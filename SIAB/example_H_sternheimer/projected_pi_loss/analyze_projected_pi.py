#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch


SIAB_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZER_ROOT = SIAB_ROOT / "opt_orb_pytorch_dpsi"
if str(OPTIMIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_ROOT))

import util
from IO.func_C import read_C_init
from IO.read_sternheimer import read_sternheimer
from IO.read_sternheimer_source import read_sternheimer_source
from IO.read_zero_order_audit import read_zero_order_audit
from projected_pi import NormalizedPhysicalFamilyProjectedPi
from sternheimer_source_pair import pair_response_and_source


THRESHOLDS = (1.0e-10, 1.0e-11, 1.0e-12)
BASIS_NAMES = (
    "initial_tzdp",
    "fixed_dzp_joint",
    "low_frequency_guarded",
)
FAMILY_NAMES = ("H", "H2")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate source-aware projected-Pi basis rankings."
    )
    for family in ("h", "h2"):
        parser.add_argument(f"--{family}-response", required=True, type=Path)
        parser.add_argument(f"--{family}-source", required=True, type=Path)
        parser.add_argument(f"--{family}-audit", required=True, type=Path)
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--joint", required=True, type=Path)
    parser.add_argument("--guarded", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path, label):
    if not path.is_file():
        raise ValueError(f"{label} is not a readable file: {path}")


def _coefficient_metadata():
    hydrogen = util.Info()
    hydrogen.index = 0
    hydrogen.Nl = 5
    hydrogen.Ne = 25
    hydrogen.Nu = [3, 2, 0, 0, 0]
    return {"H": hydrogen}


def _read_coefficients(path, label):
    coefficients, metadata = read_C_init(
        str(path),
        _coefficient_metadata(),
        return_metadata=True,
    )
    expected = frozenset(
        {
            ("H", 0, 0),
            ("H", 0, 1),
            ("H", 0, 2),
            ("H", 1, 0),
            ("H", 1, 1),
        }
    )
    if metadata.loaded_indices != expected or metadata.appended_indices:
        raise ValueError(f"{label} must contain exactly H 3s2p coefficients")
    for l in range(2, 5):
        if coefficients["H"][l].numel() != 0:
            raise ValueError(f"{label} must not contain d/f/g coefficients")
    return coefficients


def _validate_fixed_dzp(initial, candidate, label):
    differences = (
        torch.max(torch.abs(initial["H"][0][:, :2] - candidate["H"][0][:, :2])),
        torch.max(torch.abs(initial["H"][1][:, :1] - candidate["H"][1][:, :1])),
    )
    maximum = max(float(value) for value in differences)
    if not math.isfinite(maximum) or maximum > 1.0e-12:
        raise ValueError(
            f"{label} fixed DZP 1s,2s,1p columns differ by {maximum:.17g}"
        )
    return maximum


def _result_record(result):
    hermitian_candidate = float(
        torch.max(torch.abs(result.candidate_pi - result.candidate_pi.mH))
    )
    hermitian_reference = float(
        torch.max(torch.abs(result.reference_pi - result.reference_pi.mH))
    )
    return {
        "loss": float(result.loss),
        "frequency_ha": [float(value) for value in result.frequency_ha],
        "frequency_weight": [
            float(value) for value in result.frequency_weight
        ],
        "frequency_loss": [float(value) for value in result.frequency_loss],
        "reference_rank": result.reference_rank,
        "max_candidate_condition": result.max_candidate_condition,
        "candidate_hermitian_error": hermitian_candidate,
        "reference_hermitian_error": hermitian_reference,
    }


def _threshold_key(value):
    return f"{value:.0e}"


def _relative_spread(values):
    maximum = max(values)
    minimum = min(values)
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    return (maximum - minimum) / scale


def _evaluate(pairs, coefficients):
    results = {}
    for threshold in THRESHOLDS:
        family = NormalizedPhysicalFamilyProjectedPi(
            tuple((name, pairs[name]) for name in FAMILY_NAMES),
            relative_rank_tolerance=threshold,
            condition_limit=1.0e12,
        )
        threshold_results = {}
        for basis_name in BASIS_NAMES:
            value = family.evaluate(coefficients[basis_name])
            families = {
                name: _result_record(value.results[name])
                for name in FAMILY_NAMES
            }
            threshold_results[basis_name] = {
                "total_loss": float(value.loss),
                "max_candidate_condition": value.max_candidate_condition,
                "families": families,
            }
        results[_threshold_key(threshold)] = threshold_results
        del family
        gc.collect()
    return results


def _build_gates(results):
    threshold_keys = tuple(_threshold_key(value) for value in THRESHOLDS)
    joint_improves = all(
        results[key]["fixed_dzp_joint"]["total_loss"]
        < results[key]["initial_tzdp"]["total_loss"]
        for key in threshold_keys
    )
    guarded_improves = all(
        results[key]["low_frequency_guarded"]["total_loss"]
        < results[key]["initial_tzdp"]["total_loss"]
        for key in threshold_keys
    )

    spreads = {}
    all_spreads = []
    for basis_name in BASIS_NAMES:
        basis_spreads = {}
        for family_name in (*FAMILY_NAMES, "total"):
            if family_name == "total":
                values = [
                    results[key][basis_name]["total_loss"]
                    for key in threshold_keys
                ]
            else:
                values = [
                    results[key][basis_name]["families"][family_name]["loss"]
                    for key in threshold_keys
                ]
            spread = _relative_spread(values)
            basis_spreads[family_name] = spread
            all_spreads.append(spread)
        spreads[basis_name] = basis_spreads
    threshold_stable = all(value <= 0.01 for value in all_spreads)
    return {
        "joint_improves_initial": joint_improves,
        "guarded_improves_initial": guarded_improves,
        "threshold_stable_within_one_percent": threshold_stable,
        "relative_spread": spreads,
        "maximum_relative_spread": max(all_spreads),
    }


def _atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_plot(output_dir, results):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot output requires matplotlib; install it in an isolated "
            "analysis dependency directory and add that directory to PYTHONPATH"
        ) from exc

    nominal = results[_threshold_key(THRESHOLDS[-1])]
    colors = {
        "initial_tzdp": "#1f77b4",
        "fixed_dzp_joint": "#d62728",
        "low_frequency_guarded": "#2ca02c",
    }
    markers = {
        "initial_tzdp": "o",
        "fixed_dzp_joint": "s",
        "low_frequency_guarded": "^",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for axis, family_name in zip(axes, FAMILY_NAMES):
        for basis_name in BASIS_NAMES:
            record = nominal[basis_name]["families"][family_name]
            axis.plot(
                record["frequency_ha"],
                record["frequency_loss"],
                color=colors[basis_name],
                marker=markers[basis_name],
                linewidth=1.5,
                markersize=4.0,
                label=basis_name,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Imaginary frequency (Ha)")
        axis.set_ylabel("Relative projected-Pi error")
        axis.set_title(family_name)
        axis.grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=8)
    for suffix, options in (
        ("png", {"dpi": 220}),
        ("pdf", {}),
    ):
        target = output_dir / f"projected_pi_frequency.{suffix}"
        temporary = output_dir / f"projected_pi_frequency.tmp.{suffix}"
        figure.savefig(temporary, format=suffix, bbox_inches="tight", **options)
        os.replace(temporary, target)
    plt.close(figure)


def _markdown(payload):
    lines = [
        "# Projected-Pi feasibility ranking",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "| S+ threshold | Basis | H loss | H2 loss | Total loss | H rank | H2 rank | Max condition |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in payload["thresholds"]:
        key = _threshold_key(threshold)
        for basis_name in BASIS_NAMES:
            record = payload["results"][key][basis_name]
            lines.append(
                "| {threshold} | {basis} | {h:.10e} | {h2:.10e} | "
                "{total:.10e} | {hrank} | {h2rank} | {condition:.6e} |".format(
                    threshold=key,
                    basis=basis_name,
                    h=record["families"]["H"]["loss"],
                    h2=record["families"]["H2"]["loss"],
                    total=record["total_loss"],
                    hrank=record["families"]["H"]["reference_rank"],
                    h2rank=record["families"]["H2"]["reference_rank"],
                    condition=record["max_candidate_condition"],
                )
            )
    lines.extend(
        [
            "",
            "The loss uses only paired Sternheimer source/response data. SOS energy and ghost data are not inputs.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args):
    input_paths = {
        "H_response": args.h_response,
        "H_source": args.h_source,
        "H_audit": args.h_audit,
        "H2_response": args.h2_response,
        "H2_source": args.h2_source,
        "H2_audit": args.h2_audit,
        "initial_tzdp": args.initial,
        "fixed_dzp_joint": args.joint,
        "low_frequency_guarded": args.guarded,
    }
    for label, path in input_paths.items():
        _require_file(path, label)

    audits = {
        "H": read_zero_order_audit(args.h_audit, "H"),
        "H2": read_zero_order_audit(args.h2_audit, "H2"),
    }
    pairs = {}
    reader_warnings = {}
    for name, response_path, source_path in (
        ("H", args.h_response, args.h_source),
        ("H2", args.h2_response, args.h2_source),
    ):
        pair = pair_response_and_source(
            read_sternheimer(response_path),
            read_sternheimer_source(source_path),
        )
        pairs[name] = pair
        reader_warnings[name] = list(pair.provenance_warnings)

    coefficients = {
        "initial_tzdp": _read_coefficients(args.initial, "initial_tzdp"),
        "fixed_dzp_joint": _read_coefficients(
            args.joint, "fixed_dzp_joint"
        ),
        "low_frequency_guarded": _read_coefficients(
            args.guarded, "low_frequency_guarded"
        ),
    }
    fixed_dzp_max_difference = {
        "fixed_dzp_joint": _validate_fixed_dzp(
            coefficients["initial_tzdp"],
            coefficients["fixed_dzp_joint"],
            "fixed_dzp_joint",
        ),
        "low_frequency_guarded": _validate_fixed_dzp(
            coefficients["initial_tzdp"],
            coefficients["low_frequency_guarded"],
            "low_frequency_guarded",
        ),
    }
    results = _evaluate(pairs, coefficients)
    gates = _build_gates(results)
    decision = (
        "pass"
        if gates["joint_improves_initial"]
        and gates["guarded_improves_initial"]
        and gates["threshold_stable_within_one_percent"]
        else "stop_galerkin_required"
    )
    input_sha256 = {
        label: _sha256(path) for label, path in input_paths.items()
    }
    payload = {
        "schema_version": 1,
        "decision": decision,
        "thresholds": list(THRESHOLDS),
        "input_sha256": input_sha256,
        "zero_order_audit_sha256": {
            "H": input_sha256["H_audit"],
            "H2": input_sha256["H2_audit"],
        },
        "zero_order_audit_status": {
            name: "pass" if audit.passed else "fail"
            for name, audit in audits.items()
        },
        "reader_warnings": reader_warnings,
        "fixed_dzp_max_abs_difference": fixed_dzp_max_difference,
        "results": results,
        "gates": gates,
        "uses_sos_energy": False,
        "uses_ghost_family": False,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_text(
        args.output_dir / "projected_pi_ranking.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        args.output_dir / "projected_pi_ranking.md",
        _markdown(payload),
    )
    _write_plot(args.output_dir, results)
    return 0 if decision == "pass" else 2


def main(argv=None):
    try:
        return run(parse_arguments(argv))
    except Exception as exc:
        print(f"projected-Pi analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
