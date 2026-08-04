#!/usr/bin/env python3
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

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
from projected_pi_optimization import (
    NormalizedPhysicalFamilyProjectedPiOptimization,
)
from sternheimer_source_pair import pair_response_and_source


BASIS_NU = {
    "two_d": (3, 2, 2, 0, 0),
    "first_f": (3, 2, 2, 1, 0),
    "first_g": (3, 2, 2, 1, 1),
    "second_f": (3, 2, 2, 2, 1),
    "second_g": (3, 2, 2, 1, 2),
}
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)
FAMILY_NAMES = ("H", "H2")
RADIAL_SIZE = 25
FAMILY_POWER = 4
RANK_TOLERANCE = 1.0e-12
CONDITION_LIMIT = 1.0e12


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Rank five archived SIAB bases with RPA-sensitive losses."
    )
    for family in ("h", "h2"):
        parser.add_argument(f"--{family}-response", required=True, type=Path)
        parser.add_argument(f"--{family}-source", required=True, type=Path)
        parser.add_argument(f"--{family}-audit", required=True, type=Path)
    for basis_name in BASIS_NU:
        parser.add_argument(
            f"--{basis_name.replace('_', '-')}",
            required=True,
            type=Path,
        )
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


def _coefficient_metadata(nu):
    hydrogen = util.Info()
    hydrogen.index = 0
    hydrogen.Nl = len(nu)
    hydrogen.Ne = RADIAL_SIZE
    hydrogen.Nu = list(nu)
    return {"H": hydrogen}


def _read_coefficients(path, basis_name):
    nu = BASIS_NU[basis_name]
    coefficients, metadata = read_C_init(
        str(path),
        _coefficient_metadata(nu),
        return_metadata=True,
    )
    expected = frozenset(
        ("H", l_value, zeta)
        for l_value, count in enumerate(nu)
        for zeta in range(count)
    )
    if metadata.loaded_indices != expected or metadata.appended_indices:
        raise ValueError(
            f"{basis_name} must contain exactly the frozen coefficient columns"
        )
    return coefficients


def _float_list(values):
    return [float(value) for value in values]


def _family_record(result):
    return {
        "loss": float(result.loss),
        "base_loss": float(result.base_loss),
        "sensitivity_loss": float(result.sensitivity_loss),
        "frequency_ha": _float_list(result.frequency_ha),
        "frequency_weight": _float_list(result.frequency_weight),
        "frequency_loss": _float_list(result.frequency_loss),
        "frequency_base_loss": _float_list(result.frequency_base_loss),
        "frequency_sensitivity_loss": _float_list(
            result.frequency_sensitivity_loss
        ),
        "trace_log_difference": _float_list(result.trace_log_difference),
        "minimum_reference_dielectric_eigenvalue": _float_list(
            result.minimum_reference_dielectric_eigenvalue
        ),
        "minimum_candidate_dielectric_eigenvalue": _float_list(
            result.minimum_candidate_dielectric_eigenvalue
        ),
        "reference_rank": result.reference_rank,
        "max_candidate_condition": result.max_candidate_condition,
    }


def _basis_record(result):
    return {
        "loss": float(result.loss),
        "max_condition": result.max_condition,
        "frequency_ha": _float_list(result.frequency_ha),
        "frequency_loss": _float_list(result.frequency_loss),
        "families": {
            family_name: _family_record(result.family_results[family_name])
            for family_name in FAMILY_NAMES
        },
    }


def _ordering_gates(bases):
    return {
        "first_f_improves_two_d": (
            bases["first_f"]["loss"] < bases["two_d"]["loss"]
        ),
        "first_g_improves_first_f": (
            bases["first_g"]["loss"] < bases["first_f"]["loss"]
        ),
        "second_f_not_better": (
            bases["second_f"]["loss"] >= bases["first_g"]["loss"]
        ),
        "second_g_not_better": (
            bases["second_g"]["loss"] >= bases["first_g"]["loss"]
        ),
    }


def _evaluate(pairs, coefficients):
    alpha_results = []
    named_pairs = tuple(
        (family_name, pairs[family_name]) for family_name in FAMILY_NAMES
    )
    for alpha in ALPHAS:
        evaluator = NormalizedPhysicalFamilyProjectedPiOptimization(
            *named_pairs,
            relative_rank_tolerance=RANK_TOLERANCE,
            condition_limit=CONDITION_LIMIT,
            sensitivity_alpha=alpha,
            family_power=FAMILY_POWER,
        )
        with torch.no_grad():
            bases = {
                basis_name: _basis_record(evaluator.evaluate(value))
                for basis_name, value in coefficients.items()
            }
        gates = _ordering_gates(bases)
        alpha_results.append(
            {
                "alpha": alpha,
                "admissible": all(gates.values()),
                "gates": gates,
                "bases": bases,
            }
        )
        del evaluator
        gc.collect()
    return alpha_results


def _atomic_text(path, value):
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_figure(figure, path, **save_options):
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as stream:
            temporary_name = stream.name
        figure.savefig(
            temporary_name,
            format=path.suffix.lstrip("."),
            bbox_inches="tight",
            **save_options,
        )
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_plot(output_dir, alpha_result):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot output requires matplotlib in the analysis environment"
        ) from exc

    colors = {
        "two_d": "#1f77b4",
        "first_f": "#d62728",
        "first_g": "#2ca02c",
        "second_f": "#9467bd",
        "second_g": "#8c564b",
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.3),
        constrained_layout=True,
    )
    for axis, family_name in zip(axes, FAMILY_NAMES):
        for basis_name in BASIS_NU:
            record = alpha_result["bases"][basis_name]["families"][family_name]
            axis.plot(
                record["frequency_ha"],
                record["frequency_base_loss"],
                color=colors[basis_name],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                label=f"{basis_name} base",
            )
            axis.plot(
                record["frequency_ha"],
                record["frequency_sensitivity_loss"],
                color=colors[basis_name],
                linestyle="--",
                marker="s",
                linewidth=1.2,
                markersize=3.0,
                label=f"{basis_name} sensitivity",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Imaginary frequency (Ha)")
        axis.set_ylabel("Relative projected-Pi loss")
        axis.set_title(family_name)
        axis.grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    _atomic_figure(
        figure,
        output_dir / "rpa_sensitive_frequency.png",
        dpi=220,
    )
    _atomic_figure(figure, output_dir / "rpa_sensitive_frequency.pdf")
    plt.close(figure)


def _markdown(payload):
    selected = payload["selected_alpha"]
    selected_text = "none" if selected is None else str(selected)
    lines = [
        "# RPA-sensitive five-basis ranking",
        "",
        f"Decision: `{payload['decision']}`",
        f"Selected alpha: `{selected_text}`",
        "",
        "| Alpha | Basis | H Base loss | H Sensitivity loss | H2 Base loss "
        "| H2 Sensitivity loss | Fourth-order family loss | Max condition |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha_result in payload["alpha_results"]:
        for basis_name in BASIS_NU:
            basis = alpha_result["bases"][basis_name]
            h = basis["families"]["H"]
            h2 = basis["families"]["H2"]
            lines.append(
                "| {alpha:g} | {basis} | {h_base:.10e} | {h_sensitivity:.10e} "
                "| {h2_base:.10e} | {h2_sensitivity:.10e} | {total:.10e} "
                "| {condition:.6e} |".format(
                    alpha=alpha_result["alpha"],
                    basis=basis_name,
                    h_base=h["base_loss"],
                    h_sensitivity=h["sensitivity_loss"],
                    h2_base=h2["base_loss"],
                    h2_sensitivity=h2["sensitivity_loss"],
                    total=basis["loss"],
                    condition=basis["max_condition"],
                )
            )
    lines.extend(
        [
            "",
            "| Alpha | first f improves two d | first g improves first f "
            "| second f not better | second g not better | Admissible |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for alpha_result in payload["alpha_results"]:
        gates = alpha_result["gates"]
        lines.append(
            "| {alpha:g} | {first_f} | {first_g} | {second_f} | "
            "{second_g} | {admissible} |".format(
                alpha=alpha_result["alpha"],
                first_f=gates["first_f_improves_two_d"],
                first_g=gates["first_g_improves_first_f"],
                second_f=gates["second_f_not_better"],
                second_g=gates["second_g_not_better"],
                admissible=alpha_result["admissible"],
            )
        )
    lines.extend(
        [
            "",
            "This is a code-level metric gate over historical labels. SOS "
            "energy and ghost-family data are not numeric inputs, and no new "
            "candidate was evaluated.",
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
    }
    input_paths.update(
        (basis_name, getattr(args, basis_name)) for basis_name in BASIS_NU
    )
    for label, path in input_paths.items():
        _require_file(path, label)

    audits = {
        "H": read_zero_order_audit(args.h_audit, "H"),
        "H2": read_zero_order_audit(args.h2_audit, "H2"),
    }
    pairs = {}
    reader_warnings = {}
    for family_name, response_path, source_path in (
        ("H", args.h_response, args.h_source),
        ("H2", args.h2_response, args.h2_source),
    ):
        pair = pair_response_and_source(
            read_sternheimer(response_path),
            read_sternheimer_source(source_path),
        )
        pairs[family_name] = pair
        reader_warnings[family_name] = list(pair.provenance_warnings)

    coefficients = {
        basis_name: _read_coefficients(input_paths[basis_name], basis_name)
        for basis_name in BASIS_NU
    }
    alpha_results = _evaluate(pairs, coefficients)
    admissible_alphas = [
        result["alpha"] for result in alpha_results if result["admissible"]
    ]
    selected_alpha = max(admissible_alphas) if admissible_alphas else None
    decision = "pass" if selected_alpha is not None else "stop_galerkin_required"
    input_sha256 = {
        label: _sha256(path) for label, path in input_paths.items()
    }
    zero_order_audits = {}
    for family_name, audit in audits.items():
        record = asdict(audit)
        record["audit_file_sha256"] = input_sha256[f"{family_name}_audit"]
        zero_order_audits[family_name] = record
    payload = {
        "schema_version": 1,
        "decision": decision,
        "basis_nu": {
            basis_name: list(nu) for basis_name, nu in BASIS_NU.items()
        },
        "alphas": list(ALPHAS),
        "admissible_alphas": admissible_alphas,
        "selected_alpha": selected_alpha,
        "family_power": FAMILY_POWER,
        "relative_rank_tolerance": RANK_TOLERANCE,
        "condition_limit": CONDITION_LIMIT,
        "input_sha256": input_sha256,
        "zero_order_audits": zero_order_audits,
        "reader_warnings": reader_warnings,
        "alpha_results": alpha_results,
        "uses_sos_energy_as_numeric_input": False,
        "uses_ghost_family": False,
        "new_candidate_was_evaluated": False,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_text(
        args.output_dir / "rpa_sensitive_ranking.json",
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(
        args.output_dir / "rpa_sensitive_ranking.md",
        _markdown(payload),
    )
    plot_result = next(
        (
            result
            for result in alpha_results
            if result["alpha"] == selected_alpha
        ),
        alpha_results[-1],
    )
    _write_plot(args.output_dir, plot_result)
    return 0 if decision == "pass" else 2


def main(argv=None):
    try:
        return run(parse_arguments(argv))
    except Exception as exc:
        print(f"RPA-sensitive ranking failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
