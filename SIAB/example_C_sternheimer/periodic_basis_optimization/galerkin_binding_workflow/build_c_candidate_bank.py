#!/usr/bin/env python3
"""Build a deterministic C atom/diamond Galerkin Pareto candidate bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
PERIODIC_ROOT = HERE.parent
SIAB_ROOT = PERIODIC_ROOT.parents[1]
OPT_ROOT = SIAB_ROOT / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_ROOT))
sys.path.insert(0, str(PERIODIC_ROOT))

from IO.read_sternheimer import read_sternheimer  # noqa: E402
from IO.read_sternheimer_source import read_sternheimer_source  # noqa: E402
from optimize_periodic_basis import (  # noqa: E402
    validate_atomic_periodic_contract,
    validate_dataset_contract,
)
from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)
from periodic_galerkin_candidates import (  # noqa: E402
    assess_family_tradeoff,
    build_pareto_candidate_bank,
    evaluate_candidate_family_losses,
    evaluate_family_gradients,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset  # noqa: E402
from projected_pi import ProjectedPiEvaluator  # noqa: E402
from sternheimer_source_pair import pair_response_and_source  # noqa: E402


REQUIRED_CONFIG = {
    "acceptance_tolerance_ev_per_c",
    "ao_count_per_atom",
    "atom_occupations",
    "candidate_nu",
    "counterpoise",
    "coulomb",
    "element",
    "family_pair",
    "fixed_nu",
    "format_version",
    "frequency_count",
    "full_qstar_indices",
    "librpa_commit",
    "maximum_relative_family_degradation",
    "n_bands_chi0",
    "occupied_capture_floor",
    "pareto_weights",
    "pbe_max_abs_deviation_ev",
    "product_pca_threshold",
    "proxy_maximum_loo_error_ev_per_c",
    "proxy_q_indices",
    "q3_maximum_condition_ratio",
    "reference_binding_ev_per_c",
    "system",
    "tail_q_indices",
    "trust_radius",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, name, *, positive=False):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (positive and value <= 0.0)
    ):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite")
    return float(value)


def load_config(path):
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    if not isinstance(payload, dict) or set(payload) != REQUIRED_CONFIG:
        raise ValueError("C workflow config fields do not match the frozen contract")
    if payload["format_version"] != 1 or payload["system"] != "C_atom_diamond":
        raise ValueError("unsupported C workflow config")
    if payload["element"] != "C":
        raise ValueError("C workflow element must be C")
    if payload["candidate_nu"] != [3, 3, 2, 0, 0] or payload["fixed_nu"] != [
        2,
        2,
        1,
        0,
        0,
    ]:
        raise ValueError("C workflow orbital profile differs from 3s3p2d/fixed 2s2p1d")
    if payload["ao_count_per_atom"] != 22:
        raise ValueError("C workflow AO count must be 22")
    if payload["family_pair"] != ["C_atom", "C_solid"]:
        raise ValueError("C workflow family pair differs")
    if payload["atom_occupations"] != {"up": 3, "down": 1}:
        raise ValueError("C atom occupation contract differs")
    if payload["pareto_weights"] != [0.25, 0.5, 0.75]:
        raise ValueError("C workflow Pareto weights differ")
    if payload["tail_q_indices"] != [2, 6]:
        raise ValueError("C workflow tail q set differs")
    if payload["proxy_q_indices"] != [6, 7, 8]:
        raise ValueError("C workflow proxy q set differs")
    if payload["full_qstar_indices"] != [1, 2, 3, 6, 7, 8, 11, 28]:
        raise ValueError("C workflow full q-star set differs")
    for name in (
        "acceptance_tolerance_ev_per_c",
        "maximum_relative_family_degradation",
        "occupied_capture_floor",
        "pbe_max_abs_deviation_ev",
        "product_pca_threshold",
        "proxy_maximum_loo_error_ev_per_c",
        "q3_maximum_condition_ratio",
        "reference_binding_ev_per_c",
        "trust_radius",
    ):
        _finite(payload[name], name, positive=True)
    if not 0.0 < payload["occupied_capture_floor"] <= 1.0:
        raise ValueError("occupied capture floor must be in (0, 1]")
    if payload["pbe_max_abs_deviation_ev"] != 0.01:
        raise ValueError("C workflow PBE threshold differs")
    if payload["n_bands_chi0"] != -1:
        raise ValueError("C workflow must use all bands")
    if payload["product_pca_threshold"] != 1.0e-4:
        raise ValueError("C workflow product-PCA threshold differs")
    if payload["coulomb"] != "exact_grid_full_periodic":
        raise ValueError("C workflow Coulomb contract differs")
    if payload["frequency_count"] != 6 or payload["librpa_commit"] != "d4810f73":
        raise ValueError("C workflow frequency or LibRPA contract differs")
    if payload["reference_binding_ev_per_c"] != 6.902326:
        raise ValueError("C workflow reference binding differs")
    if payload["acceptance_tolerance_ev_per_c"] != 0.1:
        raise ValueError("C workflow acceptance tolerance differs")
    if payload["counterpoise"] is not False:
        raise ValueError("counterpoise is outside the C workflow")
    return payload


def build_bank_manifest(
    *,
    config,
    source_commit,
    input_hashes,
    gradient_summary,
    candidates,
):
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git hash")
    records = []
    promotable = []
    for candidate in candidates:
        evaluation = candidate["family_evaluation"]
        tradeoff = assess_family_tradeoff(
            gradient_summary["family_losses"],
            evaluation["family_losses"],
            maximum_relative_degradation=config[
                "maximum_relative_family_degradation"
            ],
        )
        record = dict(candidate)
        record.update(
            {
                "family_tradeoff_gate": tradeoff["gate"],
                "family_tradeoff": tradeoff,
            }
        )
        records.append(record)
        if tradeoff["gate"] == "pass":
            promotable.append(candidate["name"])
    return {
        "format_version": 1,
        "status": "success",
        "scope": "Galerkin candidate generation; independent ordinary SOS validation required",
        "system": config["system"],
        "source_commit": source_commit,
        "input_sha256": dict(input_hashes),
        "gradient_summary": gradient_summary,
        "candidates": records,
        "promotable_candidates": promotable,
        "candidate_bank_gate": "pass" if promotable else "fail",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--dataset-family", action="append", required=True)
    parser.add_argument("--atomic-response", type=Path, required=True)
    parser.add_argument("--atomic-source", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--block-cache-workers", type=int, default=8)
    return parser.parse_args(argv)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git hash")
    if len(args.dataset) != len(args.dataset_family):
        raise ValueError("one dataset family is required per dataset")
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(output)
    input_paths = {
        "config": args.config.resolve(),
        "initial": args.initial.resolve(),
        "atomic_response": args.atomic_response.resolve(),
        "atomic_source": args.atomic_source.resolve(),
    }
    input_paths.update(
        {
            f"dataset_{index}": path.resolve()
            for index, path in enumerate(args.dataset)
        }
    )
    for name, path in input_paths.items():
        if name.startswith("dataset_"):
            if not path.is_dir() or path.is_symlink():
                raise ValueError(f"{name} must be a real directory")
        elif not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"{name} must be a nonempty regular file")

    datasets = tuple(
        read_periodic_galerkin_dataset(
            path,
            include_reference_projection=False,
            verify_omitted_chunks=False,
        )
        for path in args.dataset
    )
    validate_dataset_contract(datasets)
    response = read_sternheimer(args.atomic_response)
    source = read_sternheimer_source(args.atomic_source)
    validate_atomic_periodic_contract(
        response,
        datasets,
        element="C",
        radial_rows=31,
    )
    pair = pair_response_and_source(response, source)
    atomic_evaluator = ProjectedPiEvaluator(pair)
    initial = read_periodic_optimizer_coefficients(
        args.initial,
        element="C",
        radial_rows=31,
        expected_nu=tuple(config["candidate_nu"]),
    )

    output.mkdir(parents=True)
    _write_json(
        output / "STATUS.json",
        {"status": "running", "source_commit": args.source_commit},
    )
    try:
        gradient = evaluate_family_gradients(
            datasets,
            initial,
            fixed_nu={"C": tuple(config["fixed_nu"])},
            dataset_families=tuple(args.dataset_family),
            additional_family_evaluators={"C_atom": atomic_evaluator},
            occupied_capture_tolerance=1.0 - config["occupied_capture_floor"],
            block_cache_workers=args.block_cache_workers,
        )
        bank = build_pareto_candidate_bank(
            gradient,
            fixed_nu={"C": tuple(config["fixed_nu"])},
            family_pair=tuple(config["family_pair"]),
            weights=tuple(config["pareto_weights"]),
            trust_radius=config["trust_radius"],
        )
        candidate_records = []
        for candidate in bank:
            label = f"pareto_w{candidate.weight:.2f}".replace(".", "p")
            directory = output / "candidates" / label
            directory.mkdir(parents=True)
            orbital = directory / "ORBITAL_RESULTS.txt"
            write_periodic_optimizer_coefficients(orbital, candidate.coefficients)
            evaluation = evaluate_candidate_family_losses(
                gradient,
                candidate.coefficients,
            )
            record = {
                "name": label,
                "weight": candidate.weight,
                "trust_radius": candidate.trust_radius,
                "coefficients_sha256": candidate.coefficients_sha256,
                "orbital_file": str(orbital),
                "orbital_sha256": sha256(orbital),
                "family_evaluation": evaluation,
            }
            _write_json(directory / "CANDIDATE.json", record)
            candidate_records.append(record)

        gradient_summary = {
            "family_order": list(gradient.family_order),
            "family_losses": gradient.family_losses,
            "gradient_norms": gradient.gradient_norms,
            "gradient_cosines": gradient.gradient_cosines,
            "minimum_occupied_capture": gradient.minimum_occupied_capture,
            "maximum_overlap_condition": gradient.maximum_overlap_condition,
        }
        input_hashes = {
            name: sha256(path)
            for name, path in input_paths.items()
            if not name.startswith("dataset_")
        }
        input_hashes.update(
            {
                name: dataset.physics_hash
                for name, dataset in zip(
                    (name for name in input_paths if name.startswith("dataset_")),
                    datasets,
                )
            }
        )
        manifest = build_bank_manifest(
            config=config,
            source_commit=args.source_commit,
            input_hashes=input_hashes,
            gradient_summary=gradient_summary,
            candidates=candidate_records,
        )
        _write_json(output / "CANDIDATE_BANK.json", manifest)
        provenance = {
            "status": "success",
            "source_commit": args.source_commit,
            "script_sha256": sha256(Path(__file__)),
            "input_sha256": input_hashes,
            "candidate_bank_sha256": sha256(output / "CANDIDATE_BANK.json"),
        }
        _write_json(output / "PROVENANCE.json", provenance)
        _write_json(
            output / "STATUS.json",
            {
                "status": "success",
                "candidate_bank_gate": manifest["candidate_bank_gate"],
                "candidate_bank_sha256": provenance["candidate_bank_sha256"],
            },
        )
        print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    except Exception as error:
        _write_json(
            output / "STATUS.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
