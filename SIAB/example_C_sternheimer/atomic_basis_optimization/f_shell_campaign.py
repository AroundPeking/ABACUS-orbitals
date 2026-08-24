#!/usr/bin/env python3
"""Prepare and audit the first C f-shell response-basis optimization."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from continuation_campaign import (
    DEFAULT_ORBITAL,
    DEFAULT_REFERENCE,
    FIXED_DZP,
    _float64_bytes,
    assess_convergence,
    freeze_keys,
    read_coefficient,
    read_final_loss,
    read_spillage_rows,
    sha256,
    validate_source_hashes as validate_continuation_source_hashes,
    variable_keys,
)


HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / "SIAB_INPUT.f_shell_optimization.json"
EXPECTED_HASHES = {
    "target": "e976c164595758029cb91ebe3913af6865780ef95fdb48954c07075bd0c7e3ff",
    "seed": "b8a2b2cacde942ea1ec5e08c37a3199b0bdb7d730a1b2ed0929f8229a6cd1a22",
    "reference": "b58a2183c3028e46e6f4bc55b0f21531f1253275d5c2f2c4ee4e27676c1b55f4",
    "orbital": "7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d",
}
PRIOR_BASIS_LOSS = 0.7114382940310687
MATERIAL_RELATIVE_GAIN = 0.01
EXPECTED_VARIABLE = frozenset(
    {
        ("C", 0, 3),
        ("C", 1, 3),
        ("C", 2, 2),
        ("C", 3, 1),
    }
)


def validate_template(template) -> None:
    if template.get("element", {}).get("Nt_all") != ["C"]:
        raise ValueError("f-shell optimization must contain only C")
    if template["element"].get("Nu") != {"C": [3, 3, 2, 1, 0]}:
        raise ValueError("f-shell optimization must use 3s3p2d1f with zero g")
    if freeze_keys(template.get("freeze_orbitals", [])) != FIXED_DZP:
        raise ValueError("f-shell optimization must use the exact frozen DZP prefix")
    if variable_keys(template) != EXPECTED_VARIABLE:
        raise ValueError("f-shell optimization must vary only 3s, 3p, 2d and 1f")
    stages = template.get("optimize", [])
    if len(stages) != 1 or stages[0].get("max_steps") != 3000:
        raise ValueError("f-shell optimization must run at most 3000 steps")
    if stages[0].get("optimizer") != "Adam" or stages[0].get("kwargs") != {
        "lr": 0.001
    }:
        raise ValueError("f-shell optimization must use Adam with lr=0.001")
    if template.get("loss", {}).get("mode") != "st_only":
        raise ValueError("f-shell optimization must use st_only")
    radial = template.get("radial", {})
    if (radial.get("Rcut"), radial.get("dr"), radial.get("Ecut")) != (
        10,
        0.01,
        100,
    ):
        raise ValueError("radial contract must be 10 au/0.01/100 Ry")


def validate_source_hashes(target, seed, reference, orbital, *, expected=None):
    expected = EXPECTED_HASHES if expected is None else expected
    records = validate_continuation_source_hashes(
        target,
        seed,
        reference,
        orbital,
        expected={
            "target": expected["target"],
            "checkpoint": expected["seed"],
            "reference": expected["reference"],
            "orbital": expected["orbital"],
        },
    )
    records["seed"] = records.pop("checkpoint")
    return records


def build_input(template, target: Path, seed: Path) -> dict:
    result = copy.deepcopy(template)
    result["file_list"] = {
        "sternheimer": [
            {
                "path": str(Path(target).resolve()),
                "family": "C_atom",
                "role": "physical",
            }
        ]
    }
    result["C_init_info"]["C_init_file"] = str(Path(seed).resolve())
    return result


def assess_f_shell_gain(
    losses,
    *,
    optimizer_rows,
    maximum_condition,
    prior_basis_loss=PRIOR_BASIS_LOSS,
    material_relative_gain=MATERIAL_RELATIVE_GAIN,
):
    losses = tuple(float(value) for value in losses)
    if not losses or not math.isfinite(losses[0]):
        raise RuntimeError("f-shell seed loss must be finite")
    if losses[0] >= prior_basis_loss:
        raise RuntimeError("f-shell seed did not improve the converged 3s3p2d space")
    report = assess_convergence(
        losses,
        optimizer_rows=optimizer_rows,
        maximum_condition=maximum_condition,
        checkpoint_loss=prior_basis_loss,
    )
    best = report["best_sternheimer_loss"]
    relative_gain = (prior_basis_loss - best) / prior_basis_loss
    seed_gain = (prior_basis_loss - losses[0]) / prior_basis_loss
    optimizer_gain = (losses[0] - best) / losses[0]
    if report["status"] == "CONTINUE_REQUIRED":
        status = "CONTINUE_REQUIRED"
    elif relative_gain >= material_relative_gain:
        status = "F_SHELL_MATERIAL_GAIN"
    else:
        status = "F_SHELL_MARGINAL_GAIN"
    report.update(
        {
            "status": status,
            "prior_3s3p2d_loss": prior_basis_loss,
            "initial_f_shell_seed_loss": losses[0],
            "relative_gain_to_3s3p2d": relative_gain,
            "seed_insertion_relative_gain": seed_gain,
            "optimizer_relative_gain_from_seed": optimizer_gain,
            "material_relative_gain_threshold": material_relative_gain,
            "advance_to_multicenter_projected_pi": status
            == "F_SHELL_MATERIAL_GAIN",
        }
    )
    return report


def prepare(args):
    template = json.loads(Path(args.template).read_text(encoding="ascii"))
    validate_template(template)
    records = validate_source_hashes(
        args.target, args.seed, args.reference, args.orbital
    )
    for key in FIXED_DZP | EXPECTED_VARIABLE:
        read_coefficient(args.seed, key)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "INPUT"
    input_path.write_text(
        json.dumps(build_input(template, args.target, args.seed), indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    manifest = {
        "status": "prepared",
        "purpose": "test whether the leading C f residual gives material response gain",
        "sources": records,
        "input": {"path": str(input_path), "sha256": sha256(input_path)},
        "fixed_dzp": sorted(FIXED_DZP),
        "variable_response_shells": sorted(EXPECTED_VARIABLE),
        "prior_3s3p2d_loss": PRIOR_BASIS_LOSS,
        "material_relative_gain_threshold": MATERIAL_RELATIVE_GAIN,
    }
    (output / "PREPARATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return manifest


def audit(args):
    output = Path(args.output).resolve()
    manifest = json.loads(
        (output / "PREPARATION_MANIFEST.json").read_text(encoding="ascii")
    )
    reference = Path(manifest["sources"]["reference"]["path"])
    seed = Path(manifest["sources"]["seed"]["path"])
    final = output / "ORBITAL_RESULTS.txt"
    spillage = output / "Spillage.dat"
    run_log = output / "run.log"
    for path in (reference, seed, final, spillage, run_log):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    frozen = {}
    for key in sorted(FIXED_DZP):
        equal = _float64_bytes(read_coefficient(reference, key)) == _float64_bytes(
            read_coefficient(final, key)
        )
        frozen["/".join(map(str, key))] = equal
        if not equal:
            raise RuntimeError(f"frozen DZP coefficient changed: {key}")

    changed = {}
    for key in sorted(EXPECTED_VARIABLE):
        changed["/".join(map(str, key))] = _float64_bytes(
            read_coefficient(seed, key)
        ) != _float64_bytes(read_coefficient(final, key))
    if not any(changed.values()):
        raise RuntimeError("f-shell optimization changed none of its variable shells")

    rows = read_spillage_rows(spillage)
    optimizer_rows = sum(row.step >= 0 for row in rows)
    report = assess_f_shell_gain(
        [row.loss for row in rows],
        optimizer_rows=optimizer_rows,
        maximum_condition=max(row.condition for row in rows),
    )
    mode, final_loss = read_final_loss(final)
    if mode != "st_only":
        raise RuntimeError(f"unexpected final loss mode: {mode}")
    if abs(final_loss - report["best_sternheimer_loss"]) > 1.0e-9:
        raise RuntimeError("final coefficient loss is not the accepted best loss")
    report.update(
        {
            "scope": "atomic first-f-shell response gate only; no SOS promotion",
            "frozen_dzp_bitwise_equal": frozen,
            "variable_shells_changed_from_seed": changed,
            "outputs": {
                label: {"size": path.stat().st_size, "sha256": sha256(path)}
                for label, path in (
                    ("coefficients", final),
                    ("spillage", spillage),
                    ("run_log", run_log),
                )
            },
        }
    )
    (output / "F_SHELL_OPTIMIZATION_RESULT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--target", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=Path, required=True)
    prepare_parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    prepare_parser.add_argument("--orbital", type=Path, default=DEFAULT_ORBITAL)
    prepare_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    prepare_parser.add_argument("--output", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = prepare(args) if args.command == "prepare" else audit(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
