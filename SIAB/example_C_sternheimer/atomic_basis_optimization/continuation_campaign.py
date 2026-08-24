#!/usr/bin/env python3
"""Prepare and audit the converged C 3s3p2d Sternheimer continuation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import NamedTuple


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_TEMPLATE = HERE / "SIAB_INPUT.tzdp_continuation.json"
DEFAULT_REFERENCE = (
    REPO_ROOT / "SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt"
)
DEFAULT_ORBITAL = (
    REPO_ROOT / "SG15_v1.0/Orbitals_v2.0/C_TZDP/C_gga_10au_100Ry_3s3p2d.orb"
)
EXPECTED_HASHES = {
    "target": "e976c164595758029cb91ebe3913af6865780ef95fdb48954c07075bd0c7e3ff",
    "checkpoint": "3e0b83c95ce744dd75d54da9128ecbadc11fb7d3357830af697a47d0c6b6d406",
    "reference": "b58a2183c3028e46e6f4bc55b0f21531f1253275d5c2f2c4ee4e27676c1b55f4",
    "orbital": "7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d",
}
EXPECTED_CHECKPOINT_LOSS = 0.7437783696
FIXED_DZP = frozenset(
    {
        ("C", 0, 1),
        ("C", 0, 2),
        ("C", 1, 1),
        ("C", 1, 2),
        ("C", 2, 1),
    }
)
EXPECTED_VARIABLE = frozenset({("C", 0, 3), ("C", 1, 3), ("C", 2, 2)})


class SpillageRow(NamedTuple):
    step: int
    loss: float
    condition: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_keys(specs) -> frozenset[tuple[str, int, int]]:
    return frozenset(
        (spec["element"], int(spec["l"]), int(spec["zeta"])) for spec in specs
    )


def all_orbital_keys(template) -> frozenset[tuple[str, int, int]]:
    result = set()
    for element, counts in template["element"]["Nu"].items():
        for l, count in enumerate(counts):
            result.update((element, l, zeta) for zeta in range(1, count + 1))
    return frozenset(result)


def variable_keys(template) -> frozenset[tuple[str, int, int]]:
    return all_orbital_keys(template) - freeze_keys(template["freeze_orbitals"])


def validate_template(template) -> None:
    if template.get("element", {}).get("Nt_all") != ["C"]:
        raise ValueError("continuation must contain only C")
    if template["element"].get("Nu") != {"C": [3, 3, 2, 0, 0]}:
        raise ValueError("continuation must use 3s3p2d with explicit zero f/g")
    if freeze_keys(template.get("freeze_orbitals", [])) != FIXED_DZP:
        raise ValueError("continuation must use the exact frozen DZP prefix")
    if variable_keys(template) != EXPECTED_VARIABLE:
        raise ValueError("continuation must vary only 3s, 3p and 2d")
    stages = template.get("optimize", [])
    if len(stages) != 1 or stages[0].get("max_steps") != 3000:
        raise ValueError("continuation must run at most 3000 steps")
    if stages[0].get("optimizer") != "Adam" or stages[0].get("kwargs") != {
        "lr": 0.001
    }:
        raise ValueError("continuation must use Adam with lr=0.001")
    if template.get("loss", {}).get("mode") != "st_only":
        raise ValueError("continuation must use st_only")
    radial = template.get("radial", {})
    if (radial.get("Rcut"), radial.get("dr"), radial.get("Ecut")) != (
        10,
        0.01,
        100,
    ):
        raise ValueError("radial contract must be 10 au/0.01/100 Ry")


def validate_source_hashes(target, checkpoint, reference, orbital, *, expected=None):
    expected = EXPECTED_HASHES if expected is None else expected
    records = {}
    for label, path in (
        ("target", target),
        ("checkpoint", checkpoint),
        ("reference", reference),
        ("orbital", orbital),
    ):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} must be a nonempty regular file")
        digest = sha256(path)
        if digest != expected[label]:
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected[label]}, got {digest}"
            )
        records[label] = {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": digest,
        }
    return records


def build_input(template, target: Path, checkpoint: Path) -> dict:
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
    result["C_init_info"]["C_init_file"] = str(Path(checkpoint).resolve())
    return result


def read_coefficient(path: Path, key: tuple[str, int, int]) -> tuple[float, ...]:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    inside = False
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line == "<Coefficient>":
            inside = True
            index += 1
            continue
        if line == "</Coefficient>":
            break
        if inside and line.startswith("Type"):
            fields = lines[index + 1].split()
            label = (fields[0], int(fields[1]), int(fields[2]))
            values = []
            index += 2
            while index < len(lines):
                value = lines[index].strip()
                if value.startswith("Type") or value == "</Coefficient>":
                    break
                fields = value.split()
                if len(fields) != 1:
                    raise ValueError(f"malformed coefficient row in {path}")
                values.append(float(fields[0]))
                index += 1
            if label == key:
                if not values:
                    raise ValueError(f"empty coefficient {key} in {path}")
                return tuple(values)
            continue
        index += 1
    raise ValueError(f"coefficient {key} not found in {path}")


def _float64_bytes(values) -> bytes:
    return b"".join(struct.pack("=d", value) for value in values)


def read_final_loss(path: Path) -> tuple[str, float]:
    mode = None
    loss = None
    for raw_line in Path(path).read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if line.startswith("Mode ="):
            mode = line.split("=", 1)[1].strip()
        if line.startswith("Sternheimer loss ="):
            loss = float(line.split("=", 1)[1])
    if mode is None or loss is None:
        raise ValueError(f"incomplete final loss metadata in {path}")
    return mode, loss


def read_spillage_rows(path: Path) -> list[SpillageRow]:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    if not lines:
        raise ValueError("Spillage.dat is empty")
    header = lines[0].split()
    required = {"istep_big", "sternheimer", "max_st_condition", "accepted"}
    if not required.issubset(header):
        raise ValueError("Spillage.dat lacks continuation columns")
    columns = {name: header.index(name) for name in required}
    rows = []
    for raw in lines[1:]:
        fields = raw.split()
        if not fields:
            continue
        if fields[columns["accepted"]].lower() != "true":
            continue
        row = SpillageRow(
            int(fields[columns["istep_big"]]),
            float(fields[columns["sternheimer"]]),
            float(fields[columns["max_st_condition"]]),
        )
        if not math.isfinite(row.loss) or not math.isfinite(row.condition):
            raise RuntimeError("accepted continuation row is non-finite")
        rows.append(row)
    if not rows or rows[0].step != -1:
        raise ValueError("Spillage.dat lacks the initial accepted row")
    return rows


def final_window_converged(losses, *, window=100, tolerance=1.0e-4):
    losses = tuple(float(value) for value in losses)
    if len(losses) <= window:
        return False, None
    best_before = min(losses[:-window])
    best_all = min(losses)
    relative_drop = max(0.0, (best_before - best_all) / best_before)
    return relative_drop < tolerance, relative_drop


def detect_nonimprovement_stop(losses, *, optimizer_rows, max_steps=3000):
    losses = tuple(float(value) for value in losses)
    if optimizer_rows < 51 or optimizer_rows >= max_steps or len(losses) < 52:
        return False
    previous_best = min(losses[:-51])
    return all(value >= previous_best for value in losses[-51:])


def assess_convergence(
    losses,
    *,
    optimizer_rows,
    maximum_condition,
    checkpoint_loss=EXPECTED_CHECKPOINT_LOSS,
    condition_limit=1.0e12,
):
    losses = tuple(float(value) for value in losses)
    if not losses or any(not math.isfinite(value) or value < 0.0 for value in losses):
        raise RuntimeError("accepted Sternheimer losses must be finite and nonnegative")
    if not math.isfinite(maximum_condition) or maximum_condition >= condition_limit:
        raise RuntimeError("maximum Sternheimer overlap condition exceeds limit")
    best = min(losses)
    if best >= checkpoint_loss:
        raise RuntimeError("continuation did not improve the accepted checkpoint")
    window_passed, relative_drop = final_window_converged(losses)
    stopped = detect_nonimprovement_stop(losses, optimizer_rows=optimizer_rows)
    return {
        "status": "TZDP_CONVERGED" if window_passed or stopped else "CONTINUE_REQUIRED",
        "best_sternheimer_loss": best,
        "checkpoint_sternheimer_loss": checkpoint_loss,
        "loss_ratio_to_checkpoint": best / checkpoint_loss,
        "optimizer_rows": optimizer_rows,
        "final_100_relative_best_loss_drop": relative_drop,
        "final_window_converged": window_passed,
        "nonimprovement_stop_detected": stopped,
        "maximum_sternheimer_overlap_condition": maximum_condition,
    }


def prepare(args):
    template = json.loads(Path(args.template).read_text(encoding="ascii"))
    validate_template(template)
    records = validate_source_hashes(
        args.target, args.checkpoint, args.reference, args.orbital
    )
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "INPUT"
    input_path.write_text(
        json.dumps(build_input(template, args.target, args.checkpoint), indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    manifest = {
        "status": "prepared",
        "purpose": "converge C 3s3p2d before residual-shell selection",
        "sources": records,
        "input": {"path": str(input_path), "sha256": sha256(input_path)},
        "fixed_dzp": sorted(FIXED_DZP),
        "variable_tzdp_excess": sorted(EXPECTED_VARIABLE),
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
    checkpoint = Path(manifest["sources"]["checkpoint"]["path"])
    final = output / "ORBITAL_RESULTS.txt"
    spillage = output / "Spillage.dat"
    run_log = output / "run.log"
    for path in (reference, checkpoint, final, spillage, run_log):
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
            read_coefficient(checkpoint, key)
        ) != _float64_bytes(read_coefficient(final, key))
    if not any(changed.values()):
        raise RuntimeError("continuation changed none of 3s, 3p and 2d")

    rows = read_spillage_rows(spillage)
    if abs(rows[0].loss - EXPECTED_CHECKPOINT_LOSS) > 1.0e-9:
        raise RuntimeError("optimizer initial loss does not reproduce the checkpoint")
    optimizer_rows = sum(row.step >= 0 for row in rows)
    report = assess_convergence(
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
            "scope": "converged atomic 3s3p2d response gate only",
            "frozen_dzp_bitwise_equal": frozen,
            "variable_tzdp_excess_changed_from_checkpoint": changed,
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
    (output / "TZDP_CONTINUATION_RESULT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--target", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
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
