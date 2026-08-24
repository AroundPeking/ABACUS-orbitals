#!/usr/bin/env python3
"""Prepare and audit the C atomic Sternheimer SIAB gradient gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_TEMPLATE = HERE / "SIAB_INPUT.atomic_gradient_gate.json"
DEFAULT_COEFFICIENTS = (
    REPO_ROOT / "SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt"
)
DEFAULT_ORBITAL = (
    REPO_ROOT / "SG15_v1.0/Orbitals_v2.0/C_TZDP/C_gga_10au_100Ry_3s3p2d.orb"
)

EXPECTED_HASHES = {
    "target": "e976c164595758029cb91ebe3913af6865780ef95fdb48954c07075bd0c7e3ff",
    "coefficients": "b58a2183c3028e46e6f4bc55b0f21531f1253275d5c2f2c4ee4e27676c1b55f4",
    "orbital": "7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d",
}
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
        raise ValueError("atomic gate must contain only C")
    if template["element"].get("Nu") != {"C": [3, 3, 2, 0, 0]}:
        raise ValueError(
            "atomic gate must use C 3s3p2d with explicit zero f/g channels"
        )
    if freeze_keys(template.get("freeze_orbitals", [])) != FIXED_DZP:
        raise ValueError("atomic gate must use the exact C DZP freeze set")
    if variable_keys(template) != EXPECTED_VARIABLE:
        raise ValueError("atomic gate must vary only C 3s, 3p and 2d")
    stages = template.get("optimize", [])
    if len(stages) != 1 or stages[0].get("max_steps") != 20:
        raise ValueError("atomic gradient gate must run exactly 20 steps")
    if stages[0].get("optimizer") != "Adam" or stages[0].get("kwargs") != {
        "lr": 0.001
    }:
        raise ValueError("atomic gradient gate must use Adam with lr=0.001")
    if template.get("loss", {}).get("mode") != "st_only":
        raise ValueError("atomic gradient gate must use st_only")
    radial = template.get("radial", {})
    if (radial.get("Rcut"), radial.get("dr"), radial.get("Ecut")) != (
        10,
        0.01,
        100,
    ):
        raise ValueError("atomic gate radial contract must be 10 au/0.01/100 Ry")


def validate_source_hashes(
    target: Path,
    coefficients: Path,
    orbital: Path,
    *,
    expected=EXPECTED_HASHES,
) -> dict[str, dict[str, object]]:
    records = {}
    for label, path in (
        ("target", target),
        ("coefficients", coefficients),
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


def build_input(template, target: Path, coefficients: Path) -> dict:
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
    result["C_init_info"]["C_init_file"] = str(Path(coefficients).resolve())
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


def read_initial_loss(path: Path) -> float:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if "sternheimer" not in fields or "istep_big" not in fields:
            continue
        column = fields.index("sternheimer")
        for row in lines[index + 1 :]:
            values = row.split()
            if values and values[0] != "istep_big":
                return float(values[column])
    raise ValueError(f"no Sternheimer loss row in {path}")


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


def prepare(args) -> dict:
    template = json.loads(Path(args.template).read_text(encoding="ascii"))
    validate_template(template)
    records = validate_source_hashes(args.target, args.coefficients, args.orbital)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    value = build_input(template, args.target, args.coefficients)
    input_path = output / "INPUT"
    input_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    manifest = {
        "status": "prepared",
        "purpose": "C atomic input-and-gradient gate, not a production basis",
        "sources": records,
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
        },
        "fixed_dzp": sorted(FIXED_DZP),
        "variable_tzdp_excess": sorted(EXPECTED_VARIABLE),
    }
    manifest_path = output / "PREPARATION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return manifest


def audit(args) -> dict:
    output = Path(args.output).resolve()
    manifest_path = output / "PREPARATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    initial = Path(manifest["sources"]["coefficients"]["path"])
    final = output / "ORBITAL_RESULTS.txt"
    spillage = output / "Spillage.dat"
    run_log = output / "run.log"
    for path in (initial, final, spillage, run_log):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    frozen = {}
    for key in sorted(FIXED_DZP):
        before = read_coefficient(initial, key)
        after = read_coefficient(final, key)
        equal = _float64_bytes(before) == _float64_bytes(after)
        frozen["/".join(map(str, key))] = equal
        if not equal:
            raise RuntimeError(f"frozen DZP coefficient changed: {key}")

    changed = {}
    for key in sorted(EXPECTED_VARIABLE):
        before = read_coefficient(initial, key)
        after = read_coefficient(final, key)
        changed["/".join(map(str, key))] = (
            _float64_bytes(before) != _float64_bytes(after)
        )
    if not any(changed.values()):
        raise RuntimeError("all C 3s/3p/2d variable coefficients remained unchanged")

    initial_loss = read_initial_loss(spillage)
    mode, final_loss = read_final_loss(final)
    if mode != "st_only":
        raise RuntimeError(f"unexpected loss mode: {mode}")
    if not all(math.isfinite(value) and value >= 0.0 for value in (initial_loss, final_loss)):
        raise RuntimeError("Sternheimer loss is non-finite or negative")
    if final_loss > initial_loss:
        raise RuntimeError("best final Sternheimer loss is worse than the initial loss")

    report = {
        "status": "ATOMIC_GRADIENT_GATE_PASSED",
        "scope": "input, freeze and optimizer-gradient gate only",
        "initial_sternheimer_loss": initial_loss,
        "final_sternheimer_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "frozen_dzp_bitwise_equal": frozen,
        "variable_tzdp_excess_changed": changed,
        "outputs": {
            name: {"size": path.stat().st_size, "sha256": sha256(path)}
            for name, path in (
                ("coefficients", final),
                ("spillage", spillage),
                ("run_log", run_log),
            )
        },
    }
    (output / "ATOMIC_GRADIENT_GATE_RESULT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--target", type=Path, required=True)
    prepare_parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    prepare_parser.add_argument("--orbital", type=Path, default=DEFAULT_ORBITAL)
    prepare_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    prepare_parser.add_argument("--output", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = prepare(args) if args.command == "prepare" else audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
