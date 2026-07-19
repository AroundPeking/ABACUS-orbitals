#!/usr/bin/env python3
"""Run and validate the deterministic H Sternheimer-only SIAB campaign."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


EXAMPLE_DIR = Path(__file__).resolve().parent
SIAB_DIR = EXAMPLE_DIR.parent
REPO_ROOT = SIAB_DIR.parent
DEFAULT_TEMPLATE = EXAMPLE_DIR / "INPUT.st_only"
DEFAULT_INITIAL = (
    REPO_ROOT
    / "Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/ORBITAL_RESULTS.txt"
)
DEFAULT_REFERENCE_ORBITAL = (
    REPO_ROOT
    / "Dojo-NC-SR/Orbitals_v2.0/H_TZDP/H_gga_8au_100Ry_3s2p.orb"
)
OPTIMIZER = SIAB_DIR / "opt_orb_pytorch_dpsi/main.py"
FIXED_DZP_ORBITALS = (
    {"label": "1s", "element": "H", "l": 0, "zeta": 1},
    {"label": "2s", "element": "H", "l": 0, "zeta": 2},
    {"label": "1p", "element": "H", "l": 1, "zeta": 1},
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_input(template, target, initial_coefficients):
    result = copy.deepcopy(template)
    result["file_list"] = {"sternheimer": [str(Path(target).resolve())]}
    result["C_init_info"]["C_init_file"] = str(
        Path(initial_coefficients).resolve()
    )
    return result


def read_coefficient(path, element, l, zeta):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    in_coefficients = False
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line == "<Coefficient>":
            in_coefficients = True
            index += 1
            continue
        if line == "</Coefficient>":
            break
        if in_coefficients and line.startswith("Type"):
            index += 1
            if index >= len(lines):
                raise ValueError(f"missing coefficient label in {path}")
            fields = lines[index].split()
            if len(fields) != 3:
                raise ValueError(f"invalid coefficient label in {path}")
            label = (fields[0], int(fields[1]), int(fields[2]))
            values = []
            index += 1
            while index < len(lines):
                value_line = lines[index].strip()
                if (
                    value_line.startswith("Type")
                    or value_line == "</Coefficient>"
                ):
                    break
                fields = value_line.split()
                if len(fields) != 1:
                    raise ValueError(f"invalid coefficient value in {path}")
                values.append(float(fields[0]))
                index += 1
            if label == (element, int(l), int(zeta)):
                if not values:
                    raise ValueError(f"empty coefficient column {label} in {path}")
                return tuple(values)
            continue
        index += 1
    raise ValueError(
        f"coefficient column {(element, int(l), int(zeta))} not found in {path}"
    )


def _float64_bytes(values):
    return b"".join(struct.pack("=d", value) for value in values)


def read_orbital(path, l, zeta):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    mesh = None
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] == "Mesh":
            mesh = int(fields[1])
            break
    if mesh is None or mesh <= 0:
        raise ValueError(f"invalid or missing Mesh in {path}")

    index = 0
    while index < len(lines):
        fields = lines[index].split()
        if fields == ["Type", "L", "N"]:
            index += 1
            if index >= len(lines):
                raise ValueError(f"missing orbital label in {path}")
            label_fields = lines[index].split()
            if len(label_fields) != 3:
                raise ValueError(f"invalid orbital label in {path}")
            label = (int(label_fields[1]), int(label_fields[2]))
            values = []
            index += 1
            while index < len(lines):
                value_fields = lines[index].split()
                if value_fields == ["Type", "L", "N"]:
                    break
                for value in value_fields:
                    values.append(float(value))
                index += 1
            if label == (int(l), int(zeta)):
                if len(values) != mesh:
                    raise ValueError(
                        f"orbital {label} has {len(values)} points, expected {mesh}"
                    )
                return tuple(values)
            continue
        index += 1
    raise ValueError(f"orbital {(int(l), int(zeta))} not found in {path}")


def _read_initial_sternheimer_loss(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if "sternheimer" not in fields or "istep_big" not in fields:
            continue
        column = fields.index("sternheimer")
        for data_line in lines[index + 1 :]:
            values = data_line.split()
            if not values or values[0] == "istep_big":
                continue
            if len(values) <= column:
                raise ValueError(f"incomplete Spillage.dat row: {data_line}")
            return float(values[column])
    raise ValueError(f"no Sternheimer loss table found in {path}")


def _read_loss_metadata(path):
    labels = {
        "DFT origin loss": "dft_origin",
        "DFT dpsi loss": "dft_dpsi",
        "Sternheimer loss": "sternheimer",
        "dpsi regularization loss": "regularization_dpsi",
        "DFT constraint loss": "constraint_dft",
        "dpsi constraint loss": "constraint_dpsi",
        "Total loss": "total",
    }
    values = {}
    mode = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("Mode ="):
            mode = line.split("=", 1)[1].strip()
        for label, key in labels.items():
            prefix = label + " ="
            if line.startswith(prefix):
                values[key] = float(line.split("=", 1)[1])
                break
    if mode is None or set(values) != set(labels.values()):
        raise ValueError(f"incomplete loss metadata in {path}")
    values["mode"] = mode
    return values


def summarize_campaign(
    target,
    initial_coefficients,
    reference_orbital,
    output_dir,
    elapsed_seconds,
):
    target = Path(target).resolve()
    initial_coefficients = Path(initial_coefficients).resolve()
    reference_orbital = Path(reference_orbital).resolve()
    output_dir = Path(output_dir).resolve()
    final_coefficients = output_dir / "ORBITAL_RESULTS.txt"
    final_orbital = output_dir / "ORBITAL_1U.dat"
    spillage = output_dir / "Spillage.dat"
    input_path = output_dir / "INPUT"
    log_path = output_dir / "run.log"
    required = (
        target,
        initial_coefficients,
        reference_orbital,
        final_coefficients,
        final_orbital,
        spillage,
        input_path,
        log_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    fixed_dzp_orbitals = []
    for orbital in FIXED_DZP_ORBITALS:
        label = orbital["label"]
        element = orbital["element"]
        l = orbital["l"]
        zeta = orbital["zeta"]
        initial_values = read_coefficient(
            initial_coefficients, element, l, zeta
        )
        final_values = read_coefficient(final_coefficients, element, l, zeta)
        coefficient_equal = _float64_bytes(initial_values) == _float64_bytes(
            final_values
        )
        if not coefficient_equal:
            raise RuntimeError(f"fixed H DZP {label} coefficient changed")

        radial_reference = read_orbital(reference_orbital, l, zeta - 1)
        radial_final = read_orbital(final_orbital, l, zeta - 1)
        if len(radial_reference) != len(radial_final):
            raise RuntimeError(f"fixed H DZP {label} radial mesh changed")
        radial_abs_errors = [
            abs(reference - final)
            for reference, final in zip(radial_reference, radial_final)
        ]
        radial_rel_errors = [
            error / max(abs(reference), 1.0e-14)
            for error, reference in zip(radial_abs_errors, radial_reference)
        ]
        radial_matches = all(
            error <= 5.0e-13 + 5.0e-13 * abs(reference)
            for error, reference in zip(radial_abs_errors, radial_reference)
        )
        if not radial_matches:
            raise RuntimeError(
                f"exported H DZP {label} radial orbital differs from "
                "the TZDP reference"
            )
        fixed_dzp_orbitals.append(
            {
                "label": label,
                "element": element,
                "l": l,
                "zeta": zeta,
                "n_coefficients": len(initial_values),
                "coefficient_bitwise_equal": coefficient_equal,
                "radial_n_points": len(radial_reference),
                "radial_matches_reference": radial_matches,
                "radial_max_abs_error": max(radial_abs_errors),
                "radial_max_rel_error": max(radial_rel_errors),
            }
        )

    initial_loss = _read_initial_sternheimer_loss(spillage)
    final_loss = _read_loss_metadata(final_coefficients)
    if final_loss["mode"] != "st_only":
        raise RuntimeError(f"unexpected loss mode {final_loss['mode']!r}")
    for name in (
        "dft_origin",
        "dft_dpsi",
        "regularization_dpsi",
        "constraint_dft",
        "constraint_dpsi",
    ):
        if final_loss[name] != 0.0:
            raise RuntimeError(f"pure ST campaign evaluated nonzero {name}")
    if final_loss["sternheimer"] > initial_loss:
        raise RuntimeError("best Sternheimer loss is worse than the initial loss")

    return {
        "mode": "st_only",
        "seed": json.loads(input_path.read_text(encoding="utf-8"))["seed"],
        "target": str(target),
        "target_sha256": sha256(target),
        "initial_coefficients": str(initial_coefficients),
        "initial_coefficients_sha256": sha256(initial_coefficients),
        "reference_orbital": str(reference_orbital),
        "reference_orbital_sha256": sha256(reference_orbital),
        "final_coefficients": str(final_coefficients),
        "final_coefficients_sha256": sha256(final_coefficients),
        "final_orbital": str(final_orbital),
        "final_orbital_sha256": sha256(final_orbital),
        "input_sha256": sha256(input_path),
        "spillage_sha256": sha256(spillage),
        "initial_sternheimer_loss": initial_loss,
        "final_sternheimer_loss": final_loss["sternheimer"],
        "loss_ratio": final_loss["sternheimer"] / initial_loss,
        "fixed_dzp_orbitals": fixed_dzp_orbitals,
        "fixed_dzp_all_coefficients_bitwise_equal": all(
            item["coefficient_bitwise_equal"] for item in fixed_dzp_orbitals
        ),
        "fixed_dzp_all_radials_match_reference": all(
            item["radial_matches_reference"] for item in fixed_dzp_orbitals
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }


def run_campaign(
    target,
    output_dir,
    template_path,
    initial_coefficients,
    reference_orbital,
    python,
):
    target = Path(target).resolve()
    output_dir = Path(output_dir).resolve()
    template_path = Path(template_path).resolve()
    initial_coefficients = Path(initial_coefficients).resolve()
    reference_orbital = Path(reference_orbital).resolve()
    for path in (
        target,
        template_path,
        initial_coefficients,
        reference_orbital,
        OPTIMIZER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"campaign output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    template = json.loads(template_path.read_text(encoding="utf-8"))
    campaign_input = build_input(template, target, initial_coefficients)
    input_path = output_dir / "INPUT"
    input_path.write_text(
        json.dumps(campaign_input, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    python_path = str(OPTIMIZER.parent)
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    environment.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    command = [str(Path(python).resolve()), str(OPTIMIZER)]
    start = time.monotonic()
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"SIAB optimizer exited with {completed.returncode}; "
            f"see {output_dir / 'run.log'}"
        )

    report = summarize_campaign(
        target,
        initial_coefficients,
        reference_orbital,
        output_dir,
        elapsed,
    )
    report["command"] = command
    report_path = output_dir / "campaign_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument(
        "--reference-orbital", type=Path, default=DEFAULT_REFERENCE_ORBITAL
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_campaign(
        target=args.target,
        output_dir=args.output,
        template_path=args.template,
        initial_coefficients=args.initial,
        reference_orbital=args.reference_orbital,
        python=args.python,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
