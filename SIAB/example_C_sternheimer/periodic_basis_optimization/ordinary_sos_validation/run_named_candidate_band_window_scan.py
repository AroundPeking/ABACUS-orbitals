#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

from extract_librpa_frequency_grid import extract_frequency_grid


HARTREE_TO_EV = 27.211386245988
OCCUPATION_TOLERANCE = 1.0e-8


def parse_band_energy_window(path: Path) -> dict[str, float | int]:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 6:
        raise ValueError(f"truncated band_out: {path}")
    nk, nspin, nbands = (int(lines[index].strip()) for index in range(3))
    if min(nk, nspin, nbands) <= 0:
        raise ValueError("band_out dimensions must be positive")

    cursor = 5
    lower_bound = math.inf
    upper_bound = -math.inf
    homo = -math.inf
    lumo = math.inf
    for _ in range(nk * nspin):
        if cursor >= len(lines):
            raise ValueError(f"missing band block header: {path}")
        block = lines[cursor].split()
        cursor += 1
        if len(block) != 2:
            raise ValueError(f"malformed band block header: {path}")

        energies: list[float] = []
        occupations: list[float] = []
        for expected_band in range(1, nbands + 1):
            if cursor >= len(lines):
                raise ValueError(f"truncated band block: {path}")
            fields = lines[cursor].split()
            cursor += 1
            if len(fields) < 4 or int(fields[0]) != expected_band:
                raise ValueError(f"malformed band row: {path}")
            occupations.append(float(fields[1]))
            energies.append(float(fields[2]))

        occupied = [
            index
            for index, value in enumerate(occupations)
            if value > OCCUPATION_TOLERANCE
        ]
        if not occupied or occupied[-1] + 1 >= nbands:
            raise ValueError("each band block must contain occupied and empty states")
        homo = max(homo, energies[occupied[-1]])
        lumo = min(lumo, energies[occupied[-1] + 1])
        lower_bound = min(lower_bound, energies[0])
        upper_bound = max(upper_bound, energies[-1])

    if cursor != len(lines):
        if any(line.strip() for line in lines[cursor:]):
            raise ValueError(f"unexpected trailing band_out content: {path}")
    gap = lumo - homo
    transition = upper_bound - lower_bound
    if not (math.isfinite(gap) and math.isfinite(transition)):
        raise ValueError("non-finite minimax window")
    if gap <= 0.0 or transition <= gap:
        raise ValueError("invalid minimax window")
    return {
        "nk": nk,
        "nspin": nspin,
        "nbands": nbands,
        "minimax_min_gap_ha": gap,
        "minimax_max_transition_ha": transition,
    }


def _read_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            values[fields[0]] = fields[1]
    if values.get("status") != "success":
        raise ValueError(f"endpoint is not successful: {path}")
    return values


def binding_row(
    *,
    atom_bands: int,
    solid_bands: int,
    atom_e0_ha: float,
    solid_e0_ha: float,
    atom_ec_ha: float,
    solid_ec_ha: float,
) -> dict[str, float | int]:
    zero_order = atom_e0_ha - 0.5 * solid_e0_ha
    correlation = atom_ec_ha - 0.5 * solid_ec_ha
    total = zero_order + correlation
    return {
        "atom_bands": atom_bands,
        "solid_bands": solid_bands,
        "atom_ecrpa_ha": atom_ec_ha,
        "solid_ecrpa_ha": solid_ec_ha,
        "zero_order_binding_ha_per_c": zero_order,
        "correlation_binding_ha_per_c": correlation,
        "total_binding_ha_per_c": total,
        "total_binding_ev_per_c": total * HARTREE_TO_EV,
    }


def _replace_parameter(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    replacement = f"{key} = {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text.rstrip() + "\n" + replacement + "\n"


def _ecrpa_from_output(path: Path) -> float:
    matches = re.findall(
        r"^\| Total EcRPA:\s+([-+0-9.eE]+)\s*$",
        path.read_text(encoding="utf-8", errors="strict"),
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError(f"expected one EcRPA value: {path}")
    value = float(matches[0])
    if not math.isfinite(value):
        raise ValueError(f"non-finite EcRPA value: {path}")
    return value


def _run_side(
    *,
    side: str,
    bands: int,
    source: Path,
    work: Path,
    executable: Path,
    mpirun: str,
    energy_window: dict[str, float | int],
    reference_frequency_manifest: Path,
    nfreq: int,
) -> float:
    work.mkdir(parents=True, exist_ok=False)
    input_text = (source / "librpa.in").read_text(encoding="ascii")
    input_text = _replace_parameter(input_text, "n_bands_chi0", str(bands))
    input_text = _replace_parameter(
        input_text,
        "minimax_min_gap",
        f"{energy_window['minimax_min_gap_ha']:.17g}",
    )
    input_text = _replace_parameter(
        input_text,
        "minimax_max_transition",
        f"{energy_window['minimax_max_transition_ha']:.17g}",
    )
    input_path = work / "librpa.in"
    input_path.write_text(input_text, encoding="ascii")

    output = work / "librpa.out"
    with output.open("wb") as handle:
        completed = subprocess.run(
            [mpirun, "-np", "1", str(executable)],
            cwd=work,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{side} LibRPA failed for {bands} bands")
    output_text = output.read_text(encoding="utf-8", errors="strict")
    if "libRPA finished successfully" not in output_text:
        raise RuntimeError(f"{side} LibRPA has no success marker for {bands} bands")

    frequency_manifest = work / "FREQUENCY_GRID.json"
    frequency = extract_frequency_grid(
        source=output,
        output=work / "FREQUENCY_GRID.dat",
        manifest=frequency_manifest,
        expected_nfreq=nfreq,
    )
    reference = json.loads(reference_frequency_manifest.read_text(encoding="ascii"))
    for key in ("frequencies_ha", "weights_ha"):
        if len(frequency[key]) != len(reference[key]):
            raise ValueError(f"{side} frequency-grid length changed")
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-13)
            for actual, expected in zip(frequency[key], reference[key])
        ):
            raise ValueError(f"{side} fixed minimax grid changed for {bands} bands")
    return _ecrpa_from_output(output)


def _parse_windows(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        atom, solid = (int(field) for field in item.split(":", maxsplit=1))
        if atom <= 0 or solid != 2 * atom:
            raise ValueError("each window must be atom:solid with solid=2*atom")
        result.append((atom, solid))
    if not result or len(set(result)) != len(result):
        raise ValueError("band windows must be nonempty and unique")
    return result


def run_scan(
    *,
    run_root: Path,
    output_root: Path,
    executable: Path,
    windows: list[tuple[int, int]],
    mpirun: str,
    nfreq: int,
) -> dict:
    run_root = run_root.resolve(strict=True)
    executable = executable.resolve(strict=True)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"scan output already exists: {output_root}")
    output_root.mkdir(parents=True)

    atom_source = run_root / "atom-sos"
    solid_source = run_root / "solid-sos"
    atom_summary = _read_summary(atom_source / "RESULT_SUMMARY.txt")
    solid_summary = _read_summary(solid_source / "RESULT_SUMMARY.txt")
    if atom_summary["selected_orbital_sha256"] != solid_summary["selected_orbital_sha256"]:
        raise ValueError("atom and solid endpoints use different orbitals")

    atom_window = parse_band_energy_window(atom_source / "reader_v1" / "band_out")
    solid_window = parse_band_energy_window(solid_source / "reader_v1" / "band_out")
    if atom_window["nbands"] * 2 != solid_window["nbands"]:
        raise ValueError("atom and solid all-band dimensions are not 1:2")

    atom_e0 = float(atom_summary["reference_ha"])
    solid_e0 = float(solid_summary["reference_ha"])
    rows: list[dict[str, float | int]] = []
    for atom_bands, solid_bands in windows:
        if atom_bands >= atom_window["nbands"]:
            raise ValueError("diagnostic windows must be smaller than all bands")
        atom_ec = _run_side(
            side="atom",
            bands=atom_bands,
            source=atom_source,
            work=output_root / f"atom-{atom_bands}",
            executable=executable,
            mpirun=mpirun,
            energy_window=atom_window,
            reference_frequency_manifest=atom_source / "ATOM_SOS_FREQUENCY_GRID.json",
            nfreq=nfreq,
        )
        solid_ec = _run_side(
            side="solid",
            bands=solid_bands,
            source=solid_source,
            work=output_root / f"solid-{solid_bands}",
            executable=executable,
            mpirun=mpirun,
            energy_window=solid_window,
            reference_frequency_manifest=solid_source / "SOLID_SOS_FREQUENCY_GRID.json",
            nfreq=nfreq,
        )
        rows.append(
            binding_row(
                atom_bands=atom_bands,
                solid_bands=solid_bands,
                atom_e0_ha=atom_e0,
                solid_e0_ha=solid_e0,
                atom_ec_ha=atom_ec,
                solid_ec_ha=solid_ec,
            )
        )

    rows.append(
        binding_row(
            atom_bands=int(atom_window["nbands"]),
            solid_bands=int(solid_window["nbands"]),
            atom_e0_ha=atom_e0,
            solid_e0_ha=solid_e0,
            atom_ec_ha=float(atom_summary["ecrpa_ha"]),
            solid_ec_ha=float(solid_summary["ecrpa_ha"]),
        )
    )
    payload = {
        "status": "success",
        "quantity": "matched_atom_solid_band_window_scan",
        "selected_orbital_sha256": atom_summary["selected_orbital_sha256"],
        "nfreq": nfreq,
        "atom_minimax_window": atom_window,
        "solid_minimax_window": solid_window,
        "rows": rows,
    }
    (output_root / "RESULT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    columns = [
        "atom_bands",
        "solid_bands",
        "atom_ecrpa_ha",
        "solid_ecrpa_ha",
        "correlation_binding_ha_per_c",
        "total_binding_ev_per_c",
    ]
    lines = ["\t".join(columns)]
    lines.extend(
        "\t".join(str(row[column]) for column in columns)
        for row in rows
    )
    (output_root / "RESULT.tsv").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )
    (output_root / "STATUS").write_text("success\n", encoding="ascii")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse completed reader-v1 outputs for a matched C atom-solid band-window scan"
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--librpa", required=True, type=Path)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--nfreq", type=int, default=6)
    args = parser.parse_args()
    result = run_scan(
        run_root=args.run_root,
        output_root=args.output_root,
        executable=args.librpa,
        windows=_parse_windows(args.windows),
        mpirun=args.mpirun,
        nfreq=args.nfreq,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
