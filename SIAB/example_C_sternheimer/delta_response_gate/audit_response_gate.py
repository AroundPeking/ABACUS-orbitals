#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


COULOMB_MARKER = -20129433
RESPONSE_MARKER = -41073291
HARTREE_TO_KCAL_PER_MOL = 627.5094740631
COULOMB_THRESHOLD = 1.0e-5
PI_RELATIVE_TOLERANCE = 1.0e-3
TRACE_LOG_RELATIVE_TOLERANCE = 1.0e-3
ECRPA_TOLERANCE_KCAL_PER_MOL = 0.1


@dataclass(frozen=True)
class MatrixRecord:
    matrix: np.ndarray
    iq: int
    atom_naux: tuple[int, ...]
    ifreq: int | None = None
    omega: float | None = None
    weight: float | None = None


def _read_exact(handle, size: int, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"truncated {label}")
    return data


def _atom_pair(pair_index: int, natoms: int) -> tuple[int, int]:
    local = pair_index
    for iatom in range(natoms):
        count = natoms - iatom
        if local < count:
            return iatom, iatom + local
        local -= count
    raise ValueError("invalid atom-pair index")


def _read_matrix_file(path: Path, response: bool) -> tuple[dict, list[tuple[int, int, np.ndarray]]]:
    path = path.resolve(strict=True)
    with path.open("rb") as handle:
        if response:
            values = struct.unpack("=6i2di", _read_exact(handle, 44, "response header"))
            marker, iq, ifreq, naux, value_flag, natoms, omega, weight, nblocks = values
            if marker != RESPONSE_MARKER or ifreq <= 0:
                raise ValueError(f"invalid Sternheimer v1 header: {path}")
            if not (math.isfinite(omega) and math.isfinite(weight) and weight > 0.0):
                raise ValueError(f"invalid Sternheimer frequency metadata: {path}")
        else:
            marker, iq, naux, value_flag, natoms, nblocks = struct.unpack(
                "=6i", _read_exact(handle, 24, "Coulomb header")
            )
            if marker != COULOMB_MARKER:
                raise ValueError(f"invalid Coulomb v1 header: {path}")
            ifreq = None
            omega = None
            weight = None
        if iq <= 0 or naux <= 0 or natoms <= 0 or value_flag not in (0, 1):
            raise ValueError(f"invalid v1 dimensions: {path}")
        atom_naux = struct.unpack(
            f"={natoms}i", _read_exact(handle, 4 * natoms, "atom sizes")
        )
        if any(value <= 0 for value in atom_naux) or sum(atom_naux) != naux:
            raise ValueError(f"invalid atom_naux table: {path}")
        expected_pairs = natoms * (natoms + 1) // 2
        if nblocks < 0 or nblocks > expected_pairs:
            raise ValueError(f"invalid block count: {path}")
        table = [
            struct.unpack("=iq", _read_exact(handle, 12, "block table"))
            for _ in range(nblocks)
        ]
        if len({pair for pair, _ in table}) != len(table):
            raise ValueError(f"duplicate atom pair in {path}")
        blocks = []
        for pair_index, offset in table:
            iatom, jatom = _atom_pair(pair_index, natoms)
            count = atom_naux[iatom] * atom_naux[jatom]
            handle.seek(offset)
            raw = _read_exact(handle, count * (16 if value_flag else 8), "matrix payload")
            dtype = np.complex128 if value_flag else np.float64
            block = np.frombuffer(raw, dtype=dtype).astype(np.complex128).reshape(
                atom_naux[iatom], atom_naux[jatom]
            )
            blocks.append((iatom, jatom, block))
    metadata = {
        "path": path,
        "iq": iq,
        "ifreq": ifreq,
        "naux": naux,
        "natoms": natoms,
        "atom_naux": tuple(atom_naux),
        "omega": omega,
        "weight": weight,
    }
    return metadata, blocks


def _assemble(metadata: dict, blocks: Iterable[tuple[int, int, np.ndarray]]) -> np.ndarray:
    atom_naux = metadata["atom_naux"]
    offsets = np.cumsum((0, *atom_naux))
    matrix = np.zeros((metadata["naux"], metadata["naux"]), dtype=np.complex128)
    seen = set()
    for iatom, jatom, block in blocks:
        if (iatom, jatom) in seen:
            raise ValueError("duplicate atom-pair block across v1 shards")
        seen.add((iatom, jatom))
        i0, i1 = offsets[iatom], offsets[iatom + 1]
        j0, j1 = offsets[jatom], offsets[jatom + 1]
        matrix[i0:i1, j0:j1] = block
        if iatom != jatom:
            matrix[j0:j1, i0:i1] = block.conj().T
    expected = len(atom_naux) * (len(atom_naux) + 1) // 2
    if len(seen) != expected:
        raise ValueError("missing atom-pair blocks in v1 matrix")
    return matrix


def read_coulomb_v1(paths: Iterable[Path]) -> MatrixRecord:
    records = [_read_matrix_file(Path(path), response=False) for path in paths]
    if not records:
        raise ValueError("no Coulomb v1 files")
    reference = records[0][0]
    for metadata, _ in records[1:]:
        for key in ("iq", "naux", "natoms", "atom_naux"):
            if metadata[key] != reference[key]:
                raise ValueError("inconsistent Coulomb shard metadata")
    matrix = _assemble(reference, (block for _, blocks in records for block in blocks))
    return MatrixRecord(matrix=matrix, iq=reference["iq"], atom_naux=reference["atom_naux"])


def read_response_v1(path: Path) -> MatrixRecord:
    metadata, blocks = _read_matrix_file(Path(path), response=True)
    matrix = _assemble(metadata, blocks)
    return MatrixRecord(
        matrix=matrix,
        iq=metadata["iq"],
        atom_naux=metadata["atom_naux"],
        ifreq=metadata["ifreq"],
        omega=metadata["omega"],
        weight=metadata["weight"],
    )


def _hermitize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def compute_pi_spectrum(coulomb: np.ndarray, response_m: np.ndarray) -> np.ndarray:
    coulomb = _hermitize(coulomb)
    response_m = _hermitize(response_m)
    eigenvalues, eigenvectors = np.linalg.eigh(coulomb)
    inverse_sqrt = np.zeros_like(eigenvalues)
    retained = eigenvalues > COULOMB_THRESHOLD
    inverse_sqrt[retained] = 1.0 / np.sqrt(eigenvalues[retained])
    metric = (eigenvectors * inverse_sqrt) @ eigenvectors.conj().T
    pi = _hermitize(metric @ response_m @ metric)
    return np.linalg.eigvalsh(pi)


def _relative_max(left: np.ndarray, right: np.ndarray, floor: float = 1.0e-10) -> float:
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), floor)
    return float(np.max(np.abs(left - right) / denominator))


def _load_completion(case: Path) -> tuple[dict, dict]:
    response = json.loads((case / "RESPONSE_COMPLETE.json").read_text(encoding="ascii"))
    librpa = json.loads(
        (case / "librpa/LIBRPA_COMPLETE.json").read_text(encoding="ascii")
    )
    if response.get("status") != "RESPONSE_COMPLETE":
        raise ValueError(f"incomplete ABACUS response in {case}")
    if librpa.get("status") != "LIBRPA_COMPLETE" or librpa.get("coulomb_kernel") != "full":
        raise ValueError(f"incomplete full-Coulomb LibRPA result in {case}")
    return response, librpa


def audit_response_gate(root: Path) -> dict:
    root = root.resolve(strict=True)
    cases = {branch: root / "branches" / branch for branch in ("fixed", "free")}
    completions = {branch: _load_completion(case) for branch, case in cases.items()}
    coulomb = {
        branch: read_coulomb_v1(sorted(case.glob("v1_coulomb_full_iq_1_rank*.dat")))
        for branch, case in cases.items()
    }
    if coulomb["fixed"].atom_naux != coulomb["free"].atom_naux:
        raise ValueError("fixed/free auxiliary layouts differ")
    vfixed = coulomb["fixed"].matrix
    vfree = coulomb["free"].matrix
    coulomb_scale = max(float(np.linalg.norm(vfixed)), float(np.linalg.norm(vfree)), 1.0)
    coulomb_relative_difference = float(np.linalg.norm(vfixed - vfree) / coulomb_scale)

    responses = {}
    for branch, case in cases.items():
        records = [
            read_response_v1(path)
            for path in sorted(case.glob("v1_sternheimer_chi0_iq_1_ifreq_*.dat"))
        ]
        records.sort(key=lambda record: record.ifreq)
        if [record.ifreq for record in records] != list(range(1, 7)):
            raise ValueError(f"{branch} response frequencies are not exactly 1..6")
        responses[branch] = records

    frequency_results = []
    max_pi_relative = 0.0
    max_m_relative = 0.0
    for fixed, free in zip(responses["fixed"], responses["free"]):
        if fixed.atom_naux != free.atom_naux:
            raise ValueError("fixed/free response layouts differ")
        if not math.isclose(fixed.omega, free.omega, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise ValueError("fixed/free frequencies differ")
        if not math.isclose(fixed.weight, free.weight, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise ValueError("fixed/free frequency weights differ")
        spectrum_fixed = compute_pi_spectrum(vfixed, fixed.matrix)
        spectrum_free = compute_pi_spectrum(vfree, free.matrix)
        pi_relative = _relative_max(spectrum_fixed, spectrum_free)
        m_scale = max(
            float(np.linalg.norm(fixed.matrix)), float(np.linalg.norm(free.matrix)), 1.0e-10
        )
        m_relative = float(np.linalg.norm(fixed.matrix - free.matrix) / m_scale)
        max_pi_relative = max(max_pi_relative, pi_relative)
        max_m_relative = max(max_m_relative, m_relative)
        frequency_results.append(
            {
                "ifreq": fixed.ifreq,
                "omega_ha": fixed.omega,
                "weight_ha": fixed.weight,
                "pi_spectrum_relative_difference": pi_relative,
                "raw_m_frobenius_relative_difference": m_relative,
                "pi_eigenvalues_fixed": spectrum_fixed.tolist(),
                "pi_eigenvalues_free": spectrum_free.tolist(),
            }
        )

    librpa = {branch: completions[branch][1] for branch in cases}
    trace_rows = {
        branch: {int(row["ifreq"]): row["values"] for row in data["trace_log_rows"]}
        for branch, data in librpa.items()
    }
    if set(trace_rows["fixed"]) != set(range(1, 7)) or set(trace_rows["free"]) != set(range(1, 7)):
        raise ValueError("LibRPA trace-log rows do not cover six frequencies")
    trace_log_relative = 0.0
    for item in frequency_results:
        ifreq = item["ifreq"]
        fixed_integrand = float(trace_rows["fixed"][ifreq][2])
        free_integrand = float(trace_rows["free"][ifreq][2])
        difference = abs(fixed_integrand - free_integrand) / max(
            abs(fixed_integrand), abs(free_integrand), 1.0e-10
        )
        item["trace_log_integrand_fixed"] = fixed_integrand
        item["trace_log_integrand_free"] = free_integrand
        item["trace_log_relative_difference"] = difference
        trace_log_relative = max(trace_log_relative, difference)

    ecrpa_fixed = float(librpa["fixed"]["ecrpa_ha"])
    ecrpa_free = float(librpa["free"]["ecrpa_ha"])
    ecrpa_difference = abs(ecrpa_fixed - ecrpa_free) * HARTREE_TO_KCAL_PER_MOL
    blocked_on = []
    if coulomb_relative_difference > 1.0e-10:
        blocked_on.append("full_coulomb_matrix")
    if max_pi_relative > PI_RELATIVE_TOLERANCE:
        blocked_on.append("pi_spectrum")
    if trace_log_relative > TRACE_LOG_RELATIVE_TOLERANCE:
        blocked_on.append("trace_log_integrand")
    if ecrpa_difference > ECRPA_TOLERANCE_KCAL_PER_MOL:
        blocked_on.append("ecrpa")
    return {
        "status": "DELTA_RESPONSE_GATE_BLOCKED" if blocked_on else "DELTA_RESPONSE_GATE_PASSED",
        "blocked_on": blocked_on or None,
        "definition": "Pi=V_full^(-1/2) M V_full^(-1/2)",
        "sqrt_coulomb_threshold": COULOMB_THRESHOLD,
        "pi_spectrum_relative_tolerance": PI_RELATIVE_TOLERANCE,
        "trace_log_relative_tolerance": TRACE_LOG_RELATIVE_TOLERANCE,
        "ecrpa_tolerance_kcal_per_mol": ECRPA_TOLERANCE_KCAL_PER_MOL,
        "full_coulomb_frobenius_relative_difference": coulomb_relative_difference,
        "max_pi_spectrum_relative_difference": max_pi_relative,
        "max_raw_m_frobenius_relative_difference": max_m_relative,
        "max_trace_log_relative_difference": trace_log_relative,
        "ecrpa_fixed_ha": ecrpa_fixed,
        "ecrpa_free_ha": ecrpa_free,
        "ecrpa_difference_kcal_per_mol": ecrpa_difference,
        "frequencies": frequency_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = audit_response_gate(args.root)
    output_json = args.root / "DELTA_RESPONSE_GATE_RESULT.json"
    output_text = args.root / "DELTA_RESPONSE_GATE_RESULT.txt"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    lines = [
        f"status={result['status']}",
        f"blocked_on={result['blocked_on']}",
        f"max_pi_spectrum_relative_difference={result['max_pi_spectrum_relative_difference']:.12e}",
        f"max_trace_log_relative_difference={result['max_trace_log_relative_difference']:.12e}",
        f"ecrpa_fixed_ha={result['ecrpa_fixed_ha']:.16e}",
        f"ecrpa_free_ha={result['ecrpa_free_ha']:.16e}",
        f"ecrpa_difference_kcal_per_mol={result['ecrpa_difference_kcal_per_mol']:.12e}",
    ]
    output_text.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "DELTA_RESPONSE_GATE_PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
