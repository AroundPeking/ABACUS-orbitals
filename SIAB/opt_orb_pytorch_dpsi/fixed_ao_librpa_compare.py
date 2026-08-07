#!/usr/bin/env python3
"""Compare fixed-AO Galerkin response with LibRPA full-band SOS."""

import argparse
import glob
import json
import math
import pathlib
import re
import struct

import numpy as np

from fixed_ao_galerkin import evaluate_fixed_ao_sidecar
from IO.read_sternheimer_fixed_ao import read_sternheimer_fixed_ao


_COULOMB_MARKER = -20129433
_SOS_NAME = re.compile(
    r"chi0fq_ifreq_(\d+)_iq_(\d+)_I_(\d+)_J_(\d+)_id_(\d+)\.mtx$"
)
_ECRPA = re.compile(r"\| Total EcRPA:\s+([-+0-9.eE]+)")


def hermitize(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conjugate().T)


def compare_response(coulomb, m_response, chi_response, threshold=0.0):
    coulomb = hermitize(coulomb)
    m_response = hermitize(m_response)
    chi_response = hermitize(chi_response)
    if coulomb.ndim != 2 or coulomb.shape[0] != coulomb.shape[1]:
        raise ValueError("Coulomb matrix must be square")
    if m_response.shape != coulomb.shape or chi_response.shape != coulomb.shape:
        raise ValueError("response dimensions must match the Coulomb matrix")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")

    eigenvalue, eigenvector = np.linalg.eigh(coulomb)
    keep = eigenvalue > threshold
    if not np.any(keep):
        raise ValueError("no positive Coulomb eigenvalue survives the threshold")
    value = eigenvalue[keep]
    vector = eigenvector[:, keep]
    m_positive = vector.conjugate().T @ m_response @ vector
    chi_positive = vector.conjugate().T @ chi_response @ vector
    m_sos = value[:, None] * chi_positive * value[None, :]
    pi_galerkin = hermitize(
        m_positive / np.sqrt(value)[:, None] / np.sqrt(value)[None, :]
    )
    pi_sos = hermitize(
        np.sqrt(value)[:, None] * chi_positive * np.sqrt(value)[None, :]
    )

    m_difference = m_positive - m_sos
    pi_difference = pi_galerkin - pi_sos
    integrand_galerkin = trace_log_integrand(pi_galerkin)
    integrand_sos = trace_log_integrand(pi_sos)
    return {
        "n_coulomb_positive": int(np.count_nonzero(keep)),
        "n_coulomb_dropped": int(len(keep) - np.count_nonzero(keep)),
        "coulomb_eigenvalue_min": float(eigenvalue[0]),
        "coulomb_eigenvalue_max": float(eigenvalue[-1]),
        "m_relative_frobenius": relative_error(m_positive, m_sos),
        "m_maximum_absolute": float(np.max(np.abs(m_difference))),
        "pi_relative_frobenius": relative_error(pi_galerkin, pi_sos),
        "pi_maximum_absolute": float(np.max(np.abs(pi_difference))),
        "integrand_galerkin": integrand_galerkin,
        "integrand_sos": integrand_sos,
        "integrand_absolute_error": abs(integrand_galerkin - integrand_sos),
        "pi_galerkin_eigenvalue_min": float(np.linalg.eigvalsh(pi_galerkin)[0]),
        "pi_sos_eigenvalue_min": float(np.linalg.eigvalsh(pi_sos)[0]),
    }


def relative_error(actual, reference):
    denominator = np.linalg.norm(reference)
    difference = np.linalg.norm(actual - reference)
    if denominator == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return float(difference / denominator)


def trace_log_integrand(pi_matrix):
    eigenvalue = np.linalg.eigvalsh(hermitize(pi_matrix))
    if np.min(1.0 - eigenvalue) <= 0.0:
        raise ValueError("non-positive trace-log argument")
    return float(np.sum(np.log(1.0 - eigenvalue) + eigenvalue))


def read_frequency_grid(path):
    rows = []
    reading = False
    for raw in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "Frequency node & weight:":
            reading = True
            continue
        if not reading:
            continue
        if line == "Time node & weight:":
            break
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("invalid LibRPA frequency-grid row")
        expected = len(rows)
        if int(fields[0]) != expected:
            raise ValueError("non-contiguous LibRPA frequency-grid index")
        rows.append((float(fields[1]), float(fields[2])))
    if not rows:
        raise ValueError("LibRPA frequency grid was not found")
    return tuple(np.asarray(values, dtype=np.float64) for values in zip(*rows))


def read_ecrpa(path):
    match = _ECRPA.search(pathlib.Path(path).read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("LibRPA EcRPA was not found")
    return float(match.group(1))


def read_matrix_market(path):
    with pathlib.Path(path).open(encoding="ascii") as handle:
        banner = handle.readline().strip().lower()
        if banner != "%%matrixmarket matrix coordinate complex general":
            raise ValueError(f"{path}: unsupported MatrixMarket banner")
        line = handle.readline()
        while line.startswith("%"):
            line = handle.readline()
        nrow, ncol, nvalue = (int(value) for value in line.split())
        matrix = np.zeros((nrow, ncol), dtype=np.complex128)
        count = 0
        for raw in handle:
            if not raw.strip() or raw.startswith("%"):
                continue
            row, column, real, imaginary = raw.split()
            matrix[int(row) - 1, int(column) - 1] += complex(
                float(real), float(imaginary)
            )
            count += 1
    if count != nvalue:
        raise ValueError(f"{path}: MatrixMarket value count mismatch")
    return matrix


def _atom_pairs(natom):
    return [
        (iatom, jatom)
        for iatom in range(natom)
        for jatom in range(iatom, natom)
    ]


def _offsets(atom_naux):
    result = [0]
    for count in atom_naux:
        if count <= 0:
            raise ValueError("per-atom auxiliary dimensions must be positive")
        result.append(result[-1] + count)
    return result


def _read_coulomb_shard(path):
    data = pathlib.Path(path).read_bytes()
    header = struct.Struct("<6i")
    if len(data) < header.size:
        raise ValueError(f"{path}: truncated Coulomb header")
    marker, iq, naux, value_flag, natom, nblock = header.unpack_from(data)
    if marker != _COULOMB_MARKER or iq <= 0 or naux <= 0 or natom <= 0:
        raise ValueError(f"{path}: invalid Coulomb header")
    if value_flag not in (0, 1) or nblock < 0:
        raise ValueError(f"{path}: invalid Coulomb value metadata")
    cursor = header.size
    atom_struct = struct.Struct(f"<{natom}i")
    atom_naux = list(atom_struct.unpack_from(data, cursor))
    cursor += atom_struct.size
    if sum(atom_naux) != naux:
        raise ValueError(f"{path}: auxiliary dimensions do not sum to naux")
    record = struct.Struct("<iq")
    table_end = cursor + nblock * record.size
    records = []
    for _ in range(nblock):
        records.append(record.unpack_from(data, cursor))
        cursor += record.size
    scalar = struct.Struct("<d" if value_flag == 0 else "<2d")
    pairs = _atom_pairs(natom)
    blocks = {}
    for pair_index, begin in records:
        if pair_index < 0 or pair_index >= len(pairs) or pair_index in blocks:
            raise ValueError(f"{path}: invalid Coulomb pair index")
        iatom, jatom = pairs[pair_index]
        count = atom_naux[iatom] * atom_naux[jatom]
        if begin < table_end or begin + count * scalar.size > len(data):
            raise ValueError(f"{path}: invalid Coulomb payload range")
        values = np.empty(count, dtype=np.complex128)
        for index in range(count):
            raw = scalar.unpack_from(data, begin + index * scalar.size)
            values[index] = complex(raw[0], 0.0 if value_flag == 0 else raw[1])
        blocks[pair_index] = values.reshape(atom_naux[iatom], atom_naux[jatom])
    return iq, atom_naux, blocks


def read_coulomb(directory, iq=1):
    directory = pathlib.Path(directory)
    paths = sorted(directory.glob(f"v1_coulomb_full_iq_{iq}_rank*.dat"))
    unsharded = directory / f"v1_coulomb_full_iq_{iq}.dat"
    if unsharded.is_file():
        paths.append(unsharded)
    if not paths:
        raise ValueError(f"full Coulomb reader-v1 files missing for iq={iq}")
    parsed = [_read_coulomb_shard(path) for path in paths]
    atom_naux = parsed[0][1]
    if any(item[0] != iq or item[1] != atom_naux for item in parsed):
        raise ValueError("inconsistent full Coulomb shards")
    pairs = _atom_pairs(len(atom_naux))
    offsets = _offsets(atom_naux)
    matrix = np.zeros((offsets[-1], offsets[-1]), dtype=np.complex128)
    seen = set()
    for _, _, blocks in parsed:
        for pair_index, block in blocks.items():
            if pair_index in seen:
                raise ValueError("duplicate full Coulomb atom-pair block")
            seen.add(pair_index)
            iatom, jatom = pairs[pair_index]
            i0, j0 = offsets[iatom], offsets[jatom]
            ni, nj = block.shape
            matrix[i0 : i0 + ni, j0 : j0 + nj] = block
            if iatom != jatom:
                matrix[j0 : j0 + nj, i0 : i0 + ni] = block.conjugate().T
    if seen != set(range(len(pairs))):
        raise ValueError("incomplete full Coulomb atom-pair blocks")
    return matrix, atom_naux


def read_sos_response(directory, ifreq, atom_naux, iq_zero_based=0):
    pattern = pathlib.Path(directory) / (
        f"chi0fq_ifreq_{ifreq}_iq_{iq_zero_based}_I_*_J_*_id_*.mtx"
    )
    paths = sorted(glob.glob(str(pattern)))
    if not paths:
        raise ValueError(f"no LibRPA SOS matrices match {pattern}")
    offsets = _offsets(atom_naux)
    matrix = np.zeros((offsets[-1], offsets[-1]), dtype=np.complex128)
    pairs_seen = set()
    for raw_path in paths:
        path = pathlib.Path(raw_path)
        match = _SOS_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"cannot parse LibRPA SOS filename {path.name}")
        _, _, iatom_text, jatom_text, _ = match.groups()
        iatom, jatom = int(iatom_text), int(jatom_text)
        if iatom >= len(atom_naux) or jatom >= len(atom_naux):
            raise ValueError("LibRPA SOS atom index exceeds auxiliary metadata")
        block = read_matrix_market(path)
        expected = (atom_naux[iatom], atom_naux[jatom])
        if block.shape != expected:
            raise ValueError(f"{path}: SOS block shape does not match metadata")
        i0, j0 = offsets[iatom], offsets[jatom]
        matrix[i0 : i0 + expected[0], j0 : j0 + expected[1]] += block
        pairs_seen.add((iatom, jatom))
    expected_pairs = set(_atom_pairs(len(atom_naux)))
    if pairs_seen != expected_pairs:
        raise ValueError("incomplete LibRPA SOS atom-pair blocks")
    for iatom, jatom in expected_pairs:
        if iatom == jatom:
            continue
        i0, j0 = offsets[iatom], offsets[jatom]
        ni, nj = atom_naux[iatom], atom_naux[jatom]
        matrix[j0 : j0 + nj, i0 : i0 + ni] = matrix[
            i0 : i0 + ni, j0 : j0 + nj
        ].conjugate().T
    return matrix


def _sidecar_atom_naux(channels):
    natom = max(channel.atom_index for channel in channels) + 1
    result = [0] * natom
    for channel in channels:
        result[channel.atom_index] += 1
    return result


def compare_files(
    sidecar,
    coulomb_directory,
    sos_directory,
    librpa_stdout,
    threshold=0.0,
):
    data = read_sternheimer_fixed_ao(sidecar)
    fixed_ao = evaluate_fixed_ao_sidecar(data)
    frequency, weight = read_frequency_grid(librpa_stdout)
    if frequency.shape != data.frequency_ha.shape or not np.allclose(
        frequency, data.frequency_ha.numpy(), rtol=0.0, atol=1.0e-14
    ):
        raise ValueError("LibRPA and fixed-AO frequency nodes differ")
    if not np.allclose(
        weight, data.frequency_weight_ha.numpy(), rtol=0.0, atol=1.0e-14
    ):
        raise ValueError("LibRPA and fixed-AO frequency weights differ")

    coulomb, atom_naux = read_coulomb(coulomb_directory)
    if atom_naux != _sidecar_atom_naux(data.channels):
        raise ValueError("LibRPA and fixed-AO auxiliary dimensions differ")
    rows = []
    galerkin_response = fixed_ao.galerkin_response.numpy()
    for ifreq, (omega, omega_weight) in enumerate(zip(frequency, weight)):
        chi_sos = read_sos_response(sos_directory, ifreq, atom_naux)
        row = compare_response(
            coulomb,
            galerkin_response[ifreq],
            chi_sos,
            threshold=threshold,
        )
        row.update(
            {
                "ifreq": ifreq,
                "frequency_ha": float(omega),
                "weight_ha": float(omega_weight),
            }
        )
        rows.append(row)
    galerkin_ecrpa = sum(
        row["weight_ha"] * row["integrand_galerkin"] for row in rows
    ) / (2.0 * math.pi)
    sos_ecrpa_integrated = sum(
        row["weight_ha"] * row["integrand_sos"] for row in rows
    ) / (2.0 * math.pi)
    sos_ecrpa_librpa = read_ecrpa(librpa_stdout)
    return {
        "status": "success",
        "n_basis": int(data.overlap.shape[0]),
        "n_auxiliary": int(coulomb.shape[0]),
        "n_frequency": len(rows),
        "coulomb_threshold": float(threshold),
        "galerkin_ecrpa_ha": float(galerkin_ecrpa),
        "sos_ecrpa_integrated_ha": float(sos_ecrpa_integrated),
        "sos_ecrpa_librpa_ha": float(sos_ecrpa_librpa),
        "sos_integration_minus_librpa_ha": float(
            sos_ecrpa_integrated - sos_ecrpa_librpa
        ),
        "maximum_pi_relative_frobenius": max(
            row["pi_relative_frobenius"] for row in rows
        ),
        "maximum_pi_absolute_error": max(
            row["pi_maximum_absolute"] for row in rows
        ),
        "frequencies": rows,
    }


def _write_tsv(path, rows):
    columns = (
        "ifreq",
        "frequency_ha",
        "weight_ha",
        "m_relative_frobenius",
        "m_maximum_absolute",
        "pi_relative_frobenius",
        "pi_maximum_absolute",
        "integrand_galerkin",
        "integrand_sos",
        "integrand_absolute_error",
    )
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append(
            "\t".join(
                str(row[column])
                if isinstance(row[column], int)
                else f"{row[column]:.16e}"
                for column in columns
            )
        )
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True, type=pathlib.Path)
    parser.add_argument("--coulomb-directory", required=True, type=pathlib.Path)
    parser.add_argument("--sos-directory", required=True, type=pathlib.Path)
    parser.add_argument("--librpa-stdout", required=True, type=pathlib.Path)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-tsv", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = compare_files(
        args.sidecar,
        args.coulomb_directory,
        args.sos_directory,
        args.librpa_stdout,
        threshold=args.threshold,
    )
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    _write_tsv(args.output_tsv, result["frequencies"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
