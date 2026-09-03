#!/usr/bin/env python3
"""Validate a periodic basis-opt-v1 dataset with bounded memory."""

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

try:
    import numpy as np
except ImportError:  # The standard-library path remains useful for small datasets.
    np = None


HEADER = struct.Struct("<16sIIiiiQQ")
PAIR = struct.Struct("<dd")
MAGIC = b"ABACUS_STBOPT_V1"
DEFAULT_BLOCK_BYTES = 1024 * 1024


def _read_header(handle, path):
    raw_header = handle.read(HEADER.size)
    if len(raw_header) != HEADER.size:
        raise RuntimeError(f"truncated header: {path}")
    magic, version, kind, iq, ik, ifrequency, rows, columns = HEADER.unpack(raw_header)
    if magic != MAGIC or version != 1:
        raise RuntimeError(f"invalid chunk header or size: {path}")
    return raw_header, {
        "kind": kind,
        "iq": iq,
        "ik": ik,
        "ifrequency": ifrequency,
        "rows": rows,
        "columns": columns,
    }


def _payload_is_finite(payload):
    if np is not None:
        return bool(np.isfinite(np.frombuffer(payload, dtype="<f8")).all())
    return all(
        math.isfinite(real) and math.isfinite(imag)
        for real, imag in struct.iter_unpack("<dd", payload)
    )


def scan_chunk(path, expected_sha256=None, block_bytes=DEFAULT_BLOCK_BYTES):
    """Check one complete chunk without retaining its payload."""
    path = Path(path)
    if block_bytes <= 0 or block_bytes % PAIR.size:
        raise ValueError("block_bytes must be a positive multiple of 16")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        raw_header, chunk = _read_header(handle, path)
        expected_size = HEADER.size + PAIR.size * chunk["rows"] * chunk["columns"]
        if os.fstat(handle.fileno()).st_size != expected_size:
            raise RuntimeError(f"invalid chunk header or size: {path}")
        digest.update(raw_header)
        remaining = expected_size - HEADER.size
        while remaining:
            payload = handle.read(min(block_bytes, remaining))
            if not payload:
                raise RuntimeError(f"invalid chunk header or size: {path}")
            if len(payload) % PAIR.size or not _payload_is_finite(payload):
                raise RuntimeError(f"non-finite payload: {path}")
            digest.update(payload)
            remaining -= len(payload)
    chunk["sha256"] = digest.hexdigest()
    if expected_sha256 is not None and chunk["sha256"] != expected_sha256:
        raise RuntimeError(f"chunk SHA256 mismatch: {path.name}")
    return chunk


def read_chunk_doubles(path, expected):
    """Read one matrix as packed doubles, avoiding Python complex objects."""
    path = Path(path)
    with path.open("rb") as handle:
        _, chunk = _read_header(handle, path)
        for key in ("kind", "iq", "ik", "ifrequency", "rows", "columns"):
            if chunk[key] != expected[key]:
                raise RuntimeError(f"chunk header mismatch: {path.name} {key}")
        values = array("d")
        try:
            values.fromfile(handle, 2 * chunk["rows"] * chunk["columns"])
        except EOFError as error:
            raise RuntimeError(f"invalid chunk header or size: {path}") from error
        if handle.read(1):
            raise RuntimeError(f"invalid chunk header or size: {path}")
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _complex_at(values, index):
    return complex(values[2 * index], values[2 * index + 1])


def hermitian_relative_error(values, dimension):
    scale = 1.0
    error = 0.0
    for row in range(dimension):
        for column in range(row, dimension):
            left = _complex_at(values, row * dimension + column)
            right = _complex_at(values, column * dimension + row)
            scale = max(scale, abs(left), abs(right))
            error = max(error, abs(left - right.conjugate()))
    return error / scale


def hermitian_relative_error_file(path, chunk, block_rows=128):
    dimension = chunk["rows"]
    if chunk["columns"] != dimension:
        raise RuntimeError(f"Hermitian chunk is not square: {path}")
    if np is None:
        values = read_chunk_doubles(path, chunk)
        return hermitian_relative_error(values, dimension)
    matrix = np.memmap(
        path,
        dtype="<c16",
        mode="r",
        offset=HEADER.size,
        shape=(dimension, dimension),
    )
    scale = max(1.0, float(np.max(np.abs(matrix))))
    error = 0.0
    for start in range(0, dimension, block_rows):
        stop = min(start + block_rows, dimension)
        difference = matrix[start:stop, :] - matrix[:, start:stop].conjugate().T
        error = max(error, float(np.max(np.abs(difference))))
    del matrix
    return error / scale


def sampled_whitening_error(metric, transform, raw_dimension, retained_rank):
    vectors = []
    for index in sorted({0, retained_rank // 2, retained_rank - 1}):
        vector = [0j] * retained_rank
        vector[index] = 1.0 + 0j
        vectors.append(vector)
    vectors.append(
        [
            complex(math.sin(index + 1.0), math.cos(2.0 * index + 1.0))
            for index in range(retained_rank)
        ]
    )
    maximum = 0.0
    for vector in vectors:
        raw_vector = [
            sum(
                _complex_at(transform, row * retained_rank + column) * vector[column]
                for column in range(retained_rank)
            )
            for row in range(raw_dimension)
        ]
        metric_vector = [
            sum(
                _complex_at(metric, row * raw_dimension + column) * raw_vector[column]
                for column in range(raw_dimension)
            )
            for row in range(raw_dimension)
        ]
        reconstructed = [
            sum(
                _complex_at(transform, row * retained_rank + column).conjugate()
                * metric_vector[row]
                for row in range(raw_dimension)
            )
            for column in range(retained_rank)
        ]
        norm = max(1.0, max(abs(value) for value in vector))
        maximum = max(
            maximum,
            max(
                abs(reconstructed[index] - vector[index])
                for index in range(retained_rank)
            )
            / norm,
        )
    return maximum


def whitening_probe_limit(retained_rank, declared_max_element_error):
    return max(1.0e-8, retained_rank * declared_max_element_error + 1.0e-12)


def parse_manifest(path):
    metadata = {}
    frequencies = []
    kpoints = []
    eigenvalues = {}
    entries = []
    with Path(path).open(encoding="ascii") as handle:
        first = handle.readline().strip()
        if first != "ABACUS_STERNHEIMER_BASIS_OPT_MANIFEST_V1":
            raise RuntimeError("unexpected manifest version")
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "frequency":
                frequencies.append((int(fields[1]), float(fields[2]), float(fields[3])))
            elif fields[0] == "kpoint":
                occupation_count = int(fields[13])
                occupations = [float(value) for value in fields[14:]]
                if len(occupations) != occupation_count:
                    raise RuntimeError("k-point occupation count mismatch")
                kpoints.append(
                    {
                        "source_ik": int(fields[1]),
                        "target_ik": int(fields[2]),
                        "k_weight": float(fields[12]),
                        "occupations": occupations,
                    }
                )
            elif fields[0] == "eigenvalues_ry":
                source_ik = int(fields[1])
                count = int(fields[2])
                values = [float(value) for value in fields[3:]]
                if len(values) != count or source_ik in eigenvalues:
                    raise RuntimeError("k-point eigenvalue count mismatch or duplicate")
                eigenvalues[source_ik] = values
            elif fields[0] == "entry":
                if len(fields) != 12:
                    raise RuntimeError("manifest entry has the wrong field count")
                entries.append(
                    {
                        "kind": int(fields[1]),
                        "iq": int(fields[2]),
                        "ik": int(fields[3]),
                        "ifrequency": int(fields[4]),
                        "rows": int(fields[5]),
                        "columns": int(fields[6]),
                        "q_weight": float(fields[7]),
                        "k_weight": float(fields[8]),
                        "frequency": float(fields[9]),
                        "path": fields[10],
                        "sha256": fields[11],
                    }
                )
            else:
                metadata[fields[0]] = fields[1:]
    for record in kpoints:
        record["eigenvalues_ry"] = eigenvalues.get(record["source_ik"], [])
        if len(record["eigenvalues_ry"]) != len(record["occupations"]):
            raise RuntimeError("occupied eigenvalue and occupation dimensions differ")
    return metadata, frequencies, kpoints, entries


def _chunk_key(entry):
    return entry["kind"], entry["ik"], entry["ifrequency"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    metadata, frequencies, kpoints, entries = parse_manifest(dataset / "manifest.dat")
    if metadata["abacus_commit"][0] != args.commit:
        raise RuntimeError("ABACUS commit provenance mismatch")
    if int(metadata["entry_count"][0]) != len(entries):
        raise RuntimeError("manifest entry count mismatch")
    expected_entries = 2 + len(frequencies) + len(kpoints) * (4 + len(frequencies))
    if len(entries) != expected_entries:
        raise RuntimeError(
            "dataset is missing an expected global, k-resolved, or frequency-resolved chunk"
        )
    if abs(sum(record["k_weight"] for record in kpoints) - 2.0) > 1.0e-12:
        raise RuntimeError("ABACUS non-spin-polarized full-k weights do not sum to two")

    primitive_count = int(metadata["primitive_count"][0])
    if primitive_count > 512 and np is None:
        raise RuntimeError("NumPy is required to validate a large basis-opt-v1 dataset")

    chunks = {}
    for entry in entries:
        path = dataset / entry["path"]
        chunk = scan_chunk(path, expected_sha256=entry["sha256"])
        for key in ("kind", "iq", "ik", "ifrequency", "rows", "columns"):
            if chunk[key] != entry[key]:
                raise RuntimeError(f"chunk header mismatch: {entry['path']} {key}")
        key = _chunk_key(entry)
        if key in chunks:
            raise RuntimeError(f"duplicate chunk key: {entry['path']}")
        chunk["path"] = path
        chunks[key] = chunk

    raw_dimension = int(metadata["raw_auxiliary_dimension"][0])
    retained_rank = int(metadata["whitened_auxiliary_rank"][0])
    if raw_dimension <= 0 or not 0 < retained_rank <= raw_dimension:
        raise RuntimeError("auxiliary dimensions are invalid")
    metric_chunk = chunks[(4, 0, -1)]
    transform_chunk = chunks[(5, 0, -1)]
    if (
        metric_chunk["rows"] != raw_dimension
        or metric_chunk["columns"] != raw_dimension
    ):
        raise RuntimeError("full-Coulomb metric dimension mismatch")
    if (
        transform_chunk["rows"] != raw_dimension
        or transform_chunk["columns"] != retained_rank
    ):
        raise RuntimeError("full-Coulomb whitening transform dimension mismatch")
    metric = read_chunk_doubles(metric_chunk["path"], metric_chunk)
    transform = read_chunk_doubles(transform_chunk["path"], transform_chunk)
    metric_hermitian_error = hermitian_relative_error(metric, raw_dimension)
    whitening_error = sampled_whitening_error(
        metric, transform, raw_dimension, retained_rank
    )
    declared_whitening_error = float(metadata["coulomb_max_orthonormality_error"][0])
    whitening_probe_limit_value = whitening_probe_limit(
        retained_rank, declared_whitening_error
    )
    if (
        metric_hermitian_error > 1.0e-10
        or declared_whitening_error > 1.0e-8
        or whitening_error > whitening_probe_limit_value
    ):
        raise RuntimeError("full-Coulomb whitening gate failed")
    del metric, transform

    reference_response_hermitian_error = 0.0
    for ifrequency, _, _ in frequencies:
        response = chunks[(8, 0, ifrequency)]
        if response["rows"] != retained_rank or response["columns"] != retained_rank:
            raise RuntimeError("exact reference-response dimension mismatch")
        values = read_chunk_doubles(response["path"], response)
        reference_response_hermitian_error = max(
            reference_response_hermitian_error,
            hermitian_relative_error(values, retained_rank),
        )
    if reference_response_hermitian_error > 1.0e-10:
        raise RuntimeError("exact reference response is non-Hermitian")

    overlap_hermitian_error = 0.0
    hamiltonian_hermitian_error = 0.0
    kpoint_by_index = {record["source_ik"]: record for record in kpoints}
    for record in kpoints:
        ik = record["source_ik"]
        overlap = chunks[(1, ik, -1)]
        source = chunks[(2, ik, -1)]
        hamiltonian = chunks[(6, ik, -1)]
        occupied_projection = chunks[(7, ik, -1)]
        target = kpoint_by_index[record["target_ik"]]
        if overlap["rows"] != primitive_count or overlap["columns"] != primitive_count:
            raise RuntimeError("overlap dimension mismatch")
        if (
            source["columns"] != primitive_count
            or source["rows"] != len(record["occupations"]) * retained_rank
        ):
            raise RuntimeError("source dimension mismatch")
        if (
            hamiltonian["rows"] != primitive_count
            or hamiltonian["columns"] != primitive_count
        ):
            raise RuntimeError("Hamiltonian dimension mismatch")
        if (
            occupied_projection["rows"] != len(target["occupations"])
            or occupied_projection["columns"] != primitive_count
        ):
            raise RuntimeError("occupied-projection dimension mismatch")
        overlap_hermitian_error = max(
            overlap_hermitian_error,
            hermitian_relative_error_file(overlap["path"], overlap),
        )
        hamiltonian_hermitian_error = max(
            hamiltonian_hermitian_error,
            hermitian_relative_error_file(hamiltonian["path"], hamiltonian),
        )
        for ifrequency, _, _ in frequencies:
            response = chunks[(3, ik, ifrequency)]
            if (
                response["rows"] != source["rows"]
                or response["columns"] != primitive_count
            ):
                raise RuntimeError("response dimension mismatch")
    if overlap_hermitian_error > 1.0e-10:
        raise RuntimeError("Bloch primitive overlap is non-Hermitian")
    if hamiltonian_hermitian_error > 1.0e-8:
        raise RuntimeError("Bloch primitive Hamiltonian is non-Hermitian")

    with (dataset / "status.dat").open(encoding="ascii") as handle:
        status = dict(line.strip().split(maxsplit=1) for line in handle if line.strip())
    if status.get("status") != "success" or status.get("all_converged") != "yes":
        raise RuntimeError("dataset status is not converged success")
    if status.get("physics_hash") != metadata["physics_hash"][0]:
        raise RuntimeError("dataset status and manifest physics hashes differ")

    payload = {
        "status": "success",
        "entries": len(entries),
        "kpoints": len(kpoints),
        "frequencies": len(frequencies),
        "raw_auxiliary_dimension": raw_dimension,
        "whitened_auxiliary_rank": retained_rank,
        "primitive_count": primitive_count,
        "metric_hermitian_relative_error": metric_hermitian_error,
        "declared_whitening_max_error": declared_whitening_error,
        "sampled_whitening_max_error": whitening_error,
        "sampled_whitening_limit": whitening_probe_limit_value,
        "reference_response_hermitian_relative_error": reference_response_hermitian_error,
        "overlap_hermitian_relative_error": overlap_hermitian_error,
        "hamiltonian_hermitian_relative_error": hamiltonian_hermitian_error,
    }
    print(json.dumps(payload, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
