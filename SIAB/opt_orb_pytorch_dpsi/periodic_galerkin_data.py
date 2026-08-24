"""Read the versioned ABACUS periodic Galerkin Sternheimer dataset."""

from dataclasses import dataclass
import hashlib
import math
import os
import struct
from typing import Dict, Tuple

import numpy as np
import torch


_MANIFEST_MAGIC = "ABACUS_STERNHEIMER_BASIS_OPT_MANIFEST_V1"
_CHUNK_MAGIC = b"ABACUS_STBOPT_V1"
_HEADER = struct.Struct("<16sIIiiiQQ")
_KNOWN_KINDS = frozenset(range(1, 9))
_GLOBAL_KINDS = frozenset((4, 5, 8))
_FREQUENCY_KINDS = frozenset((3, 8))
_RY_TO_HA = 0.5


@dataclass(frozen=True)
class PeriodicGalerkinKPoint:
    source_ik: int
    target_ik: int
    source_kpoint: Tuple[float, float, float]
    target_kpoint: Tuple[float, float, float]
    reciprocal_shift: Tuple[int, int, int]
    k_weight: float
    occupation: torch.Tensor
    source_eigenvalue_ha: torch.Tensor
    overlap: torch.Tensor
    hamiltonian_ha: torch.Tensor
    occupied_projection: torch.Tensor
    source: torch.Tensor
    reference_projection: torch.Tensor


@dataclass(frozen=True)
class PeriodicGalerkinDataset:
    abacus_commit: str
    executable_sha256: str
    orbital_sha256: str
    pseudopotential_sha256: str
    auxiliary_basis_sha256: str
    primitive_blocks_sha256: str
    physics_hash: str
    selected_iq: int
    q_count: int
    qpoint: Tuple[float, float, float]
    q_weight: float
    primitive_count: int
    raw_auxiliary_dimension: int
    whitened_auxiliary_rank: int
    frequency_ha: torch.Tensor
    frequency_weights_ha: torch.Tensor
    coulomb_metric: torch.Tensor
    coulomb_whitening: torch.Tensor
    reference_response: torch.Tensor
    kpoints: Tuple[PeriodicGalerkinKPoint, ...]


@dataclass(frozen=True)
class _Entry:
    kind: int
    iq: int
    ik: int
    ifrequency: int
    rows: int
    columns: int
    q_weight: float
    k_weight: float
    frequency: float
    relative_path: str
    sha256: str


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _read_status(directory):
    path = os.path.join(directory, "status.dat")
    _require(os.path.isfile(path), "periodic Galerkin dataset is missing status.dat")
    values = {}
    with open(path, "r", encoding="ascii") as handle:
        for line in handle:
            fields = line.split()
            if fields:
                _require(len(fields) == 2, "invalid periodic Galerkin status line")
                _require(fields[0] not in values, "duplicate periodic Galerkin status field")
                values[fields[0]] = fields[1]
    _require(values.get("status") == "success", "periodic Galerkin producer status is not success")
    _require(values.get("all_converged") == "yes", "periodic Galerkin equations did not all converge")
    return values


def _read_manifest(directory):
    path = os.path.join(directory, "manifest.dat")
    _require(os.path.isfile(path), "periodic Galerkin dataset is missing manifest.dat")
    with open(path, "r", encoding="ascii") as handle:
        lines = [line.rstrip("\n") for line in handle]
    _require(lines and lines[0] == _MANIFEST_MAGIC, "invalid periodic Galerkin manifest version")

    scalar = {}
    frequencies = {}
    kpoints = {}
    eigenvalues = {}
    entries = []
    for line in lines[1:]:
        _require(line, "blank line in periodic Galerkin manifest")
        fields = line.split()
        key = fields[0]
        if key == "frequency":
            _require(len(fields) == 4, "invalid periodic Galerkin frequency record")
            index = int(fields[1])
            _require(index not in frequencies, "duplicate periodic Galerkin frequency record")
            frequencies[index] = (float(fields[2]), float(fields[3]))
        elif key == "kpoint":
            _require(len(fields) >= 15, "invalid periodic Galerkin k-point record")
            source_ik = int(fields[1])
            target_ik = int(fields[2])
            count = int(fields[13])
            _require(count > 0 and len(fields) == 14 + count,
                     "invalid periodic Galerkin occupation record")
            _require(source_ik not in kpoints, "duplicate periodic Galerkin k-point record")
            kpoints[source_ik] = {
                "source_ik": source_ik,
                "target_ik": target_ik,
                "source_kpoint": tuple(float(value) for value in fields[3:6]),
                "target_kpoint": tuple(float(value) for value in fields[6:9]),
                "reciprocal_shift": tuple(int(value) for value in fields[9:12]),
                "k_weight": float(fields[12]),
                "occupation": tuple(float(value) for value in fields[14:]),
            }
        elif key == "eigenvalues_ry":
            _require(len(fields) >= 4, "invalid periodic Galerkin eigenvalue record")
            source_ik = int(fields[1])
            count = int(fields[2])
            _require(count > 0 and len(fields) == 3 + count,
                     "invalid periodic Galerkin eigenvalue record")
            _require(source_ik not in eigenvalues, "duplicate periodic Galerkin eigenvalue record")
            eigenvalues[source_ik] = tuple(float(value) for value in fields[3:])
        elif key == "entry":
            fields = line.split("\t")
            _require(len(fields) == 12, "invalid periodic Galerkin chunk entry")
            entries.append(_Entry(
                kind=int(fields[1]), iq=int(fields[2]), ik=int(fields[3]),
                ifrequency=int(fields[4]), rows=int(fields[5]), columns=int(fields[6]),
                q_weight=float(fields[7]), k_weight=float(fields[8]),
                frequency=float(fields[9]), relative_path=fields[10], sha256=fields[11]
            ))
        else:
            _require(len(fields) >= 2, "invalid periodic Galerkin scalar record")
            _require(key not in scalar, "duplicate periodic Galerkin scalar field")
            scalar[key] = tuple(fields[1:])
    return scalar, frequencies, kpoints, eigenvalues, entries


def _one(scalar, key):
    _require(key in scalar and len(scalar[key]) == 1,
             "missing or invalid periodic Galerkin manifest field: " + key)
    return scalar[key][0]


def _three(scalar, key, conversion=float):
    _require(key in scalar and len(scalar[key]) == 3,
             "missing or invalid periodic Galerkin manifest field: " + key)
    return tuple(conversion(value) for value in scalar[key])


def _valid_hex(value, lengths):
    return len(value) in lengths and all(character in "0123456789abcdefABCDEF" for character in value)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_chunk_path(directory, relative_path):
    _require(relative_path and not os.path.isabs(relative_path), "invalid periodic Galerkin chunk path")
    root = os.path.realpath(directory)
    path = os.path.realpath(os.path.join(root, relative_path))
    _require(os.path.commonpath((root, path)) == root and path != root,
             "periodic Galerkin chunk path escapes the dataset")
    return path


def _read_chunk(directory, entry):
    path = _safe_chunk_path(directory, entry.relative_path)
    _require(os.path.isfile(path), "periodic Galerkin dataset is missing chunk: " + entry.relative_path)
    _require(_sha256(path).lower() == entry.sha256.lower(),
             "periodic Galerkin chunk SHA256 mismatch: " + entry.relative_path)
    with open(path, "rb") as handle:
        header_bytes = handle.read(_HEADER.size)
        _require(len(header_bytes) == _HEADER.size, "truncated periodic Galerkin chunk header")
        header = _HEADER.unpack(header_bytes)
        payload = handle.read()
    magic, version, kind, iq, ik, ifrequency, rows, columns = header
    _require(magic == _CHUNK_MAGIC and version == 1, "invalid periodic Galerkin chunk version")
    expected = (entry.kind, entry.iq, entry.ik, entry.ifrequency, entry.rows, entry.columns)
    _require((kind, iq, ik, ifrequency, rows, columns) == expected,
             "periodic Galerkin chunk header differs from manifest")
    _require(len(payload) == 16 * rows * columns,
             "periodic Galerkin chunk payload has the wrong size")
    array = np.frombuffer(payload, dtype=np.dtype("<c16")).copy().reshape((rows, columns))
    _require(np.isfinite(array.real).all() and np.isfinite(array.imag).all(),
             "periodic Galerkin chunk contains non-finite values")
    return torch.from_numpy(array)


def _hermitian(matrix, label, tolerance=1.0e-9):
    _require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], label + " is not square")
    scale = max(1.0, float(torch.linalg.norm(matrix).item()))
    adjoint = matrix.transpose(-2, -1).conj()
    error = float(torch.linalg.norm(matrix - adjoint).item()) / scale
    _require(error <= tolerance, label + " is not Hermitian")


def read_periodic_galerkin_dataset(directory):
    """Read and validate one complete-q periodic Galerkin training dataset."""
    directory = os.path.realpath(os.fspath(directory))
    status = _read_status(directory)
    scalar, frequencies, kpoint_metadata, eigenvalues, entries = _read_manifest(directory)

    physics_hash = _one(scalar, "physics_hash")
    _require(status.get("physics_hash") == physics_hash,
             "periodic Galerkin status and manifest physics hashes differ")
    _require(_one(scalar, "kernel") == "full_coulomb",
             "periodic Galerkin optimization requires full_coulomb")
    provenance = {
        "abacus_commit": _one(scalar, "abacus_commit"),
        "executable_sha256": _one(scalar, "executable_sha256"),
        "orbital_sha256": _one(scalar, "orbital_sha256"),
        "pseudopotential_sha256": _one(scalar, "pseudopotential_sha256"),
        "auxiliary_basis_sha256": _one(scalar, "auxiliary_basis_sha256"),
        "primitive_blocks_sha256": _one(scalar, "primitive_blocks_sha256"),
    }
    _require(_valid_hex(provenance["abacus_commit"], (40, 64))
             and all(_valid_hex(value, (64,)) for key, value in provenance.items()
                     if key != "abacus_commit")
             and _valid_hex(physics_hash, (64,)),
             "invalid periodic Galerkin provenance hash")

    selected_iq = int(_one(scalar, "selected_iq"))
    q_count = int(_one(scalar, "q_count"))
    k_count = int(_one(scalar, "k_count"))
    nfrequency = int(_one(scalar, "frequency_count"))
    primitive_count = int(_one(scalar, "primitive_count"))
    raw_aux = int(_one(scalar, "raw_auxiliary_dimension"))
    white_aux = int(_one(scalar, "whitened_auxiliary_rank"))
    _require(selected_iq > 0 and selected_iq <= q_count and k_count > 0
             and nfrequency > 0 and primitive_count > 0 and raw_aux >= white_aux > 0,
             "invalid periodic Galerkin dimensions")
    _require(int(_one(scalar, "entry_count")) == len(entries),
             "periodic Galerkin manifest entry count mismatch")
    _require(set(frequencies) == set(range(nfrequency)),
             "periodic Galerkin frequency grid is incomplete")
    _require(set(kpoint_metadata) == set(range(1, k_count + 1))
             and set(eigenvalues) == set(kpoint_metadata),
             "periodic Galerkin k-point metadata are incomplete")
    qpoint = _three(scalar, "qpoint")
    q_weight = float(_one(scalar, "q_weight"))
    _require(all(math.isfinite(value) for value in qpoint)
             and math.isfinite(q_weight) and 0.0 < q_weight <= 1.0,
             "invalid periodic Galerkin q-point metadata")

    entry_map: Dict[Tuple[int, int, int], _Entry] = {}
    for entry in entries:
        _require(entry.kind in _KNOWN_KINDS and entry.iq == selected_iq,
                 "invalid periodic Galerkin chunk kind or q index")
        key = (entry.kind, entry.ik, entry.ifrequency)
        _require(key not in entry_map, "duplicate periodic Galerkin chunk record")
        _require((entry.kind in _GLOBAL_KINDS) == (entry.ik == 0),
                 "periodic Galerkin chunk has invalid k index")
        _require((entry.kind in _FREQUENCY_KINDS) == (entry.ifrequency >= 0),
                 "periodic Galerkin chunk has invalid frequency index")
        _require(math.isfinite(entry.q_weight) and abs(entry.q_weight - q_weight) <= 1.0e-12,
                 "periodic Galerkin chunk has inconsistent q weight")
        if entry.kind in _GLOBAL_KINDS:
            _require(abs(entry.k_weight - 1.0) <= 1.0e-12,
                     "periodic Galerkin global chunk has invalid k weight")
        else:
            _require(entry.ik in kpoint_metadata
                     and abs(entry.k_weight - kpoint_metadata[entry.ik]["k_weight"]) <= 1.0e-12,
                     "periodic Galerkin chunk has inconsistent k weight")
        if entry.kind in _FREQUENCY_KINDS:
            _require(entry.ifrequency in frequencies
                     and abs(entry.frequency - frequencies[entry.ifrequency][0]) <= 1.0e-12,
                     "periodic Galerkin chunk has inconsistent frequency")
        else:
            _require(entry.frequency == -1.0,
                     "periodic Galerkin frequency-independent chunk has invalid frequency")
        entry_map[key] = entry

    expected_keys = {(4, 0, -1), (5, 0, -1)}
    expected_keys.update((8, 0, iw) for iw in range(nfrequency))
    for ik in range(1, k_count + 1):
        expected_keys.update(((1, ik, -1), (2, ik, -1), (6, ik, -1), (7, ik, -1)))
        expected_keys.update((3, ik, iw) for iw in range(nfrequency))
    missing = expected_keys - set(entry_map)
    extra = set(entry_map) - expected_keys
    _require(not missing, "periodic Galerkin dataset is missing chunk records")
    _require(not extra, "periodic Galerkin dataset contains unexpected chunk records")

    chunks = {key: _read_chunk(directory, entry) for key, entry in entry_map.items()}
    metric = chunks[(4, 0, -1)]
    whitening = chunks[(5, 0, -1)]
    _require(metric.shape == (raw_aux, raw_aux) and whitening.shape == (raw_aux, white_aux),
             "periodic Galerkin Coulomb matrices have inconsistent dimensions")
    _hermitian(metric, "periodic Galerkin Coulomb metric")
    whitened_metric = whitening.transpose(-2, -1).conj().matmul(metric).matmul(whitening)
    identity = torch.eye(white_aux, dtype=torch.complex128)
    whitening_error = float(torch.linalg.norm(whitened_metric - identity).item()) / max(1.0, white_aux ** 0.5)
    declared_error = float(_one(scalar, "coulomb_max_orthonormality_error"))
    _require(math.isfinite(declared_error) and declared_error >= 0.0
             and whitening_error <= max(1.0e-8, 2.0 * declared_error + 1.0e-12),
             "periodic Galerkin Coulomb whitening is inconsistent with its metric")

    frequency_ha = torch.tensor([frequencies[i][0] for i in range(nfrequency)], dtype=torch.float64)
    frequency_weights_ha = torch.tensor([frequencies[i][1] for i in range(nfrequency)], dtype=torch.float64)
    _require(bool(torch.isfinite(frequency_ha).all()) and bool((frequency_ha >= 0.0).all())
             and bool(torch.isfinite(frequency_weights_ha).all())
             and bool((frequency_weights_ha > 0.0).all()),
             "invalid periodic Galerkin frequency grid")

    reference_response = torch.stack([chunks[(8, 0, iw)] for iw in range(nfrequency)])
    _require(reference_response.shape == (nfrequency, white_aux, white_aux),
             "periodic Galerkin exact response has inconsistent dimensions")
    for iw in range(nfrequency):
        _hermitian(reference_response[iw], "periodic Galerkin exact response")

    records = []
    for ik in range(1, k_count + 1):
        metadata = kpoint_metadata[ik]
        _require(1 <= metadata["target_ik"] <= k_count
                 and math.isfinite(metadata["k_weight"]) and metadata["k_weight"] > 0.0
                 and all(math.isfinite(value) for value in metadata["source_kpoint"])
                 and all(math.isfinite(value) for value in metadata["target_kpoint"])
                 and all(math.isfinite(value) and value > 0.0 for value in metadata["occupation"])
                 and all(math.isfinite(value) for value in eigenvalues[ik]),
                 "invalid periodic Galerkin k-point values")
        occupation = torch.tensor(metadata["occupation"], dtype=torch.float64)
        source_eigenvalue_ha = torch.tensor(eigenvalues[ik], dtype=torch.float64) * _RY_TO_HA
        noccupied = occupation.numel()
        _require(source_eigenvalue_ha.numel() == noccupied,
                 "periodic Galerkin occupations and eigenvalues have inconsistent dimensions")
        overlap = chunks[(1, ik, -1)]
        hamiltonian_ha = chunks[(6, ik, -1)] * _RY_TO_HA
        occupied_projection = chunks[(7, ik, -1)]
        source_flat = chunks[(2, ik, -1)]
        reference_flat = torch.stack([chunks[(3, ik, iw)] for iw in range(nfrequency)])
        _require(overlap.shape == (primitive_count, primitive_count)
                 and hamiltonian_ha.shape == overlap.shape,
                 "periodic Galerkin primitive matrices have inconsistent dimensions")
        _require(occupied_projection.shape == (noccupied, primitive_count),
                 "periodic Galerkin occupied projection has inconsistent dimensions")
        _require(source_flat.shape == (noccupied * white_aux, primitive_count)
                 and reference_flat.shape == (nfrequency, noccupied * white_aux, primitive_count),
                 "periodic Galerkin source or reference projection has inconsistent dimensions")
        _hermitian(overlap, "periodic Galerkin primitive overlap")
        _hermitian(hamiltonian_ha, "periodic Galerkin primitive Hamiltonian")
        records.append(PeriodicGalerkinKPoint(
            source_ik=ik,
            target_ik=metadata["target_ik"],
            source_kpoint=metadata["source_kpoint"],
            target_kpoint=metadata["target_kpoint"],
            reciprocal_shift=metadata["reciprocal_shift"],
            k_weight=metadata["k_weight"],
            occupation=occupation,
            source_eigenvalue_ha=source_eigenvalue_ha,
            overlap=overlap,
            hamiltonian_ha=hamiltonian_ha,
            occupied_projection=occupied_projection,
            source=source_flat.reshape(noccupied, white_aux, primitive_count),
            reference_projection=reference_flat.reshape(
                nfrequency, noccupied, white_aux, primitive_count),
        ))

    weight_sum = sum(record.k_weight for record in records)
    _require(abs(weight_sum - 2.0) <= 1.0e-10,
             "periodic Galerkin ABACUS k-point weights must sum to 2")

    return PeriodicGalerkinDataset(
        abacus_commit=provenance["abacus_commit"],
        executable_sha256=provenance["executable_sha256"],
        orbital_sha256=provenance["orbital_sha256"],
        pseudopotential_sha256=provenance["pseudopotential_sha256"],
        auxiliary_basis_sha256=provenance["auxiliary_basis_sha256"],
        primitive_blocks_sha256=provenance["primitive_blocks_sha256"],
        physics_hash=physics_hash,
        selected_iq=selected_iq,
        q_count=q_count,
        qpoint=qpoint,
        q_weight=q_weight,
        primitive_count=primitive_count,
        raw_auxiliary_dimension=raw_aux,
        whitened_auxiliary_rank=white_aux,
        frequency_ha=frequency_ha,
        frequency_weights_ha=frequency_weights_ha,
        coulomb_metric=metric,
        coulomb_whitening=whitening,
        reference_response=reference_response,
        kpoints=tuple(records),
    )
