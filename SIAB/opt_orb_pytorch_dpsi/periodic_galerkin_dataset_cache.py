"""Opt-in exact active-dataset snapshots: JSON and hashed, non-pickled NPY files.

Write only datasets from a fully verified reader. This derivative is not a new
validation of producer chunks. Standalone reads require the accepted completion
SHA256 and source identifiers; optional source_directory rechecks small metadata
files only. Callers must not mutate the dataset while it is being written.
"""

from dataclasses import fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import torch

import periodic_galerkin_data as reader
from periodic_galerkin_data import (
    PeriodicGalerkinActivePrimitiveReduction,
    PeriodicGalerkinDataset,
    PeriodicGalerkinKPoint,
    PeriodicGalerkinPrimitiveBlock,
)
from periodic_galerkin_reduction import validate_active_primitive_profile


_VERSION = 1
_SCHEMA = {
    PeriodicGalerkinPrimitiveBlock: "element atom_index l m n_primitive offset".split(),
    PeriodicGalerkinKPoint: (
        "source_ik target_ik source_kpoint target_kpoint reciprocal_shift k_weight occupation "
        "source_eigenvalue_ha overlap hamiltonian_ha occupied_projection source reference_projection "
        "occupied_projection_normalization block_contraction_cache"
    ).split(),
    PeriodicGalerkinActivePrimitiveReduction: (
        "original_primitive_count original_primitive_blocks_sha256 source_indices "
        "coefficient_profile mapping_sha256"
    ).split(),
    PeriodicGalerkinDataset: (
        "abacus_commit executable_sha256 orbital_sha256 pseudopotential_sha256 auxiliary_basis_sha256 "
        "primitive_blocks_sha256 physics_hash selected_iq q_count qpoint q_weight primitive_count "
        "raw_auxiliary_dimension whitened_auxiliary_rank frequency_ha frequency_weights_ha "
        "coulomb_metric coulomb_whitening reference_response primitive_blocks kpoints active_primitive_reduction"
    ).split(),
}
_TYPES = {cls.__name__: cls for cls in _SCHEMA}
_DTYPES = {np.dtype(name).str for name in (
    "bool", "uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64",
    "complex64", "complex128",
)}


def _require(condition, message):
    if not condition:
        raise ValueError("periodic dataset cache: " + message)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _hash_stream(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _read_hashed(path, expected):
    _require(not path.is_symlink(), "symlinks are not supported: " + str(path))
    content = path.read_bytes()
    _require(_digest(content) == expected, "SHA256 mismatch: " + str(path))
    return content


def _binding(manifest_sha256, status_sha256, identifiers):
    identifiers = {} if identifiers is None else identifiers
    _require(type(identifiers) is dict and all(type(k) is str and type(v) is str
             for k, v in identifiers.items()), "identifiers must be a string dictionary")
    for name, value in dict(identifiers, manifest_sha256=manifest_sha256,
                            status_sha256=status_sha256).items():
        _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                 "invalid SHA256 identifier: " + name)
    return {"manifest_sha256": manifest_sha256, "status_sha256": status_sha256,
            "identifiers": dict(identifiers)}


def _source_snapshot(directory, binding):
    root = Path(directory)
    paths = {"manifest.dat": binding["manifest_sha256"], "status.dat": binding["status_sha256"]}
    for name, expected in paths.items():
        _read_hashed(root / name, expected)
    try:
        status = reader._read_status(str(root))
        scalar, frequency, kpoints, eigenvalues, _ = reader._read_manifest(str(root))
        one = lambda key: reader._one(scalar, key)
        _require(status.get("physics_hash") == one("physics_hash"), "source status physics mismatch")
        blocks = reader._read_primitive_blocks(str(root), one("primitive_blocks_sha256"),
                                                int(one("primitive_count")))
        paths["primitive_blocks.dat"] = one("primitive_blocks_sha256")
        snapshot = {
            "scalar": scalar,
            "frequency": [frequency[i] for i in range(int(one("frequency_count")))],
            "kpoints": [kpoints[i] for i in range(1, int(one("k_count")) + 1)],
            "eigenvalues_ry": [eigenvalues[i] for i in range(1, int(one("k_count")) + 1)],
            "blocks": [{f.name: getattr(block, f.name) for f in fields(block)} for block in blocks],
        }
    except (RuntimeError, KeyError, IndexError) as error:
        raise ValueError("periodic dataset cache: invalid source metadata: " + str(error)) from error
    for name, expected in paths.items():
        _read_hashed(root / name, expected)
    return json.loads(_json_bytes(snapshot))


def _validate_dataset(dataset, source, binding):
    _require(type(dataset) is PeriodicGalerkinDataset, "supported active dataset required")
    one = lambda key: reader._one(source["scalar"], key)
    for key in ("abacus_commit", "executable_sha256", "orbital_sha256", "pseudopotential_sha256",
                "auxiliary_basis_sha256", "primitive_blocks_sha256", "physics_hash"):
        _require(getattr(dataset, key) == one(key), "source metadata mismatch: " + key)
    for key in ("selected_iq", "q_count", "raw_auxiliary_dimension", "whitened_auxiliary_rank"):
        _require(getattr(dataset, key) == int(one(key)), "source metadata mismatch: " + key)
    _require(dataset.q_weight == float(one("q_weight")) and dataset.qpoint == tuple(
        float(v) for v in source["scalar"]["qpoint"]), "source q metadata mismatch")
    reduction = dataset.active_primitive_reduction
    _require(type(reduction) is PeriodicGalerkinActivePrimitiveReduction, "active reduction is required")
    _require(reduction.original_primitive_count == int(one("primitive_count"))
             and reduction.original_primitive_blocks_sha256 == one("primitive_blocks_sha256"),
             "source mother metadata mismatch")
    signature = {"format_version": 1, "original_primitive_count": reduction.original_primitive_count,
                 "original_primitive_blocks_sha256": reduction.original_primitive_blocks_sha256,
                 "source_indices": reduction.source_indices, "coefficient_profile": reduction.coefficient_profile}
    _require(_digest(_json_bytes(signature)) == reduction.mapping_sha256, "active mapping SHA256 mismatch")
    expected_mapping = binding["identifiers"].get("mapping_sha256", reduction.mapping_sha256)
    _require(expected_mapping == reduction.mapping_sha256, "source mapping identifier mismatch")
    profile = {(element, l): (rows, columns) for element, l, rows, columns in reduction.coefficient_profile}
    blocks, indices = [], []
    for values in source["blocks"]:
        block = PeriodicGalerkinPrimitiveBlock(**values)
        rows, columns = profile.get((block.element, block.l), (None, None))
        _require(rows == block.n_primitive and type(columns) is int and columns >= 0,
                 "source active coefficient profile mismatch")
        if columns:
            blocks.append(replace(block, offset=len(indices)))
            indices.extend(range(block.offset, block.offset + block.n_primitive))
    _require(tuple(indices) == reduction.source_indices and tuple(blocks) == dataset.primitive_blocks
             and dataset.primitive_count == len(indices), "source active mapping metadata mismatch")
    for tensor, values in ((dataset.frequency_ha, [row[0] for row in source["frequency"]]),
                           (dataset.frequency_weights_ha, [row[1] for row in source["frequency"]])):
        _require(torch.equal(tensor, torch.tensor(values, dtype=torch.float64)), "source frequency metadata mismatch")
    _require(len(dataset.kpoints) == len(source["kpoints"]), "source k count mismatch")
    for record, metadata, eigenvalues in zip(dataset.kpoints, source["kpoints"], source["eigenvalues_ry"]):
        _require(type(record) is PeriodicGalerkinKPoint, "unsupported k-point type")
        for key in ("source_ik", "target_ik", "source_kpoint", "target_kpoint", "reciprocal_shift", "k_weight"):
            expected = tuple(metadata[key]) if isinstance(metadata[key], list) else metadata[key]
            _require(getattr(record, key) == expected, "source k metadata mismatch: " + key)
        _require(torch.equal(record.occupation, torch.tensor(metadata["occupation"], dtype=torch.float64))
                 and torch.equal(record.source_eigenvalue_ha, torch.tensor(eigenvalues, dtype=torch.float64) * .5),
                 "source occupation/eigenvalue metadata mismatch")
        _require(record.block_contraction_cache is None, "block_contraction_cache is not supported")
        nocc, n = record.occupation.numel(), dataset.primitive_count
        _require(isinstance(record.occupied_projection_normalization, torch.Tensor)
                 and record.occupied_projection_normalization.shape == (nocc, nocc),
                 "full-mother occupied_projection_normalization is required")
        _require(record.overlap.shape == (n, n) and record.hamiltonian_ha.shape == (n, n)
                 and record.occupied_projection.shape == (nocc, n)
                 and record.source.shape == (nocc, dataset.whitened_auxiliary_rank, n),
                 "active operator shape mismatch")
        _require(record.reference_projection is None or record.reference_projection.numel() == 0,
                 "active dataset must omit reference projection")


def _write_bytes(path, content):
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _check_schema():
    for cls, names in _SCHEMA.items():
        _require([f.name for f in fields(cls)] == names, "unsupported dataclass schema: " + cls.__name__)


def write_periodic_galerkin_dataset_cache(directory, dataset, *, source_directory,
                                          manifest_sha256, status_sha256, identifiers=None):
    """Write a new cache and return COMPLETE.json SHA256; never overwrite/resume.

    identifiers may bind freeze_sha256, acceptance_sha256 and mapping_sha256.
    A failed write leaves an incomplete directory that readers will reject.
    Only frozen, dense CPU tensors in active reader datasets are supported.
    """
    _check_schema()
    root = Path(directory)
    if root.exists():
        raise FileExistsError(root)
    binding = _binding(manifest_sha256, status_sha256, identifiers)
    source = _source_snapshot(source_directory, binding)
    _validate_dataset(dataset, source, binding)
    root.mkdir(parents=True, exist_ok=False)
    arrays, versions = [], []

    def encode(value):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            _require(math.isfinite(value), "nonfinite scalar metadata")
            return value
        if type(value) is tuple:
            return {"type": "tuple", "items": [encode(v) for v in value]}
        if type(value) in _SCHEMA:
            return {"type": type(value).__name__,
                    "fields": {key: encode(getattr(value, key)) for key in _SCHEMA[type(value)]}}
        _require(type(value) is torch.Tensor, "unsupported Python object in dataset")
        _require(value.device.type == "cpu" and value.layout == torch.strided and not value.requires_grad,
                 "only frozen dense CPU tensors without gradients are supported")
        versions.append((value, value._version))
        array = value.detach().resolve_conj().resolve_neg().numpy()
        _require(array.dtype.str in _DTYPES, "unsupported tensor dtype")
        index = len(arrays)
        path = root / "array_{:06d}.npy".format(index)
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        with path.open("rb") as handle:
            digest = _hash_stream(handle)
        arrays.append({"file": path.name, "sha256": digest, "dtype": array.dtype.str,
                       "shape": list(array.shape)})
        return {"type": "tensor", "index": index}

    tree = encode(dataset)
    _require(all(tensor._version == version for tensor, version in versions), "frozen input mutated during write")
    _require(_source_snapshot(source_directory, binding) == source, "source changed during write")
    metadata = {"format_version": _VERSION, "binding": binding, "source": source,
                "arrays": arrays, "dataset": tree}
    content = _json_bytes(metadata)
    _write_bytes(root / "dataset.json", content)
    completion = _json_bytes({"format_version": _VERSION, "status": "complete", "metadata_sha256": _digest(content)})
    temporary = root / ".COMPLETE.tmp"
    _write_bytes(temporary, completion)
    # Hard-link publication is atomic and refuses an existing completion record.
    os.link(str(temporary), str(root / "COMPLETE.json"))
    temporary.unlink()
    return _digest(completion)


def read_periodic_galerkin_dataset_cache(directory, *, cache_sha256, manifest_sha256,
                                         status_sha256, identifiers=None, source_directory=None,
                                         active_coefficients=None):
    """Read an accepted derivative with fresh CPU tensor ownership, no chunk reads.

    cache_sha256 is the SHA256 returned by the writer, retained by the caller.
    Supply source_directory to additionally recheck the three small source files.
    Tensor storage aliases, strides and autograd graphs are not serialized.
    """
    _check_schema()
    root = Path(directory)
    binding = _binding(manifest_sha256, status_sha256, identifiers)
    completion = json.loads(_read_hashed(root / "COMPLETE.json", cache_sha256))
    _require(set(completion) == {"format_version", "status", "metadata_sha256"}
             and completion["format_version"] == _VERSION and completion["status"] == "complete",
             "invalid completion schema")
    metadata = json.loads(_read_hashed(root / "dataset.json", completion["metadata_sha256"]))
    _require(set(metadata) == {"format_version", "binding", "source", "arrays", "dataset"}
             and metadata["format_version"] == _VERSION, "unsupported metadata schema")
    _require(metadata["binding"] == binding, "source binding mismatch")
    if source_directory is not None:
        _require(_source_snapshot(source_directory, binding) == metadata["source"], "source metadata changed")
    arrays = metadata["arrays"]
    _require(type(arrays) is list, "invalid arrays schema")
    used = set()

    def decode(value):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            _require(math.isfinite(value), "nonfinite scalar metadata")
            return value
        _require(type(value) is dict and "type" in value, "unsupported node schema")
        kind = value["type"]
        if kind == "tuple":
            _require(set(value) == {"type", "items"} and type(value["items"]) is list, "invalid tuple schema")
            return tuple(decode(v) for v in value["items"])
        if kind in _TYPES:
            cls = _TYPES[kind]
            _require(set(value) == {"type", "fields"} and type(value["fields"]) is dict
                     and set(value["fields"]) == set(_SCHEMA[cls]), "unsupported dataclass fields")
            return cls(**{name: decode(value["fields"][name]) for name in _SCHEMA[cls]})
        _require(kind == "tensor" and set(value) == {"type", "index"}, "unsupported type schema")
        index = value["index"]
        _require(type(index) is int and 0 <= index < len(arrays) and index not in used, "invalid array index")
        used.add(index)
        item = arrays[index]
        _require(type(item) is dict and set(item) == {"file", "sha256", "dtype", "shape"}, "invalid array schema")
        _require(item["file"] == "array_{:06d}.npy".format(index), "unsafe array path")
        _require(item["dtype"] in _DTYPES and type(item["shape"]) is list
                 and all(type(n) is int and n >= 0 for n in item["shape"]), "unsupported array dtype/shape")
        path = root / item["file"]
        _require(not path.is_symlink(), "array symlinks are not supported")
        with path.open("rb") as handle:
            _require(_hash_stream(handle) == item["sha256"], "array SHA256 mismatch: " + item["file"])
            handle.seek(0)
            array = np.load(handle, allow_pickle=False)
            _require(type(array) is np.ndarray and array.dtype.str == item["dtype"]
                     and list(array.shape) == item["shape"] and handle.read(1) == b"", "array dtype/shape mismatch")
        return torch.from_numpy(array)

    dataset = decode(metadata["dataset"])
    _require(len(used) == len(arrays), "unreferenced cache arrays")
    _validate_dataset(dataset, metadata["source"], binding)
    if active_coefficients is not None:
        validate_active_primitive_profile(dataset, active_coefficients)
    return dataset
