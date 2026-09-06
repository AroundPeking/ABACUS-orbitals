"""Exact active-channel views for coefficient response, not mother diagnostics."""

from dataclasses import replace
import hashlib
import json

import torch

from periodic_galerkin_basis import _coefficient_profile, build_primitive_to_candidate
from periodic_galerkin_data import (
    PeriodicGalerkinActivePrimitiveReduction,
    PeriodicGalerkinDataset,
)
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


def validate_active_primitive_profile(dataset, coefficients):
    reduction = dataset.active_primitive_reduction
    if reduction is not None and reduction.coefficient_profile != _coefficient_profile(coefficients):
        raise ValueError("active primitive coefficient profile changed; reload unreduced data")


def reduce_periodic_active_primitives(dataset, coefficients):
    """Drop only identically unused angular blocks with an explicit index map.

    The input must already have passed the complete reader/hash contract. This
    opt-in in-memory transformation does not replace that validation or modify
    the input. Callers must release the original dataset to realize savings.
    Occupied normalization is computed BEFORE slicing, in the full mother
    space. The full reference Pi and Coulomb whitening are never reduced.
    """
    if not isinstance(dataset, PeriodicGalerkinDataset):
        raise ValueError("dataset must be a PeriodicGalerkinDataset")
    validate_active_primitive_profile(dataset, coefficients)
    if dataset.active_primitive_reduction is not None:
        return dataset
    if any(record.reference_projection is not None and record.reference_projection.numel() > 0
           for record in dataset.kpoints):
        raise ValueError("reference_projection must be omitted; full projection diagnostics need unreduced data")
    # Validate the complete coefficient/block contract before removing channels.
    build_primitive_to_candidate(dataset.primitive_blocks, dataset.primitive_count, coefficients)
    prepared = prepare_periodic_occupied_reference(dataset)
    indices, blocks = [], []
    for block in dataset.primitive_blocks:
        if coefficients[block.element][block.l].shape[1] == 0:
            continue
        blocks.append(replace(block, offset=len(indices)))
        indices.extend(range(block.offset, block.offset + block.n_primitive))
    source_indices = tuple(indices)
    index = torch.tensor(indices, dtype=torch.int64)
    profile = _coefficient_profile(coefficients)
    signature = {
        "format_version": 1,
        "original_primitive_count": dataset.primitive_count,
        "original_primitive_blocks_sha256": dataset.primitive_blocks_sha256,
        "source_indices": source_indices,
        "coefficient_profile": profile,
    }
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    metadata = PeriodicGalerkinActivePrimitiveReduction(
        dataset.primitive_count, dataset.primitive_blocks_sha256,
        source_indices, profile, fingerprint,
    )
    records = []
    n = dataset.primitive_count
    for record in prepared.kpoints:
        if (record.overlap.shape != (n, n) or record.hamiltonian_ha.shape != (n, n)
                or record.source.shape[-1] != n or record.occupied_projection.shape[-1] != n):
            raise ValueError("original primitive operator dimensions are inconsistent")
        records.append(replace(
            record,
            overlap=record.overlap.index_select(0, index).index_select(1, index),
            hamiltonian_ha=record.hamiltonian_ha.index_select(0, index).index_select(1, index),
            source=record.source.index_select(-1, index),
            occupied_projection=record.occupied_projection.index_select(-1, index),
            block_contraction_cache=None,
        ))
    return replace(
        prepared, primitive_count=len(indices), primitive_blocks=tuple(blocks),
        kpoints=tuple(records), active_primitive_reduction=metadata,
    )
