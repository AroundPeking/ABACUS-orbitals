#!/usr/bin/env python3
"""Resolve a reader-v1 Coulomb discrepancy by auxiliary angular momentum."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np


COULOMB_MARKER = -20129433


def expand_angular_momentum_labels(radial_angular_momenta: list[int]) -> list[int]:
    labels: list[int] = []
    for angular_momentum in radial_angular_momenta:
        if angular_momentum < 0:
            raise ValueError("auxiliary angular momentum must be non-negative")
        labels.extend([angular_momentum] * (2 * angular_momentum + 1))
    return labels


def read_auxiliary_angular_momenta(path: Path) -> list[int]:
    lines = [line.split() for line in path.read_text(encoding="ascii").splitlines() if line.split()]
    if len(lines) < 3 or len(lines[0]) < 2:
        raise ValueError("invalid basis_aux_out header")
    number_of_types = int(lines[0][0])
    if number_of_types != 1:
        raise ValueError("the angular-momentum diagnostic currently requires one atom type")
    number_of_radial_functions = int(lines[2][1])
    if len(lines) < 3 + number_of_radial_functions:
        raise ValueError("truncated basis_aux_out angular-momentum list")
    return [int(lines[index][0]) for index in range(3, 3 + number_of_radial_functions)]


def _atom_pair(pair_index: int, number_of_atoms: int) -> tuple[int, int]:
    local_index = pair_index
    for first_atom in range(number_of_atoms):
        number_of_pairs = number_of_atoms - first_atom
        if local_index < number_of_pairs:
            return first_atom, first_atom + local_index
        local_index -= number_of_pairs
    raise ValueError("invalid reader-v1 atom-pair index")


def read_coulomb_v1(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise ValueError("truncated reader-v1 Coulomb header")
        marker, iq, naux, value_flag, number_of_atoms, number_of_blocks = struct.unpack("=6i", header)
        if marker != COULOMB_MARKER or value_flag != 1:
            raise ValueError("invalid reader-v1 Coulomb header")
        atom_naux = struct.unpack(f"={number_of_atoms}i", handle.read(4 * number_of_atoms))
        block_table = [struct.unpack("=iq", handle.read(12)) for _ in range(number_of_blocks)]
        offsets = np.cumsum((0, *atom_naux))
        matrix = np.zeros((naux, naux), dtype=np.complex128)
        for pair_index, file_offset in block_table:
            first_atom, second_atom = _atom_pair(pair_index, number_of_atoms)
            shape = (atom_naux[first_atom], atom_naux[second_atom])
            handle.seek(file_offset)
            raw = handle.read(16 * shape[0] * shape[1])
            if len(raw) != 16 * shape[0] * shape[1]:
                raise ValueError("truncated reader-v1 Coulomb block")
            block = np.frombuffer(raw, dtype=np.complex128).reshape(shape)
            first_start, first_stop = offsets[first_atom], offsets[first_atom + 1]
            second_start, second_stop = offsets[second_atom], offsets[second_atom + 1]
            matrix[first_start:first_stop, second_start:second_stop] = block
            if first_atom != second_atom:
                matrix[second_start:second_stop, first_start:first_stop] = block.conj().T
    return {
        "iq": iq,
        "atom_naux": tuple(int(value) for value in atom_naux),
        "matrix": matrix,
    }


def _eigenvector_l_weights(vector: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        str(angular_momentum): float(np.sum(np.abs(vector[labels == angular_momentum]) ** 2))
        for angular_momentum in sorted(set(int(value) for value in labels))
    }


def summarize_coulomb_pair(
    native_data: dict[str, Any],
    grid_data: dict[str, Any],
    local_angular_momentum_labels: list[int],
) -> dict[str, Any]:
    if native_data["iq"] != grid_data["iq"] or native_data["atom_naux"] != grid_data["atom_naux"]:
        raise ValueError("native and grid Coulomb metadata differ")
    atom_naux = native_data["atom_naux"]
    if any(dimension != len(local_angular_momentum_labels) for dimension in atom_naux):
        raise ValueError("basis_aux_out labels do not match reader-v1 atom dimensions")

    native_raw = native_data["matrix"]
    grid_raw = grid_data["matrix"]
    native_hermiticity = np.linalg.norm(native_raw - native_raw.conj().T) / max(np.linalg.norm(native_raw), 1.0)
    grid_hermiticity = np.linalg.norm(grid_raw - grid_raw.conj().T) / max(np.linalg.norm(grid_raw), 1.0)
    native = 0.5 * (native_raw + native_raw.conj().T)
    grid = 0.5 * (grid_raw + grid_raw.conj().T)
    error = native - grid
    labels = np.asarray(local_angular_momentum_labels * len(atom_naux), dtype=int)

    blocks: list[dict[str, Any]] = []
    angular_momenta = sorted(set(int(value) for value in labels))
    for row_l in angular_momenta:
        row_mask = labels == row_l
        for column_l in angular_momenta:
            if column_l < row_l:
                continue
            column_mask = labels == column_l
            native_block = native[np.ix_(row_mask, column_mask)]
            grid_block = grid[np.ix_(row_mask, column_mask)]
            error_block = error[np.ix_(row_mask, column_mask)]
            grid_norm = np.linalg.norm(grid_block)
            blocks.append(
                {
                    "l_row": row_l,
                    "l_column": column_l,
                    "native_frobenius": float(np.linalg.norm(native_block)),
                    "grid_frobenius": float(grid_norm),
                    "error_frobenius": float(np.linalg.norm(error_block)),
                    "error_relative_to_grid": float(np.linalg.norm(error_block) / max(grid_norm, 1.0e-300)),
                    "native_max_abs": float(np.max(np.abs(native_block))),
                    "grid_max_abs": float(np.max(np.abs(grid_block))),
                    "error_max_abs": float(np.max(np.abs(error_block))),
                }
            )

    eigenvalues, eigenvectors = np.linalg.eigh(native)
    principal_subspaces = []
    for maximum_l in angular_momenta:
        mask = labels <= maximum_l
        native_subspace = native[np.ix_(mask, mask)]
        grid_subspace = grid[np.ix_(mask, mask)]
        error_subspace = native_subspace - grid_subspace
        if maximum_l == angular_momenta[-1]:
            native_subspace_eigenvalues = eigenvalues
        else:
            native_subspace_eigenvalues = np.linalg.eigvalsh(native_subspace)
        grid_subspace_eigenvalues = np.linalg.eigvalsh(grid_subspace)
        principal_subspaces.append(
            {
                "maximum_l": maximum_l,
                "dimension": int(np.count_nonzero(mask)),
                "native_frobenius": float(np.linalg.norm(native_subspace)),
                "grid_frobenius": float(np.linalg.norm(grid_subspace)),
                "error_frobenius": float(np.linalg.norm(error_subspace)),
                "error_relative_to_grid": float(
                    np.linalg.norm(error_subspace) / max(np.linalg.norm(grid_subspace), 1.0e-300)
                ),
                "native_minimum_eigenvalue": float(native_subspace_eigenvalues[0]),
                "native_maximum_eigenvalue": float(native_subspace_eigenvalues[-1]),
                "grid_minimum_eigenvalue": float(grid_subspace_eigenvalues[0]),
                "grid_maximum_eigenvalue": float(grid_subspace_eigenvalues[-1]),
                "negative_eigenvalue_count_below_minus_1e_8": int(
                    np.count_nonzero(native_subspace_eigenvalues < -1.0e-8)
                ),
            }
        )
    result = {
        "status": "success",
        "iq": native_data["iq"],
        "atom_naux": list(atom_naux),
        "native_hermiticity_relative": float(native_hermiticity),
        "grid_hermiticity_relative": float(grid_hermiticity),
        "native_frobenius": float(np.linalg.norm(native)),
        "grid_frobenius": float(np.linalg.norm(grid)),
        "error_frobenius": float(np.linalg.norm(error)),
        "error_relative_to_grid": float(np.linalg.norm(error) / max(np.linalg.norm(grid), 1.0)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "negative_eigenvalue_count_below_minus_1e_8": int(np.count_nonzero(eigenvalues < -1.0e-8)),
        "minimum_eigenvector_l_weights": _eigenvector_l_weights(eigenvectors[:, 0], labels),
        "maximum_eigenvector_l_weights": _eigenvector_l_weights(eigenvectors[:, -1], labels),
        "angular_momentum_blocks": blocks,
        "principal_subspace_by_maximum_l": principal_subspaces,
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ValueError("non-finite Coulomb angular-momentum diagnostic")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-aux", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--comparison-native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    radial_l = read_auxiliary_angular_momenta(args.basis_aux)
    local_labels = expand_angular_momentum_labels(radial_l)
    native_data = read_coulomb_v1(args.native)
    result = summarize_coulomb_pair(native_data, read_coulomb_v1(args.grid), local_labels)
    result["radial_function_count_by_l"] = {
        str(angular_momentum): radial_l.count(angular_momentum)
        for angular_momentum in sorted(set(radial_l))
    }
    if args.comparison_native is not None:
        comparison = read_coulomb_v1(args.comparison_native)
        if comparison["iq"] != native_data["iq"] or comparison["atom_naux"] != native_data["atom_naux"]:
            raise ValueError("comparison native Coulomb metadata differ")
        native_matrix = 0.5 * (native_data["matrix"] + native_data["matrix"].conj().T)
        comparison_matrix = 0.5 * (comparison["matrix"] + comparison["matrix"].conj().T)
        result["native_minus_comparison_relative"] = float(
            np.linalg.norm(native_matrix - comparison_matrix) / max(np.linalg.norm(comparison_matrix), 1.0)
        )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
