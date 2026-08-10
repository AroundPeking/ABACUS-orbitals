#!/usr/bin/env python3
"""Validate a response-only virtual eigensystem for the H atom."""

import argparse
import hashlib
import json
import pathlib
import sys
import types

import torch


SIAB_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPT_DIR = SIAB_ROOT / "opt_orb_pytorch_dpsi"
EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OPT_DIR))
sys.path.insert(0, str(EXAMPLE_DIR))

from IO.func_C import read_C_init
from IO.read_sternheimer_fixed_ao import read_sternheimer_fixed_ao
from IO.read_sternheimer_primitive_galerkin import (
    read_sternheimer_primitive_galerkin,
)
from delta_st_parent_space import (
    load_delta_st_reference,
    load_full_coulomb_matrix,
    rpa_correlation_energy,
    symmetric_response,
)
from delta_st_response_compression import _fixed_ao_eigensystem
from frozen_occupied_delta_st import evaluate_frozen_occupied_delta_st
from response_only_virtual import (
    assemble_response_only_union,
    evaluate_response_only_sos,
    solve_response_only_virtual_eigensystem,
)
from run_h_gradient_gate import HARTREE_TO_KCAL_MOL, _align_candidate, _primitive_layout
from sternheimer_spillage import assemble_orbital_coefficients


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=pathlib.Path)
    parser.add_argument("primitive_file", type=pathlib.Path)
    parser.add_argument("fixed_ao_file", type=pathlib.Path)
    parser.add_argument("optimized_coefficients", type=pathlib.Path)
    parser.add_argument("output_json", type=pathlib.Path)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--sidecar-commit", required=True)
    parser.add_argument("--nu", type=int, nargs="+", default=(3, 3, 2))
    parser.add_argument(
        "--excluded-response-columns", type=int, nargs="*", default=(0,)
    )
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--condition-limit", type=float, default=1.0e12)
    parser.add_argument(
        "--spectral-direct-relative-tolerance", type=float, default=1.0e-3
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_dir = args.reference_dir.resolve()
    primitive_path = args.primitive_file.resolve()
    fixed_path = args.fixed_ao_file.resolve()
    coefficient_path = args.optimized_coefficients.resolve()
    reference = load_delta_st_reference(reference_dir)
    if reference.provenance["abacus_commit"] != args.reference_commit:
        raise ValueError("reference ABACUS commit differs from --reference-commit")
    primitive = _align_candidate(
        read_sternheimer_primitive_galerkin(primitive_path),
        args.sidecar_commit,
        args.reference_commit,
    )
    fixed_ao = _align_candidate(
        read_sternheimer_fixed_ao(fixed_path),
        args.sidecar_commit,
        args.reference_commit,
    )
    coulomb = load_full_coulomb_matrix(
        reference_dir,
        reference.provenance,
        iq=1,
    )

    element, nprimitive_by_l, lmax = _primitive_layout(primitive)
    if element != "H" or len(args.nu) != lmax + 1:
        raise ValueError("the H gate requires one Nu value for every primitive l")
    info_element = {
        "H": types.SimpleNamespace(
            Nl=lmax + 1,
            Ne=nprimitive_by_l,
            Nu=list(args.nu),
        )
    }
    radial_coefficients, initialization = read_C_init(
        coefficient_path,
        info_element,
        return_metadata=True,
    )
    if len(initialization.loaded_indices) != sum(args.nu):
        raise ValueError("optimized coefficient file is incomplete")
    response_coefficients, labels = assemble_orbital_coefficients(
        primitive,
        radial_coefficients,
    )
    excluded = _validate_excluded_columns(
        args.excluded_response_columns,
        response_coefficients.shape[1],
    )
    active_columns = tuple(
        index
        for index in range(response_coefficients.shape[1])
        if index not in excluded
    )
    response_active = response_coefficients[:, active_columns]

    response_only = evaluate_frozen_occupied_delta_st(
        primitive,
        fixed_ao,
        response_coefficients,
        include_fixed_ao_virtual=False,
        relative_rank_tolerance=args.relative_rank_tolerance,
        condition_limit=args.condition_limit,
        active_spin_excluded_columns=excluded,
    )
    union_direct = evaluate_frozen_occupied_delta_st(
        primitive,
        fixed_ao,
        response_coefficients,
        include_fixed_ao_virtual=True,
        relative_rank_tolerance=args.relative_rank_tolerance,
        condition_limit=args.condition_limit,
        active_spin_excluded_columns=excluded,
    )
    union_spectral_response, spectral_diagnostics = _evaluate_union_spectral(
        primitive,
        fixed_ao,
        response_active,
        args.relative_rank_tolerance,
        args.condition_limit,
    )

    spectral_direct_difference = union_spectral_response - union_direct.response
    spectral_direct_relative = float(
        torch.linalg.vector_norm(spectral_direct_difference)
        / torch.linalg.vector_norm(union_direct.response)
    )
    spectral_direct_maximum = float(
        torch.max(torch.abs(spectral_direct_difference))
    )
    reference_pi, coulomb_metadata = symmetric_response(
        coulomb.matrix,
        reference.response_m,
    )
    reference_ec = rpa_correlation_energy(
        reference_pi,
        reference.frequency_weight_ha,
    )
    lanes = {
        "response_ao_only_direct": _analyze_response(
            response_only.response,
            reference_pi,
            reference_ec,
            coulomb.matrix,
            reference.frequency_weight_ha,
        ),
        "response_ao_plus_fixed_virtual_direct": _analyze_response(
            union_direct.response,
            reference_pi,
            reference_ec,
            coulomb.matrix,
            reference.frequency_weight_ha,
        ),
        "response_ao_plus_fixed_virtual_spectral": _analyze_response(
            union_spectral_response,
            reference_pi,
            reference_ec,
            coulomb.matrix,
            reference.frequency_weight_ha,
        ),
    }

    overlap_difference = fixed_ao.overlap - primitive.fixed_ao_grid_overlap
    hamiltonian_difference = (
        fixed_ao.hamiltonian_ha - primitive.fixed_ao_grid_hamiltonian_ha
    )
    payload = {
        "method": "h_response_only_virtual_gate_v1",
        "protocol": dict(reference.provenance),
        "reference_abacus_commit": args.reference_commit,
        "sidecar_abacus_commit": args.sidecar_commit,
        "relative_rank_tolerance": args.relative_rank_tolerance,
        "condition_limit": args.condition_limit,
        "dimensions": {
            "primitive": int(primitive.overlap.shape[0]),
            "fixed_ao": int(fixed_ao.overlap.shape[0]),
            "response_ao_total": int(response_coefficients.shape[1]),
            "response_ao_active": int(response_active.shape[1]),
            "union": int(fixed_ao.overlap.shape[0] + response_active.shape[1]),
        },
        "radial_orbitals_by_l": list(args.nu),
        "excluded_response_columns": list(excluded),
        "excluded_response_labels": [str(labels[index]) for index in excluded],
        "same_metric_diagnostics": {
            "fixed_grid_overlap_max_abs_difference": float(
                torch.max(torch.abs(overlap_difference))
            ),
            "fixed_grid_overlap_relative_frobenius": float(
                torch.linalg.vector_norm(overlap_difference)
                / torch.linalg.vector_norm(fixed_ao.overlap)
            ),
            "fixed_grid_hamiltonian_max_abs_difference_by_spin": [
                float(torch.max(torch.abs(value)))
                for value in hamiltonian_difference
            ],
            "fixed_grid_hamiltonian_relative_frobenius_by_spin": [
                float(
                    torch.linalg.vector_norm(value)
                    / torch.linalg.vector_norm(fixed_ao.hamiltonian_ha[spin])
                )
                for spin, value in enumerate(hamiltonian_difference)
            ],
        },
        "direct_solver_diagnostics": {
            "response_ao_only": _direct_diagnostics(response_only),
            "response_ao_plus_fixed_virtual": _direct_diagnostics(union_direct),
        },
        "spectral_solver_diagnostics": spectral_diagnostics,
        "spectral_direct_response_relative_frobenius": spectral_direct_relative,
        "spectral_direct_response_max_abs_difference": spectral_direct_maximum,
        "spectral_direct_relative_tolerance": (
            args.spectral_direct_relative_tolerance
        ),
        "spectral_direct_gate_passed": (
            spectral_direct_relative <= args.spectral_direct_relative_tolerance
        ),
        "reference_ec_ha": reference_ec,
        "lanes": lanes,
        "coulomb_transform": {
            "retained_rank": coulomb_metadata.retained_rank,
            "dropped_rank": coulomb_metadata.dropped_rank,
            "minimum_eigenvalue": coulomb_metadata.minimum_eigenvalue,
            "maximum_eigenvalue": coulomb_metadata.maximum_eigenvalue,
            "eigenvalue_threshold": coulomb_metadata.eigenvalue_threshold,
        },
        "input_sha256": {
            str(path): _sha256(path)
            for path in (primitive_path, fixed_path, coefficient_path)
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["spectral_direct_gate_passed"]:
        raise RuntimeError("spectral and direct projected responses differ")


def _evaluate_union_spectral(
    primitive,
    fixed_ao,
    response_coefficients,
    relative_rank_tolerance,
    condition_limit,
):
    response_overlap = _hermitize(
        response_coefficients.mH @ primitive.overlap @ response_coefficients
    )
    response_fixed_overlap = (
        response_coefficients.mH @ primitive.primitive_ao_overlap
    )
    response_hamiltonian = torch.stack(
        tuple(
            _hermitize(response_coefficients.mH @ value @ response_coefficients)
            for value in primitive.hamiltonian_ha
        )
    )
    response_fixed_hamiltonian = torch.stack(
        tuple(
            response_coefficients.mH @ value
            for value in primitive.primitive_ao_hamiltonian_ha
        )
    )
    response_perturbation = torch.stack(
        tuple(
            _hermitize(response_coefficients.mH @ value @ response_coefficients)
            for value in primitive.perturbation_ha
        )
    )
    response_fixed_perturbation = torch.stack(
        tuple(
            response_coefficients.mH @ value
            for value in primitive.primitive_ao_perturbation_ha
        )
    )

    total = torch.zeros(
        (
            primitive.frequency_ha.shape[0],
            len(primitive.channels),
            len(primitive.channels),
        ),
        dtype=torch.complex128,
    )
    diagnostics = []
    for spin in range(fixed_ao.hamiltonian_ha.shape[0]):
        occupied = fixed_ao.occupation[spin] > 0.0
        if not bool(torch.any(occupied)):
            continue
        if int(torch.count_nonzero(occupied)) != 1:
            raise ValueError("the H gate requires one occupied state per active spin")
        energy, coefficient = _fixed_ao_eigensystem(
            fixed_ao.overlap,
            fixed_ao.hamiltonian_ha[spin],
        )
        eigenvalue_error = float(
            torch.max(torch.abs(energy - fixed_ao.eigenvalue_ha[spin]))
        )
        occupied_coefficient = coefficient[:, occupied]
        grid_norm = torch.real(
            occupied_coefficient.mH
            @ primitive.fixed_ao_grid_overlap
            @ occupied_coefficient
        ).reshape(())
        if grid_norm <= 0.0:
            raise RuntimeError("fixed occupied state has non-positive grid norm")
        occupied_coefficient = occupied_coefficient / torch.sqrt(grid_norm)
        embedded_occupied = torch.cat(
            (
                occupied_coefficient,
                torch.zeros(
                    (response_coefficients.shape[1], 1),
                    dtype=torch.complex128,
                ),
            ),
            dim=0,
        )
        union = assemble_response_only_union(
            primitive.fixed_ao_grid_overlap,
            primitive.fixed_ao_grid_hamiltonian_ha[spin],
            fixed_ao.perturbation_ha,
            response_overlap,
            response_hamiltonian[spin],
            response_perturbation,
            response_fixed_overlap,
            response_fixed_hamiltonian[spin],
            response_fixed_perturbation,
        )
        eigensystem = solve_response_only_virtual_eigensystem(
            union.overlap,
            union.hamiltonian_ha,
            embedded_occupied,
            fixed_ao.eigenvalue_ha[spin, occupied],
            relative_rank_tolerance=relative_rank_tolerance,
            condition_limit=condition_limit,
        )
        response = evaluate_response_only_sos(
            eigensystem,
            union.perturbation_ha,
            fixed_ao.occupation[spin, occupied],
            primitive.frequency_ha,
        )
        total += response.response
        diagnostics.append(
            {
                "spin": spin,
                "fixed_ao_eigenvalue_max_abs_error_ha": eigenvalue_error,
                "occupied_grid_norm_before_normalization": float(grid_norm),
                "retained_virtual_rank": eigensystem.retained_virtual_rank,
                "dropped_trial_rank": eigensystem.dropped_trial_rank,
                "projected_overlap_condition": (
                    eigensystem.projected_overlap_condition
                ),
                "occupied_orthonormality_max_abs_error": (
                    eigensystem.occupied_orthonormality_max_abs_error
                ),
                "occupied_virtual_max_abs_overlap": (
                    eigensystem.occupied_virtual_max_abs_overlap
                ),
                "virtual_orthonormality_max_abs_error": (
                    eigensystem.virtual_orthonormality_max_abs_error
                ),
                "minimum_virtual_energy_ha": float(
                    eigensystem.virtual_energy_ha[0]
                ),
                "maximum_virtual_energy_ha": float(
                    eigensystem.virtual_energy_ha[-1]
                ),
            }
        )
    if not diagnostics:
        raise ValueError("the H gate requires one active spin channel")
    return total, diagnostics


def _analyze_response(
    response,
    reference_pi,
    reference_ec,
    coulomb,
    frequency_weight,
):
    pi, _ = symmetric_response(coulomb, response)
    difference = pi - reference_pi
    relative = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(reference_pi)
    )
    ec = rpa_correlation_energy(pi, frequency_weight)
    return {
        "pi_relative_frobenius": relative,
        "pi_squared_relative_frobenius": relative * relative,
        "ec_ha": ec,
        "ec_error_ha": ec - reference_ec,
        "ec_error_kcal_mol": (ec - reference_ec) * HARTREE_TO_KCAL_MOL,
    }


def _direct_diagnostics(result):
    return {
        "retained_rank_by_spin": list(result.retained_parent_rank_by_spin),
        "dropped_rank_by_spin": list(result.dropped_parent_rank_by_spin),
        "projected_overlap_condition_by_spin": list(
            result.projected_overlap_condition_by_spin
        ),
        "fixed_ao_overlap_condition": result.fixed_ao_overlap_condition,
        "fixed_ao_eigenvalue_max_abs_error_ha": (
            result.fixed_ao_eigenvalue_max_abs_error_ha
        ),
    }


def _validate_excluded_columns(value, dimension):
    result = []
    for index in value:
        if index < 0 or index >= dimension:
            raise ValueError("excluded response column is outside the basis")
        if index in result:
            raise ValueError("excluded response columns must be unique")
        result.append(index)
    if len(result) >= dimension:
        raise ValueError("excluded response columns remove the full basis")
    return tuple(result)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hermitize(value):
    return 0.5 * (value + value.mH)


if __name__ == "__main__":
    main()
