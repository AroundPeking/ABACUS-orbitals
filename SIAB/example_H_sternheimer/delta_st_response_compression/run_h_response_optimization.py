#!/usr/bin/env python3
"""Optimize compact H orbitals against the full-grid Delta-ST response."""

import argparse
import json
import pathlib
import resource
import sys
import time
import types

import torch


SIAB_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPT_DIR = SIAB_ROOT / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from IO.func_C import read_C_init, write_C
from IO.read_sternheimer_fixed_ao import read_sternheimer_fixed_ao
from IO.read_sternheimer_primitive_galerkin import (
    read_sternheimer_primitive_galerkin,
)
from delta_st_gradient_gate import (
    require_file_sha256,
    run_delta_st_response_optimization,
)
from delta_st_basis_extension import select_metric_complement_shell
from delta_st_parent_space import (
    load_delta_st_reference,
    load_full_coulomb_matrix,
    rpa_correlation_energy,
)
from delta_st_response_compression import (
    FrozenOccupiedDeltaSTCompression,
    anchor_atomic_occupied_radial,
)
from run_h_gradient_gate import (
    FREEZE_SPECS,
    HARTREE_TO_KCAL_MOL,
    _align_candidate,
    _input_hashes,
    _primitive_layout,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=pathlib.Path)
    parser.add_argument("primitive_file", type=pathlib.Path)
    parser.add_argument("fixed_ao_file", type=pathlib.Path)
    parser.add_argument("initial_coefficients", type=pathlib.Path)
    parser.add_argument("initial_orbital", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--sidecar-commit", required=True)
    parser.add_argument("--siab-commit", required=True)
    parser.add_argument("--nu", type=int, nargs="+", default=(3, 2, 0))
    parser.add_argument(
        "--append-l",
        type=int,
        nargs="+",
        default=None,
        help="append deterministic metric-complement radials in this l order",
    )
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--initial-step", type=float, default=0.2)
    parser.add_argument("--maximum-step", type=float, default=2.0)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--relative-loss-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--relative-loss-patience", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_dir = args.reference_dir.resolve()
    primitive_file = args.primitive_file.resolve()
    fixed_ao_file = args.fixed_ao_file.resolve()
    initial_file = args.initial_coefficients.resolve()
    initial_orbital = args.initial_orbital.resolve()
    reference = load_delta_st_reference(reference_dir)
    if reference.provenance["abacus_commit"] != args.reference_commit:
        raise ValueError("reference ABACUS commit differs from --reference-commit")
    initial_orbital_hash = require_file_sha256(
        initial_orbital, reference.provenance["orbital_sha256"]
    )
    primitive = _align_candidate(
        read_sternheimer_primitive_galerkin(primitive_file),
        args.sidecar_commit,
        args.reference_commit,
    )
    fixed_ao = _align_candidate(
        read_sternheimer_fixed_ao(fixed_ao_file),
        args.sidecar_commit,
        args.reference_commit,
    )
    coulomb = load_full_coulomb_matrix(reference_dir, reference.provenance, iq=1)

    element, nprimitive, lmax = _primitive_layout(primitive)
    if element != "H" or len(args.nu) != lmax + 1:
        raise ValueError("the H optimizer requires one Nu value for every primitive l")
    info_element = {
        "H": types.SimpleNamespace(
            Nl=lmax + 1,
            Ne=nprimitive,
            Nu=list(args.nu),
        )
    }
    coefficients, initialization = read_C_init(
        initial_file, info_element, return_metadata=True
    )
    if len(initialization.loaded_indices) != sum(args.nu):
        raise ValueError("initial coefficient file does not define every requested AO")
    coefficients, occupied_anchor = anchor_atomic_occupied_radial(
        primitive,
        fixed_ao,
        coefficients,
        element="H",
    )

    objective = FrozenOccupiedDeltaSTCompression(
        reference,
        primitive,
        fixed_ao,
        coulomb,
        family_name="H",
        relative_rank_tolerance=args.relative_rank_tolerance,
        active_spin_excluded_columns=(0,),
    )
    extensions = []
    for l in args.append_l or ():
        extension = select_metric_complement_shell(
            primitive,
            objective,
            coefficients,
            element="H",
            l=l,
        )
        coefficients = extension.coefficients
        extensions.append(extension)
    optimization = run_delta_st_response_optimization(
        objective,
        coefficients,
        FREEZE_SPECS,
        max_steps=args.max_steps,
        initial_step=args.initial_step,
        maximum_step=args.maximum_step,
        gradient_tolerance=args.gradient_tolerance,
        relative_loss_tolerance=args.relative_loss_tolerance,
        relative_loss_patience=args.relative_loss_patience,
    )

    initial_family = optimization.initial_result.family_results["H"]
    final_family = optimization.final_result.family_results["H"]
    initial_ec = rpa_correlation_energy(
        initial_family.candidate_pi, initial_family.frequency_weight
    )
    final_ec = rpa_correlation_energy(
        final_family.candidate_pi, final_family.frequency_weight
    )
    reference_ec = rpa_correlation_energy(
        initial_family.reference_pi, initial_family.frequency_weight
    )

    initial_output = output_dir / "INITIAL_ORBITAL_RESULTS.txt"
    optimized_output = output_dir / "OPTIMIZED_ORBITAL_RESULTS.txt"
    write_C(initial_output, coefficients, optimization.initial_loss)
    write_C(optimized_output, optimization.coefficients, optimization.final_loss)
    elapsed = time.perf_counter() - started
    radial_orbitals_by_l = [
        int(coefficients["H"][l].shape[1]) for l in range(lmax + 1)
    ]
    diagnostics = {
        "method": "h_delta_st_response_compression_optimization_v4",
        "siab_commit": args.siab_commit,
        "reference_abacus_commit": args.reference_commit,
        "sidecar_abacus_commit": args.sidecar_commit,
        "initial_orbital_sha256": initial_orbital_hash,
        "protocol": dict(reference.provenance),
        "primitive_dimension": int(primitive.overlap.shape[0]),
        "fixed_ao_dimension": int(fixed_ao.overlap.shape[0]),
        "radial_orbitals_by_l": radial_orbitals_by_l,
        "candidate_ao_dimension": int(
            sum(
                (2 * l + 1) * count
                for l, count in enumerate(radial_orbitals_by_l)
            )
        ),
        "frozen_orbitals": list(FREEZE_SPECS),
        "frozen_orbital_roles": [
            "exact_atomic_occupied_s",
            "fixed_s_complement",
            "fixed_first_p",
        ],
        "variable_orbitals": _variable_orbitals(radial_orbitals_by_l),
        "occupied_anchor": _anchor_payload(occupied_anchor),
        "basis_extension": (
            _extension_payload(extensions[0]) if len(extensions) == 1 else None
        ),
        "basis_extensions": [
            _extension_payload(extension) for extension in extensions
        ],
        "active_spin_excluded_columns": [0],
        "relative_rank_tolerance": args.relative_rank_tolerance,
        "optimization_parameters": {
            "max_steps": args.max_steps,
            "initial_step": args.initial_step,
            "maximum_step": args.maximum_step,
            "gradient_tolerance": args.gradient_tolerance,
            "relative_loss_tolerance": args.relative_loss_tolerance,
            "relative_loss_patience": args.relative_loss_patience,
        },
        "stop_reason": optimization.stop_reason,
        "accepted_steps": len(optimization.history) - 1,
        "initial_loss": optimization.initial_loss,
        "final_loss": optimization.final_loss,
        "relative_loss_reduction": (
            optimization.initial_loss - optimization.final_loss
        )
        / optimization.initial_loss,
        "initial_frequency_loss": [
            float(value) for value in initial_family.frequency_loss
        ],
        "final_frequency_loss": [
            float(value) for value in final_family.frequency_loss
        ],
        "history": [_history_payload(record) for record in optimization.history],
        "reference_ec_ha": reference_ec,
        "initial_ec_ha": initial_ec,
        "final_ec_ha": final_ec,
        "initial_ec_error_kcal_mol": (
            initial_ec - reference_ec
        )
        * HARTREE_TO_KCAL_MOL,
        "final_ec_error_kcal_mol": (final_ec - reference_ec)
        * HARTREE_TO_KCAL_MOL,
        "input_sha256": _input_hashes(
            reference_dir,
            primitive_file,
            fixed_ao_file,
            initial_file,
            initial_orbital,
        ),
        "output_sha256": {
            initial_output.name: require_file_sha256(initial_output, None),
            optimized_output.name: require_file_sha256(optimized_output, None),
        },
        "elapsed_seconds": elapsed,
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    diagnostics_path = output_dir / "optimization.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


def _history_payload(record):
    return {
        "iteration": int(record.iteration),
        "loss": float(record.loss),
        "accepted_step": float(record.accepted_step),
        "relative_loss_reduction": float(record.relative_loss_reduction),
        "raw_fixed_gradient_norm": float(record.raw_fixed_gradient_norm),
        "masked_fixed_gradient_norm": float(record.masked_fixed_gradient_norm),
        "variable_gradient_norm": float(record.variable_gradient_norm),
        "maximum_frequency_loss": float(record.maximum_frequency_loss),
        "maximum_condition": float(record.maximum_condition),
        "retained_rank_by_spin": list(record.retained_rank_by_spin),
        "dropped_rank_by_spin": list(record.dropped_rank_by_spin),
    }


def _anchor_payload(anchor):
    return {
        "occupied_band_index": int(anchor.occupied_band_index),
        "omitted_original_s_zeta": int(anchor.omitted_original_s_zeta),
        "fixed_ao_coefficients": [
            float(value) for value in anchor.fixed_ao_coefficients
        ],
        "maximum_off_s_coefficient": float(anchor.maximum_off_s_coefficient),
        "eigenvalue_max_abs_error_ha": float(
            anchor.eigenvalue_max_abs_error_ha
        ),
    }


def _extension_payload(extension):
    return {
        "element": extension.element,
        "l": int(extension.l),
        "selected_mode": int(extension.selected_mode),
        "initial_loss": float(extension.initial_loss),
        "selected_loss": float(extension.selected_loss),
        "candidate_losses": [
            float(value) for value in extension.candidate_losses
        ],
        "radial_metric_condition": float(extension.radial_metric_condition),
        "maximum_metric_orthogonality": float(
            extension.maximum_metric_orthogonality
        ),
        "metric_normalization_error": float(
            extension.metric_normalization_error
        ),
    }


def _variable_orbitals(radial_orbitals_by_l):
    frozen = {
        (spec["element"], int(spec["l"]), int(spec["zeta"]))
        for spec in FREEZE_SPECS
    }
    return [
        f"H/l{l}/zeta{zeta}"
        for l, count in enumerate(radial_orbitals_by_l)
        for zeta in range(1, count + 1)
        if ("H", l, zeta) not in frozen
    ]


if __name__ == "__main__":
    main()
