#!/usr/bin/env python3
"""Run the real-H Delta-ST response-compression gradient gate."""

import argparse
import dataclasses
import hashlib
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
from delta_st_gradient_gate import run_delta_st_gradient_gate
from delta_st_parent_space import (
    load_delta_st_reference,
    load_full_coulomb_matrix,
    rpa_correlation_energy,
)
from delta_st_response_compression import FrozenOccupiedDeltaSTCompression


HARTREE_TO_KCAL_MOL = 627.5094740631
FREEZE_SPECS = (
    {"element": "H", "l": 0, "zeta": 1},
    {"element": "H", "l": 0, "zeta": 2},
    {"element": "H", "l": 1, "zeta": 1},
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=pathlib.Path)
    parser.add_argument("primitive_file", type=pathlib.Path)
    parser.add_argument("fixed_ao_file", type=pathlib.Path)
    parser.add_argument("initial_coefficients", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--sidecar-commit", required=True)
    parser.add_argument("--siab-commit", required=True)
    parser.add_argument("--nu", type=int, nargs="+", default=(3, 2, 0))
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=(0.02, 0.01, 0.005, 0.002, 0.001),
    )
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
    reference = load_delta_st_reference(reference_dir)
    if reference.provenance["abacus_commit"] != args.reference_commit:
        raise ValueError("reference ABACUS commit differs from --reference-commit")
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
    coulomb = load_full_coulomb_matrix(
        reference_dir, reference.provenance, iq=1
    )

    element, nprimitive, lmax = _primitive_layout(primitive)
    if element != "H" or len(args.nu) != lmax + 1:
        raise ValueError("the H gate requires one Nu value for every primitive l")
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
    requested = sum(args.nu)
    if len(initialization.loaded_indices) != requested:
        raise ValueError("initial coefficient file does not define every requested AO")

    objective = FrozenOccupiedDeltaSTCompression(
        reference, primitive, fixed_ao, coulomb, family_name="H"
    )
    gate = run_delta_st_gradient_gate(
        objective,
        coefficients,
        FREEZE_SPECS,
        step_sizes=args.step_sizes,
    )
    initial_family = gate.initial_result.family_results["H"]
    accepted_family = gate.accepted_result.family_results["H"]
    initial_ec = rpa_correlation_energy(
        initial_family.candidate_pi, initial_family.frequency_weight
    )
    accepted_ec = rpa_correlation_energy(
        accepted_family.candidate_pi, accepted_family.frequency_weight
    )
    reference_ec = rpa_correlation_energy(
        initial_family.reference_pi, initial_family.frequency_weight
    )

    initial_output = output_dir / "INITIAL_ORBITAL_RESULTS.txt"
    accepted_output = output_dir / "ACCEPTED_ORBITAL_RESULTS.txt"
    write_C(initial_output, coefficients, gate.initial_loss)
    write_C(accepted_output, gate.coefficients, gate.accepted_loss)
    input_hashes = _input_hashes(
        reference_dir, primitive_file, fixed_ao_file, initial_file
    )
    elapsed = time.perf_counter() - started
    diagnostics = {
        "method": "h_delta_st_response_compression_gradient_gate_v1",
        "siab_commit": args.siab_commit,
        "reference_abacus_commit": args.reference_commit,
        "sidecar_abacus_commit": args.sidecar_commit,
        "protocol": dict(reference.provenance),
        "primitive_dimension": int(primitive.overlap.shape[0]),
        "fixed_ao_dimension": int(fixed_ao.overlap.shape[0]),
        "radial_orbitals_by_l": list(args.nu),
        "candidate_ao_dimension": int(
            sum((2 * l + 1) * count for l, count in enumerate(args.nu))
        ),
        "frozen_orbitals": list(FREEZE_SPECS),
        "variable_orbitals": ["H/l0/zeta3", "H/l1/zeta2"],
        "initial_loss": gate.initial_loss,
        "accepted_loss": gate.accepted_loss,
        "relative_loss_reduction": (
            gate.initial_loss - gate.accepted_loss
        ) / gate.initial_loss,
        "accepted_step": gate.accepted_step,
        "raw_fixed_gradient_norm": gate.raw_fixed_gradient_norm,
        "masked_fixed_gradient_norm": gate.masked_fixed_gradient_norm,
        "variable_gradient_norm": gate.variable_gradient_norm,
        "initial_frequency_loss": [
            float(value) for value in initial_family.frequency_loss
        ],
        "accepted_frequency_loss": [
            float(value) for value in accepted_family.frequency_loss
        ],
        "initial_maximum_frequency_loss": float(
            torch.max(initial_family.frequency_loss)
        ),
        "accepted_maximum_frequency_loss": float(
            torch.max(accepted_family.frequency_loss)
        ),
        "retained_rank_by_spin": list(accepted_family.retained_rank_by_spin),
        "dropped_rank_by_spin": list(accepted_family.dropped_rank_by_spin),
        "maximum_candidate_condition": accepted_family.max_candidate_condition,
        "reference_ec_ha": reference_ec,
        "initial_ec_ha": initial_ec,
        "accepted_ec_ha": accepted_ec,
        "initial_ec_error_kcal_mol": (
            initial_ec - reference_ec
        ) * HARTREE_TO_KCAL_MOL,
        "accepted_ec_error_kcal_mol": (
            accepted_ec - reference_ec
        ) * HARTREE_TO_KCAL_MOL,
        "input_sha256": input_hashes,
        "output_sha256": {
            initial_output.name: _sha256(initial_output),
            accepted_output.name: _sha256(accepted_output),
        },
        "elapsed_seconds": elapsed,
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    diagnostics_path = output_dir / "gradient_gate.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


def _align_candidate(value, sidecar_commit, reference_commit):
    if value.provenance["abacus_commit"] != sidecar_commit:
        raise ValueError("sidecar ABACUS commit differs from --sidecar-commit")
    provenance = dict(value.provenance)
    provenance["abacus_commit"] = reference_commit
    return dataclasses.replace(value, provenance=provenance)


def _primitive_layout(primitive):
    elements = {block.element for block in primitive.blocks}
    if len(elements) != 1:
        raise ValueError("the H gate requires one primitive element")
    radial_counts = {}
    for block in primitive.blocks:
        radial_counts.setdefault(block.l, set()).add(block.n_primitive)
    if sorted(radial_counts) != list(range(max(radial_counts) + 1)):
        raise ValueError("primitive angular channels must be contiguous from l=0")
    counts = {next(iter(values)) for values in radial_counts.values() if len(values) == 1}
    if len(counts) != 1 or any(len(values) != 1 for values in radial_counts.values()):
        raise ValueError("the H gate requires one radial primitive count")
    return next(iter(elements)), counts.pop(), max(radial_counts)


def _input_hashes(reference_dir, primitive_file, fixed_ao_file, initial_file):
    paths = [
        primitive_file,
        fixed_ao_file,
        initial_file,
        reference_dir / "reference_protocol.json",
        reference_dir / "STERNHEIMER_ABFS_CHANNELS.dat",
        reference_dir / "v1_coulomb_full_iq_1_rank0.dat",
    ]
    paths.extend(
        sorted(reference_dir.glob("v1_sternheimer_chi0_iq_1_ifreq_*_rank*.dat"))
    )
    return {str(path): _sha256(path) for path in paths}


def _sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
