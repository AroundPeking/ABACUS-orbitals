#!/usr/bin/env python3
"""Diagnose a periodic SIAB candidate's generalized KS spectrum."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from periodic_galerkin_basis import (  # noqa: E402
    build_primitive_to_candidate,
    contract_periodic_candidate_operators,
    read_periodic_optimizer_coefficients,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset  # noqa: E402


HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class GeneralizedSpectrum:
    eigenvalue_ha: torch.Tensor
    rank: int
    condition: float


def _adjoint(value):
    return value.transpose(-2, -1).conj()


def solve_generalized_spectrum(
    overlap,
    hamiltonian_ha,
    *,
    relative_rank_tolerance=1.0e-12,
):
    """Solve H c = epsilon S c without hiding dependent AO directions."""
    if (
        not isinstance(overlap, torch.Tensor)
        or not isinstance(hamiltonian_ha, torch.Tensor)
        or overlap.device.type != "cpu"
        or hamiltonian_ha.device.type != "cpu"
        or overlap.dtype != torch.complex128
        or hamiltonian_ha.dtype != torch.complex128
        or overlap.ndim != 2
        or overlap.shape[0] != overlap.shape[1]
        or hamiltonian_ha.shape != overlap.shape
        or not bool(torch.isfinite(overlap).all())
        or not bool(torch.isfinite(hamiltonian_ha).all())
    ):
        raise ValueError("overlap and Hamiltonian must be finite CPU complex128 squares")
    if (
        not isinstance(relative_rank_tolerance, (int, float))
        or isinstance(relative_rank_tolerance, bool)
        or not math.isfinite(relative_rank_tolerance)
        or not 0.0 < relative_rank_tolerance < 1.0
    ):
        raise ValueError("relative_rank_tolerance must lie between zero and one")

    overlap = 0.5 * (overlap + _adjoint(overlap))
    hamiltonian_ha = 0.5 * (hamiltonian_ha + _adjoint(hamiltonian_ha))
    overlap_eigenvalue, overlap_eigenvector = torch.linalg.eigh(overlap)
    maximum = torch.max(overlap_eigenvalue)
    if float(maximum.detach()) <= 0.0:
        raise RuntimeError("candidate overlap has no positive direction")
    retained = overlap_eigenvalue > relative_rank_tolerance * maximum
    if bool(torch.any(~retained)):
        raise RuntimeError("candidate overlap is rank deficient")
    minimum = torch.min(overlap_eigenvalue)
    condition = float((maximum / minimum).detach())
    lowdin = overlap_eigenvector @ torch.diag(
        overlap_eigenvalue.rsqrt()
    ).to(torch.complex128)
    orthonormal_hamiltonian = (
        _adjoint(lowdin).matmul(hamiltonian_ha).matmul(lowdin)
    )
    eigenvalue_ha = torch.linalg.eigvalsh(
        0.5 * (orthonormal_hamiltonian + _adjoint(orthonormal_hamiltonian))
    )
    return GeneralizedSpectrum(
        eigenvalue_ha=eigenvalue_ha,
        rank=int(eigenvalue_ha.numel()),
        condition=condition,
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_nu(value):
    try:
        counts = tuple(int(field.strip()) for field in value.split(","))
    except ValueError as error:
        raise ValueError("nu must be a comma-separated integer list") from error
    if not counts or any(count < 0 for count in counts) or not any(counts):
        raise ValueError("nu must contain nonnegative counts and be nonempty")
    return counts


def analyze_candidate(dataset, coefficients, *, relative_rank_tolerance):
    candidate = build_primitive_to_candidate(
        dataset.primitive_blocks,
        dataset.primitive_count,
        coefficients,
    )
    records = []
    for record in dataset.kpoints:
        operators = contract_periodic_candidate_operators(
            record,
            dataset.primitive_blocks,
            coefficients,
        )
        spectrum = solve_generalized_spectrum(
            operators.overlap,
            operators.hamiltonian_ha,
            relative_rank_tolerance=relative_rank_tolerance,
        )
        noccupied = int(record.occupation.numel())
        if not 0 < noccupied < spectrum.rank:
            raise RuntimeError("candidate spectrum has no virtual complement")
        value = spectrum.eigenvalue_ha
        records.append(
            {
                "source_ik": record.source_ik,
                "target_ik": record.target_ik,
                "target_kpoint": list(record.target_kpoint),
                "rank": spectrum.rank,
                "overlap_condition": spectrum.condition,
                "minimum_eigenvalue_ev": float(value[0]) * HARTREE_TO_EV,
                "highest_occupied_eigenvalue_ev": (
                    float(value[noccupied - 1]) * HARTREE_TO_EV
                ),
                "lowest_virtual_eigenvalue_ev": (
                    float(value[noccupied]) * HARTREE_TO_EV
                ),
                "maximum_eigenvalue_ev": float(value[-1]) * HARTREE_TO_EV,
                "maximum_transition_energy_ev": (
                    float(value[-1] - torch.min(record.source_eigenvalue_ha))
                    * HARTREE_TO_EV
                ),
            }
        )
    worst_condition = max(records, key=lambda item: item["overlap_condition"])
    highest_energy = max(records, key=lambda item: item["maximum_eigenvalue_ev"])
    largest_transition = max(
        records, key=lambda item: item["maximum_transition_energy_ev"]
    )
    smallest_gap = min(
        records,
        key=lambda item: (
            item["lowest_virtual_eigenvalue_ev"]
            - item["highest_occupied_eigenvalue_ev"]
        ),
    )
    return {
        "scope": (
            "fixed-Hamiltonian Galerkin spectrum diagnostic; not an independent "
            "ABACUS SCF, SOS, or Delta-ST energy validation"
        ),
        "ao_count_cell": int(candidate.transform.shape[1]),
        "maximum_overlap_condition": worst_condition["overlap_condition"],
        "maximum_overlap_condition_record": worst_condition,
        "maximum_eigenvalue_ev": highest_energy["maximum_eigenvalue_ev"],
        "maximum_eigenvalue_record": highest_energy,
        "maximum_transition_energy_ev": largest_transition[
            "maximum_transition_energy_ev"
        ],
        "maximum_transition_record": largest_transition,
        "minimum_candidate_gap_ev": (
            smallest_gap["lowest_virtual_eigenvalue_ev"]
            - smallest_gap["highest_occupied_eigenvalue_ev"]
        ),
        "minimum_candidate_gap_record": smallest_gap,
        "kpoints": records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--nu", required=True)
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-12)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_path = args.dataset.resolve()
    coefficient_path = args.coefficients.resolve()
    output_path = args.output.resolve()
    if not dataset_path.is_dir() or dataset_path.is_symlink():
        raise ValueError("dataset must be a real directory")
    if not coefficient_path.is_file() or coefficient_path.is_symlink():
        raise ValueError("coefficients must be a regular file")
    if output_path.exists():
        raise FileExistsError(output_path)
    nu = parse_nu(args.nu)
    dataset = read_periodic_galerkin_dataset(dataset_path)
    coefficients = read_periodic_optimizer_coefficients(
        coefficient_path,
        element=args.element,
        radial_rows=args.radial_rows,
        expected_nu=nu,
    )
    report = analyze_candidate(
        dataset,
        coefficients,
        relative_rank_tolerance=args.relative_rank_tolerance,
    )
    report.update(
        {
            "format_version": 1,
            "label": args.label,
            "nu": list(nu),
            "inputs": {
                "dataset": str(dataset_path),
                "physics_hash": dataset.physics_hash,
                "selected_iq": dataset.selected_iq,
                "coefficients": str(coefficient_path),
                "coefficients_sha256": _sha256(coefficient_path),
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
