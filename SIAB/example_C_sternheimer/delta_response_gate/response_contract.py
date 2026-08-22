from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EV_PER_HA = 27.211386245988
INTEGER_TOL = 1.0e-10
ACCEPTED_PHASES = {
    "fixed": "runs/fixed/fixed_zero_restart",
    "free": "runs/dir0/free_restart2",
}

FROZEN_RESPONSE_PROTOCOL = (
    ("suffix", "C_DELTA_RESPONSE_GATE"),
    ("calculation", "scf"),
    ("ntype", "1"),
    ("nelec", "4"),
    ("nspin", "2"),
    ("nupdown", "2"),
    ("nbands", "22"),
    ("basis_type", "lcao"),
    ("ecutwfc", "30"),
    ("lcao_ecut", "100"),
    ("nx", "135"),
    ("ny", "135"),
    ("nz", "135"),
    ("ks_solver", "genelpa"),
    ("dft_functional", "pbe"),
    ("symmetry", "0"),
    ("gamma_only", "1"),
    ("kpar", "1"),
    ("pseudo_dir", "./"),
    ("orbital_dir", "./"),
    ("scf_thr", "1e-10"),
    ("scf_nmax", "300"),
    ("mixing_type", "broyden"),
    ("mixing_beta", "0.3"),
    ("mixing_beta_mag", "0.3"),
    ("smearing_method", "fixed"),
    ("out_chg", "1"),
    ("out_wfc_lcao", "1"),
    ("out_app_flag", "1"),
    ("out_mul", "1"),
    ("efield_flag", "0"),
    ("efield_amp", "0"),
    ("init_wfc", "file"),
    ("init_chg", "file"),
    ("rpa", "1"),
    ("out_librpa_reader_version", "1"),
    ("exx_pca_threshold", "1e-4"),
    ("exx_singularity_correction", "massidda"),
    ("exx_ccp_rmesh_times", "1"),
    ("rpa_ccp_rmesh_times", "1"),
    ("out_sternheimer_librpa", "1"),
    ("sternheimer_nfreq", "6"),
    ("sternheimer_frequency_grid_file", "fixed_frequency_grid.dat"),
    ("sternheimer_frequency_mpi", "1"),
    ("sternheimer_delta", "1"),
    ("sternheimer_fd_order", "8"),
    ("sternheimer_delta_max_states", "0"),
    ("sternheimer_delta_norm_tol", "1e-10"),
)

_SPIN_HEADER = re.compile(r"^\s*spin=(\d+)\s+k-point=(\d+)/(\d+)\b")
_STATE_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s*$")


@dataclass(frozen=True)
class EigOccRecord:
    path: Path
    spin_counts: dict[int, int]
    nbands_by_spin: dict[int, int]
    transition_min_ha: float
    transition_max_ha: float


def parse_eig_occ(path: Path) -> EigOccRecord:
    rows: dict[int, list[tuple[float, float]]] = {}
    current_spin: int | None = None
    for line in path.read_text(encoding="ascii").splitlines():
        header = _SPIN_HEADER.match(line)
        if header:
            spin, kpoint, total_kpoints = map(int, header.groups())
            if kpoint != 1 or total_kpoints != 1 or spin in rows:
                raise ValueError("eig_occ must contain one Gamma record for each spin")
            current_spin = spin
            rows[spin] = []
            continue
        state = _STATE_ROW.match(line)
        if state and current_spin is not None:
            _, energy_token, occupation_token = state.groups()
            energy = float(energy_token)
            occupation = float(occupation_token)
            if not (math.isfinite(energy) and math.isfinite(occupation)):
                raise ValueError("eig_occ contains non-finite data")
            if min(abs(occupation), abs(occupation - 1.0)) > INTEGER_TOL:
                raise ValueError("eig_occ occupations must be integer zero or one")
            rows[current_spin].append((energy, round(occupation)))

    if set(rows) != {1, 2} or any(not values for values in rows.values()):
        raise ValueError("eig_occ must contain two nonempty spin records")

    spin_counts = {spin: sum(occupation for _, occupation in values) for spin, values in rows.items()}
    if spin_counts != {1: 3, 2: 1}:
        raise ValueError("eig_occ does not describe the C triplet with spin counts 3 and 1")

    minima = []
    maxima = []
    for spin, values in rows.items():
        occupied = [energy for energy, occupation in values if occupation == 1]
        virtual = [energy for energy, occupation in values if occupation == 0]
        if not occupied or not virtual:
            raise ValueError(f"spin {spin} lacks occupied or virtual states")
        minima.append(min(virtual) - max(occupied))
        maxima.append(max(virtual) - min(occupied))
    transition_min_ev = min(minima)
    transition_max_ev = max(maxima)
    if transition_min_ev <= 0.0 or transition_max_ev <= transition_min_ev:
        raise ValueError("eig_occ has an invalid occupied-to-virtual transition window")

    return EigOccRecord(
        path=path.resolve(),
        spin_counts=spin_counts,
        nbands_by_spin={spin: len(values) for spin, values in rows.items()},
        transition_min_ha=transition_min_ev / EV_PER_HA,
        transition_max_ha=transition_max_ev / EV_PER_HA,
    )


def union_transition_window(records: Iterable[EigOccRecord]) -> tuple[float, float]:
    records = tuple(records)
    if not records:
        raise ValueError("at least one eig_occ record is required")
    minimum = min(record.transition_min_ha for record in records)
    maximum = max(record.transition_max_ha for record in records)
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0.0 < minimum < maximum):
        raise ValueError("invalid union transition window")
    return minimum, maximum


def render_response_input(branch: str) -> str:
    if branch not in ACCEPTED_PHASES:
        raise ValueError(f"unsupported response branch: {branch}")
    values = [("INPUT_PARAMETERS", None), *FROZEN_RESPONSE_PROTOCOL]
    values.append(("ocp", "1" if branch == "fixed" else "0"))
    if branch == "fixed":
        values.append(("ocp_set", "3*1 19*0 1*1 21*0"))
    return "\n".join(
        key if value is None else f"{key} {value}" for key, value in values
    ) + "\n"
