from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ZeroOrderAudit:
    case: str
    passed: bool
    occupied_state_count: int
    grid: Tuple[int, int, int]
    max_occupation_abs_diff: float
    max_occupied_eigenvalue_abs_diff_ha: float
    final_total_energy_abs_diff_ha: float
    source_file_paths: Tuple[Tuple[str, str], ...]
    source_file_sha256: Tuple[Tuple[str, str], ...]
    thresholds: Tuple[Tuple[str, float], ...]
