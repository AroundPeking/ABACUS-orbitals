#!/usr/bin/env python3
"""Gate a relaxed C DZP candidate against the original-TZDP PBE energies."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


REFERENCE_ATOM_EV = -147.4773363622974
REFERENCE_SOLID_C2_EV = -309.8590826137842
DEFAULT_TOLERANCE_EV = 0.010


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def collect_pbe_gate(
    *,
    atom_energy_ev: float,
    solid_c2_energy_ev: float,
    tolerance_ev: float = DEFAULT_TOLERANCE_EV,
) -> dict:
    atom = _finite(atom_energy_ev, "atom energy")
    solid = _finite(solid_c2_energy_ev, "solid C2 energy")
    tolerance = _finite(tolerance_ev, "PBE tolerance")
    if tolerance <= 0.0:
        raise ValueError("PBE tolerance must be positive")

    reference_binding = REFERENCE_ATOM_EV - 0.5 * REFERENCE_SOLID_C2_EV
    candidate_binding = atom - 0.5 * solid
    atom_difference = atom - REFERENCE_ATOM_EV
    solid_difference_per_c = 0.5 * (solid - REFERENCE_SOLID_C2_EV)
    binding_difference = candidate_binding - reference_binding
    checks = {
        "atom_energy": abs(atom_difference) <= tolerance,
        "solid_energy_per_c": abs(solid_difference_per_c) <= tolerance,
        "binding_energy": abs(binding_difference) <= tolerance,
    }
    return {
        "status": "success",
        "quantity": "c_relaxed_dzp_pbe_reference_gate",
        "reference_basis": "original_unoptimized_sg15_tzdp",
        "candidate_atom_energy_ev": atom,
        "candidate_solid_c2_energy_ev": solid,
        "reference_atom_energy_ev": REFERENCE_ATOM_EV,
        "reference_solid_c2_energy_ev": REFERENCE_SOLID_C2_EV,
        "candidate_binding_ev_per_c": candidate_binding,
        "reference_binding_ev_per_c": reference_binding,
        "atom_energy_difference_ev": atom_difference,
        "solid_energy_difference_ev_per_c": solid_difference_per_c,
        "binding_energy_difference_ev_per_c": binding_difference,
        "tolerance_ev": tolerance,
        "checks": checks,
        "pbe_gate": "pass" if all(checks.values()) else "fail",
    }


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-energy-ev", required=True, type=float)
    parser.add_argument("--solid-c2-energy-ev", required=True, type=float)
    parser.add_argument("--tolerance-ev", type=float, default=DEFAULT_TOLERANCE_EV)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = collect_pbe_gate(
        atom_energy_ev=args.atom_energy_ev,
        solid_c2_energy_ev=args.solid_c2_energy_ev,
        tolerance_ev=args.tolerance_ev,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["pbe_gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
