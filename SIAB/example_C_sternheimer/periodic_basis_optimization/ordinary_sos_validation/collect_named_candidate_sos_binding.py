#!/usr/bin/env python3
"""Collect an ordinary-SOS C atom-solid binding energy against a fixed reference."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
BINDING_GATE_KCAL_MOL_PER_C = 0.1
REQUIRED = {
    "status",
    "side",
    "method",
    "scope",
    "coulomb_kernel",
    "selected_orbital_sha256",
    "frequency_grid_sha256",
    "naux",
    "reference_ha",
    "ecrpa_ha",
}


def _read_endpoint(path: Path, side: str) -> dict:
    values = {}
    for number, raw in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 1)
        if len(fields) != 2 or fields[0] in values:
            raise ValueError(f"{path}: malformed or duplicate line {number}")
        values[fields[0]] = fields[1]
    missing = REQUIRED - values.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    if values["status"] != "success" or values["side"] != side or values["method"] != "sos":
        raise ValueError(f"{path}: invalid endpoint identity")
    if values["scope"] != "body_only_no_analytic_headwing":
        raise ValueError(f"{path}: invalid endpoint scope")
    if values["coulomb_kernel"] != "full_periodic_poisson":
        raise ValueError(f"{path}: invalid Coulomb kernel")
    for key in ("selected_orbital_sha256", "frequency_grid_sha256"):
        value = values[key]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{path}: invalid {key}")
    values["naux"] = int(values["naux"])
    values["reference_ha"] = float(values["reference_ha"])
    values["ecrpa_ha"] = float(values["ecrpa_ha"])
    if values["naux"] <= 0 or not all(
        math.isfinite(values[key]) for key in ("reference_ha", "ecrpa_ha")
    ):
        raise ValueError(f"{path}: invalid numeric endpoint")
    return values


def collect_binding_energy(*, atom_sos: Path, solid_sos: Path, delta_reference_ev_per_c: float) -> dict:
    atom = _read_endpoint(atom_sos, "atom")
    solid = _read_endpoint(solid_sos, "solid")
    if atom["selected_orbital_sha256"] != solid["selected_orbital_sha256"]:
        raise ValueError("atom and solid orbital SHA256 mismatch")
    if solid["naux"] != 2 * atom["naux"]:
        raise ValueError("solid auxiliary dimension must be twice the atom dimension")
    reference = float(delta_reference_ev_per_c)
    if not math.isfinite(reference):
        raise ValueError("fixed reference must be finite")

    zero_order_ha = atom["reference_ha"] - 0.5 * solid["reference_ha"]
    correlation_ha = atom["ecrpa_ha"] - 0.5 * solid["ecrpa_ha"]
    total_ev = (zero_order_ha + correlation_ha) * HARTREE_TO_EV
    difference_ev = total_ev - reference
    difference_kcal = difference_ev / HARTREE_TO_EV * HARTREE_TO_KCAL_MOL
    return {
        "status": "success",
        "quantity": "solid_binding_energy_per_carbon_atom",
        "scope": "body_only_no_analytic_headwing",
        "selected_orbital_sha256": atom["selected_orbital_sha256"],
        "atom_frequency_grid_sha256": atom["frequency_grid_sha256"],
        "solid_frequency_grid_sha256": solid["frequency_grid_sha256"],
        "atom_naux": atom["naux"],
        "solid_naux": solid["naux"],
        "zero_order_binding_ha_per_c": zero_order_ha,
        "correlation_binding_ha_per_c": correlation_ha,
        "sos_total_binding_ha_per_c": zero_order_ha + correlation_ha,
        "zero_order_binding_ev_per_c": zero_order_ha * HARTREE_TO_EV,
        "correlation_binding_ev_per_c": correlation_ha * HARTREE_TO_EV,
        "sos_total_binding_ev_per_c": total_ev,
        "delta_st_reference_ev_per_c": reference,
        "difference_from_delta_ev_per_c": difference_ev,
        "difference_from_delta_kcal_mol_per_c": difference_kcal,
        "binding_gate_kcal_mol_per_c": BINDING_GATE_KCAL_MOL_PER_C,
        "basis_full_body_gate": (
            "pass" if abs(difference_kcal) < BINDING_GATE_KCAL_MOL_PER_C else "fail"
        ),
        "atom_endpoint": str(Path(atom_sos).resolve()),
        "solid_endpoint": str(Path(solid_sos).resolve()),
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
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
    parser.add_argument("--atom-sos", required=True, type=Path)
    parser.add_argument("--solid-sos", required=True, type=Path)
    parser.add_argument("--delta-reference-ev-per-c", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    payload = collect_binding_energy(
        atom_sos=args.atom_sos,
        solid_sos=args.solid_sos,
        delta_reference_ev_per_c=args.delta_reference_ev_per_c,
    )
    _atomic_write(args.output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
