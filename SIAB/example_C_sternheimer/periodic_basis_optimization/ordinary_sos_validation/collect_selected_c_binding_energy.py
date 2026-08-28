#!/usr/bin/env python3
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

REQUIRED_KEYS = {
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


def _read_summary(path: Path) -> dict[str, str]:
    path = Path(path).resolve(strict=True)
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 1)
        if len(fields) != 2:
            raise ValueError(f"{path}: malformed line {number}")
        key, value = fields
        if key in values:
            raise ValueError(f"{path}: duplicate key {key}")
        values[key] = value.strip()
    missing = sorted(REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"{path}: missing keys: {', '.join(missing)}")
    return values


def _validated_endpoint(path: Path, *, side: str, method: str) -> dict:
    raw = _read_summary(path)
    if raw["status"] != "success":
        raise ValueError(f"{path}: endpoint status is not success")
    if raw["side"] != side or raw["method"] != method:
        raise ValueError(
            f"{path}: expected side={side} method={method}, "
            f"found side={raw['side']} method={raw['method']}"
        )
    if raw["scope"] != "body_only_no_analytic_headwing":
        raise ValueError(f"{path}: endpoint is not the body-only comparison")
    if raw["coulomb_kernel"] != "full_periodic_poisson":
        raise ValueError(f"{path}: endpoint does not use full periodic Coulomb")
    for key in ("selected_orbital_sha256", "frequency_grid_sha256"):
        value = raw[key]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{path}: invalid {key}")
    try:
        naux = int(raw["naux"])
        reference = float(raw["reference_ha"])
        ecrpa = float(raw["ecrpa_ha"])
    except ValueError as error:
        raise ValueError(f"{path}: invalid numeric endpoint field") from error
    if naux <= 0 or not all(math.isfinite(value) for value in (reference, ecrpa)):
        raise ValueError(f"{path}: non-finite or non-positive endpoint field")
    return {
        **raw,
        "path": str(Path(path).resolve()),
        "naux": naux,
        "reference_ha": reference,
        "ecrpa_ha": ecrpa,
    }


def _require_equal(left: dict, right: dict, key: str, label: str) -> None:
    if left[key] != right[key]:
        raise ValueError(f"{label} mismatch: {key}")


def _validate_pair(sos: dict, delta: dict, label: str) -> None:
    for key in (
        "scope",
        "coulomb_kernel",
        "selected_orbital_sha256",
        "frequency_grid_sha256",
        "naux",
    ):
        _require_equal(sos, delta, key, label)
    if not math.isclose(
        sos["reference_ha"], delta["reference_ha"], rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError(f"{label} mismatch: reference_ha")


def collect_binding_energy(
    *,
    atom_sos: Path,
    atom_delta: Path,
    solid_sos: Path,
    solid_delta: Path,
) -> dict:
    endpoints = {
        "atom_sos": _validated_endpoint(atom_sos, side="atom", method="sos"),
        "atom_delta": _validated_endpoint(
            atom_delta, side="atom", method="delta_st"
        ),
        "solid_sos": _validated_endpoint(solid_sos, side="solid", method="sos"),
        "solid_delta": _validated_endpoint(
            solid_delta, side="solid", method="delta_st"
        ),
    }
    _validate_pair(endpoints["atom_sos"], endpoints["atom_delta"], "atom frequency")
    _validate_pair(endpoints["solid_sos"], endpoints["solid_delta"], "solid frequency")
    _require_equal(
        endpoints["atom_sos"],
        endpoints["solid_sos"],
        "selected_orbital_sha256",
        "atom/solid selected orbital",
    )
    _require_equal(
        endpoints["atom_sos"],
        endpoints["solid_sos"],
        "coulomb_kernel",
        "atom/solid Coulomb kernel",
    )
    if endpoints["solid_sos"]["naux"] != 2 * endpoints["atom_sos"]["naux"]:
        raise ValueError("solid auxiliary dimension must be twice the atomic dimension")

    atom_e0 = endpoints["atom_sos"]["reference_ha"]
    solid_e0 = endpoints["solid_sos"]["reference_ha"]
    sos_atom_ec = endpoints["atom_sos"]["ecrpa_ha"]
    sos_solid_ec = endpoints["solid_sos"]["ecrpa_ha"]
    delta_atom_ec = endpoints["atom_delta"]["ecrpa_ha"]
    delta_solid_ec = endpoints["solid_delta"]["ecrpa_ha"]

    sos_correlation_binding = sos_atom_ec - 0.5 * sos_solid_ec
    delta_correlation_binding = delta_atom_ec - 0.5 * delta_solid_ec
    sos_total_binding = (atom_e0 + sos_atom_ec) - 0.5 * (solid_e0 + sos_solid_ec)
    delta_total_binding = (atom_e0 + delta_atom_ec) - 0.5 * (
        solid_e0 + delta_solid_ec
    )
    difference = sos_total_binding - delta_total_binding
    difference_kcal = difference * HARTREE_TO_KCAL_MOL

    return {
        "status": "success",
        "quantity": "solid_binding_energy_per_carbon_atom",
        "solid_atoms_per_cell": 2,
        "selected_orbital_sha256": endpoints["atom_sos"][
            "selected_orbital_sha256"
        ],
        "atom_frequency_grid_sha256": endpoints["atom_sos"][
            "frequency_grid_sha256"
        ],
        "solid_frequency_grid_sha256": endpoints["solid_sos"][
            "frequency_grid_sha256"
        ],
        "atom_naux": endpoints["atom_sos"]["naux"],
        "solid_naux": endpoints["solid_sos"]["naux"],
        "atom_reference_ha": atom_e0,
        "solid_reference_ha": solid_e0,
        "sos_correlation_binding_ha_per_c": sos_correlation_binding,
        "delta_st_correlation_binding_ha_per_c": delta_correlation_binding,
        "sos_total_binding_ha_per_c": sos_total_binding,
        "delta_st_total_binding_ha_per_c": delta_total_binding,
        "sos_total_binding_ev_per_c": sos_total_binding * HARTREE_TO_EV,
        "delta_st_total_binding_ev_per_c": delta_total_binding * HARTREE_TO_EV,
        "binding_difference_ha_per_c": difference,
        "binding_difference_ev_per_c": difference * HARTREE_TO_EV,
        "binding_difference_kcal_mol_per_c": difference_kcal,
        "binding_gate_kcal_mol_per_c": BINDING_GATE_KCAL_MOL_PER_C,
        "basis_full_body_gate": (
            "pass"
            if abs(difference_kcal) < BINDING_GATE_KCAL_MOL_PER_C
            else "fail"
        ),
        "endpoint_files": {
            name: endpoint["path"] for name, endpoint in endpoints.items()
        },
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
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
    parser = argparse.ArgumentParser(
        description="Compare selected-basis SOS and Delta-ST solid binding energies"
    )
    parser.add_argument("--atom-sos", required=True, type=Path)
    parser.add_argument("--atom-delta", required=True, type=Path)
    parser.add_argument("--solid-sos", required=True, type=Path)
    parser.add_argument("--solid-delta", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect_binding_energy(
        atom_sos=args.atom_sos,
        atom_delta=args.atom_delta,
        solid_sos=args.solid_sos,
        solid_delta=args.solid_delta,
    )
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    _atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
