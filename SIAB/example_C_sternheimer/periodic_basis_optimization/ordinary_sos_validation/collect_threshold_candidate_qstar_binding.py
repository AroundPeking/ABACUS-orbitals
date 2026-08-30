#!/usr/bin/env python3
"""Collect an atom/solid sparse-q SOS binding energy against Delta-ST."""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


HARTREE_TO_EV = 27.211386245988


def _read_summary(path):
    path = Path(path).resolve(strict=True)
    values = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            values[key] = value.strip()
    if values.get("status") != "success":
        raise ValueError(f"summary is not successful: {path}")
    return path, values


def _finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def collect_binding(
    *,
    atom_summary,
    solid_summary,
    delta_reference_ev_per_c,
    tolerance_ev_per_c,
):
    atom_path, atom = _read_summary(atom_summary)
    solid_path, solid = _read_summary(solid_summary)
    if atom.get("side") != "atom" or solid.get("side") != "solid":
        raise ValueError("summary sides must be atom and solid")
    orbital_sha = atom.get("selected_orbital_sha256")
    if not orbital_sha or orbital_sha != solid.get("selected_orbital_sha256"):
        raise ValueError("atom and solid orbital SHA256 values do not match")
    delta_reference = _finite(delta_reference_ev_per_c, "Delta-ST reference")
    tolerance = _finite(tolerance_ev_per_c, "binding tolerance")
    if tolerance <= 0.0:
        raise ValueError("binding tolerance must be positive")

    atom_zero = _finite(atom["reference_ha"], "atom zero-order energy")
    solid_zero = _finite(solid["reference_ha"], "solid zero-order energy")
    atom_correlation = _finite(atom["ecrpa_ha"], "atom correlation energy")
    solid_correlation = _finite(solid["ecrpa_ha"], "solid correlation energy")
    zero_binding = (atom_zero - 0.5 * solid_zero) * HARTREE_TO_EV
    correlation_binding = (
        atom_correlation - 0.5 * solid_correlation
    ) * HARTREE_TO_EV
    total_binding = zero_binding + correlation_binding
    difference = total_binding - delta_reference
    return {
        "status": "success",
        "quantity": "c_atom_diamond_ordinary_sos_binding_energy",
        "atom_summary": str(atom_path),
        "solid_summary": str(solid_path),
        "selected_orbital_sha256": orbital_sha,
        "atom_zero_order_ha": atom_zero,
        "solid_zero_order_ha": solid_zero,
        "atom_ecrpa_ha": atom_correlation,
        "solid_ecrpa_qstar_reconstructed_ha": solid_correlation,
        "zero_order_binding_ev_per_c": zero_binding,
        "correlation_binding_ev_per_c": correlation_binding,
        "sos_total_binding_ev_per_c": total_binding,
        "delta_st_reference_ev_per_c": delta_reference,
        "difference_from_delta_ev_per_c": difference,
        "absolute_difference_from_delta_ev_per_c": abs(difference),
        "binding_tolerance_ev_per_c": tolerance,
        "binding_gate": "pass" if abs(difference) < tolerance else "fail",
    }


def _atomic_write(path, text):
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-summary", required=True, type=Path)
    parser.add_argument("--solid-summary", required=True, type=Path)
    parser.add_argument("--delta-reference-ev-per-c", type=float, default=6.902326)
    parser.add_argument("--tolerance-ev-per-c", type=float, default=0.1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = collect_binding(
        atom_summary=args.atom_summary,
        solid_summary=args.solid_summary,
        delta_reference_ev_per_c=args.delta_reference_ev_per_c,
        tolerance_ev_per_c=args.tolerance_ev_per_c,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(args.output, encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
