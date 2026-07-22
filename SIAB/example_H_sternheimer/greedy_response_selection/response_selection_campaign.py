"""I/O contracts shared by the physical response-shell selection campaign."""

import math
from pathlib import Path

import torch

from IO.read_sternheimer import read_sternheimer
from response_selection import ResponseTargetFamily
from sternheimer_targets import (
    apply_target_element_aliases,
    parse_target_entries,
)


_FAMILY_ROLES = {
    "atom": "physical",
    "multicenter": "physical",
    "fragment_ghost": "ghost",
}


def _validate_nu(expected_nu, max_l):
    try:
        values = tuple(expected_nu)
    except TypeError as exc:
        raise TypeError("expected_nu must be a sequence") from exc
    if len(values) != max_l + 1:
        raise ValueError("expected_nu must contain one count for every l")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("expected_nu counts must be nonnegative integers")
    return values


def _next_nonempty(lines, index):
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise ValueError("coefficient file ended unexpectedly")
    return lines[index].strip(), index + 1


def read_optimizer_coefficients(
    path,
    *,
    element,
    radial_rows,
    max_l,
    expected_nu,
):
    """Read one element from SIAB's native ORBITAL_RESULTS coefficient block."""
    if not isinstance(element, str) or not element:
        raise ValueError("element must be nonempty")
    if type(radial_rows) is not int or radial_rows <= 0:
        raise ValueError("radial_rows must be a positive integer")
    if type(max_l) is not int or max_l < 0:
        raise ValueError("max_l must be a nonnegative integer")
    expected_nu = _validate_nu(expected_nu, max_l)

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        index = next(
            position + 1
            for position, line in enumerate(lines)
            if line.strip() == "<Coefficient>"
        )
    except StopIteration as exc:
        raise ValueError("missing <Coefficient> section") from exc

    declared_total = None
    columns = {}
    closed = False
    while index < len(lines):
        line, index = _next_nonempty(lines, index)
        if line == "</Coefficient>":
            closed = True
            break
        if "Total number of radial orbitals" in line:
            fields = line.split()
            try:
                declared_total = int(fields[0])
            except (IndexError, ValueError) as exc:
                raise ValueError("invalid declared coefficient count") from exc
            continue
        if not line.startswith("Type"):
            raise ValueError(f"unexpected coefficient row: {line}")

        label, index = _next_nonempty(lines, index)
        fields = label.split()
        if len(fields) != 3:
            raise ValueError(f"invalid coefficient label: {label}")
        label_element = fields[0]
        try:
            l = int(fields[1])
            zeta = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"invalid coefficient label: {label}") from exc
        if label_element != element:
            raise ValueError(
                f"unexpected coefficient element {label_element!r}; expected {element!r}"
            )
        if l < 0 or l > max_l or zeta <= 0 or zeta > expected_nu[l]:
            raise ValueError(f"coefficient column {(element, l, zeta)!r} is unexpected")
        key = (l, zeta)
        if key in columns:
            raise ValueError(f"duplicate coefficient column {(element, l, zeta)!r}")

        values = []
        while len(values) < radial_rows:
            value_line, index = _next_nonempty(lines, index)
            if value_line == "</Coefficient>" or value_line.startswith("Type"):
                raise ValueError(f"coefficient column {(element, l, zeta)!r} is incomplete")
            try:
                values.extend(float(value) for value in value_line.split())
            except ValueError as exc:
                raise ValueError(
                    f"coefficient column {(element, l, zeta)!r} is not numeric"
                ) from exc
        if len(values) != radial_rows or any(not math.isfinite(value) for value in values):
            raise ValueError(f"coefficient column {(element, l, zeta)!r} is invalid")
        columns[key] = values

    if not closed:
        raise ValueError("missing </Coefficient> section")
    expected_total = sum(expected_nu)
    if declared_total is not None and declared_total != expected_total:
        raise ValueError("declared coefficient count does not match expected_nu")

    by_l = []
    for l, count in enumerate(expected_nu):
        missing = [zeta for zeta in range(1, count + 1) if (l, zeta) not in columns]
        if missing:
            raise ValueError(f"missing coefficient column {(element, l, missing[0])!r}")
        if count:
            by_l.append(
                torch.tensor(
                    [columns[(l, zeta)] for zeta in range(1, count + 1)],
                    dtype=torch.float64,
                ).transpose(0, 1).contiguous()
            )
        else:
            by_l.append(torch.empty((radial_rows, 0), dtype=torch.float64))
    return {element: by_l}


def write_optimizer_coefficients(path, coefficients):
    """Write coefficients in the native format consumed by SIAB main.py."""
    if not isinstance(coefficients, dict) or not coefficients:
        raise TypeError("coefficients must be a nonempty dictionary")
    total = 0
    validated = []
    for element in sorted(coefficients):
        if not isinstance(element, str) or not element:
            raise ValueError("coefficient element must be nonempty")
        for l, channel in enumerate(coefficients[element]):
            if (
                not isinstance(channel, torch.Tensor)
                or channel.ndim != 2
                or channel.dtype != torch.float64
                or channel.is_complex()
                or channel.device.type != "cpu"
                or not bool(torch.all(torch.isfinite(channel)))
            ):
                raise ValueError("optimizer coefficients must be finite CPU float64 matrices")
            for column in range(channel.shape[1]):
                validated.append((element, l, column + 1, channel[:, column]))
                total += 1

    output = ["<Coefficient>", f"\t {total} Total number of radial orbitals."]
    for element, l, zeta, column in validated:
        output.extend(
            (
                "\tType\tL\tZeta-Orbital",
                f"\t  {element} \t{l}\t    {zeta}",
            )
        )
        output.extend(f"\t {float(value):18.14f}" for value in column)
    output.extend(
        (
            "</Coefficient>",
            "<Mkb>",
            "Left spillage = 0.0000000000e+00",
            "</Mkb>",
            "",
        )
    )
    Path(path).write_text("\n".join(output), encoding="utf-8")


def assemble_response_families(loaded_entries):
    """Build the frozen atom, multicenter, and fragment/ghost family tuple."""
    loaded_entries = tuple(loaded_entries)
    families = {}
    for entry, data in loaded_entries:
        expected_role = _FAMILY_ROLES.get(entry.family)
        if expected_role is None or entry.role != expected_role:
            raise ValueError(
                "response selection requires exactly atom, multicenter, and "
                "fragment_ghost with their frozen roles"
            )
        if entry.family in families:
            raise ValueError(f"duplicate response target family {entry.family!r}")
        families[entry.family] = data
    if set(families) != set(_FAMILY_ROLES):
        raise ValueError(
            "response selection requires exactly atom, multicenter, and "
            "fragment_ghost with their frozen roles"
        )
    return (
        ResponseTargetFamily("atom", (families["atom"],), "physical"),
        ResponseTargetFamily(
            "multicenter", (families["multicenter"],), "physical"
        ),
        ResponseTargetFamily(
            "fragment_ghost",
            (families["fragment_ghost"],),
            "ghost",
            real_atom_index=0,
        ),
    )


def load_response_families(targets):
    """Read, alias, and assemble the three physical campaign targets once."""
    entries = parse_target_entries(targets)
    loaded = tuple(
        (
            entry,
            apply_target_element_aliases(read_sternheimer(entry.path), entry),
        )
        for entry in entries
    )
    return assemble_response_families(loaded)
