from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path


_REQUIRED_FIELDS = frozenset({"path", "family", "role"})
_OPTIONAL_FIELDS = frozenset(
    {"element_aliases", "source_path", "zero_order_audit_path"}
)
_FORBIDDEN_ENERGY_FIELDS = frozenset(
    {"rpa_binding", "h2_energy", "delta_st_energy"}
)
_TARGET_ROLES = frozenset({"physical", "ghost"})


@dataclass(frozen=True)
class SternheimerTargetEntry:
    path: Path
    family: str
    role: str
    element_aliases: tuple = ()
    source_path: Path | None = None
    zero_order_audit_path: Path | None = None


def _nonempty_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"target {field} must be a nonempty string")
    return value


def _parse_element_aliases(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("target element_aliases must be a nonempty object")
    aliases = {}
    for source, target in value.items():
        source = _nonempty_string(source, "element_aliases source")
        target = _nonempty_string(target, "element_aliases target")
        if source == target:
            raise ValueError("target element_aliases cannot map an element to itself")
        aliases[source] = target
    if set(aliases) & set(aliases.values()):
        raise ValueError("target element_aliases cannot contain chains or cycles")
    return tuple(sorted(aliases.items()))


def apply_target_element_aliases(data, entry):
    if not isinstance(entry, SternheimerTargetEntry):
        raise TypeError("entry must be a SternheimerTargetEntry")
    if not entry.element_aliases:
        return data

    aliases = dict(entry.element_aliases)
    present = {block.element for block in data.blocks}
    missing = sorted(set(aliases) - present)
    if missing:
        raise ValueError(f"target alias source is absent: {missing[0]}")

    blocks = tuple(
        replace(block, element=aliases.get(block.element, block.element))
        for block in data.blocks
    )
    keys = [block.key for block in blocks]
    if len(keys) != len(set(keys)):
        raise ValueError("target element_aliases create duplicate PrimitiveBlock keys")
    return replace(data, blocks=blocks)


def parse_target_entries(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("sternheimer targets must be a nonempty list")

    entries = []
    seen = set()
    for value in values:
        if isinstance(value, str):
            entry = SternheimerTargetEntry(
                Path(_nonempty_string(value, "path")),
                "default",
                "physical",
            )
        else:
            if not isinstance(value, dict):
                raise ValueError(
                    "sternheimer target entries must be paths or objects"
                )
            if set(value) & _FORBIDDEN_ENERGY_FIELDS:
                raise ValueError("RPA energy is not a selector input")
            fields = set(value)
            if not _REQUIRED_FIELDS <= fields or fields - (
                _REQUIRED_FIELDS | _OPTIONAL_FIELDS
            ):
                raise ValueError(
                    "named target requires path, family, and role; "
                    "only element_aliases, source_path, and "
                    "zero_order_audit_path are optional"
                )
            path = _nonempty_string(value["path"], "path")
            family = _nonempty_string(value["family"], "family")
            role = _nonempty_string(value["role"], "role")
            if role not in _TARGET_ROLES:
                raise ValueError("target role must be physical or ghost")
            aliases = (
                _parse_element_aliases(value["element_aliases"])
                if "element_aliases" in value
                else ()
            )
            source_path = (
                Path(_nonempty_string(value["source_path"], "source_path"))
                if "source_path" in value
                else None
            )
            zero_order_audit_path = (
                Path(
                    _nonempty_string(
                        value["zero_order_audit_path"],
                        "zero_order_audit_path",
                    )
                )
                if "zero_order_audit_path" in value
                else None
            )
            if role == "ghost" and (
                source_path is not None or zero_order_audit_path is not None
            ):
                raise ValueError(
                    "ghost target cannot carry source or zero-order audit paths"
                )
            entry = SternheimerTargetEntry(
                path=Path(path),
                family=family,
                role=role,
                element_aliases=aliases,
                source_path=source_path,
                zero_order_audit_path=zero_order_audit_path,
            )

        key = (
            entry.path,
            entry.family,
            entry.role,
            entry.element_aliases,
            entry.source_path,
            entry.zero_order_audit_path,
        )
        if key in seen:
            raise ValueError(f"duplicate Sternheimer target {key!r}")
        seen.add(key)
        entries.append(entry)

    return tuple(entries)
