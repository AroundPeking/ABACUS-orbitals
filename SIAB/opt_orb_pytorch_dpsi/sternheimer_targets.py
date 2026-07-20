from dataclasses import dataclass
from pathlib import Path


_REQUIRED_FIELDS = frozenset({"path", "family", "role"})
_FORBIDDEN_ENERGY_FIELDS = frozenset(
    {"rpa_binding", "h2_energy", "delta_st_energy"}
)
_TARGET_ROLES = frozenset({"physical", "ghost"})


@dataclass(frozen=True)
class SternheimerTargetEntry:
    path: Path
    family: str
    role: str


def _nonempty_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"target {field} must be a nonempty string")
    return value


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
            if set(value) != _REQUIRED_FIELDS:
                raise ValueError("named target requires path, family, and role")
            path = _nonempty_string(value["path"], "path")
            family = _nonempty_string(value["family"], "family")
            role = _nonempty_string(value["role"], "role")
            if role not in _TARGET_ROLES:
                raise ValueError("target role must be physical or ghost")
            entry = SternheimerTargetEntry(Path(path), family, role)

        key = (entry.path, entry.family, entry.role)
        if key in seen:
            raise ValueError(f"duplicate Sternheimer target {key!r}")
        seen.add(key)
        entries.append(entry)

    return tuple(entries)
