from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


VALID_MODES = {"fixed", "fixed_field", "field", "free"}
FROZEN_PROTOCOL = (
    ("suffix", "C_PBE_REFERENCE_GATE"),
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
)
_INTEGER_PROTOCOL_KEYS = frozenset(
    {
        "ntype",
        "nelec",
        "nspin",
        "nupdown",
        "nbands",
        "nx",
        "ny",
        "nz",
        "symmetry",
        "gamma_only",
        "kpar",
        "scf_nmax",
        "out_chg",
        "out_wfc_lcao",
        "out_app_flag",
        "out_mul",
    }
)
_FLOAT_PROTOCOL_KEYS = frozenset(
    {"ecutwfc", "lcao_ecut", "scf_thr", "mixing_beta", "mixing_beta_mag"}
)
_FROZEN_PROTOCOL_KEYS = frozenset(key for key, _ in FROZEN_PROTOCOL)
_BASE_MODE_KEYS = frozenset({"ocp", "efield_flag", "efield_amp"})
_FIELD_ONLY_KEYS = frozenset(
    {"dip_cor_flag", "efield_dir", "efield_pos_max", "efield_pos_dec"}
)
_RESTART_ONLY_KEYS = frozenset({"init_wfc", "init_chg"})
HA_TO_EV = 27.211386245988
HA_TO_KCAL_MOL = 627.5094740631
INTEGER_TOL = 1e-10
DRIFT_TOL_KCAL = 0.001
ENERGY_TOL_HA = 1e-5

_CONVERGENCE_MARKER = "#SCF IS CONVERGED#"
_FINAL_ENERGY_RE = re.compile(
    r"^\s*!FINAL_ETOT_IS\s+(\S+)\s+eV\s*$", re.MULTILINE
)
_IONIC_STEP_RE = re.compile(r"^\s*\d+\s+#\s*ionic step\s*$", re.MULTILINE)
_SPIN_NUMBER_RE = re.compile(r"^\s*Spin number\s+(\d+)\s*$", re.MULTILINE)
_SPIN_HEADER_RE = re.compile(
    r"^\s*spin=(\d+)\s+k-point=(\d+)/(\d+)\b.*$", re.MULTILINE
)
_OCCUPATION_ROW_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s*$")


@dataclass(frozen=True)
class PhaseResult:
    path: str
    expected_mode: str
    expected_restart: bool
    expected_field_dir: int | None
    energy_ev: float
    energy_ha: float
    spin_counts: Mapping[int, float]
    occupations: Mapping[int, tuple[float, ...]]
    integer_occupations: bool
    input_values: Mapping[str, str]
    file_hashes: Mapping[str, str]
    stage_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "spin_counts", MappingProxyType(dict(self.spin_counts))
        )
        object.__setattr__(
            self,
            "occupations",
            MappingProxyType(
                {spin: tuple(values) for spin, values in self.occupations.items()}
            ),
        )
        object.__setattr__(
            self, "input_values", MappingProxyType(dict(self.input_values))
        )
        object.__setattr__(
            self, "file_hashes", MappingProxyType(dict(self.file_hashes))
        )


def render_input(
    *, mode: str, field_dir: int | None = None, restart: bool = False
) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if mode == "fixed" and field_dir is not None:
        raise ValueError("fixed mode does not accept field_dir")
    if mode in {"fixed_field", "field", "free"} and (
        type(field_dir) is not int or field_dir not in {0, 1, 2}
    ):
        raise ValueError("field_dir must be an integer 0, 1, or 2")
    if mode in {"fixed_field", "field"} and restart:
        raise ValueError(f"{mode} mode requires restart=False")
    if mode == "free" and not restart:
        raise ValueError("free mode requires restart=True")

    values = [
        ("INPUT_PARAMETERS", None),
        *FROZEN_PROTOCOL,
        ("ocp", "1" if mode in {"fixed", "fixed_field"} else "0"),
    ]
    if mode in {"fixed", "fixed_field"}:
        values.append(("ocp_set", "3*1 19*0 1*1 21*0"))
    if mode in {"fixed_field", "field"}:
        values.extend(
            [
                ("efield_flag", "1"),
                ("dip_cor_flag", "0"),
                ("efield_dir", str(field_dir)),
                ("efield_pos_max", "0.8"),
                ("efield_pos_dec", "0.1"),
                ("efield_amp", "1e-4"),
            ]
        )
    else:
        values.extend([("efield_flag", "0"), ("efield_amp", "0")])
    if restart:
        values.extend([("init_wfc", "file"), ("init_chg", "file")])

    return "\n".join(
        key if value is None else f"{key} {value}" for key, value in values
    ) + "\n"


def _parse_input(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read INPUT: {exc}") from exc

    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty or nonempty[0] != "INPUT_PARAMETERS":
        raise ValueError("INPUT must start with INPUT_PARAMETERS")
    if nonempty.count("INPUT_PARAMETERS") != 1:
        raise ValueError("INPUT contains ambiguous INPUT_PARAMETERS headers")

    values: dict[str, str] = {}
    for line in nonempty[1:]:
        if line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"ambiguous INPUT line: {line}")
        key, value = fields
        if key in values:
            raise ValueError(f"duplicate INPUT key: {key}")
        values[key] = value
    return values


def _require_finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _require_integer_input(values: Mapping[str, str], key: str) -> int:
    if key not in values:
        raise ValueError(f"INPUT is missing {key}")
    token = values[key]
    if not re.fullmatch(r"[+-]?\d+", token):
        raise ValueError(f"INPUT {key} must be an integer")
    return int(token)


def _validate_frozen_protocol(values: Mapping[str, str]) -> None:
    for key, expected in FROZEN_PROTOCOL:
        if key not in values:
            raise ValueError(f"frozen protocol key {key} is missing")
        actual = values[key]
        if key in _INTEGER_PROTOCOL_KEYS:
            if not re.fullmatch(r"[+-]?\d+", actual) or int(actual) != int(expected):
                raise ValueError(
                    f"frozen protocol key {key} must equal {expected}; got {actual}"
                )
        elif key in _FLOAT_PROTOCOL_KEYS:
            actual_number = _require_finite_float(
                actual, f"frozen protocol key {key}"
            )
            if actual_number != float(expected):
                raise ValueError(
                    f"frozen protocol key {key} must equal {expected}; got {actual}"
                )
        elif actual != expected:
            raise ValueError(
                f"frozen protocol key {key} must equal {expected}; got {actual}"
            )


def _require_float_input(values: Mapping[str, str], key: str) -> float:
    if key not in values:
        raise ValueError(f"INPUT is missing {key}")
    return _require_finite_float(values[key], f"INPUT {key}")


def _validate_restart_input(
    values: Mapping[str, str], expected_restart: bool
) -> None:
    # Task2 checks only the declared INPUT semantics. Task4 must prove that
    # these files were copied from, and loaded from, the preceding phase.
    if type(expected_restart) is not bool:
        raise ValueError("expected_restart must be a boolean")
    restart_values = (values.get("init_wfc"), values.get("init_chg"))
    if expected_restart:
        if restart_values != ("file", "file"):
            raise ValueError(
                "restart input requires init_wfc=file and init_chg=file"
            )
    elif any(value is not None for value in restart_values):
        raise ValueError(
            "cold/field input must not contain init_wfc or init_chg"
        )


def _allowed_input_keys(
    expected_mode: str, expected_restart: bool
) -> frozenset[str]:
    allowed = set(_FROZEN_PROTOCOL_KEYS | _BASE_MODE_KEYS)
    if expected_mode in {"fixed", "fixed_field"}:
        allowed.add("ocp_set")
    if expected_mode in {"fixed_field", "field"}:
        allowed.update(_FIELD_ONLY_KEYS)
    if expected_restart:
        allowed.update(_RESTART_ONLY_KEYS)
    return frozenset(allowed)


def _validate_input_key_whitelist(
    values: Mapping[str, str], expected_mode: str, expected_restart: bool
) -> None:
    allowed = _allowed_input_keys(expected_mode, expected_restart)
    actual = set(values)
    missing = sorted(allowed - actual)
    unexpected = sorted(actual - allowed)
    if missing:
        raise ValueError("missing INPUT keys: " + ", ".join(missing))
    if unexpected:
        raise ValueError("unexpected INPUT keys: " + ", ".join(unexpected))


def _validate_phase_input(
    values: Mapping[str, str],
    expected_mode: str,
    expected_restart: bool,
    expected_field_dir: int | None,
) -> None:
    if expected_mode not in VALID_MODES:
        raise ValueError(f"unsupported expected_mode: {expected_mode}")
    _validate_frozen_protocol(values)

    if expected_mode in {"fixed_field", "field"}:
        if expected_restart:
            raise ValueError(f"{expected_mode} phase cannot use restart input")
        if (
            type(expected_field_dir) is not int
            or expected_field_dir not in {0, 1, 2}
        ):
            raise ValueError(
                f"{expected_mode} phase requires expected_field_dir 0, 1, or 2"
            )
    else:
        if expected_field_dir is not None:
            raise ValueError(
                "expected_field_dir is only valid for field-bearing phases"
            )
        if expected_mode == "free" and not expected_restart:
            raise ValueError("free phase requires expected_restart=True")
    _validate_restart_input(values, expected_restart)
    _validate_input_key_whitelist(values, expected_mode, expected_restart)

    ocp = _require_integer_input(values, "ocp")
    if expected_mode in {"fixed", "fixed_field"}:
        if ocp != 1:
            raise ValueError(f"{expected_mode} phase requires ocp=1")
        if values.get("ocp_set") != "3*1 19*0 1*1 21*0":
            raise ValueError(
                f"{expected_mode} phase has missing or unexpected ocp_set"
            )
    else:
        if ocp != 0:
            raise ValueError(f"{expected_mode} phase requires ocp=0")
        if "ocp_set" in values:
            raise ValueError(f"{expected_mode} phase must not contain ocp_set")

    if expected_mode in {"fixed_field", "field"}:
        field_contract = {
            "efield_flag": 1,
            "dip_cor_flag": 0,
            "efield_dir": expected_field_dir,
        }
        for key, expected in field_contract.items():
            if _require_integer_input(values, key) != expected:
                raise ValueError(
                    f"{expected_mode} phase requires {key}={expected}"
                )
        float_contract = {
            "efield_pos_max": 0.8,
            "efield_pos_dec": 0.1,
            "efield_amp": 1e-4,
        }
        for key, expected in float_contract.items():
            if _require_float_input(values, key) != expected:
                raise ValueError(
                    f"{expected_mode} phase requires {key}={expected:.16g}"
                )
        return

    if _require_integer_input(values, "efield_flag") != 0:
        raise ValueError(
            "accepted fixed/free phase violates the zero-field contract"
        )
    if _require_float_input(values, "efield_amp") != 0.0:
        raise ValueError(
            "accepted fixed/free phase violates the zero-field contract"
        )
    forbidden_field_keys = {
        "dip_cor_flag",
        "efield_dir",
        "efield_pos_max",
        "efield_pos_dec",
    }
    present = sorted(forbidden_field_keys.intersection(values))
    if present:
        raise ValueError(
            "fixed/free zero-field input contains field-only keys: "
            + ", ".join(present)
        )


def _parse_final_energy(path: Path) -> float:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ValueError(f"cannot read running_scf.log: {exc}") from exc

    if text.count(_CONVERGENCE_MARKER) != 1:
        raise ValueError(
            "running_scf.log must contain exactly one ABACUS SCF convergence marker"
        )
    marker_lines = re.findall(
        r"^\s*!FINAL_ETOT_IS\b.*$", text, flags=re.MULTILINE
    )
    matches = _FINAL_ENERGY_RE.findall(text)
    if len(marker_lines) != 1 or len(matches) != 1:
        raise ValueError(
            "running_scf.log must contain exactly one final total energy"
        )
    return _require_finite_float(matches[0], "final total energy")


def _parse_occupations(path: Path) -> dict[int, tuple[float, ...]]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ValueError(f"cannot read eig_occ.txt: {exc}") from exc

    if len(_IONIC_STEP_RE.findall(text)) != 1:
        raise ValueError("eig_occ.txt must contain exactly one ionic step")
    spin_numbers = _SPIN_NUMBER_RE.findall(text)
    if spin_numbers != ["2"]:
        raise ValueError("eig_occ.txt must contain exactly one Spin number 2 line")

    headers = list(_SPIN_HEADER_RE.finditer(text))
    if len(headers) != 2 or {int(match.group(1)) for match in headers} != {1, 2}:
        raise ValueError(
            "eig_occ.txt must contain one explicit spin=1 and spin=2 block"
        )
    if any((int(match.group(2)), int(match.group(3))) != (1, 1) for match in headers):
        raise ValueError("eig_occ.txt must contain exactly one k-point per spin")

    occupations: dict[int, tuple[float, ...]] = {}
    for header_index, header in enumerate(headers):
        spin = int(header.group(1))
        end = headers[header_index + 1].start() if header_index + 1 < len(headers) else len(text)
        block = text[header.end():end]
        indices: list[int] = []
        values: list[float] = []
        for line in block.splitlines():
            if not line.strip():
                continue
            match = _OCCUPATION_ROW_RE.fullmatch(line)
            if match is None:
                raise ValueError(f"ambiguous occupation row in spin={spin}: {line.strip()}")
            index = int(match.group(1))
            _require_finite_float(match.group(2), f"spin={spin} eigenvalue")
            occupation = _require_finite_float(
                match.group(3), f"spin={spin} occupation"
            )
            indices.append(index)
            values.append(occupation)
        if not values or indices != list(range(1, len(values) + 1)):
            raise ValueError(f"spin={spin} occupation rows are missing or ambiguous")
        occupations[spin] = tuple(values)

    if len(occupations[1]) != len(occupations[2]):
        raise ValueError("spin occupation blocks have inconsistent band counts")
    return occupations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash {path.name}: {exc}") from exc
    return digest.hexdigest()


def _resolve_output_files(phase: Path, suffix: str) -> tuple[Path, Path]:
    root_outputs = [
        phase / name
        for name in ("running_scf.log", "eig_occ.txt")
        if (phase / name).exists()
    ]
    if root_outputs:
        names = ", ".join(path.name for path in root_outputs)
        raise ValueError(f"root-level output files are not accepted: {names}")

    expected_directory = phase / f"OUT.{suffix}"
    if expected_directory.is_symlink():
        raise ValueError(
            f"expected output directory {expected_directory.name} is a symlink"
        )
    if not expected_directory.is_dir():
        raise ValueError(
            f"expected output directory {expected_directory.name} is missing"
        )

    stale_directories = sorted(
        path.name
        for path in phase.glob("OUT.*")
        if path.is_dir() and path != expected_directory
    )
    if stale_directories:
        raise ValueError(
            "ambiguous/stale output directories are present: "
            + ", ".join(stale_directories)
        )

    log_path = expected_directory / "running_scf.log"
    eig_path = expected_directory / "eig_occ.txt"
    for path in (log_path, eig_path):
        if path.is_symlink():
            raise ValueError(f"{path.name} is a symlink, not local output evidence")
        if not path.is_file():
            raise ValueError(
                f"missing {path.name} in expected output directory "
                f"{expected_directory.name}"
            )
    return log_path, eig_path


def audit_phase(
    path: str | Path,
    expected_mode: str,
    expected_restart: bool,
    expected_field_dir: int | None = None,
) -> PhaseResult:
    phase_argument = Path(path)
    if phase_argument.is_symlink():
        raise ValueError("phase directory must not be a symlink")
    phase = phase_argument.resolve()
    if not phase.is_dir():
        raise ValueError(f"phase directory does not exist: {phase}")

    input_path = phase / "INPUT"
    if input_path.is_symlink():
        raise ValueError("INPUT is a symlink, not local input evidence")
    if not input_path.is_file():
        raise ValueError(f"missing INPUT in phase {phase}")

    input_values = _parse_input(input_path)
    _validate_phase_input(
        input_values,
        expected_mode,
        expected_restart,
        expected_field_dir,
    )
    log_path, eig_path = _resolve_output_files(
        phase, input_values["suffix"]
    )
    energy_ev = _parse_final_energy(log_path)
    occupations = _parse_occupations(eig_path)

    expected_band_count = int(input_values["nbands"])
    for spin, spin_values in occupations.items():
        if len(spin_values) != expected_band_count:
            raise ValueError(
                f"spin={spin} band count must equal nbands={expected_band_count}; "
                f"got {len(spin_values)}"
            )

    snapped: dict[int, tuple[int, ...]] = {}
    for spin, spin_values in occupations.items():
        snapped_values = []
        for occupation in spin_values:
            if abs(occupation) <= INTEGER_TOL:
                snapped_values.append(0)
            elif abs(occupation - 1.0) <= INTEGER_TOL:
                snapped_values.append(1)
            else:
                raise ValueError(
                    f"fractional occupation in spin={spin}: {occupation:.16g}"
                )
        snapped[spin] = tuple(snapped_values)

    spin_counts = {spin: float(sum(values)) for spin, values in snapped.items()}
    if spin_counts != {1: 3.0, 2: 1.0}:
        raise ValueError(
            "spin electron counts must be spin1=3 and spin2=1; "
            f"got spin1={spin_counts.get(1)} spin2={spin_counts.get(2)}"
        )

    files = {
        "INPUT": input_path,
        "running_scf.log": log_path,
        "eig_occ.txt": eig_path,
    }
    file_hashes = {name: _sha256(file_path) for name, file_path in files.items()}
    manifest = {
        name: {
            "relative_path": str(file_path.relative_to(phase)),
            "sha256": file_hashes[name],
        }
        for name, file_path in files.items()
    }
    stage_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return PhaseResult(
        path=str(phase),
        expected_mode=expected_mode,
        expected_restart=expected_restart,
        expected_field_dir=expected_field_dir,
        energy_ev=energy_ev,
        energy_ha=energy_ev / HA_TO_EV,
        spin_counts=spin_counts,
        occupations=occupations,
        integer_occupations=True,
        input_values=input_values,
        file_hashes=file_hashes,
        stage_hash=stage_hash,
    )


def _validate_directions(values: Mapping[int, object], label: str) -> None:
    if any(type(direction) is not int for direction in values) or set(values) != {
        0,
        1,
        2,
    }:
        raise ValueError(f"{label} must contain exactly directions {{0, 1, 2}}")


def compare_zero_field_results(
    *,
    fixed_energy_ha: float,
    free_energies_ha: Mapping[int, float],
    fixed_drift_kcal: float,
    free_drifts_kcal: Mapping[int, float],
) -> dict[str, object]:
    fixed_energy = _require_finite_float(fixed_energy_ha, "fixed energy")
    fixed_drift = _require_finite_float(fixed_drift_kcal, "fixed drift")
    _validate_directions(free_energies_ha, "free energies")
    _validate_directions(free_drifts_kcal, "free drifts")

    free_energies = {
        direction: _require_finite_float(
            free_energies_ha[direction], f"free direction {direction} energy"
        )
        for direction in range(3)
    }
    free_drifts = {
        direction: _require_finite_float(
            free_drifts_kcal[direction], f"free direction {direction} drift"
        )
        for direction in range(3)
    }

    if fixed_drift >= DRIFT_TOL_KCAL:
        raise ValueError(
            f"fixed seed-to-zero-restart drift {fixed_drift:.16g} kcal/mol is not "
            f"below {DRIFT_TOL_KCAL}"
        )
    for direction, drift in free_drifts.items():
        if drift >= DRIFT_TOL_KCAL:
            raise ValueError(
                f"free direction {direction} drift {drift:.16g} kcal/mol is not "
                f"below {DRIFT_TOL_KCAL}"
            )

    fixed_free_differences = {
        direction: abs(energy - fixed_energy)
        for direction, energy in free_energies.items()
    }
    for direction, difference in fixed_free_differences.items():
        if difference >= ENERGY_TOL_HA:
            raise ValueError(
                f"fixed/free energy difference for direction {direction} is "
                f"{difference:.16g} Ha, not below {ENERGY_TOL_HA}"
            )

    free_pair_differences = {
        f"{left}-{right}": abs(free_energies[left] - free_energies[right])
        for left in range(3)
        for right in range(left + 1, 3)
    }
    for pair, difference in free_pair_differences.items():
        if difference >= ENERGY_TOL_HA:
            raise ValueError(
                f"free-direction energy difference for {pair} is "
                f"{difference:.16g} Ha, not below {ENERGY_TOL_HA}"
            )

    return {
        "status": "ZERO_FIELD_COMPARISON_PASSED",
        "thresholds": {
            "drift_kcal_mol": DRIFT_TOL_KCAL,
            "energy_ha": ENERGY_TOL_HA,
            "integer_occupation": INTEGER_TOL,
        },
        "fixed_energy_ha": fixed_energy,
        "free_energies_ha": free_energies,
        "fixed_drift_kcal": fixed_drift,
        "free_drifts_kcal": free_drifts,
        "fixed_free_differences_ha": fixed_free_differences,
        "free_pair_differences_ha": free_pair_differences,
    }
