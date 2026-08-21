#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .gate_contract import (
        HA_TO_KCAL_MOL,
        PhaseResult,
        audit_phase,
        compare_zero_field_results,
    )
else:
    from gate_contract import (
        HA_TO_KCAL_MOL,
        PhaseResult,
        audit_phase,
        compare_zero_field_results,
    )


AUTHORITATIVE_RESULT = "RESULT_SUMMARY.json"
TEXT_SUMMARY = "RESULT_SUMMARY.txt"
RESTART_CHAIN_STATUS = "PENDING_TASK4"
RESTART_CHAIN_NOTE = (
    "Task2 verifies INPUT restart semantics only; actual WFC/CHG copy and load "
    "provenance must be verified by the Task4 runner before production use."
)


@dataclass(frozen=True)
class PhaseSpec:
    relative: str
    mode: str
    restart: bool
    field_dir: int | None = None


PHASES = (
    PhaseSpec("runs/fixed/fixed_cold", "fixed", False),
    PhaseSpec("runs/fixed/fixed_restart", "fixed", True),
    PhaseSpec("runs/dir0/field_seed", "field", False, 0),
    PhaseSpec("runs/dir0/free_restart1", "free", True),
    PhaseSpec("runs/dir0/free_restart2", "free", True),
    PhaseSpec("runs/dir1/field_seed", "field", False, 1),
    PhaseSpec("runs/dir1/free_restart1", "free", True),
    PhaseSpec("runs/dir1/free_restart2", "free", True),
    PhaseSpec("runs/dir2/field_seed", "field", False, 2),
    PhaseSpec("runs/dir2/free_restart1", "free", True),
    PhaseSpec("runs/dir2/free_restart2", "free", True),
)


def _phase_dict(phase: PhaseResult, root: Path) -> dict[str, object]:
    return {
        "path": str(Path(phase.path).relative_to(root)),
        "expected_mode": phase.expected_mode,
        "expected_restart": phase.expected_restart,
        "expected_field_dir": phase.expected_field_dir,
        "energy_ev": phase.energy_ev,
        "energy_ha": phase.energy_ha,
        "spin_counts": {str(key): value for key, value in phase.spin_counts.items()},
        "occupations": {
            str(key): list(values) for key, values in phase.occupations.items()
        },
        "integer_occupations": phase.integer_occupations,
        "file_sha256": dict(phase.file_hashes),
        "stage_sha256": phase.stage_hash,
    }


def audit_gate(root: str | Path) -> dict[str, object]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"gate root does not exist: {root_path}")

    phases = {
        spec.relative: audit_phase(
            root_path / spec.relative,
            expected_mode=spec.mode,
            expected_restart=spec.restart,
            expected_field_dir=spec.field_dir,
        )
        for spec in PHASES
    }
    fixed_cold = phases["runs/fixed/fixed_cold"]
    fixed_restart = phases["runs/fixed/fixed_restart"]
    fixed_drift = abs(fixed_restart.energy_ha - fixed_cold.energy_ha) * HA_TO_KCAL_MOL
    free_energies = {
        direction: phases[f"runs/dir{direction}/free_restart2"].energy_ha
        for direction in range(3)
    }
    free_drifts = {
        direction: abs(
            phases[f"runs/dir{direction}/free_restart2"].energy_ha
            - phases[f"runs/dir{direction}/free_restart1"].energy_ha
        )
        * HA_TO_KCAL_MOL
        for direction in range(3)
    }
    comparison = compare_zero_field_results(
        fixed_energy_ha=fixed_restart.energy_ha,
        free_energies_ha=free_energies,
        fixed_drift_kcal=fixed_drift,
        free_drifts_kcal=free_drifts,
    )
    return {
        "status": "DIAGNOSTIC_ONLY",
        "zero_field_comparison_status": comparison["status"],
        "authoritative_result": AUTHORITATIVE_RESULT,
        "restart_chain_evidence": {
            "status": RESTART_CHAIN_STATUS,
            "note": RESTART_CHAIN_NOTE,
        },
        "blocked_on": "restart_chain_evidence",
        "phases": {
            spec.relative: _phase_dict(phases[spec.relative], root_path)
            for spec in PHASES
        },
        "comparison": comparison,
    }


def _summary_text(summary: dict[str, object]) -> str:
    lines = [
        f"status={summary['status']}",
        f"authoritative_result={AUTHORITATIVE_RESULT}",
    ]
    if summary["status"] == "PBE_GATE_FAILED":
        lines.append(f"error={summary.get('error', 'unknown audit failure')}")
        return "\n".join(lines) + "\n"
    if summary["status"] != "DIAGNOSTIC_ONLY":
        raise ValueError(f"unsupported audit status: {summary['status']}")

    lines.append(
        "zero_field_comparison_status="
        f"{summary['zero_field_comparison_status']}"
    )
    lines.append(f"restart_chain_evidence={RESTART_CHAIN_STATUS}")
    lines.append(f"blocked_on={summary['blocked_on']}")
    lines.append(f"restart_chain_note={RESTART_CHAIN_NOTE}")
    phases = summary["phases"]
    for spec in PHASES:
        phase = phases[spec.relative]
        counts = phase["spin_counts"]
        occupations = phase["occupations"]
        spin1_occupations = ",".join(
            f"{value:.16g}" for value in occupations["1"]
        )
        spin2_occupations = ",".join(
            f"{value:.16g}" for value in occupations["2"]
        )
        lines.append(
            f"phase={spec.relative} energy_ev={phase['energy_ev']:.16g} "
            f"energy_ha={phase['energy_ha']:.16g} spin1={counts['1']:.1f} "
            f"spin2={counts['2']:.1f} integer_occupations="
            f"{str(phase['integer_occupations']).lower()} "
            f"spin1_occupations={spin1_occupations} "
            f"spin2_occupations={spin2_occupations} "
            f"INPUT_sha256={phase['file_sha256']['INPUT']} "
            f"running_scf.log_sha256="
            f"{phase['file_sha256']['running_scf.log']} "
            f"eig_occ.txt_sha256={phase['file_sha256']['eig_occ.txt']} "
            f"stage_sha256={phase['stage_sha256']}"
        )

    comparison = summary["comparison"]
    lines.append(f"fixed_drift_kcal={comparison['fixed_drift_kcal']:.16g}")
    for direction in range(3):
        lines.append(
            f"free_direction_{direction}_drift_kcal="
            f"{comparison['free_drifts_kcal'][direction]:.16g}"
        )
        lines.append(
            f"fixed_free_direction_{direction}_difference_ha="
            f"{comparison['fixed_free_differences_ha'][direction]:.16g}"
        )
    for pair, difference in comparison["free_pair_differences_ha"].items():
        lines.append(f"free_pair_{pair}_difference_ha={difference:.16g}")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def _summary_paths(root: Path) -> tuple[Path, Path]:
    return root / AUTHORITATIVE_RESULT, root / TEXT_SUMMARY


def _best_effort_remove_summaries(root: Path) -> None:
    for path in _summary_paths(root):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def invalidate_summaries(root: Path) -> None:
    errors = []
    for path in _summary_paths(root):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(exc)
    if errors:
        raise OSError(f"cannot invalidate old audit summaries: {errors[0]}")


def write_summaries(root: Path, summary: dict[str, object]) -> None:
    authoritative_path, text_path = _summary_paths(root)
    try:
        json_content = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        text_content = _summary_text(summary)
        _atomic_write(text_path, text_content)
        # JSON is the sole authority and is therefore published last.
        _atomic_write(authoritative_path, json_content)
    except Exception:
        # Cleanup is best-effort; the original exception is always re-raised.
        _best_effort_remove_summaries(root)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the C atom PBE reference gate")
    parser.add_argument("root_positional", nargs="?", help="gate root directory")
    parser.add_argument("--root", dest="root_option", help="gate root directory")
    args = parser.parse_args(argv)
    if args.root_positional is not None and args.root_option is not None:
        parser.error("positional root and --root cannot be used together")
    root_argument = args.root_option or args.root_positional or "."
    root = Path(root_argument).resolve()

    try:
        if not root.is_dir():
            raise ValueError(f"gate root does not exist: {root}")
        invalidate_summaries(root)
        summary = audit_gate(root)
    except (ValueError, OSError) as exc:
        failure = {
            "status": "PBE_GATE_FAILED",
            "authoritative_result": AUTHORITATIVE_RESULT,
            "error": str(exc),
        }
        if root.is_dir():
            _best_effort_remove_summaries(root)
            try:
                write_summaries(root, failure)
            except OSError:
                _best_effort_remove_summaries(root)
        print(f"PBE gate audit failed: {exc}", file=sys.stderr)
        return 1

    try:
        write_summaries(root, summary)
    except OSError as exc:
        _best_effort_remove_summaries(root)
        print(f"PBE gate summary write failed: {exc}", file=sys.stderr)
        return 1
    print(f"status={summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
