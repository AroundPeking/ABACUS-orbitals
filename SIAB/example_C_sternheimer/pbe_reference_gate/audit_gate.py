#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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


PHASES = (
    ("runs/fixed/fixed_cold", "fixed"),
    ("runs/fixed/fixed_restart", "fixed"),
    ("runs/dir0/free_restart1", "free"),
    ("runs/dir0/free_restart2", "free"),
    ("runs/dir1/free_restart1", "free"),
    ("runs/dir1/free_restart2", "free"),
    ("runs/dir2/free_restart1", "free"),
    ("runs/dir2/free_restart2", "free"),
)


def _phase_dict(phase: PhaseResult, root: Path) -> dict[str, object]:
    return {
        "path": str(Path(phase.path).relative_to(root)),
        "expected_mode": phase.expected_mode,
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
        relative: audit_phase(root_path / relative, expected_mode=mode)
        for relative, mode in PHASES
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
        "status": comparison["status"],
        "phases": {
            relative: _phase_dict(phases[relative], root_path)
            for relative, _ in PHASES
        },
        "comparison": comparison,
    }


def _summary_text(summary: dict[str, object]) -> str:
    lines = [f"status={summary['status']}"]
    if summary["status"] != "PBE_GATE_PASSED":
        lines.append(f"error={summary.get('error', 'unknown audit failure')}")
        return "\n".join(lines) + "\n"

    phases = summary["phases"]
    for relative, _ in PHASES:
        phase = phases[relative]
        counts = phase["spin_counts"]
        occupations = phase["occupations"]
        spin1_occupations = ",".join(
            f"{value:.16g}" for value in occupations["1"]
        )
        spin2_occupations = ",".join(
            f"{value:.16g}" for value in occupations["2"]
        )
        lines.append(
            f"phase={relative} energy_ev={phase['energy_ev']:.16g} "
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


def write_summaries(root: Path, summary: dict[str, object]) -> None:
    _atomic_write(
        root / "RESULT_SUMMARY.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(root / "RESULT_SUMMARY.txt", _summary_text(summary))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the C atom PBE reference gate")
    parser.add_argument("root", nargs="?", default=".", help="gate root directory")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    try:
        summary = audit_gate(root)
    except Exception as exc:
        failure = {"status": "PBE_GATE_FAILED", "error": str(exc)}
        if root.is_dir():
            write_summaries(root, failure)
        print(f"PBE gate audit failed: {exc}", file=sys.stderr)
        return 1

    write_summaries(root, summary)
    print(f"status={summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
