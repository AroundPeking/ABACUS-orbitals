#!/usr/bin/env python3
"""Freeze deterministic SIAB response-shell sequences without energy feedback."""

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from response_selection import (
    CandidateEvaluation,
    append_response_shell,
    borrowing_gap,
    evaluate_response_candidates,
    normalized_family_loss,
    select_best_candidate,
)
from sternheimer_targets import parse_target_entries


_OPTIMIZER_ARTIFACTS = (
    "ORBITAL_RESULTS.txt",
    "ORBITAL_1U.dat",
    "Spillage.dat",
)


@dataclass(frozen=True)
class FrozenSelectionStep:
    selected: CandidateEvaluation
    candidates: tuple
    coefficients: dict

    def __post_init__(self):
        candidates = tuple(self.candidates)
        object.__setattr__(self, "candidates", candidates)
        if not isinstance(self.selected, CandidateEvaluation):
            raise TypeError("selected must be a CandidateEvaluation")
        if not self.selected.admissible:
            raise ValueError("selected candidate must be admissible")
        if self.selected not in candidates:
            raise ValueError("selected candidate must occur in candidates")
        if not isinstance(self.coefficients, dict) or not self.coefficients:
            raise TypeError("coefficients must be a nonempty dictionary")


@dataclass(frozen=True)
class SelectionMetrics:
    atom_loss: float
    multicenter_loss: float
    global_capture: float
    borrowing: float
    per_l_residual_ratio: dict


@dataclass(frozen=True)
class NestedSelectionResult:
    status: str
    steps: tuple
    metrics: SelectionMetrics


def _reject_energy_fields(value):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"h2_energy", "rpa_binding"}:
                raise ValueError("selector inputs and records cannot contain energy fields")
            _reject_energy_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_energy_fields(child)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(path, value):
    _reject_energy_fields(value)
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_channel(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError(f"{name} must be a rank-2 torch.Tensor")
    if value.dtype != torch.float64 or value.is_complex():
        raise ValueError(f"{name} must be real float64")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")


def _coefficient_payload(coefficients):
    if not isinstance(coefficients, dict) or not coefficients:
        raise TypeError("coefficients must be a nonempty dictionary")
    elements = {}
    for element in sorted(coefficients):
        channels = []
        for l, value in enumerate(coefficients[element]):
            _validate_channel(value, f"coefficients[{element!r}][{l}]")
            channels.append(
                {
                    "l": l,
                    "shape": list(value.shape),
                    "values": [
                        float(item).hex() for item in value.detach().reshape(-1)
                    ],
                }
            )
        elements[element] = channels
    return {"format_version": 1, "dtype": "float64", "elements": elements}


def _write_coefficients_json(path, coefficients):
    _canonical_json(path, _coefficient_payload(coefficients))


def load_coefficients_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or payload.get("dtype") != "float64":
        raise ValueError("unsupported coefficient JSON format")
    result = {}
    for element, channels in payload["elements"].items():
        by_l = []
        for expected_l, channel in enumerate(channels):
            if channel["l"] != expected_l:
                raise ValueError("coefficient channels must be ordered by l")
            shape = tuple(channel["shape"])
            if len(shape) != 2 or shape[0] < 0 or shape[1] < 0:
                raise ValueError("invalid coefficient channel shape")
            values = [float.fromhex(item) for item in channel["values"]]
            if len(values) != shape[0] * shape[1]:
                raise ValueError("coefficient value count does not match shape")
            by_l.append(torch.tensor(values, dtype=torch.float64).reshape(shape))
        result[element] = by_l
    return result


def _column_hex(coefficients, element, l, zeta):
    try:
        column = coefficients[element][l][:, zeta - 1]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"fixed orbital {(element, l, zeta)!r} is missing"
        ) from exc
    _validate_channel(column.reshape(-1, 1), "fixed orbital column")
    return tuple(float(value).hex() for value in column)


def _validate_fixed_columns(baseline, current, fixed_specs):
    for spec in fixed_specs:
        if set(spec) != {"element", "l", "zeta"}:
            raise ValueError("fixed orbital spec requires element, l, and zeta")
        key = (spec["element"], int(spec["l"]), int(spec["zeta"]))
        if _column_hex(baseline, *key) != _column_hex(current, *key):
            raise RuntimeError(f"fixed orbital {key!r} changed")


def _candidate_record(value):
    if not isinstance(value, CandidateEvaluation):
        raise TypeError("candidates must be CandidateEvaluation values")
    score = value.score if math.isfinite(value.score) else None
    return {
        "key": [value.gain.l, value.gain.mode],
        "l": value.gain.l,
        "mode": value.gain.mode,
        "cost": value.gain.cost,
        "gain_atom": value.gain.atom,
        "gain_multicenter": value.gain.multicenter,
        "gain_balance": value.gain.balance,
        "score": score,
        "admissible": value.admissible,
        "rejection_reason": value.rejection_reason,
    }


def _basis_counts(coefficients):
    nu = {}
    radial_shells = 0
    ao_functions = 0
    for element in sorted(coefficients):
        nu[element] = []
        for l, value in enumerate(coefficients[element]):
            _validate_channel(value, f"coefficients[{element!r}][{l}]")
            count = value.shape[1]
            nu[element].append(count)
            radial_shells += count
            ao_functions += (2 * l + 1) * count
    return nu, radial_shells, ao_functions


def freeze_selection_sequence(
    output_dir,
    config,
    fixed_dzp,
    fixed_specs,
    steps,
):
    _reject_energy_fields(config)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"selection output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_specs = tuple(copy.deepcopy(fixed_specs))
    steps = tuple(steps)
    if not steps:
        raise ValueError("frozen selection sequence must be nonempty")

    previous_nu, _, _ = _basis_counts(fixed_dzp)
    manifest_steps = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, FrozenSelectionStep):
            raise TypeError("steps must contain FrozenSelectionStep values")
        _validate_fixed_columns(fixed_dzp, step.coefficients, fixed_specs)
        nu, radial_shells, ao_functions = _basis_counts(step.coefficients)
        selected_l = step.selected.gain.l
        if set(nu) != set(previous_nu):
            raise ValueError("coefficient elements changed within the sequence")
        increments = []
        for element in nu:
            if len(nu[element]) != len(previous_nu[element]):
                raise ValueError("angular channel count changed within the sequence")
            increments.extend(
                (element, l)
                for l, (new, old) in enumerate(
                    zip(nu[element], previous_nu[element])
                )
                for _ in range(new - old)
            )
            if any(new < old for new, old in zip(nu[element], previous_nu[element])):
                raise ValueError("a frozen sequence cannot remove radial shells")
        if len(increments) != 1 or increments[0][1] != selected_l:
            raise ValueError("each selection step must append its one chosen shell")

        step_dir = output_dir / "selection_steps" / f"step_{index:03d}"
        step_dir.mkdir(parents=True)
        coefficients_path = step_dir / "coefficients.json"
        _write_coefficients_json(coefficients_path, step.coefficients)
        record = {
            "format_version": 1,
            "step": index,
            "selected": _candidate_record(step.selected),
            "candidates": [_candidate_record(value) for value in step.candidates],
            "nu": nu,
            "radial_shell_count": radial_shells,
            "ao_function_count": ao_functions,
            "coefficients": "coefficients.json",
            "coefficients_sha256": _sha256(coefficients_path),
        }
        _canonical_json(step_dir / "selection_record.json", record)
        relative_coefficients = coefficients_path.relative_to(output_dir)
        manifest_steps.append(
            {
                "step": index,
                "selected_key": list(step.selected.gain.key),
                "nu": nu,
                "radial_shell_count": radial_shells,
                "ao_function_count": ao_functions,
                "coefficients": str(relative_coefficients),
                "coefficients_sha256": record["coefficients_sha256"],
                "selection_record": str(
                    (step_dir / "selection_record.json").relative_to(output_dir)
                ),
            }
        )
        previous_nu = nu

    manifest = {
        "format_version": 1,
        "config": copy.deepcopy(config),
        "fixed_orbitals": list(fixed_specs),
        "steps": manifest_steps,
    }
    manifest_path = output_dir / "selection_manifest.json"
    _canonical_json(manifest_path, manifest)
    return manifest_path


def select_one_response_shell(
    current,
    fixed_dzp,
    spectra,
    atom_family,
    multicenter_family,
    ghost_family,
):
    spectra = tuple(spectra)
    evaluations = evaluate_response_candidates(
        spectra,
        current,
        fixed_dzp,
        atom_family,
        multicenter_family,
        ghost_family,
    )
    selected_gain = select_best_candidate(
        value.gain for value in evaluations if value.admissible
    )
    selected = next(
        value for value in evaluations if value.gain.key == selected_gain.key
    )
    spectrum_by_l = {value.l: value for value in spectra}
    coefficients = append_response_shell(
        current,
        spectrum_by_l[selected.gain.l],
        selected.gain.mode,
    )
    return FrozenSelectionStep(
        selected=selected,
        candidates=evaluations,
        coefficients=coefficients,
    )


def _spectrum_weights(spectra):
    result = {}
    for spectrum in spectra:
        if spectrum.l in result:
            raise ValueError("response spectra must contain one value per l")
        weight = float(torch.sum(spectrum.eigenvalues).item())
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("residual spectrum weight must be finite and nonnegative")
        result[spectrum.l] = weight
    if not result:
        raise ValueError("response spectra must be nonempty")
    return result


def _selection_metrics(
    current,
    fixed_dzp,
    spectra,
    baseline_weights,
    atom_family,
    multicenter_family,
    ghost_family,
):
    current_weights = _spectrum_weights(spectra)
    if set(current_weights) != set(baseline_weights):
        raise ValueError("response spectrum l channels changed within the sequence")
    ratios = {}
    for l, baseline in baseline_weights.items():
        current_weight = current_weights[l]
        if baseline > 0.0:
            ratios[l] = current_weight / baseline
        elif current_weight > 1.0e-14:
            raise RuntimeError("a zero-baseline l channel gained residual weight")

    atom_loss = normalized_family_loss(atom_family, current, fixed_dzp)
    multicenter_loss = normalized_family_loss(
        multicenter_family, current, fixed_dzp
    )
    return SelectionMetrics(
        atom_loss=atom_loss,
        multicenter_loss=multicenter_loss,
        global_capture=1.0 - 0.5 * (atom_loss + multicenter_loss),
        borrowing=borrowing_gap(ghost_family, current, fixed_dzp),
        per_l_residual_ratio=ratios,
    )


def _stopping_satisfied(metrics, baseline_borrowing, config):
    try:
        global_capture = float(config["global_capture"])
        per_l_limit = float(config["per_l_residual_limit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "selection config requires finite global_capture and "
            "per_l_residual_limit"
        ) from exc
    if (
        not math.isfinite(global_capture)
        or not 0.0 < global_capture <= 1.0
        or not math.isfinite(per_l_limit)
        or not 0.0 <= per_l_limit < 1.0
    ):
        raise ValueError("invalid response-selection stopping thresholds")
    borrowing_tolerance = 1.0e-12 * max(abs(baseline_borrowing), 1.0)
    return (
        metrics.global_capture >= global_capture
        and all(
            value <= per_l_limit
            for value in metrics.per_l_residual_ratio.values()
        )
        and metrics.borrowing <= baseline_borrowing + borrowing_tolerance
    )


def run_nested_selection(
    config,
    initial,
    fixed_dzp,
    fixed_specs,
    atom_family,
    multicenter_family,
    ghost_family,
    spectrum_builder,
    optimize_step,
    max_steps=64,
):
    if not callable(spectrum_builder):
        raise TypeError("spectrum_builder must be callable")
    if not callable(optimize_step):
        raise TypeError("optimize_step must be callable")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    fixed_specs = tuple(fixed_specs)
    _validate_fixed_columns(fixed_dzp, initial, fixed_specs)
    baseline_spectra = tuple(spectrum_builder(fixed_dzp))
    baseline_weights = _spectrum_weights(baseline_spectra)
    baseline_borrowing = borrowing_gap(ghost_family, fixed_dzp, fixed_dzp)

    current = initial
    steps = []
    while True:
        spectra = tuple(spectrum_builder(current))
        metrics = _selection_metrics(
            current,
            fixed_dzp,
            spectra,
            baseline_weights,
            atom_family,
            multicenter_family,
            ghost_family,
        )
        if _stopping_satisfied(metrics, baseline_borrowing, config):
            return NestedSelectionResult(
                status="converged",
                steps=tuple(steps),
                metrics=metrics,
            )
        if len(steps) >= max_steps:
            raise RuntimeError(
                "response-shell selection exceeded max_steps before convergence"
            )

        initialized = select_one_response_shell(
            current,
            fixed_dzp,
            spectra,
            atom_family,
            multicenter_family,
            ghost_family,
        )
        optimized = optimize_step(
            len(steps) + 1,
            initialized.coefficients,
            initialized.selected,
        )
        _validate_fixed_columns(fixed_dzp, optimized, fixed_specs)
        initialized_nu, _, _ = _basis_counts(initialized.coefficients)
        optimized_nu, _, _ = _basis_counts(optimized)
        if initialized_nu != optimized_nu:
            raise RuntimeError("SIAB optimization changed selected shell counts")
        step = FrozenSelectionStep(
            selected=initialized.selected,
            candidates=initialized.candidates,
            coefficients=optimized,
        )
        steps.append(step)
        current = optimized


def build_step_input(
    template,
    targets,
    initial_coefficients,
    nu,
    fixed_specs,
    seed,
):
    entries = parse_target_entries(targets)
    if not entries:
        raise ValueError("step input requires Sternheimer targets")
    if not any(entry.role == "physical" for entry in entries):
        raise ValueError("step input requires a physical Sternheimer target")
    result = copy.deepcopy(template)
    _reject_energy_fields(result)
    result.setdefault("file_list", {})["sternheimer"] = copy.deepcopy(targets)
    result["element"]["Nu"] = copy.deepcopy(nu)
    result["C_init_info"]["init_from_file"] = True
    result["C_init_info"]["C_init_file"] = str(initial_coefficients)
    result["freeze_orbitals"] = list(copy.deepcopy(fixed_specs))
    result["seed"] = int(seed)
    return result


def run_joint_optimizer(input_payload, output_dir, optimizer, python):
    _reject_energy_fields(input_payload)
    output_dir = Path(output_dir)
    optimizer = Path(optimizer).resolve()
    python = Path(python).resolve()
    for path in (optimizer, python):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"optimizer output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _canonical_json(output_dir / "INPUT", input_payload)

    environment = os.environ.copy()
    python_path = str(optimizer.parent)
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    environment.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(python), str(optimizer)],
            cwd=output_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"SIAB optimizer exited with {completed.returncode}; "
            f"see {output_dir / 'run.log'}"
        )

    result = {}
    for name in _OPTIMIZER_ARTIFACTS:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"SIAB optimizer did not produce {name}")
        result[name] = _sha256(path)
    return result
