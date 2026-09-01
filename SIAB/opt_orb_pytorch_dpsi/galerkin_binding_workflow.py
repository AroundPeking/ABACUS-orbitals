"""Pure stage decisions for Galerkin-generated binding-basis candidates."""

import math
import re


_STAGE_ORDER = (
    "galerkin_screen",
    "pbe_gate",
    "tail_gate",
    "proxy_gate",
    "full_q_gate",
)


def _finite_positive(value, name):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("workflow config must be a dictionary")
    fingerprint = config.get("candidate_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise ValueError("candidate fingerprint must be a lowercase SHA256")
    try:
        stages = tuple(config["stages"])
    except (KeyError, TypeError) as error:
        raise ValueError("workflow stages must be a sequence") from error
    if (
        not stages
        or len(stages) != len(set(stages))
        or any(stage not in _STAGE_ORDER for stage in stages)
        or tuple(sorted(stages, key=_STAGE_ORDER.index)) != stages
        or stages[-1] != "full_q_gate"
    ):
        raise ValueError("workflow stages must be unique, ordered, and end at full_q_gate")
    reference = config.get("reference_binding_ev")
    if (
        not isinstance(reference, (int, float))
        or isinstance(reference, bool)
        or not math.isfinite(reference)
    ):
        raise ValueError("reference binding must be finite")
    tolerance = _finite_positive(
        config.get("acceptance_tolerance_ev"),
        "acceptance tolerance",
    )
    return fingerprint, stages, float(reference), tolerance


def _existing_action(stage, fingerprint, existing_actions):
    action = existing_actions.get(stage)
    if action is None:
        return None
    if not isinstance(action, dict):
        raise ValueError("existing action must be a dictionary")
    if action.get("candidate_fingerprint") != fingerprint:
        return None
    status = action.get("status")
    if status not in ("pending", "running", "completed"):
        raise ValueError("existing action status must be pending, running, or completed")
    job_id = str(action.get("job_id", "")).strip()
    if not job_id:
        raise ValueError("existing action job id must be nonempty")
    return {
        "next_action": "collect" if status == "completed" else "wait",
        "existing_job_id": job_id,
        "existing_job_status": status,
    }


def assess_workflow(config, evidence, *, existing_actions=None):
    """Return one monotone workflow decision without submitting any work."""
    fingerprint, stages, reference, tolerance = _validate_config(config)
    if not isinstance(evidence, dict):
        raise ValueError("workflow evidence must be a dictionary")
    if any(stage not in stages for stage in evidence):
        raise ValueError("workflow evidence contains an undeclared stage")
    if existing_actions is None:
        existing_actions = {}
    if not isinstance(existing_actions, dict):
        raise ValueError("existing_actions must be a dictionary")

    for stage in stages:
        record = evidence.get(stage)
        if record is None:
            existing = _existing_action(stage, fingerprint, existing_actions)
            result = {
                "status": "success",
                "decision": "continue",
                "workflow_state": stage,
                "next_action": "submit",
                "candidate_fingerprint": fingerprint,
            }
            if existing is not None:
                result.update(existing)
            return result
        if not isinstance(record, dict):
            raise ValueError("stage evidence must be a dictionary")
        if record.get("candidate_fingerprint") != fingerprint:
            raise ValueError("stage evidence candidate fingerprint does not match")
        status = record.get("status")
        gate = record.get("gate")
        if status not in ("success", "failed") or gate not in ("pass", "fail"):
            raise ValueError("stage evidence requires success/failed status and pass/fail gate")
        if status != "success" or gate != "pass":
            return {
                "status": "success",
                "decision": "rejected",
                "workflow_state": "rejected",
                "failed_stage": stage,
                "failure_reasons": list(record.get("failure_reasons", ())),
                "next_action": None,
                "candidate_fingerprint": fingerprint,
            }

    final = evidence["full_q_gate"]
    binding = final.get("binding_energy_ev")
    if (
        not isinstance(binding, (int, float))
        or isinstance(binding, bool)
        or not math.isfinite(binding)
    ):
        raise ValueError("full_q_gate requires a finite binding energy")
    binding = float(binding)
    error = abs(binding - reference)
    accepted = error < tolerance
    return {
        "status": "success",
        "decision": "accepted" if accepted else "rejected",
        "workflow_state": "accepted" if accepted else "rejected",
        "failed_stage": None if accepted else "full_q_gate",
        "next_action": None,
        "candidate_fingerprint": fingerprint,
        "binding_energy_ev": binding,
        "reference_binding_ev": reference,
        "binding_error_ev": error,
        "acceptance_tolerance_ev": tolerance,
    }
