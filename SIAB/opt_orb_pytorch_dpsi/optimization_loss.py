import math
import numbers

import torch


LOSS_DEFAULTS = {
    "epsilon": 1e-14,
    "condition_limit": 1e12,
    "tau_dft": 0.05,
    "tau_dpsi": 0.10,
    "constraint_penalty_dft": 10.0,
    "constraint_penalty_dpsi": 10.0,
    "joint_dpsi_weight": 1.0,
}

_LOSS_MODES = frozenset({"st_only", "st_constrained", "st_dpsi_joint"})


def _validate_mode(mode):
    if not isinstance(mode, str) or mode not in _LOSS_MODES:
        raise ValueError(
            f"invalid loss mode {mode!r}; expected st_only, st_constrained, "
            "or st_dpsi_joint"
        )


def _validate_real(name, value, minimum, strict=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if strict and value <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    if not strict and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def normalize_loss_config(config):
    if not isinstance(config, dict):
        raise TypeError("loss config must be a dictionary")
    allowed = set(LOSS_DEFAULTS) | {"mode"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown loss config keys: {sorted(unknown)!r}")
    if "mode" not in config:
        raise ValueError("loss config requires mode")

    _validate_mode(config["mode"])
    normalized = dict(LOSS_DEFAULTS)
    normalized.update(config)
    _validate_real("epsilon", normalized["epsilon"], 0.0, strict=True)
    _validate_real("condition_limit", normalized["condition_limit"], 1.0)
    _validate_real("tau_dft", normalized["tau_dft"], 0.0)
    _validate_real("tau_dpsi", normalized["tau_dpsi"], 0.0)
    _validate_real(
        "constraint_penalty_dft", normalized["constraint_penalty_dft"], 0.0
    )
    _validate_real(
        "constraint_penalty_dpsi", normalized["constraint_penalty_dpsi"], 0.0
    )
    _validate_real("joint_dpsi_weight", normalized["joint_dpsi_weight"], 0.0)
    return normalized


def _validate_loss_tensor(name, value):
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise TypeError(f"{name} must be a scalar tensor")
    if value.is_complex() or value.dtype == torch.bool:
        raise TypeError(f"{name} must be a real scalar tensor")
    if not bool(torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if bool(value < 0):
        raise ValueError(f"{name} must be nonnegative")


def _baseline_tensor(name, baseline, reference):
    if not isinstance(baseline, dict) or name not in baseline:
        raise ValueError(f"baseline requires finite nonnegative {name}")
    value = baseline[name]
    if isinstance(value, torch.Tensor):
        if (
            value.ndim != 0
            or value.is_complex()
            or value.dtype == torch.bool
            or not bool(torch.isfinite(value))
            or bool(value < 0)
        ):
            raise ValueError(f"baseline {name} must be finite and nonnegative")
        return value.to(dtype=reference.dtype, device=reference.device)
    _validate_real(f"baseline {name}", value, 0.0)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _ratios(dft, dpsi, baseline, config):
    baseline_dft = _baseline_tensor("dft_origin", baseline, dft)
    baseline_dpsi = _baseline_tensor("dft_dpsi", baseline, dpsi)
    dft_denominator = torch.clamp(baseline_dft, min=config["epsilon"])
    dpsi_denominator = torch.clamp(baseline_dpsi, min=config["epsilon"])
    return dft / dft_denominator, dpsi / dpsi_denominator


def compose_loss(mode, st, dft, dpsi, baseline, config):
    _validate_mode(mode)
    normalized = normalize_loss_config(config)
    if normalized["mode"] != mode:
        raise ValueError(
            f"loss mode {mode!r} does not match config mode {normalized['mode']!r}"
        )
    _validate_loss_tensor("sternheimer", st)
    _validate_loss_tensor("dft_origin", dft)
    _validate_loss_tensor("dft_dpsi", dpsi)
    dft_ratio, dpsi_ratio = _ratios(dft, dpsi, baseline, normalized)

    regularization_dpsi = torch.zeros_like(dpsi)
    if mode == "st_only":
        constraint_dft = torch.zeros_like(dft)
        constraint_dpsi = torch.zeros_like(dpsi)
        total = st
    else:
        constraint_dft = normalized["constraint_penalty_dft"] * torch.relu(
            dft_ratio - 1.0 - normalized["tau_dft"]
        ).square()
        constraint_dpsi = normalized["constraint_penalty_dpsi"] * torch.relu(
            dpsi_ratio - 1.0 - normalized["tau_dpsi"]
        ).square()
        if mode == "st_dpsi_joint":
            regularization_dpsi = normalized["joint_dpsi_weight"] * dpsi_ratio
        total = (
            st + regularization_dpsi + constraint_dft + constraint_dpsi
        )

    return {
        "dft_origin": dft,
        "dft_dpsi": dpsi,
        "sternheimer": st,
        "regularization_dpsi": regularization_dpsi,
        "constraint_dft": constraint_dft,
        "constraint_dpsi": constraint_dpsi,
        "total": total,
    }


def selection_component(mode):
    _validate_mode(mode)
    return "total" if mode == "st_dpsi_joint" else "sternheimer"


def constraints_satisfied(dft, dpsi, baseline, config):
    normalized = normalize_loss_config(config)
    _validate_loss_tensor("dft_origin", dft)
    _validate_loss_tensor("dft_dpsi", dpsi)
    dft_ratio, dpsi_ratio = _ratios(dft, dpsi, baseline, normalized)
    return bool(
        (dft_ratio <= 1.0 + normalized["tau_dft"])
        & (dpsi_ratio <= 1.0 + normalized["tau_dpsi"])
    )
