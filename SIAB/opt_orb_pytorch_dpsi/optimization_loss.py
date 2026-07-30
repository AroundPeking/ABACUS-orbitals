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
    "radial_tail_weight": 0.0,
    "radial_tail_radius": 0.0,
    "radial_tail_condition_limit": 1.0e10,
    "low_frequency_guard_weight": 0.0,
    "low_frequency_guard_tolerance": 0.0,
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
    _validate_real("radial_tail_weight", normalized["radial_tail_weight"], 0.0)
    _validate_real("radial_tail_radius", normalized["radial_tail_radius"], 0.0)
    _validate_real(
        "radial_tail_condition_limit",
        normalized["radial_tail_condition_limit"],
        1.0,
    )
    _validate_real(
        "low_frequency_guard_weight",
        normalized["low_frequency_guard_weight"],
        0.0,
    )
    _validate_real(
        "low_frequency_guard_tolerance",
        normalized["low_frequency_guard_tolerance"],
        0.0,
    )
    if (
        normalized["radial_tail_weight"] > 0.0
        and normalized["radial_tail_radius"] <= 0.0
    ):
        raise ValueError(
            "radial_tail_radius must be positive when radial_tail_weight is positive"
        )
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


def compose_loss(
    mode,
    st,
    dft,
    dpsi,
    baseline,
    config,
    radial_tail=None,
    *,
    st_low_frequency=None,
):
    _validate_mode(mode)
    normalized = normalize_loss_config(config)
    if normalized["mode"] != mode:
        raise ValueError(
            f"loss mode {mode!r} does not match config mode {normalized['mode']!r}"
        )
    _validate_loss_tensor("sternheimer", st)
    _validate_loss_tensor("dft_origin", dft)
    _validate_loss_tensor("dft_dpsi", dpsi)
    if radial_tail is None:
        if normalized["radial_tail_weight"] > 0.0:
            raise ValueError(
                "radial_tail is required when radial_tail_weight is positive"
            )
        radial_tail = torch.zeros_like(st)
    else:
        _validate_loss_tensor("radial_tail", radial_tail)
    dft_ratio, dpsi_ratio = _ratios(dft, dpsi, baseline, normalized)

    regularization_dpsi = torch.zeros_like(dpsi)
    regularization_locality = (
        normalized["radial_tail_weight"] * radial_tail
    )
    guard_active = normalized["low_frequency_guard_weight"] > 0.0
    if guard_active:
        if st_low_frequency is None:
            raise ValueError(
                "st_low_frequency is required when the low-frequency guard "
                "is active"
            )
        _validate_loss_tensor(
            "sternheimer_lowest_frequency", st_low_frequency
        )
        baseline_low_frequency = _baseline_tensor(
            "sternheimer_lowest_frequency", baseline, st_low_frequency
        )
        if not bool(baseline_low_frequency > normalized["epsilon"]):
            raise ValueError(
                "baseline sternheimer_lowest_frequency must exceed epsilon "
                "when the low-frequency guard is active"
            )
        low_frequency_excess = torch.relu(
            st_low_frequency / baseline_low_frequency
            - 1.0
            - normalized["low_frequency_guard_tolerance"]
        )
        regularization_low_frequency = (
            normalized["low_frequency_guard_weight"]
            * low_frequency_excess.square()
        )
    else:
        regularization_low_frequency = torch.zeros_like(st)
    if mode == "st_only":
        constraint_dft = torch.zeros_like(dft)
        constraint_dpsi = torch.zeros_like(dpsi)
        total = (
            st
            if normalized["radial_tail_weight"] == 0.0 and not guard_active
            else st
            + regularization_locality
            + regularization_low_frequency
        )
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
            st
            + regularization_dpsi
            + regularization_locality
            + regularization_low_frequency
            + constraint_dft
            + constraint_dpsi
        )

    components = {
        "dft_origin": dft,
        "dft_dpsi": dpsi,
        "sternheimer": st,
        "regularization_dpsi": regularization_dpsi,
        "constraint_dft": constraint_dft,
        "constraint_dpsi": constraint_dpsi,
        "radial_tail": radial_tail,
        "regularization_locality": regularization_locality,
    }
    if guard_active:
        components["sternheimer_lowest_frequency"] = st_low_frequency
        components["regularization_low_frequency"] = (
            regularization_low_frequency
        )
    components["total"] = total
    return components


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


def low_frequency_guard_satisfied(current, baseline, config):
    normalized = normalize_loss_config(config)
    if normalized["low_frequency_guard_weight"] == 0.0:
        return True
    _validate_loss_tensor("sternheimer_lowest_frequency", current)
    reference = _baseline_tensor(
        "sternheimer_lowest_frequency", baseline, current
    )
    if not bool(reference > normalized["epsilon"]):
        raise ValueError(
            "baseline sternheimer_lowest_frequency must exceed epsilon "
            "when the low-frequency guard is active"
        )
    limit = (1.0 + normalized["low_frequency_guard_tolerance"]) * reference
    allowance = 1.0e-12 * torch.maximum(limit.abs(), torch.ones_like(limit))
    return bool(current <= limit + allowance)
