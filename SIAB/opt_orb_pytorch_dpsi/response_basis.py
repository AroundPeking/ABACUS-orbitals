import torch


def _validate_coefficients(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64 or value.is_complex():
        raise ValueError(f"{name} must be real float64")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.ndim != 2:
        raise ValueError(f"{name} must have rank 2")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")


def canonicalize_columns(coefficients):
    """Fix eigenvector signs by making each largest entry positive."""
    _validate_coefficients(coefficients, "coefficients")
    result = coefficients.detach().clone()
    for column in range(result.shape[1]):
        values = result[:, column]
        pivot = int(torch.argmax(torch.abs(values)).item())
        if float(values[pivot].item()) < 0.0:
            values.mul_(-1.0)
    return result


def replace_channel_coefficients(coefficients, element, l, values):
    """Replace one SIAB radial channel without touching other channels."""
    try:
        current = coefficients[element][l]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"missing coefficient channel {element}/{l}") from exc
    _validate_coefficients(current, f"coefficients[{element!r}][{l}]")
    _validate_coefficients(values, "values")
    if values.shape != current.shape:
        raise ValueError(
            f"values shape {tuple(values.shape)} does not match channel shape "
            f"{tuple(current.shape)}"
        )
    coefficients[element][l] = values.detach().clone().requires_grad_(True)
