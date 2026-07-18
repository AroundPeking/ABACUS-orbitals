import numbers

import torch


def _coefficient_matrix(c, element, l, invalid_value):
    try:
        coefficient = c[element][l]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"invalid freeze orbital {invalid_value!r}") from exc
    if not isinstance(coefficient, torch.Tensor) or coefficient.ndim != 2:
        raise ValueError(f"invalid freeze orbital {invalid_value!r}")
    return coefficient


def _is_integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def validate_freeze_orbitals(specs, c):
    if not isinstance(specs, (list, tuple)):
        raise TypeError(f"invalid freeze_orbitals value {specs!r}")

    indices = set()
    expected_keys = {"element", "l", "zeta"}
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != expected_keys:
            raise ValueError(f"invalid freeze orbital specification {spec!r}")

        element = spec["element"]
        l = spec["l"]
        zeta = spec["zeta"]
        invalid_value = (element, l, zeta)
        if not isinstance(element, str) or not element:
            raise ValueError(f"invalid freeze orbital {invalid_value!r}")
        if not _is_integer(l) or not _is_integer(zeta) or element not in c:
            raise ValueError(f"invalid freeze orbital {invalid_value!r}")
        if l < 0 or l >= len(c[element]):
            raise ValueError(f"invalid freeze orbital {invalid_value!r}")

        coefficient = _coefficient_matrix(c, element, l, invalid_value)
        if zeta < 1 or zeta > coefficient.shape[1]:
            raise ValueError(f"invalid freeze orbital {invalid_value!r}")

        index = (element, int(l), int(zeta) - 1)
        if index in indices:
            raise ValueError(f"duplicate freeze orbital {invalid_value!r}")
        indices.add(index)

    return frozenset(indices)


def zero_frozen_gradients(c, indices):
    validated = []
    try:
        iterator = iter(indices)
    except TypeError as exc:
        raise TypeError(f"invalid frozen orbital indices {indices!r}") from exc

    for index in iterator:
        if (
            not isinstance(index, tuple)
            or len(index) != 3
            or not _is_integer(index[1])
            or not _is_integer(index[2])
        ):
            raise ValueError(f"invalid frozen orbital index {index!r}")
        element, l, zeta = index
        if l < 0:
            raise ValueError(f"invalid frozen orbital index {index!r}")
        coefficient = _coefficient_matrix(c, element, l, index)
        if zeta < 0 or zeta >= coefficient.shape[1]:
            raise ValueError(f"invalid frozen orbital index {index!r}")
        if coefficient.grad is None:
            raise ValueError(f"gradient is missing for frozen orbital {index!r}")
        if coefficient.grad.shape != coefficient.shape:
            raise ValueError(f"invalid gradient for frozen orbital {index!r}")
        validated.append((coefficient, zeta))

    with torch.no_grad():
        for coefficient, zeta in validated:
            coefficient.grad[:, zeta].zero_()
