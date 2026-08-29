from dataclasses import dataclass
import math

import torch

from sternheimer_data import SternheimerData
from sternheimer_source_data import SternheimerSourceData


_PHYSICAL_PROVENANCE_KEYS = (
    "auxiliary_basis_sha256",
    "cell_bohr",
    "ecut_ry",
    "kernel",
    "orbital_sha256",
    "pseudopotential_sha256",
    "spin_convention",
    "exx_pca_thr",
    "auxiliary_whitening",
    "raw_auxiliary_dimension",
    "whitened_auxiliary_rank",
    "discarded_auxiliary_rank",
    "coulomb_relative_threshold",
    "coulomb_transform_sha256",
)
_EXECUTION_PROVENANCE_KEYS = (
    "abacus_commit",
    "executable_sha256",
    "mpi_ranks",
    "omp_threads",
)
_OVERLAP_RELATIVE_TOLERANCE = 1.0e-13
_OVERLAP_ABSOLUTE_TOLERANCE = 1.0e-14
_OCCUPATION_ABSOLUTE_TOLERANCE = 1.0e-14
_AUXILIARY_SPACE_IDENTITY_KEYS = (
    "auxiliary_whitening",
    "raw_auxiliary_dimension",
    "whitened_auxiliary_rank",
    "discarded_auxiliary_rank",
    "coulomb_relative_threshold",
    "coulomb_transform_sha256",
)


@dataclass(frozen=True)
class SternheimerResponseSourcePair:
    response: SternheimerData
    source: SternheimerSourceData
    source_row_for_response_key: dict
    provenance_warnings: tuple


def pair_response_and_source(response, source):
    if not isinstance(response, SternheimerData):
        raise ValueError("response must be SternheimerData")
    if not isinstance(source, SternheimerSourceData):
        raise ValueError("source must be SternheimerSourceData")
    if response.format_version != source.format_version:
        raise ValueError("response and source format versions differ")
    if response.grid_volume_bohr3 != source.grid_volume_bohr3:
        raise ValueError("response and source grid_volume_bohr3 differ")
    if response.blocks != source.blocks:
        raise ValueError("response and source primitive blocks differ")

    _validate_overlap(response.overlap, source.overlap)
    warnings = _compare_provenance(response.provenance, source.provenance)

    whitened_rank = _positive_integer_provenance(
        response.provenance,
        "whitened_auxiliary_rank",
    )
    response_channels = sorted(set(response.auxiliary_channel.tolist()))
    expected_channels = list(range(whitened_rank))
    if response_channels != expected_channels:
        raise ValueError(
            "response channel IDs must be contiguous from zero and equal "
            f"whitened_auxiliary_rank={whitened_rank}; got {response_channels}"
        )

    response_keys = set(
        zip(
            response.occupied_state.tolist(),
            response.auxiliary_channel.tolist(),
        )
    )
    source_keys = list(
        zip(
            source.occupied_state.tolist(),
            source.auxiliary_channel.tolist(),
        )
    )
    source_key_set = set(source_keys)
    if response_keys != source_key_set:
        missing = sorted(response_keys - source_key_set)
        extra = sorted(source_key_set - response_keys)
        raise ValueError(
            f"response and source keys differ: missing={missing}, extra={extra}"
        )

    source_channels = sorted(set(source.auxiliary_channel.tolist()))
    if source_channels != expected_channels:
        raise ValueError(
            "source channel IDs must be contiguous from zero and equal "
            f"whitened_auxiliary_rank={whitened_rank}; got {source_channels}"
        )

    source_row_for_key = {
        key: source_row for source_row, key in enumerate(source_keys)
    }
    for response_row, key in enumerate(
        zip(
            response.occupied_state.tolist(),
            response.auxiliary_channel.tolist(),
        )
    ):
        source_row = source_row_for_key[key]
        difference = abs(
            float(response.occupation[response_row])
            - float(source.occupation[source_row])
        )
        if difference > _OCCUPATION_ABSOLUTE_TOLERANCE:
            raise ValueError(
                f"occupation differs for source key {key}: "
                f"absolute difference {difference:.17g} exceeds "
                f"{_OCCUPATION_ABSOLUTE_TOLERANCE:.1e}"
            )

    return SternheimerResponseSourcePair(
        response=response,
        source=source,
        source_row_for_response_key=source_row_for_key,
        provenance_warnings=warnings,
    )


def _validate_overlap(response_overlap, source_overlap):
    if response_overlap.shape != source_overlap.shape:
        raise ValueError(
            "response and source overlap shapes differ: "
            f"{tuple(response_overlap.shape)} != {tuple(source_overlap.shape)}"
        )
    difference = response_overlap - source_overlap
    if difference.numel() == 0:
        maximum_absolute = 0.0
        maximum_scale = 0.0
    else:
        maximum_absolute = float(torch.max(torch.abs(difference)))
        maximum_scale = max(
            float(torch.max(torch.abs(response_overlap))),
            float(torch.max(torch.abs(source_overlap))),
        )
    maximum_allowed = (
        _OVERLAP_ABSOLUTE_TOLERANCE
        + _OVERLAP_RELATIVE_TOLERANCE * maximum_scale
    )
    if maximum_absolute > maximum_allowed:
        raise ValueError(
            "response/source overlap maximum absolute difference "
            f"{maximum_absolute:.17g} exceeds "
            f"scale-aware tolerance {maximum_allowed:.17g}"
        )

    difference_norm = float(torch.linalg.vector_norm(difference))
    scale = max(
        float(torch.linalg.vector_norm(response_overlap)),
        float(torch.linalg.vector_norm(source_overlap)),
    )
    relative = difference_norm / scale if scale > 0.0 else 0.0
    if relative > _OVERLAP_RELATIVE_TOLERANCE:
        raise ValueError(
            "response/source overlap relative Frobenius difference "
            f"{relative:.17g} exceeds "
            f"{_OVERLAP_RELATIVE_TOLERANCE:.1e}"
        )


def _compare_provenance(response, source):
    warnings = []
    for key in _PHYSICAL_PROVENANCE_KEYS:
        missing_from = []
        if key not in response:
            missing_from.append("response")
        if key not in source:
            missing_from.append("source")
        if missing_from:
            raise ValueError(
                f"missing physical provenance key {key} from "
                + " and ".join(missing_from)
            )
        if not _values_equal(response[key], source[key]):
            if key != "auxiliary_basis_sha256" or not _same_auxiliary_space(
                response,
                source,
            ):
                raise ValueError(f"physical provenance differs: {key}")
            warnings.append(
                "physical provenance definition differs: "
                "auxiliary_basis_sha256; accepted because the complete "
                "whitened Coulomb space identity is equal"
            )

    for key in _EXECUTION_PROVENANCE_KEYS:
        response_value = response.get(key)
        source_value = source.get(key)
        if not _values_equal(response_value, source_value):
            warnings.append(
                f"execution provenance differs: {key}: "
                f"response={response_value!r}, source={source_value!r}"
            )

    ignored = set(_PHYSICAL_PROVENANCE_KEYS) | set(
        _EXECUTION_PROVENANCE_KEYS
    )
    for key in sorted((set(response) | set(source)) - ignored):
        if key not in response or key not in source:
            raise ValueError(f"provenance key set differs: {key}")
        if not _values_equal(response[key], source[key]):
            raise ValueError(f"provenance differs: {key}")
    return tuple(warnings)


def _same_auxiliary_space(response, source):
    for key in _AUXILIARY_SPACE_IDENTITY_KEYS:
        if key not in response or key not in source:
            return False
        if not _values_equal(response[key], source[key]):
            return False
    return True


def _positive_integer_provenance(provenance, key):
    value = provenance[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"physical provenance {key} must be a positive integer")
    return value


def _values_equal(left, right):
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right, (list, tuple)
        ):
            return False
        return len(left) == len(right) and all(
            _values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return set(left) == set(right) and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, float) or isinstance(right, float):
        try:
            return (
                math.isfinite(float(left))
                and math.isfinite(float(right))
                and left == right
            )
        except (TypeError, ValueError):
            return False
    return left == right
