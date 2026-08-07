"""Protocol-checked comparison of Bessel parent spaces with grid Delta-ST."""

from dataclasses import dataclass
import json
import math
import pathlib
import struct
from typing import Dict, Tuple

import torch

from fixed_ao_librpa_compare import read_coulomb
from primitive_galerkin import evaluate_primitive_galerkin
from sternheimer_fixed_ao_data import AuxiliaryChannel
from sternheimer_primitive_galerkin_data import SternheimerPrimitiveGalerkinData


_RESPONSE_MARKER = -41073291
_PROTOCOL_PROVENANCE_KEYS = (
    "abacus_commit",
    "auxiliary_basis_sha256",
    "cell_bohr",
    "ecut_ry",
    "kernel",
    "orbital_sha256",
    "pseudopotential_sha256",
    "spin_convention",
)
_HARTREE_TO_KCAL_MOL = 627.5094740631


@dataclass(frozen=True)
class DeltaSTReference:
    response_m: torch.Tensor
    frequency_ha: torch.Tensor
    frequency_weight_ha: torch.Tensor
    channels: Tuple[AuxiliaryChannel, ...]
    atom_naux: Tuple[int, ...]
    occupied_occupation_by_spin: Tuple[Tuple[float, ...], ...]
    provenance: Dict[str, object]

    def __post_init__(self):
        _require_tensor("response_m", self.response_m, torch.complex128, 3)
        _require_tensor("frequency_ha", self.frequency_ha, torch.float64, 1)
        _require_tensor(
            "frequency_weight_ha", self.frequency_weight_ha, torch.float64, 1
        )
        nfrequency = self.frequency_ha.shape[0]
        nauxiliary = len(self.channels)
        if self.response_m.shape != (nfrequency, nauxiliary, nauxiliary):
            raise ValueError(
                "response_m shape must be (n_frequency, n_auxiliary, n_auxiliary)"
            )
        if self.frequency_weight_ha.shape != self.frequency_ha.shape:
            raise ValueError("frequency weight shape must match frequency")
        if nfrequency == 0 or not bool(torch.all(self.frequency_ha > 0.0)):
            raise ValueError("frequency_ha must be nonempty and positive")
        if nfrequency > 1 and not bool(
            torch.all(self.frequency_ha[1:] > self.frequency_ha[:-1])
        ):
            raise ValueError("frequency_ha must be strictly increasing")
        if bool(torch.any(self.frequency_weight_ha < 0.0)):
            raise ValueError("frequency weights must be nonnegative")
        _require_hermitian("response_m", self.response_m)
        if not self.channels:
            raise ValueError("at least one auxiliary channel is required")
        if any(not isinstance(value, AuxiliaryChannel) for value in self.channels):
            raise ValueError("channels must contain AuxiliaryChannel values")
        if not self.atom_naux or any(
            type(value) is not int or value <= 0 for value in self.atom_naux
        ):
            raise ValueError("atom_naux must contain positive integers")
        if sum(self.atom_naux) != nauxiliary:
            raise ValueError("atom_naux must sum to the auxiliary dimension")
        _require_occupations(self.occupied_occupation_by_spin)
        _require_provenance(self.provenance)


@dataclass(frozen=True)
class FullCoulombMatrix:
    matrix: torch.Tensor
    atom_naux: Tuple[int, ...]
    provenance: Dict[str, object]

    def __post_init__(self):
        _require_tensor("Coulomb matrix", self.matrix, torch.complex128, 2)
        if self.matrix.shape[0] == 0 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Coulomb matrix must be nonempty and square")
        if not self.atom_naux or any(
            type(value) is not int or value <= 0 for value in self.atom_naux
        ):
            raise ValueError("Coulomb atom_naux must contain positive integers")
        if sum(self.atom_naux) != self.matrix.shape[0]:
            raise ValueError("Coulomb atom_naux must sum to the matrix dimension")
        _require_hermitian("Coulomb matrix", self.matrix)
        _require_provenance(self.provenance)


@dataclass(frozen=True)
class CoulombTransformMetadata:
    retained_rank: int
    dropped_rank: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    eigenvalue_threshold: float


@dataclass(frozen=True)
class ParentSpaceAnalysis:
    parent_dimension: int
    radial_count: object
    lmax: int
    parent_response_m: torch.Tensor
    parent_pi: torch.Tensor
    reference_pi: torch.Tensor
    per_frequency_pi_relative_frobenius: Tuple[float, ...]
    maximum_pi_relative_frobenius: float
    all_frequency_pi_relative_frobenius: float
    parent_energy_ha: float
    reference_energy_ha: float
    energy_error_ha: float
    energy_error_kcal_mol: float
    overlap_condition_by_spin: Tuple[float, ...]
    coulomb_transform: CoulombTransformMetadata


def validate_parent_space_protocol(reference, primitive, coulomb):
    if not isinstance(reference, DeltaSTReference):
        raise ValueError("reference must be a DeltaSTReference")
    if not isinstance(primitive, SternheimerPrimitiveGalerkinData):
        raise ValueError("primitive must be SternheimerPrimitiveGalerkinData")
    if not isinstance(coulomb, FullCoulombMatrix):
        raise ValueError("coulomb must be a FullCoulombMatrix")
    if reference.provenance["kernel"] != "full_coulomb":
        raise ValueError("kernel must be full_coulomb")
    if primitive.provenance["kernel"] != "full_coulomb":
        raise ValueError("kernel must be full_coulomb")
    if coulomb.provenance["kernel"] != "full_coulomb":
        raise ValueError("kernel must be full_coulomb")

    if not _same_real_tensor(reference.frequency_ha, primitive.frequency_ha):
        raise ValueError("frequency grids differ")
    if not _same_real_tensor(
        reference.frequency_weight_ha, primitive.frequency_weight_ha
    ):
        raise ValueError("frequency weights differ")
    if reference.channels != primitive.channels:
        raise ValueError("auxiliary channel order or metadata differ")
    primitive_occupation = _occupied_occupation_by_spin(primitive.occupation)
    if not _same_occupation(
        reference.occupied_occupation_by_spin, primitive_occupation
    ):
        raise ValueError("spin occupation metadata differ")

    nauxiliary = len(reference.channels)
    if coulomb.matrix.shape != (nauxiliary, nauxiliary):
        raise ValueError("Coulomb dimension differs from the response")
    if reference.atom_naux != coulomb.atom_naux:
        raise ValueError("Coulomb dimension by atom differs from the response")
    if reference.atom_naux != _atom_naux_from_channels(primitive.channels):
        raise ValueError("auxiliary channel atom dimensions differ")

    for key in _PROTOCOL_PROVENANCE_KEYS:
        values = (
            reference.provenance[key],
            primitive.provenance[key],
            coulomb.provenance[key],
        )
        if not _values_equal(values[0], values[1]) or not _values_equal(
            values[0], values[2]
        ):
            raise ValueError(f"physical provenance differs: {key}")


def build_parent_coefficients(data, radial_count, lmax):
    if not isinstance(data, SternheimerPrimitiveGalerkinData):
        raise ValueError("data must be SternheimerPrimitiveGalerkinData")
    if type(lmax) is not int or lmax < 0:
        raise ValueError("lmax must be a nonnegative integer")
    if radial_count is not None and (
        type(radial_count) is not int or radial_count <= 0
    ):
        raise ValueError("radial_count must be None or a positive integer")
    selected = []
    for block in data.blocks:
        if block.l > lmax:
            continue
        count = block.n_primitive
        if radial_count is not None:
            count = min(count, radial_count)
        selected.extend(range(block.offset, block.offset + count))
    if not selected:
        raise ValueError("parent selection is empty")
    identity = torch.eye(data.overlap.shape[0], dtype=torch.complex128)
    return identity[:, selected]


def build_parent_occupation(data, parent_dimension):
    if not isinstance(data, SternheimerPrimitiveGalerkinData):
        raise ValueError("data must be SternheimerPrimitiveGalerkinData")
    if type(parent_dimension) is not int or parent_dimension <= 0:
        raise ValueError("parent_dimension must be a positive integer")
    result = torch.zeros(
        (data.occupation.shape[0], parent_dimension), dtype=torch.float64
    )
    for spin, row in enumerate(data.occupation):
        occupied = row[row > 0.0]
        if occupied.shape[0] >= parent_dimension:
            raise ValueError(
                "parent space must contain at least one virtual state per active spin"
            )
        result[spin, : occupied.shape[0]] = occupied
    if not bool(torch.any(result > 0.0)):
        raise ValueError("at least one occupied spin channel is required")
    return result


def symmetric_response(coulomb, response_m, eigenvalue_threshold=0.0):
    _require_tensor("Coulomb matrix", coulomb, torch.complex128, 2)
    _require_tensor("response_m", response_m, torch.complex128, 3)
    if coulomb.shape[0] == 0 or coulomb.shape[0] != coulomb.shape[1]:
        raise ValueError("Coulomb matrix must be nonempty and square")
    if response_m.shape[1:] != coulomb.shape:
        raise ValueError("response_m dimensions must match the Coulomb matrix")
    _require_hermitian("Coulomb matrix", coulomb)
    _require_hermitian("response_m", response_m)
    try:
        threshold = float(eigenvalue_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("eigenvalue_threshold must be finite and nonnegative") from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("eigenvalue_threshold must be finite and nonnegative")

    eigenvalue, eigenvector = torch.linalg.eigh(coulomb)
    keep = eigenvalue > threshold
    if not bool(torch.any(keep)):
        raise ValueError("no positive Coulomb eigenvalue survives the threshold")
    positive = eigenvalue[keep]
    vectors = eigenvector[:, keep]
    response_positive = vectors.mH.unsqueeze(0) @ response_m @ vectors.unsqueeze(0)
    inverse_scale = positive.rsqrt()
    pi = (
        inverse_scale.reshape(1, -1, 1)
        * response_positive
        * inverse_scale.reshape(1, 1, -1)
    )
    pi = _hermitize(pi)
    metadata = CoulombTransformMetadata(
        retained_rank=int(torch.count_nonzero(keep)),
        dropped_rank=int(keep.shape[0] - torch.count_nonzero(keep)),
        minimum_eigenvalue=float(eigenvalue[0]),
        maximum_eigenvalue=float(eigenvalue[-1]),
        eigenvalue_threshold=threshold,
    )
    return pi, metadata


def rpa_correlation_energy(pi, frequency_weight_ha):
    _require_tensor("pi", pi, torch.complex128, 3)
    _require_tensor(
        "frequency_weight_ha", frequency_weight_ha, torch.float64, 1
    )
    if pi.shape[0] != frequency_weight_ha.shape[0]:
        raise ValueError("frequency weight count must match pi")
    _require_hermitian("pi", pi)
    eigenvalue = torch.linalg.eigvalsh(pi)
    if bool(torch.any(1.0 - eigenvalue <= 0.0)):
        raise ValueError("non-positive trace-log argument")
    integrand = torch.sum(torch.log1p(-eigenvalue) + eigenvalue, dim=1)
    return integrate_trace_log_integrand(integrand, frequency_weight_ha)


def integrate_trace_log_integrand(integrand, frequency_weight_ha):
    _require_tensor("integrand", integrand, torch.float64, 1)
    _require_tensor(
        "frequency_weight_ha", frequency_weight_ha, torch.float64, 1
    )
    if integrand.shape != frequency_weight_ha.shape:
        raise ValueError("integrand and frequency weight shapes must match")
    return float(torch.dot(frequency_weight_ha, integrand) / (2.0 * math.pi))


def analyze_parent_response(
    reference,
    primitive,
    coulomb,
    *,
    radial_count,
    lmax,
    eigenvalue_threshold=0.0,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
):
    validate_parent_space_protocol(reference, primitive, coulomb)
    coefficients = build_parent_coefficients(primitive, radial_count, lmax)
    occupation = build_parent_occupation(primitive, coefficients.shape[1])
    parent = evaluate_primitive_galerkin(
        primitive.overlap,
        primitive.hamiltonian_ha,
        primitive.perturbation_ha,
        coefficients,
        occupation,
        primitive.frequency_ha,
        relative_rank_tolerance=relative_rank_tolerance,
        condition_limit=condition_limit,
    )
    reference_pi, transform = symmetric_response(
        coulomb.matrix, reference.response_m, eigenvalue_threshold
    )
    parent_pi, parent_transform = symmetric_response(
        coulomb.matrix, parent.response, eigenvalue_threshold
    )
    if parent_transform != transform:
        raise RuntimeError("Coulomb transforms differ")
    per_frequency = tuple(
        _relative_frobenius(parent_pi[index], reference_pi[index])
        for index in range(reference_pi.shape[0])
    )
    parent_energy = rpa_correlation_energy(
        parent_pi, reference.frequency_weight_ha
    )
    reference_energy = rpa_correlation_energy(
        reference_pi, reference.frequency_weight_ha
    )
    energy_error = parent_energy - reference_energy
    return ParentSpaceAnalysis(
        parent_dimension=coefficients.shape[1],
        radial_count=radial_count,
        lmax=lmax,
        parent_response_m=parent.response,
        parent_pi=parent_pi,
        reference_pi=reference_pi,
        per_frequency_pi_relative_frobenius=per_frequency,
        maximum_pi_relative_frobenius=max(per_frequency),
        all_frequency_pi_relative_frobenius=_relative_frobenius(
            parent_pi, reference_pi
        ),
        parent_energy_ha=parent_energy,
        reference_energy_ha=reference_energy,
        energy_error_ha=energy_error,
        energy_error_kcal_mol=energy_error * _HARTREE_TO_KCAL_MOL,
        overlap_condition_by_spin=parent.overlap_condition_by_spin,
        coulomb_transform=transform,
    )


def load_delta_st_reference(directory, iq=1):
    directory = pathlib.Path(directory)
    manifest_path = directory / "reference_protocol.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reference protocol {manifest_path}") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "occupied_occupation_by_spin",
        "provenance",
    }:
        raise ValueError("reference protocol must contain occupation and provenance")
    occupation = _normalize_occupation(manifest["occupied_occupation_by_spin"])
    provenance = manifest["provenance"]
    _require_provenance(provenance)
    channels = _read_auxiliary_channels(
        directory / "STERNHEIMER_ABFS_CHANNELS.dat"
    )

    pattern = f"v1_sternheimer_chi0_iq_{iq}_ifreq_*_rank*.dat"
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise ValueError(f"no Delta-ST response shards match {pattern}")
    by_frequency = {}
    for path in paths:
        shard = _read_delta_response_shard(path)
        if shard["iq"] != iq:
            raise ValueError("Delta-ST response iq differs from request")
        by_frequency.setdefault(shard["ifreq"], []).append(shard)
    expected = list(range(1, len(by_frequency) + 1))
    if sorted(by_frequency) != expected:
        raise ValueError("Delta-ST frequency indices must be contiguous from one")

    responses = []
    frequencies = []
    weights = []
    reference_metadata = None
    for ifreq in expected:
        matrix, metadata = _assemble_response_shards(by_frequency[ifreq])
        if reference_metadata is None:
            reference_metadata = metadata
        elif (
            metadata["naux"] != reference_metadata["naux"]
            or metadata["atom_naux"] != reference_metadata["atom_naux"]
        ):
            raise ValueError("Delta-ST auxiliary metadata differs across frequencies")
        responses.append(matrix)
        frequencies.append(metadata["omega"])
        weights.append(metadata["weight"])
    if reference_metadata["naux"] != len(channels):
        raise ValueError("Delta-ST response and channel dimensions differ")
    return DeltaSTReference(
        response_m=torch.stack(tuple(responses)),
        frequency_ha=torch.tensor(frequencies, dtype=torch.float64),
        frequency_weight_ha=torch.tensor(weights, dtype=torch.float64),
        channels=channels,
        atom_naux=tuple(reference_metadata["atom_naux"]),
        occupied_occupation_by_spin=occupation,
        provenance=provenance,
    )


def load_full_coulomb_matrix(directory, provenance, iq=1):
    matrix, atom_naux = read_coulomb(directory, iq=iq)
    return FullCoulombMatrix(
        matrix=torch.tensor(matrix, dtype=torch.complex128),
        atom_naux=tuple(atom_naux),
        provenance=dict(provenance),
    )


def _read_auxiliary_channels(path):
    channels = []
    try:
        lines = pathlib.Path(path).read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read auxiliary channels {path}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"auxiliary channel row {line_number} must have 9 fields")
        try:
            angular_momentum = int(fields[4])
            real_harmonic_index = int(fields[6])
            channel = AuxiliaryChannel(
                channel_index=int(fields[0]),
                atom_index=int(fields[1]),
                l=angular_momentum,
                radial_index=int(fields[5]),
                m=real_harmonic_index - angular_momentum,
                label=fields[7],
            )
            float(fields[8])
        except ValueError as exc:
            raise ValueError(f"invalid auxiliary channel row {line_number}") from exc
        if channel.channel_index != len(channels):
            raise ValueError("auxiliary channel indices must be contiguous from zero")
        channels.append(channel)
    if not channels:
        raise ValueError("auxiliary channel file is empty")
    return tuple(channels)


def _read_delta_response_shard(path):
    data = pathlib.Path(path).read_bytes()
    header = struct.Struct("<6i2di")
    if len(data) < header.size:
        raise ValueError(f"{path}: truncated Delta-ST response header")
    values = header.unpack_from(data)
    marker, iq, ifreq, naux, value_flag, natom, omega, weight, nblock = values
    if (
        marker != _RESPONSE_MARKER
        or iq <= 0
        or ifreq <= 0
        or naux <= 0
        or value_flag not in (0, 1)
        or natom <= 0
        or nblock < 0
        or not math.isfinite(omega)
        or not math.isfinite(weight)
    ):
        raise ValueError(f"{path}: invalid Delta-ST response header")
    cursor = header.size
    atom_struct = struct.Struct(f"<{natom}i")
    if len(data) < cursor + atom_struct.size:
        raise ValueError(f"{path}: truncated Delta-ST atom dimensions")
    atom_naux = tuple(atom_struct.unpack_from(data, cursor))
    cursor += atom_struct.size
    if any(value <= 0 for value in atom_naux) or sum(atom_naux) != naux:
        raise ValueError(f"{path}: invalid Delta-ST atom dimensions")
    record = struct.Struct("<iq")
    table_end = cursor + nblock * record.size
    if len(data) < table_end:
        raise ValueError(f"{path}: truncated Delta-ST block table")
    records = []
    for _ in range(nblock):
        records.append(record.unpack_from(data, cursor))
        cursor += record.size
    pairs = _atom_pairs(natom)
    scalar = struct.Struct("<d" if value_flag == 0 else "<2d")
    blocks = {}
    for pair_index, offset in records:
        if pair_index < 0 or pair_index >= len(pairs) or pair_index in blocks:
            raise ValueError(f"{path}: invalid Delta-ST atom-pair index")
        iatom, jatom = pairs[pair_index]
        count = atom_naux[iatom] * atom_naux[jatom]
        if offset < table_end or offset + count * scalar.size > len(data):
            raise ValueError(f"{path}: invalid Delta-ST payload range")
        block = torch.empty(count, dtype=torch.complex128)
        for index in range(count):
            raw = scalar.unpack_from(data, offset + index * scalar.size)
            block[index] = complex(raw[0], 0.0 if value_flag == 0 else raw[1])
        blocks[pair_index] = block.reshape(
            atom_naux[iatom], atom_naux[jatom]
        )
    return {
        "iq": iq,
        "ifreq": ifreq,
        "omega": omega,
        "weight": weight,
        "naux": naux,
        "natom": natom,
        "atom_naux": atom_naux,
        "blocks": blocks,
    }


def _assemble_response_shards(shards):
    metadata = shards[0]
    for shard in shards[1:]:
        for key in ("iq", "ifreq", "omega", "weight", "naux", "natom", "atom_naux"):
            if shard[key] != metadata[key]:
                raise ValueError("inconsistent Delta-ST response shard metadata")
    pairs = _atom_pairs(metadata["natom"])
    blocks = {}
    for shard in shards:
        for pair_index, block in shard["blocks"].items():
            if pair_index in blocks:
                raise ValueError("duplicate Delta-ST atom-pair block")
            blocks[pair_index] = block
    if set(blocks) != set(range(len(pairs))):
        raise ValueError("incomplete Delta-ST atom-pair blocks")
    offsets = _offsets(metadata["atom_naux"])
    matrix = torch.zeros(
        (metadata["naux"], metadata["naux"]), dtype=torch.complex128
    )
    for pair_index, block in blocks.items():
        iatom, jatom = pairs[pair_index]
        i0, j0 = offsets[iatom], offsets[jatom]
        ni, nj = block.shape
        matrix[i0 : i0 + ni, j0 : j0 + nj] = block
        if iatom != jatom:
            matrix[j0 : j0 + nj, i0 : i0 + ni] = block.mH
    _require_hermitian("Delta-ST response", matrix)
    return matrix, metadata


def _require_tensor(name, value, dtype, rank):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype or value.device.type != "cpu" or value.ndim != rank:
        raise ValueError(f"{name} must be CPU {dtype} with rank {rank}")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")


def _require_hermitian(name, value):
    if not torch.allclose(value, value.mH, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{name} must be Hermitian")


def _require_provenance(provenance):
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a dictionary")
    for key in _PROTOCOL_PROVENANCE_KEYS:
        if key not in provenance:
            raise ValueError(f"missing physical provenance key: {key}")


def _require_occupations(occupation):
    if not isinstance(occupation, tuple) or not occupation:
        raise ValueError("occupied_occupation_by_spin must be a nonempty tuple")
    for row in occupation:
        if not isinstance(row, tuple):
            raise ValueError("each spin occupation row must be a tuple")
        for value in row:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("occupied spin occupations must be positive and finite")


def _normalize_occupation(value):
    if not isinstance(value, list) or not value:
        raise ValueError("occupied_occupation_by_spin must be a nonempty array")
    result = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError("spin occupation rows must be arrays")
        result.append(tuple(float(item) for item in row))
    normalized = tuple(result)
    _require_occupations(normalized)
    return normalized


def _occupied_occupation_by_spin(occupation):
    return tuple(
        tuple(float(value) for value in row[row > 0.0]) for row in occupation
    )


def _same_occupation(left, right):
    return len(left) == len(right) and all(
        len(left_row) == len(right_row)
        and all(abs(a - b) <= 1.0e-14 for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def _same_real_tensor(left, right):
    return left.shape == right.shape and bool(
        torch.allclose(left, right, rtol=0.0, atol=1.0e-14)
    )


def _atom_naux_from_channels(channels):
    natom = max(channel.atom_index for channel in channels) + 1
    counts = [0] * natom
    for channel in channels:
        counts[channel.atom_index] += 1
    if any(count == 0 for count in counts):
        raise ValueError("auxiliary channels contain an empty atom block")
    return tuple(counts)


def _atom_pairs(natom):
    return [
        (iatom, jatom)
        for iatom in range(natom)
        for jatom in range(iatom, natom)
    ]


def _offsets(atom_naux):
    result = [0]
    for count in atom_naux:
        result.append(result[-1] + count)
    return result


def _relative_frobenius(actual, reference):
    difference = float(torch.linalg.vector_norm(actual - reference))
    denominator = float(torch.linalg.vector_norm(reference))
    if denominator == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / denominator


def _hermitize(value):
    return 0.5 * (value + value.mH)


def _values_equal(left, right):
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_values_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) == float(right)
        except (TypeError, ValueError):
            return False
    return left == right
