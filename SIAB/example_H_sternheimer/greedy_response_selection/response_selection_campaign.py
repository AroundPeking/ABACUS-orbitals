"""I/O contracts shared by the physical response-shell selection campaign."""

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from IO.read_sternheimer import read_sternheimer
from response_selection import ResponseTargetFamily
from select_response_shells import (
    SelectionOptimization,
    build_step_input,
    freeze_selection_sequence,
    run_joint_optimizer,
    run_nested_selection,
)
from sternheimer_spillage import (
    RadialResidualSpectrum,
    radial_residual_spectrum_many,
)
from sternheimer_targets import (
    apply_target_element_aliases,
    parse_target_entries,
)


_FAMILY_ROLES = {
    "atom": "physical",
    "multicenter": "physical",
}

_LOCALITY_OVERRIDE_KEYS = frozenset(
    {
        "radial_tail_weight",
        "radial_tail_radius",
        "radial_tail_condition_limit",
    }
)


@dataclass(frozen=True)
class OptimizedResponseStep:
    coefficients: dict
    artifact_sha256: dict
    input_path: Path
    metrics: dict


@dataclass(frozen=True)
class ResponseSelectionCampaignResult:
    selection: object
    selection_manifest: Path
    campaign_manifest: Path


def apply_optimizer_loss_overrides(template, config):
    """Apply only the predeclared radial-locality lane to a joint template."""
    if not isinstance(template, dict):
        raise TypeError("optimizer template must be a dictionary")
    if not isinstance(config, dict):
        raise TypeError("selection config must be a dictionary")
    result = copy.deepcopy(template)
    overrides = config.get("optimizer_loss")
    if overrides is None:
        return result
    if not isinstance(overrides, dict) or set(overrides) != _LOCALITY_OVERRIDE_KEYS:
        raise ValueError(
            "optimizer_loss may contain only radial locality weight, radius, "
            "and condition limit"
        )
    try:
        loss = result["loss"]
    except (KeyError, TypeError) as exc:
        raise ValueError("optimizer template requires a loss dictionary") from exc
    if not isinstance(loss, dict):
        raise ValueError("optimizer template requires a loss dictionary")
    loss.update(copy.deepcopy(overrides))
    return result


def _validate_nu(expected_nu, max_l):
    try:
        values = tuple(expected_nu)
    except TypeError as exc:
        raise TypeError("expected_nu must be a sequence") from exc
    if len(values) != max_l + 1:
        raise ValueError("expected_nu must contain one count for every l")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("expected_nu counts must be nonnegative integers")
    return values


def _validate_coefficient_channel(channel, name):
    if not isinstance(channel, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if channel.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if channel.shape[0] <= 0:
        raise ValueError(f"{name} must contain radial rows")
    if not bool(torch.all(torch.isfinite(channel))):
        raise ValueError(f"{name} must contain only finite coefficients")


def extract_fixed_reference_coefficients(coefficients, fixed_specs):
    """Extract the contiguous fixed-zeta prefix from a full initial basis."""
    if not isinstance(coefficients, dict) or not coefficients:
        raise TypeError("coefficients must be a nonempty dictionary")
    requested = {}
    for spec in fixed_specs:
        if not isinstance(spec, dict) or set(spec) != {
            "element",
            "l",
            "zeta",
        }:
            raise ValueError("fixed orbital spec requires element, l, and zeta")
        element = spec["element"]
        l = spec["l"]
        zeta = spec["zeta"]
        if element not in coefficients:
            raise ValueError(f"fixed orbital element {element!r} is missing")
        if type(l) is not int or l < 0 or l >= len(coefficients[element]):
            raise ValueError(f"fixed orbital l={l!r} is missing")
        if type(zeta) is not int or zeta <= 0:
            raise ValueError("fixed orbital zeta must be a positive integer")
        requested.setdefault((element, l), set()).add(zeta)

    result = {}
    for element, channels in coefficients.items():
        result[element] = []
        for l, channel in enumerate(channels):
            _validate_coefficient_channel(
                channel, f"coefficients[{element!r}][{l}]"
            )
            zetas = sorted(requested.get((element, l), ()))
            if zetas and zetas != list(range(1, len(zetas) + 1)):
                raise ValueError(
                    "fixed orbital zetas must form a contiguous prefix"
                )
            if len(zetas) > channel.shape[1]:
                raise ValueError(
                    f"fixed orbital {element}/{l}/zeta{zetas[-1]} is missing"
                )
            result[element].append(channel[:, : len(zetas)].detach().clone())
    if not requested:
        raise ValueError("fixed orbital specs must be nonempty")
    return result


def _next_nonempty(lines, index):
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise ValueError("coefficient file ended unexpectedly")
    return lines[index].strip(), index + 1


def read_optimizer_coefficients(
    path,
    *,
    element,
    radial_rows,
    max_l,
    expected_nu,
):
    """Read one element from SIAB's native ORBITAL_RESULTS coefficient block."""
    if not isinstance(element, str) or not element:
        raise ValueError("element must be nonempty")
    if type(radial_rows) is not int or radial_rows <= 0:
        raise ValueError("radial_rows must be a positive integer")
    if type(max_l) is not int or max_l < 0:
        raise ValueError("max_l must be a nonnegative integer")
    if expected_nu is not None:
        expected_nu = _validate_nu(expected_nu, max_l)

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        index = next(
            position + 1
            for position, line in enumerate(lines)
            if line.strip() == "<Coefficient>"
        )
    except StopIteration as exc:
        raise ValueError("missing <Coefficient> section") from exc

    declared_total = None
    columns = {}
    closed = False
    while index < len(lines):
        line, index = _next_nonempty(lines, index)
        if line == "</Coefficient>":
            closed = True
            break
        if "Total number of radial orbitals" in line:
            fields = line.split()
            try:
                declared_total = int(fields[0])
            except (IndexError, ValueError) as exc:
                raise ValueError("invalid declared coefficient count") from exc
            continue
        if not line.startswith("Type"):
            raise ValueError(f"unexpected coefficient row: {line}")

        label, index = _next_nonempty(lines, index)
        fields = label.split()
        if len(fields) != 3:
            raise ValueError(f"invalid coefficient label: {label}")
        label_element = fields[0]
        try:
            l = int(fields[1])
            zeta = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"invalid coefficient label: {label}") from exc
        if label_element != element:
            raise ValueError(
                f"unexpected coefficient element {label_element!r}; expected {element!r}"
            )
        if (
            l < 0
            or l > max_l
            or zeta <= 0
            or (expected_nu is not None and zeta > expected_nu[l])
        ):
            raise ValueError(f"coefficient column {(element, l, zeta)!r} is unexpected")
        key = (l, zeta)
        if key in columns:
            raise ValueError(f"duplicate coefficient column {(element, l, zeta)!r}")

        values = []
        while len(values) < radial_rows:
            value_line, index = _next_nonempty(lines, index)
            if value_line == "</Coefficient>" or value_line.startswith("Type"):
                raise ValueError(f"coefficient column {(element, l, zeta)!r} is incomplete")
            try:
                values.extend(float(value) for value in value_line.split())
            except ValueError as exc:
                raise ValueError(
                    f"coefficient column {(element, l, zeta)!r} is not numeric"
                ) from exc
        if len(values) != radial_rows or any(not math.isfinite(value) for value in values):
            raise ValueError(f"coefficient column {(element, l, zeta)!r} is invalid")
        columns[key] = values

    if not closed:
        raise ValueError("missing </Coefficient> section")
    if expected_nu is None:
        inferred_nu = [0] * (max_l + 1)
        for l, zeta in columns:
            inferred_nu[l] = max(inferred_nu[l], zeta)
        expected_nu = tuple(inferred_nu)
    expected_total = sum(expected_nu)
    if declared_total is not None and declared_total != expected_total:
        raise ValueError("declared coefficient count does not match expected_nu")

    by_l = []
    for l, count in enumerate(expected_nu):
        missing = [zeta for zeta in range(1, count + 1) if (l, zeta) not in columns]
        if missing:
            raise ValueError(f"missing coefficient column {(element, l, missing[0])!r}")
        if count:
            by_l.append(
                torch.tensor(
                    [columns[(l, zeta)] for zeta in range(1, count + 1)],
                    dtype=torch.float64,
                ).transpose(0, 1).contiguous()
            )
        else:
            by_l.append(torch.empty((radial_rows, 0), dtype=torch.float64))
    return {element: by_l}


def write_optimizer_coefficients(path, coefficients):
    """Write coefficients in the native format consumed by SIAB main.py."""
    if not isinstance(coefficients, dict) or not coefficients:
        raise TypeError("coefficients must be a nonempty dictionary")
    total = 0
    validated = []
    for element in sorted(coefficients):
        if not isinstance(element, str) or not element:
            raise ValueError("coefficient element must be nonempty")
        for l, channel in enumerate(coefficients[element]):
            if (
                not isinstance(channel, torch.Tensor)
                or channel.ndim != 2
                or channel.dtype != torch.float64
                or channel.is_complex()
                or channel.device.type != "cpu"
                or not bool(torch.all(torch.isfinite(channel)))
            ):
                raise ValueError("optimizer coefficients must be finite CPU float64 matrices")
            for column in range(channel.shape[1]):
                validated.append((element, l, column + 1, channel[:, column]))
                total += 1

    output = ["<Coefficient>", f"\t {total} Total number of radial orbitals."]
    for element, l, zeta, column in validated:
        output.extend(
            (
                "\tType\tL\tZeta-Orbital",
                f"\t  {element} \t{l}\t    {zeta}",
            )
        )
        output.extend(f"\t {float(value):18.14f}" for value in column)
    output.extend(
        (
            "</Coefficient>",
            "<Mkb>",
            "Left spillage = 0.0000000000e+00",
            "</Mkb>",
            "",
        )
    )
    Path(path).write_text("\n".join(output), encoding="utf-8")


_OPTIMIZER_METRIC_LABELS = {
    "Sternheimer loss": "sternheimer",
    "Radial tail fraction": "radial_tail",
    "Radial locality regularization loss": "regularization_locality",
    "Total loss": "total",
    "Maximum ST overlap condition": "max_st_condition",
    "Maximum radial locality condition": "max_locality_condition",
}


def read_optimizer_metrics(path):
    """Read the accepted SIAB loss and condition diagnostics from <Mkb>."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    inside = False
    closed = False
    values = {}
    for raw in lines:
        line = raw.strip()
        if line == "<Mkb>":
            if inside:
                raise ValueError("duplicate <Mkb> section")
            inside = True
            continue
        if line == "</Mkb>" and inside:
            closed = True
            break
        if not inside or "=" not in line:
            continue
        label, value = (field.strip() for field in line.split("=", 1))
        name = _OPTIMIZER_METRIC_LABELS.get(label)
        if name is None:
            continue
        if name in values:
            raise ValueError(f"duplicate optimizer metric {name}")
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"optimizer metric {name} is not numeric") from exc
        lower_bound = 1.0 if name.startswith("max_") else 0.0
        if not math.isfinite(number) or number < lower_bound:
            raise ValueError(
                f"optimizer metric {name} must be finite and at least "
                f"{lower_bound:g}"
            )
        values[name] = number
    if not inside or not closed:
        raise ValueError("optimizer result requires one closed <Mkb> section")
    expected = set(_OPTIMIZER_METRIC_LABELS.values())
    if set(values) != expected:
        missing = ", ".join(sorted(expected - set(values)))
        raise ValueError(f"optimizer result is missing metrics: {missing}")
    return values


def _absolute_path(value, base_dir, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} paths must be nonempty strings")
    path = Path(value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


def resolve_optimizer_template_paths(template, base_dir):
    """Resolve legacy origin/dpsi paths before optimization changes cwd."""
    if not isinstance(template, dict):
        raise TypeError("optimizer template must be a dictionary")
    result = copy.deepcopy(template)
    try:
        file_list = result["file_list"]
        origin = file_list["origin"]
        linear = file_list["linear"]
    except (KeyError, TypeError) as exc:
        raise ValueError("optimizer template requires origin and linear files") from exc
    if not isinstance(origin, list) or not origin:
        raise ValueError("optimizer template origin must be a nonempty list")
    if not isinstance(linear, list) or not linear:
        raise ValueError("optimizer template linear must be a nonempty list")
    file_list["origin"] = [
        _absolute_path(path, base_dir, "origin") for path in origin
    ]
    resolved_linear = []
    for group in linear:
        if not isinstance(group, list) or not group:
            raise ValueError("each optimizer linear group must be nonempty")
        resolved_linear.append(
            [_absolute_path(path, base_dir, "linear") for path in group]
        )
    file_list["linear"] = resolved_linear
    return result


def _coefficient_shape(coefficients):
    if not isinstance(coefficients, dict) or len(coefficients) != 1:
        raise ValueError("the H response campaign requires exactly one element")
    element = next(iter(coefficients))
    channels = coefficients[element]
    if not isinstance(channels, (list, tuple)) or not channels:
        raise ValueError("response coefficients require angular channels")
    radial_rows = None
    nu = []
    for channel in channels:
        if not isinstance(channel, torch.Tensor) or channel.ndim != 2:
            raise ValueError("response coefficient channels must be matrices")
        if radial_rows is None:
            radial_rows = channel.shape[0]
        elif channel.shape[0] != radial_rows:
            raise ValueError("response coefficient channels must share radial rows")
        nu.append(channel.shape[1])
    if radial_rows is None or radial_rows <= 0:
        raise ValueError("response coefficients require positive radial rows")
    return element, radial_rows, tuple(nu)


def optimize_response_step(
    *,
    step,
    coefficients,
    template,
    targets,
    fixed_specs,
    seed,
    output_dir,
    optimizer,
    python,
    require_metrics=False,
):
    """Run one checked native SIAB joint optimization and read its coefficients."""
    if type(step) is not int or step <= 0:
        raise ValueError("step must be a positive integer")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"optimization step output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    element, radial_rows, nu = _coefficient_shape(coefficients)
    initial_path = output_dir / "INITIAL_ORBITAL_RESULTS.txt"
    write_optimizer_coefficients(initial_path, coefficients)
    input_value = build_step_input(
        template,
        targets,
        initial_path.resolve(),
        {element: list(nu)},
        fixed_specs,
        seed,
    )
    optimizer_dir = output_dir / "optimizer"
    artifacts = run_joint_optimizer(
        input_value,
        optimizer_dir,
        optimizer,
        python,
    )
    optimized = read_optimizer_coefficients(
        optimizer_dir / "ORBITAL_RESULTS.txt",
        element=element,
        radial_rows=radial_rows,
        max_l=len(nu) - 1,
        expected_nu=nu,
    )
    result_path = optimizer_dir / "ORBITAL_RESULTS.txt"
    result_text = result_path.read_text(encoding="utf-8")
    if require_metrics or "Maximum ST overlap condition" in result_text:
        metrics = read_optimizer_metrics(result_path)
    else:
        metrics = {}
    return OptimizedResponseStep(
        coefficients=optimized,
        artifact_sha256=artifacts,
        input_path=optimizer_dir / "INPUT",
        metrics=metrics,
    )


def assemble_response_families(loaded_entries):
    """Build the atom and multicenter physical selection families."""
    loaded_entries = tuple(loaded_entries)
    families = {}
    for entry, data in loaded_entries:
        expected_role = _FAMILY_ROLES.get(entry.family)
        if expected_role is None or entry.role != expected_role:
            raise ValueError(
                "response selection requires exactly atom, multicenter "
                "with physical roles"
            )
        if entry.family in families:
            raise ValueError(f"duplicate response target family {entry.family!r}")
        families[entry.family] = data
    if set(families) != set(_FAMILY_ROLES):
        raise ValueError(
            "response selection requires exactly atom, multicenter "
            "with physical roles"
        )
    return (
        ResponseTargetFamily("atom", (families["atom"],), "physical"),
        ResponseTargetFamily(
            "multicenter", (families["multicenter"],), "physical"
        ),
    )


def load_response_families(targets):
    """Read, alias, and assemble the two physical campaign targets once."""
    entries = parse_target_entries(targets)
    loaded = tuple(
        (
            entry,
            apply_target_element_aliases(read_sternheimer(entry.path), entry),
        )
        for entry in entries
    )
    return assemble_response_families(loaded)


def _current_radial_specs(coefficients):
    specs = []
    for element in sorted(coefficients):
        for l, channel in enumerate(coefficients[element]):
            if not isinstance(channel, torch.Tensor) or channel.ndim != 2:
                raise ValueError("response coefficient channels must be matrices")
            specs.extend(
                {"element": element, "l": l, "zeta": zeta}
                for zeta in range(1, channel.shape[1] + 1)
            )
    if not specs:
        raise ValueError("response spectrum requires at least one current orbital")
    return tuple(specs)


def build_response_spectrum_builder(
    atom_family,
    multicenter_family,
    *,
    element,
    max_l,
    relative_rank_tolerance,
    magnetic_overlap_tolerance,
    condition_limit,
):
    """Return atomic candidate spectra for physical atom+multicenter scoring."""
    for family, name in (
        (atom_family, "atom"),
        (multicenter_family, "multicenter"),
    ):
        if not isinstance(family, ResponseTargetFamily):
            raise TypeError(f"{name} family must be a ResponseTargetFamily")
        if family.name != name or family.role != "physical":
            raise ValueError(f"{name} family has the wrong name or role")
    if not isinstance(element, str) or not element:
        raise ValueError("element must be nonempty")
    if type(max_l) is not int or max_l < 0:
        raise ValueError("max_l must be a nonnegative integer")
    # A common m-resolved radial metric is exact for the spherical atom target.
    # The multicenter family enters the full-projector candidate score and the
    # joint optimization, where molecular m splitting is represented directly.
    data_items = atom_family.data

    def build(coefficients):
        current_specs = _current_radial_specs(coefficients)
        spectra = []
        for l in range(max_l + 1):
            try:
                spectrum = radial_residual_spectrum_many(
                    data_items,
                    coefficients,
                    current_specs,
                    element,
                    l,
                    relative_rank_tolerance=relative_rank_tolerance,
                    magnetic_overlap_tolerance=magnetic_overlap_tolerance,
                    condition_limit=condition_limit,
                )
            except RuntimeError as exc:
                if str(exc) != "projected primitive overlap has no positive modes":
                    raise
                radial_rows = coefficients[element][l].shape[0]
                atom_indices = sorted(
                    {
                        block.atom_index
                        for data in data_items
                        for block in data.blocks
                        if block.element == element and block.l == l
                    }
                )
                spectrum = RadialResidualSpectrum(
                    element=element,
                    atom_index=None,
                    l=l,
                    magnetic_channels=tuple(range(-l, l + 1)),
                    numerical_rank=0,
                    eigenvalues=torch.zeros(1, dtype=torch.float64),
                    cumulative_capture=torch.zeros(1, dtype=torch.float64),
                    coefficients=torch.zeros((radial_rows, 1), dtype=torch.float64),
                    overlap_relative_deviation=0.0,
                    atom_indices=tuple(atom_indices),
                )
            spectra.append(spectrum)
        return tuple(spectra)

    return build


def _metrics_payload(metrics):
    return {
        "atom_loss": metrics.atom_loss,
        "multicenter_loss": metrics.multicenter_loss,
        "atom_floor": metrics.atom_floor,
        "multicenter_floor": metrics.multicenter_floor,
        "atom_representable_loss": metrics.atom_representable_loss,
        "multicenter_representable_loss": (
            metrics.multicenter_representable_loss
        ),
        "global_capture": metrics.global_capture,
        "per_l_residual_ratio": {
            str(l): value
            for l, value in sorted(metrics.per_l_residual_ratio.items())
        },
    }


def run_response_selection_campaign(
    *,
    config,
    initial,
    fixed_specs,
    families,
    optimizer_template,
    targets,
    output_dir,
    optimizer,
    python,
    condition_limit,
    max_steps=64,
):
    """Run the nested H-only response selection without any H2 feedback."""
    if not isinstance(config, dict):
        raise TypeError("selection config must be a dictionary")
    optimizer_template = apply_optimizer_loss_overrides(
        optimizer_template, config
    )
    try:
        seed = config["seed"]
        max_l = config["max_l"]
        relative_rank_tolerance = config["relative_rank_tolerance"]
        magnetic_overlap_tolerance = config["magnetic_overlap_tolerance"]
    except KeyError as exc:
        raise ValueError(f"selection config is missing {exc.args[0]}") from exc
    if type(seed) is not int or seed < 0 or seed >= 2**32:
        raise ValueError("selection seed must satisfy 0 <= seed < 2**32")
    if type(max_l) is not int or max_l < 0:
        raise ValueError("selection max_l must be a nonnegative integer")
    element, _, nu = _coefficient_shape(initial)
    if len(nu) != max_l + 1:
        raise ValueError("initial coefficient channels do not match max_l")
    families = tuple(families)
    if len(families) != 2:
        raise ValueError("selection campaign requires two physical response families")
    atom_family, multicenter_family = families

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"campaign output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir()
    spectrum_builder = build_response_spectrum_builder(
        atom_family,
        multicenter_family,
        element=element,
        max_l=max_l,
        relative_rank_tolerance=relative_rank_tolerance,
        magnetic_overlap_tolerance=magnetic_overlap_tolerance,
        condition_limit=condition_limit,
    )
    optimizer_records = {}
    compact = config.get("selection_mode") == "ao_budget_frontier"
    fixed_reference = extract_fixed_reference_coefficients(
        initial, fixed_specs
    )

    def optimize_step(index, coefficients, selected):
        del selected
        value = optimize_response_step(
            step=index,
            coefficients=coefficients,
            template=optimizer_template,
            targets=targets,
            fixed_specs=fixed_specs,
            seed=seed,
            output_dir=work_dir / f"step_{index:03d}",
            optimizer=optimizer,
            python=python,
            require_metrics=compact,
        )
        optimizer_records[str(index)] = {
            "input": str(value.input_path.relative_to(output_dir)),
            "artifacts": value.artifact_sha256,
            "optimization_metrics": value.metrics,
        }
        if compact:
            return SelectionOptimization(value.coefficients, value.metrics)
        return value.coefficients

    selection = run_nested_selection(
        config,
        initial,
        fixed_reference,
        fixed_specs,
        atom_family,
        multicenter_family,
        spectrum_builder,
        optimize_step,
        max_steps=max_steps,
        condition_limit=condition_limit,
    )
    selection_manifest = freeze_selection_sequence(
        output_dir / "frozen",
        config,
        initial,
        fixed_specs,
        selection.steps,
    )
    campaign_payload = {
        "format_version": 1,
        "status": selection.status,
        "steps": len(selection.steps),
        "metrics": _metrics_payload(selection.metrics),
        "selection_manifest": str(selection_manifest.relative_to(output_dir)),
        "optimizer_steps": optimizer_records,
        "targets": copy.deepcopy(targets),
    }
    campaign_manifest = output_dir / "campaign_manifest.json"
    campaign_manifest.write_text(
        json.dumps(
            campaign_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return ResponseSelectionCampaignResult(
        selection=selection,
        selection_manifest=selection_manifest,
        campaign_manifest=campaign_manifest,
    )
