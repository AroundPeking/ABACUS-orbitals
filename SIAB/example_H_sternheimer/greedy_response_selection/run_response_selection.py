#!/usr/bin/env python3
"""Run the frozen H response-shell campaign from validated physical targets."""

import argparse
import json
from pathlib import Path

from response_selection_campaign import (
    load_response_families,
    read_optimizer_coefficients,
    resolve_optimizer_template_paths,
    run_response_selection_campaign,
)


def _json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _required_file(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def _fixed_nu(fixed_specs, max_l, element):
    counts = [0] * (max_l + 1)
    seen = set()
    for spec in fixed_specs:
        if not isinstance(spec, dict) or set(spec) != {"element", "l", "zeta"}:
            raise ValueError("fixed orbital requires element, l, and zeta")
        if spec["element"] != element:
            raise ValueError("fixed orbital element does not match campaign element")
        l = spec["l"]
        zeta = spec["zeta"]
        if type(l) is not int or l < 0 or l > max_l:
            raise ValueError("fixed orbital l is outside the campaign")
        if type(zeta) is not int or zeta <= 0 or (l, zeta) in seen:
            raise ValueError("fixed orbital zeta is invalid or duplicated")
        seen.add((l, zeta))
        counts[l] = max(counts[l], zeta)
    for l, count in enumerate(counts):
        if any((l, zeta) not in seen for zeta in range(1, count + 1)):
            raise ValueError("fixed orbitals must be contiguous within each l")
    if not seen:
        raise ValueError("fixed orbital list must be nonempty")
    return tuple(counts)


def _template_files(template):
    file_list = template["file_list"]
    return tuple(file_list["origin"]) + tuple(
        path for group in file_list["linear"] for path in group
    )


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--optimizer-template", required=True, type=Path)
    value.add_argument("--asset-root", required=True, type=Path)
    value.add_argument("--baseline", required=True, type=Path)
    value.add_argument("--atom-target", required=True, type=Path)
    value.add_argument("--multicenter-target", required=True, type=Path)
    value.add_argument("--ghost-target", required=True, type=Path)
    value.add_argument("--optimizer", required=True, type=Path)
    value.add_argument("--python", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--condition-limit", type=float, default=1.0e12)
    value.add_argument("--max-steps", type=int, default=64)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    config_path = _required_file(args.config)
    template_path = _required_file(args.optimizer_template)
    baseline_path = _required_file(args.baseline)
    optimizer = _required_file(args.optimizer)
    python = _required_file(args.python)
    target_paths = tuple(
        _required_file(path)
        for path in (
            args.atom_target,
            args.multicenter_target,
            args.ghost_target,
        )
    )

    config = _json(config_path)
    fixed_specs = tuple(config["fixed_orbitals"])
    max_l = config["max_l"]
    element = config["element"]
    expected_nu = _fixed_nu(fixed_specs, max_l, element)
    initial = read_optimizer_coefficients(
        baseline_path,
        element=element,
        radial_rows=25,
        max_l=max_l,
        expected_nu=expected_nu,
    )

    optimizer_template = resolve_optimizer_template_paths(
        _json(template_path), args.asset_root.resolve()
    )
    for path in _template_files(optimizer_template):
        _required_file(path)
    targets = [
        {
            "path": str(target_paths[0]),
            "family": "atom",
            "role": "physical",
        },
        {
            "path": str(target_paths[1]),
            "family": "multicenter",
            "role": "physical",
        },
        {
            "path": str(target_paths[2]),
            "family": "fragment_ghost",
            "role": "ghost",
            "element_aliases": {"H_empty": "H"},
        },
    ]
    families = load_response_families(targets)
    result = run_response_selection_campaign(
        config=config,
        initial=initial,
        fixed_specs=fixed_specs,
        families=families,
        optimizer_template=optimizer_template,
        targets=targets,
        output_dir=args.output,
        optimizer=optimizer,
        python=python,
        condition_limit=args.condition_limit,
        max_steps=args.max_steps,
    )
    print(
        json.dumps(
            {
                "status": result.selection.status,
                "steps": len(result.selection.steps),
                "selection_manifest": str(result.selection_manifest),
                "campaign_manifest": str(result.campaign_manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
