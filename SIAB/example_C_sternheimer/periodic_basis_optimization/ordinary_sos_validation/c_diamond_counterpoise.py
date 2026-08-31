#!/usr/bin/env python3
"""Prepare and collect finite-shell counterpoise diagnostics for diamond C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HARTREE_TO_EV = 27.211386245988
_DIAMOND_BASIS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _diamond_shells(lattice_constant_angstrom: float, shell_count: int) -> list[list[dict]]:
    if not math.isfinite(lattice_constant_angstrom) or lattice_constant_angstrom <= 0.0:
        raise ValueError("lattice constant must be positive and finite")
    if type(shell_count) is not int or shell_count <= 0:
        raise ValueError("shell count must be a positive integer")

    extent = shell_count + 2
    points: list[dict] = []
    seen: set[tuple[float, float, float]] = set()
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for basis in _DIAMOND_BASIS:
                    conventional = (i + basis[0], j + basis[1], k + basis[2])
                    if all(abs(value) <= 1.0e-14 for value in conventional):
                        continue
                    cartesian = tuple(lattice_constant_angstrom * value for value in conventional)
                    rounded = tuple(round(value, 12) for value in cartesian)
                    if rounded in seen:
                        continue
                    seen.add(rounded)
                    distance = math.sqrt(sum(value * value for value in cartesian))
                    points.append(
                        {
                            "cartesian_angstrom": list(cartesian),
                            "distance_angstrom": distance,
                        }
                    )
    points.sort(key=lambda row: (row["distance_angstrom"], row["cartesian_angstrom"]))

    shells: list[list[dict]] = []
    for point in points:
        if not shells or abs(point["distance_angstrom"] - shells[-1][0]["distance_angstrom"]) > 1.0e-9:
            shells.append([])
        shells[-1].append(point)
        if len(shells) > shell_count:
            break
    if len(shells) < shell_count:
        raise RuntimeError("failed to generate requested diamond shells")
    return shells[:shell_count]


def build_ghost_cluster(
    *,
    lattice_constant_angstrom: float,
    box_angstrom: float,
    shell_count: int,
) -> dict:
    if not math.isfinite(box_angstrom) or box_angstrom <= 0.0:
        raise ValueError("box size must be positive and finite")
    shells = _diamond_shells(lattice_constant_angstrom, shell_count)
    ghosts = []
    for shell_index, shell in enumerate(shells, start=1):
        for point in shell:
            fractional = [0.5 + value / box_angstrom for value in point["cartesian_angstrom"]]
            if any(value <= 0.0 or value >= 1.0 for value in fractional):
                raise ValueError("ghost cluster does not fit inside the requested box")
            ghosts.append(
                {
                    **point,
                    "fractional": fractional,
                    "shell_index": shell_index,
                }
            )
    return {
        "status": "success",
        "quantity": "diamond_c_finite_shell_counterpoise_cluster",
        "lattice_constant_angstrom": lattice_constant_angstrom,
        "box_angstrom": box_angstrom,
        "shell_count": shell_count,
        "shell_populations": [len(shell) for shell in shells],
        "shell_distances_angstrom": [shell[0]["distance_angstrom"] for shell in shells],
        "ghost_count": len(ghosts),
        "real_atom_fractional": [0.5, 0.5, 0.5],
        "ghosts": ghosts,
    }


def render_stru(cluster: dict, *, orbital_filename: str, pseudopotential_filename: str) -> str:
    if not orbital_filename or not pseudopotential_filename:
        raise ValueError("orbital and pseudopotential filenames are required")
    lines = [
        "ATOMIC_SPECIES",
        f"C 12.011 {pseudopotential_filename}",
        f"C_empty 12.011 {pseudopotential_filename}",
        "",
        "NUMERICAL_ORBITAL",
        orbital_filename,
        orbital_filename,
        "",
        "LATTICE_CONSTANT",
        "1.8897261254578281",
        "",
        "LATTICE_VECTORS",
        f'{cluster["box_angstrom"]:.12f} 0.0 0.0',
        f'0.0 {cluster["box_angstrom"]:.12f} 0.0',
        f'0.0 0.0 {cluster["box_angstrom"]:.12f}',
        "",
        "ATOMIC_POSITIONS",
        "Direct",
        "",
        "C",
        "0.0",
        "1",
        "0.5 0.5 0.5 0 0 0 mag 2.0",
        "",
        "C_empty",
        "0.0",
        str(cluster["ghost_count"]),
    ]
    for ghost in cluster["ghosts"]:
        lines.append("{:.15f} {:.15f} {:.15f} 0 0 0".format(*ghost["fractional"]))
    return "\n".join(lines) + "\n"


def collect_counterpoise(raw: dict, ghost: dict) -> dict:
    if raw.get("selected_orbital_sha256") != ghost.get("selected_orbital_sha256"):
        raise ValueError("raw and ghost orbital hash mismatch")
    required_raw = (
        "atom_zero_order_ha",
        "atom_ecrpa_ha",
        "zero_order_binding_ev_per_c",
        "correlation_binding_ev_per_c",
        "sos_total_binding_ev_per_c",
        "delta_st_reference_ev_per_c",
    )
    required_ghost = ("reference_ha", "ecrpa_ha", "shell_count", "ghost_count")
    values = [float(raw[key]) for key in required_raw] + [float(ghost[key]) for key in required_ghost[:2]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("counterpoise inputs must be finite")

    zero_correction = (float(ghost["reference_ha"]) - float(raw["atom_zero_order_ha"])) * HARTREE_TO_EV
    correlation_correction = (float(ghost["ecrpa_ha"]) - float(raw["atom_ecrpa_ha"])) * HARTREE_TO_EV
    total_correction = zero_correction + correlation_correction
    zero_cp = float(raw["zero_order_binding_ev_per_c"]) + zero_correction
    correlation_cp = float(raw["correlation_binding_ev_per_c"]) + correlation_correction
    total_cp = float(raw["sos_total_binding_ev_per_c"]) + total_correction
    reference = float(raw["delta_st_reference_ev_per_c"])
    difference = total_cp - reference
    return {
        "status": "success",
        "quantity": "c_atom_diamond_ordinary_sos_counterpoise_binding_energy",
        "scope": "first_shell_counterpoise_diagnostic_requires_shell_convergence"
        if int(ghost["shell_count"]) == 1
        else "finite_shell_counterpoise_diagnostic_requires_shell_convergence",
        "shell_count": int(ghost["shell_count"]),
        "ghost_count": int(ghost["ghost_count"]),
        "selected_orbital_sha256": raw["selected_orbital_sha256"],
        "zero_order_binding_raw_ev_per_c": float(raw["zero_order_binding_ev_per_c"]),
        "correlation_binding_raw_ev_per_c": float(raw["correlation_binding_ev_per_c"]),
        "sos_total_binding_raw_ev_per_c": float(raw["sos_total_binding_ev_per_c"]),
        "zero_order_binding_cp_ev_per_c": zero_cp,
        "correlation_binding_cp_ev_per_c": correlation_cp,
        "sos_total_binding_cp_ev_per_c": total_cp,
        "bsse_zero_order_ev_per_c": -zero_correction,
        "bsse_correlation_ev_per_c": -correlation_correction,
        "bsse_total_ev_per_c": -total_correction,
        "delta_st_reference_ev_per_c": reference,
        "difference_cp_from_delta_ev_per_c": difference,
        "absolute_difference_cp_from_delta_ev_per_c": abs(difference),
        "counterpoise_acceptance": "requires_shell_convergence",
    }


def _prepare(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    cluster = build_ghost_cluster(
        lattice_constant_angstrom=args.lattice_constant_angstrom,
        box_angstrom=args.box_angstrom,
        shell_count=args.shell_count,
    )
    stru = output_root / "STRU"
    stru.write_text(
        render_stru(
            cluster,
            orbital_filename=args.orbital_filename,
            pseudopotential_filename=args.pseudopotential_filename,
        ),
        encoding="ascii",
    )
    cluster["stru_sha256"] = _sha256(stru)
    (output_root / "CLUSTER.json").write_text(
        json.dumps(cluster, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def _collect(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    raw = json.loads(args.raw_binding.read_text(encoding="ascii"))
    ghost = json.loads(args.ghost_summary.read_text(encoding="ascii"))
    result = collect_counterpoise(raw, ghost)
    result["raw_binding_path"] = str(args.raw_binding.resolve())
    result["raw_binding_sha256"] = _sha256(args.raw_binding)
    result["ghost_summary_path"] = str(args.ghost_summary.resolve())
    result["ghost_summary_sha256"] = _sha256(args.ghost_summary)
    (output_root / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (output_root / "STATUS").write_text("success\n", encoding="ascii")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--orbital-filename", required=True)
    prepare.add_argument("--pseudopotential-filename", default="C_ONCV_PBE-1.0.upf")
    prepare.add_argument("--lattice-constant-angstrom", type=float, default=3.6)
    prepare.add_argument("--box-angstrom", type=float, default=20.0)
    prepare.add_argument("--shell-count", type=int, default=1)
    prepare.set_defaults(function=_prepare)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--raw-binding", required=True, type=Path)
    collect.add_argument("--ghost-summary", required=True, type=Path)
    collect.add_argument("--output-root", required=True, type=Path)
    collect.set_defaults(function=_collect)

    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
