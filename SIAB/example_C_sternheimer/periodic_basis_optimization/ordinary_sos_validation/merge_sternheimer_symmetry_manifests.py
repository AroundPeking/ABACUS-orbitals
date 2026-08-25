#!/usr/bin/env python3
"""Merge per-q ABACUS symmetry fragments into LibRPA input manifests."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QPoint:
    iq: int
    qx: float
    qy: float
    qz: float
    weight: float


@dataclass(frozen=True)
class PartialResponse:
    iq: int
    ik_full: int
    ifreq: int
    path: Path


@dataclass(frozen=True)
class FixedQRoute:
    iq: int
    representative_ik: int
    member_ik: int
    spatial_isym: int
    time_reversal: int
    fold_gx: int
    fold_gy: int
    fold_gz: int


@dataclass(frozen=True)
class QStarRoute:
    representative_iq: int
    member_iq: int
    spatial_isym: int
    time_reversal: int
    fold_gx: int
    fold_gy: int
    fold_gz: int


def _data_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].split("!", 1)[0].strip()
        if line:
            rows.append(line.split())
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def read_qpoint_fragment(path: Path) -> QPoint:
    rows = _data_rows(path)
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError(f"{path}: expected one row with iq qx qy qz qweight")
    iq = int(rows[0][0])
    values = [float(value) for value in rows[0][1:]]
    if iq <= 0 or not all(math.isfinite(value) for value in values) or values[3] <= 0.0:
        raise ValueError(f"{path}: invalid q-point row")
    return QPoint(iq, values[0], values[1], values[2], values[3])


def read_partial_fragment(path: Path) -> list[PartialResponse]:
    records: list[PartialResponse] = []
    for row in _data_rows(path):
        if len(row) != 4:
            raise ValueError(f"{path}: expected iq ik_full ifreq response_file")
        iq, ik_full, ifreq = (int(value) for value in row[:3])
        response = (path.parent / row[3]).resolve()
        if iq <= 0 or ik_full < 0 or ifreq <= 0 or not response.is_file():
            raise ValueError(f"{path}: invalid or missing partial response in row {' '.join(row)}")
        records.append(PartialResponse(iq, ik_full, ifreq, response))
    return records


def read_route_fragment(path: Path) -> list[FixedQRoute]:
    rows = _data_rows(path)
    if rows[0] != ["version", "1"]:
        raise ValueError(f"{path}: expected version 1")
    routes: list[FixedQRoute] = []
    seen_members: set[tuple[int, int]] = set()
    for row in rows[1:]:
        if len(row) != 8:
            raise ValueError(
                f"{path}: expected iq representative_ik member_ik spatial_isym "
                "time_reversal fold_Gx fold_Gy fold_Gz"
            )
        values = [int(value) for value in row]
        route = FixedQRoute(*values)
        if (
            route.iq <= 0
            or route.representative_ik < 0
            or route.member_ik < 0
            or route.spatial_isym < 0
            or route.time_reversal not in (0, 1)
        ):
            raise ValueError(f"{path}: invalid fixed-q route row {' '.join(row)}")
        key = (route.iq, route.member_ik)
        if key in seen_members:
            raise ValueError(f"{path}: duplicate fixed-q route member {key}")
        seen_members.add(key)
        routes.append(route)
    if not routes:
        raise ValueError(f"{path}: no fixed-q routes")
    return routes


def read_qstar_route_fragment(path: Path) -> list[QStarRoute]:
    rows = _data_rows(path)
    if rows[0] != ["version", "1"]:
        raise ValueError(f"{path}: expected version 1")
    routes: list[QStarRoute] = []
    seen_members: set[int] = set()
    for row in rows[1:]:
        if len(row) != 7:
            raise ValueError(
                f"{path}: expected representative_iq member_iq spatial_isym "
                "time_reversal fold_Gx fold_Gy fold_Gz"
            )
        values = [int(value) for value in row]
        route = QStarRoute(*values)
        if (
            route.representative_iq <= 0
            or route.member_iq <= 0
            or route.spatial_isym < 0
            or route.time_reversal not in (0, 1)
        ):
            raise ValueError(f"{path}: invalid q-star route row {' '.join(row)}")
        if route.member_iq in seen_members:
            raise ValueError(f"{path}: duplicate q-star route member {route.member_iq}")
        seen_members.add(route.member_iq)
        routes.append(route)
    if not routes:
        raise ValueError(f"{path}: no q-star routes")
    return routes


def read_full_kpoint_fragment(path: Path) -> tuple[tuple[int, float, float, float], ...]:
    points: dict[int, tuple[float, float, float]] = {}
    for row in _data_rows(path):
        if len(row) != 4:
            raise ValueError(f"{path}: expected ik_full kx ky kz")
        ik_full = int(row[0])
        values = tuple(float(value) for value in row[1:])
        if ik_full < 0 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}: invalid full-k-point row {' '.join(row)}")
        if ik_full in points:
            raise ValueError(f"{path}: duplicate ik_full={ik_full}")
        points[ik_full] = values
    if sorted(points) != list(range(len(points))):
        raise ValueError(f"{path}: full-k-point indices must be contiguous from zero")
    return tuple((ik_full, *points[ik_full]) for ik_full in range(len(points)))


def merge_manifests(run_directories: list[Path], output_directory: Path) -> tuple[Path, Path]:
    if not run_directories:
        raise ValueError("at least one ABACUS q-run directory is required")
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    qpoints: dict[int, QPoint] = {}
    partials: dict[tuple[int, int, int], PartialResponse] = {}
    routes: dict[tuple[int, int], FixedQRoute] = {}
    qstar_routes: dict[int, QStarRoute] = {}
    representatives_by_q_frequency: dict[tuple[int, int], set[int]] = {}
    full_kpoints: tuple[tuple[int, float, float, float], ...] | None = None
    for run_directory in run_directories:
        run_directory = run_directory.resolve()
        q_fragments = sorted(run_directory.glob("v1_sternheimer_qpoint_iq_*.dat"))
        partial_fragments = sorted(run_directory.glob("v1_sternheimer_partial_manifest_iq_*.dat"))
        route_fragments = sorted(run_directory.glob("v1_sternheimer_symmetry_routes_iq_*.dat"))
        qstar_route_fragments = sorted(
            run_directory.glob("v1_sternheimer_qstar_routes_iq_*.dat")
        )
        full_kpoint_fragments = sorted(run_directory.glob("v1_sternheimer_full_kpoints.dat"))
        if (len(q_fragments) != 1 or len(partial_fragments) != 1
                or len(route_fragments) != 1
                or len(qstar_route_fragments) != 1
                or len(full_kpoint_fragments) != 1):
            raise ValueError(
                f"{run_directory}: expected exactly one q fragment, partial manifest, "
                "fixed-q route fragment, q-star route fragment, and full-k-point manifest"
            )

        run_full_kpoints = read_full_kpoint_fragment(full_kpoint_fragments[0])
        if full_kpoints is None:
            full_kpoints = run_full_kpoints
        elif full_kpoints != run_full_kpoints:
            raise ValueError(f"{run_directory}: full-k-point manifest disagrees with other q runs")

        qpoint = read_qpoint_fragment(q_fragments[0])
        if qpoint.iq in qpoints:
            raise ValueError(f"duplicate q-point iq={qpoint.iq}")
        qpoints[qpoint.iq] = qpoint
        records = read_partial_fragment(partial_fragments[0])
        if not records or any(record.iq != qpoint.iq for record in records):
            raise ValueError(f"{partial_fragments[0]}: partial iq does not match q fragment")
        for record in records:
            key = (record.iq, record.ik_full, record.ifreq)
            if key in partials:
                raise ValueError(f"duplicate partial response key={key}")
            partials[key] = record
            representatives_by_q_frequency.setdefault((record.iq, record.ifreq), set()).add(
                record.ik_full
            )
        run_routes = read_route_fragment(route_fragments[0])
        if any(route.iq != qpoint.iq for route in run_routes):
            raise ValueError(f"{route_fragments[0]}: route iq does not match q fragment")
        expected_members = set(range(len(run_full_kpoints)))
        actual_members = {route.member_ik for route in run_routes}
        if actual_members != expected_members:
            raise ValueError(f"{route_fragments[0]}: routes do not cover the full k grid")
        route_representatives = {route.representative_ik for route in run_routes}
        response_representatives = {record.ik_full for record in records}
        if route_representatives != response_representatives:
            raise ValueError(
                f"{route_fragments[0]}: route representatives disagree with partial responses"
            )
        for route in run_routes:
            if route.representative_ik not in expected_members:
                raise ValueError(f"{route_fragments[0]}: route representative leaves the full k grid")
            key = (route.iq, route.member_ik)
            if key in routes:
                raise ValueError(f"duplicate fixed-q route key={key}")
            routes[key] = route

        run_qstar_routes = read_qstar_route_fragment(qstar_route_fragments[0])
        if any(route.representative_iq != qpoint.iq for route in run_qstar_routes):
            raise ValueError(
                f"{qstar_route_fragments[0]}: q-star representative does not match q fragment"
            )
        for route in run_qstar_routes:
            if route.member_iq in qstar_routes:
                raise ValueError(f"duplicate q-star route member {route.member_iq}")
            qstar_routes[route.member_iq] = route

    if abs(sum(point.weight for point in qpoints.values()) - 1.0) > 1.0e-10:
        raise ValueError("combined q-point weights do not sum to one")
    assert full_kpoints is not None
    expected_q_members = set(range(1, len(full_kpoints) + 1))
    if set(qstar_routes) != expected_q_members:
        raise ValueError("combined q-star routes do not cover the full q grid")
    if {route.representative_iq for route in qstar_routes.values()} != set(qpoints):
        raise ValueError("q-star route representatives disagree with q-point manifest")
    qstar_sizes = {
        iq: sum(route.representative_iq == iq for route in qstar_routes.values())
        for iq in qpoints
    }
    for iq, point in qpoints.items():
        expected_weight = qstar_sizes[iq] / len(full_kpoints)
        if abs(point.weight - expected_weight) > 1.0e-10:
            raise ValueError(f"iq={iq}: q-point weight disagrees with discrete q-star size")
    frequency_sets = {
        iq: {ifreq for q_iq, ifreq in representatives_by_q_frequency if q_iq == iq}
        for iq in qpoints
    }
    if len({tuple(sorted(values)) for values in frequency_sets.values()}) != 1:
        raise ValueError("representative q points do not contain the same frequency indices")
    for iq, frequencies in frequency_sets.items():
        reference: set[int] | None = None
        for ifreq in sorted(frequencies):
            representatives = representatives_by_q_frequency[(iq, ifreq)]
            if reference is None:
                reference = representatives
            elif representatives != reference:
                raise ValueError(f"iq={iq}: k representatives differ between frequencies")

    qpoint_manifest = output_directory / "v1_sternheimer_qpoints.dat"
    partial_manifest = output_directory / "v1_sternheimer_partial_manifest.dat"
    full_kpoint_manifest = output_directory / "v1_sternheimer_full_kpoints.dat"
    route_manifest = output_directory / "v1_sternheimer_symmetry_routes.dat"
    qstar_route_manifest = output_directory / "v1_sternheimer_qstar_routes.dat"
    relative_response_paths: dict[tuple[int, int, int], str] = {}
    for key, record in partials.items():
        relative_path = os.path.relpath(record.path, output_directory)
        if any(character.isspace() for character in relative_path):
            raise ValueError(
                "response path cannot be represented in a whitespace-delimited manifest: "
                f"{record.path}"
            )
        relative_response_paths[key] = relative_path
    qpoint_text = "# iq qx qy qz qweight\n" + "".join(
        f"{point.iq} {point.qx:.17g} {point.qy:.17g} {point.qz:.17g} {point.weight:.17g}\n"
        for point in sorted(qpoints.values(), key=lambda item: item.iq)
    )
    partial_text = "# iq ik_full ifreq response_file\n" + "".join(
        f"{record.iq} {record.ik_full} {record.ifreq} "
        f"{relative_response_paths[(record.iq, record.ik_full, record.ifreq)]}\n"
        for record in sorted(partials.values(), key=lambda item: (item.iq, item.ik_full, item.ifreq))
    )
    qpoint_manifest.write_text(qpoint_text, encoding="utf-8")
    partial_manifest.write_text(partial_text, encoding="utf-8")
    full_kpoint_text = "# ik_full kx ky kz\n" + "".join(
        f"{ik_full} {kx:.17g} {ky:.17g} {kz:.17g}\n"
        for ik_full, kx, ky, kz in full_kpoints
    )
    full_kpoint_manifest.write_text(full_kpoint_text, encoding="utf-8")
    route_text = (
        "version 1\n"
        "# iq representative_ik member_ik spatial_isym time_reversal fold_Gx fold_Gy fold_Gz\n"
        + "".join(
            f"{route.iq} {route.representative_ik} {route.member_ik} "
            f"{route.spatial_isym} {route.time_reversal} "
            f"{route.fold_gx} {route.fold_gy} {route.fold_gz}\n"
            for route in sorted(routes.values(), key=lambda item: (item.iq, item.member_ik))
        )
    )
    route_manifest.write_text(route_text, encoding="utf-8")
    qstar_route_text = (
        "version 1\n"
        "# representative_iq member_iq spatial_isym time_reversal fold_Gx fold_Gy fold_Gz\n"
        + "".join(
            f"{route.representative_iq} {route.member_iq} "
            f"{route.spatial_isym} {route.time_reversal} "
            f"{route.fold_gx} {route.fold_gy} {route.fold_gz}\n"
            for route in sorted(qstar_routes.values(), key=lambda item: item.member_iq)
        )
    )
    qstar_route_manifest.write_text(qstar_route_text, encoding="utf-8")
    return qpoint_manifest, partial_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    qpoints, partials = merge_manifests(args.run_dirs, args.output_dir)
    print(qpoints)
    print(partials)


if __name__ == "__main__":
    main()
