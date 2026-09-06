"""Export only eight bounded, cheap-screened actual-PBE diagnostic probes."""

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]/"opt_orb_pytorch_dpsi"))

import torch

from check_c_optimized_pbe import _optimizer_artifacts, _read_hashed, _sha256, _write_json, BACKOFF_SCOPE
from diagnose_c_radial_gradients import validate_pbe_binding
from export_periodic_orbitals import write_abacus_orbital
from optimize_c_all_radial_fast import load_frozen_c
from periodic_galerkin_basis import read_periodic_optimizer_coefficients, write_periodic_optimizer_coefficients
from periodic_galerkin_direction_calibration import build_directions, signed_probes
from periodic_galerkin_fit import CandidateGuardError, _minimum_occupied_capture


def export_probes(coefficients, artifact, *, center, center_result_sha256, output, screen):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/"DIRECTIONS.json", artifact)
    digest = _sha256((output/"DIRECTIONS.json").read_bytes())
    rows = []
    with torch.no_grad():
        for index, trial in enumerate(signed_probes(coefficients, artifact)):
            destination = output/("probe_%02d" % index)
            destination.mkdir()
            try:
                gate = screen(trial["coefficients"])
                if gate.get("gate") is not True:
                    raise CandidateGuardError("cheap screen rejected probe")
            except CandidateGuardError as error:
                gate = dict(gate=False, reason=str(error))
            probe = dict(status="prepared" if gate["gate"] else "rejected",
                scope="direction_calibration_probe", direction_name=trial["direction_name"],
                signed_radius=trial["signed_radius"], center_result_sha256=center_result_sha256,
                center_coefficient_sha256=center["coefficient_sha256"],
                center_orbital_sha256=center["orbital_sha256"], directions_sha256=digest,
                cheap_gate=gate, galerkin_energy="unmeasured", physical_release_gate="hold")
            if gate["gate"]:
                for filename, writer in (("COEFFICIENTS.txt", write_periodic_optimizer_coefficients),
                                         ("C_3s3p2d_probe.orb", None)):
                    if writer is not None:
                        writer(destination/filename, trial["coefficients"])
                    else:
                        write_abacus_orbital(destination/filename, trial["coefficients"], element="C",
                            ecut_ry=100., rcut_bohr=10., dr_bohr=.01, smoothing_sigma_bohr=.1)
                probe.update(coefficient_filename="COEFFICIENTS.txt", orbital_filename="C_3s3p2d_probe.orb",
                    coefficient_sha256=_sha256((destination/"COEFFICIENTS.txt").read_bytes()),
                    orbital_sha256=_sha256((destination/"C_3s3p2d_probe.orb").read_bytes()))
            _write_json(destination/"PROBE.json", probe)
            rows.append(dict(path=str((destination/"PROBE.json").relative_to(output)), status=probe["status"],
                sha256=_sha256((destination/"PROBE.json").read_bytes()),
                direction_name=trial["direction_name"], signed_radius=trial["signed_radius"]))
    return rows


def prepare_calibration(*, center_result, center_result_sha256, pbe_collection,
                        pbe_collection_sha256, diagnostic, diagnostic_sha256, output, source_commit):
    started = time.perf_counter()
    output, center_path = Path(output), Path(center_result)
    if output.exists():
        raise FileExistsError(output)
    center, _, _ = _optimizer_artifacts(center_path, center_result_sha256)
    if center.get("scope") != BACKOFF_SCOPE or center.get("alpha") != .25:
        raise ValueError("only the accepted quarter center is eligible")
    pbe_bytes = _read_hashed(Path(pbe_collection), pbe_collection_sha256)
    pbe = json.loads(pbe_bytes)
    validate_pbe_binding(center, center_result_sha256, pbe)
    _read_hashed(Path(pbe["candidate_log_path"]), pbe["candidate_log_sha256"])
    diagnostic_bytes = _read_hashed(Path(diagnostic), diagnostic_sha256)
    d = json.loads(diagnostic_bytes)
    expected = dict(candidate_result_sha256=center_result_sha256, pbe_collection_sha256=pbe_collection_sha256,
        status="success", nu=[3, 3, 2, 0, 0], fixed_nu=[0]*5, optimizer_steps=0)
    expected.update({key: center[key] for key in ("coefficient_sha256", "orbital_sha256", "freeze_sha256", "active_cache_index_sha256")})
    if any(d.get(key) != value for key, value in expected.items()):
        raise ValueError("diagnostic does not bind the accepted center and frozen inputs")
    root = center_path.parent
    coefficients = read_periodic_optimizer_coefficients(root/"INTERPOLATED_COEFFICIENTS.txt",
        element="C", radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
    original = read_periodic_optimizer_coefficients(root/"ORIGINAL_COEFFICIENTS.txt",
        element="C", radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
    artifact = build_directions(coefficients, d["energy_gradient"], d["guard_sensitivity"]["energy_gradient"])
    artifact.update(source_commit=source_commit, diagnostic_sha256=diagnostic_sha256,
                    center_result_sha256=center_result_sha256, center_coefficient_sha256=center["coefficient_sha256"],
                    center_orbital_sha256=center["orbital_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(center["configuration"]["threads"])
    datasets, records, guard = load_frozen_c(root/"INPUT_FREEZE.json", center["freeze_sha256"], original,
        output, active_cache_index=root/"ACTIVE_DATA_CACHE.json",
        active_cache_index_sha256=center["active_cache_index_sha256"])
    loaded = time.perf_counter()
    floor = max(0., center["initial"]["minimum_occupied_capture"]-1e-4)
    def screen(c):
        bands = guard(c)
        capture = _minimum_occupied_capture(datasets, c, relative_rank_tolerance=1e-12, condition_limit=1e12)
        if not math.isfinite(capture) or capture < floor:
            raise CandidateGuardError("original occupied capture floor failed")
        return dict(gate=True, band_screen=bands, minimum_occupied_capture=capture,
            occupied_capture_floor=floor, overlap_rank_and_condition_gate="pass",
            overlap_relative_rank_tolerance=1e-12, overlap_condition_limit=1e12)
    rows = export_probes(coefficients, artifact, center=center, center_result_sha256=center_result_sha256,
                         output=output/"probes", screen=screen)
    for name, content in (("DIAGNOSTIC_INPUT.json", diagnostic_bytes), ("CENTER_PBE_COLLECTION.json", pbe_bytes)):
        with (output/name).open("xb") as stream:
            stream.write(content)
    _optimizer_artifacts(center_path, center_result_sha256)
    _read_hashed(Path(diagnostic), diagnostic_sha256)
    _read_hashed(Path(pbe_collection), pbe_collection_sha256)
    _read_hashed(Path(pbe["candidate_log_path"]), pbe["candidate_log_sha256"])
    result = dict(status="success", scope="bounded_pbe_probe_preparation", source_commit=source_commit,
        center_result_sha256=center_result_sha256, center_result_path=str(center_path.resolve()),
        center_coefficient_sha256=center["coefficient_sha256"], center_orbital_sha256=center["orbital_sha256"],
        diagnostic_sha256=diagnostic_sha256, pbe_collection_sha256=pbe_collection_sha256,
        directions_sha256=_sha256((output/"probes/DIRECTIONS.json").read_bytes()),
        probes=rows, cache_load_seconds=loaded-started, total_seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, load_records=records,
        rpa_evaluations=0, optimizer_steps=0, physical_release_gate="hold")
    _write_json(output/"CALIBRATION_PREPARED.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("center-result", "center-result-sha256", "pbe-collection", "pbe-collection-sha256",
                "diagnostic", "diagnostic-sha256", "output", "source-commit"):
        parser.add_argument("--"+key, required=True)
    result = prepare_calibration(**vars(parser.parse_args()))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
