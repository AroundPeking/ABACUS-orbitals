"""One Ec gradient at a strictly accepted combined endpoint; no new candidates.

CLI pins the accepted RESULT, acceptance, new DEPLOYMENT and new source commit.
DEPLOYMENT uses source_directory/source_commit/files/archive_path/archive_sha256.
The stage must be separate from the accepted stage; result/ is exclusively
reserved. All SIAB Python sources and this batch script must be hash-listed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import sys
import time
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SOURCE / "SIAB/opt_orb_pytorch_dpsi"),
               str(SOURCE / "SIAB/example_C_sternheimer/periodic_basis_optimization")]

import check_c_accepted_combined_step as accepted
from check_c_accepted_combined_step import load_accepted_combined_step
import check_c_optimized_pbe as endpoint
from check_c_direction_probe_pbe import _digest, _equal, _json
from check_c_combined_step_pbe import _number, _root


_WORKFLOW = "SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/"
REQUIRED_SOURCE_FILES = tuple(_WORKFLOW + name for name in (
    "refresh_c_combined_step_gradient.py", "run_c_combined_step_gradient.slurm",
    "check_c_accepted_combined_step.py", "check_c_optimized_pbe.py",
    "check_c_combined_step_pbe.py", "check_c_direction_probe_pbe.py",
    "optimize_c_all_radial_fast.py")) + tuple("SIAB/opt_orb_pytorch_dpsi/" + name for name in (
    "periodic_galerkin_basis.py", "periodic_galerkin_combined_step.py",
    "periodic_galerkin_direction_calibration.py", "periodic_galerkin_radial_diagnostics.py"))
DIAGNOSTIC = "EC_GRADIENT_REFRESH.json"


def _check_file(path, digest):
    expected = _digest(digest)
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(chunk)
    _equal(checksum.hexdigest(), expected, "SHA256 " + str(path))


def verify_source(stage, deployment_sha256, source_commit):
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("exact new source commit required")
    manifest = _json(accepted._small(stage, "DEPLOYMENT.json", _digest(deployment_sha256)))
    _equal(manifest.get("source_commit"), source_commit, "new deployment source commit")
    source = _root(manifest["source_directory"])
    _equal(source, SOURCE, "running source versus immutable deployment")
    files = manifest.get("files")
    required = set(REQUIRED_SOURCE_FILES)
    required.update(str(path.relative_to(source)) for path in (source / "SIAB").rglob("*.py"))
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("complete SIAB Python and gradient batch source hashes required")
    for name, digest in files.items():
        _check_file(endpoint._within(source, name), digest)
    _check_file(Path(manifest["archive_path"]), manifest.get("archive_sha256"))
    return manifest


def _runtime():
    # Keep native numerical imports out of the CLI parser and mock unit tests.
    import torch
    from optimize_c_all_radial_fast import load_frozen_c
    from periodic_galerkin_basis import read_periodic_optimizer_coefficients
    from periodic_galerkin_combined_step import combine_pbe_tangent
    from periodic_galerkin_radial_diagnostics import evaluate_radial_gradients, transported_descent_report
    return SimpleNamespace(torch=torch, load_frozen_c=load_frozen_c,
        read_coefficients=read_periodic_optimizer_coefficients,
        combine_pbe_tangent=combine_pbe_tangent, evaluate_radial_gradients=evaluate_radial_gradients,
        transported_descent_report=transported_descent_report)


def _read_coefficients(runtime, path):
    return runtime.read_coefficients(path, element="C", radial_rows=31, expected_nu=(3, 3, 2, 0, 0))


def _old_direction(runtime, loaded, coefficients):
    root = loaded["coefficient_path"].parent
    directions, calibration = (_json(accepted._small(root, filename, loaded["result"][identity]))
        for filename, identity in (("DIRECTIONS.json", "directions_sha256"),
                                    ("CALIBRATION.json", "calibration_sha256")))
    previous_path = loaded["freeze_path"].parent / "INTERPOLATED_COEFFICIENTS.txt"
    _check_file(previous_path, loaded["quarter"]["coefficient_sha256"])
    previous = _read_coefficients(runtime, previous_path)
    with runtime.torch.no_grad():
        reconstructed = runtime.combine_pbe_tangent(previous, directions, calibration)
    accepted._expect(reconstructed, dict(scope="actual_pbe_tangent_candidate",
                                       direction_name="actual_pbe_tangent"), "old direction reconstruction")
    for key in ("radius", "mixing_ratio"):
        _equal(reconstructed.get(key), loaded["candidate"][key], "old direction " + key)
    recovered = reconstructed.get("coefficients")
    if (not isinstance(recovered, dict) or set(recovered) != set(coefficients)
            or any(len(recovered[e]) != len(channels) or
                   any(not runtime.torch.equal(a, b) for a, b in zip(recovered[e], channels))
                   for e, channels in coefficients.items())):
        raise ValueError("old direction does not reconstruct the exact accepted coefficients")
    return reconstructed["direction"]


def validate_dataset_extent(datasets, loaded):
    """The frozen loader validates identities; do not truncate its full result."""
    if len(datasets) != 8:
        raise ValueError("one complete eight-q dataset set required")
    for dataset, expected in zip(datasets, loaded["result"]["candidate"]["rpa"]["per_q"]):
        if (dataset.selected_iq != expected["selected_iq"] or dataset.q_weight != expected["q_weight"]
                or dataset.q_count != 64 or len(dataset.kpoints) != 64
                or dataset.frequency_ha.numel() != 12
                or dataset.frequency_ha.tolist() != expected["frequency_ha"]):
            raise ValueError("full eight-q/64-k/twelve-frequency extent changed")


def validate_reproduction(loaded, diagnostic, band_screen):
    accepted._expect(diagnostic, dict(scope="radial_gradients_at_fixed_candidate_no_optimization",
        gradient_mode="energy_only", backward_passes=1, physical_release_gate="hold",
        energy_gradient_units="Ha_per_cell_per_unit_coefficient"), "Ec-only kernel")
    if "loss_gradient" in diagnostic or not isinstance(diagnostic.get("energy_gradient"), dict):
        raise ValueError("exactly one energy gradient and no loss gradient required")
    expected = loaded["result"]["candidate"]
    # Adapt only the reporting shape to reuse the accepted full-q/guard checker.
    measured = dict(diagnostic, coefficient_guard=band_screen, energy_quantity=expected["energy_quantity"])
    for prefix, key in (("rpa", "candidate_energy_ha"), ("reference_rpa", "reference_energy_ha")):
        energy = _number(diagnostic["rpa"].get(key), key)*endpoint.HARTREE_TO_EV
        for suffix, divisor in (("cell", 1), ("c", 2)):
            measured[prefix+"_correlation_energy_ev_per_"+suffix] = energy/divisor
    accepted._record(measured, loaded["occupied_capture_floor"], loaded["quarter"]["training_weights"])
    endpoint._match_parent_initial(measured, expected)
    accepted._same_grid_reference(measured, expected, reproduce=True)
    for name in ("pi", "trace_log", "energy"):
        key = name+"_relative_squared_error"
        accepted._near(measured["rpa"][key], expected["rpa"][key], "reproduced " + key)
    for row, previous in zip(measured["rpa"]["per_q"], expected["rpa"]["per_q"]):
        _equal(row["q_weight"], previous["q_weight"], "reproduced q weight")
    for key in ("minimum_occupied_capture", "maximum_overlap_condition"):
        if not math.isclose(_number(measured.get(key), key), _number(expected.get(key), key),
                            rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("accepted endpoint reproduction failed: " + key)


def refresh_gradient(*, accepted_stage, result_sha256, acceptance_sha256,
                     stage, deployment_sha256, source_commit):
    started = time.perf_counter()
    stage, accepted_stage = _root(stage), _root(accepted_stage)
    if stage == accepted_stage or accepted_stage in stage.parents or stage in accepted_stage.parents:
        raise ValueError("new diagnostic stage must be separate from accepted evidence")
    output = stage / "result"
    for path in (output, stage / "PROVENANCE.json"):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    verify_source(stage, deployment_sha256, source_commit)
    loaded = load_accepted_combined_step(accepted_stage, result_sha256, acceptance_sha256)
    if source_commit == loaded["result"]["source_commit"]:
        raise ValueError("new gradient source must not claim the accepted old source commit")
    runtime = _runtime()
    runtime.torch.set_num_threads(28)
    coefficients = _read_coefficients(runtime, loaded["coefficient_path"])
    original = _read_coefficients(runtime, loaded["original_coefficient_path"])
    previous_direction = _old_direction(runtime, loaded, coefficients)
    output.mkdir()
    load_started = time.perf_counter()
    datasets, records, guard = runtime.load_frozen_c(
        loaded["freeze_path"], loaded["result"]["freeze_sha256"], original, output,
        active_cache_index=loaded["active_cache_index_path"],
        active_cache_index_sha256=loaded["result"]["active_cache_index_sha256"])
    cache_seconds = time.perf_counter()-load_started
    validate_dataset_extent(datasets, loaded)
    with runtime.torch.no_grad():
        band_screen = guard(coefficients)
    accepted._band_guard(band_screen)
    diagnostic = runtime.evaluate_radial_gradients(datasets, coefficients,
        occupied_capture_tolerance=1-loaded["occupied_capture_floor"],
        frequency_batch_size=loaded["quarter"]["configuration"]["frequency_batch_size"],
        weights=loaded["quarter"]["training_weights"], energy_only=True)
    validate_reproduction(loaded, diagnostic, band_screen)
    transport = runtime.transported_descent_report(coefficients, diagnostic["energy_gradient"], previous_direction)
    if not isinstance(transport, dict):
        raise ValueError("transported descent report object required")
    result = dict(diagnostic, scope="accepted_combined_step_ec_gradient_refresh", status="success",
        source_commit=source_commit, deployment_sha256=deployment_sha256,
        accepted_stage=str(accepted_stage), accepted_result_sha256=result_sha256,
        accepted_acceptance_sha256=acceptance_sha256, accepted_source_commit=loaded["result"]["source_commit"],
        original_quarter_result_sha256=loaded["result"]["center_result_sha256"],
        coefficient_sha256=loaded["result"]["coefficient_sha256"],
        orbital_sha256=loaded["result"]["orbital_sha256"], freeze_sha256=loaded["result"]["freeze_sha256"],
        active_cache_index_sha256=loaded["result"]["active_cache_index_sha256"],
        directions_sha256=loaded["result"]["directions_sha256"], calibration_sha256=loaded["result"]["calibration_sha256"],
        occupied_capture_floor=loaded["occupied_capture_floor"], initial_frozen_band_screen=band_screen,
        transported_descent=transport, old_direction_reconstruction="exact_accepted_coefficients",
        center_reproduction="pass", load_records=records, cache_load_seconds=cache_seconds,
        total_seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cache_loads=1, rpa_evaluations=1, actual_scf_count=0, optimizer_steps=0, candidate_count=0,
        coefficient_update="none", exported_candidate="none", actual_pbe_direction_derivatives="unmeasured",
        nu=[3, 3, 2, 0, 0], fixed_nu=[0]*5, ao_per_C=22, physical_release_gate="hold",
        ordinary_sos_qavg="pending", gw="pending")
    json.dumps(result, allow_nan=False)
    # Recheck only small evidence/source files, never reload the data cache.
    load_accepted_combined_step(accepted_stage, result_sha256, acceptance_sha256)
    verify_source(stage, deployment_sha256, source_commit)
    endpoint._write_json(output / DIAGNOSTIC, result)
    provenance = {key: result[key] for key in (
        "status", "scope", "source_commit", "deployment_sha256", "accepted_result_sha256",
        "accepted_acceptance_sha256", "accepted_source_commit", "coefficient_sha256",
        "freeze_sha256", "active_cache_index_sha256", "occupied_capture_floor", "center_reproduction",
        "cache_loads", "rpa_evaluations", "gradient_mode", "backward_passes", "actual_scf_count",
        "optimizer_steps", "candidate_count", "coefficient_update", "exported_candidate",
        "physical_release_gate", "ordinary_sos_qavg", "gw")}
    provenance.update(diagnostic_filename="result/"+DIAGNOSTIC,
                      diagnostic_sha256=endpoint._sha256((output / DIAGNOSTIC).read_bytes()))
    endpoint._write_json(stage / "PROVENANCE.json", provenance)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accepted-stage", "result-sha256", "acceptance-sha256",
                 "stage", "deployment-sha256", "source-commit"):
        parser.add_argument("--"+name, required=True)
    result = refresh_gradient(**vars(parser.parse_args(argv)))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
