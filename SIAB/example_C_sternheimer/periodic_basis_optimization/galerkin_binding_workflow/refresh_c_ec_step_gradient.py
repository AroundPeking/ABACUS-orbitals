"""One full Ec derivative at an independently accepted Ec-step center.

No candidate, SCF, ordinary SOS, or optimization is performed. The original
reference and guards remain unchanged; the preceding gradient is provenance,
never substituted for the derivative measured at this center.
"""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

SOURCE = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SOURCE / "SIAB/opt_orb_pytorch_dpsi"),
               str(SOURCE / "SIAB/example_C_sternheimer/periodic_basis_optimization")]

import check_c_accepted_combined_step as accepted
from check_c_accepted_ec_step import load_accepted_ec_step
import check_c_optimized_pbe as endpoint
from check_c_combined_step_pbe import _root
from check_c_direction_probe_pbe import _equal, _json
import refresh_c_combined_step_gradient as common

DIAGNOSTIC = "EC_GRADIENT_REFRESH.json"
_WORKFLOW = "SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/"
REQUIRED_SOURCE_FILES = tuple(_WORKFLOW + name for name in (
    "refresh_c_ec_step_gradient.py", "run_c_ec_step_gradient.slurm", "check_c_accepted_ec_step.py"))
PROVENANCE_KEYS = (
    "status", "scope", "source_commit", "deployment_sha256", "accepted_stage",
    "accepted_result_sha256", "accepted_acceptance_sha256", "accepted_source_commit",
    "coefficient_sha256", "orbital_sha256", "freeze_sha256", "active_cache_index_sha256",
    "preceding_gradient_result_sha256", "preceding_gradient_acceptance_sha256", "gradient_step_sha256",
    "occupied_capture_floor", "center_reproduction", "cache_loads", "rpa_evaluations",
    "gradient_mode", "backward_passes", "actual_scf_count", "optimizer_steps", "candidate_count",
    "coefficient_update", "exported_candidate", "physical_release_gate", "ordinary_sos_qavg", "gw")


def verify_source(stage, deployment_sha256, source_commit):
    manifest = common.verify_source(stage, deployment_sha256, source_commit)
    if not set(REQUIRED_SOURCE_FILES).issubset(manifest["files"]):
        raise ValueError("Ec-step reader, runner and batch must be hash-pinned")
    return manifest


def _runtime():
    return common._runtime()


def _preceding_direction(runtime, loaded):
    """Admission has already reconstructed the signed-QR candidate exactly."""
    root = loaded["coefficient_path"].parent
    step = _json(accepted._small(root, "GRADIENT_STEP.json", loaded["candidate"]["gradient_step_sha256"]))
    return {element: [runtime.torch.tensor(block, dtype=runtime.torch.float64) for block in blocks]
            for element, blocks in step["direction"].items()}


def refresh_gradient(*, accepted_stage, result_sha256, acceptance_sha256,
                     stage, deployment_sha256, source_commit):
    started = time.perf_counter()
    stage, accepted_stage = _root(stage), _root(accepted_stage)
    if stage == accepted_stage or accepted_stage in stage.parents or stage in accepted_stage.parents:
        raise ValueError("new derivative stage must be separate from accepted evidence")
    output = stage / "result"
    for path in (output, stage / "PROVENANCE.json"):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    verify_source(stage, deployment_sha256, source_commit)
    loaded = load_accepted_ec_step(accepted_stage, result_sha256, acceptance_sha256)
    _equal(loaded["result"].get("scope"), "one_accepted_ec_gradient_step", "Ec-step parent scope")
    if source_commit == loaded["result"]["source_commit"]:
        raise ValueError("new derivative source must not claim the old step source")
    runtime = _runtime()
    runtime.torch.set_num_threads(28)
    coefficients = common._read_coefficients(runtime, loaded["coefficient_path"])
    original = common._read_coefficients(runtime, loaded["original_coefficient_path"])
    previous_direction = _preceding_direction(runtime, loaded)
    output.mkdir()
    load_started = time.perf_counter()
    datasets, records, guard = runtime.load_frozen_c(
        loaded["freeze_path"], loaded["result"]["freeze_sha256"], original, output,
        active_cache_index=loaded["active_cache_index_path"],
        active_cache_index_sha256=loaded["result"]["active_cache_index_sha256"])
    cache_seconds = time.perf_counter()-load_started
    common.validate_dataset_extent(datasets, loaded)
    with runtime.torch.no_grad():
        band_screen = guard(coefficients)
    accepted._band_guard(band_screen)
    diagnostic = runtime.evaluate_radial_gradients(datasets, coefficients,
        occupied_capture_tolerance=1-loaded["occupied_capture_floor"],
        frequency_batch_size=loaded["quarter"]["configuration"]["frequency_batch_size"],
        weights=loaded["quarter"]["training_weights"], energy_only=True)
    common.validate_reproduction(loaded, diagnostic, band_screen)
    transport = runtime.transported_descent_report(coefficients, diagnostic["energy_gradient"], previous_direction)
    if not isinstance(transport, dict):
        raise ValueError("transported descent report required")
    result = dict(diagnostic, scope="accepted_ec_step_ec_gradient_refresh", status="success",
        source_commit=source_commit, deployment_sha256=deployment_sha256, accepted_stage=str(accepted_stage),
        accepted_result_sha256=result_sha256, accepted_acceptance_sha256=acceptance_sha256,
        accepted_source_commit=loaded["result"]["source_commit"],
        preceding_gradient_result_sha256=loaded["result"]["gradient_result_sha256"],
        preceding_gradient_acceptance_sha256=loaded["result"]["gradient_acceptance_sha256"],
        gradient_step_sha256=loaded["candidate"]["gradient_step_sha256"],
        occupied_capture_floor=loaded["occupied_capture_floor"], initial_frozen_band_screen=band_screen,
        transported_descent=transport, old_direction_reconstruction="validated_accepted_ec_step",
        center_reproduction="pass", load_records=records, cache_load_seconds=cache_seconds,
        total_seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cache_loads=1, rpa_evaluations=1, actual_scf_count=0, optimizer_steps=0, candidate_count=0,
        coefficient_update="none", exported_candidate="none", actual_pbe_direction_derivatives="unmeasured",
        nu=[3, 3, 2, 0, 0], fixed_nu=[0]*5, ao_per_C=22, physical_release_gate="hold",
        ordinary_sos_qavg="pending", gw="pending")
    result.update({key: loaded["result"][key] for key in
        ("coefficient_sha256", "orbital_sha256", "freeze_sha256", "active_cache_index_sha256")})
    json.dumps(result, allow_nan=False)
    load_accepted_ec_step(accepted_stage, result_sha256, acceptance_sha256)
    verify_source(stage, deployment_sha256, source_commit)
    endpoint._write_json(output / DIAGNOSTIC, result)
    provenance = {key: result[key] for key in PROVENANCE_KEYS}
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
