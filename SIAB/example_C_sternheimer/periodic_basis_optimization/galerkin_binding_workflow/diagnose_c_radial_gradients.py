"""Diagnose the frozen C quarter candidate without exporting or optimizing."""

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "opt_orb_pytorch_dpsi"))

import torch

from check_c_optimized_pbe import _optimizer_artifacts, _read_hashed, _sha256, BACKOFF_SCOPE
from optimize_c_all_radial_fast import load_frozen_c, save_json
from periodic_galerkin_basis import read_periodic_optimizer_coefficients
from periodic_galerkin_radial_diagnostics import evaluate_radial_gradients, radial_guard_sensitivity


def validate_pbe_binding(candidate, digest, pbe):
    required = (pbe.get("pbe_total_energy_gate") == "pass", pbe.get("scf_log_gate") == "pass",
        pbe.get("band_count_check") == "pass", pbe.get("band_counts") == {"occupied": 4, "total": 44},
        pbe.get("candidate_result_sha256") == digest,
        pbe.get("candidate_orbital_sha256") == candidate["orbital_sha256"],
        pbe.get("tolerance_ev_per_c") == .01)
    if not all(required):
        raise ValueError("actual PBE pass must bind the same candidate and original tolerance")
    delta = (pbe["candidate_energy_ev"] - pbe["baseline_energy_ev"])/2
    if (not math.isfinite(delta) or abs(delta) > .010 or not math.isclose(
            delta, pbe["energy_delta_ev_per_c"], rel_tol=0, abs_tol=1e-12)):
        raise ValueError("actual PBE energy deviation mismatch")


def validate_reproduction(candidate, result):
    expected = candidate["candidate"]
    rpa = result["rpa"]
    for actual, before in [(result["loss"], expected["loss"])] + [
            (rpa[k], expected["rpa"][k]) for k in ("candidate_energy_ha", "reference_energy_ha")]:
        if not math.isfinite(actual) or not math.isclose(actual, before, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("diagnostic does not reproduce the accepted quarter endpoint")
    if (rpa.get("complete_q_weight") is not True
            or not math.isclose(rpa.get("q_weight_coverage", 0), 1., rel_tol=0, abs_tol=1e-12)
            or [q["selected_iq"] for q in rpa["per_q"]] != [1, 22, 43, 6, 27, 23, 11, 55]
            or any(len(q["frequency_ha"]) != 12 for q in rpa["per_q"])):
        raise ValueError("complete canonical eight-q twelve-frequency result required")
    if "per_q" in expected["rpa"]:
        for actual, before in zip(rpa["per_q"], expected["rpa"]["per_q"]):
            if actual["frequency_ha"] != before["frequency_ha"] or actual["q_weight"] != before["q_weight"]:
                raise ValueError("q or frequency weights changed")
            for key in ("candidate_contributions_ha", "reference_contributions_ha"):
                if len(actual[key]) != 12 or any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                                                for a, b in zip(actual[key], before[key])):
                    raise ValueError("q/frequency contributions changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-result", "candidate-result-sha256", "pbe-collection",
                 "pbe-collection-sha256", "source-commit", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source = Path(args.candidate_result)
    candidate, _, _ = _optimizer_artifacts(source, args.candidate_result_sha256)
    if candidate.get("scope") != BACKOFF_SCOPE or candidate.get("alpha") != .25:
        raise ValueError("this bounded diagnostic accepts only the existing quarter candidate")
    pbe = json.loads(_read_hashed(Path(args.pbe_collection), args.pbe_collection_sha256))
    validate_pbe_binding(candidate, args.candidate_result_sha256, pbe)
    _read_hashed(Path(pbe["candidate_log_path"]), pbe["candidate_log_sha256"])
    root = source.parent
    original = read_periodic_optimizer_coefficients(root / "ORIGINAL_COEFFICIENTS.txt",
        element="C", radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
    coefficients = read_periodic_optimizer_coefficients(root / "INTERPOLATED_COEFFICIENTS.txt",
        element="C", radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(candidate["configuration"]["threads"])
    datasets, records, guard = load_frozen_c(root / "INPUT_FREEZE.json", candidate["freeze_sha256"],
        original, output, active_cache_index=root / "ACTIVE_DATA_CACHE.json",
        active_cache_index_sha256=candidate["active_cache_index_sha256"])
    cache_load_seconds = time.perf_counter()-started
    initial_screen = guard(coefficients)
    tolerance = 1-max(0., candidate["initial"]["minimum_occupied_capture"]-1e-4)
    save_json(output / "PROGRESS.json", dict(stage="one_forward_two_backward_diagnostics"))
    result = evaluate_radial_gradients(datasets, coefficients,
        occupied_capture_tolerance=tolerance,
        frequency_batch_size=candidate["configuration"]["frequency_batch_size"],
        weights=candidate["training_weights"])
    validate_reproduction(candidate, result)
    save_json(output / "PROGRESS.json", dict(stage="frozen_band_radial_stencils"))
    guard_started = time.perf_counter()
    result["guard_sensitivity"] = {name: radial_guard_sensitivity(coefficients, result[name], guard)
                                   for name in ("loss_gradient", "energy_gradient")}
    result.update(status="success", source_commit=args.source_commit,
        candidate_result_sha256=args.candidate_result_sha256,
        pbe_collection_sha256=args.pbe_collection_sha256,
        coefficient_sha256=candidate["coefficient_sha256"], orbital_sha256=candidate["orbital_sha256"],
        freeze_sha256=candidate["freeze_sha256"], active_cache_index_sha256=candidate["active_cache_index_sha256"],
        initial_frozen_band_screen=initial_screen, load_records=records,
        cache_load_seconds=cache_load_seconds, guard_seconds=time.perf_counter()-guard_started,
        total_seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        nu=[3, 3, 2, 0, 0], fixed_nu=[0, 0, 0, 0, 0], ao_per_C=22,
        actual_pbe_direction_derivatives="unmeasured", optimizer_steps=0,
        coefficient_update="none", exported_candidate="none", physical_release_gate="hold")
    _optimizer_artifacts(source, args.candidate_result_sha256)
    _read_hashed(Path(args.pbe_collection), args.pbe_collection_sha256)
    _read_hashed(Path(pbe["candidate_log_path"]), pbe["candidate_log_sha256"])
    save_json(output / "RADIAL_GRADIENT_DIAGNOSTIC.json", result)
    provenance = dict(status="success", scope=result["scope"], source_commit=args.source_commit,
        candidate_result=str(source.resolve()), candidate_result_sha256=args.candidate_result_sha256,
        pbe_collection=str(Path(args.pbe_collection).resolve()), pbe_collection_sha256=args.pbe_collection_sha256,
        diagnostic_sha256=_sha256((output / "RADIAL_GRADIENT_DIAGNOSTIC.json").read_bytes()),
        actual_pbe_direction_derivatives="unmeasured", physical_release_gate="hold")
    save_json(output / "PROVENANCE.json", provenance)
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
