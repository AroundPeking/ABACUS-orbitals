"""Read-only admission of an improved Ec step, never a combined/physics release.

The caller supplies independent RESULT and STEP_MEASUREMENT_ACCEPTANCE pins.
Historical source is checked as data, never executed. Existing validators own
the parent provenance, coefficient reconstruction and PBE runtime identities.
No SCF, response, cache payload or scheduler is invoked. Reconstruction renders
one orbital in the existing validator's temporary directory, not in the stage.
"""

import math
from pathlib import Path

import check_c_accepted_combined_step as common
import check_c_ec_gradient_step as admission
import check_c_ec_gradient_step_pbe as pbe
import check_c_optimized_pbe as endpoint
from check_c_combined_step_pbe import _near, _number, _root
from check_c_direction_probe_pbe import _digest, _equal, _json, _strict_log


SCOPE = "one_accepted_ec_gradient_step"
ACCEPTANCE = "STEP_MEASUREMENT_ACCEPTANCE.json"
_WORKFLOW = "SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/"
_SOURCE_FILES = tuple(_WORKFLOW + name for name in (
    "check_c_ec_gradient_step.py", "check_c_ec_gradient_step_pbe.py",
    "run_c_ec_gradient_step.py", "run_c_ec_gradient_step.slurm")) + (
    "SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_ec_gradient_step.py",)
_PROVENANCE = ("status", "source_commit", "candidate_gate", "actual_scf_count",
               "rpa_evaluations", "backward_passes", "physical_release_gate")
_BINDINGS = ("center_stage", "center_result_sha256", "center_acceptance_sha256",
             "gradient_stage", "gradient_result_sha256", "gradient_acceptance_sha256",
             "coefficient_sha256", "orbital_sha256", "gradient_step_sha256", "radius")


def _execution(stage, result, acceptance):
    commit = admission._commit(result.get("source_commit"))
    deployment_sha = _digest(result.get("deployment_sha256"))
    deployment = admission._archived_source(stage, deployment_sha, commit)
    if not set(_SOURCE_FILES).issubset(deployment["files"]):
        raise ValueError("complete Ec-step source archive required")
    scheduler = acceptance.get("scheduler")
    if not isinstance(scheduler, str):
        raise ValueError("archived Ec-step scheduler text required")
    common._scheduler(dict(acceptance, scheduler=[line.split("|") for line in scheduler.splitlines() if line.strip()]))
    job = acceptance["job_id"]
    _equal(common._small(stage, "slurm-"+job+".err"), b"", "Ec-step stderr")
    timing = common._small(stage, "ec-step.time").decode().strip().splitlines()
    if (not timing or timing[-1].strip() != "Exit status: 0"
            or sum("Exit status:" in line for line in timing) != 1):
        raise ValueError("successful Ec-step process timing required")
    submission = _json(common._small(stage, "SUBMISSION.json"))
    common._expect(submission, dict(status="submitted_not_accepted", job_id=job, stage=str(stage),
        source_commit=commit, deployment_sha256=deployment_sha, archive_sha256=deployment["archive_sha256"],
        source_file_count=len(deployment["files"])), "submission")
    _digest(submission.get("fingerprint"))
    execution = _json(common._small(stage, "EXECUTION_PROVENANCE.json"))
    common._expect(execution, dict(status="launching", scheduler_job=job, source_commit=commit,
        actual_scf_budget=1, backward_budget=0, delta_st="not_run", librpa="not_run",
        ordinary_sos_qavg="pending", physical_release_gate="hold"), "execution provenance")
    return deployment, execution


def _actual_pbe(stage, result, candidate):
    slot = _root(stage / "result/pbe")
    collected = _json(common._small(slot, pbe.COLLECTION))
    admission._same_json(collected, result.get("actual_pbe"), "persisted Ec PBE collection")
    preparation_sha = _digest(collected.get("preparation_sha256"))
    manifest = _json(common._small(slot, pbe.PREPARATION, preparation_sha))
    total = 0
    for key, count in (("prepared_input_sha256", 5), ("evidence_sha256", 8)):
        values = manifest.get(key)
        if not isinstance(values, dict) or len(values) != count:
            raise ValueError("bounded complete Ec PBE " + key + " required")
        for name, sha in values.items():
            total += len(common._small(slot, name, _digest(sha)))
            if total > common._MAX_EVIDENCE_BYTES:
                raise ValueError("Ec PBE evidence exceeds validation budget")
    common._expect(manifest, dict(candidate_root=str(stage / "result/candidate"),
        candidate_sha256=result["candidate_sha256"], center_stage=candidate["center_stage"],
        **{key:candidate[key] for key in pbe._IDENTITY}), "Ec PBE candidate binding")
    # _prepared re-creates the exact snapshot in memory, checks live runtime
    # hashes and every frozen input/proof copy, and does not write a collection.
    admission._same_json(pbe._prepared(slot, preparation_sha), manifest, "validated Ec PBE preparation")
    _near(manifest.get("baseline_energy_ev"), pbe.ORIGINAL_ENERGY_EV, "original PBE baseline")
    log_name = manifest["candidate_log_relative_path"]
    measured = _strict_log(common._small(slot, log_name, _digest(collected.get("candidate_log_sha256"))))
    delta = (measured["energy_ev"]-manifest["baseline_energy_ev"])/2
    if abs(delta) > endpoint.TOLERANCE_EV_PER_C+1e-12:
        raise ValueError("actual PBE exceeds original 10 meV/C bound")
    expected = dict(pbe._COMMON, status="collected", preparation_sha256=preparation_sha,
        candidate_energy_ev=measured["energy_ev"], energy_delta_ev_per_c=delta,
        delta_from_center_ev_per_c=(measured["energy_ev"]-manifest["center_energy_ev"])/2,
        comparison="absolute_candidate_minus_original_per_C_lte_tolerance", pbe_gate="pass",
        scf_log_gate="pass", band_count_check="pass", band_counts=measured["band_counts"],
        candidate_log_path=str(slot / log_name), candidate_log_sha256=measured["log_sha256"])
    expected.update({key:manifest[key] for key in pbe._IDENTITY + (
        "candidate_sha256", "center_stage", "center_preparation_sha256", "baseline_energy_ev",
        "center_energy_ev", "baseline_log_sha256", "center_log_sha256",
        "abacus_binary", "abacus_sha256", "mpi_library", "mpi_sha256")})
    admission._same_json(collected, expected, "complete measured Ec PBE collection")
    _equal(common._small(slot, ".provenance/CANDIDATE.json", result["candidate_sha256"]),
           common._small(stage, "result/candidate/CANDIDATE.json", result["candidate_sha256"]),
           "prepared candidate bytes")
    return collected


def _mathematics(result, candidate, parent, gradient):
    quarter = parent["quarter"]
    initial_capture = _number(quarter["initial"].get("minimum_occupied_capture"), "original initial capture")
    common._capture(initial_capture, 0.)
    floor = max(0., initial_capture-1e-4)
    for actual in (result.get("occupied_capture_floor"), parent.get("occupied_capture_floor"),
                   candidate.get("cheap_gate", {}).get("occupied_capture_floor")):
        _equal(_number(actual, "original occupied floor"), floor, "original occupied floor")
    cheap = candidate["cheap_gate"]
    common._expect(cheap, dict(gate=True), "cheap gate")
    for key, expected in (("overlap_relative_rank_tolerance", 1e-12), ("overlap_condition_limit", 1e12)):
        _equal(_number(cheap.get(key), key), expected, "original cheap guard " + key)
    first, new, previous = result.get("center"), result.get("candidate"), parent["result"]["candidate"]
    for value in (previous, first, new):
        common._record(value, floor, quarter["training_weights"])
        common._same_grid_reference(value, previous)
        for row, weight in zip(value["rpa"]["per_q"], common._WEIGHTS):
            _equal(row["q_weight"], weight, "exact frozen q weight")
    endpoint._match_parent_initial(first, previous)
    common._same_grid_reference(first, previous, reproduce=True)
    for name in ("pi", "trace_log", "energy"):
        key = name+"_relative_squared_error"
        _near(first["rpa"][key], previous["rpa"][key], "parent reproduction " + key)
    for key in ("minimum_occupied_capture", "maximum_overlap_condition"):
        if not math.isclose(_number(first[key], key), _number(previous[key], key), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("parent reproduction failed: " + key)
    _near(cheap.get("minimum_occupied_capture"), new["minimum_occupied_capture"], "cheap/full capture")
    admission._same_json(cheap.get("band_screen"), new["coefficient_guard"], "cheap/full frozen band guard")
    ec, ref, old = new["rpa"]["candidate_energy_ha"], new["rpa"]["reference_energy_ha"], first["rpa"]["candidate_energy_ha"]
    if not (abs(ec-ref) < abs(old-ref) and new["loss"] <= first["loss"]):
        raise ValueError("Ec-step requires closer reference energy and nonincreasing loss")
    norm = _number(gradient["energy_gradient"].get("horizontal_gradient_norm"), "accepted Ec gradient norm")
    if norm <= 1e-14:
        raise ValueError("resolved parent Ec gradient required")
    prediction = -candidate["radius"]*norm*endpoint.HARTREE_TO_EV/2
    _near(candidate.get("predicted_ec_delta_ha_per_cell"), -candidate["radius"]*norm, "candidate Ec prediction")
    for key, expected in (("predicted_ec_delta_ev_per_c", prediction),
            ("measured_ec_delta_ev_per_c", (ec-old)*endpoint.HARTREE_TO_EV/2),
            ("body_error_ev_per_c", abs(ec-ref)*endpoint.HARTREE_TO_EV/2),
            ("measured_to_predicted_ec_gain", (ec-old)*endpoint.HARTREE_TO_EV/2/prediction)):
        _near(result.get(key), expected, "Ec-step " + key)
    return floor


def _measurement_acceptance(result, acceptance):
    first, new, measured = result["center"], result["candidate"], result["actual_pbe"]
    expected = dict(actual_pbe_energy_ev_per_cell=measured["candidate_energy_ev"],
        actual_pbe_delta_mev_per_c=measured["energy_delta_ev_per_c"]*1000,
        ec_ha_per_cell=new["rpa"]["candidate_energy_ha"], reference_ec_ha_per_cell=new["rpa"]["reference_energy_ha"],
        body_error_ev_per_c=result["body_error_ev_per_c"], improvement_mev_per_c=-result["measured_ec_delta_ev_per_c"]*1000,
        predicted_improvement_mev_per_c=-result["predicted_ec_delta_ev_per_c"]*1000,
        initial_loss=first["loss"], loss=new["loss"], capture=new["minimum_occupied_capture"],
        condition=new["maximum_overlap_condition"])
    for key in ("total_seconds", "cache_load_seconds", "peak_rss_kib", "scf_seconds", "response_seconds"):
        value = _number(result.get(key), key)
        if value <= 0 or (key == "total_seconds" and value >= 3600):
            raise ValueError("invalid Ec-step timing/resource report: " + key)
        if key in ("total_seconds", "cache_load_seconds", "peak_rss_kib"):
            expected[key] = value
    for key, value in expected.items():
        _near(acceptance.get(key), value, "measurement acceptance " + key)
    for label, record in (("before", first), ("after", new)):
        admission._same_json(acceptance.get("per_q_"+label), record["rpa"]["per_q"], "accepted full-q " + label)
    errors = {name:dict(before=first["rpa"][name+"_relative_squared_error"],
                        after=new["rpa"][name+"_relative_squared_error"]) for name in ("pi", "trace_log", "energy")}
    admission._same_json(acceptance.get("errors"), errors, "accepted RPA errors")


def load_accepted_ec_step(stage, result_sha256, acceptance_sha256):
    """Return own result/candidate/acceptance, original quarter and frozen paths.

    result['candidate'] is the newly measured eight-q/twelve-frequency endpoint;
    candidate is its immutable proposal manifest. quarter retains ORIGINAL
    training weights/configuration. Historical parent/gradient/runtime stages
    must still exist; this is not a self-contained portable archive adapter.
    """
    _digest(result_sha256)
    _digest(acceptance_sha256)
    stage = _root(stage)
    result = _json(common._small(stage, "result/RESULT.json", result_sha256))
    acceptance = _json(common._small(stage, ACCEPTANCE, acceptance_sha256))
    _equal(common._small(stage, "STATUS").decode().strip(), "success", "Ec-step STATUS")
    common._expect(result, dict(status="success", scope=SCOPE, candidate_gate="improved_frozen_body",
        actual_scf_count=1, rpa_evaluations=2, cache_loads=1, backward_passes=0, optimizer_steps=0,
        physical_release_gate="hold", ordinary_sos_qavg="pending", gw="pending",
        actual_pbe_direction_derivative="unmeasured"), "accepted Ec-step result")
    admission._profile(result)
    provenance = _json(common._small(stage, "PROVENANCE.json"))
    admission._same_json(provenance, dict({key:result[key] for key in _PROVENANCE},
                                         result_sha256=result_sha256), "Ec-step provenance")
    common._expect(acceptance, dict(status="success", scope="ec_gradient_step_measurement_acceptance",
        result_sha256=result_sha256, source_commit=result["source_commit"], deployment_sha256=result.get("deployment_sha256"),
        candidate_gate="improved_frozen_body", physical_release_gate="hold", ordinary_sos_qavg="pending", gw="pending",
        pbe_gate="pass", coefficient_sha256=result.get("coefficient_sha256"), orbital_sha256=result.get("orbital_sha256")),
        "Ec-step acceptance")
    _digest(acceptance.get("verification_script_sha256"))
    deployment, execution = _execution(stage, result, acceptance)
    root = _root(stage / "result/candidate")
    candidate_sha = _digest(result.get("candidate_sha256"))
    candidate_bytes = common._small(root, "CANDIDATE.json", candidate_sha)
    candidate = _json(candidate_bytes)
    common._expect(candidate, dict(status="prepared", scope=admission.SCOPE, source_commit=result["source_commit"]),
                   "own immutable Ec-step candidate")
    admission._profile(candidate)
    for key in _BINDINGS:
        if key.endswith("sha256"):
            _digest(result.get(key))
        admission._same_json(candidate.get(key), result.get(key), "candidate/result " + key)
    for name, key in pbe._FILES.items():
        common._small(root, name, _digest(candidate.get(key)))
    if candidate.get("radius") == .018:
        reduction = candidate.get("reduction_evidence")
        admission._same_json(result.get("reduction_evidence"), reduction, "pinned radius reduction")
        if not isinstance(reduction, dict):
            raise ValueError("bounded reduction evidence required")
        admission._same_json(reduction, dict(rejected_stage=str(admission.REJECTED_STAGE),
            rejected_result_sha256=admission.REJECTED_RESULT,
            rejected_acceptance_sha256=admission.REJECTED_ACCEPTANCE,
            rejected_diagnostic_sha256=admission.REJECTED_DIAGNOSTIC,
            gradient_result_sha256=admission.GRADIENT_RESULT,
            prior_radius=.02, next_radius=.018, reduction_factor=.9, automatic_radius_scan=False),
            "exact legacy reduction evidence graph")
        rejected = admission._absolute(reduction.get("rejected_stage"))
        for name, key in (("result/RESULT.json", "rejected_result_sha256"),
                (ACCEPTANCE, "rejected_acceptance_sha256"), ("GUARD_REJECTION_DIAGNOSTIC.json", "rejected_diagnostic_sha256")):
            common._small(rejected, name, _digest(reduction.get(key)))
    admission._same_json(admission.validate_gradient_step_candidate(root), candidate, "fully reconstructed Ec-step candidate")
    _equal(common._small(root, "CANDIDATE.json", candidate_sha), candidate_bytes, "unchanged reconstructed candidate")
    loaded = admission.load_accepted_ec_gradient(admission._absolute(candidate.get("gradient_stage")),
        _digest(candidate.get("gradient_result_sha256")), _digest(candidate.get("gradient_acceptance_sha256")))
    parent, gradient = loaded["center"], loaded["gradient"]
    _equal(admission._absolute(candidate.get("center_stage")), loaded["center_stage"], "accepted parent stage")
    for key, expected in (("center_result_sha256", gradient["accepted_result_sha256"]),
            ("center_acceptance_sha256", gradient["accepted_acceptance_sha256"]),
            ("center_coefficient_sha256", parent["result"]["coefficient_sha256"]),
            ("center_orbital_sha256", parent["result"]["orbital_sha256"])):
        _equal(candidate.get(key), expected, "accepted parent " + key)
    floor = _mathematics(result, candidate, parent, gradient)
    paths = dict(coefficient_path=root / "COEFFICIENTS.txt", orbital_path=root / "C_3s3p2d_ec_step.orb")
    for key, identity in (("original_coefficient_path", "initial_sha256"),
                          ("freeze_path", "freeze_sha256"), ("active_cache_index_path", "active_cache_index_sha256")):
        path = Path(parent[key])
        sha = parent["quarter"][identity] if identity == "initial_sha256" else parent["result"][identity]
        common._small(_root(path.parent), path.name, _digest(sha))
        if identity != "initial_sha256":
            _equal(_digest(result.get(identity)), sha, "original frozen " + identity)
        paths[key] = path
    measured = _actual_pbe(stage, result, candidate)
    for key, identity in (("abacus_sha256", "abacus_sha256"), ("pmi_sha256", "mpi_sha256")):
        _equal(_digest(execution.get(key)), _digest(measured.get(identity)), "execution runtime " + key)
    _measurement_acceptance(result, acceptance)
    return dict(result=result, candidate=candidate, acceptance=acceptance, quarter=parent["quarter"],
        occupied_capture_floor=floor, center=parent, gradient=gradient, stage=stage, deployment=deployment,
        result_sha256=result_sha256, acceptance_sha256=acceptance_sha256, **paths)
