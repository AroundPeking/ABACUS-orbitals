"""Frozen-band sensitivities at an accepted Ec-gradient center, no RPA or SCF."""

import argparse
import json
import math
from pathlib import Path
import resource
import time

import refresh_c_ec_step_gradient as refresh
import check_c_ec_gradient_step as checks
from check_c_combined_step_pbe import _root
from check_c_direction_probe_pbe import _digest, _equal, _json

common = refresh.common


def verify_source(stage, deployment_sha256, source_commit):
    manifest = refresh.verify_source(stage, deployment_sha256, source_commit)
    required = {refresh._WORKFLOW+n for n in (
        "screen_c_ec_step_constraints.py", "run_c_ec_constraint_screen.slurm")}
    if not required.issubset(manifest["files"]):
        raise ValueError("constraint runner and batch source pins required")
    return manifest


def _runtime():
    rt = common._runtime()
    from periodic_galerkin_radial_diagnostics import radial_guard_sensitivity, radial_gradient_report
    rt.radial_guard_sensitivity = radial_guard_sensitivity
    rt.radial_gradient_report = radial_gradient_report
    return rt


def admit_gradient(stage, result_sha, acceptance_sha):
    stage = _root(stage)
    small, expect = refresh.accepted._small, refresh.accepted._expect
    g = _json(small(stage, "result/EC_GRADIENT_REFRESH.json", _digest(result_sha)))
    a = _json(small(stage, "EC_STEP_GRADIENT_ACCEPTANCE.json", _digest(acceptance_sha)))
    expect(a, dict(status="success", scope="ec_step_gradient_acceptance_without_replay_not_optimized_basis",
        result_sha256=result_sha, deployment_sha256=g["deployment_sha256"],
        accepted_result_sha256=g["accepted_result_sha256"],
        accepted_acceptance_sha256=g["accepted_acceptance_sha256"],
        new_candidate_count=0, new_scf_count=0, physical_release_gate="hold"), "independent gradient acceptance")
    expect(g, dict(status="success", scope="accepted_ec_step_ec_gradient_refresh",
        candidate_count=0, actual_scf_count=0, optimizer_steps=0, coefficient_update="none",
        gradient_mode="energy_only", cache_loads=1, rpa_evaluations=1, backward_passes=1,
        physical_release_gate="hold"), "measured gradient")
    _equal(small(stage, "STATUS"), b"success\n", "gradient STATUS")
    checks._archived_source(stage, g["deployment_sha256"], g["source_commit"])
    provenance = {k:g[k] for k in refresh.PROVENANCE_KEYS}
    provenance.update(diagnostic_filename="result/EC_GRADIENT_REFRESH.json", diagnostic_sha256=result_sha)
    checks._same_json(_json(small(stage, "PROVENANCE.json")), provenance, "gradient provenance")
    audit = a["stderr_audit"]
    small(stage, "slurm-"+a["job_id"]+".err", audit["sha256"])
    if audit["gate"] not in ("empty", "audited_startup_locale_warning"):
        raise ValueError("independently audited stderr required")
    center = refresh.load_accepted_ec_step(g["accepted_stage"],
        g["accepted_result_sha256"], g["accepted_acceptance_sha256"])
    for key in ("source_commit", "coefficient_sha256", "orbital_sha256", "freeze_sha256", "active_cache_index_sha256"):
        _equal(g["accepted_source_commit" if key=="source_commit" else key], center["result"][key], "center "+key)
    common.validate_reproduction(center, dict(g,scope="radial_gradients_at_fixed_candidate_no_optimization"),
                                g["initial_frozen_band_screen"])
    rt = _runtime()
    rt.torch.set_num_threads(28)
    c = common._read_coefficients(rt, center["coefficient_path"])
    rows = g["energy_gradient"]["channels"]
    raw = {"C":[rt.torch.tensor(row["raw_gradient"],dtype=rt.torch.float64) for row in rows]}
    checks._same_json(rt.radial_gradient_report(c,raw),g["energy_gradient"],"existing raw gradient")
    return center,g


def target_band_slopes(sensitivity):
    rows=sensitivity.get("radials",[])
    if (sensitivity.get("scope")!="frozen_band_sensitivity_not_actual_PBE_derivative"
            or [(r["element"],r["l"],r["zeta"]) for r in rows] !=
            [("C",l,z+1) for l,n in enumerate([3,3,2]) for z in range(n)]):
        raise ValueError("complete ordered C radial sensitivities required")
    result=[]
    for row in rows:
        if row["status"]!="success" or [s["radius"] for s in row["stencils"]] != [1e-4,5e-5]:
            raise ValueError("unresolved two-radius sensitivity")
        slopes=[]
        for s in row["stencils"]:
            a,b=s["minus"],s["plus"]
            values=[v["maximum_target_band_change_ev"] for v in (a,b)]
            if not all(v["gate"] for v in (a,b)) or not all(math.isfinite(x) for x in values):
                raise ValueError("invalid target-band stencil")
            slopes.append(dict(radius=s["radius"],maximum_band_slope_ev=(values[1]-values[0])/(2*s["radius"])))
        result.append(dict(element=row["element"],l=row["l"],zeta=row["zeta"],stencils=slopes,
            two_radius_slope_difference_ev=abs(slopes[0]["maximum_band_slope_ev"]-slopes[1]["maximum_band_slope_ev"]),
            active_band_identity="unmeasured"))
    return result


def compare_center_guard(actual, expected):
    if set(actual)!=set(expected):
        raise ValueError("same-center band guard fields changed")
    for key,value in expected.items():
        if type(value) is float:
            if (type(actual[key]) not in (int,float) or not math.isfinite(actual[key])
                    or not math.isclose(actual[key],value,rel_tol=1e-10,abs_tol=1e-12)):
                raise ValueError("same-center band guard "+key)
        elif type(actual[key]) is not type(value) or actual[key]!=value:
            raise ValueError("same-center band contract "+key)


def screen(*,stage,source_commit,deployment_sha256,gradient_stage,gradient_result_sha256,gradient_acceptance_sha256):
    started=time.perf_counter()
    stage,previous=_root(stage),_root(gradient_stage)
    if stage==previous or stage in previous.parents or previous in stage.parents:
        raise ValueError("separate constraint stage required")
    output=stage/"result"
    if output.exists() or output.is_symlink() or (stage/"PROVENANCE.json").exists():
        raise FileExistsError(output)
    verify_source(stage,deployment_sha256,source_commit)
    center,g=admit_gradient(previous,gradient_result_sha256,gradient_acceptance_sha256)
    rt=_runtime()
    rt.torch.set_num_threads(28)
    c=common._read_coefficients(rt,center["coefficient_path"])
    original=common._read_coefficients(rt,center["original_coefficient_path"])
    output.mkdir()
    load_started=time.perf_counter()
    datasets,records,guard=rt.load_frozen_c(center["freeze_path"],center["result"]["freeze_sha256"],
        original,output,active_cache_index=center["active_cache_index_path"],
        active_cache_index_sha256=center["result"]["active_cache_index_sha256"])
    load_seconds=time.perf_counter()-load_started
    common.validate_dataset_extent(datasets,center)
    with rt.torch.no_grad():
        measured=guard(c)
    compare_center_guard(measured,center["result"]["candidate"]["coefficient_guard"])
    screen_started=time.perf_counter()
    sensitivity=rt.radial_guard_sensitivity(c,g["energy_gradient"],guard,radii=(1e-4,5e-5))
    target=target_band_slopes(sensitivity)
    result=dict(status="success",scope="current_ec_step_frozen_constraint_sensitivity_not_pbe",
        source_commit=source_commit,deployment_sha256=deployment_sha256,gradient_stage=str(previous),
        gradient_result_sha256=gradient_result_sha256,gradient_acceptance_sha256=gradient_acceptance_sha256,
        coefficient_sha256=g["coefficient_sha256"],orbital_sha256=g["orbital_sha256"],
        accepted_result_sha256=g["accepted_result_sha256"],center_band_screen=measured,
        guard_sensitivity=sensitivity,target_band_slopes=target,load_records=records,
        cache_load_seconds=load_seconds,screen_seconds=time.perf_counter()-screen_started,
        total_seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cache_loads=1,guard_evaluations=33,rpa_evaluations=0,backward_passes=0,actual_scf_count=0,
        candidate_count=0,optimizer_steps=0,coefficient_update="none",exported_candidate="none",
        actual_pbe_direction_derivative="unmeasured",physical_release_gate="hold")
    json.dumps(result,allow_nan=False)
    admit_gradient(previous,gradient_result_sha256,gradient_acceptance_sha256)
    verify_source(stage,deployment_sha256,source_commit)
    refresh.endpoint._write_json(output/"CONSTRAINT_SCREEN.json",result)
    refresh.endpoint._write_json(stage/"PROVENANCE.json",dict(status="success",scope=result["scope"],
        source_commit=source_commit,deployment_sha256=deployment_sha256,
        gradient_result_sha256=gradient_result_sha256,gradient_acceptance_sha256=gradient_acceptance_sha256,
        result_sha256=refresh.endpoint._sha256((output/"CONSTRAINT_SCREEN.json").read_bytes()),
        physical_release_gate="hold"))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("stage","source-commit","deployment-sha256","gradient-stage","gradient-result-sha256","gradient-acceptance-sha256"):
        parser.add_argument("--"+name,required=True)
    print(json.dumps(screen(**vars(parser.parse_args())),allow_nan=False))
