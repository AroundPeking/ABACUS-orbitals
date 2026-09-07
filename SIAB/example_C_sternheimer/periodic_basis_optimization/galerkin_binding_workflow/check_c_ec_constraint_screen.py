"""Reconstruct saved current-center stencils without cache, SCF or response replay."""

import math

import screen_c_ec_step_constraints as screen

IDENTITIES=[("C",l,z+1) for l,n in enumerate([3,3,2]) for z in range(n)]


def _finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError("finite numeric metric required")
    return value


def _close(a,b):
    if not math.isclose(_finite(a),_finite(b),rel_tol=1e-10,abs_tol=1e-11):
        raise ValueError("saved finite difference does not reconstruct")


def _expect(actual,expected):
    if any(type(actual.get(k)) is not type(v) or actual[k]!=v for k,v in expected.items()):
        raise ValueError("constraint screen identity changed")


def validate_scope(r):
    _expect(r,dict(status="success",scope="current_ec_step_frozen_constraint_sensitivity_not_pbe",
        cache_loads=1,guard_evaluations=33,rpa_evaluations=0,backward_passes=0,actual_scf_count=0,
        candidate_count=0,optimizer_steps=0,coefficient_update="none",exported_candidate="none",
        actual_pbe_direction_derivative="unmeasured",physical_release_gate="hold"))


def _guard(g):
    _expect(g,dict(gate=True,scope="frozen_h_band_screen_not_scf_energy",scf_pbe_gate="pending",
        occupied_band_sum_limit_ev_per_atom=.01,target_band_change_limit_ev=.05,
        k_weight_sum=2.,k_weight_convention="ABACUS_spin_included_no_renormalization"))
    if (abs(_finite(g["occupied_band_sum_change_ev_per_atom"]))>.01 or
            not 0<=_finite(g["maximum_target_band_change_ev"])<=.05 or
            _finite(g["minimum_gap_ev"])<=0):
        raise ValueError("original band guard failed")


def analyze_stencils(r):
    validate_scope(r)
    _guard(r["center_band_screen"])
    sensitivity=r["guard_sensitivity"]
    _expect(sensitivity,dict(scope="frozen_band_sensitivity_not_actual_PBE_derivative",
        direction="unit_negative_horizontal_gradient_per_saved_radial",
        actual_pbe_sensitivity="unmeasured",physical_release_gate="hold"))
    rows,targets=sensitivity["radials"],r["target_band_slopes"]
    for data in (rows,targets):
        if [(x["element"],x["l"],x["zeta"]) for x in data]!=IDENTITIES:
            raise ValueError("all eight ordered radial identities required")
    output=[]
    for row,target in zip(rows,targets):
        _expect(row,dict(status="success"))
        _expect(target,dict(active_band_identity="unmeasured"))
        if any([s["radius"] for s in data["stencils"]]!=[1e-4,5e-5] for data in (row,target)):
            raise ValueError("two prescribed stencil radii required")
        occupied,maximum=[],[]
        for s,t in zip(row["stencils"],target["stencils"]):
            a,b=s["minus"],s["plus"]
            _guard(a)
            _guard(b)
            occupied.append((b["occupied_band_sum_change_ev_per_atom"]-a["occupied_band_sum_change_ev_per_atom"])/(2*s["radius"]))
            maximum.append((b["maximum_target_band_change_ev"]-a["maximum_target_band_change_ev"])/(2*s["radius"]))
            _close(occupied[-1],s["band_sum_derivative_ev_per_atom"])
            _close(maximum[-1],t["maximum_band_slope_ev"])
        od,md=abs(occupied[0]-occupied[1]),abs(maximum[0]-maximum[1])
        _close(od,row["two_size_derivative_difference"])
        _close(md,target["two_radius_slope_difference_ev"])
        output.append(dict(element=row["element"],l=row["l"],zeta=row["zeta"],
            occupied_slope_ev_per_c=occupied[1],target_slope_ev=maximum[1],
            occupied_two_radius_difference=od,target_two_radius_difference=md,
            active_band_identity="unmeasured"))
    return output


def validate_timing(r,process_peak):
    keys=("cache_load_seconds","screen_seconds","total_seconds","peak_rss_kib")
    if (any(_finite(r[k])<=0 for k in keys) or
            not r["cache_load_seconds"]+r["screen_seconds"]<r["total_seconds"]<3600 or
            _finite(process_peak)<r["peak_rss_kib"] or process_peak>102400*1024):
        raise ValueError("invalid screen timing or memory")
    return dict({k:r[k] for k in keys},process_peak_rss_kib=process_peak,
        memory_snapshots="result_before_postflight_and_process_exit_separate")


def load_screen(stage,result_sha256,deployment_sha256,source_commit,gradient_result_sha256,gradient_acceptance_sha256):
    stage=screen._root(stage)
    small=screen.refresh.accepted._small
    manifest=screen.checks._archived_source(stage,deployment_sha256,source_commit)
    if not {screen.refresh._WORKFLOW+n for n in ("screen_c_ec_step_constraints.py",
            "run_c_ec_constraint_screen.slurm")}.issubset(manifest["files"]):
        raise ValueError("archived screen runner source required")
    r=screen._json(small(stage,"result/CONSTRAINT_SCREEN.json",screen._digest(result_sha256)))
    _expect(r,dict(source_commit=source_commit,deployment_sha256=deployment_sha256,
        gradient_result_sha256=gradient_result_sha256,gradient_acceptance_sha256=gradient_acceptance_sha256))
    center,g=screen.admit_gradient(r["gradient_stage"],gradient_result_sha256,gradient_acceptance_sha256)
    for key in ("coefficient_sha256","orbital_sha256","accepted_result_sha256"):
        if r[key]!=g[key]: raise ValueError("same center binding required")
    screen.compare_center_guard(r["center_band_screen"],center["result"]["candidate"]["coefficient_guard"])
    rows=analyze_stencils(r)
    screen._equal(small(stage,"STATUS"),b"success\n","STATUS")
    expected=dict(status="success",scope=r["scope"],source_commit=source_commit,
        deployment_sha256=deployment_sha256,gradient_result_sha256=gradient_result_sha256,
        gradient_acceptance_sha256=gradient_acceptance_sha256,result_sha256=result_sha256,physical_release_gate="hold")
    screen.checks._same_json(screen._json(small(stage,"PROVENANCE.json")),expected,"constraint provenance")
    if ([x["label"] for x in r["load_records"]]!=[1,2,3,6,7,8,11,28] or
            any(x["mapping_sha256"]!=g["load_records"][i]["mapping_sha256"] or
                _finite(x["seconds"])<=0 for i,x in enumerate(r["load_records"]))):
        raise ValueError("cache mapping or coverage mismatch")
    return center,g,r,rows


def load_accepted_screen(stage,result_sha256,acceptance_sha256,deployment_sha256,source_commit,
                         gradient_result_sha256,gradient_acceptance_sha256):
    center,g,r,rows=load_screen(stage,result_sha256,deployment_sha256,source_commit,
                               gradient_result_sha256,gradient_acceptance_sha256)
    stage=screen._root(stage)
    small=screen.refresh.accepted._small
    a=screen._json(small(stage,"CONSTRAINT_SCREEN_ACCEPTANCE.json",screen._digest(acceptance_sha256)))
    _expect(a,dict(status="success",scope="saved_constraint_screen_acceptance_no_replay_not_pbe",
        result_sha256=result_sha256,deployment_sha256=deployment_sha256,source_commit=source_commit,
        gradient_result_sha256=gradient_result_sha256,gradient_acceptance_sha256=gradient_acceptance_sha256,
        coefficient_sha256=r["coefficient_sha256"],accepted_result_sha256=r["accepted_result_sha256"],
        stderr_gate="empty",new_candidate_count=0,new_scf_count=0,physical_release_gate="hold"))
    screen.checks._same_json(a["radials"],rows,"independent stencil analysis")
    job=screen.checks._scheduler(a)
    screen._equal(small(stage,"slurm-"+job+".err"),b"","screen stderr")
    return center,g,r,rows
