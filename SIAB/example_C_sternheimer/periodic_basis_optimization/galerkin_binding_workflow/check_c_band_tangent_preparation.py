"""Read-only admission of eight saved C band-tangent preparation probes.

Archived sources are verified as evidence, never executed or compared with this
checkout. Only the small signed-QR coefficient construction is repeated; no
cache, diagonalization, SCF, response, backward or scheduler replay is performed.
The authoritative parent loader retains all original quarter-center checks
(including its legacy temporary evidence reconstruction). Nothing in a stage is
written. Probe orbitals are hash-verified, not re-exported.

BAND_TANGENT_ACCEPTANCE.json is external. Required fields are status='success',
scope='saved_band_tangent_preparation_acceptance_no_replay', result_sha256,
source_commit, deployment_sha256, directions_sha256, center_result_sha256,
center_coefficient_sha256, center_orbital_sha256, stderr_gate='empty',
physical_release_gate='hold', job_id (decimal string), and scheduler (three
pipe-delimited parent/batch/extern COMPLETED|0:0 rows). Other audit notes are
permitted. Acceptance never constitutes final RPA, actual probe PBE or basis
release, nor proof of smooth eigenvalue branches.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import check_c_ec_constraint_screen as admission
from prepare_c_band_tangent_probes import summarize_target_branches


screen = admission.screen
checks = screen.checks
_small = screen.refresh.accepted._small
_json, _digest, _equal = screen._json, screen._digest, screen._equal
RESULT = "result/BAND_TANGENT_PREPARED.json"
ACCEPTANCE = "BAND_TANGENT_ACCEPTANCE.json"
_SLOTS = [(name, radius) for name in ("T", "N") for radius in (-.001, .001, -.002, .002)]
_LABELS = [1, 2, 3, 6, 7, 8, 11, 28]
_CENTER_HASHES = ("center_result_sha256", "center_coefficient_sha256", "center_orbital_sha256")
_DIRECTION_BINDINGS = ("source_commit", "screen_result_sha256", "screen_acceptance_sha256") + _CENTER_HASHES
_BAND_KEYS = (
    "gate", "scope", "scf_pbe_gate", "occupied_band_sum_change_ev_per_atom",
    "maximum_target_band_change_ev", "minimum_gap_ev", "occupied_band_sum_limit_ev_per_atom",
    "target_band_change_limit_ev", "k_weight_sum", "k_weight_convention")
_SCOPE = dict(status="success", scope="bounded_current_center_band_tangent_probe_preparation",
    preparation_gate="pass", cache_loads=1, probe_count=8, actual_scf_count=0,
    rpa_evaluations=0, backward_passes=0, optimizer_steps=0, physical_candidate_count=0,
    coefficient_update="none", physical_release_gate="hold")
_RESULT_KEYS = set(_SCOPE) | set(_DIRECTION_BINDINGS) | {
    "deployment_sha256", "screen_stage", "gradient_result_sha256", "gradient_acceptance_sha256",
    "center_actual_pbe", "directions_sha256", "probes", "center_band_detail", "branch_diagnostic",
    "load_records", "cache_load_seconds", "total_seconds", "peak_rss_kib"}
_REQUIRED_SOURCE = {screen.refresh._WORKFLOW+n for n in (
    "prepare_c_band_tangent_probes.py", "run_c_band_tangent_preparation.slurm",
    "check_c_ec_constraint_screen.py", "screen_c_ec_step_constraints.py",
    "run_c_ec_constraint_screen.slurm", "prepare_c_pbe_direction_calibration.py")}
_REQUIRED_SOURCE.update("SIAB/opt_orb_pytorch_dpsi/"+n for n in (
    "periodic_galerkin_band_tangent.py", "periodic_galerkin_ec_gradient_step.py",
    "periodic_galerkin_pbe_guard.py", "periodic_galerkin_fit.py"))
_REQUIRED_SOURCE.add("SIAB/example_C_sternheimer/periodic_basis_optimization/export_periodic_orbitals.py")


def _keys(record, expected, name):
    if not isinstance(record, dict) or set(record) != set(expected):
        raise ValueError("exact " + name + " fields required")


def _expect(record, expected, name):
    if not isinstance(record, dict) or not set(expected).issubset(record):
        raise ValueError(name + " object/fields required")
    checks._same_json({k:record[k] for k in expected}, expected, name)


def _runtime():
    # Do not import a preparation runner's cache-backed numerical runtime.
    import torch
    from periodic_galerkin_basis import read_periodic_optimizer_coefficients
    from periodic_galerkin_band_tangent import build_band_tangent
    from periodic_galerkin_direction_calibration import signed_probes
    return SimpleNamespace(torch=torch, read_coefficients=read_periodic_optimizer_coefficients,
                           build_band_tangent=build_band_tangent, signed_probes=signed_probes)


def _band_detail(record):
    _keys(record, set(_BAND_KEYS) | {"target_band_identity", "target_band_details"}, "band detail")
    admission._guard(record)
    _expect(record, dict(target_band_identity="sorted_eigenvalue_index_not_eigenvector_tracking"),
            "sorted band identity")
    rows = record["target_band_details"]
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError("full 64-k-point band detail required")
    values = []
    # The frozen guard uses q=1: source and target are the same full 4x4x4 grid.
    for ik, row in enumerate(rows, 1):
        _keys(row, ("source_ik", "target_ik", "band_indices", "signed_changes_ev"), "k/band row")
        _expect(row, dict(source_ik=ik, target_ik=ik, band_indices=list(range(1, 9))), "k/band grid")
        changes = row["signed_changes_ev"]
        if not isinstance(changes, list) or len(changes) != 8:
            raise ValueError("all eight signed occupied/target band changes required")
        values.extend(admission._finite(v) for v in changes)
    if not math.isclose(max(map(abs, values)), record["maximum_target_band_change_ev"],
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("saved band detail does not reproduce maximum")


def _cheap_guard(cheap, floor):
    _keys(cheap, ("gate", "band_screen", "minimum_occupied_capture", "occupied_capture_floor",
        "overlap_rank_and_condition_gate", "overlap_relative_rank_tolerance", "overlap_condition_limit"),
        "original cheap guard")
    _expect(cheap, dict(gate=True, occupied_capture_floor=floor, overlap_rank_and_condition_gate="pass",
        overlap_relative_rank_tolerance=1e-12, overlap_condition_limit=1e12), "original cheap protections")
    screen.refresh.accepted._capture(cheap["minimum_occupied_capture"], floor)
    _band_detail(cheap["band_screen"])


def _load_records(result, gradient, parent):
    records = result["load_records"]
    for data in (records, gradient["load_records"], parent["load_records"]):
        if not isinstance(data, list) or len(data) != len(_LABELS):
            raise ValueError("eight ordered q load records required")
        for row, label in zip(data, _LABELS):
            _keys(row, ("label", "mapping_sha256", "seconds"), "q load record")
            _expect(row, dict(label=label), "q label")
            _digest(row["mapping_sha256"])
            if admission._finite(row["seconds"]) <= 0:
                raise ValueError("positive finite cache load timing required")
    for i, row in enumerate(records):
        for previous in (gradient, parent):
            _equal(row["mapping_sha256"], previous["load_records"][i]["mapping_sha256"], "q mapping binding")
    load, total, peak = (admission._finite(result[k]) for k in (
        "cache_load_seconds", "total_seconds", "peak_rss_kib"))
    if (not 0 < load < total < 3600 or not 0 < peak <= 102400*1024
            or math.fsum(row["seconds"] for row in records) > load+1e-9):
        raise ValueError("invalid preparation timing or memory")


def _probe(stage, row, index, result, floor):
    name, radius = _SLOTS[index]
    relative = "probe_%02d/PROBE.json" % index
    _keys(row, ("path", "status", "sha256", "direction_name", "signed_radius"), "probe index")
    _expect(row, dict(path=relative, status="prepared", direction_name=name, signed_radius=radius),
            "ordered signed probe")
    path = stage / "result/probes" / relative
    files = {"PROBE.json":_small(stage, "result/probes/"+relative, _digest(row["sha256"]))}
    manifest = _json(files["PROBE.json"])
    expected = dict(status="prepared", scope="direction_calibration_probe", direction_name=name,
        signed_radius=radius, **{k:result[k] for k in _CENTER_HASHES},
        directions_sha256=result["directions_sha256"], galerkin_energy="unmeasured",
        physical_release_gate="hold", coefficient_filename="COEFFICIENTS.txt",
        orbital_filename="C_3s3p2d_probe.orb")
    _keys(manifest, set(expected) | {"cheap_gate", "coefficient_sha256", "orbital_sha256"}, "probe manifest")
    _expect(manifest, expected, "probe identity")
    _cheap_guard(manifest["cheap_gate"], floor)
    for kind in ("coefficient", "orbital"):
        filename = manifest[kind+"_filename"]
        files[filename] = _small(stage, str(path.parent.relative_to(stage) / filename),
                                 _digest(manifest[kind+"_sha256"]))
        if not files[filename]:
            raise ValueError("empty probe " + kind)
    _equal({p.name for p in path.parent.iterdir()}, set(files), "exact saved probe files")
    return dict(path=path, manifest=manifest, files=files)


def load_preparation(stage, result_sha256, deployment_sha256, source_commit):
    """Return accepted center/gradient, saved result/directions and eight probes.

    Each probe contains path (Path to PROBE.json), manifest, and files mapping
    PROBE.json/COEFFICIENTS.txt/C_3s3p2d_probe.orb to their verified raw bytes.
    Historical parent stages must remain accessible to load_accepted_screen.
    """
    stage = screen._root(stage)
    _digest(result_sha256)
    _digest(deployment_sha256)
    checks._commit(source_commit)
    deployment = checks._archived_source(stage, deployment_sha256, source_commit)
    if not _REQUIRED_SOURCE.issubset(deployment["files"]):
        raise ValueError("archived band-tangent runner and kernel source required")
    result = _json(_small(stage, RESULT, result_sha256))
    _keys(result, _RESULT_KEYS, "preparation result")
    _expect(result, dict(_SCOPE, source_commit=source_commit, deployment_sha256=deployment_sha256),
            "preparation scope and source")
    for key in _RESULT_KEYS:
        if key.endswith("sha256"):
            _digest(result[key])
    _equal(_small(stage, "STATUS"), b"success\n", "preparation STATUS")
    checks._same_json(_json(_small(stage, "PROVENANCE.json")), dict(
        {k:result[k] for k in ("status", "scope", "source_commit", "deployment_sha256",
                              "preparation_gate", "physical_release_gate")}, result_sha256=result_sha256),
        "preparation provenance")
    previous = checks._absolute(result["screen_stage"])
    if stage == previous or stage in previous.parents or previous in stage.parents:
        raise ValueError("separate preparation and parent screen stages required")
    # The pinned parent result supplies its historical deployment/source, not
    # this checkout's source identity and not unpinned acceptance metadata.
    parent = _json(_small(previous, "result/CONSTRAINT_SCREEN.json", result["screen_result_sha256"]))
    center, gradient, admitted, rows = admission.load_accepted_screen(previous,
        result["screen_result_sha256"], result["screen_acceptance_sha256"],
        _digest(parent.get("deployment_sha256")), checks._commit(parent.get("source_commit")),
        result["gradient_result_sha256"], result["gradient_acceptance_sha256"])
    checks._same_json(admitted, parent, "independently admitted constraint screen")
    _expect(parent, {k:result[k] for k in ("gradient_result_sha256", "gradient_acceptance_sha256")},
            "screen gradient binding")
    _expect(result, dict(center_result_sha256=center["result_sha256"],
        center_coefficient_sha256=gradient["coefficient_sha256"],
        center_orbital_sha256=gradient["orbital_sha256"], center_actual_pbe=center["result"]["actual_pbe"]),
        "accepted center binding")
    _expect(gradient, dict(accepted_result_sha256=center["result_sha256"]), "gradient center")
    _band_detail(result["center_band_detail"])
    for guard in (parent["center_band_screen"], center["result"]["candidate"]["coefficient_guard"]):
        screen.compare_center_guard({k:result["center_band_detail"][k] for k in _BAND_KEYS}, guard)
    _load_records(result, gradient, parent)
    artifact = _json(_small(stage, "result/probes/DIRECTIONS.json", result["directions_sha256"]))
    entries = result["probes"]
    if not isinstance(entries, list) or len(entries) != 8:
        raise ValueError("exactly eight ordered saved probes required")
    probes = [_probe(stage, row, i, result, center["occupied_capture_floor"]) for i, row in enumerate(entries)]
    _equal({p.name for p in (stage / "result/probes").iterdir()},
           {"DIRECTIONS.json"} | {"probe_%02d" % i for i in range(8)}, "exact eight-probe directory")
    branches = summarize_target_branches(result["center_band_detail"],
        [p["manifest"]["cheap_gate"]["band_screen"] for p in probes])
    checks._same_json(result["branch_diagnostic"], branches, "independent saved branch analysis")
    for key, identity in (("coefficient_path", "coefficient_sha256"), ("orbital_path", "orbital_sha256")):
        path = Path(center[key])
        _small(screen._root(path.parent), path.name, _digest(gradient[identity]))
        _equal(center["result"][identity], gradient[identity], "accepted center " + identity)
    rt = _runtime()
    rt.torch.set_num_threads(28)
    with rt.torch.no_grad():
        coefficients = screen.common._read_coefficients(rt, center["coefficient_path"])
        expected = rt.build_band_tangent(coefficients, gradient["energy_gradient"], rows)
        expected.update({k:result[k] for k in _DIRECTION_BINDINGS})
        checks._same_json(artifact, expected, "reconstructed band-tangent DIRECTIONS metadata")
        trials = rt.signed_probes(coefficients, artifact)
        if len(trials) != 8:
            raise ValueError("eight reconstructed signed-QR probes required")
        for probe, trial, (name, radius) in zip(probes, trials, _SLOTS):
            _expect(trial, dict(direction_name=name, signed_radius=radius), "signed-QR probe slot")
            path = probe["path"].parent / "COEFFICIENTS.txt"
            saved = screen.common._read_coefficients(rt, path)
            checks._same_json({e:[v.tolist() for v in cs] for e, cs in saved.items()},
                {e:[v.tolist() for v in cs] for e, cs in trial["coefficients"].items()},
                "exact signed-QR probe coefficients")
            _equal(_small(stage, str(path.relative_to(stage))), probe["files"][path.name],
                   "unchanged reconstructed probe bytes")
    return dict(center=center, gradient=gradient, result=result, directions=artifact, probes=probes)


def load_accepted_preparation(stage, result_sha256, acceptance_sha256, deployment_sha256, source_commit):
    """Additionally admit externally pinned acceptance and exact job stderr."""
    _digest(acceptance_sha256)
    loaded = load_preparation(stage, result_sha256, deployment_sha256, source_commit)
    stage = screen._root(stage)
    acceptance = _json(_small(stage, ACCEPTANCE, acceptance_sha256))
    _expect(acceptance, dict(status="success", scope="saved_band_tangent_preparation_acceptance_no_replay",
        result_sha256=result_sha256, source_commit=source_commit, deployment_sha256=deployment_sha256,
        **{k:loaded["result"][k] for k in ("directions_sha256",) + _CENTER_HASHES},
        stderr_gate="empty", physical_release_gate="hold"), "band-tangent acceptance")
    job = checks._scheduler(acceptance)
    _equal(_small(stage, "slurm-"+job+".err"), b"", "band-tangent job stderr")
    return loaded
