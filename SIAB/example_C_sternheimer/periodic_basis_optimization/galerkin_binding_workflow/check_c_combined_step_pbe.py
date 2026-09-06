"""Stage/collect one actual-PBE tangent candidate, never a calibration probe.

Use a dedicated, initially empty output parent for the single candidate slot.
No scheduler, electronic-structure, direction construction, or RPA calls occur.
"""

import math
from pathlib import Path
import sys

import check_c_optimized_pbe as endpoint
from check_c_direction_probe_pbe import (
    _center_snapshot, _json, _digest, _hashed, _group, _equal, _strict_log,
)


PREPARATION = "COMBINED_STEP_PBE_PREPARED.json"
COLLECTION = "COMBINED_STEP_PBE_COLLECTION.json"
SCOPE = "actual_pbe_tangent_candidate"
_CENTER = ".provenance/center/"
_LOCK = ".combined_step_pbe_prepare.lock"
_BINDINGS = ("center_result_sha256", "center_coefficient_sha256", "center_orbital_sha256")
_IDENTITY = ("direction_name", "radius", "mixing_ratio") + _BINDINGS + (
    "directions_sha256", "calibration_sha256", "coefficient_sha256", "orbital_sha256")
_COMMON = dict(scope=SCOPE, atoms_per_cell=2, physical_release_gate="hold",
    galerkin_energy="unmeasured", energy_quantity="PBE_total_energy_not_RPA_E0",
    tolerance_ev_per_c=endpoint.TOLERANCE_EV_PER_C, scheduler_gate="pending_external_validation")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite real number")
    return value


def _near(actual, expected, name):
    if not math.isclose(_number(actual, name), _number(expected, name), rel_tol=0, abs_tol=1e-12):
        raise ValueError(name + " mismatch")


def _regular(root, name):
    endpoint._within(root, name)
    path = root
    for part in Path(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink evidence/input forbidden: " + str(path))
    if not path.is_file():
        raise ValueError("regular evidence/input file required: " + str(path))


def _checked_group(root, values):
    content = _group(root, values)
    for name in content:
        _regular(root, name)
    return content


def _calibration(calibration, candidate, center, baseline, measured, prep_sha, coll_sha):
    for key, value in (("status", "success"),
                       ("scope", "actual_pbe_direction_calibration_not_optimization"),
                       ("consistency_gate", "pass"), ("center_collection_sha256", coll_sha)):
        _equal(calibration.get(key), value, "calibration " + key)
    if type(calibration.get("actual_scf_count")) is not int or calibration["actual_scf_count"] != 8:
        raise ValueError("calibration requires exactly eight actual SCFs")
    _near(calibration.get("center_energy_ev"), measured["energy_ev"], "calibration center energy")
    samples = calibration.get("samples")
    if not isinstance(samples, list) or len(samples) != 8:
        raise ValueError("exact eight calibration samples required")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("calibration sample object required")
        radius = _number(sample.get("signed_radius"), "sample signed radius")
        if sample.get("direction_name") not in ("T", "N") or radius not in (-.002, -.001, .001, .002):
            raise ValueError("sample must be an approved T/N signed-radius probe")
        for key in _BINDINGS + ("directions_sha256",):
            _equal(_digest(sample.get(key)), candidate[key], "sample " + key)
        for key, value in (("status", "collected"), ("scope", "direction_calibration_probe"),
                           ("center_preparation_sha256", prep_sha), ("center_collection_sha256", coll_sha),
                           ("pbe_gate", "pass"), ("scf_log_gate", "pass"), ("band_count_check", "pass"),
                           ("band_counts", dict(occupied=4, total=44)), ("physical_release_gate", "hold"),
                           ("energy_quantity", _COMMON["energy_quantity"]),
                           ("tolerance_ev_per_c", endpoint.TOLERANCE_EV_PER_C)):
            _equal(sample.get(key), value, "sample " + key)
        energy = _number(sample.get("candidate_energy_ev"), "sample PBE energy")
        delta = (energy-baseline["energy_ev"])/2
        for key, value in (("baseline_energy_ev", baseline["energy_ev"]),
                           ("center_energy_ev", measured["energy_ev"]), ("energy_delta_ev_per_c", delta),
                           ("delta_from_center_ev_per_c", (energy-measured["energy_ev"])/2)):
            _near(sample.get(key), value, "sample " + key)
        if abs(delta) > endpoint.TOLERANCE_EV_PER_C + 1e-12:
            raise ValueError("sample falsely claims PBE 10 meV/C gate pass")
    # The existing analyzer is authoritative; do not duplicate its finite-difference math.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "opt_orb_pytorch_dpsi"))
    from periodic_galerkin_direction_calibration import analyze_pbe_axes
    analysis = analyze_pbe_axes(measured["energy_ev"], samples)
    _equal(analysis["consistency_gate"], "pass", "recomputed calibration consistency")
    for key, value in analysis.items():
        _equal(calibration.get(key), value, "recomputed calibration " + key)
    normal = analysis["axes"]["N"][0]["derivative_ev_per_c"]
    tangent = analysis["axes"]["T"][0]["derivative_ev_per_c"]
    if normal == 0:
        raise ValueError("zero normal PBE derivative cannot define mixing ratio")
    _near(candidate.get("mixing_ratio"), tangent/normal, "actual PBE mixing ratio at radius .001")


def _candidate(root, digest, center, baseline, measured, prep_sha, coll_sha):
    _regular(root, "CANDIDATE.json")
    content = _hashed(root, "CANDIDATE.json", digest)
    candidate = _json(content)
    for key, value in (("status", "prepared"), ("scope", SCOPE),
                       ("direction_name", "actual_pbe_tangent"), ("physical_release_gate", "hold"),
                       ("galerkin_energy", "unmeasured"), ("coefficient_filename", "COEFFICIENTS.txt"),
                       ("orbital_filename", "C_3s3p2d_combined.orb")):
        _equal(candidate.get(key), value, "candidate " + key)
    _equal(_number(candidate.get("radius"), "candidate radius"), .02, "fixed combined radius")
    _number(candidate.get("mixing_ratio"), "candidate mixing ratio")
    if not isinstance(candidate.get("cheap_gate"), dict) or candidate["cheap_gate"].get("gate") is not True:
        raise ValueError("candidate cheap gate must be true")
    for key in _IDENTITY[3:]:
        _digest(candidate.get(key))
    for name in ("result", "coefficient", "orbital"):
        _equal(candidate["center_" + name + "_sha256"], center["candidate_" + name + "_sha256"],
               "candidate center " + name)
    files = _checked_group(root, {
        "DIRECTIONS.json": candidate["directions_sha256"],
        "CALIBRATION.json": candidate["calibration_sha256"],
        "COEFFICIENTS.txt": candidate["coefficient_sha256"],
        "C_3s3p2d_combined.orb": candidate["orbital_sha256"],
    })
    if any(not value for value in files.values()):
        raise ValueError("empty combined candidate artifact")
    directions = _json(files["DIRECTIONS.json"])
    for key in _BINDINGS:
        _equal(_digest(directions.get(key)), candidate[key], "directions " + key)
    for key, value in (("status", "success"), ("scope", "two_direction_actual_pbe_calibration"),
                       ("physical_release_gate", "hold")):
        _equal(directions.get(key), value, "directions " + key)
    _calibration(_json(files["CALIBRATION.json"]), candidate, center, baseline, measured, prep_sha, coll_sha)
    files["CANDIDATE.json"] = content
    return candidate, files


def _inputs(center, center_files, candidate_files):
    inputs = {name: center_files[name] for name in center["prepared_input_sha256"]}
    if set(inputs) & {PREPARATION, COLLECTION, ".provenance"}:
        raise ValueError("combined candidate input filename collision")
    inputs[center["candidate_orbital_filename"]] = candidate_files["C_3s3p2d_combined.orb"]
    return inputs


def _root(path):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("symlink candidate/center directory forbidden")
    return path.resolve()


def _prepared(root, digest):
    """Runtime preflight: revalidate archived center, calibration math and exact inputs."""
    root = _root(root)
    _regular(root, PREPARATION)
    manifest = _json(_hashed(root, PREPARATION, digest))
    for key, value in dict(_COMMON, status="prepared", format_version=1).items():
        _equal(manifest.get(key), value, "combined preparation " + key)
    evidence = _checked_group(root, manifest.get("evidence_sha256"))
    center, baseline, measured, center_files = _center_snapshot(
        endpoint._within(root, _CENTER), manifest.get("center_preparation_sha256"),
        manifest.get("center_collection_sha256"))
    candidate, candidate_files = _candidate(endpoint._within(root, ".provenance"),
        manifest.get("candidate_sha256"), center, baseline, measured,
        manifest["center_preparation_sha256"], manifest["center_collection_sha256"])
    expected = {_CENTER + name: content for name, content in center_files.items()}
    expected.update({".provenance/" + name: content for name, content in candidate_files.items()})
    _equal(evidence, expected, "complete frozen combined evidence")
    inputs = _checked_group(root, manifest.get("prepared_input_sha256"))
    _equal(inputs, _inputs(center, center_files, candidate_files), "combined input replacement")
    for key in _IDENTITY:
        _equal(manifest.get(key), candidate[key], "frozen combined " + key)
    _equal(manifest.get("candidate_log_relative_path"), center["candidate_log_relative_path"], "combined log path")
    _near(manifest.get("baseline_energy_ev"), baseline["energy_ev"], "combined original baseline")
    _near(manifest.get("center_energy_ev"), measured["energy_ev"], "combined center energy")
    return manifest, baseline, measured


def prepare_combined_pbe(*, center_dir, center_preparation_sha256, center_collection_sha256,
                         candidate_path, candidate_sha256, output):
    """Reserve one slot under an empty parent, retaining incomplete slots on failure."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = _root(output)
    root = _root(center_dir)
    center, baseline, measured, center_files = _center_snapshot(
        root, center_preparation_sha256, center_collection_sha256)
    for name in center_files:
        _regular(root, name)
    source = Path(candidate_path)
    _equal(source.name, "CANDIDATE.json", "candidate manifest filename")
    candidate, candidate_files = _candidate(_root(source.parent), candidate_sha256,
        center, baseline, measured, center_preparation_sha256, center_collection_sha256)
    inputs = _inputs(center, center_files, candidate_files)
    evidence = {_CENTER + name: content for name, content in center_files.items()}
    evidence.update({".provenance/" + name: content for name, content in candidate_files.items()})
    manifest = dict(_COMMON, format_version=1, status="prepared",
        center_preparation_sha256=center_preparation_sha256, center_collection_sha256=center_collection_sha256,
        candidate_sha256=candidate_sha256, prepared_input_sha256={n: endpoint._sha256(c) for n, c in inputs.items()},
        evidence_sha256={n: endpoint._sha256(c) for n, c in evidence.items()},
        baseline_energy_ev=baseline["energy_ev"], center_energy_ev=measured["energy_ev"],
        candidate_log_relative_path=center["candidate_log_relative_path"])
    manifest.update({key: candidate[key] for key in _IDENTITY})
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / _LOCK
    lock.mkdir()
    try:
        # An empty/incomplete or symlink sibling still consumes the only slot.
        if any(path != lock for path in output.parent.iterdir()):
            raise ValueError("single candidate campaign slot already occupied")
        output.mkdir()
        for name, content in dict(inputs, **evidence).items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
        endpoint._write_json(output / PREPARATION, manifest)
    finally:
        lock.rmdir()
    return manifest


def collect_combined_pbe(*, prepared_dir, preparation_sha256):
    """Record actual PBE diagnostics; SCF failure raises and no evidence is overwritten."""
    root = _root(prepared_dir)
    destination = root / COLLECTION
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest, baseline, center = _prepared(root, preparation_sha256)
    name = manifest["candidate_log_relative_path"]
    log_path = endpoint._within(root, name)
    content = log_path.read_bytes()
    _regular(root, name)
    candidate = _strict_log(content)
    delta = (candidate["energy_ev"]-baseline["energy_ev"])/2
    center_delta = (candidate["energy_ev"]-center["energy_ev"])/2
    _number(delta, "PBE energy difference")
    _number(center_delta, "PBE center difference")
    result = dict(_COMMON, status="collected", preparation_sha256=preparation_sha256,
        candidate_sha256=manifest["candidate_sha256"],
        center_preparation_sha256=manifest["center_preparation_sha256"],
        center_collection_sha256=manifest["center_collection_sha256"],
        baseline_energy_ev=baseline["energy_ev"], center_energy_ev=center["energy_ev"],
        candidate_energy_ev=candidate["energy_ev"], energy_delta_ev_per_c=delta,
        delta_from_center_ev_per_c=center_delta,
        comparison="absolute_candidate_minus_original_per_C_lte_tolerance",
        pbe_gate="pass" if abs(delta) <= endpoint.TOLERANCE_EV_PER_C + 1e-12 else "fail",
        scf_log_gate="pass", band_count_check="pass", band_counts=candidate["band_counts"],
        candidate_log_path=str(log_path), candidate_log_sha256=candidate["log_sha256"],
        baseline_log_sha256=baseline["log_sha256"], center_log_sha256=center["log_sha256"])
    result.update({key: manifest[key] for key in _IDENTITY})
    endpoint._read_hashed(log_path, candidate["log_sha256"])
    endpoint._write_json(destination, result)
    return result
