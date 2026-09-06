"""Stage and collect bounded, diagnostic-only C direction PBE probes.

Keep all output directories in one campaign parent: preparation reserves one
of the eight T/N signed-radius slots there. No jobs, optimizer steps, or
Galerkin evaluations are performed. The accepted quarter SCF is reused.
"""

import json
import math
from pathlib import Path
import re
import tempfile

import check_c_optimized_pbe as endpoint


PREPARATION = "DIRECTION_PROBE_PBE_PREPARED.json"
COLLECTION = "DIRECTION_PROBE_PBE_COLLECTION.json"
SCOPE = "direction_calibration_probe"
RADII = (-.002, -.001, .001, .002)
_IDENTITY = ("direction_name", "signed_radius", "center_result_sha256",
             "center_coefficient_sha256", "center_orbital_sha256", "directions_sha256",
             "coefficient_sha256", "orbital_sha256")
_CENTER = ".provenance/center/"
_BACKOFF_LAYOUT = {
    "backoff_RESULT.json": "RESULT.json",
    "BACKOFF_PROVENANCE.json": "BACKOFF_PROVENANCE.json",
    "backoff_ORIGINAL_COEFFICIENTS.txt": "ORIGINAL_COEFFICIENTS.txt",
    "backoff_INPUT_FREEZE.json": "INPUT_FREEZE.json",
    "backoff_ACTIVE_DATA_CACHE.json": "ACTIVE_DATA_CACHE.json",
    "INTERPOLATED_COEFFICIENTS.txt": "INTERPOLATED_COEFFICIENTS.txt",
    "C_3s3p2d_interpolated.orb": "C_3s3p2d_interpolated.orb",
    "INITIAL_RETRACTED_COEFFICIENTS.txt": "INITIAL_RETRACTED_COEFFICIENTS.txt",
    "C_3s3p2d_initial_retracted.orb": "C_3s3p2d_initial_retracted.orb",
}
for _name in ("RESULT.json", "BEST_CHECKPOINT.json", "ORBITAL_RESULTS.txt",
              "BEST_ORBITAL_CHECKPOINT.txt", "C_3s3p2d_optimized.orb"):
    _BACKOFF_LAYOUT["backoff_parent_" + _name] = "parent/" + _name


def _json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value: " + value)

    result = json.loads(content, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    # Reject numeric overflow such as 1e999 as well as explicit NaN tokens.
    json.dumps(result, allow_nan=False)
    return result


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase SHA256 digest required")
    return value


def _hashed(root, name, digest):
    return endpoint._read_hashed(endpoint._within(root, name), _digest(digest))


def _group(root, values):
    if not isinstance(values, dict) or not values:
        raise ValueError("nonempty input/evidence hash map required")
    return {name: _hashed(root, name, digest) for name, digest in values.items()}


def _equal(actual, expected, name):
    if actual != expected:
        raise ValueError(name + " mismatch")


def _energy(actual, expected, name):
    if not math.isclose(endpoint._finite(actual, name), expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(name + " mismatch")


def _strict_log(content):
    result = endpoint._scf_log(content)
    _equal(result["band_counts"], {"occupied": 4, "total": 44}, "required C band counts")
    return result


def _backoff(evidence, digest):
    # Endpoint archives flatten parent filenames. Reconstruct only the known
    # layout so the existing strict backoff/parent validator remains authoritative.
    with tempfile.TemporaryDirectory(prefix="c-probe-center-") as directory:
        root = Path(directory)
        (root / "parent").mkdir()
        for archived, name in _BACKOFF_LAYOUT.items():
            content = evidence.get(".provenance/" + archived)
            if content is None:
                raise ValueError("missing center backoff evidence: " + archived)
            (root / name).write_bytes(content)
        result, orbital, artifacts = endpoint._optimizer_artifacts(root / "RESULT.json", digest)
    _equal(result.get("scope"), endpoint.BACKOFF_SCOPE, "center backoff scope")
    _equal(result.get("alpha"), .25, "accepted quarter alpha")
    return result, orbital, artifacts


def _center_snapshot(root, preparation_sha256, collection_sha256):
    preparation = _hashed(root, endpoint.PREPARATION, preparation_sha256)
    collection = _hashed(root, endpoint.COLLECTION, collection_sha256)
    manifest, measured = _json(preparation), _json(collection)
    for key, value in (("status", "prepared"), ("scope", endpoint.SCOPE),
                       ("atoms_per_cell", 2), ("candidate_scope", endpoint.BACKOFF_SCOPE),
                       ("candidate_alpha", .25), ("physical_release_gate", "hold"),
                       ("tolerance_ev_per_c", endpoint.TOLERANCE_EV_PER_C)):
        _equal(manifest.get(key), value, "center preparation " + key)
    inputs = _group(root, manifest.get("prepared_input_sha256"))
    evidence = _group(root, manifest.get("evidence_sha256"))
    if not {"INPUT", "STRU", "KPT"}.issubset(inputs):
        raise ValueError("center is missing frozen INPUT/STRU/KPT")
    pseudo, orbital_name = endpoint._c2_files(inputs["STRU"])
    names = {"INPUT", "STRU", "KPT", pseudo, orbital_name}
    _equal(set(inputs), names, "center input file set")
    if {pseudo, orbital_name} & {PREPARATION, COLLECTION, endpoint.COLLECTION}:
        raise ValueError("probe input filename collision")
    if any(not content for content in inputs.values()):
        raise ValueError("empty center input")
    values = endpoint._input_values(inputs["INPUT"])
    _equal(values, endpoint._pure_pbe(values), "center pure PBE atomic initialization")
    original_hashes = manifest.get("original_input_sha256")
    if not isinstance(original_hashes, dict) or set(original_hashes) != names:
        raise ValueError("incomplete center original input hashes")
    originals = {}
    for name in names:
        content = evidence.get(".provenance/original_" + name)
        if content is None or endpoint._sha256(content) != _digest(original_hashes[name]):
            raise ValueError("original center input SHA256 mismatch: " + name)
        originals[name] = content
    _equal(values, endpoint._pure_pbe(endpoint._input_values(originals["INPUT"])),
           "center versus original PBE numerics")
    for name in ("STRU", "KPT", pseudo):
        _equal(inputs[name], originals[name], "unchanged center " + name)
    result, orbital, artifacts = _backoff(evidence, _digest(manifest.get("candidate_result_sha256")))
    required_evidence = {".provenance/" + name for name in artifacts}
    required_evidence.update(".provenance/original_" + name for name in names)
    required_evidence.add(".provenance/original_running_scf.log")
    _equal(set(evidence), required_evidence, "complete center evidence file set")
    for name, content in artifacts.items():
        _equal(evidence[".provenance/" + name], content, "center artifact " + name)
    _equal(inputs[orbital_name], orbital, "center orbital bytes")
    for key, value in (("candidate_orbital_filename", orbital_name),
                       ("candidate_orbital_sha256", result["orbital_sha256"]),
                       ("candidate_coefficient_sha256", result["coefficient_sha256"]),
                       ("parent_optimizer_result_sha256", result["parent_result_sha256"])):
        _equal(manifest.get(key), value, "center " + key)
    baseline = _strict_log(evidence[".provenance/original_running_scf.log"])
    saved_baseline = manifest.get("baseline", {})
    for key in ("log_sha256", "band_counts", "energy_record"):
        _equal(saved_baseline.get(key), baseline[key], "center baseline " + key)
    _energy(saved_baseline.get("energy_ev"), baseline["energy_ev"], "center baseline energy")
    log_name = "OUT." + endpoint._basename(values.get("suffix", "ABACUS")) + "/running_scf.log"
    _equal(manifest.get("candidate_log_relative_path"), log_name, "center SCF log path")
    log_content = _hashed(root, log_name, measured.get("candidate_log_sha256"))
    center = _strict_log(log_content)
    delta = (center["energy_ev"] - baseline["energy_ev"])/2
    for key, value in (("status", "collected"), ("scope", endpoint.SCOPE),
                       ("preparation_sha256", preparation_sha256),
                       ("scf_log_gate", "pass"), ("band_count_check", "pass"),
                       ("band_counts", center["band_counts"]), ("pbe_total_energy_gate", "pass"),
                       ("energy_quantity", "PBE_total_energy_not_RPA_E0"),
                       ("tolerance_ev_per_c", endpoint.TOLERANCE_EV_PER_C),
                       ("baseline_log_sha256", baseline["log_sha256"]),
                       ("physical_release_gate", "hold")):
        _equal(measured.get(key), value, "center collection " + key)
    for key in ("candidate_scope", "candidate_alpha", "candidate_result_sha256",
                "candidate_orbital_sha256", "parent_optimizer_result_sha256"):
        _equal(measured.get(key), manifest[key], "center collection " + key)
    for key, value in (("baseline_energy_ev", baseline["energy_ev"]),
                       ("candidate_energy_ev", center["energy_ev"]), ("energy_delta_ev_per_c", delta)):
        _energy(measured.get(key), value, "center collection " + key)
    if not math.isfinite(delta) or abs(delta) > endpoint.TOLERANCE_EV_PER_C + 1e-12:
        raise ValueError("center violates actual PBE 10 meV/C constraint")
    files = dict(inputs, **evidence)
    files.update({endpoint.PREPARATION: preparation, endpoint.COLLECTION: collection, log_name: log_content})
    return manifest, baseline, center, files


def _probe(root, name, digest, center):
    content = _hashed(root, name, digest)
    probe = _json(content)
    for key, value in (("status", "prepared"), ("scope", SCOPE), ("physical_release_gate", "hold"),
                       ("coefficient_filename", "COEFFICIENTS.txt"),
                       ("orbital_filename", "C_3s3p2d_probe.orb")):
        _equal(probe.get(key), value, "probe " + key)
    radius = probe.get("signed_radius")
    if (probe.get("direction_name") not in ("T", "N") or type(radius) not in (int, float)
            or radius not in RADII):
        raise ValueError("probe must use a bounded T/N signed-radius slot")
    if not isinstance(probe.get("cheap_gate"), dict) or probe["cheap_gate"].get("gate") is not True:
        raise ValueError("probe cheap gate must be true")
    _equal(probe.get("galerkin_energy", "unmeasured"), "unmeasured", "probe Galerkin energy status")
    for key in _IDENTITY[2:]:
        _digest(probe.get(key))
    for name in ("result", "coefficient", "orbital"):
        _equal(probe["center_" + name + "_sha256"], center["candidate_" + name + "_sha256"],
               "probe center " + name)
    files = {"PROBE.json": content}
    for kind in ("coefficient", "orbital"):
        filename = probe[kind + "_filename"]
        files[filename] = _hashed(root, filename, probe[kind + "_sha256"])
        if not files[filename]:
            raise ValueError("empty probe " + kind)
    return probe, files


def _inputs(center, center_files, probe_files):
    inputs = {name: center_files[name] for name in center["prepared_input_sha256"]}
    inputs[center["candidate_orbital_filename"]] = probe_files["C_3s3p2d_probe.orb"]
    return inputs


def _prepared(root, digest):
    manifest = _json(_hashed(root, PREPARATION, digest))
    for key, value in (("status", "prepared"), ("scope", SCOPE), ("atoms_per_cell", 2),
                       ("physical_release_gate", "hold"),
                       ("galerkin_energy", "unmeasured"),
                       ("tolerance_ev_per_c", endpoint.TOLERANCE_EV_PER_C)):
        _equal(manifest.get(key), value, "probe preparation " + key)
    declared = _group(root, manifest.get("evidence_sha256"))
    center, baseline, measured, files = _center_snapshot(
        endpoint._within(root, _CENTER), manifest.get("center_preparation_sha256"),
        manifest.get("center_collection_sha256"))
    probe, probe_files = _probe(endpoint._within(root, ".provenance"), "PROBE.json",
                                 manifest.get("probe_sha256"), center)
    expected = {_CENTER + name: content for name, content in files.items()}
    expected.update({".provenance/" + name: content for name, content in probe_files.items()})
    _equal(declared, expected, "complete frozen probe evidence")
    inputs = _group(root, manifest.get("prepared_input_sha256"))
    _equal(inputs, _inputs(center, files, probe_files), "probe input replacement")
    for key in _IDENTITY:
        _equal(manifest.get(key), probe[key], "frozen probe " + key)
    _equal(manifest.get("candidate_log_relative_path"), center["candidate_log_relative_path"], "probe log path")
    _energy(manifest.get("baseline_energy_ev"), baseline["energy_ev"], "probe original baseline")
    _energy(manifest.get("center_energy_ev"), measured["energy_ev"], "probe center energy")
    return manifest, baseline, measured


def _check_slots(parent, manifest):
    paths = list(parent.glob("*/" + PREPARATION))
    if len(paths) >= 8:
        raise ValueError("eight-probe campaign budget exhausted")
    for path in paths:
        previous, _, _ = _prepared(path.parent.resolve(), endpoint._sha256(path.read_bytes()))
        for key in ("center_preparation_sha256", "center_collection_sha256", "directions_sha256"):
            _equal(previous[key], manifest[key], "single campaign " + key)
        if (previous["direction_name"], previous["signed_radius"]) == (
                manifest["direction_name"], manifest["signed_radius"]):
            raise ValueError("duplicate direction/radius probe in campaign")


def prepare_probe_pbe(*, center_dir, center_preparation_sha256, center_collection_sha256,
                      probe_path, probe_sha256, output):
    """Stage one probe in a new directory; sibling outputs share an eight-slot budget."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    center, baseline, measured, files = _center_snapshot(
        Path(center_dir).resolve(), center_preparation_sha256, center_collection_sha256)
    source = Path(probe_path)
    _equal(source.name, "PROBE.json", "probe manifest filename")
    probe, probe_files = _probe(source.parent.resolve(), source.name, probe_sha256, center)
    inputs = _inputs(center, files, probe_files)
    evidence = {_CENTER + name: content for name, content in files.items()}
    evidence.update({".provenance/" + name: content for name, content in probe_files.items()})
    manifest = dict(format_version=1, status="prepared", scope=SCOPE, atoms_per_cell=2,
        center_preparation_sha256=center_preparation_sha256, center_collection_sha256=center_collection_sha256,
        probe_sha256=probe_sha256, prepared_input_sha256={n: endpoint._sha256(c) for n, c in inputs.items()},
        evidence_sha256={n: endpoint._sha256(c) for n, c in evidence.items()},
        baseline_energy_ev=baseline["energy_ev"], center_energy_ev=measured["energy_ev"],
        candidate_log_relative_path=center["candidate_log_relative_path"],
        energy_quantity="PBE_total_energy_not_RPA_E0", tolerance_ev_per_c=endpoint.TOLERANCE_EV_PER_C,
        galerkin_energy="unmeasured", scheduler_gate="pending_external_validation", physical_release_gate="hold")
    manifest.update({key: probe[key] for key in _IDENTITY})
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ".direction_probe_pbe_prepare.lock"
    lock.mkdir()
    try:
        _check_slots(output.parent, manifest)
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


def collect_probe_pbe(*, prepared_dir, preparation_sha256):
    """Collect actual SCF evidence; a constraint failure remains a collected diagnostic."""
    root = Path(prepared_dir).resolve()
    if (root / COLLECTION).exists() or (root / COLLECTION).is_symlink():
        raise FileExistsError(root / COLLECTION)
    manifest, baseline, center = _prepared(root, preparation_sha256)
    log_path = endpoint._within(root, manifest["candidate_log_relative_path"])
    candidate = _strict_log(log_path.read_bytes())
    delta = (candidate["energy_ev"] - baseline["energy_ev"])/2
    center_delta = (candidate["energy_ev"] - center["energy_ev"])/2
    if not math.isfinite(delta) or not math.isfinite(center_delta):
        raise ValueError("nonfinite PBE energy difference")
    result = dict(status="collected", scope=SCOPE, preparation_sha256=preparation_sha256,
        probe_sha256=manifest["probe_sha256"], center_preparation_sha256=manifest["center_preparation_sha256"],
        center_collection_sha256=manifest["center_collection_sha256"],
        baseline_energy_ev=baseline["energy_ev"], center_energy_ev=center["energy_ev"],
        candidate_energy_ev=candidate["energy_ev"], energy_delta_ev_per_c=delta,
        delta_from_center_ev_per_c=center_delta, energy_quantity="PBE_total_energy_not_RPA_E0",
        tolerance_ev_per_c=endpoint.TOLERANCE_EV_PER_C,
        comparison="absolute_probe_minus_original_per_C_lte_tolerance",
        pbe_gate="pass" if abs(delta) <= endpoint.TOLERANCE_EV_PER_C + 1e-12 else "fail",
        scf_log_gate="pass", band_count_check="pass", band_counts=candidate["band_counts"],
        candidate_log_path=str(log_path), candidate_log_sha256=candidate["log_sha256"],
        baseline_log_sha256=baseline["log_sha256"], center_log_sha256=center["log_sha256"],
        galerkin_energy="unmeasured", scheduler_gate="pending_external_validation", physical_release_gate="hold")
    result.update({key: manifest[key] for key in _IDENTITY})
    endpoint._read_hashed(log_path, candidate["log_sha256"])
    endpoint._write_json(root / COLLECTION, result)
    return result
