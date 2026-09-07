"""Own-scope pure-PBE staging for one accepted-center negative Ec-gradient step.

Only the orbital bytes change. Small proof copies are retained, but the frozen
external center, candidate and runtime must remain available and are revalidated
before collection. No process, SCF, response or scheduler is launched here.
"""

import hashlib
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(SOURCE / "SIAB/opt_orb_pytorch_dpsi"),
               str(SOURCE / "SIAB/example_C_sternheimer/periodic_basis_optimization")]

import check_c_optimized_pbe as endpoint
from check_c_accepted_combined_step import load_accepted_combined_step, _small, _expect
from check_c_combined_step_pbe import _root, _regular, _number, _near
from check_c_direction_probe_pbe import _digest, _equal, _json, _strict_log


PREPARATION = "EC_GRADIENT_STEP_PBE_PREPARED.json"
COLLECTION = "EC_GRADIENT_STEP_PBE_COLLECTION.json"
SCOPE = "accepted_ec_gradient_step_candidate"
ORIGINAL_ENERGY_EV = -309.8590820440637
_FILES = {"COEFFICIENTS.txt": "coefficient_sha256", "C_3s3p2d_ec_step.orb": "orbital_sha256",
          "GRADIENT_STEP.json": "gradient_step_sha256"}
_IDENTITY = ("direction_name", "radius", "center_result_sha256", "center_acceptance_sha256",
             "center_coefficient_sha256", "center_orbital_sha256", "gradient_stage",
             "gradient_result_sha256", "gradient_acceptance_sha256",
             "coefficient_sha256", "orbital_sha256", "gradient_step_sha256")
_COMMON = dict(scope=SCOPE, atoms_per_cell=2, physical_release_gate="hold",
    energy_quantity="PBE_total_energy_not_RPA_E0", galerkin_energy="unmeasured",
    tolerance_ev_per_c=endpoint.TOLERANCE_EV_PER_C, scheduler_gate="pending_external_validation")


def validate_gradient_step_candidate(candidate_root):
    # The parent owns reconstruction and accepted-gradient physics contracts.
    from check_c_ec_gradient_step import validate_gradient_step_candidate as validate
    return validate(candidate_root)


def _runtime_file(path, expected):
    expected = _digest(expected)
    # System MPI installs use directory and SONAME aliases. Pin the real file;
    # the stricter no-symlink rule still applies to scientific evidence/inputs.
    path = Path(path).resolve(strict=True)
    root = _root(path.parent)
    _regular(root, path.name)
    path = root / path.name
    if path.stat().st_size == 0:
        raise ValueError("empty runtime artifact")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    _equal(digest.hexdigest(), expected, "runtime SHA256 " + str(path))
    return str(path)


def _candidate(root, center, center_sha, acceptance_sha, result):
    content = _small(root, "CANDIDATE.json")
    candidate = _json(content)
    _expect(candidate, dict(status="prepared", scope=SCOPE, physical_release_gate="hold",
                           direction_name="negative_horizontal_ec_gradient"), "Ec candidate")
    if _number(candidate.get("radius"), "radius") not in (.02, .018):
        raise ValueError("only the original radius or explicit ten-percent reduction is eligible")
    if not isinstance(candidate.get("center_stage"), str) or _root(candidate["center_stage"]) != center:
        raise ValueError("candidate must bind the accepted combined center stage")
    for key, value in (("center_result_sha256", center_sha), ("center_acceptance_sha256", acceptance_sha),
                       ("center_coefficient_sha256", result["coefficient_sha256"]),
                       ("center_orbital_sha256", result["orbital_sha256"])):
        _equal(_digest(candidate.get(key)), value, "candidate " + key)
    for key in ("gradient_result_sha256", "gradient_acceptance_sha256"):
        _digest(candidate.get(key))
    if not isinstance(candidate.get("gradient_stage"), str) or not Path(candidate["gradient_stage"]).is_absolute():
        raise ValueError("absolute frozen gradient stage required")
    _root(candidate["gradient_stage"])
    for key, value in (("coefficient_filename", "COEFFICIENTS.txt"), ("orbital_filename", "C_3s3p2d_ec_step.orb"),
                       ("actual_pbe_gate", "pending"), ("galerkin_energy", "unmeasured")):
        if key in candidate:
            _equal(candidate[key], value, "candidate " + key)
    files = {"CANDIDATE.json": content}
    for name, key in _FILES.items():
        files[name] = _small(root, name, _digest(candidate.get(key)))
        if not files[name]:
            raise ValueError("empty candidate artifact: " + name)
    verified = validate_gradient_step_candidate(root)
    _equal(verified, candidate, "verified candidate metadata")
    for name, saved in files.items():
        _equal(_small(root, name), saved, "unchanged validated candidate " + name)
    return candidate, files


def _snapshot(center_stage, center_result_sha256, center_acceptance_sha256, candidate_root, *,
              abacus_binary, expected_abacus_sha256, mpi_library, expected_mpi_sha256):
    center, candidate_root = _root(center_stage), _root(candidate_root)
    center_sha, acceptance_sha = _digest(center_result_sha256), _digest(center_acceptance_sha256)
    runtime = dict(abacus_binary=_runtime_file(abacus_binary, expected_abacus_sha256),
                   abacus_sha256=_digest(expected_abacus_sha256),
                   mpi_library=_runtime_file(mpi_library, expected_mpi_sha256),
                   mpi_sha256=_digest(expected_mpi_sha256))
    loaded = load_accepted_combined_step(center, center_sha, acceptance_sha)
    result_content = _small(center, "result/RESULT.json", center_sha)
    result = _json(result_content)
    _equal(result, loaded["result"], "accepted center result")
    acceptance = _small(center, "COMBINED_STEP_ACCEPTANCE.json", acceptance_sha)
    candidate, candidate_files = _candidate(candidate_root, center, center_sha, acceptance_sha, result)
    pbe = result["actual_pbe"]
    _near(pbe.get("baseline_energy_ev"), ORIGINAL_ENERGY_EV, "original C2 PBE energy")
    center_energy = _number(pbe.get("candidate_energy_ev"), "accepted center PBE energy")
    slot = center / "result/pbe_slots/candidate"
    prep_content = _small(slot, "COMBINED_STEP_PBE_PREPARED.json", _digest(pbe.get("preparation_sha256")))
    prep = _json(prep_content)
    hashes = prep.get("prepared_input_sha256")
    if not isinstance(hashes, dict) or len(hashes) != 5:
        raise ValueError("exact five frozen pure-PBE inputs required")
    inputs = {name: _small(slot, name, _digest(digest)) for name, digest in hashes.items()}
    if not {"INPUT", "STRU", "KPT"}.issubset(inputs) or any(not content for content in inputs.values()):
        raise ValueError("missing or empty pure-PBE input")
    pseudo, orbital = endpoint._c2_files(inputs["STRU"])
    _equal(set(inputs), {"INPUT", "STRU", "KPT", pseudo, orbital}, "frozen PBE file set")
    if set(inputs) & {PREPARATION, COLLECTION, ".provenance"}:
        raise ValueError("PBE input filename collision")
    values = endpoint._input_values(inputs["INPUT"])
    _equal(values, endpoint._pure_pbe(values), "unchanged pure PBE numerics")
    log_name = "OUT."+endpoint._basename(values.get("suffix", "ABACUS"))+"/running_scf.log"
    _equal(prep.get("candidate_log_relative_path"), log_name, "frozen SCF log path")
    log = _small(slot, log_name, _digest(pbe.get("candidate_log_sha256")))
    measured = _strict_log(log)
    _near(measured["energy_ev"], center_energy, "accepted center SCF energy")
    _equal(endpoint._sha256(inputs[orbital]), result["orbital_sha256"], "accepted center orbital")
    inputs[orbital] = candidate_files["C_3s3p2d_ec_step.orb"]
    evidence = {".provenance/"+name: content for name, content in candidate_files.items()}
    evidence.update({".provenance/center_RESULT.json": result_content,
                     ".provenance/center_ACCEPTANCE.json": acceptance,
                     ".provenance/center_PBE_PREPARED.json": prep_content,
                     ".provenance/center_running_scf.log": log})
    manifest = dict(_COMMON, format_version=1, status="prepared", center_stage=str(center),
        candidate_root=str(candidate_root), candidate_sha256=endpoint._sha256(candidate_files["CANDIDATE.json"]),
        center_preparation_sha256=pbe["preparation_sha256"],
        baseline_energy_ev=ORIGINAL_ENERGY_EV, center_energy_ev=center_energy,
        baseline_log_sha256=_digest(pbe.get("baseline_log_sha256")), center_log_sha256=measured["log_sha256"],
        candidate_log_relative_path=log_name, **runtime)
    manifest.update({key: candidate[key] for key in _IDENTITY})
    manifest.update(prepared_input_sha256={n: endpoint._sha256(c) for n, c in inputs.items()},
                    evidence_sha256={n: endpoint._sha256(c) for n, c in evidence.items()})
    return manifest, dict(inputs, **evidence)


def _prepared(output_root, expected_preparation_sha256):
    root = _root(output_root)
    manifest = _json(_small(root, PREPARATION, _digest(expected_preparation_sha256)))
    _expect(manifest, dict(_COMMON, format_version=1, status="prepared"), "Ec PBE preparation")
    expected, files = _snapshot(manifest["center_stage"], manifest["center_result_sha256"],
        manifest["center_acceptance_sha256"], manifest["candidate_root"],
        abacus_binary=manifest["abacus_binary"], expected_abacus_sha256=manifest["abacus_sha256"],
        mpi_library=manifest["mpi_library"], expected_mpi_sha256=manifest["mpi_sha256"])
    _equal(manifest, expected, "complete immutable Ec PBE preparation")
    for name, content in files.items():
        _equal(_small(root, name), content, "frozen Ec PBE input/evidence " + name)
    return manifest


def prepare_ec_gradient_step_pbe(center_stage, center_result_sha256, center_acceptance_sha256,
                                 candidate_root, output_root, *, abacus_binary,
                                 expected_abacus_sha256, mpi_library, expected_mpi_sha256):
    """Reserve a new directory; partial preparations are retained and never reused."""
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = _root(output)
    manifest, files = _snapshot(center_stage, center_result_sha256, center_acceptance_sha256, candidate_root,
        abacus_binary=abacus_binary, expected_abacus_sha256=expected_abacus_sha256,
        mpi_library=mpi_library, expected_mpi_sha256=expected_mpi_sha256)
    for name in ("center_stage", "candidate_root", "gradient_stage"):
        original = _root(manifest[name])
        if output == original or original in output.parents or output in original.parents:
            raise ValueError("PBE output must be separate from immutable " + name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    endpoint._write_json(output / PREPARATION, manifest)
    _prepared(output, endpoint._sha256((output / PREPARATION).read_bytes()))
    return manifest


def collect_ec_gradient_step_pbe(output_root, expected_preparation_sha256):
    """Read one strict 4-occupied/44-band SCF log; scheduler acceptance stays external."""
    root = _root(output_root)
    destination = root / COLLECTION
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest = _prepared(root, expected_preparation_sha256)
    name = manifest["candidate_log_relative_path"]
    content = _small(root, name)
    measured = _strict_log(content)
    delta = (measured["energy_ev"]-manifest["baseline_energy_ev"])/2
    center_delta = (measured["energy_ev"]-manifest["center_energy_ev"])/2
    _number(delta, "PBE original energy difference")
    _number(center_delta, "PBE center energy difference")
    result = dict(_COMMON, status="collected", preparation_sha256=expected_preparation_sha256,
        candidate_sha256=manifest["candidate_sha256"], center_stage=manifest["center_stage"],
        center_preparation_sha256=manifest["center_preparation_sha256"],
        baseline_energy_ev=manifest["baseline_energy_ev"], center_energy_ev=manifest["center_energy_ev"],
        candidate_energy_ev=measured["energy_ev"], energy_delta_ev_per_c=delta,
        delta_from_center_ev_per_c=center_delta,
        comparison="absolute_candidate_minus_original_per_C_lte_tolerance",
        pbe_gate="pass" if abs(delta) <= endpoint.TOLERANCE_EV_PER_C+1e-12 else "fail",
        scf_log_gate="pass", band_count_check="pass", band_counts=measured["band_counts"],
        candidate_log_path=str(root / name), candidate_log_sha256=measured["log_sha256"],
        baseline_log_sha256=manifest["baseline_log_sha256"], center_log_sha256=manifest["center_log_sha256"])
    result.update({key: manifest[key] for key in _IDENTITY + ("abacus_binary", "abacus_sha256", "mpi_library", "mpi_sha256")})
    _small(root, name, measured["log_sha256"])
    endpoint._write_json(destination, result)
    return result
