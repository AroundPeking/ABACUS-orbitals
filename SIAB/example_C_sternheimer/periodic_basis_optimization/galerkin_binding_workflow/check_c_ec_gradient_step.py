"""Admit frozen Ec gradients and verify one distinct, diagnostic-only proposal.

No cache, response, backward, SCF or scheduler calls occur. Numerical imports
are lazy and used only for small coefficient/report reconstruction. Historical
source archives are checked as data, never imported as the running validator.
"""

import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import check_c_accepted_combined_step as accepted
from check_c_accepted_combined_step import load_accepted_combined_step
import check_c_optimized_pbe as endpoint
from check_c_combined_step_pbe import _near, _number, _regular, _root
from check_c_direction_probe_pbe import _digest, _equal, _json
import refresh_c_combined_step_gradient as refresh


SCOPE = "accepted_ec_gradient_step_candidate"
RADIUS = .02
NU = [3, 3, 2, 0, 0]
GRADIENT_FILE = "result/EC_GRADIENT_REFRESH.json"
ACCEPTANCE_FILE = "GRADIENT_ACCEPTANCE.json"
_TIMING = ("cache_load_seconds", "forward_seconds", "forward_and_backward_seconds", "total_seconds", "peak_rss_kib")
_PROVENANCE = (
    "status", "scope", "source_commit", "deployment_sha256", "accepted_result_sha256",
    "accepted_acceptance_sha256", "accepted_source_commit", "coefficient_sha256",
    "freeze_sha256", "active_cache_index_sha256", "occupied_capture_floor", "center_reproduction",
    "cache_loads", "rpa_evaluations", "gradient_mode", "backward_passes", "actual_scf_count",
    "optimizer_steps", "candidate_count", "coefficient_update", "exported_candidate",
    "physical_release_gate", "ordinary_sos_qavg", "gw")


def _commit(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("exact lowercase source commit required")
    return value


def _absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("absolute evidence path required")
    return _root(value)


def _same_json(actual, expected, name):
    # Unlike dict equality, canonical JSON also distinguishes booleans/numbers.
    _equal(json.dumps(actual, sort_keys=True, allow_nan=False),
           json.dumps(expected, sort_keys=True, allow_nan=False), name)


def _source_bytes(root, name, digest):
    # Source JSON may contain arrays; hash its raw bytes without schema parsing.
    _regular(root, name)
    with endpoint._within(root, name).open("rb") as stream:
        data = stream.read(accepted._MAX_FILE_BYTES + 1)
    if len(data) > accepted._MAX_FILE_BYTES:
        raise ValueError("oversized archived source file")
    _equal(endpoint._sha256(data), _digest(digest), "source SHA256 " + name)
    return data


def _archived_source(stage, digest, commit):
    manifest = _json(accepted._small(stage, "DEPLOYMENT.json", digest))
    _equal(_commit(manifest.get("source_commit")), commit, "gradient deployment commit")
    source = _absolute(manifest.get("source_directory"))
    _equal(source, stage / "source", "archived source location")
    archive = _absolute(manifest.get("archive_path"))
    _equal(archive.parent, stage, "archived source tar location")
    files = manifest.get("files")
    if (not isinstance(files, dict) or not 0 < len(files) <= 2048
            or not set(refresh.REQUIRED_SOURCE_FILES).issubset(files)):
        raise ValueError("complete bounded archived source manifest required")
    actual = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink archived source forbidden")
        if path.is_file():
            actual.add(path.relative_to(source).as_posix())
    _equal(actual, set(files), "exact deployed source file set")
    payload = _source_bytes(stage, archive.name, manifest.get("archive_sha256"))
    seen, members, total = set(), set(), 0
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            for member in tar:
                name = member.name
                path = PurePosixPath(name)
                if (not name or path.is_absolute() or ".." in path.parts
                        or str(path) != name.rstrip("/") or name in seen or len(seen) >= 4096):
                    raise ValueError("unsafe or duplicate archived member")
                seen.add(name)
                if member.isdir():
                    continue
                if not member.isfile() or name not in files:
                    raise ValueError("unlisted or nonregular archived member: " + name)
                if not 0 <= member.size <= accepted._MAX_FILE_BYTES:
                    raise ValueError("oversized archived member")
                total += member.size
                if total > accepted._MAX_EVIDENCE_BYTES:
                    raise ValueError("archived source exceeds validation budget")
                stream = tar.extractfile(member)
                with stream:
                    data = stream.read(accepted._MAX_FILE_BYTES + 1)
                if len(data) != member.size:
                    raise ValueError("incomplete archived member")
                _equal(endpoint._sha256(data), _digest(files[name]), "archived member SHA256 " + name)
                _equal(_source_bytes(source, name, files[name]), data, "deployed/archive bytes " + name)
                members.add(name)
    except (tarfile.TarError, EOFError) as error:
        raise ValueError("invalid source archive") from error
    _equal(members, set(files), "exact archived source file set")
    return manifest


def _runtime():
    source = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(source / "SIAB/opt_orb_pytorch_dpsi"),
                   str(source / "SIAB/example_C_sternheimer/periodic_basis_optimization")]
    import torch
    from periodic_galerkin_basis import read_periodic_optimizer_coefficients
    from periodic_galerkin_radial_diagnostics import radial_gradient_report, transported_descent_report
    from periodic_galerkin_combined_step import combine_pbe_tangent
    from periodic_galerkin_ec_gradient_step import propose_ec_gradient_step, _same_report
    from export_periodic_orbitals import write_abacus_orbital
    return SimpleNamespace(torch=torch, read_coefficients=read_periodic_optimizer_coefficients,
        radial_gradient_report=radial_gradient_report, transported_descent_report=transported_descent_report,
        combine_pbe_tangent=combine_pbe_tangent, propose_ec_gradient_step=propose_ec_gradient_step,
        same_report=_same_report, write_abacus_orbital=write_abacus_orbital)


def _profile(record):
    accepted._expect(record, dict(nu=NU, fixed_nu=[0]*5, ao_per_C=22), "C radial profile")
    if any(type(n) is not int for key in ("nu", "fixed_nu") for n in record[key]):
        raise ValueError("integer C radial profile required")


def _scheduler(acceptance):
    job, text = acceptance.get("job_id"), acceptance.get("scheduler")
    if not isinstance(job, str) or re.fullmatch(r"[0-9]+", job) is None or not isinstance(text, str):
        raise ValueError("gradient scheduler parent and archived rows required")
    rows = [line.split("|") for line in text.splitlines() if line.strip()]
    if len(rows) != 3 or any(len(row) < 3 or row[1:3] != ["COMPLETED", "0:0"] for row in rows):
        raise ValueError("three COMPLETED/0:0 gradient scheduler rows required")
    _equal({row[0] for row in rows}, {job, job+".batch", job+".extern"}, "gradient scheduler identities")
    return job


def load_accepted_ec_gradient(stage, result_sha256, acceptance_sha256):
    """Return gradient, accepted combined center, stage and existing frozen paths.

    This revalidates small archived evidence and tensors, never the response or
    cache payloads. Source paths describe the archived stage, not this checkout.
    """
    _digest(result_sha256)
    _digest(acceptance_sha256)
    stage = _root(stage)
    gradient = _json(accepted._small(stage, GRADIENT_FILE, result_sha256))
    acceptance = _json(accepted._small(stage, ACCEPTANCE_FILE, acceptance_sha256))
    _equal(accepted._small(stage, "STATUS").decode().strip(), "success", "gradient STATUS")
    accepted._expect(gradient, dict(status="success", scope="accepted_combined_step_ec_gradient_refresh",
        gradient_mode="energy_only", backward_passes=1, cache_loads=1, rpa_evaluations=1,
        actual_scf_count=0, optimizer_steps=0, candidate_count=0, coefficient_update="none", exported_candidate="none",
        center_reproduction="pass", old_direction_reconstruction="exact_accepted_coefficients",
        actual_pbe_direction_derivatives="unmeasured", physical_release_gate="hold",
        ordinary_sos_qavg="pending", gw="pending"), "accepted Ec gradient")
    _profile(gradient)
    commit = _commit(gradient.get("source_commit"))
    provenance = _json(accepted._small(stage, "PROVENANCE.json"))
    expected = {key:gradient[key] for key in _PROVENANCE}
    expected.update(diagnostic_filename=GRADIENT_FILE, diagnostic_sha256=result_sha256)
    _same_json(provenance, expected, "gradient provenance")
    accepted._expect(acceptance, dict(status="success", scope="gradient_acceptance_without_replay_not_optimized_basis",
        result_sha256=result_sha256, deployment_sha256=gradient["deployment_sha256"],
        new_candidate_count=0, new_scf_count=0, physical_release_gate="hold"), "gradient acceptance")
    _digest(acceptance.get("verification_script_sha256"))
    job = _scheduler(acceptance)
    _equal(accepted._small(stage, "slurm-"+job+".err"), b"", "gradient stderr")
    deployment = _archived_source(stage, _digest(gradient["deployment_sha256"]), commit)
    center_stage = _absolute(gradient.get("accepted_stage"))
    center = load_accepted_combined_step(center_stage, _digest(gradient.get("accepted_result_sha256")),
                                         _digest(gradient.get("accepted_acceptance_sha256")))
    for key in ("coefficient_sha256", "orbital_sha256", "freeze_sha256", "active_cache_index_sha256",
                "directions_sha256", "calibration_sha256"):
        _equal(_digest(gradient.get(key)), center["result"][key], "gradient center " + key)
    _equal(gradient.get("original_quarter_result_sha256"), center["result"]["center_result_sha256"], "original quarter")
    _equal(_commit(gradient.get("accepted_source_commit")), center["result"]["source_commit"], "accepted center source")
    if commit == gradient["accepted_source_commit"]:
        raise ValueError("gradient source cannot claim the old combined source")
    _equal(_number(gradient.get("occupied_capture_floor"), "original floor"), center["occupied_capture_floor"], "original floor")
    refresh.validate_reproduction(center, dict(gradient, scope="radial_gradients_at_fixed_candidate_no_optimization"),
                                  gradient.get("initial_frozen_band_screen"))
    runtime = _runtime()
    runtime.torch.set_num_threads(28)
    coefficients = refresh._read_coefficients(runtime, center["coefficient_path"])
    report = gradient["energy_gradient"]
    rows = report.get("channels")
    if (not isinstance(rows, list) or [(r.get("element"), r.get("l"), r.get("shape")) for r in rows]
            != [("C", l, [31, n]) for l, n in enumerate(NU)]):
        raise ValueError("complete C signed-QR gradient channels required")
    with runtime.torch.no_grad():
        raw = {"C": [runtime.torch.tensor(row["raw_gradient"], dtype=runtime.torch.float64) for row in rows]}
        if not runtime.same_report(runtime.radial_gradient_report(coefficients, raw), report):
            raise ValueError("inconsistent accepted Ec gradient report")
        previous = refresh._old_direction(runtime, center, coefficients)
        transport = runtime.transported_descent_report(coefficients, report, previous)
    _same_json(gradient.get("transported_descent"), transport, "reconstructed transported direction")
    _same_json(acceptance.get("transported_direction"), transport, "accepted transported direction")
    norm = _number(report.get("horizontal_gradient_norm"), "horizontal Ec norm")
    if norm <= 1e-14:
        raise ValueError("accepted Ec gradient must have a resolved nonzero norm")
    _near(acceptance.get("gradient_norm"), norm, "accepted gradient norm")
    _near(acceptance.get("loss"), gradient["loss"], "accepted loss")
    _near(acceptance.get("center_energy_ha"), gradient["rpa"]["candidate_energy_ha"], "accepted center energy")
    shares = {str(r["l"])+":"+str(r["zeta"]):r["gradient_norm"]**2/norm**2 for r in transport["radials"]}
    _same_json(acceptance.get("radial_gradient_squared_shares"), shares, "accepted radial shares")
    timing = {key:_number(gradient.get(key), key) for key in _TIMING}
    if (any(value <= 0 for value in timing.values()) or not
            timing["forward_seconds"] < timing["forward_and_backward_seconds"] < timing["total_seconds"] < 3600):
        raise ValueError("invalid gradient timing")
    _same_json(acceptance.get("timing"), timing, "accepted timing")
    return dict(gradient=gradient, center=center, stage=stage, acceptance=acceptance, deployment=deployment,
        result_sha256=result_sha256, acceptance_sha256=acceptance_sha256, center_stage=center_stage,
        occupied_capture_floor=center["occupied_capture_floor"], coefficient_path=center["coefficient_path"],
        orbital_path=center["orbital_path"], original_coefficient_path=center["original_coefficient_path"],
        freeze_path=center["freeze_path"], active_cache_index_path=center["active_cache_index_path"])


def validate_gradient_step_candidate(candidate_root):
    """Return the manifest only after exact kernel, coefficient and orbital checks.

    CANDIDATE.json uses center_{stage,result_sha256,acceptance_sha256,
    coefficient_sha256,orbital_sha256} and gradient_{stage,result_sha256,
    acceptance_sha256}. GRADIENT_STEP.json is the serialized kernel proposal
    without coefficients. The only writes are a temporary orbital rendering.
    """
    root = _root(candidate_root)
    candidate = _json(accepted._small(root, "CANDIDATE.json"))
    accepted._expect(candidate, dict(status="prepared", scope=SCOPE,
        direction_name="negative_horizontal_ec_gradient", coefficient_filename="COEFFICIENTS.txt",
        orbital_filename="C_3s3p2d_ec_step.orb", actual_pbe_direction_derivative="unmeasured",
        finite_step_safety="unmeasured", actual_pbe_gate="pending", galerkin_energy="unmeasured",
        physical_release_gate="hold"), "Ec step candidate")
    _equal(_number(candidate.get("radius"), "radius"), RADIUS, "fixed Ec step radius")
    _profile(candidate)
    _commit(candidate.get("source_commit"))
    artifacts = {}
    for name, key in (("COEFFICIENTS.txt", "coefficient_sha256"), ("C_3s3p2d_ec_step.orb", "orbital_sha256"),
                      ("GRADIENT_STEP.json", "gradient_step_sha256")):
        artifacts[name] = accepted._small(root, name, _digest(candidate.get(key)))
        if not artifacts[name]:
            raise ValueError("empty Ec step artifact")
    step = _json(artifacts["GRADIENT_STEP.json"])
    loaded = load_accepted_ec_gradient(_absolute(candidate.get("gradient_stage")),
        _digest(candidate.get("gradient_result_sha256")), _digest(candidate.get("gradient_acceptance_sha256")))
    gradient, center = loaded["gradient"], loaded["center"]
    _equal(_absolute(candidate.get("center_stage")), loaded["center_stage"], "Ec step center stage")
    for name, expected in (("center_result_sha256", gradient["accepted_result_sha256"]),
                           ("center_acceptance_sha256", gradient["accepted_acceptance_sha256"]),
                           ("center_coefficient_sha256", gradient["coefficient_sha256"]),
                           ("center_orbital_sha256", gradient["orbital_sha256"])):
        _equal(_digest(candidate.get(name)), expected, "Ec step " + name)
    cheap = candidate.get("cheap_gate")
    accepted._expect(cheap, dict(gate=True), "Ec step cheap gate")
    for key, value in (("occupied_capture_floor", loaded["occupied_capture_floor"]),
                       ("overlap_relative_rank_tolerance", 1e-12), ("overlap_condition_limit", 1e12)):
        _equal(_number(cheap.get(key), key), value, "original cheap guard " + key)
    accepted._capture(cheap.get("minimum_occupied_capture"), loaded["occupied_capture_floor"])
    accepted._band_guard(cheap.get("band_screen"))
    runtime = _runtime()
    runtime.torch.set_num_threads(28)
    coefficients = refresh._read_coefficients(runtime, center["coefficient_path"])
    saved = refresh._read_coefficients(runtime, root / "COEFFICIENTS.txt")
    with runtime.torch.no_grad():
        proposal = runtime.propose_ec_gradient_step(coefficients, gradient["energy_gradient"], radius=RADIUS)
        serialized = {key:value for key,value in proposal.items() if key not in ("coefficients", "direction")}
        serialized["direction"] = {e:[d.tolist() for d in channels] for e,channels in proposal["direction"].items()}
        _same_json(step, serialized, "exact serialized Ec step proposal")
        for key,value in serialized.items():
            if key != "direction":
                _same_json(candidate.get(key), value, "candidate proposal " + key)
        expected = proposal["coefficients"]
        if (not isinstance(saved, dict) or set(saved) != set(expected)
                or any(len(saved[e]) != len(channels) or any(not runtime.torch.equal(a, b)
                       for a,b in zip(saved[e], channels)) for e,channels in expected.items())):
            raise ValueError("Ec step coefficients do not reconstruct exactly")
        with tempfile.TemporaryDirectory(prefix="c-ec-orbital-check-") as directory:
            orbital = Path(directory) / "C_3s3p2d_ec_step.orb"
            runtime.write_abacus_orbital(orbital, saved, element="C", ecut_ry=100.,
                rcut_bohr=10., dr_bohr=.01, smoothing_sigma_bohr=.1)
            _equal(orbital.read_bytes(), artifacts["C_3s3p2d_ec_step.orb"], "exact Ec step orbital rendering")
    return candidate
