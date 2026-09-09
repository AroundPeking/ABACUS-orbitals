"""Admit one completed combined step as a diagnostic center, never physics release.

The caller pins RESULT and acceptance independently. Only small archived evidence
is read; cache index paths are returned, never followed. Existing quarter-only
validators remain authoritative for the original center. No prepared record is
updated and no SCF, response, coefficient reconstruction or scheduler call occurs.
"""

import math
from pathlib import Path
import re

import check_c_optimized_pbe as endpoint
import check_c_combined_step_pbe as combined
from check_c_direction_probe_pbe import _digest, _equal, _json, _strict_log
from check_c_combined_step_pbe import _near, _number


_SELECTED = [1, 22, 43, 6, 27, 23, 11, 55]
_WEIGHTS = [v/64 for v in (1, 8, 4, 6, 24, 12, 3, 6)]
_MAX_FILE_BYTES = 8 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_ACCEPTANCE = "COMBINED_STEP_ACCEPTANCE.json"


def _small(root, name, digest=None):
    combined._regular(root, name)
    path = endpoint._within(root, name)
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError("oversized diagnostic evidence: " + str(path))
    with path.open("rb") as stream:
        content = stream.read(_MAX_FILE_BYTES + 1)
    if len(content) > _MAX_FILE_BYTES:
        raise ValueError("oversized diagnostic evidence: " + str(path))
    if digest is not None:
        _equal(endpoint._sha256(content), _digest(digest), "SHA256 " + str(path))
    if path.suffix == ".json":
        _json(content)
    return content


def _expect(record, expected, name):
    if not isinstance(record, dict):
        raise ValueError(name + " object required")
    for key, value in expected.items():
        actual = record.get(key)
        if type(value) in (bool, int) and type(actual) is not type(value):
            raise ValueError(name + " " + key + " has wrong type")
        _equal(actual, value, name + " " + key)


def _scheduler(acceptance):
    job = acceptance.get("job_id")
    rows = acceptance.get("scheduler")
    if not isinstance(job, str) or re.fullmatch(r"[0-9]+", job) is None:
        raise ValueError("accepted scheduler parent job ID required")
    if (not isinstance(rows, list) or len(rows) != 4 or
            any(not isinstance(row, list) or len(row) < 3 or
                not isinstance(row[0], str) or row[1:3] != ["COMPLETED", "0:0"] for row in rows)):
        raise ValueError("four COMPLETED/0:0 scheduler rows required")
    _equal({row[0] for row in rows}, {job, job+".batch", job+".extern", job+".0"},
           "scheduler parent/batch/extern/.0 identities")


def _bounded_preparation(root, digest):
    """Bound the complete declared graph before the legacy helpers read it."""
    manifest = _json(_small(root, combined.PREPARATION, _digest(digest)))
    files, total = {}, 0
    for key in ("evidence_sha256", "prepared_input_sha256"):
        values = manifest.get(key)
        if not isinstance(values, dict) or not 0 < len(values) <= 64:
            raise ValueError("bounded nonempty preparation " + key + " required")
        for name, expected in values.items():
            content = _small(root, name, _digest(expected))
            total += len(content)
            if total > _MAX_EVIDENCE_BYTES:
                raise ValueError("combined evidence exceeds bounded validation budget")
            files[name] = content
    prefix = combined._CENTER
    center_name = prefix + endpoint.PREPARATION
    if center_name not in files:
        raise ValueError("missing original quarter preparation")
    center = _json(files[center_name])
    # The nested validator must not follow undeclared or unbounded evidence.
    for key in ("prepared_input_sha256", "evidence_sha256"):
        values = center.get(key)
        if not isinstance(values, dict) or not 0 < len(values) <= 64:
            raise ValueError("bounded quarter evidence map required")
        for name, expected in values.items():
            endpoint._within(root / prefix, name)
            content = files.get(prefix + name)
            if content is None or endpoint._sha256(content) != _digest(expected):
                raise ValueError("undeclared quarter evidence: " + name)
    for name in (endpoint.COLLECTION, center.get("candidate_log_relative_path")):
        if not isinstance(name, str) or prefix + name not in files:
            raise ValueError("undeclared quarter collection/log")
    return combined._prepared(root, digest)


def _band_guard(guard, *, enforce_accuracy=True):
    if type(enforce_accuracy) is not bool:
        raise ValueError('explicit boolean band accuracy switch required')
    _expect(guard, dict(scf_pbe_gate="pending",
        scope="frozen_h_band_screen_not_scf_energy",
        k_weight_convention="ABACUS_spin_included_no_renormalization"), "band guard")
    for key, value in (("k_weight_sum", 2.), ("occupied_band_sum_limit_ev_per_atom", .01),
                       ("target_band_change_limit_ev", .05)):
        _near(guard.get(key), value, "band guard " + key)
    band_sum=_number(guard.get("occupied_band_sum_change_ev_per_atom"), "occupied band change")
    shift=_number(guard.get("maximum_target_band_change_ev"), "target band change")
    gap=_number(guard.get("minimum_gap_ev"), "minimum gap")
    if (gap <= 0 or shift < 0 or (enforce_accuracy and
            (guard.get('gate') is not True or abs(band_sum) > .01 or shift > .05))):
        raise ValueError("numerical frozen band protection failed")


def _capture(value, floor):
    if not floor <= _number(value, "occupied capture") <= 1.+1e-12:
        raise ValueError("original occupied capture floor failed")


def _record(record, floor, weights, *, enforce_band_accuracy=True):
    _expect(record, dict(energy_quantity="frozen_body_RPA_correlation_not_PBE_total"), "RPA record")
    if _number(record.get("loss"), "loss") < 0:
        raise ValueError("negative loss")
    _capture(record.get("minimum_occupied_capture"), floor)
    if not 0 < _number(record.get("maximum_overlap_condition"), "overlap condition") <= 1e12:
        raise ValueError("overlap condition gate failed")
    _band_guard(record.get("coefficient_guard"),enforce_accuracy=enforce_band_accuracy)
    rpa = record.get("rpa")
    _expect(rpa, dict(complete_q_weight=True), "RPA integration")
    names = ("pi", "trace_log", "energy")
    if not isinstance(weights, dict) or set(weights) != {name+"_weight" for name in names}:
        raise ValueError("original RPA training weights required")
    terms = []
    for name in names:
        weight = _number(weights[name+"_weight"], "RPA weight")
        error = _number(rpa.get(name+"_relative_squared_error"), "RPA squared error")
        if weight < 0 or error < 0:
            raise ValueError("nonnegative RPA weights and squared errors required")
        terms.append(weight*error)
    if not any(weights[name+"_weight"] > 0 for name in names):
        raise ValueError("at least one positive RPA weight required")
    _near(record["loss"], math.fsum(terms), "RPA weighted loss")
    energy = _number(rpa.get("candidate_energy_ha"), "candidate energy")
    reference = _number(rpa.get("reference_energy_ha"), "reference energy")
    if reference == 0:
        raise ValueError("nonzero reference energy required")
    _near(rpa["energy_relative_squared_error"], ((energy-reference)/reference)**2,
          "RPA relative energy error")
    _near(rpa.get("q_weight_coverage"), 1., "q weight coverage")
    rows = rpa.get("per_q")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("full eight q representatives required")
    for row, iq, weight in zip(rows, _SELECTED, _WEIGHTS):
        _expect(row, dict(selected_iq=iq), "q representative")
        _near(row.get("q_weight"), weight, "q star weight")
        for key in ("frequency_ha", "candidate_contributions_ha", "reference_contributions_ha"):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != 12:
                raise ValueError("twelve frequencies/contributions required: " + key)
            for value in values:
                _number(value, key)
        frequencies = row["frequency_ha"]
        if frequencies[0] <= 0 or any(a >= b for a, b in zip(frequencies, frequencies[1:])):
            raise ValueError("positive increasing frequency nodes required")
    for contributions, energy, prefix in (
            ("candidate_contributions_ha", "candidate_energy_ha", "rpa"),
            ("reference_contributions_ha", "reference_energy_ha", "reference_rpa")):
        # These contributions already contain the q weights.
        _near(math.fsum(v for row in rows for v in row[contributions]), rpa.get(energy),
              "aggregate " + energy)
        for suffix, divisor in (("cell", 1), ("c", 2)):
            _near(record.get(prefix + "_correlation_energy_ev_per_" + suffix),
                  rpa[energy]*endpoint.HARTREE_TO_EV/divisor, "RPA reported energy units")


def _same_grid_reference(record, original, reproduce=False):
    for row, previous in zip(record["rpa"]["per_q"], original["rpa"]["per_q"]):
        _equal(row["frequency_ha"], previous["frequency_ha"], "unchanged frequency grid")
        keys = ["reference_contributions_ha"]
        if reproduce:
            keys.append("candidate_contributions_ha")
        for key in keys:
            for value, expected in zip(row[key], previous[key]):
                _near(value, expected, "quarter reproduction " + key)


def _actual_pbe(root, pbe, candidate_sha):
    if not isinstance(pbe, dict):
        raise ValueError("actual PBE collection required")
    _equal(_json(_small(root, combined.COLLECTION)), pbe, "persisted actual PBE collection")
    manifest, baseline, center = _bounded_preparation(root, pbe.get("preparation_sha256"))
    _expect(pbe, dict(combined._COMMON, status="collected", pbe_gate="pass", scf_log_gate="pass",
        band_count_check="pass", band_counts=dict(occupied=4, total=44),
        comparison="absolute_candidate_minus_original_per_C_lte_tolerance",
        candidate_sha256=candidate_sha, baseline_log_sha256=baseline["log_sha256"],
        center_log_sha256=center["log_sha256"]), "actual PBE")
    for key in combined._IDENTITY + ("candidate_sha256", "center_preparation_sha256", "center_collection_sha256"):
        _equal(pbe.get(key), manifest[key], "actual PBE preparation " + key)
    log_name = manifest["candidate_log_relative_path"]
    measured = _strict_log(_small(root, log_name, _digest(pbe.get("candidate_log_sha256"))))
    # Historical absolute paths are descriptive. Read only the local frozen slot.
    declared = pbe.get("candidate_log_path")
    if not isinstance(declared, str) or not declared.endswith("/result/pbe_slots/candidate/" + log_name):
        raise ValueError("actual PBE log identity mismatch")
    delta = (measured["energy_ev"]-baseline["energy_ev"])/2
    for key, value in (("candidate_energy_ev", measured["energy_ev"]),
                       ("baseline_energy_ev", baseline["energy_ev"]), ("center_energy_ev", center["energy_ev"]),
                       ("energy_delta_ev_per_c", delta),
                       ("delta_from_center_ev_per_c", (measured["energy_ev"]-center["energy_ev"])/2)):
        _near(pbe.get(key), value, "actual PBE " + key)
    if abs(delta) > endpoint.TOLERANCE_EV_PER_C + 1e-12:
        raise ValueError("actual PBE exceeds original 10 meV/C bound")
    return manifest


def load_accepted_combined_step(stage, result_sha256, acceptance_sha256):
    """Return result/candidate/quarter dicts and existing coefficient/freeze paths.

    Returned paths point into the supplied stage, including its portable quarter
    archive. The capture floor always comes from the ORIGINAL quarter initial,
    not either later candidate. Tensor-backed legacy imports may be needed by
    _prepared's scalar calibration analyzer, but no tensor work is invoked here.
    """
    _digest(result_sha256)
    _digest(acceptance_sha256)
    stage = combined._root(stage)
    result = _json(_small(stage, "result/RESULT.json", result_sha256))
    acceptance = _json(_small(stage, _ACCEPTANCE, acceptance_sha256))
    provenance = _json(_small(stage, "PROVENANCE.json"))
    _equal(_small(stage, "STATUS").decode().strip(), "success", "stage STATUS")
    common = dict(status="success", candidate_gate="improved_frozen_body", actual_scf_count=1,
                  rpa_evaluations=2, physical_release_gate="hold")
    _expect(result, dict(common, scope="one_actual_pbe_tangent_step", optimizer_steps=0,
                        ordinary_sos_qavg="pending", gw="pending"), "combined result")
    source = result.get("source_commit")
    if not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("exact source commit required")
    _expect(provenance, dict(common, result_sha256=result_sha256, source_commit=source), "provenance")
    _expect(acceptance, dict(common, scope="read_only_combined_step_acceptance",
        result_sha256=result_sha256, source_commit=source, source_runtime_input_gate="pass",
        scheduler_gate="pass", pbe_gate="pass", center_reproduction="pass",
        new_calculations_in_this_audit=0, ordinary_sos_qavg="pending", gw="pending"), "acceptance")
    _scheduler(acceptance)
    candidate_root = stage / "result/candidate"
    candidate_sha = _digest(result.get("candidate_sha256"))
    candidate_bytes = _small(candidate_root, "CANDIDATE.json", candidate_sha)
    candidate = _json(candidate_bytes)
    _expect(candidate, dict(source_commit=source, actual_pbe_gate="pending", galerkin_energy="unmeasured",
                           physical_release_gate="hold", status="prepared"), "immutable candidate")
    for key in ("center_result_sha256", "coefficient_sha256", "orbital_sha256",
                "directions_sha256", "calibration_sha256"):
        _equal(_digest(result.get(key)), _digest(candidate.get(key)), "candidate/result " + key)
    for key in ("radius", "mixing_ratio"):
        _near(result.get(key), candidate.get(key), "candidate/result " + key)
    for name, key in (("COEFFICIENTS.txt", "coefficient_sha256"), ("C_3s3p2d_combined.orb", "orbital_sha256"),
                      ("DIRECTIONS.json", "directions_sha256"), ("CALIBRATION.json", "calibration_sha256")):
        _small(candidate_root, name, candidate[key])
    slot = stage / "result/pbe_slots/candidate"
    _actual_pbe(slot, result.get("actual_pbe"), candidate_sha)
    _equal(_small(slot, ".provenance/CANDIDATE.json", candidate_sha), candidate_bytes, "prepared candidate bytes")
    # _prepared -> _center_snapshot -> _backoff already ran _optimizer_artifacts
    # on the exact reconstructed archive; keep its quarter-only contract intact.
    original_root = slot / ".provenance/center/.provenance"
    quarter = _json(_small(original_root, "backoff_RESULT.json", result["center_result_sha256"]))
    _expect(quarter, dict(scope=endpoint.BACKOFF_SCOPE, alpha=.25), "original quarter")
    paths = dict(coefficient_path=candidate_root / "COEFFICIENTS.txt",
                 orbital_path=candidate_root / "C_3s3p2d_combined.orb")
    for key, name, identity in (("freeze_path", "backoff_INPUT_FREEZE.json", "freeze_sha256"),
            ("active_cache_index_path", "backoff_ACTIVE_DATA_CACHE.json", "active_cache_index_sha256"),
            ("original_coefficient_path", "backoff_ORIGINAL_COEFFICIENTS.txt", "initial_sha256")):
        _small(original_root, name, _digest(quarter.get(identity)))
        if identity != "initial_sha256":
            _equal(_digest(result.get(identity)), quarter[identity], "original " + identity)
        paths[key] = original_root / name
    initial_capture = _number(quarter["initial"].get("minimum_occupied_capture"), "original initial capture")
    floor = max(0., initial_capture-1e-4)
    cheap = candidate.get("cheap_gate")
    _expect(cheap, dict(gate=True), "cheap gate")
    _near(cheap.get("occupied_capture_floor"), floor, "original occupied capture floor")
    for key, value in (("overlap_relative_rank_tolerance", 1e-12), ("overlap_condition_limit", 1e12)):
        _equal(_number(cheap.get(key), key), value, "fixed cheap gate " + key)
    _capture(cheap.get("minimum_occupied_capture"), floor)
    _band_guard(cheap.get("band_screen"))
    center, evaluated = result.get("center"), result.get("candidate")
    for record in (quarter["initial"], quarter["candidate"], center, evaluated):
        _record(record, floor, quarter["training_weights"])
        _same_grid_reference(record, quarter["initial"])
    endpoint._match_parent_initial(center, quarter["candidate"])
    _same_grid_reference(center, quarter["candidate"], reproduce=True)
    endpoint._require_backoff_improvement(center, evaluated)
    _near(cheap["minimum_occupied_capture"], evaluated["minimum_occupied_capture"], "cheap/full occupied capture")
    _equal(cheap["band_screen"], evaluated["coefficient_guard"], "cheap/full band guard")
    energy, reference = (evaluated["rpa"][key] for key in ("candidate_energy_ha", "reference_energy_ha"))
    error = abs(energy-reference)*endpoint.HARTREE_TO_EV/2
    change = (energy-center["rpa"]["candidate_energy_ha"])*endpoint.HARTREE_TO_EV/2
    _near(result.get("body_error_ev_per_c"), error, "body error")
    _near(result.get("measured_ec_delta_ev_per_c"), change, "measured Ec change")
    return dict(result=result, candidate=candidate, quarter=quarter,
                occupied_capture_floor=floor, **paths)
