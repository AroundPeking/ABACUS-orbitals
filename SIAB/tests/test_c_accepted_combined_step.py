"""Read-only admission tests; no Torch, cache payloads, SCF or response work.

Set C_ACCEPTED_COMBINED_ARCHIVE to a copy of the small accepted stage evidence
on other hosts. Only explicitly named metadata, inputs and logs are copied.
"""

import ast
from contextlib import ExitStack
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SIAB = Path(__file__).resolve().parents[1]
WORKFLOW = SIAB / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
sys.path.insert(0, str(WORKFLOW))
ARCHIVE = Path(os.environ.get("C_ACCEPTED_COMBINED_ARCHIVE",
    "/Users/ghj/同步空间/AITP_project/gw_basis_optimization/provenance/actual-pbe-combined-step-20260907"))
RESULT_SHA = "2d073f68cd0fde81e6205b4c5e9c159c0be5513ccb47cfd7f83cdd9a45a051b7"
ACCEPTANCE_SHA = "d2802620a3880f1b490c3e33d4ff495ce61701e68077f08e8839fdd5c964fa7e"
SLOT = Path("result/pbe_slots/candidate")
PREP = "COMBINED_STEP_PBE_PREPARED.json"
COLL = "COMBINED_STEP_PBE_COLLECTION.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=True) + "\n")


def scalar_calibration_module():
    # Execute the real scalar analyzer unchanged, excluding Torch-only imports
    # and tensor functions. A new dependency fails instead of silently mocking it.
    path = SIAB / "opt_orb_pytorch_dpsi/periodic_galerkin_direction_calibration.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name in ("_finite", "analyze_pbe_axes")) or (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
        t.id == "SIGNED_RADII" for t in node.targets))]
    if len(nodes) != 3:
        raise AssertionError("scalar calibration source contract changed")
    module = types.ModuleType("periodic_galerkin_direction_calibration")
    module.math = math
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


class AcceptedCombinedStepTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec("check_c_accepted_combined_step")
        self.assertIsNotNone(spec, "strict accepted combined-step adapter missing")
        self.adapter = importlib.import_module("check_c_accepted_combined_step")
        if not (ARCHIVE / "COMBINED_STEP_ACCEPTANCE.json").is_file():
            self.skipTest("set C_ACCEPTED_COMBINED_ARCHIVE to the small accepted stage archive")
        self.assertEqual(digest(ARCHIVE / "result/RESULT.json"), RESULT_SHA)
        self.assertEqual(digest(ARCHIVE / "COMBINED_STEP_ACCEPTANCE.json"), ACCEPTANCE_SHA)
        temporary = tempfile.TemporaryDirectory(prefix="c-accepted-step-test-")
        self.addCleanup(temporary.cleanup)
        self.stage = Path(temporary.name)
        self.result_sha, self.acceptance_sha = RESULT_SHA, ACCEPTANCE_SHA
        manifest = read_json(ARCHIVE / SLOT / PREP)
        names = {Path(n) for n in ("STATUS", "PROVENANCE.json", "COMBINED_STEP_ACCEPTANCE.json",
                                  "result/RESULT.json")}
        names.update(SLOT / n for n in (PREP, COLL, manifest["candidate_log_relative_path"]))
        names.update(SLOT / n for key in ("evidence_sha256", "prepared_input_sha256")
                     for n in manifest[key])
        names.update(Path("result/candidate") / n for n in (
            "CANDIDATE.json", "COEFFICIENTS.txt", "C_3s3p2d_combined.orb", "DIRECTIONS.json", "CALIBRATION.json"))
        self.assertLess(len(names), 80)
        for name in names:
            source, target = ARCHIVE / name, self.stage / name
            self.assertLess(source.stat().st_size, 8 * 1024 * 1024)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(mock.patch.dict(sys.modules, {
            "periodic_galerkin_direction_calibration": scalar_calibration_module(),
            "torch": None, "numpy": None}))
        # Any accidental execution or tensor import must fail, not run locally.
        contexts.enter_context(mock.patch("subprocess.run", side_effect=AssertionError("process execution")))
        contexts.enter_context(mock.patch("subprocess.check_output", side_effect=AssertionError("process execution")))

    def load(self, **changes):
        options = dict(stage=self.stage, result_sha256=self.result_sha,
                       acceptance_sha256=self.acceptance_sha)
        return self.adapter.load_accepted_combined_step(**dict(options, **changes))

    def repin(self, result=None, acceptance=None):
        if result is not None:
            write_json(self.stage / "result/RESULT.json", result)
        self.result_sha = digest(self.stage / "result/RESULT.json")
        provenance = read_json(self.stage / "PROVENANCE.json")
        provenance["result_sha256"] = self.result_sha
        write_json(self.stage / "PROVENANCE.json", provenance)
        if acceptance is None:
            acceptance = read_json(self.stage / "COMBINED_STEP_ACCEPTANCE.json")
        acceptance["result_sha256"] = self.result_sha
        write_json(self.stage / "COMBINED_STEP_ACCEPTANCE.json", acceptance)
        self.acceptance_sha = digest(self.stage / "COMBINED_STEP_ACCEPTANCE.json")

    def refresh_pbe(self, result):
        root = self.stage / SLOT
        manifest = read_json(root / PREP)
        for key in ("evidence_sha256", "prepared_input_sha256"):
            manifest[key] = {n: digest(root / n) for n in manifest[key]}
        candidate = read_json(self.stage / "result/candidate/CANDIDATE.json")
        manifest["candidate_sha256"] = digest(self.stage / "result/candidate/CANDIDATE.json")
        for key in ("directions_sha256", "calibration_sha256"):
            manifest[key] = candidate[key]
        write_json(root / PREP, manifest)
        pbe = result["actual_pbe"]
        pbe["preparation_sha256"] = digest(root / PREP)
        for key in ("candidate_sha256", "directions_sha256", "calibration_sha256"):
            pbe[key] = result[key] = manifest[key]
        write_json(root / COLL, pbe)
        self.repin(result)

    def test_happy_path_is_portable_read_only_and_keeps_original_floor(self):
        before = {p.relative_to(self.stage): p.read_bytes()
                  for p in self.stage.rglob("*") if p.is_file()}
        loaded = self.load()
        self.assertEqual(loaded["result"], read_json(self.stage / "result/RESULT.json"))
        self.assertEqual(loaded["candidate"], read_json(self.stage / "result/candidate/CANDIDATE.json"))
        self.assertEqual(loaded["quarter"]["alpha"], .25)
        self.assertEqual(loaded["occupied_capture_floor"],
                         max(0., loaded["quarter"]["initial"]["minimum_occupied_capture"]-1e-4))
        self.assertNotEqual(loaded["occupied_capture_floor"],
                            loaded["result"]["candidate"]["minimum_occupied_capture"]-1e-4)
        for key, expected in (("coefficient_path", "coefficient_sha256"),
                              ("orbital_path", "orbital_sha256"),
                              ("freeze_path", "freeze_sha256"),
                              ("active_cache_index_path", "active_cache_index_sha256")):
            self.assertEqual(digest(loaded[key]), loaded["result"][expected])
        self.assertEqual(digest(loaded["original_coefficient_path"]), loaded["quarter"]["initial_sha256"])
        self.assertEqual(before, {p.relative_to(self.stage): p.read_bytes()
                                for p in self.stage.rglob("*") if p.is_file()})

    def test_external_pins_are_mandatory_and_never_self_derived(self):
        for key in ("result_sha256", "acceptance_sha256"):
            for value in (None, "", "bad", "0"*64, RESULT_SHA.upper()):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(**{key: value})
        with self.assertRaises(TypeError):
            self.adapter.load_accepted_combined_step(self.stage)

    def test_original_optimizer_validator_does_not_admit_new_result_scope(self):
        import check_c_optimized_pbe as endpoint
        with self.assertRaisesRegex(ValueError, "completed solid optimizer result required"):
            endpoint._optimizer_artifacts(self.stage / "result/RESULT.json", self.result_sha)

    def test_tampered_small_evidence_rejects(self):
        for path in sorted(p for p in self.stage.rglob("*") if p.is_file()):
            saved = path.read_bytes()
            path.write_bytes(saved + b"tampered")
            with self.subTest(path=path.relative_to(self.stage)), self.assertRaises(ValueError):
                self.load()
            path.write_bytes(saved)

    def test_scheduler_requires_exact_four_successful_steps(self):
        original = read_json(self.stage / "COMBINED_STEP_ACCEPTANCE.json")
        rows = original["scheduler"]
        changes = [dict(scheduler=[]), dict(scheduler=rows[:3]), dict(scheduler=rows+[rows[0]]),
                   dict(scheduler=rows[:3]+[rows[0]]), dict(scheduler=None), dict(job_id="other")]
        for i in range(4):
            for column, value in ((0, "other.0"), (1, "FAILED"), (1, "RUNNING"), (2, "1:0")):
                altered = copy.deepcopy(rows)
                altered[i][column] = value
                changes.append(dict(scheduler=altered))
        changes.append(dict(scheduler=[[]]+rows[1:]))
        for change in changes:
            self.repin(acceptance=dict(original, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_acceptance_and_provenance_cannot_overstate_or_reject_scope(self):
        for filename in ("COMBINED_STEP_ACCEPTANCE.json", "PROVENANCE.json"):
            path = self.stage / filename
            original = read_json(path)
            changes = [dict(status="failed"), dict(candidate_gate="rejected_actual_pbe"),
                       dict(actual_scf_count=True), dict(actual_scf_count=2), dict(rpa_evaluations=1),
                       dict(physical_release_gate="pass"), dict(source_commit="other")]
            if filename.startswith("COMBINED"):
                changes += [dict(scope="optimization"), dict(scheduler_gate="pending"),
                            dict(pbe_gate="fail"), dict(center_reproduction="fail"),
                            dict(ordinary_sos_qavg="pass"), dict(gw="pass")]
            for change in changes:
                write_json(path, dict(original, **change))
                if filename.startswith("COMBINED"):
                    self.acceptance_sha = digest(path)
                with self.subTest(filename=filename, change=change), self.assertRaises(ValueError):
                    self.load()
            write_json(path, original)
            self.acceptance_sha = digest(self.stage / "COMBINED_STEP_ACCEPTANCE.json")

    def test_result_status_counts_scope_and_frozen_identities(self):
        original = read_json(self.stage / "result/RESULT.json")
        changes = [dict(status="failed"), dict(scope="optimized_full_q_frozen_body_rpa_calibration"),
                   dict(candidate_gate="rejected_no_improvement"), dict(actual_scf_count=True),
                   dict(rpa_evaluations=2.), dict(optimizer_steps=1), dict(physical_release_gate="pass"),
                   dict(ordinary_sos_qavg="pass"), dict(gw="pass"), dict(radius=.001)]
        changes += [{key: "0"*64} for key in ("center_result_sha256", "freeze_sha256",
                    "active_cache_index_sha256", "coefficient_sha256", "orbital_sha256",
                    "directions_sha256", "calibration_sha256")]
        for change in changes:
            self.repin(dict(original, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_pbe_collection_must_match_prepared_identity_and_measured_numbers(self):
        original = read_json(self.stage / "result/RESULT.json")
        for change in (dict(pbe_gate="fail"), dict(scf_log_gate="fail"), dict(band_count_check="missing"),
                       dict(band_counts=dict(occupied=3, total=44)), dict(tolerance_ev_per_c=.02),
                       dict(baseline_energy_ev=-300), dict(candidate_energy_ev=-300),
                       dict(energy_delta_ev_per_c=0), dict(delta_from_center_ev_per_c=0),
                       dict(candidate_sha256="0"*64), dict(center_log_sha256="0"*64)):
            result = copy.deepcopy(original)
            result["actual_pbe"].update(change)
            write_json(self.stage / SLOT / COLL, result["actual_pbe"])
            self.repin(result)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_rehashed_scf_log_still_requires_convergence_counts_and_original_pbe_bound(self):
        original = read_json(self.stage / "result/RESULT.json")
        manifest = read_json(self.stage / SLOT / PREP)
        path = self.stage / SLOT / manifest["candidate_log_relative_path"]
        saved = path.read_text()
        for text in (saved.replace("#SCF IS CONVERGED#", "SCF NOT CONVERGED"),
                     saved + "\n !FINAL_ETOT_IS -300 eV\n",
                     "#SCF IS CONVERGED#\n!FINAL_ETOT_IS -309.85 eV\n",
                     "Occupied electronic states = 3\nNBANDS = 44\n#SCF IS CONVERGED#\n!FINAL_ETOT_IS -309.85 eV\n",
                     "Occupied electronic states = 4\nNBANDS = 44\n#SCF IS CONVERGED#\n!FINAL_ETOT_IS -309.8 eV\n"):
            path.write_text(text)
            result = copy.deepcopy(original)
            pbe = result["actual_pbe"]
            pbe["candidate_log_sha256"] = digest(path)
            if text.endswith("-309.8 eV\n"):
                pbe["candidate_energy_ev"] = -309.8
                pbe["energy_delta_ev_per_c"] = (-309.8-pbe["baseline_energy_ev"])/2
                pbe["delta_from_center_ev_per_c"] = (-309.8-pbe["center_energy_ev"])/2
            write_json(self.stage / SLOT / COLL, pbe)
            self.repin(result)
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                self.load()

    def test_rehashed_cheap_guards_and_prepared_metadata_remain_strict(self):
        source = self.stage / "result/candidate/CANDIDATE.json"
        original = read_json(source)
        result = read_json(self.stage / "result/RESULT.json")
        changes = [dict(gate=False), dict(occupied_capture_floor=.9), dict(minimum_occupied_capture=.9),
                   dict(overlap_relative_rank_tolerance=1e-8), dict(overlap_relative_rank_tolerance=0.),
                   dict(overlap_relative_rank_tolerance=2e-12), dict(overlap_condition_limit=1e15)]
        for change in changes:
            candidate = copy.deepcopy(original)
            candidate["cheap_gate"].update(change)
            write_json(source, candidate)
            (self.stage / SLOT / ".provenance/CANDIDATE.json").write_bytes(source.read_bytes())
            self.refresh_pbe(copy.deepcopy(result))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_capture_overlap_and_band_numbers_are_enforced_for_both_records(self):
        original = read_json(self.stage / "result/RESULT.json")
        changes = [("minimum_occupied_capture", .9), ("minimum_occupied_capture", True),
                   ("maximum_overlap_condition", 0), ("maximum_overlap_condition", 1.01e12)]
        guards = [("gate", 1), ("k_weight_sum", 1.), ("k_weight_convention", "normalized"),
                  ("occupied_band_sum_change_ev_per_atom", -.01001),
                  ("maximum_target_band_change_ev", .05001), ("maximum_target_band_change_ev", -.1),
                  ("minimum_gap_ev", 0), ("occupied_band_sum_limit_ev_per_atom", .1),
                  ("target_band_change_limit_ev", .1), ("scf_pbe_gate", "pass")]
        for name in ("center", "candidate"):
            for key, value in changes + guards:
                result = copy.deepcopy(original)
                target = result[name] if (key, value) in changes else result[name]["coefficient_guard"]
                target[key] = value
                self.repin(result)
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    self.load()

    def test_q_frequency_reference_and_aggregation_are_strict(self):
        original = read_json(self.stage / "result/RESULT.json")
        mutations = [lambda r: r.update(complete_q_weight=1), lambda r: r.update(q_weight_coverage=.9),
                     lambda r: r["per_q"].pop(), lambda r: r["per_q"][0].update(selected_iq=True),
                     lambda r: r["per_q"][0].update(q_weight=.125),
                     lambda r: r["per_q"][0]["frequency_ha"].pop(),
                     lambda r: r["per_q"][0]["frequency_ha"].__setitem__(0, -1),
                     lambda r: r["per_q"][0]["frequency_ha"].__setitem__(0, .05),
                     lambda r: r["per_q"][0]["candidate_contributions_ha"].__setitem__(0, .1),
                     lambda r: r["per_q"][0]["reference_contributions_ha"].__setitem__(0, .1),
                     lambda r: r["per_q"][0]["candidate_contributions_ha"].__setitem__(0, float("nan"))]
        for name in ("center", "candidate"):
            for index, mutate in enumerate(mutations):
                result = copy.deepcopy(original)
                mutate(result[name]["rpa"])
                self.repin(result)
                with self.subTest(name=name, mutation=index), self.assertRaises(ValueError):
                    self.load()
        # Preserving totals must not hide a changed frozen reference or center.
        for name, key in (("center", "candidate_contributions_ha"),
                          ("center", "reference_contributions_ha"),
                          ("candidate", "reference_contributions_ha")):
            result = copy.deepcopy(original)
            values = result[name]["rpa"]["per_q"][0][key]
            values[0] += 1e-5
            values[1] -= 1e-5
            self.repin(result)
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                self.load()

    def test_reproduction_loss_error_and_reported_units_cannot_be_spoofed(self):
        original = read_json(self.stage / "result/RESULT.json")
        for name, key, value in (("center", "loss", .1),
                                  ("candidate", "loss", original["center"]["loss"]),
                                  ("candidate", "rpa_correlation_energy_ev_per_c", -4.)):
            result = copy.deepcopy(original)
            result[name][key] = value
            self.repin(result)
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                self.load()
        result = copy.deepcopy(original)
        result["candidate"] = copy.deepcopy(result["center"])
        result["candidate"]["loss"] /= 2
        self.repin(result)
        with self.assertRaises(ValueError):
            self.load()
        for key in ("body_error_ev_per_c", "measured_ec_delta_ev_per_c"):
            result = dict(original, **{key: 0.})
            self.repin(result)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()

    def test_finite_but_inconsistent_loss_and_error_components_reject(self):
        original = read_json(self.stage / "result/RESULT.json")
        for key in ("loss", "pi_relative_squared_error", "trace_log_relative_squared_error",
                    "energy_relative_squared_error"):
            result = copy.deepcopy(original)
            target = result["candidate"] if key == "loss" else result["candidate"]["rpa"]
            target[key] *= .9
            self.repin(result)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()

    def test_symlinks_missing_status_duplicate_json_and_oversized_files_reject(self):
        path = self.stage / "STATUS"
        path.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.load()
        path.symlink_to(ARCHIVE / "STATUS")
        with self.assertRaises(ValueError):
            self.load()
        path.unlink()
        path.write_text("success\n")
        path = self.stage / "result/RESULT.json"
        saved = path.read_bytes()
        path.write_bytes(saved.rstrip()[:-1] + b', "status": "success"}\n')
        self.repin()
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.load()
        path.write_bytes(saved)
        self.repin()
        path = self.stage / "result/candidate/COEFFICIENTS.txt"
        with path.open("wb") as stream:
            stream.truncate(8 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
