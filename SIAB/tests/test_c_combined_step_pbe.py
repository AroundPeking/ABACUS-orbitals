"""Bounded combined-step staging only; no electronic-structure execution."""

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

WORKFLOW = (Path(__file__).resolve().parents[1] / "example_C_sternheimer" /
            "periodic_basis_optimization" / "galerkin_binding_workflow")
sys.path.insert(0, str(WORKFLOW))


class CCombinedStepPbeAvailabilityTest(unittest.TestCase):
    def test_combined_step_helper_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("check_c_combined_step_pbe"),
                             "combined-step PBE helper missing")


class CCombinedStepPbeTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("check_c_combined_step_pbe"),
                             "combined-step PBE helper missing")
        self.helper = importlib.import_module("check_c_combined_step_pbe")
        import test_c_direction_probe_pbe as probes
        self.pbe = probes.pbe
        self.probes = probes.CDirectionProbePbeTest()
        self.probes.setUp()
        self.addCleanup(self.probes.doCleanups)
        self.root, self.center = self.probes.root, self.probes.center
        self.source = self.root / "combined"
        self.source.mkdir()
        self.output = self.root / "combined_pbe" / "candidate"
        self.center_prep_sha = self.pbe.digest(self.center / self.pbe.endpoint.PREPARATION)
        self.center_coll_sha = self.pbe.digest(self.center / self.pbe.endpoint.COLLECTION)
        self.center_energy = self.probes.center_collection["candidate_energy_ev"]
        self.bindings = {key: self.probes.probe[key] for key in (
            "center_result_sha256", "center_coefficient_sha256", "center_orbital_sha256")}
        self.directions = dict(self.bindings, status="success",
            scope="two_direction_actual_pbe_calibration", physical_release_gate="hold")
        self.write_json(self.source / "DIRECTIONS.json", self.directions)
        self.directions_sha = self.pbe.digest(self.source / "DIRECTIONS.json")
        samples = []
        for name, slope in (("T", .2), ("N", 1.)):
            for radius in (-.001, .001, -.002, .002):
                energy = self.center_energy + 2*slope*radius + .4*radius**2
                samples.append(dict(self.bindings, status="collected",
                    scope="direction_calibration_probe", direction_name=name, signed_radius=radius,
                    directions_sha256=self.directions_sha, center_preparation_sha256=self.center_prep_sha,
                    center_collection_sha256=self.center_coll_sha, center_energy_ev=self.center_energy,
                    baseline_energy_ev=self.pbe.BASELINE, candidate_energy_ev=energy,
                    energy_delta_ev_per_c=(energy-self.pbe.BASELINE)/2,
                    delta_from_center_ev_per_c=(energy-self.center_energy)/2,
                    energy_quantity="PBE_total_energy_not_RPA_E0", tolerance_ev_per_c=.010,
                    pbe_gate="pass", scf_log_gate="pass", band_count_check="pass",
                    band_counts=dict(occupied=4, total=44), physical_release_gate="hold"))
        from periodic_galerkin_direction_calibration import analyze_pbe_axes
        self.analyze = analyze_pbe_axes
        self.calibration = dict(self.analyze(self.center_energy, samples), samples=samples,
            center_collection_sha256=self.center_coll_sha, actual_scf_count=8)
        self.write_json(self.source / "CALIBRATION.json", self.calibration)
        (self.source / "COEFFICIENTS.txt").write_text("combined coefficients\n")
        (self.source / "C_3s3p2d_combined.orb").write_text("Element C\ncombined orbital\n")
        axes = self.calibration["axes"]
        self.candidate = dict(self.bindings, status="prepared", scope="actual_pbe_tangent_candidate",
            direction_name="actual_pbe_tangent", radius=.02,
            mixing_ratio=axes["T"][0]["derivative_ev_per_c"]/axes["N"][0]["derivative_ev_per_c"],
            directions_sha256=self.directions_sha, calibration_sha256=self.pbe.digest(self.source / "CALIBRATION.json"),
            cheap_gate=dict(gate=True), coefficient_filename="COEFFICIENTS.txt",
            coefficient_sha256=self.pbe.digest(self.source / "COEFFICIENTS.txt"),
            orbital_filename="C_3s3p2d_combined.orb",
            orbital_sha256=self.pbe.digest(self.source / "C_3s3p2d_combined.orb"),
            galerkin_energy="unmeasured", physical_release_gate="hold")
        self.candidate_path = self.source / "CANDIDATE.json"
        self.write_candidate()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value))

    def write_candidate(self, **changes):
        self.write_json(self.candidate_path, dict(self.candidate, **changes))

    def write_calibration(self, calibration):
        path = self.source / "CALIBRATION.json"
        self.write_json(path, calibration)
        self.write_candidate(calibration_sha256=self.pbe.digest(path))

    def prepare(self, **changes):
        options = dict(center_dir=self.center, center_preparation_sha256=self.center_prep_sha,
            center_collection_sha256=self.center_coll_sha, candidate_path=self.candidate_path,
            candidate_sha256=self.pbe.digest(self.candidate_path), output=self.output)
        return self.helper.prepare_combined_pbe(**dict(options, **changes))

    def collect(self, text=None, **changes):
        if text is not None:
            path = self.output / "OUT.C_Q1/running_scf.log"
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        options = dict(prepared_dir=self.output,
            preparation_sha256=self.pbe.digest(self.output / self.helper.PREPARATION))
        return self.helper.collect_combined_pbe(**dict(options, **changes))

    def test_prepare_exact_center_inputs_and_complete_portable_evidence(self):
        before = {p: p.read_bytes() for p in self.center.rglob("*") if p.is_file()}
        manifest = self.prepare()
        self.assertEqual(self.helper.PREPARATION, "COMBINED_STEP_PBE_PREPARED.json")
        self.assertEqual(self.helper.COLLECTION, "COMBINED_STEP_PBE_COLLECTION.json")
        for name in ("INPUT", "STRU", "KPT", "C.upf"):
            self.assertEqual((self.output / name).read_bytes(), (self.center / name).read_bytes())
        self.assertEqual((self.output / "C_original.orb").read_bytes(),
                         (self.source / "C_3s3p2d_combined.orb").read_bytes())
        for name in ("CANDIDATE.json", "COEFFICIENTS.txt", "C_3s3p2d_combined.orb",
                     "DIRECTIONS.json", "CALIBRATION.json"):
            self.assertEqual((self.output / ".provenance" / name).read_bytes(),
                             (self.source / name).read_bytes())
        values = self.pbe.endpoint.read_input(self.output / "INPUT")
        for key, value in (("rpa", "0"), ("dft_functional", "pbe"),
                           ("init_chg", "atomic"), ("init_wfc", "atomic")):
            self.assertEqual(values[key], value)
        self.assertEqual(manifest["candidate_sha256"], self.pbe.digest(self.candidate_path))
        self.assertEqual(manifest["scope"], "actual_pbe_tangent_candidate")
        self.assertEqual(manifest["galerkin_energy"], "unmeasured")
        self.assertEqual(manifest["physical_release_gate"], "hold")
        self.assertNotIn("probe_sha256", manifest)
        self.assertEqual(set(p.name for p in self.output.iterdir()),
            {"INPUT", "STRU", "KPT", "C.upf", "C_original.orb", ".provenance", self.helper.PREPARATION})
        frozen, baseline, center = self.helper._prepared(self.output, self.pbe.digest(self.output / self.helper.PREPARATION))
        self.assertEqual(frozen, manifest)
        self.assertEqual(baseline["energy_ev"], self.pbe.BASELINE)
        self.assertEqual(center["energy_ev"], self.center_energy)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        shutil.rmtree(self.center)
        shutil.rmtree(self.source)
        shutil.rmtree(self.probes.backoff.output)
        shutil.rmtree(self.probes.fixture.original)
        self.assertEqual(self.collect(self.pbe.log_text())["pbe_gate"], "pass")

    def test_wrong_hashes_and_all_candidate_artifact_mutations_reject(self):
        for key in ("center_preparation_sha256", "center_collection_sha256", "candidate_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(**{key: "0"*64})
        for name in ("DIRECTIONS.json", "CALIBRATION.json", "COEFFICIENTS.txt", "C_3s3p2d_combined.orb"):
            path = self.source / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"mutation")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.prepare()
            path.write_bytes(saved)
        self.assertFalse(self.output.exists())

    def test_candidate_identity_radius_numbers_and_cheap_gate_are_strict(self):
        changes = [{key: "0"*64} for key in self.bindings]
        changes += [dict(status="success"), dict(scope="direction_calibration_probe"),
            dict(direction_name="T"), dict(radius=.001), dict(radius=-.02), dict(radius=True),
            dict(radius="0.02"), dict(radius=float("nan")), dict(radius=float("inf")),
            dict(mixing_ratio=True), dict(mixing_ratio="0.2"), dict(mixing_ratio=float("nan")),
            dict(mixing_ratio=float("inf")), dict(mixing_ratio=.3), dict(directions_sha256="bad"),
            dict(cheap_gate=dict(gate=1)), dict(cheap_gate=dict(gate=False)),
            dict(coefficient_filename="../COEFFICIENTS.txt"), dict(orbital_filename="C_3s3p2d_probe.orb"),
            dict(galerkin_energy=-.4), dict(physical_release_gate="pass")]
        for change in changes:
            self.write_candidate(**change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())
        for key in ("galerkin_energy", "mixing_ratio", "radius", "cheap_gate"):
            changed = dict(self.candidate)
            changed.pop(key)
            self.write_json(self.candidate_path, changed)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.prepare()

    def test_wrong_filename_duplicate_json_and_symlink_artifacts_reject(self):
        alias = self.source / "PROBE.json"
        alias.write_bytes(self.candidate_path.read_bytes())
        with self.assertRaises(ValueError):
            self.prepare(candidate_path=alias)
        saved = self.candidate_path.read_bytes()
        self.candidate_path.write_bytes(saved[:-1] + b', "radius": 0.02}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.prepare()
        self.candidate_path.write_bytes(saved)
        for name in ("CANDIDATE.json", "DIRECTIONS.json", "CALIBRATION.json", "COEFFICIENTS.txt", "C_3s3p2d_combined.orb"):
            path = self.source / name
            data = path.read_bytes()
            target = self.source / "alias"
            target.write_bytes(data)
            path.unlink()
            path.symlink_to(target)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.prepare()
            path.unlink()
            path.write_bytes(data)
            target.unlink()

    def test_rehashed_directions_and_calibration_must_match_center(self):
        for key in self.bindings:
            self.write_json(self.source / "DIRECTIONS.json", dict(self.directions, **{key: "0"*64}))
            digest = self.pbe.digest(self.source / "DIRECTIONS.json")
            changed = copy.deepcopy(self.calibration)
            for sample in changed["samples"]:
                sample["directions_sha256"] = digest
            self.write_calibration(changed)
            current = json.loads(self.candidate_path.read_text())
            self.write_json(self.candidate_path, dict(current, directions_sha256=digest))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare()
        self.write_json(self.source / "DIRECTIONS.json", self.directions)
        for change in (dict(status="prepared"), dict(scope="direction_calibration_probe"),
                       dict(actual_scf_count=7), dict(actual_scf_count=True), dict(actual_scf_count=8.),
                       dict(consistency_gate="fail"), dict(center_collection_sha256="0"*64),
                       dict(center_energy_ev=self.center_energy+.001), dict(physical_release_gate="pass")):
            self.write_calibration(dict(self.calibration, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()

    def test_calibration_samples_bind_exact_eight_passed_measurements(self):
        cases = [{key: "0"*64} for key in self.bindings]
        cases += [dict(directions_sha256="0"*64), dict(center_collection_sha256="0"*64),
            dict(center_preparation_sha256="0"*64), dict(scope="actual_pbe_tangent_candidate"),
            dict(status="prepared"), dict(pbe_gate="fail"), dict(scf_log_gate="fail"),
            dict(band_count_check="unavailable"), dict(band_counts=dict(occupied=3, total=44)),
            dict(band_counts=dict(occupied=4, total=43)), dict(signed_radius=True),
            dict(signed_radius=.003), dict(direction_name="X"), dict(candidate_energy_ev=float("nan")),
            dict(candidate_energy_ev=True), dict(energy_delta_ev_per_c=0.),
            dict(delta_from_center_ev_per_c=0.), dict(center_energy_ev=self.pbe.BASELINE)]
        for change in cases:
            calibration = copy.deepcopy(self.calibration)
            calibration["samples"][0].update(change)
            self.write_calibration(calibration)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
        for samples in (self.calibration["samples"][:7], self.calibration["samples"]*2,
                        self.calibration["samples"][:-1] + [self.calibration["samples"][0]]):
            self.write_calibration(dict(self.calibration, samples=samples))
            with self.assertRaises(ValueError):
                self.prepare()

    def test_recompute_axes_consistency_mixing_and_false_pass(self):
        changed = copy.deepcopy(self.calibration)
        changed["axes"]["T"][0]["derivative_ev_per_c"] += .1
        self.write_calibration(changed)
        with self.assertRaises(ValueError):
            self.prepare()
        for mode in ("inconsistent", "over_tolerance", "zero_normal"):
            changed = copy.deepcopy(self.calibration)
            for sample in changed["samples"]:
                if mode == "over_tolerance":
                    sample["candidate_energy_ev"] += .03
                elif mode == "zero_normal" and sample["direction_name"] == "N":
                    sample["candidate_energy_ev"] = self.center_energy
            if mode == "inconsistent":
                changed["samples"][0]["candidate_energy_ev"] += .001
            for sample in changed["samples"]:
                energy = sample["candidate_energy_ev"]
                sample.update(energy_delta_ev_per_c=(energy-self.pbe.BASELINE)/2,
                              delta_from_center_ev_per_c=(energy-self.center_energy)/2)
            changed.update(self.analyze(self.center_energy, changed["samples"]))
            changed["consistency_gate"] = "pass"
            self.write_calibration(changed)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.prepare()

    def test_center_pure_pbe_atomic_init_cannot_be_rehashed_away(self):
        path = self.center / "INPUT"
        original = path.read_text()
        for text in (original.replace("rpa 0", "rpa 1"),
                     original.replace("init_wfc atomic", "init_wfc file"),
                     original.replace("init_chg atomic", "init_chg file"),
                     original.replace("dft_functional pbe", "dft_functional hse"),
                     original + "restart_load true\n", original + "read_file_dir ../old/\n"):
            path.write_text(text)
            manifest = copy.deepcopy(self.probes.center_preparation)
            manifest["prepared_input_sha256"]["INPUT"] = self.pbe.digest(path)
            self.probes.change_center_preparation(manifest)
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.prepare(center_preparation_sha256=self.pbe.digest(self.center / self.pbe.endpoint.PREPARATION),
                             center_collection_sha256=self.pbe.digest(self.center / self.pbe.endpoint.COLLECTION))

    def test_one_slot_per_parent_rejects_duplicate_even_without_preparation(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        other = self.output.parent / "second"
        with self.assertRaisesRegex(ValueError, "slot|campaign|duplicate"):
            self.prepare(output=other)
        (self.output / self.helper.PREPARATION).unlink()
        with self.assertRaises(ValueError):
            self.prepare(output=other)
        self.assertFalse(other.exists())

    def test_preexisting_empty_incomplete_symlink_and_locked_parent_reject(self):
        self.output.parent.mkdir()
        for mode in ("empty", "incomplete", "symlink", "file"):
            entry = self.output.parent / "old_slot"
            if mode in ("empty", "incomplete"):
                entry.mkdir()
                if mode == "incomplete":
                    (entry / "INPUT").write_text("incomplete")
            elif mode == "symlink":
                entry.symlink_to(self.root / "absent", target_is_directory=True)
            else:
                entry.write_text("preexisting")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.prepare()
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            else:
                shutil.rmtree(entry)
        self.output.symlink_to(self.root / "absent", target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.output.unlink()
        lock = self.output.parent / ".combined_step_pbe_prepare.lock"
        lock.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertTrue(lock.is_dir())
        self.assertFalse(self.output.exists())

    def test_failed_write_reserves_slot_and_does_not_overwrite(self):
        original = self.helper.endpoint._write_json
        def fail(path, result):
            if Path(path).name == self.helper.PREPARATION:
                raise OSError("interrupted staging")
            return original(path, result)
        with mock.patch.object(self.helper.endpoint, "_write_json", side_effect=fail):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.prepare()
        saved = (self.output / "INPUT").read_bytes()
        with self.assertRaises(ValueError):
            self.prepare(output=self.output.parent / "retry")
        self.assertEqual((self.output / "INPUT").read_bytes(), saved)

    def test_lock_is_held_until_preparation_is_written(self):
        original = self.helper.endpoint._write_json
        def compete(path, result):
            self.assertTrue((self.output.parent / ".combined_step_pbe_prepare.lock").is_dir())
            with self.assertRaises(FileExistsError):
                self.prepare(output=self.output.parent / "competing_candidate")
            return original(path, result)
        with mock.patch.object(self.helper.endpoint, "_write_json", side_effect=compete):
            self.prepare()
        self.assertFalse((self.output.parent / ".combined_step_pbe_prepare.lock").exists())
        self.assertFalse((self.output.parent / "competing_candidate").exists())

    def test_collection_compares_original_and_center_but_keeps_release_on_hold(self):
        for index, delta in enumerate((.008, .010, -.010, .011, -.011)):
            self.output = self.root / ("campaign_%d" % index) / "candidate"
            self.prepare()
            result = self.collect(self.pbe.log_text(self.pbe.BASELINE+2*delta))
            self.assertEqual(result["status"], "collected")
            self.assertEqual(result["scope"], "actual_pbe_tangent_candidate")
            self.assertAlmostEqual(result["energy_delta_ev_per_c"], delta, places=12)
            self.assertAlmostEqual(result["delta_from_center_ev_per_c"], delta-.005, places=12)
            self.assertEqual(result["pbe_gate"], "pass" if abs(delta) <= .010 else "fail")
            self.assertEqual(result["scf_log_gate"], "pass")
            self.assertEqual(result["band_counts"], dict(occupied=4, total=44))
            self.assertEqual(result["physical_release_gate"], "hold")
            self.assertEqual(result["galerkin_energy"], "unmeasured")
            self.assertNotIn("candidate_energy_ha", result)

    def test_collection_rejects_nonconvergence_bad_counts_and_nonfinite(self):
        self.prepare()
        with self.assertRaises(FileNotFoundError):
            self.collect()
        log = self.pbe.log_text()
        texts = [self.pbe.log_text(counts=False), log.replace("states = 4", "states = 3"),
            log.replace("NBANDS) = 44", "NBANDS) = 43"), log.replace("#SCF IS CONVERGED#", ""),
            log + "!!SCF IS NOT CONVERGED!!\n", log + "!FINAL_ETOT_IS -310 eV\n",
            log.replace("!FINAL_ETOT_IS", "Etot_without_rpa(Ha):")]
        texts += [self.pbe.log_text(value) for value in (float("nan"), float("inf"), -float("inf"))]
        for text in texts:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.collect(text)
            self.assertFalse((self.output / self.helper.COLLECTION).exists())

    def test_collect_revalidates_every_hash_and_never_overwrites(self):
        manifest = self.prepare()
        for name in list(manifest["prepared_input_sha256"]) + list(manifest["evidence_sha256"]):
            path = self.output / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"mutation")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.collect(self.pbe.log_text())
            path.write_bytes(saved)
        with self.assertRaises(ValueError):
            self.collect(self.pbe.log_text(), preparation_sha256="0"*64)
        self.collect(self.pbe.log_text())
        path = self.output / self.helper.COLLECTION
        saved = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.collect(self.pbe.log_text(self.pbe.BASELINE+.03))
        self.assertEqual(path.read_bytes(), saved)
        path.unlink()
        target = self.root / "untouched"
        path.symlink_to(target)
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertFalse(target.exists())

    def test_runtime_preflight_rejects_rehashed_input_scope_and_math_tampering(self):
        manifest = self.prepare()
        prep = self.output / self.helper.PREPARATION
        path = self.output / "INPUT"
        saved = path.read_bytes()
        path.write_bytes(saved + b"scf_nmax 200\n")
        changed = copy.deepcopy(manifest)
        changed["prepared_input_sha256"]["INPUT"] = self.pbe.digest(path)
        self.write_json(prep, changed)
        with self.assertRaises(ValueError):
            self.collect(self.pbe.log_text())
        path.write_bytes(saved)
        for change in (dict(scope="direction_calibration_probe"), dict(radius=.001),
                       dict(mixing_ratio=.3), dict(directions_sha256="0"*64),
                       dict(calibration_sha256="0"*64), dict(physical_release_gate="pass")):
            self.write_json(prep, dict(manifest, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.collect()
        evidence = self.output / ".provenance"
        candidate = dict(self.candidate, mixing_ratio=.3)
        self.write_json(evidence / "CANDIDATE.json", candidate)
        changed = copy.deepcopy(manifest)
        digest = self.pbe.digest(evidence / "CANDIDATE.json")
        changed.update(candidate_sha256=digest, mixing_ratio=.3)
        changed["evidence_sha256"][".provenance/CANDIDATE.json"] = digest
        self.write_json(prep, changed)
        with self.assertRaises(ValueError):
            self.collect()


if __name__ == "__main__":
    unittest.main()
