"""Hash-bound PBE-only probes; no scheduler or electronic-structure execution."""

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest

import test_c_optimized_pbe as pbe

sys.path.insert(0, str(pbe.MODULE.parent))


class CDirectionProbePbeTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("check_c_direction_probe_pbe"),
                             "direction probe PBE helper missing")
        self.helper = importlib.import_module("check_c_direction_probe_pbe")
        import test_periodic_galerkin_c_backoff as backoff
        self.backoff = backoff.CBackoffTest()
        self.backoff.setUp()
        self.addCleanup(self.backoff.doCleanups)
        self.backoff.produce()
        self.fixture = self.backoff.fixture
        self.root = self.fixture.root
        self.center = self.fixture.output
        result = self.backoff.output / "RESULT.json"
        self.center_preparation = self.fixture.prepare(
            optimizer_result=result, optimizer_result_sha256=pbe.digest(result))
        self.center_collection = self.fixture.collect(pbe.log_text(pbe.BASELINE + .010))
        self.probe_dir = self.root / "probe"
        self.probe_dir.mkdir()
        (self.probe_dir / "COEFFICIENTS.txt").write_text("signed probe coefficients\n")
        (self.probe_dir / "C_3s3p2d_probe.orb").write_text("Element C\nsigned probe orbital\n")
        self.probe = dict(status="prepared", scope="direction_calibration_probe",
            direction_name="T", signed_radius=.001,
            center_result_sha256=pbe.digest(result),
            center_coefficient_sha256=self.center_preparation["candidate_coefficient_sha256"],
            center_orbital_sha256=self.center_preparation["candidate_orbital_sha256"],
            directions_sha256="d"*64, coefficient_filename="COEFFICIENTS.txt",
            coefficient_sha256=pbe.digest(self.probe_dir / "COEFFICIENTS.txt"),
            orbital_filename="C_3s3p2d_probe.orb",
            orbital_sha256=pbe.digest(self.probe_dir / "C_3s3p2d_probe.orb"),
            cheap_gate=dict(gate=True), galerkin_energy="unmeasured", physical_release_gate="hold")
        self.probe_path = self.probe_dir / "PROBE.json"
        self.write_probe()
        self.output = self.root / "pbe_probes" / "T_plus_001"

    def write_probe(self, **changes):
        self.probe_path.write_text(json.dumps(dict(self.probe, **changes)))

    def prepare(self, **changes):
        options = dict(center_dir=self.center,
            center_preparation_sha256=pbe.digest(self.center / pbe.endpoint.PREPARATION),
            center_collection_sha256=pbe.digest(self.center / pbe.endpoint.COLLECTION),
            probe_path=self.probe_path, probe_sha256=pbe.digest(self.probe_path), output=self.output)
        options.update(changes)
        return self.helper.prepare_probe_pbe(**options)

    def collect(self, text=None, **changes):
        if text is not None:
            path = self.output / "OUT.C_Q1" / "running_scf.log"
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        options = dict(prepared_dir=self.output,
            preparation_sha256=pbe.digest(self.output / self.helper.PREPARATION))
        options.update(changes)
        return self.helper.collect_probe_pbe(**options)

    def change_center_preparation(self, manifest):
        path = self.center / pbe.endpoint.PREPARATION
        path.write_text(json.dumps(manifest))
        collection = dict(self.center_collection, preparation_sha256=pbe.digest(path))
        (self.center / pbe.endpoint.COLLECTION).write_text(json.dumps(collection))

    def test_prepare_copies_only_pure_pbe_inputs_and_freezes_probe_evidence(self):
        before = {p: p.read_bytes() for p in self.center.rglob("*") if p.is_file()}
        manifest = self.prepare()
        for name in ("INPUT", "STRU", "KPT", "C.upf"):
            self.assertEqual((self.output / name).read_bytes(), (self.center / name).read_bytes())
        self.assertEqual((self.output / "C_original.orb").read_bytes(),
                         (self.probe_dir / "C_3s3p2d_probe.orb").read_bytes())
        values = pbe.endpoint.read_input(self.output / "INPUT")
        for name, expected in (("rpa", "0"), ("init_chg", "atomic"), ("init_wfc", "atomic")):
            self.assertEqual(values[name], expected)
        self.assertEqual(manifest["scope"], "direction_calibration_probe")
        self.assertEqual(manifest["physical_release_gate"], "hold")
        self.assertEqual(manifest.get("galerkin_energy"), "unmeasured")
        self.assertEqual(manifest["probe_sha256"], pbe.digest(self.probe_path))
        for name in ("PROBE.json", "COEFFICIENTS.txt", "C_3s3p2d_probe.orb"):
            frozen = ".provenance/" + name
            self.assertEqual(manifest["evidence_sha256"][frozen], pbe.digest(self.output / frozen))
        self.assertEqual(set(p.name for p in self.output.iterdir()),
            {"INPUT", "STRU", "KPT", "C.upf", "C_original.orb", ".provenance", self.helper.PREPARATION})
        self.assertFalse({"best", "optimized_result", "candidate_energy_ha"} & set(manifest))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_wrong_hashes_and_mutated_probe_artifacts_reject_before_staging(self):
        for key in ("center_preparation_sha256", "center_collection_sha256", "probe_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(**{key: "0"*64})
        for name in ("COEFFICIENTS.txt", "C_3s3p2d_probe.orb"):
            path = self.probe_dir / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"mutation")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.prepare()
            path.write_bytes(saved)
        self.assertFalse(self.output.exists())

    def test_wrong_center_unsafe_names_and_out_of_budget_probes_reject(self):
        changes = [{key: "0"*64} for key in (
            "center_result_sha256", "center_coefficient_sha256", "center_orbital_sha256")]
        changes += [dict(direction_name="X"), dict(signed_radius=0), dict(signed_radius=.003),
            dict(signed_radius=True), dict(signed_radius=float("nan")),
            dict(coefficient_filename="../COEFFICIENTS.txt"),
            dict(orbital_filename="/tmp/C.orb"), dict(directions_sha256="bad"),
            dict(cheap_gate=dict(gate=1)), dict(cheap_gate=dict(gate=False)),
            dict(galerkin_energy=-.43),
            dict(physical_release_gate="pass"), dict(status="success"),
            dict(scope="optimized_full_q_frozen_body_rpa_calibration")]
        for change in changes:
            self.write_probe(**change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_center_evidence_logs_and_input_mutations_reject(self):
        paths = [self.center / name for name in self.center_preparation["prepared_input_sha256"]]
        paths += [self.center / name for name in self.center_preparation["evidence_sha256"]]
        paths.append(self.center / "OUT.C_Q1/running_scf.log")
        for path in paths:
            saved = path.read_bytes()
            path.write_bytes(saved + b"changed")
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.prepare()
            path.write_bytes(saved)
            self.assertFalse(self.output.exists())

    def test_center_collection_must_match_logs_quarter_binding_and_actual_gate(self):
        cases = [dict(pbe_total_energy_gate="fail"), dict(scf_log_gate="fail"),
            dict(band_count_check="unavailable"), dict(band_counts=dict(occupied=None, total=None)),
            dict(candidate_energy_ev=pbe.BASELINE), dict(energy_delta_ev_per_c=0),
            dict(tolerance_ev_per_c=.1), dict(candidate_alpha=.5),
            dict(candidate_result_sha256="0"*64), dict(candidate_orbital_sha256="0"*64),
            dict(preparation_sha256="0"*64)]
        for change in cases:
            (self.center / pbe.endpoint.COLLECTION).write_text(json.dumps(dict(self.center_collection, **change)))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_center_missing_counts_nonconvergence_and_over_tolerance_reject_even_rehashed(self):
        path = self.center / "OUT.C_Q1/running_scf.log"
        for text in (pbe.log_text(pbe.BASELINE+.010, counts=False),
                     pbe.log_text(pbe.BASELINE+.010).replace("states = 4", "states = 3"),
                     pbe.log_text(pbe.BASELINE+.010).replace("#SCF IS CONVERGED#", ""),
                     pbe.log_text(pbe.BASELINE+.022)):
            path.write_text(text)
            collection = dict(self.center_collection, candidate_log_sha256=pbe.digest(path))
            if text == pbe.log_text(pbe.BASELINE+.022):
                collection.update(candidate_energy_ev=pbe.BASELINE+.022, energy_delta_ev_per_c=.011)
            (self.center / pbe.endpoint.COLLECTION).write_text(json.dumps(collection))
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.prepare()

    def test_rehashed_center_input_cannot_restore_restart_or_old_wfc(self):
        input_path = self.center / "INPUT"
        original = input_path.read_text()
        for text in (original.replace("rpa 0", "rpa 1"),
                     original.replace("init_wfc atomic", "init_wfc file"),
                     original + "read_file_dir ../old/\n",
                     original + "restart_load true\n"):
            input_path.write_text(text)
            manifest = copy.deepcopy(self.center_preparation)
            manifest["prepared_input_sha256"]["INPUT"] = pbe.digest(input_path)
            self.change_center_preparation(manifest)
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.prepare()

    def test_unsafe_center_paths_and_extra_old_wfc_inputs_reject(self):
        for group, name in (("evidence_sha256", "../escape"),
                            ("prepared_input_sha256", "old.wfc")):
            manifest = copy.deepcopy(self.center_preparation)
            (self.center / "old.wfc").write_text("old wavefunction")
            manifest[group][name] = "a"*64
            self.change_center_preparation(manifest)
            with self.subTest(group=group), self.assertRaises(ValueError):
                self.prepare()
        outside = self.root / "outside.orb"
        outside.write_bytes((self.probe_dir / "C_3s3p2d_probe.orb").read_bytes())
        (self.probe_dir / "C_3s3p2d_probe.orb").unlink()
        (self.probe_dir / "C_3s3p2d_probe.orb").symlink_to(outside)
        self.change_center_preparation(self.center_preparation)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_eight_unique_slots_only_and_duplicate_output_or_slot_rejected(self):
        for direction in ("T", "N"):
            for radius in (-.002, -.001, .001, .002):
                self.write_probe(direction_name=direction, signed_radius=radius)
                path = self.output.parent / (direction + str(radius))
                self.prepare(output=path)
                with self.assertRaises(FileExistsError):
                    self.prepare(output=path)
        self.write_probe(direction_name="T", signed_radius=.001)
        with self.assertRaises(ValueError):
            self.prepare(output=self.output.parent / "duplicate_slot")
        self.assertEqual(len(list(self.output.parent.glob("*/" + self.helper.PREPARATION))), 8)

    def test_duplicate_slot_before_budget_exhaustion_and_changed_campaign_reject(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare(output=self.output.parent / "same_slot_again")
        self.write_probe(direction_name="N", directions_sha256="e"*64)
        with self.assertRaisesRegex(ValueError, "campaign"):
            self.prepare(output=self.output.parent / "different_directions")

    def test_missing_scf_is_not_valid_and_duplicate_json_keys_reject(self):
        self.prepare()
        with self.assertRaises(FileNotFoundError):
            self.collect()
        self.assertFalse((self.output / self.helper.COLLECTION).exists())
        self.probe_path.write_text(self.probe_path.read_text()[:-1] + ', "direction_name": "N"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.prepare(output=self.output.parent / "duplicate_json")

    def test_collect_reuses_original_and_center_and_constraint_failure_is_not_scf_failure(self):
        for index, delta in enumerate((.008, .010, .011, -.011)):
            self.write_probe(signed_radius=(-.002, -.001, .001, .002)[index])
            self.output = self.output.parent / ("case" + str(index))
            self.prepare()
            result = self.collect(pbe.log_text(pbe.BASELINE + 2*delta))
            self.assertEqual(result["status"], "collected")
            self.assertEqual(result["scope"], "direction_calibration_probe")
            self.assertAlmostEqual(result["energy_delta_ev_per_c"], delta, places=12)
            self.assertAlmostEqual(result["delta_from_center_ev_per_c"], delta-.005, places=12)
            self.assertEqual(result["pbe_gate"], "pass" if abs(delta) <= .010 else "fail")
            self.assertEqual(result["scf_log_gate"], "pass")
            self.assertEqual(result["band_count_check"], "pass")
            self.assertEqual(result["band_counts"], dict(occupied=4, total=44))
            self.assertEqual(result["physical_release_gate"], "hold")
            self.assertEqual(result.get("galerkin_energy"), "unmeasured")
            self.assertNotIn("candidate_energy_ha", result)

    def test_collect_rejects_missing_counts_nonconvergence_and_duplicate_energy(self):
        self.prepare()
        for text in (pbe.log_text(counts=False),
                     pbe.log_text().replace("states = 4", "states = 3"),
                     pbe.log_text().replace("NBANDS) = 44", "NBANDS) = 43"),
                     pbe.log_text().replace("#SCF IS CONVERGED#", ""),
                     pbe.log_text() + "!!SCF IS NOT CONVERGED!!\n",
                     pbe.log_text() + "!FINAL_ETOT_IS -310 eV\n",
                     pbe.log_text().replace("!FINAL_ETOT_IS", "Etot_without_rpa(Ha):")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.collect(text)
            self.assertFalse((self.output / self.helper.COLLECTION).exists())

    def test_collect_validates_frozen_evidence_and_inputs_and_never_overwrites(self):
        manifest = self.prepare()
        for name in list(manifest["prepared_input_sha256"]) + list(manifest["evidence_sha256"]):
            path = self.output / name
            before = path.read_bytes()
            path.write_bytes(before + b"mutation")
            with self.subTest(path=name), self.assertRaises(ValueError):
                self.collect(pbe.log_text())
            path.write_bytes(before)
        with self.assertRaises(ValueError):
            self.collect(pbe.log_text(), preparation_sha256="0"*64)
        self.collect(pbe.log_text())
        with self.assertRaises(FileExistsError):
            self.collect()

    def test_collect_works_from_evidence_after_original_center_and_probe_removed(self):
        self.prepare()
        shutil.rmtree(self.center)
        shutil.rmtree(self.probe_dir)
        shutil.rmtree(self.backoff.output)
        shutil.rmtree(self.fixture.original)
        result = self.collect(pbe.log_text(pbe.BASELINE+.016))
        self.assertAlmostEqual(result["delta_from_center_ev_per_c"], .003, places=12)


if __name__ == "__main__":
    unittest.main()
