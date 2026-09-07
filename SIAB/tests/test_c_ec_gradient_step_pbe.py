"""Pure stdlib PBE staging tests; scientific candidate/center APIs are mocked."""

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import test_c_optimized_pbe as fixture


WORKFLOW = fixture.MODULE.parent
sys.path.insert(0, str(WORKFLOW))
PREPARATION = "EC_GRADIENT_STEP_PBE_PREPARED.json"
COLLECTION = "EC_GRADIENT_STEP_PBE_COLLECTION.json"
SCOPE = "accepted_ec_gradient_step_candidate"
BASELINE = fixture.BASELINE
LOG = "OUT.C_Q1/running_scf.log"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


class EcGradientStepPbeTest(unittest.TestCase):
    def patch(self, target, name, **options):
        patcher = mock.patch.object(target, name, **options)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("check_c_ec_gradient_step_pbe"),
                             "own-scope Ec-step PBE adapter missing")
        self.helper = importlib.import_module("check_c_ec_gradient_step_pbe")
        temporary = tempfile.TemporaryDirectory(prefix="c-ec-step-pbe-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.center = self.root / "accepted-center"
        self.slot = self.center / "result/pbe_slots/candidate"
        self.slot.mkdir(parents=True)
        values = fixture.endpoint._pure_pbe(fixture.endpoint._input_values(fixture.INPUT.encode()))
        contents = {"INPUT": "INPUT_PARAMETERS\n"+"\n".join(k+" "+v for k, v in values.items())+"\n",
                    "STRU": fixture.STRU, "KPT": "K_POINTS\n0\nGamma\n4 4 4 0 0 0\n",
                    "C.upf": "C PBE pseudopotential\n", "C_original.orb": "accepted center orbital\n"}
        for name, content in contents.items():
            (self.slot / name).write_text(content)
        self.center_energy = BASELINE + .005
        (self.slot / LOG).parent.mkdir()
        (self.slot / LOG).write_text(fixture.log_text(self.center_energy))
        old_preparation = self.slot / "COMBINED_STEP_PBE_PREPARED.json"
        save(old_preparation, dict(prepared_input_sha256={n: fixture.digest(self.slot / n) for n in contents},
                                  candidate_log_relative_path=LOG, scope="actual_pbe_tangent_candidate"))
        self.center_result = dict(status="success", scope="one_actual_pbe_tangent_step",
            coefficient_sha256="a"*64, orbital_sha256=fixture.digest(self.slot / "C_original.orb"),
            actual_pbe=dict(preparation_sha256=fixture.digest(old_preparation),
                candidate_log_sha256=fixture.digest(self.slot / LOG), candidate_energy_ev=self.center_energy,
                baseline_energy_ev=BASELINE, baseline_log_sha256="b"*64))
        save(self.center / "result/RESULT.json", self.center_result)
        save(self.center / "COMBINED_STEP_ACCEPTANCE.json", dict(status="success", job_id="21894543"))
        self.center_sha = fixture.digest(self.center / "result/RESULT.json")
        self.acceptance_sha = fixture.digest(self.center / "COMBINED_STEP_ACCEPTANCE.json")
        self.candidate_root = self.root / "new/candidate"
        self.candidate_root.mkdir(parents=True)
        (self.candidate_root / "COEFFICIENTS.txt").write_text("new Ec-step coefficients\n")
        (self.candidate_root / "C_3s3p2d_ec_step.orb").write_text("new Ec-step orbital\n")
        save(self.candidate_root / "GRADIENT_STEP.json", dict(scope=SCOPE))
        self.gradient_stage = self.root / "gradient"
        self.gradient_stage.mkdir()
        self.candidate = dict(status="prepared", scope=SCOPE, center_stage=str(self.center),
            center_result_sha256=self.center_sha, center_acceptance_sha256=self.acceptance_sha,
            center_coefficient_sha256=self.center_result["coefficient_sha256"],
            center_orbital_sha256=self.center_result["orbital_sha256"],
            gradient_stage=str(self.gradient_stage), gradient_result_sha256="c"*64,
            gradient_acceptance_sha256="d"*64, direction_name="negative_horizontal_ec_gradient",
            radius=.02, physical_release_gate="hold",
            coefficient_filename="COEFFICIENTS.txt", orbital_filename="C_3s3p2d_ec_step.orb",
            coefficient_sha256=fixture.digest(self.candidate_root / "COEFFICIENTS.txt"),
            orbital_sha256=fixture.digest(self.candidate_root / "C_3s3p2d_ec_step.orb"),
            gradient_step_sha256=fixture.digest(self.candidate_root / "GRADIENT_STEP.json"))
        self.write_candidate()
        self.binary, self.mpi = self.root / "abacus", self.root / "libpmi.so"
        self.binary.write_bytes(b"pinned ABACUS executable")
        self.mpi.write_bytes(b"pinned MPI library")
        self.output = self.root / "new/pbe"
        self.options = dict(center_stage=self.center, center_result_sha256=self.center_sha,
            center_acceptance_sha256=self.acceptance_sha, candidate_root=self.candidate_root,
            output_root=self.output, abacus_binary=self.binary, expected_abacus_sha256=fixture.digest(self.binary),
            mpi_library=self.mpi, expected_mpi_sha256=fixture.digest(self.mpi))
        self.admit = self.patch(self.helper, "load_accepted_combined_step",
                               return_value=dict(result=self.center_result))
        self.verify_candidate = self.patch(self.helper, "validate_gradient_step_candidate",
            side_effect=lambda root: json.loads((Path(root) / "CANDIDATE.json").read_text()))

    def write_candidate(self, **changes):
        save(self.candidate_root / "CANDIDATE.json", dict(self.candidate, **changes))

    def prepare(self, **changes):
        return self.helper.prepare_ec_gradient_step_pbe(**dict(self.options, **changes))

    def collect(self, text=None, digest=None):
        if text is not None:
            path = self.output / LOG
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        return self.helper.collect_ec_gradient_step_pbe(self.output,
            fixture.digest(self.output / PREPARATION) if digest is None else digest)

    def test_prepare_owns_scope_and_changes_only_orbital_preserving_small_proof(self):
        before = {p: p.read_bytes() for root in (self.center, self.candidate_root)
                  for p in root.rglob("*") if p.is_file()}
        prepared = self.prepare()
        self.assertEqual(self.helper.PREPARATION, PREPARATION)
        self.assertEqual(self.helper.COLLECTION, COLLECTION)
        self.assertEqual(prepared["scope"], SCOPE)
        self.assertEqual(prepared["direction_name"], "negative_horizontal_ec_gradient")
        self.assertEqual(prepared["radius"], .02)
        self.assertEqual(prepared["physical_release_gate"], "hold")
        self.assertEqual(prepared["baseline_energy_ev"], BASELINE)
        self.assertEqual(prepared["center_energy_ev"], self.center_energy)
        self.assertEqual(prepared["candidate_log_relative_path"], LOG)
        for name in ("INPUT", "STRU", "KPT", "C.upf"):
            self.assertEqual((self.output / name).read_bytes(), (self.slot / name).read_bytes())
        self.assertEqual((self.output / "C_original.orb").read_bytes(),
                         (self.candidate_root / "C_3s3p2d_ec_step.orb").read_bytes())
        for name in ("CANDIDATE.json", "COEFFICIENTS.txt", "C_3s3p2d_ec_step.orb", "GRADIENT_STEP.json"):
            self.assertEqual((self.output / ".provenance" / name).read_bytes(),
                             (self.candidate_root / name).read_bytes())
        self.assertNotIn("candidate_alpha", prepared)
        self.assertNotIn("mixing_ratio", prepared)
        self.assertFalse((self.output / LOG).exists())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.admit.assert_called_with(self.center, self.center_sha, self.acceptance_sha)
        self.verify_candidate.assert_called_with(self.candidate_root)

    def test_reduced_radius_preserves_original_pbe_admission(self):
        self.write_candidate(radius=.018)
        prepared = self.prepare()
        self.assertEqual(prepared['radius'], .018)
        result = self.collect(fixture.log_text(BASELINE+.016))
        self.assertEqual(result['pbe_gate'], 'pass')
        self.assertAlmostEqual(result['energy_delta_ev_per_c'], .008)

    def test_system_runtime_directory_alias_is_resolved_and_hash_pinned(self):
        system = self.root/'system'
        system.mkdir()
        library = system/'libpmi.so.0.0.0'
        library.write_bytes(self.mpi.read_bytes())
        (system/'libpmi.so').symlink_to(library.name)
        alias = self.root/'runtime-alias'
        alias.symlink_to(system, target_is_directory=True)
        prepared = self.prepare(mpi_library=alias/'libpmi.so')
        self.assertEqual(prepared['mpi_library'], str(library))
        library.write_bytes(b'changed library')
        with self.assertRaises(ValueError):
            self.collect()

    def test_collection_uses_original_baseline_and_reports_center_shift(self):
        self.prepare()
        self.admit.reset_mock()
        self.verify_candidate.reset_mock()
        energy = BASELINE + .016
        result = self.collect(fixture.log_text(energy))
        self.assertEqual(result["scope"], SCOPE)
        self.assertEqual(result["pbe_gate"], "pass")
        self.assertEqual(result["scf_log_gate"], "pass")
        self.assertEqual(result["band_counts"], dict(occupied=4, total=44))
        self.assertEqual(result["scheduler_gate"], "pending_external_validation")
        self.assertEqual(result["physical_release_gate"], "hold")
        self.assertAlmostEqual(result["energy_delta_ev_per_c"], (energy-BASELINE)/2)
        self.assertAlmostEqual(result["delta_from_center_ev_per_c"], (energy-self.center_energy)/2)
        self.assertEqual(result["candidate_sha256"], fixture.digest(self.candidate_root / "CANDIDATE.json"))
        self.assertEqual(result["preparation_sha256"], fixture.digest(self.output / PREPARATION))
        self.admit.assert_called_with(self.center, self.center_sha, self.acceptance_sha)
        self.verify_candidate.assert_called_with(self.candidate_root)
        self.assertEqual(json.loads((self.output / COLLECTION).read_text()), result)
        with self.assertRaises(FileExistsError):
            self.collect()

    def test_pbe_outside_original_bound_fails_even_when_close_to_center(self):
        self.prepare()
        result = self.collect(fixture.log_text(BASELINE+.0201))
        self.assertEqual(result["pbe_gate"], "fail")
        self.assertLess(abs(result["delta_from_center_ev_per_c"]), .01)
        self.assertGreater(abs(result["energy_delta_ev_per_c"]), .01)

    def test_negative_overbound_also_fails(self):
        self.prepare()
        self.assertEqual(self.collect(fixture.log_text(BASELINE-.0201))["pbe_gate"], "fail")

    def test_wrong_external_hashes_and_failed_parent_or_candidate_stop_preparation(self):
        for key in ("center_result_sha256", "center_acceptance_sha256",
                    "expected_abacus_sha256", "expected_mpi_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(**{key: "0"*64})
            self.assertFalse(self.output.exists())
        self.admit.side_effect = ValueError("rejected center")
        with self.assertRaisesRegex(ValueError, "rejected center"):
            self.prepare()
        self.admit.side_effect = None
        self.verify_candidate.side_effect = ValueError("rejected gradient candidate")
        with self.assertRaisesRegex(ValueError, "rejected gradient candidate"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_candidate_scope_center_identity_radius_and_names_are_not_counterfeited(self):
        changes = [dict(scope="actual_pbe_tangent_candidate"), dict(status="success"),
            dict(direction_name="actual_pbe_tangent"), dict(radius=True), dict(radius=.01),
            dict(physical_release_gate="pass"), dict(center_stage=str(self.gradient_stage)),
            dict(coefficient_filename="../COEFFICIENTS.txt"), dict(orbital_filename="C_original.orb")]
        changes += [{k: "0"*64} for k in ("center_result_sha256", "center_acceptance_sha256",
                    "center_coefficient_sha256", "center_orbital_sha256",
                    "coefficient_sha256", "orbital_sha256", "gradient_step_sha256")]
        for change in changes:
            self.write_candidate(**change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_symlinks_duplicate_json_and_oversized_candidate_evidence_are_rejected(self):
        for path in (self.candidate_root / n for n in
                     ("CANDIDATE.json", "COEFFICIENTS.txt", "C_3s3p2d_ec_step.orb", "GRADIENT_STEP.json")):
            saved = path.read_bytes()
            target = self.root / "alias"
            target.write_bytes(saved)
            path.unlink()
            path.symlink_to(target)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.prepare()
            path.unlink()
            path.write_bytes(saved)
            target.unlink()
        path = self.candidate_root / "CANDIDATE.json"
        path.write_bytes(path.read_bytes().rstrip()[:-1] + b', "radius": 0.02}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.prepare()
        self.write_candidate()
        path = self.candidate_root / "COEFFICIENTS.txt"
        with path.open("wb") as stream:
            stream.truncate(8*1024*1024+1)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_existing_or_input_contained_output_is_never_modified(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.output.rmdir()
        for root in (self.center, self.candidate_root, self.gradient_stage):
            with self.subTest(root=root), self.assertRaises(ValueError):
                self.prepare(output_root=root / "pbe")
            self.assertFalse((root / "pbe").exists())

    def test_collect_rejects_every_tampered_staged_input_and_evidence(self):
        self.prepare()
        prepared = json.loads((self.output / PREPARATION).read_text())
        for name in list(prepared["prepared_input_sha256"]) + list(prepared["evidence_sha256"]):
            path = self.output / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"changed")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.collect()
            self.assertFalse((self.output / COLLECTION).exists())
            path.write_bytes(saved)
        with self.assertRaises(ValueError):
            self.collect(digest="0"*64)

    def test_collect_revalidates_external_parent_candidate_and_runtime(self):
        self.prepare()
        for path in (self.slot / "INPUT", self.candidate_root / "COEFFICIENTS.txt", self.binary, self.mpi):
            saved = path.read_bytes()
            path.write_bytes(saved+b"mutation")
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.collect()
            path.write_bytes(saved)
        self.admit.side_effect = ValueError("parent no longer accepted")
        with self.assertRaisesRegex(ValueError, "parent no longer accepted"):
            self.collect()
        self.admit.side_effect = None
        self.verify_candidate.side_effect = ValueError("gradient proof changed")
        with self.assertRaisesRegex(ValueError, "gradient proof changed"):
            self.collect()
        self.assertFalse((self.output / COLLECTION).exists())

    def test_collect_rejects_failed_ambiguous_nonfinite_and_wrong_band_logs(self):
        self.prepare()
        texts = [fixture.log_text().replace("#SCF IS CONVERGED#", "SCF NOT CONVERGED"),
                 fixture.log_text(counts=False), fixture.log_text().replace("states = 4", "states = 3"),
                 fixture.log_text().replace("= 44", "= 43"), fixture.log_text(float("nan")),
                 fixture.log_text()+"\n!FINAL_ETOT_IS -309.8 eV\n"]
        for text in texts:
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                self.collect(text)
            self.assertFalse((self.output / COLLECTION).exists())

    def test_rehashed_preparation_cannot_change_scope_baseline_inputs_or_runtime(self):
        self.prepare()
        path = self.output / PREPARATION
        original = json.loads(path.read_text())
        for change in (dict(scope="actual_pbe_tangent_candidate"), dict(baseline_energy_ev=self.center_energy),
                       dict(tolerance_ev_per_c=.1), dict(candidate_log_relative_path="../outside.log"),
                       dict(physical_release_gate="pass"), dict(candidate_sha256="0"*64),
                       dict(prepared_input_sha256={"../outside": "0"*64})):
            save(path, dict(original, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.collect()
            self.assertFalse((self.output / COLLECTION).exists())

    def test_collection_rejects_symlink_log_without_touching_target(self):
        self.prepare()
        target = self.root / "external.log"
        target.write_text(fixture.log_text())
        path = self.output / LOG
        path.parent.mkdir()
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(target.read_text(), fixture.log_text())


if __name__ == "__main__":
    unittest.main()
