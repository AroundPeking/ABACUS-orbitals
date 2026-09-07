"""Small-evidence admission tests; numerical reconstruction is a lazy boundary."""

import copy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_c_combined_step_gradient_refresh import record


WF = Path(__file__).resolve().parents[1] / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
sys.path.insert(0, str(WF))
RESULT = "result/EC_GRADIENT_REFRESH.json"
ACCEPTANCE = "GRADIENT_ACCEPTANCE.json"
NU = [3, 3, 2, 0, 0]
PROVENANCE_KEYS = (
    "status", "scope", "source_commit", "deployment_sha256", "accepted_result_sha256",
    "accepted_acceptance_sha256", "accepted_source_commit", "coefficient_sha256",
    "freeze_sha256", "active_cache_index_sha256", "occupied_capture_floor", "center_reproduction",
    "cache_loads", "rpa_evaluations", "gradient_mode", "backward_passes", "actual_scf_count",
    "optimizer_steps", "candidate_count", "coefficient_update", "exported_candidate",
    "physical_release_gate", "ordinary_sos_qavg", "gw")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True) + "\n")


class Array(list):
    def tolist(self):
        return list(self)


class EcGradientAdmissionTest(unittest.TestCase):
    def patch(self, target, name, **kwargs):
        patcher = mock.patch.object(target, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("check_c_ec_gradient_step"),
                             "strict Ec gradient sidecar missing")
        self.module = importlib.import_module("check_c_ec_gradient_step")
        temporary = tempfile.TemporaryDirectory(prefix="c-ec-step-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage = self.root / "gradient"
        self.stage.mkdir()
        self.center_stage = self.root / "combined"
        self.center_stage.mkdir()
        paths = {}
        for key, name in (("coefficient_path", "COEFFICIENTS.txt"), ("orbital_path", "C.orb"),
                          ("original_coefficient_path", "ORIGINAL.txt"), ("freeze_path", "FREEZE.json"),
                          ("active_cache_index_path", "CACHE.json")):
            path = self.center_stage / name
            path.write_text("{}" if path.suffix == ".json" else name)
            paths[key] = path
        for name in ("DIRECTIONS.json", "CALIBRATION.json"):
            save(self.center_stage / name, {})
        previous_path = self.center_stage / "INTERPOLATED_COEFFICIENTS.txt"
        previous_path.write_text("previous coefficients")
        self.coefficients = {"C": [Array([[float(i == j) for j in range(n)] for i in range(31)]) for n in NU]}
        self.report = dict(coordinate_metric="Euclidean_coefficient_signed_QR_frame",
            projection="G_horizontal=G-C(C^T G)",
            per_radial_scope="saved_frame_dependent_not_rotation_invariant",
            raw_gradient_norm=1., horizontal_gradient_norm=1., channels=[])
        radials = []
        for l, n in enumerate(NU):
            values = [[float(l == 0 and i == 3 and j == 0) for j in range(n)] for i in range(31)]
            self.report["channels"].append(dict(element="C", l=l, shape=[31, n],
                raw_gradient=values, horizontal_gradient=values, raw_norm=float(l == 0),
                horizontal_norm=float(l == 0), span_component_norm=0., horizontal_residual_norm=0.,
                radials=[dict(zeta=z+1, raw_norm=float(l == 0 and z == 0),
                              horizontal_norm=float(l == 0 and z == 0)) for z in range(n)]))
            radials.extend(dict(element="C", l=l, zeta=z+1, gradient_norm=float(l == 0 and z == 0)) for z in range(n))
        self.transport = dict(gradient_norm=1., radials=radials, physical_release_gate="hold",
                              actual_pbe_direction_derivative="unmeasured")
        self.expected = record()
        self.center = dict(paths, occupied_capture_floor=.9998999,
            candidate=dict(radius=.02, mixing_ratio=.1),
            quarter=dict(initial=dict(minimum_occupied_capture=.9999999), coefficient_sha256=digest(previous_path),
                         training_weights=dict(pi_weight=1., trace_log_weight=1., energy_weight=1.)),
            result=dict(source_commit="b"*40, candidate=self.expected, center_result_sha256="3"*64,
                directions_sha256=digest(self.center_stage / "DIRECTIONS.json"),
                calibration_sha256=digest(self.center_stage / "CALIBRATION.json"),
                **{identity: digest(paths[key]) for identity, key in (
                    ("coefficient_sha256", "coefficient_path"), ("orbital_sha256", "orbital_path"),
                    ("freeze_sha256", "freeze_path"), ("active_cache_index_sha256", "active_cache_index_path"))}))
        self.gradient = {k: copy.deepcopy(self.expected[k]) for k in
                         ("loss", "rpa", "minimum_occupied_capture", "maximum_overlap_condition")}
        self.gradient.update(status="success", scope="accepted_combined_step_ec_gradient_refresh",
            source_commit="a"*40, accepted_source_commit="b"*40,
            accepted_stage=str(self.center_stage), accepted_result_sha256="1"*64, accepted_acceptance_sha256="2"*64,
            original_quarter_result_sha256="3"*64,
            energy_gradient=copy.deepcopy(self.report), transported_descent=copy.deepcopy(self.transport),
            occupied_capture_floor=self.center["occupied_capture_floor"], initial_frozen_band_screen=self.expected["coefficient_guard"],
            center_reproduction="pass", old_direction_reconstruction="exact_accepted_coefficients",
            gradient_mode="energy_only", backward_passes=1, cache_loads=1, rpa_evaluations=1,
            actual_scf_count=0, optimizer_steps=0, candidate_count=0, coefficient_update="none", exported_candidate="none",
            actual_pbe_direction_derivatives="unmeasured", energy_gradient_units="Ha_per_cell_per_unit_coefficient",
            nu=NU, fixed_nu=[0]*5, ao_per_C=22, physical_release_gate="hold", ordinary_sos_qavg="pending", gw="pending",
            cache_load_seconds=2., forward_seconds=3., forward_and_backward_seconds=5., total_seconds=8., peak_rss_kib=100)
        for key in ("coefficient_sha256", "orbital_sha256", "freeze_sha256", "active_cache_index_sha256",
                    "directions_sha256", "calibration_sha256"):
            self.gradient[key] = self.center["result"][key]
        import refresh_c_combined_step_gradient as refresh
        self.source = self.stage / "source"
        self.source_files = {name: ("# archived " + name + "\n").encode() for name in refresh.REQUIRED_SOURCE_FILES}
        self.source_files["SIAB/README.md"] = b"archived non-Python source\n"
        for name, data in self.source_files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.deployment = dict(source_directory=str(self.source), source_commit="a"*40,
            archive_path=str(self.stage / "source.tar.gz"), files={name: hashlib.sha256(data).hexdigest()
                                                                 for name, data in self.source_files.items()})
        self.write_archive()
        self.acceptance = dict(status="success", scope="gradient_acceptance_without_replay_not_optimized_basis",
            job_id="21895652", scheduler="21895652|COMPLETED|0:0|00:06:19|\n21895652.batch|COMPLETED|0:0|00:06:19|1K\n21895652.extern|COMPLETED|0:0|00:06:20|1K\n",
            verification_script_sha256="e"*64, center_energy_ha=self.expected["rpa"]["candidate_energy_ha"],
            loss=self.expected["loss"], gradient_norm=1., transported_direction=copy.deepcopy(self.transport),
            radial_gradient_squared_shares={str(r["l"])+":"+str(r["zeta"]):r["gradient_norm"]**2 for r in radials},
            timing={key:self.gradient[key] for key in ("cache_load_seconds", "forward_seconds", "forward_and_backward_seconds", "total_seconds", "peak_rss_kib")},
            new_candidate_count=0, new_scf_count=0, physical_release_gate="hold")
        (self.stage / "STATUS").write_text("success\n")
        (self.stage / "slurm-21895652.err").write_bytes(b"")
        self.repin()
        self.admit = self.patch(self.module, "load_accepted_combined_step", return_value=self.center)
        self.runtime = SimpleNamespace(
            torch=SimpleNamespace(set_num_threads=mock.Mock(), no_grad=mock.MagicMock(), equal=lambda a,b:a==b,
                                  tensor=lambda a,**kw:Array(a), float64="float64"),
            read_coefficients=mock.Mock(return_value=self.coefficients),
            radial_gradient_report=mock.Mock(return_value=self.report),
            same_report=lambda actual, expected: actual == expected,
            transported_descent_report=mock.Mock(return_value=self.transport),
            combine_pbe_tangent=mock.Mock(return_value=dict(coefficients=self.coefficients, direction=self.coefficients,
                scope="actual_pbe_tangent_candidate", direction_name="actual_pbe_tangent", radius=.02, mixing_ratio=.1)))
        self.runtime_call = self.patch(self.module, "_runtime", return_value=self.runtime)
        patcher = mock.patch.dict(sys.modules, {"torch": None, "numpy": None})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.patch(__import__("subprocess"), "run", side_effect=AssertionError("no subprocess"))
        self.patch(__import__("subprocess"), "check_output", side_effect=AssertionError("no scheduler"))

    def write_archive(self, changes=None, extra=None):
        files = dict(self.source_files, **(changes or {}))
        if extra:
            files.update(extra)
        with tarfile.open(self.deployment["archive_path"], "w:gz") as archive:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        self.deployment["archive_sha256"] = digest(Path(self.deployment["archive_path"]))
        save(self.stage / "DEPLOYMENT.json", self.deployment)
        self.gradient["deployment_sha256"] = digest(self.stage / "DEPLOYMENT.json")

    def repin(self):
        save(self.stage / RESULT, self.gradient)
        self.result_sha = digest(self.stage / RESULT)
        provenance = {k:self.gradient[k] for k in PROVENANCE_KEYS}
        provenance.update(diagnostic_filename=RESULT, diagnostic_sha256=self.result_sha)
        save(self.stage / "PROVENANCE.json", provenance)
        self.acceptance.update(result_sha256=self.result_sha, deployment_sha256=self.gradient["deployment_sha256"])
        save(self.stage / ACCEPTANCE, self.acceptance)
        self.acceptance_sha = digest(self.stage / ACCEPTANCE)

    def load(self, **changes):
        options = dict(stage=self.stage, result_sha256=self.result_sha, acceptance_sha256=self.acceptance_sha)
        return self.module.load_accepted_ec_gradient(**dict(options, **changes))

    def test_loader_preserves_archive_and_uses_explicit_center_pins(self):
        before = {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        loaded = self.load()
        self.assertEqual(loaded["gradient"], self.gradient)
        self.assertEqual(loaded["stage"], self.stage)
        self.assertIs(loaded["center"], self.center)
        self.admit.assert_called_once_with(self.center_stage, "1"*64, "2"*64)
        self.runtime.torch.set_num_threads.assert_called_with(28)
        self.assertEqual(before, {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_both_external_pins_are_required(self):
        for key in ("result_sha256", "acceptance_sha256"):
            for value in (None, "", "0"*64, "A"*64):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(**{key:value})

    def test_status_provenance_and_all_small_artifacts_reject_tampering(self):
        for path in (self.stage / "STATUS", self.stage / "PROVENANCE.json", self.stage / RESULT,
                     self.stage / ACCEPTANCE, self.stage / "DEPLOYMENT.json", self.stage / "source.tar.gz"):
            original = path.read_bytes()
            path.write_bytes(original + b"corruption")
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.load()
            path.write_bytes(original)

    def test_exact_three_scheduler_identities_and_success_are_required(self):
        original = self.acceptance["scheduler"]
        for value in ("", original.replace("COMPLETED", "FAILED", 1), original.replace("0:0", "1:0", 1),
                      original + original.splitlines()[0] + "\n", original.replace(".extern", ".0")):
            self.acceptance["scheduler"] = value
            self.repin()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load()

    def test_no_physics_scope_counts_and_original_bindings_can_be_relaxed(self):
        original = copy.deepcopy(self.gradient)
        mutations = dict(actual_scf_count=True, backward_passes=2, cache_loads=2, candidate_count=1,
            optimizer_steps=1, physical_release_gate="pass", scope="accepted_ec_gradient_step_candidate",
            ordinary_sos_qavg="pass", gw="pass", coefficient_update="updated", exported_candidate="C.orb",
            occupied_capture_floor=.9, coefficient_sha256="0"*64, freeze_sha256="0"*64,
            active_cache_index_sha256="0"*64, accepted_source_commit="0"*40, original_quarter_result_sha256="0"*64)
        for key, value in mutations.items():
            self.gradient = dict(copy.deepcopy(original), **{key:value})
            self.repin()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()

    def test_archived_member_bytes_must_match_manifest_even_after_repin(self):
        name = next(iter(self.source_files))
        self.write_archive(changes={name:b"different archived implementation\n"})
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_archived_member_set_is_exact_including_non_python_files(self):
        self.write_archive(extra={"SIAB/unlisted.txt":b"not declared"})
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_source_file_and_manifest_omissions_are_not_hidden_by_archive_pin(self):
        path = self.source / "SIAB/README.md"
        path.write_text("mutated deployed file")
        with self.assertRaises(ValueError):
            self.load()
        path.write_bytes(self.source_files["SIAB/README.md"])
        del self.deployment["files"]["SIAB/README.md"]
        save(self.stage / "DEPLOYMENT.json", self.deployment)
        self.gradient["deployment_sha256"] = digest(self.stage / "DEPLOYMENT.json")
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_reproduction_checks_all_error_components_not_only_weighted_sum(self):
        self.gradient["rpa"]["pi_relative_squared_error"] += .005
        self.gradient["rpa"]["trace_log_relative_squared_error"] -= .005
        self.repin()
        with self.assertRaisesRegex(ValueError, "reproduced pi_relative"):
            self.load()

    def test_gradient_matrix_and_transport_are_reconstructed_not_trusted(self):
        original = copy.deepcopy(self.gradient)
        self.gradient["energy_gradient"]["channels"][0]["horizontal_gradient"][3][0] = 2.
        self.repin()
        with self.assertRaises(ValueError):
            self.load()
        self.gradient = original
        self.gradient["transported_descent"]["gradient_norm"] = 2.
        self.acceptance["transported_direction"] = copy.deepcopy(self.gradient["transported_descent"])
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_acceptance_summary_numbers_and_timing_cannot_contradict_gradient(self):
        original = copy.deepcopy(self.acceptance)
        for key, value in (("gradient_norm", 0.), ("loss", .01), ("center_energy_ha", -.1),
                           ("physical_release_gate", "pass"), ("new_scf_count", 1),
                           ("radial_gradient_squared_shares", {}), ("timing", {})):
            self.acceptance = dict(copy.deepcopy(original), **{key:value})
            self.repin()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()

    def test_duplicate_nonfinite_and_overflow_json_reject_before_runtime(self):
        path = self.stage / RESULT
        for content in (b'{"status":"success","status":"failed"}', b'{"x":NaN}',
                        b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e999}', b'{bad', b'[]'):
            path.write_bytes(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.load(result_sha256=digest(path))
        self.runtime_call.assert_not_called()

    def candidate_fixture(self):
        self.candidate_root = self.root / "candidate"
        self.candidate_root.mkdir()
        self.proposed = copy.deepcopy(self.coefficients)
        self.proposed["C"][0][3][0] = -.02
        self.direction = {"C": [Array([[-float(l == 0 and i == 3 and z == 0) for z in range(n)]
                                       for i in range(31)]) for l, n in enumerate(NU)]}
        self.proposal = dict(scope="accepted_ec_gradient_step_candidate", direction_name="negative_horizontal_ec_gradient",
            radius=.02, coefficients=self.proposed, direction=self.direction, predicted_ec_delta_ha_per_cell=-.02,
            actual_pbe_direction_derivative="unmeasured", finite_step_safety="unmeasured", physical_release_gate="hold",
            actual_pbe_gate="pending", galerkin_energy="unmeasured")
        self.step = {key:copy.deepcopy(value) for key,value in self.proposal.items() if key != "coefficients"}
        save(self.candidate_root / "GRADIENT_STEP.json", self.step)
        save(self.candidate_root / "COEFFICIENTS.txt", self.proposed)
        (self.candidate_root / "C_3s3p2d_ec_step.orb").write_bytes(b"exact rendered orbital\n")
        self.candidate = {key:value for key,value in self.proposal.items() if key not in ("coefficients", "direction")}
        self.candidate.update(status="prepared", source_commit="c"*40, nu=NU, fixed_nu=[0]*5, ao_per_C=22,
            center_stage=str(self.center_stage), center_result_sha256="1"*64, center_acceptance_sha256="2"*64,
            center_coefficient_sha256=self.center["result"]["coefficient_sha256"],
            center_orbital_sha256=self.center["result"]["orbital_sha256"], gradient_stage=str(self.stage),
            gradient_result_sha256=self.result_sha, gradient_acceptance_sha256=self.acceptance_sha,
            coefficient_filename="COEFFICIENTS.txt", orbital_filename="C_3s3p2d_ec_step.orb",
            cheap_gate=dict(gate=True, minimum_occupied_capture=.999995, occupied_capture_floor=.9998999,
                overlap_relative_rank_tolerance=1e-12, overlap_condition_limit=1e12,
                band_screen=copy.deepcopy(self.expected["coefficient_guard"])))
        self.repin_candidate()
        def read(path, **kwargs):
            if Path(path) == self.candidate_root / "COEFFICIENTS.txt":
                value = json.loads(Path(path).read_bytes())
                return {e:[Array(c) for c in channels] for e,channels in value.items()}
            return self.coefficients
        self.runtime.read_coefficients.side_effect = read
        self.runtime.propose_ec_gradient_step = mock.Mock(side_effect=lambda *args,**kwargs:copy.deepcopy(self.proposal))
        self.runtime.write_abacus_orbital = mock.Mock(side_effect=lambda path,*args,**kwargs:
            Path(path).write_bytes(b"exact rendered orbital\n"))

    def repin_candidate(self):
        for name,key in (("COEFFICIENTS.txt", "coefficient_sha256"), ("C_3s3p2d_ec_step.orb", "orbital_sha256"),
                         ("GRADIENT_STEP.json", "gradient_step_sha256")):
            self.candidate[key] = digest(self.candidate_root / name)
        save(self.candidate_root / "CANDIDATE.json", self.candidate)

    def verify_candidate(self):
        function = getattr(self.module, "validate_gradient_step_candidate", None)
        self.assertTrue(callable(function), "strict Ec step candidate verifier missing")
        return function(self.candidate_root)

    def test_candidate_exact_proposal_coefficients_and_orbital_without_writes(self):
        self.candidate_fixture()
        before = {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.verify_candidate(), self.candidate)
        self.runtime.propose_ec_gradient_step.assert_called_once_with(self.coefficients, self.report, radius=.02)
        self.runtime.write_abacus_orbital.assert_called_once()
        self.assertEqual(before, {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_candidate_cannot_claim_old_scope_other_radius_or_measured_physics(self):
        self.candidate_fixture()
        original = copy.deepcopy(self.candidate)
        for key,value in (("scope", "actual_pbe_tangent_candidate"), ("radius", .01), ("radius", True),
                          ("actual_pbe_gate", "pass"), ("galerkin_energy", "measured"),
                          ("physical_release_gate", "pass"), ("finite_step_safety", "pass"),
                          ("actual_pbe_direction_derivative", 0.), ("source_commit", "bad"),
                          ("nu", [3,3,2,False,0]), ("ao_per_C", 21)):
            self.candidate = dict(copy.deepcopy(original), **{key:value})
            self.repin_candidate()
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                self.verify_candidate()

    def test_candidate_center_and_gradient_pins_cannot_be_crossed(self):
        self.candidate_fixture()
        original = copy.deepcopy(self.candidate)
        for key in ("center_result_sha256", "center_acceptance_sha256", "center_coefficient_sha256",
                    "center_orbital_sha256", "gradient_result_sha256", "gradient_acceptance_sha256"):
            self.candidate = dict(original, **{key:"0"*64})
            self.repin_candidate()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify_candidate()

    def test_candidate_cheap_gate_keeps_original_floor_and_limits(self):
        self.candidate_fixture()
        original = copy.deepcopy(self.candidate)
        for key,value in (("gate", 1), ("minimum_occupied_capture", .9), ("occupied_capture_floor", .9),
                          ("overlap_relative_rank_tolerance", 2e-12), ("overlap_condition_limit", 1e13)):
            self.candidate = copy.deepcopy(original)
            self.candidate["cheap_gate"][key] = value
            self.repin_candidate()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify_candidate()

    def test_rehashed_direction_or_proposal_changes_still_reject(self):
        self.candidate_fixture()
        original = copy.deepcopy(self.step)
        for mutation in (lambda s:s["direction"]["C"][0][3].__setitem__(0, -.5),
                         lambda s:s["direction"]["C"].pop(),
                         lambda s:s.update(predicted_ec_delta_ha_per_cell=0.),
                         lambda s:s.update(physical_release_gate="pass"),
                         lambda s:s.update(unapproved_extra=True)):
            self.step = copy.deepcopy(original)
            mutation(self.step)
            save(self.candidate_root / "GRADIENT_STEP.json", self.step)
            self.repin_candidate()
            with self.subTest(step=self.step.keys()), self.assertRaises(ValueError):
                self.verify_candidate()

    def test_rehashed_coefficients_and_orbital_must_reconstruct_exactly(self):
        self.candidate_fixture()
        changed = copy.deepcopy(self.proposed)
        changed["C"][0][3][0] += 1e-15
        save(self.candidate_root / "COEFFICIENTS.txt", changed)
        self.repin_candidate()
        with self.assertRaises(ValueError):
            self.verify_candidate()
        save(self.candidate_root / "COEFFICIENTS.txt", self.proposed)
        (self.candidate_root / "C_3s3p2d_ec_step.orb").write_bytes(b"changed rendering\n")
        self.repin_candidate()
        with self.assertRaises(ValueError):
            self.verify_candidate()

    def test_candidate_nonfinite_step_and_symlink_artifact_reject(self):
        self.candidate_fixture()
        path = self.candidate_root / "GRADIENT_STEP.json"
        original = path.read_bytes()
        for content in (b'{"direction":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}'):
            path.write_bytes(content)
            self.repin_candidate()
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.verify_candidate()
        path.write_bytes(original)
        target = self.candidate_root / "C_3s3p2d_ec_step.orb"
        target.unlink()
        target.symlink_to(self.center["orbital_path"])
        self.repin_candidate()
        with self.assertRaises(ValueError):
            self.verify_candidate()


class RealAcceptedGradientSmokeTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("C_ACCEPTED_EC_GRADIENT_STAGE"), "remote small-evidence smoke is opt-in")
    def test_frozen_21895652_without_cache_or_response(self):
        import check_c_ec_gradient_step as validator
        loaded = validator.load_accepted_ec_gradient(os.environ["C_ACCEPTED_EC_GRADIENT_STAGE"],
            "b3e46a26e07a76a0931e8418c185e613bdef346fa31e52d6fb33ed8ba2bd02b5",
            "3991384900936811cfb9f5708446e2c344698bb5e455078c7a792d31e40f4c3e")
        self.assertEqual(loaded["gradient"]["actual_scf_count"], 0)
        self.assertEqual(loaded["stage"], Path(os.environ["C_ACCEPTED_EC_GRADIENT_STAGE"]).resolve())


if __name__ == "__main__":
    unittest.main()
