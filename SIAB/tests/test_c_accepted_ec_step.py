"""Stdlib admission tests; only remote tensor/provenance boundaries are mocked."""

import copy
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_c_combined_step_gradient_refresh import record, HA_EV


WF = Path(__file__).resolve().parents[1] / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
sys.path.insert(0, str(WF))
RESULT = "result/RESULT.json"
ACCEPTANCE = "STEP_MEASUREMENT_ACCEPTANCE.json"
SOURCE_FILES = tuple("SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/" + n
    for n in ("check_c_ec_gradient_step.py", "check_c_ec_gradient_step_pbe.py",
              "run_c_ec_gradient_step.py", "run_c_ec_gradient_step.slurm")) + (
    "SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_ec_gradient_step.py",)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True) + "\n")


def energy_record(energy):
    value = record()
    rpa = value["rpa"]
    rpa["candidate_energy_ha"] = energy
    rpa["energy_relative_squared_error"] = ((energy-rpa["reference_energy_ha"])/rpa["reference_energy_ha"])**2
    value["loss"] = sum(rpa[n+"_relative_squared_error"] for n in ("pi", "trace_log", "energy"))
    for row in rpa["per_q"]:
        row["candidate_contributions_ha"] = [energy*row["q_weight"]/12]*12
    value["rpa_correlation_energy_ev_per_cell"] = energy*HA_EV
    value["rpa_correlation_energy_ev_per_c"] = energy*HA_EV/2
    return value


class AcceptedEcStepTest(unittest.TestCase):
    def patch(self, target, name, **options):
        patcher = mock.patch.object(target, name, **options)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("check_c_accepted_ec_step"),
                             "independent accepted Ec-step reader missing")
        self.module = importlib.import_module("check_c_accepted_ec_step")
        self.admission = importlib.import_module("check_c_ec_gradient_step")
        self.pbe = importlib.import_module("check_c_ec_gradient_step_pbe")
        self.refresh = importlib.import_module("refresh_c_combined_step_gradient")
        temporary = tempfile.TemporaryDirectory(prefix="c-accepted-ec-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage = self.root / "step"
        self.stage.mkdir()
        center_stage = self.root / "center"
        center_stage.mkdir()
        gradient_stage = self.root / "gradient"
        gradient_stage.mkdir()
        paths = {}
        for key, name in (("coefficient_path", "COEFFICIENTS.txt"), ("orbital_path", "C.orb"),
                          ("original_coefficient_path", "ORIGINAL.txt"), ("freeze_path", "FREEZE.json"),
                          ("active_cache_index_path", "CACHE.json")):
            path = center_stage / name
            path.write_text("{}" if path.suffix == ".json" else name)
            paths[key] = path
        self.center = dict(paths, occupied_capture_floor=.9999999-1e-4,
            quarter=dict(initial=dict(minimum_occupied_capture=.9999999),
                initial_sha256=digest(paths["original_coefficient_path"]),
                training_weights=dict(pi_weight=1., trace_log_weight=1., energy_weight=1.),
                configuration=dict(frequency_batch_size=3)),
            result=dict(candidate=record(), source_commit="b"*40, **{identity:digest(paths[key])
                for key, identity in (("coefficient_path", "coefficient_sha256"),
                                     ("orbital_path", "orbital_sha256"), ("freeze_path", "freeze_sha256"),
                                     ("active_cache_index_path", "active_cache_index_sha256"))}))
        gradient = dict(accepted_result_sha256="1"*64, accepted_acceptance_sha256="2"*64,
            energy_gradient=dict(horizontal_gradient_norm=1.))
        self.loaded_gradient = dict(gradient=gradient, center=self.center, center_stage=center_stage)
        self.proposal = dict(status="prepared", scope="accepted_ec_gradient_step_candidate", source_commit="a"*40,
            center_stage=str(center_stage), center_result_sha256="1"*64, center_acceptance_sha256="2"*64,
            center_coefficient_sha256=self.center["result"]["coefficient_sha256"],
            center_orbital_sha256=self.center["result"]["orbital_sha256"],
            gradient_stage=str(gradient_stage), gradient_result_sha256="3"*64, gradient_acceptance_sha256="4"*64,
            direction_name="negative_horizontal_ec_gradient", radius=.02,
            coefficient_filename="COEFFICIENTS.txt", orbital_filename="C_3s3p2d_ec_step.orb",
            actual_pbe_direction_derivative="unmeasured", finite_step_safety="unmeasured",
            actual_pbe_gate="pending", galerkin_energy="unmeasured", physical_release_gate="hold",
            predicted_ec_delta_ha_per_cell=-.02, nu=[3, 3, 2, 0, 0], fixed_nu=[0]*5, ao_per_C=22,
            cheap_gate=dict(gate=True, occupied_capture_floor=self.center["occupied_capture_floor"],
                minimum_occupied_capture=record()["minimum_occupied_capture"],
                overlap_relative_rank_tolerance=1e-12, overlap_condition_limit=1e12,
                band_screen=record()["coefficient_guard"]))
        self.candidate_root = self.stage / "result/candidate"
        self.candidate_root.mkdir(parents=True)
        for name, key in self.pbe._FILES.items():
            (self.candidate_root / name).write_text("{}" if name.endswith(".json") else name)
            self.proposal[key] = digest(self.candidate_root / name)
        save(self.candidate_root / "CANDIDATE.json", self.proposal)
        self.result = dict(status="success", scope="one_accepted_ec_gradient_step", source_commit="a"*40,
            candidate_gate="improved_frozen_body", actual_scf_count=1, rpa_evaluations=2,
            cache_loads=1, backward_passes=0, optimizer_steps=0, physical_release_gate="hold",
            ordinary_sos_qavg="pending", gw="pending", actual_pbe_direction_derivative="unmeasured",
            nu=[3, 3, 2, 0, 0], fixed_nu=[0]*5, ao_per_C=22,
            occupied_capture_floor=self.center["occupied_capture_floor"],
            center=record(), candidate=energy_record(-.431), total_seconds=10., cache_load_seconds=1.,
            scf_seconds=2., response_seconds=3., peak_rss_kib=100,
            predicted_ec_delta_ev_per_c=-.02*HA_EV/2)
        for key in ("radius", "center_stage", "center_result_sha256", "center_acceptance_sha256",
                    "gradient_stage", "gradient_result_sha256", "gradient_acceptance_sha256",
                    "coefficient_sha256", "orbital_sha256", "gradient_step_sha256"):
            self.result[key] = self.proposal[key]
        for key in ("freeze_sha256", "active_cache_index_sha256"):
            self.result[key] = self.center["result"][key]
        self.slot = self.stage / "result/pbe"
        self.slot.mkdir()
        self.log_name = "OUT.C/running_scf.log"
        log = self.slot / self.log_name
        log.parent.mkdir()
        log.write_text("Occupied electronic states = 4\nNBANDS = 44\n#SCF IS CONVERGED#\n"
                       "!FINAL_ETOT_IS -309.8405190693191 eV\n")
        self.prepared = dict(self.pbe._COMMON, format_version=1, status="prepared",
            center_stage=str(center_stage), candidate_root=str(self.candidate_root),
            candidate_sha256=digest(self.candidate_root / "CANDIDATE.json"),
            center_preparation_sha256="5"*64, baseline_energy_ev=self.pbe.ORIGINAL_ENERGY_EV,
            center_energy_ev=-309.8544130538425, baseline_log_sha256="6"*64, center_log_sha256="7"*64,
            candidate_log_relative_path=self.log_name, abacus_binary="/remote/abacus", abacus_sha256="8"*64,
            mpi_library="/remote/libpmi.so", mpi_sha256="9"*64,
            **{key:self.proposal[key] for key in self.pbe._IDENTITY})
        self.prepared_files = {n:n.encode() for n in ("INPUT", "STRU", "KPT", "C.upf", "C.orb")}
        self.prepared_files.update({".provenance/"+p.name:p.read_bytes() for p in self.candidate_root.iterdir()})
        self.prepared_files.update({".provenance/center_"+n:b"{}" for n in (
            "RESULT.json", "ACCEPTANCE.json", "PBE_PREPARED.json", "running_scf.log")})
        for name, data in self.prepared_files.items():
            path = self.slot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for key, evidence in (("evidence_sha256", True), ("prepared_input_sha256", False)):
            self.prepared[key] = {n:hashlib.sha256(data).hexdigest() for n, data in self.prepared_files.items()
                                  if n.startswith(".provenance/") == evidence}
        save(self.slot / self.pbe.PREPARATION, self.prepared)
        self.collection = dict(self.pbe._COMMON, status="collected", pbe_gate="pass", scf_log_gate="pass",
            band_count_check="pass", band_counts=dict(occupied=4, total=44),
            comparison="absolute_candidate_minus_original_per_C_lte_tolerance",
            preparation_sha256=digest(self.slot / self.pbe.PREPARATION),
            candidate_log_path=str(log), candidate_log_sha256=digest(log), candidate_energy_ev=-309.8405190693191)
        for key in self.pbe._IDENTITY + ("candidate_sha256", "center_stage", "center_preparation_sha256",
                "baseline_energy_ev", "center_energy_ev", "baseline_log_sha256", "center_log_sha256",
                "abacus_binary", "abacus_sha256", "mpi_library", "mpi_sha256"):
            self.collection[key] = self.prepared[key]
        self.collection["energy_delta_ev_per_c"] = (self.collection["candidate_energy_ev"]-self.prepared["baseline_energy_ev"])/2
        self.collection["delta_from_center_ev_per_c"] = (self.collection["candidate_energy_ev"]-self.prepared["center_energy_ev"])/2
        self.result["candidate_sha256"] = self.prepared["candidate_sha256"]
        self.result["actual_pbe"] = self.collection
        self.source_files = {name:b"# historical source, not executed\n" for name in
                             self.refresh.REQUIRED_SOURCE_FILES + SOURCE_FILES}
        self.deployment = dict(source_commit="a"*40, source_directory=str(self.stage / "source"),
                               archive_path=str(self.stage / "source.tar.gz"))
        self.archive()
        job = "12345"
        self.acceptance = dict(status="success", scope="ec_gradient_step_measurement_acceptance", job_id=job,
            source_commit="a"*40, candidate_gate="improved_frozen_body", physical_release_gate="hold",
            ordinary_sos_qavg="pending", gw="pending", pbe_gate="pass", verification_script_sha256="f"*64,
            scheduler="".join(job+s+"|COMPLETED|0:0|00:01:00|\n" for s in ("", ".batch", ".extern", ".0")))
        self.execution = dict(status="launching", scheduler_job=job, source_commit="a"*40,
            actual_scf_budget=1, backward_budget=0, delta_st="not_run", librpa="not_run",
            ordinary_sos_qavg="pending", physical_release_gate="hold", server="df_dcu",
            abacus_sha256="8"*64, pmi_sha256="9"*64)
        self.submission = dict(status="submitted_not_accepted", job_id=job, stage=str(self.stage), source_commit="a"*40,
                               source_file_count=len(self.source_files), fingerprint="e"*64)
        (self.stage / "STATUS").write_text("success\n")
        (self.stage / ("slurm-"+job+".err")).write_bytes(b"")
        (self.stage / "ec-step.time").write_text("Exit status: 0\n")
        self.repin()
        self.reconstruct = self.patch(self.admission, "validate_gradient_step_candidate",
                                      side_effect=lambda root:copy.deepcopy(self.proposal))
        self.gradient_loader = self.patch(self.admission, "load_accepted_ec_gradient", return_value=self.loaded_gradient)
        self.snapshot = self.patch(self.pbe, "_snapshot", side_effect=lambda *a, **kw:(self.prepared, self.prepared_files))
        self.patch(self.pbe, "collect_ec_gradient_step_pbe", side_effect=AssertionError("collector writes"))
        self.patch(self.pbe, "prepare_ec_gradient_step_pbe", side_effect=AssertionError("preparation writes"))
        self.patch(__import__("subprocess"), "run", side_effect=AssertionError("no process execution"))
        self.patch(__import__("subprocess"), "check_output", side_effect=AssertionError("no scheduler"))
        self.patch(self.refresh, "_runtime", side_effect=AssertionError("no cache/response runtime"))
        native = mock.patch.dict(sys.modules, {"torch":None, "numpy":None})
        native.start()
        self.addCleanup(native.stop)

    def archive(self):
        with tarfile.open(self.deployment["archive_path"], "w:gz") as archive:
            for name, data in self.source_files.items():
                path = self.stage / "source" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        self.deployment.update(archive_sha256=digest(Path(self.deployment["archive_path"])),
            files={n:hashlib.sha256(data).hexdigest() for n,data in self.source_files.items()})
        save(self.stage / "DEPLOYMENT.json", self.deployment)
        self.result["deployment_sha256"] = digest(self.stage / "DEPLOYMENT.json")

    def repin(self, measurements=True):
        r, a = self.result, self.acceptance
        first, new = r["center"], r["candidate"]
        ec, ref = new["rpa"]["candidate_energy_ha"], new["rpa"]["reference_energy_ha"]
        if measurements:
            r["measured_ec_delta_ev_per_c"] = (ec-first["rpa"]["candidate_energy_ha"])*HA_EV/2
            r["body_error_ev_per_c"] = abs(ec-ref)*HA_EV/2
            r["measured_to_predicted_ec_gain"] = r["measured_ec_delta_ev_per_c"]/r["predicted_ec_delta_ev_per_c"]
            a.update(ec_ha_per_cell=ec, reference_ec_ha_per_cell=ref, body_error_ev_per_c=r["body_error_ev_per_c"],
                improvement_mev_per_c=-r["measured_ec_delta_ev_per_c"]*1000,
                predicted_improvement_mev_per_c=-r["predicted_ec_delta_ev_per_c"]*1000,
                initial_loss=first["loss"], loss=new["loss"], capture=new["minimum_occupied_capture"],
                condition=new["maximum_overlap_condition"], per_q_before=copy.deepcopy(first["rpa"]["per_q"]),
                per_q_after=copy.deepcopy(new["rpa"]["per_q"]),
                errors={n:dict(before=first["rpa"][n+"_relative_squared_error"], after=new["rpa"][n+"_relative_squared_error"])
                        for n in ("pi", "trace_log", "energy")},
                actual_pbe_energy_ev_per_cell=self.collection["candidate_energy_ev"],
                actual_pbe_delta_mev_per_c=self.collection["energy_delta_ev_per_c"]*1000)
            a.update({k:r[k] for k in ("coefficient_sha256", "orbital_sha256", "total_seconds", "cache_load_seconds", "peak_rss_kib")})
        save(self.slot / self.pbe.COLLECTION, self.collection)
        save(self.stage / RESULT, r)
        self.result_sha = digest(self.stage / RESULT)
        provenance = {k:r[k] for k in ("status", "source_commit", "candidate_gate", "actual_scf_count",
                                       "rpa_evaluations", "backward_passes", "physical_release_gate")}
        save(self.stage / "PROVENANCE.json", dict(provenance, result_sha256=self.result_sha))
        a.update(result_sha256=self.result_sha, deployment_sha256=r["deployment_sha256"])
        save(self.stage / ACCEPTANCE, a)
        self.acceptance_sha = digest(self.stage / ACCEPTANCE)
        self.submission.update(deployment_sha256=r["deployment_sha256"], archive_sha256=self.deployment["archive_sha256"])
        save(self.stage / "SUBMISSION.json", self.submission)
        save(self.stage / "EXECUTION_PROVENANCE.json", self.execution)

    def load(self, **changes):
        return self.module.load_accepted_ec_step(**dict(dict(stage=self.stage, result_sha256=self.result_sha,
                                                           acceptance_sha256=self.acceptance_sha), **changes))

    def sync_candidate(self):
        save(self.candidate_root / "CANDIDATE.json", self.proposal)
        sha = digest(self.candidate_root / "CANDIDATE.json")
        for value in (self.result, self.prepared, self.collection):
            value["candidate_sha256"] = sha
        for key in self.pbe._IDENTITY:
            self.prepared[key] = self.collection[key] = self.proposal[key]
        self.prepared_files[".provenance/CANDIDATE.json"] = (self.candidate_root / "CANDIDATE.json").read_bytes()
        (self.slot / ".provenance/CANDIDATE.json").write_bytes(self.prepared_files[".provenance/CANDIDATE.json"])
        self.prepared["evidence_sha256"][".provenance/CANDIDATE.json"] = sha
        save(self.slot / self.pbe.PREPARATION, self.prepared)
        self.collection["preparation_sha256"] = digest(self.slot / self.pbe.PREPARATION)
        self.repin()

    def reduced_fixture(self):
        rejected = self.root / "rejected"
        reduction = dict(rejected_stage=str(rejected), prior_radius=.02, next_radius=.018,
            reduction_factor=.9, automatic_radius_scan=False, gradient_result_sha256=self.result["gradient_result_sha256"])
        for name, key, constant in ((RESULT, "rejected_result_sha256", "REJECTED_RESULT"),
                (ACCEPTANCE, "rejected_acceptance_sha256", "REJECTED_ACCEPTANCE"),
                ("GUARD_REJECTION_DIAGNOSTIC.json", "rejected_diagnostic_sha256", "REJECTED_DIAGNOSTIC")):
            save(rejected / name, {})
            reduction[key] = digest(rejected / name)
            self.patch(self.admission, constant, new=reduction[key])
        self.patch(self.admission, "REJECTED_STAGE", new=rejected)
        self.patch(self.admission, "GRADIENT_RESULT", new=self.result["gradient_result_sha256"])
        for value in (self.result, self.proposal):
            value.update(radius=.018, reduction_evidence=copy.deepcopy(reduction))
        self.proposal["predicted_ec_delta_ha_per_cell"] = -.018
        self.result["predicted_ec_delta_ev_per_c"] = -.018*HA_EV/2
        self.sync_candidate()
        return rejected

    def test_own_scope_paths_read_only_and_common_refresh_contract(self):
        before = {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        loaded = self.load()
        self.assertEqual(loaded["result"], self.result)
        self.assertEqual(loaded["candidate"], self.proposal)
        self.assertEqual(loaded["acceptance"], self.acceptance)
        self.assertIs(loaded["quarter"], self.center["quarter"])
        self.assertEqual(loaded["occupied_capture_floor"], self.center["occupied_capture_floor"])
        self.assertEqual(loaded["coefficient_path"], self.candidate_root / "COEFFICIENTS.txt")
        self.assertEqual(loaded["orbital_path"], self.candidate_root / "C_3s3p2d_ec_step.orb")
        for key in ("original_coefficient_path", "freeze_path", "active_cache_index_path"):
            self.assertEqual(loaded[key], self.center[key])
        self.reconstruct.assert_called_with(self.candidate_root)
        self.gradient_loader.assert_called_with(Path(self.proposal["gradient_stage"]), "3"*64, "4"*64)
        diagnostic = dict(copy.deepcopy(self.result["candidate"]), scope="radial_gradients_at_fixed_candidate_no_optimization",
            gradient_mode="energy_only", backward_passes=1, physical_release_gate="hold", energy_gradient={},
            energy_gradient_units="Ha_per_cell_per_unit_coefficient")
        self.refresh.validate_reproduction(loaded, diagnostic, diagnostic["coefficient_guard"])
        datasets = [SimpleNamespace(selected_iq=q["selected_iq"], q_weight=q["q_weight"], q_count=64,
            kpoints=[None]*64, frequency_ha=SimpleNamespace(numel=lambda:12, tolist=lambda q=q:q["frequency_ha"]))
            for q in self.result["candidate"]["rpa"]["per_q"]]
        self.refresh.validate_dataset_extent(datasets, loaded)
        self.assertEqual(before, {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_external_pins_are_required(self):
        for key in ("result_sha256", "acceptance_sha256"):
            for value in (None, "", "A"*64, "0"*64):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(**{key:value})
        with self.assertRaises(TypeError):
            self.module.load_accepted_ec_step(self.stage)

    def test_old_scopes_rejections_counts_and_identity_reject_after_repin(self):
        original = copy.deepcopy(self.result)
        changes = dict(scope="one_actual_pbe_tangent_step", candidate_gate="rejected_no_improvement",
            status="failed", actual_scf_count=True, rpa_evaluations=1, cache_loads=2,
            backward_passes=1, optimizer_steps=1, physical_release_gate="pass", ordinary_sos_qavg="pass",
            gw="pass", radius=.018, source_commit="b"*40, occupied_capture_floor=.9,
            actual_pbe_direction_derivative="measured", fixed_nu=[0., 0, 0, 0, 0])
        changes.update({k:"0"*64 for k in ("candidate_sha256", "coefficient_sha256", "orbital_sha256",
            "gradient_step_sha256", "freeze_sha256", "active_cache_index_sha256", "center_result_sha256",
            "center_acceptance_sha256", "gradient_result_sha256", "gradient_acceptance_sha256")})
        for key, value in changes.items():
            self.result = dict(copy.deepcopy(original), **{key:value})
            self.repin()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()

    def test_full_reconstruction_and_prepared_failures_propagate(self):
        self.reconstruct.side_effect = ValueError("coefficient reconstruction failed")
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            self.load()
        self.reconstruct.side_effect = lambda root:self.proposal
        self.snapshot.side_effect = ValueError("runtime provenance failed")
        with self.assertRaisesRegex(ValueError, "runtime provenance"):
            self.load()

    def test_scheduler_requires_four_completed_distinct_rows(self):
        original = self.acceptance["scheduler"]
        for value in ("", original.replace(".0|", ".batch|"), original.replace("COMPLETED", "FAILED", 1),
                      original.replace("0:0", "1:0", 1), "\n".join(original.splitlines()[:3]),
                      original+original.splitlines()[0], original.splitlines()):
            self.acceptance["scheduler"] = value
            self.repin()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load()

    def test_repinning_acceptance_does_not_allow_inconsistent_claims(self):
        original = copy.deepcopy(self.acceptance)
        for key, value in dict(scope="read_only_combined_step_acceptance", pbe_gate="fail", status="failed",
                physical_release_gate="pass", ordinary_sos_qavg="pass", gw="pass", source_commit="b"*40,
                candidate_gate="rejected_cheap_guard", verification_script_sha256="bad", loss=.1,
                actual_pbe_delta_mev_per_c=0., coefficient_sha256="0"*64, orbital_sha256="0"*64,
                improvement_mev_per_c=0., capture=.9, condition=1., per_q_after=[], errors={}).items():
            self.acceptance = dict(copy.deepcopy(original), **{key:value})
            self.repin(measurements=False)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()

    def test_execution_submission_provenance_and_status_are_checked(self):
        for name, changes in (("PROVENANCE.json", dict(source_commit="b"*40, actual_scf_count=True)),
                ("EXECUTION_PROVENANCE.json", dict(actual_scf_budget=2, backward_budget=True, scheduler_job="999",
                                                  abacus_sha256="0"*64, pmi_sha256="0"*64)),
                ("SUBMISSION.json", dict(stage="/other", source_file_count=0, job_id="999", archive_sha256="0"*64))):
            path = self.stage / name
            original = json.loads(path.read_text())
            for key, value in changes.items():
                save(path, dict(original, **{key:value}))
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    self.load()
            save(path, original)
        for name, content in (("STATUS", "failed\n"), ("slurm-12345.err", "warning"), ("ec-step.time", "Exit status: 1\n")):
            path = self.stage / name
            original = path.read_bytes()
            path.write_text(content)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load()
            path.write_bytes(original)

    def test_repinned_source_archive_and_manifest_are_semantically_checked(self):
        path = self.stage / "source" / SOURCE_FILES[0]
        path.write_text("changed")
        with self.assertRaises(ValueError):
            self.load()
        self.source_files.pop(SOURCE_FILES[0])
        path.unlink()
        self.archive()
        self.submission["source_file_count"] = len(self.source_files)
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_repinned_pbe_collection_requires_every_prepared_identity_and_measured_value(self):
        original = copy.deepcopy(self.collection)
        changes = dict(scope="actual_pbe_tangent_candidate", tolerance_ev_per_c=.1, atoms_per_cell=True,
            pbe_gate="fail", scf_log_gate="fail", band_count_check="pending", band_counts=dict(occupied=3, total=44),
            energy_delta_ev_per_c=0., delta_from_center_ev_per_c=0., baseline_energy_ev=-300.,
            center_energy_ev=-300., candidate_log_path="/other/log", abacus_sha256="0"*64,
            center_log_sha256="0"*64, gradient_result_sha256="0"*64, radius=.1)
        for key,value in changes.items():
            self.collection = dict(copy.deepcopy(original), **{key:value})
            self.result["actual_pbe"] = self.collection
            self.repin()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()

    def test_rehashed_pbe_log_requires_strict_scf_and_original_ten_mev_bound(self):
        path = self.slot / self.log_name
        original = path.read_text()
        for content in (original.replace("#SCF IS CONVERGED#", "not converged"),
                        original.replace("states = 4", "states = 3"),
                        original+"!FINAL_ETOT_IS -309.8 eV\n", original.replace("-309.8405190693191", "-309.8")):
            path.write_text(content)
            self.collection["candidate_log_sha256"] = digest(path)
            if content.endswith("-309.8 eV\n"):
                self.collection.update(candidate_energy_ev=-309.8,
                    energy_delta_ev_per_c=(-309.8-self.prepared["baseline_energy_ev"])/2,
                    delta_from_center_ev_per_c=(-309.8-self.prepared["center_energy_ev"])/2)
            self.repin()
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.load()

    def test_q_grid_contributions_and_loss_math_reject_even_with_matching_acceptance(self):
        original = copy.deepcopy(self.result)
        changes = [lambda r:r["rpa"].update(complete_q_weight=1), lambda r:r["rpa"]["per_q"].pop(),
            lambda r:r["rpa"]["per_q"][0].update(selected_iq=True),
            lambda r:r["rpa"]["per_q"][0].update(q_weight=.015625+1e-13),
            lambda r:r["rpa"]["per_q"][0]["frequency_ha"].pop(),
            lambda r:r["rpa"]["per_q"][0]["frequency_ha"].__setitem__(0, .5),
            lambda r:r["rpa"]["per_q"][0]["candidate_contributions_ha"].__setitem__(0, .1),
            lambda r:r.update(loss=r["loss"]*.9),
            lambda r:r["rpa"].update(pi_relative_squared_error=.03),
            lambda r:r.update(rpa_correlation_energy_ev_per_c=-4.)]
        for name in ("center", "candidate"):
            for i, change in enumerate(changes):
                self.result = copy.deepcopy(original)
                change(self.result[name])
                self.repin()
                with self.subTest(name=name, mutation=i), self.assertRaises(ValueError):
                    self.load()
        for name, key in (("center", "candidate_contributions_ha"), ("center", "reference_contributions_ha"),
                          ("candidate", "reference_contributions_ha")):
            self.result = copy.deepcopy(original)
            values = self.result[name]["rpa"]["per_q"][0][key]
            values[0] += 1e-5
            values[1] -= 1e-5
            self.repin()
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                self.load()

    def test_parent_reproduction_requires_each_error_and_guard_not_just_totals(self):
        for name in ("pi", "trace_log"):
            key = name+"_relative_squared_error"
            self.center["result"]["candidate"]["rpa"][key] += .001
            other = ("trace_log" if name == "pi" else "pi")+"_relative_squared_error"
            self.center["result"]["candidate"]["rpa"][other] -= .001
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load()
            self.center["result"]["candidate"] = record()
        self.result["center"]["maximum_overlap_condition"] += 10
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_improvement_requires_closer_energy_and_nonincreasing_loss(self):
        for energy in (-.43, -.429):
            self.result["candidate"] = energy_record(energy)
            self.repin()
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                self.load()
        self.result["candidate"] = energy_record(-.431)
        self.result["candidate"]["rpa"]["pi_relative_squared_error"] += .01
        self.result["candidate"]["loss"] += .01
        self.repin()
        with self.assertRaises(ValueError):
            self.load()
        # Unlike the old backoff gate, the measured Ec-step permits equal loss.
        delta = self.result["candidate"]["loss"]-self.result["center"]["loss"]
        self.result["candidate"]["rpa"]["pi_relative_squared_error"] -= delta
        self.result["candidate"]["loss"] = self.result["center"]["loss"]
        self.repin()
        self.load()

    def test_original_floor_cannot_be_replaced_by_a_later_capture_or_lowered(self):
        self.center["occupied_capture_floor"] = self.result["occupied_capture_floor"] = .9
        self.proposal["cheap_gate"]["occupied_capture_floor"] = .9
        save(self.candidate_root / "CANDIDATE.json", self.proposal)
        self.result["candidate_sha256"] = digest(self.candidate_root / "CANDIDATE.json")
        self.repin()
        with self.assertRaisesRegex(ValueError, "floor"):
            self.load()

    def test_capture_condition_band_and_reported_gains_reject(self):
        original = copy.deepcopy(self.result)
        for name in ("center", "candidate"):
            for key, value in (("minimum_occupied_capture", .9), ("maximum_overlap_condition", 1.1e12)):
                self.result = copy.deepcopy(original)
                self.result[name][key] = value
                self.repin()
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    self.load()
            self.result = copy.deepcopy(original)
            self.result[name]["coefficient_guard"]["maximum_target_band_change_ev"] = .051
            self.repin()
            with self.assertRaises(ValueError):
                self.load()
        for key in ("body_error_ev_per_c", "measured_ec_delta_ev_per_c", "measured_to_predicted_ec_gain",
                    "predicted_ec_delta_ev_per_c"):
            self.result = dict(copy.deepcopy(original), **{key:0.})
            self.repin(measurements=False)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()

    def test_all_declared_files_are_bounded_regular_hashed_and_unchanged(self):
        for path in sorted(p for p in self.stage.rglob("*") if p.is_file()):
            saved = path.read_bytes()
            path.write_bytes(saved+b"tampered")
            with self.subTest(path=path.relative_to(self.stage)), self.assertRaises(ValueError):
                self.load()
            path.write_bytes(saved)
        path = self.candidate_root / "COEFFICIENTS.txt"
        saved = path.read_bytes()
        path.unlink()
        path.symlink_to(self.center["coefficient_path"])
        with self.assertRaises(ValueError):
            self.load()
        path.unlink()
        path.write_bytes(saved)
        with path.open("wb") as stream:
            stream.truncate(8*1024*1024+1)
        with self.assertRaises(ValueError):
            self.load()

    def test_repinned_prepared_extra_unsafe_and_oversized_graphs_reject(self):
        original = copy.deepcopy(self.prepared)
        for values in ({"../escape":"0"*64}, {}, {str(i):"0"*64 for i in range(65)}):
            changed = dict(original, evidence_sha256=values)
            save(self.slot / self.pbe.PREPARATION, changed)
            self.collection["preparation_sha256"] = digest(self.slot / self.pbe.PREPARATION)
            self.repin()
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.load()

    def test_reduced_radius_preflights_the_exact_legacy_rejection_graph(self):
        rejected = self.reduced_fixture()
        self.load()
        # Otherwise the legacy validator would read a different, unbounded
        # hard-coded stage after this reader had preflighted a decoy graph.
        decoy = self.root / "decoy"
        for name in (RESULT, ACCEPTANCE, "GUARD_REJECTION_DIAGNOSTIC.json"):
            save(decoy / name, {})
        for value in (self.result, self.proposal):
            value["reduction_evidence"]["rejected_stage"] = str(decoy)
        self.sync_candidate()
        self.reconstruct.reset_mock()
        with self.assertRaisesRegex(ValueError, "reduction"):
            self.load()
        self.reconstruct.assert_not_called()
        for value in (self.result, self.proposal):
            value["reduction_evidence"]["rejected_stage"] = str(rejected)
        self.sync_candidate()
        path = rejected / "GUARD_REJECTION_DIAGNOSTIC.json"
        path.unlink()
        path.symlink_to(decoy / "GUARD_REJECTION_DIAGNOSTIC.json")
        with self.assertRaisesRegex(ValueError, "symlink|escapes endpoint"):
            self.load()

    def test_repinned_malformed_json_and_nonfinite_values_reject(self):
        path = self.stage / RESULT
        original = path.read_bytes()
        for data in (original.rstrip()[:-1]+b', "status": "success"}', b'[]',
                     original.replace(b'"loss": 0.08', b'"loss": NaN', 1)):
            self.assertNotEqual(data, original)
            path.write_bytes(data)
            self.result_sha = digest(path)
            with self.subTest(data=data[:40]), self.assertRaises(ValueError):
                self.load()
        path.write_bytes(original)
        self.repin()

    def test_fully_repinned_archive_still_rejects_nonregular_members(self):
        with tarfile.open(self.deployment["archive_path"], "w:gz") as archive:
            for name, data in self.source_files.items():
                member = tarfile.TarInfo(name)
                if name == SOURCE_FILES[0]:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/external/source"
                    archive.addfile(member)
                else:
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
        self.deployment["archive_sha256"] = digest(Path(self.deployment["archive_path"]))
        save(self.stage / "DEPLOYMENT.json", self.deployment)
        self.result["deployment_sha256"] = digest(self.stage / "DEPLOYMENT.json")
        self.repin()
        with self.assertRaisesRegex(ValueError, "nonregular"):
            self.load()


if __name__ == "__main__":
    unittest.main()
