"""Small archived evidence tests; opt in to tensor reconstruction on Linux."""

import copy
from contextlib import nullcontext
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


WF = Path(__file__).resolve().parents[1] / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
sys.path.insert(0, str(WF))
RESULT = "result/BAND_TANGENT_PREPARED.json"
ACCEPTANCE = "BAND_TANGENT_ACCEPTANCE.json"
SLOTS = [(n, r) for n in ("T", "N") for r in (-.001, .001, -.002, .002)]
LABELS = [1, 2, 3, 6, 7, 8, 11, 28]
WORKFLOW = "SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/"
SOURCE_FILES = (
    "prepare_c_band_tangent_probes.py", "run_c_band_tangent_preparation.slurm",
    "check_c_ec_constraint_screen.py", "screen_c_ec_step_constraints.py",
    "run_c_ec_constraint_screen.slurm", "prepare_c_pbe_direction_calibration.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True) + "\n")


def guard():
    return dict(gate=True, scope="frozen_h_band_screen_not_scf_energy", scf_pbe_gate="pending",
        occupied_band_sum_change_ev_per_atom=.0001, maximum_target_band_change_ev=.048,
        minimum_gap_ev=4.3, occupied_band_sum_limit_ev_per_atom=.01,
        target_band_change_limit_ev=.05, k_weight_sum=2.,
        k_weight_convention="ABACUS_spin_included_no_renormalization")


def detail():
    return dict(guard(), target_band_identity="sorted_eigenvalue_index_not_eigenvector_tracking",
        target_band_details=[dict(source_ik=k, target_ik=k, band_indices=list(range(1, 9)),
            signed_changes_ev=[0.]*7 + ([.048] if k == 43 else [.001])) for k in range(1, 65)])


class Array(list):
    def tolist(self):
        return list(self)


class BandTangentAcceptanceTest(unittest.TestCase):
    def patch(self, obj, name, **kwargs):
        p = mock.patch.object(obj, name, **kwargs)
        self.addCleanup(p.stop)
        return p.start()

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("check_c_band_tangent_preparation"),
                             "read-only band-tangent admission checker required")
        self.m = importlib.import_module("check_c_band_tangent_preparation")
        temporary = tempfile.TemporaryDirectory(prefix="c-band-admission-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage = self.root / "preparation"
        self.stage.mkdir()
        self.previous = self.root / "screen"
        self.previous.mkdir()
        center_path = self.root / "center" / "COEFFICIENTS.txt"
        center_path.parent.mkdir()
        center_path.write_text("accepted coefficients\n")
        orbital_path = center_path.with_suffix(".orb")
        orbital_path.write_text("accepted orbital\n")
        self.center = dict(result_sha256="c"*64, occupied_capture_floor=.9998,
            coefficient_path=center_path, orbital_path=orbital_path,
            result=dict(candidate=dict(coefficient_guard=guard()), actual_pbe=dict(
                status="collected", pbe_gate="pass", physical_release_gate="hold"),
                coefficient_sha256=sha(center_path), orbital_sha256=sha(orbital_path)))
        self.g = dict(energy_gradient=dict(channels=[]), coefficient_sha256=sha(center_path),
            orbital_sha256=sha(orbital_path), accepted_result_sha256=self.center["result_sha256"],
            load_records=[dict(label=k, mapping_sha256="a"*64, seconds=1.) for k in LABELS])
        self.parent = dict(source_commit="b"*40, deployment_sha256="b"*64,
            center_band_screen=guard(), load_records=copy.deepcopy(self.g["load_records"]),
            coefficient_sha256=self.g["coefficient_sha256"], orbital_sha256=self.g["orbital_sha256"],
            accepted_result_sha256=self.center["result_sha256"],
            gradient_result_sha256="d"*64, gradient_acceptance_sha256="e"*64)
        save(self.previous / "result/CONSTRAINT_SCREEN.json", self.parent)
        self.rows = [dict(element="C", l=l, zeta=z) for l, n in enumerate((3, 3, 2))
                     for z in range(1, n+1)]
        self.coefficients = {"C": [Array([[float(i == j) for j in range(n)] for i in range(31)])
                                    for n in (3, 3, 2, 0, 0)]}
        self.kernel_artifact = dict(status="success", scope="two_direction_band_tangent_pbe_calibration",
            signed_radii=[-.001, .001, -.002, .002], radials=copy.deepcopy(self.rows),
            directions={n: dict(matrices=copy.deepcopy(self.coefficients), radial_weights=[.1]*8,
                maximum_band_slope_ev=0., frozen_band_slope_ev_per_c=0., ec_slope_ha_per_cell=-1.)
                for n in ("T", "N")}, physical_release_gate="hold",
            actual_pbe_direction_derivatives="unmeasured")
        self.trials = [dict(direction_name=n, signed_radius=r, coefficients=copy.deepcopy(self.coefficients))
                       for n, r in SLOTS]
        for i, trial in enumerate(self.trials):
            trial["coefficients"]["C"][0][10][0] = (i+1)*.0001
        self.r = dict(status="success", scope="bounded_current_center_band_tangent_probe_preparation",
            source_commit="a"*40, screen_stage=str(self.previous),
            screen_result_sha256=sha(self.previous / "result/CONSTRAINT_SCREEN.json"),
            screen_acceptance_sha256="f"*64, gradient_result_sha256="d"*64,
            gradient_acceptance_sha256="e"*64, center_result_sha256=self.center["result_sha256"],
            center_coefficient_sha256=self.g["coefficient_sha256"],
            center_orbital_sha256=self.g["orbital_sha256"],
            center_actual_pbe=copy.deepcopy(self.center["result"]["actual_pbe"]),
            center_band_detail=detail(), preparation_gate="pass", cache_loads=1, probe_count=8,
            actual_scf_count=0, rpa_evaluations=0, backward_passes=0, optimizer_steps=0,
            physical_candidate_count=0, coefficient_update="none", physical_release_gate="hold",
            load_records=copy.deepcopy(self.g["load_records"]), cache_load_seconds=10.,
            total_seconds=20., peak_rss_kib=1000)
        self.artifact = dict(copy.deepcopy(self.kernel_artifact), **{k:self.r[k] for k in (
            "source_commit", "screen_result_sha256", "screen_acceptance_sha256",
            "center_result_sha256", "center_coefficient_sha256", "center_orbital_sha256")})
        self.probes = []
        self.r["probes"] = []
        for i, (name, radius) in enumerate(SLOTS):
            path = self.stage / ("result/probes/probe_%02d" % i)
            path.mkdir(parents=True)
            (path / "COEFFICIENTS.txt").write_text("probe %d\n" % i)
            (path / "C_3s3p2d_probe.orb").write_text("orbital %d\n" % i)
            self.probes.append(dict(status="prepared", scope="direction_calibration_probe",
                direction_name=name, signed_radius=radius,
                **{k:self.r[k] for k in ("center_result_sha256", "center_coefficient_sha256", "center_orbital_sha256")},
                cheap_gate=dict(gate=True, band_screen=detail(), minimum_occupied_capture=.9999,
                    occupied_capture_floor=.9998, overlap_rank_and_condition_gate="pass",
                    overlap_relative_rank_tolerance=1e-12, overlap_condition_limit=1e12),
                galerkin_energy="unmeasured", physical_release_gate="hold",
                coefficient_filename="COEFFICIENTS.txt", orbital_filename="C_3s3p2d_probe.orb",
                coefficient_sha256=sha(path / "COEFFICIENTS.txt"), orbital_sha256=sha(path / "C_3s3p2d_probe.orb")))
            self.r["probes"].append(dict(path="probe_%02d/PROBE.json" % i, status="prepared",
                                         direction_name=name, signed_radius=radius))
        import prepare_c_band_tangent_probes as preparation
        self.r["branch_diagnostic"] = preparation.summarize_target_branches(
            self.r["center_band_detail"], [p["cheap_gate"]["band_screen"] for p in self.probes])
        self.a = dict(status="success", scope="saved_band_tangent_preparation_acceptance_no_replay",
            job_id="7654321", scheduler="".join("7654321%s|COMPLETED|0:0|00:02:23|\n" % s
                for s in ("", ".batch", ".extern")), stderr_gate="empty", physical_release_gate="hold")
        (self.stage / "STATUS").write_bytes(b"success\n")
        (self.stage / "slurm-7654321.err").write_bytes(b"")
        import refresh_c_combined_step_gradient as refresh
        files = set(refresh.REQUIRED_SOURCE_FILES)
        files.update(WORKFLOW+n for n in SOURCE_FILES)
        files.update("SIAB/opt_orb_pytorch_dpsi/"+n for n in (
            "periodic_galerkin_band_tangent.py", "periodic_galerkin_ec_gradient_step.py",
            "periodic_galerkin_pbe_guard.py", "periodic_galerkin_fit.py"))
        files.add("SIAB/example_C_sternheimer/periodic_basis_optimization/export_periodic_orbitals.py")
        self.source_files = {n:("# old archived source, never execute: " + n + "\n").encode() for n in files}
        self.archive()
        self.repin()
        self.admit = self.patch(self.m.admission, "load_accepted_screen",
                               return_value=(self.center, self.g, self.parent, self.rows))
        self.rt = SimpleNamespace(torch=SimpleNamespace(no_grad=nullcontext,
            set_num_threads=mock.Mock(), equal=lambda a, b:a == b),
            read_coefficients=mock.Mock(side_effect=self.read_coefficients),
            build_band_tangent=mock.Mock(side_effect=lambda *a:copy.deepcopy(self.kernel_artifact)),
            signed_probes=mock.Mock(side_effect=lambda *a:copy.deepcopy(self.trials)))
        self.runtime = self.patch(self.m, "_runtime", return_value=self.rt)
        p = mock.patch.dict(sys.modules, {"torch": None, "numpy": None})
        p.start()
        self.addCleanup(p.stop)
        self.patch(self.m.admission.screen, "verify_source", side_effect=AssertionError("no running-source equality"))
        self.patch(self.m.admission.screen, "_runtime", side_effect=AssertionError("no frozen cache runtime"))
        self.patch(__import__("subprocess"), "run", side_effect=AssertionError("no processes or scheduler"))
        self.patch(__import__("subprocess"), "Popen", side_effect=AssertionError("no jobs"))

    def read_coefficients(self, path, **kwargs):
        if Path(path) == self.center["coefficient_path"]:
            return copy.deepcopy(self.coefficients)
        return copy.deepcopy(self.trials[int(Path(path).parent.name[-2:])]["coefficients"])

    def archive(self):
        for name, data in self.source_files.items():
            path = self.stage / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        path = self.stage / "source.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, data in self.source_files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        save(self.stage / "DEPLOYMENT.json", dict(source_commit="a"*40,
            source_directory=str(self.stage / "source"), archive_path=str(path), archive_sha256=sha(path),
            files={n:hashlib.sha256(v).hexdigest() for n, v in self.source_files.items()}))
        self.r["deployment_sha256"] = sha(self.stage / "DEPLOYMENT.json")

    def repin(self):
        path = self.stage / "result/probes/DIRECTIONS.json"
        save(path, self.artifact)
        self.r["directions_sha256"] = sha(path)
        for i, probe in enumerate(self.probes):
            probe["directions_sha256"] = self.r["directions_sha256"]
            path = self.stage / ("result/probes/probe_%02d/PROBE.json" % i)
            save(path, probe)
            self.r["probes"][i]["sha256"] = sha(path)
        save(self.stage / RESULT, self.r)
        self.result_sha = sha(self.stage / RESULT)
        save(self.stage / "PROVENANCE.json", dict(
            {k:self.r[k] for k in ("status", "scope", "source_commit", "deployment_sha256",
                                  "preparation_gate", "physical_release_gate")}, result_sha256=self.result_sha))
        self.a.update({k:self.r[k] for k in ("source_commit", "deployment_sha256", "directions_sha256",
            "center_result_sha256", "center_coefficient_sha256", "center_orbital_sha256")})
        self.a["result_sha256"] = self.result_sha
        save(self.stage / ACCEPTANCE, self.a)
        self.acceptance_sha = sha(self.stage / ACCEPTANCE)

    def load(self, accepted=False, **changes):
        args = dict(stage=self.stage, result_sha256=self.result_sha,
                    deployment_sha256=self.r["deployment_sha256"], source_commit="a"*40)
        if accepted:
            args["acceptance_sha256"] = self.acceptance_sha
        return (self.m.load_accepted_preparation if accepted else self.m.load_preparation)(**dict(args, **changes))

    def test_admits_archive_and_reconstructs_from_independently_accepted_center(self):
        before = {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with mock.patch.object(Path, "write_bytes", side_effect=AssertionError("read only")), \
                mock.patch.object(Path, "write_text", side_effect=AssertionError("read only")), \
                mock.patch.object(Path, "mkdir", side_effect=AssertionError("no staging")):
            loaded = self.load(accepted=True)
        self.assertIs(loaded["center"], self.center)
        self.assertIs(loaded["gradient"], self.g)
        self.assertEqual(loaded["result"], self.r)
        self.assertEqual(loaded["directions"], self.artifact)
        self.admit.assert_called_once_with(self.previous, self.r["screen_result_sha256"], "f"*64,
                                           "b"*64, "b"*40, "d"*64, "e"*64)
        self.rt.build_band_tangent.assert_called_once_with(self.coefficients, self.g["energy_gradient"], self.rows)
        self.rt.signed_probes.assert_called_once_with(self.coefficients, self.artifact)
        self.assertEqual(len(loaded["probes"]), 8)
        for i, row in enumerate(loaded["probes"]):
            path = self.stage / ("result/probes/probe_%02d/PROBE.json" % i)
            self.assertEqual(row["path"], path)
            self.assertEqual(row["manifest"], self.probes[i])
            self.assertEqual(row["files"], {n:(path.parent/n).read_bytes() for n in (
                "PROBE.json", "COEFFICIENTS.txt", "C_3s3p2d_probe.orb")})
        self.assertEqual(before, {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_external_pins_and_commit_are_mandatory(self):
        for key in ("result_sha256", "acceptance_sha256", "deployment_sha256", "source_commit"):
            for bad in (None, "", "A"*64, "0"*(40 if key == "source_commit" else 64)):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    self.load(accepted=True, **{key:bad})

    def test_28_threads_are_set_before_exact_direction_and_signed_qr_builds(self):
        calls = mock.Mock()
        calls.attach_mock(self.rt.torch.set_num_threads, "threads")
        calls.attach_mock(self.rt.build_band_tangent, "directions")
        calls.attach_mock(self.rt.signed_probes, "probes")
        self.load()
        self.assertEqual([c[0] for c in calls.mock_calls], ["threads", "directions", "probes"])
        self.assertEqual(calls.mock_calls[0], mock.call.threads(28))

    def test_external_verifier_audit_metadata_is_compatible(self):
        self.a.update(preparation_gate="pass", probe_count=8, new_scf_count=0, new_rpa_count=0,
            coefficient_update="none", branch_diagnostic=copy.deepcopy(self.r["branch_diagnostic"]),
            probes=copy.deepcopy(self.r["probes"]),
            timing=dict(cache_load_seconds=self.r["cache_load_seconds"], total_seconds=self.r["total_seconds"],
                result_peak_rss_kib=self.r["peak_rss_kib"], process_peak_rss_kib=self.r["peak_rss_kib"]+100),
            checker_sha256="1"*64, verifier_sha256="2"*64)
        self.repin()
        loaded = self.load(accepted=True)
        self.assertEqual(loaded["result"], self.r)
        self.assertEqual(loaded["result"]["physical_release_gate"], "hold")

    def test_status_provenance_archived_source_and_every_probe_hash(self):
        paths = [self.stage/n for n in ("STATUS", "PROVENANCE.json", "DEPLOYMENT.json", "source.tar.gz", RESULT,
                                       ACCEPTANCE, "result/probes/DIRECTIONS.json")]
        paths.append(self.stage / "source" / next(iter(self.source_files)))
        paths.extend(self.stage / ("result/probes/probe_%02d/" % i) / n for i in range(8)
                     for n in ("PROBE.json", "COEFFICIENTS.txt", "C_3s3p2d_probe.orb"))
        for path in paths:
            data = path.read_bytes()
            path.write_bytes(data+b"tampered")
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.load(accepted=True)
            path.write_bytes(data)

    def test_provenance_exact_fields_and_status_bytes(self):
        path = self.stage / "PROVENANCE.json"
        original = json.loads(path.read_text())
        for key in original:
            data = dict(original)
            del data[key]
            save(path, data)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
        save(path, dict(original, extra="not in preparation provenance"))
        with self.assertRaises(ValueError):
            self.load()
        save(path, original)
        for status in (b"success", b" success\n", b"success\n\n"):
            (self.stage / "STATUS").write_bytes(status)
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.load()

    def test_required_historical_runner_and_kernel_cannot_be_omitted(self):
        for name in (WORKFLOW+"prepare_c_band_tangent_probes.py",
                     WORKFLOW+"run_c_band_tangent_preparation.slurm",
                     WORKFLOW+"check_c_ec_constraint_screen.py",
                     "SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_band_tangent.py"):
            data = self.source_files.pop(name)
            (self.stage / "source" / name).unlink()
            self.archive()
            self.repin()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load()
            self.source_files[name] = data
            self.archive()
            self.repin()

    def test_parent_hash_and_readmission_fail_closed(self):
        path = self.previous / "result/CONSTRAINT_SCREEN.json"
        path.write_text("{}")
        with self.assertRaises(ValueError):
            self.load()
        save(path, self.parent)
        self.admit.side_effect = ValueError("parent acceptance failed")
        with self.assertRaisesRegex(ValueError, "parent acceptance failed"):
            self.load()
        self.runtime.assert_not_called()

    def test_rehashed_result_identity_scope_and_zero_budgets(self):
        changes = dict(status="prepared", scope="optimized_basis", preparation_gate="rejected_cheap_probe",
            cache_loads=2, probe_count=9, actual_scf_count=1, rpa_evaluations=1, backward_passes=1,
            optimizer_steps=1, physical_candidate_count=1, coefficient_update="updated",
            physical_release_gate="pass", center_result_sha256="0"*64,
            center_coefficient_sha256="0"*64, center_orbital_sha256="0"*64,
            gradient_result_sha256="0"*64, gradient_acceptance_sha256="0"*64,
            center_actual_pbe=dict(status="unmeasured"))
        original = copy.deepcopy(self.r)
        for key, value in changes.items():
            self.r = dict(copy.deepcopy(original), **{key:value})
            self.repin()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
        for key in ("cache_loads", "probe_count", "actual_scf_count", "rpa_evaluations", "optimizer_steps"):
            self.r = dict(copy.deepcopy(original), **{key:float(original[key])})
            self.repin()
            with self.subTest(type_key=key), self.assertRaises(ValueError):
                self.load()

    def test_center_is_not_a_rebased_probe_reference(self):
        self.r["center_band_detail"]["minimum_gap_ev"] += .1
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_direction_metadata_is_rebuilt_not_merely_hash_checked(self):
        original = copy.deepcopy(self.artifact)
        for key, value in dict(source_commit="0"*40, screen_result_sha256="0"*64,
                screen_acceptance_sha256="0"*64, center_result_sha256="0"*64,
                center_coefficient_sha256="0"*64, center_orbital_sha256="0"*64,
                physical_release_gate="pass", actual_pbe_direction_derivatives="measured",
                signed_radii=[-.002, .002], radials=[]).items():
            self.artifact = dict(copy.deepcopy(original), **{key:value})
            self.repin()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
        self.artifact = copy.deepcopy(original)
        self.artifact["directions"]["T"]["matrices"]["C"][0][0][0] += .001
        self.repin()
        with self.assertRaises(ValueError):
            self.load()
        self.artifact = dict(original, invented_metadata="not in reconstructed artifact")
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_all_eight_ordered_slots_and_both_manifest_identities(self):
        original = copy.deepcopy(self.r["probes"])
        for key, bad in (("path", "../probe_00/PROBE.json"), ("path", "probe_01/PROBE.json"),
                         ("direction_name", "N"), ("signed_radius", .001), ("status", "rejected")):
            self.r["probes"] = copy.deepcopy(original)
            self.r["probes"][0][key] = bad
            self.repin()
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                self.load()
        self.r["probes"] = copy.deepcopy(original)
        self.repin()
        self.r["probes"].reverse()
        save(self.stage / RESULT, self.r)
        self.result_sha = sha(self.stage / RESULT)
        with self.assertRaises(ValueError):
            self.load()

    def test_probe_manifest_and_original_cheap_protections(self):
        original = copy.deepcopy(self.probes[0])
        changes = dict(status="rejected", scope="actual_pbe_tangent_candidate", direction_name="N",
            signed_radius=.002, center_result_sha256="0"*64, center_coefficient_sha256="0"*64,
            center_orbital_sha256="0"*64, galerkin_energy="measured", physical_release_gate="pass",
            coefficient_filename="../COEFFICIENTS.txt", orbital_filename="elsewhere.orb")
        for key, bad in changes.items():
            self.probes[0] = dict(copy.deepcopy(original), **{key:bad})
            self.repin()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
        for key, bad in dict(gate=False, occupied_capture_floor=.9, minimum_occupied_capture=.9997,
                overlap_rank_and_condition_gate="pending", overlap_relative_rank_tolerance=1e-10,
                overlap_condition_limit=1e13).items():
            self.probes[0] = copy.deepcopy(original)
            self.probes[0]["cheap_gate"][key] = bad
            self.repin()
            with self.subTest(cheap=key), self.assertRaises(ValueError):
                self.load()

    def test_every_saved_guard_enforces_original_limits_and_finite_metrics(self):
        original = copy.deepcopy(self.probes)
        changes = dict(gate=False, scf_pbe_gate="pass", scope="scf_energy", k_weight_sum=1.,
            k_weight_convention="normalized", occupied_band_sum_limit_ev_per_atom=.02,
            target_band_change_limit_ev=.06, occupied_band_sum_change_ev_per_atom=-.01001,
            maximum_target_band_change_ev=.05001, minimum_gap_ev=0.)
        for key, bad in changes.items():
            self.probes = copy.deepcopy(original)
            self.probes[7]["cheap_gate"]["band_screen"][key] = bad
            self.repin()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
        for bad in (True, "4.3", float("nan"), float("inf")):
            self.probes = copy.deepcopy(original)
            self.probes[3]["cheap_gate"]["band_screen"]["minimum_gap_ev"] = bad
            self.repin()
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.load()

    def test_probe_coefficients_must_equal_signed_qr_reconstruction(self):
        def changed(path, **kwargs):
            c = self.read_coefficients(path, **kwargs)
            if Path(path) != self.center["coefficient_path"]:
                c["C"][0][0][0] += 1e-7
            return c
        self.rt.read_coefficients.side_effect = changed
        with self.assertRaises(ValueError):
            self.load()

    def test_full_saved_band_detail_not_only_the_reported_maximum(self):
        original = copy.deepcopy(self.probes)
        mutations = (
            lambda d:d["target_band_details"].pop(),
            lambda d:d["target_band_details"].reverse(),
            lambda d:d["target_band_details"][0].update(source_ik=True),
            lambda d:d["target_band_details"][0].update(target_ik=2),
            lambda d:d["target_band_details"][0].update(band_indices=list(range(2, 10))),
            lambda d:d["target_band_details"][0]["signed_changes_ev"].pop(),
            lambda d:d["target_band_details"][0]["signed_changes_ev"].__setitem__(0, True),
            lambda d:d["target_band_details"][0]["signed_changes_ev"].__setitem__(0, float("nan")),
            lambda d:d["target_band_details"][42]["signed_changes_ev"].__setitem__(7, .047),
            lambda d:d.update(target_band_identity="tracked_eigenvectors"),
        )
        for i, mutate in enumerate(mutations):
            self.probes = copy.deepcopy(original)
            mutate(self.probes[5]["cheap_gate"]["band_screen"])
            self.repin()
            with self.subTest(mutation=i), self.assertRaises(ValueError):
                self.load()

    def test_center_detail_has_the_same_full_grid_requirement(self):
        self.r["center_band_detail"]["target_band_details"].pop(0)
        for p in self.probes:
            p["cheap_gate"]["band_screen"]["target_band_details"].pop(0)
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_branch_diagnostic_is_reanalyzed_including_ties_and_switches(self):
        d = self.probes[0]["cheap_gate"]["band_screen"]
        d["target_band_details"][0]["signed_changes_ev"][0] = .048
        self.repin()
        with self.assertRaises(ValueError):
            self.load()
        import prepare_c_band_tangent_probes as preparation
        self.r["branch_diagnostic"] = preparation.summarize_target_branches(
            self.r["center_band_detail"], [p["cheap_gate"]["band_screen"] for p in self.probes])
        self.repin()
        result = self.load()["result"]
        self.assertFalse(result["branch_diagnostic"]["probe_maxima_within_center_near_active_set"])
        self.assertEqual(result["physical_release_gate"], "hold")
        self.r["branch_diagnostic"]["smoothness_gate"] = "proven"
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_q_mapping_order_hashes_and_timing_bind_both_parents(self):
        original = copy.deepcopy(self.r)
        mutations = (
            lambda r:r["load_records"].reverse(),
            lambda r:r["load_records"].pop(),
            lambda r:r["load_records"][0].update(label=True),
            lambda r:r["load_records"][3].update(mapping_sha256="0"*64),
            lambda r:r["load_records"][3].update(seconds=-1.),
            lambda r:r.update(cache_load_seconds=0.),
            lambda r:r.update(total_seconds=9.),
            lambda r:r.update(peak_rss_kib=True),
            lambda r:r.update(total_seconds=float("nan")),
            lambda r:r.update(cache_load_seconds=float("inf")),
        )
        for i, mutate in enumerate(mutations):
            self.r = copy.deepcopy(original)
            mutate(self.r)
            self.repin()
            with self.subTest(mutation=i), self.assertRaises(ValueError):
                self.load()
        self.r = copy.deepcopy(original)
        self.parent["load_records"][0]["mapping_sha256"] = "0"*64
        save(self.previous / "result/CONSTRAINT_SCREEN.json", self.parent)
        self.r["screen_result_sha256"] = sha(self.previous / "result/CONSTRAINT_SCREEN.json")
        self.artifact["screen_result_sha256"] = self.r["screen_result_sha256"]
        self.repin()
        with self.assertRaises(ValueError):
            self.load()

    def test_acceptance_is_external_and_not_needed_for_preparation(self):
        (self.stage / ACCEPTANCE).unlink()
        self.load()
        with self.assertRaises(ValueError):
            self.load(accepted=True)

    def test_acceptance_contract_fields_are_required_and_hash_bound(self):
        original = copy.deepcopy(self.a)
        for key in original:
            a = dict(original)
            del a[key]
            save(self.stage / ACCEPTANCE, a)
            self.acceptance_sha = sha(self.stage / ACCEPTANCE)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.load(accepted=True)
        for key in ("result_sha256", "deployment_sha256", "directions_sha256", "center_result_sha256",
                    "center_coefficient_sha256", "center_orbital_sha256", "source_commit",
                    "status", "scope", "stderr_gate", "physical_release_gate"):
            save(self.stage / ACCEPTANCE, dict(original, **{key:"wrong"}))
            self.acceptance_sha = sha(self.stage / ACCEPTANCE)
            with self.subTest(changed=key), self.assertRaises(ValueError):
                self.load(accepted=True)

    def test_all_three_scheduler_records_and_exact_job_stderr(self):
        original = self.a["scheduler"]
        for text in (original.replace("COMPLETED", "RUNNING", 1), original.replace("0:0", "1:0", 1),
                     original.replace(".extern", ".0"), "\n".join(original.splitlines()[:2]),
                     original+original.splitlines()[0]+"\n", original.replace("7654321", "1234567")):
            self.a["scheduler"] = text
            self.repin()
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(accepted=True)
        self.a["scheduler"] = original
        self.repin()
        for data in (b"\n", b"warning\n", b" "):
            (self.stage / "slurm-7654321.err").write_bytes(data)
            with self.subTest(stderr=data), self.assertRaises(ValueError):
                self.load(accepted=True)
        (self.stage / "slurm-7654321.err").write_bytes(b"")
        self.a.update(job_id="1234567", scheduler=original.replace("7654321", "1234567"))
        self.repin()
        with self.assertRaises(ValueError):
            self.load(accepted=True)
        (self.stage / "slurm-1234567.err").write_bytes(b"")
        self.load(accepted=True)

    def test_symlinks_extra_slots_and_noncanonical_paths_are_rejected(self):
        path = self.stage / "result/probes/probe_00/COEFFICIENTS.txt"
        content = path.read_bytes()
        outside = self.root / "outside.txt"
        outside.write_bytes(content)
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.load()
        path.unlink()
        path.write_bytes(content)
        extra = self.stage / "result/probes/probe_08"
        extra.mkdir()
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_json_nonfinite_json_and_bounded_reads(self):
        path = self.stage / "result/probes/DIRECTIONS.json"
        for data in (b'{"x":1,"x":2}', b'{"x":1e999}'):
            path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load()
        self.repin()
        with mock.patch.object(self.m.admission.screen.refresh.accepted, "_MAX_FILE_BYTES", 64), \
                self.assertRaises(ValueError):
            self.load()


@unittest.skipUnless(os.environ.get("C_BAND_TANGENT_NUMERICAL_TESTS") == "1" and sys.platform != "darwin",
                     "explicit remote tensor test; no local Torch")
class BandTangentNumericalTest(unittest.TestCase):
    def test_real_kernels_and_saved_coefficient_round_trip_without_physics(self):
        m = importlib.import_module("check_c_band_tangent_preparation")
        rt = m._runtime()
        torch = rt.torch
        torch.set_num_threads(28)
        from periodic_galerkin_radial_diagnostics import radial_gradient_report
        from periodic_galerkin_basis import write_periodic_optimizer_coefficients
        from periodic_galerkin_band_tangent import build_band_tangent
        from periodic_galerkin_direction_calibration import signed_probes
        c = {"C":[torch.eye(31, dtype=torch.float64)[:, :n].clone() for n in (3, 3, 2, 0, 0)]}
        raw = {"C":[torch.zeros_like(v) for v in c["C"]]}
        rows = []
        b = [.4, -.2, -.01, .94, -.37, -.013, .73, 1.55]
        for l, block in enumerate(raw["C"]):
            for z in range(block.shape[1]):
                i = len(rows)
                block[10+z, z] = i+1.
                rows.append(dict(element="C", l=l, zeta=z+1, occupied_slope_ev_per_c=(-1.)**z,
                    target_slope_ev=b[i], occupied_two_radius_difference=0., target_two_radius_difference=0.,
                    active_band_identity="unmeasured"))
        report = radial_gradient_report(c, raw)
        with mock.patch.object(torch.linalg, "eigvalsh", side_effect=AssertionError("no diagonalization")), \
                mock.patch.object(torch.linalg, "eigh", side_effect=AssertionError("no diagonalization")):
            expected = build_band_tangent(c, report, rows)
            self.assertEqual(rt.build_band_tangent(c, report, rows), expected)
            probes = rt.signed_probes(c, expected)
            self.assertEqual([(p["direction_name"], p["signed_radius"]) for p in probes], SLOTS)
            with tempfile.TemporaryDirectory() as tmp:
                for i, (p, reference) in enumerate(zip(probes, signed_probes(c, expected))):
                    path = Path(tmp) / ("coef_%d.txt" % i)
                    write_periodic_optimizer_coefficients(path, p["coefficients"])
                    saved = rt.read_coefficients(path, element="C", radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
                    for a, b in zip(saved["C"], reference["coefficients"]["C"]):
                        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
