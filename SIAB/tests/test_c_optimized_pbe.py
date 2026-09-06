"""Solid-only endpoint staging and log evidence; never launches ABACUS."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODULE = (Path(__file__).resolve().parents[1] / "example_C_sternheimer" /
          "periodic_basis_optimization/galerkin_binding_workflow/check_c_optimized_pbe.py")
endpoint = None
if MODULE.is_file():
    spec = importlib.util.spec_from_file_location("check_c_optimized_pbe", str(MODULE))
    endpoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(endpoint)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


BASELINE = -309.8590820440637
INPUT = """INPUT_PARAMETERS
suffix C_Q1
calculation scf
basis_type lcao
dft_functional pbe
ntype 1
nspin 1
ecutwfc 45
lcao_ecut 100
nx 24
ny 24
nz 24
kpar 4
nbands 44
symmetry 1
smearing_method fixed
scf_thr 1e-8
scf_nmax 120
mixing_beta 0.4
rpa 1
out_chg 0
out_wfc_lcao 0
sternheimer_q_index 1
sternheimer_frequency_grid_file FREQUENCY_GRID.tsv
out_sternheimer_basis_opt 1
out_sternheimer_librpa 1
bessel_ecut 100
bessel_rcut 10
out_librpa_reader_version 1
exx_pca_threshold 1e-6
rpa_ccp_rmesh_times 2
init_chg file
init_wfc file
read_file_dir ../old-output/
restart_load true
"""
STRU = """ATOMIC_SPECIES
C 12.011 C.upf

NUMERICAL_ORBITAL
C_original.orb

LATTICE_CONSTANT
6.7410

LATTICE_VECTORS
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0

ATOMIC_POSITIONS
Direct
C
0.0
2
0.0 0.0 0.0 0 0 0
0.25 0.25 0.25 0 0 0
"""


def log_text(energy=BASELINE, counts=True):
    return ((" Occupied electronic states = 4\n"
             " Number of electronic states (NBANDS) = 44\n") if counts else "") + (
        " #SCF IS CONVERGED#\n"
        " #TOTAL ENERGY# {:.16f} eV\n"
        " !FINAL_ETOT_IS {:.16f} eV\n".format(energy, energy)
    )


class COptimizedPbeTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(endpoint, "solid-only PBE endpoint module is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "q1"
        self.original.mkdir()
        for name, text in (("INPUT", INPUT), ("STRU", STRU),
                           ("KPT", "K_POINTS\n0\nGamma\n4 4 4 0 0 0\n"),
                           ("C.upf", "original C PBE pseudopotential\n"),
                           ("C_original.orb", "original C orbital\n"),
                           ("old.wfc", "old wavefunctions\n"),
                           ("SPIN1_CHG.cube", "old density\n")):
            (self.original / name).write_text(text)
        out = self.original / "OUT.C_Q1"
        out.mkdir()
        self.baseline_log = out / "running_scf.log"
        self.baseline_log.write_text(log_text())
        self.optimized = self.root / "optimized"
        self.optimized.mkdir()
        self.coefficient = self.optimized / "ORBITAL_RESULTS.txt"
        self.coefficient.write_text("candidate coefficients\n")
        self.best_coefficient = self.optimized / "BEST_ORBITAL_CHECKPOINT.txt"
        self.best_coefficient.write_bytes(self.coefficient.read_bytes())
        self.orbital = self.optimized / "C_3s3p2d_optimized.orb"
        self.orbital.write_text("Element C\ncandidate 3s3p2d orbital\n")
        self.metadata = self.optimized / "BEST_CHECKPOINT.json"
        self.metadata.write_text(json.dumps({
            "objective": "rpa", "step": 2, "loss": 0.4,
            "orbital_file": self.best_coefficient.name,
            "orbital_sha256": digest(self.best_coefficient),
        }))
        self.result = self.optimized / "RESULT.json"
        self.result.write_text(json.dumps({
            "status": "success", "scope": "optimized_full_q_frozen_body_rpa_calibration",
            "steps_completed": 3, "best_step": 2,
            "nu": [3, 3, 2, 0, 0], "ao_per_C": 22, "fixed_nu": [0, 0, 0, 0, 0],
            "initial": {"step": 0, "loss": 0.8}, "best": {"step": 2, "loss": 0.4},
            "coefficient_sha256": digest(self.coefficient), "orbital_sha256": digest(self.orbital),
        }))
        self.output = self.root / "endpoint"

    def prepare(self, output=None, **options):
        values = dict(
            optimizer_result=self.result, optimizer_result_sha256=digest(self.result),
            original_q1=self.original, baseline_log_sha256=digest(self.baseline_log),
            output=self.output if output is None else output,
        )
        values.update(options)
        return endpoint.prepare_pbe_endpoint(**values)

    def collect(self, text=None, output=None, **options):
        work = self.output if output is None else output
        if text is not None:
            out = work / "OUT.C_Q1"
            out.mkdir(exist_ok=True)
            (out / "running_scf.log").write_text(text)
        values = dict(prepared_dir=work, preparation_sha256=digest(work / "PBE_ENDPOINT_PREPARED.json"))
        values.update(options)
        return endpoint.collect_pbe_endpoint(**values)

    def test_prepare_preserves_geometry_kpoints_pp_and_installs_exact_candidate(self):
        result = self.prepare()
        for name in ("STRU", "KPT", "C.upf"):
            self.assertEqual((self.output / name).read_bytes(), (self.original / name).read_bytes())
        self.assertEqual((self.output / "C_original.orb").read_bytes(), self.orbital.read_bytes())
        self.assertEqual(result["candidate_orbital_sha256"], digest(self.orbital))
        self.assertEqual(result["baseline"]["energy_ev"], BASELINE)
        self.assertEqual(result["baseline"]["log_sha256"], digest(self.baseline_log))
        for name in ("INPUT", "STRU", "KPT", "C.upf", "C_original.orb"):
            self.assertEqual(result["original_input_sha256"][name], digest(self.original / name))
            self.assertEqual(result["prepared_input_sha256"][name], digest(self.output / name))
        self.assertEqual(result["scheduler_gate"], "pending_external_validation")

    def test_prepared_input_is_pure_pbe_with_frozen_numerics_and_no_restart(self):
        self.prepare()
        values = endpoint.read_input(self.output / "INPUT")
        for key, value in (("calculation", "scf"), ("rpa", "0"), ("nspin", "1"),
                           ("out_chg", "1"), ("out_wfc_lcao", "1"),
                           ("init_chg", "atomic"), ("init_wfc", "atomic")):
            self.assertEqual(values[key], value)
        for key in ("ecutwfc", "lcao_ecut", "nx", "ny", "nz", "kpar", "nbands",
                    "symmetry", "smearing_method", "mixing_beta", "scf_thr", "scf_nmax"):
            self.assertEqual(values[key], endpoint.read_input(self.original / "INPUT")[key])
        self.assertFalse(any(key.startswith(("sternheimer", "out_sternheimer", "bessel",
                                            "restart", "exx_", "rpa_", "out_librpa")) for key in values))
        self.assertNotIn("read_file_dir", values)
        self.assertFalse((self.output / "old.wfc").exists())
        self.assertFalse((self.output / "SPIN1_CHG.cube").exists())
        self.assertFalse((self.output / "OUT.C_Q1").exists())

    def test_prepare_retains_original_scf_threshold_and_iteration_limit(self):
        original = INPUT.replace("scf_thr 1e-8", "scf_thr 3.0e-10").replace(
            "scf_nmax 120", "scf_nmax 280"
        )
        (self.original / "INPUT").write_text(original)
        self.prepare()
        values = endpoint.read_input(self.output / "INPUT")
        self.assertEqual(values["scf_thr"], "3.0e-10")
        self.assertEqual(values["scf_nmax"], "280")

    def test_prepare_rejects_wrong_or_missing_all_radial_basis_metadata(self):
        original = json.loads(self.result.read_text())
        cases = (
            ("nu", [2, 2, 1, 0, 0]), ("nu", [3, 3, 2]), ("nu", [3, 3, 2, False, 0]),
            ("nu", [3.0, 3, 2, 0, 0]), ("ao_per_C", 21), ("ao_per_C", 22.0),
            ("fixed_nu", [1, 0, 0, 0, 0]), ("fixed_nu", [0, 0, 0]),
            ("fixed_nu", [False, 0, 0, 0, 0]),
            ("nu", None), ("ao_per_C", None), ("fixed_nu", None),
        )
        for index, (key, value) in enumerate(cases):
            data = dict(original)
            if value is None:
                data.pop(key)
            else:
                data[key] = value
            self.result.write_text(json.dumps(data))
            output = self.root / ("invalid_metadata_" + str(index))
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    self.prepare(output=output)
                self.assertFalse(output.exists())

    def test_prepare_rejects_incomplete_unimproved_or_inconsistent_result(self):
        original = self.result.read_text()
        for change in ({"status": "running"}, {"steps_completed": 0}, {"steps_completed": True},
                       {"best": {"step": 2, "loss": 0.8}},
                       {"best": {"step": 2, "loss": float("nan")}},
                       {"initial": {"step": 0, "loss": float("inf")}},
                       {"best_step": 4}, {"scope": "atom_binding"}):
            data = json.loads(original)
            data.update(change)
            self.result.write_text(json.dumps(data))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_prepare_requires_matching_result_and_baseline_hashes(self):
        for option in ("optimizer_result_sha256", "baseline_log_sha256"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "SHA256"):
                self.prepare(**{option: "0" * 64})
        self.assertFalse(self.output.exists())

    def test_prepare_rejects_tampered_coefficient_checkpoint_or_export(self):
        for path in (self.coefficient, self.best_coefficient, self.orbital, self.metadata):
            saved = path.read_bytes()
            path.write_bytes(saved + b"corrupt")
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.prepare()
            path.write_bytes(saved)
            self.assertFalse(self.output.exists())

    def test_prepare_rejects_wrong_best_metadata_even_with_equal_coefficient_bytes(self):
        data = json.loads(self.metadata.read_text())
        for change in ({"step": 1}, {"loss": 0.3}, {"orbital_sha256": "0" * 64}):
            self.metadata.write_text(json.dumps(dict(data, **change)))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()

    def test_prepare_rejects_wrong_spin_or_changed_frozen_numerics(self):
        for old, new in (("nspin 1", "nspin 2"), ("ecutwfc 45", "ecutwfc 46"),
                         ("lcao_ecut 100", "lcao_ecut 90"), ("nx 24", "nx 22"),
                         ("ny 24", "ny 22"), ("nz 24", "nz 22"), ("kpar 4", "kpar 1"),
                         ("nbands 44", "nbands 40"), ("symmetry 1", "symmetry 0"),
                         ("smearing_method fixed", "smearing_method gauss"),
                         ("dft_functional pbe", "dft_functional hse")):
            (self.original / "INPUT").write_text(INPUT.replace(old, new))
            with self.subTest(parameter=old), self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_prepare_rejects_non_c2_structure_and_unsafe_filenames(self):
        for text in (STRU.replace("\n2\n", "\n1\n"), STRU.replace("C 12.011", "O 15.999"),
                     STRU.replace("C.upf", "../C.upf"), STRU.replace("C_original.orb", "/tmp/C.orb"),
                     STRU + "\nABFS_ORBITAL\nC.abfs\n"):
            (self.original / "STRU").write_text(text)
            with self.subTest(stru=text), self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_prepare_rejects_active_non_pbe_or_spin_extensions(self):
        for key in ("dft_plus_u", "deepks_scf", "lspinorb", "noncolin"):
            (self.original / "INPUT").write_text(INPUT + key + " 1\n")
            with self.subTest(parameter=key), self.assertRaisesRegex(ValueError, key):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_prepare_rejects_unconverged_or_ambiguous_baseline(self):
        for text in (log_text().replace("#SCF IS CONVERGED#", ""),
                     log_text() + " !!SCF IS NOT CONVERGED!!\n",
                     log_text() + " !FINAL_ETOT_IS -300 eV\n",
                     "#SCF IS CONVERGED#\nEtot_without_rpa(Ha): -10\n",
                     log_text().replace(str(BASELINE), "nan")):
            self.baseline_log.write_text(text)
            with self.subTest(log=text), self.assertRaises(ValueError):
                self.prepare()

    def test_prepare_requires_new_directory_and_does_not_modify_sources(self):
        paths = list(self.original.rglob("*")) + list(self.optimized.rglob("*"))
        saved = {path: path.read_bytes() for path in paths if path.is_file()}
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        for path, content in saved.items():
            self.assertEqual(path.read_bytes(), content)

    def test_collect_reports_absolute_solid_pbe_total_energy_gate(self):
        for index, (delta, gate) in enumerate(((0.009, "pass"), (0.010, "pass"),
                                              (0.011, "fail"), (-0.009, "pass"),
                                              (-0.010, "pass"), (-0.011, "fail"),
                                              (-0.020, "fail"))):
            work = self.root / ("endpoint" + str(index))
            self.prepare(output=work)
            result = self.collect(log_text(BASELINE + 2 * delta), output=work)
            self.assertAlmostEqual(result["energy_delta_ev_per_c"], delta, places=12)
            self.assertEqual(result["pbe_total_energy_gate"], gate)
            self.assertEqual(result["comparison"], "absolute_candidate_minus_original_per_C_lte_tolerance")
            self.assertEqual(result["energy_quantity"], "PBE_total_energy_not_RPA_E0")
            self.assertEqual(result["scheduler_gate"], "pending_external_validation")
            self.assertEqual(result["physical_release_gate"], "hold")
            self.assertEqual(result["band_counts"], {"occupied": 4, "total": 44})

    def test_collect_rejects_large_negative_energy_shift(self):
        self.prepare()
        result = self.collect(log_text(BASELINE - 0.4))
        self.assertAlmostEqual(result["energy_delta_ev_per_c"], -0.2, places=12)
        self.assertEqual(result["pbe_total_energy_gate"], "fail")
        self.assertEqual(result["scheduler_gate"], "pending_external_validation")

    def test_collect_does_not_use_status_as_scf_or_scheduler_evidence(self):
        self.prepare()
        (self.output / "STATUS").write_text("success\n")
        with self.assertRaises(ValueError):
            self.collect("!FINAL_ETOT_IS -309.85 eV\n")
        (self.output / "STATUS").write_text("failed\n")
        result = self.collect(log_text())
        self.assertEqual(result["scf_log_gate"], "pass")
        self.assertEqual(result["scheduler_gate"], "pending_external_validation")

    def test_collect_rejects_nonconvergence_nonfinite_duplicate_or_wrong_energy(self):
        self.prepare()
        for text in (log_text().replace("#SCF IS CONVERGED#", ""),
                     log_text() + " !!SCF IS NOT CONVERGED!!\n",
                     log_text() + " convergence has NOT been achieved\n",
                     log_text() + " !FINAL_ETOT_IS -300 eV\n",
                     "#SCF IS CONVERGED#\n!FINAL_ETOT_IS nan eV\n",
                     "#SCF IS CONVERGED#\n!FINAL_ETOT_IS +inf eV\n",
                     "#SCF IS CONVERGED#\n!FINAL_ETOT_IS -310 Ry\n",
                     "#SCF IS CONVERGED#\n!FINAL_ETOT_IS -310 Ha\n",
                     "#SCF IS CONVERGED#\n!FINAL_ETOT_IS -310\n",
                     "!FINAL_ETOT_IS -310 eV\n#SCF IS CONVERGED#\n",
                     log_text() + "#SCF IS CONVERGED#\n",
                     "#SCF IS CONVERGED#\nEtot_without_rpa(Ha): -310\n"):
            with self.subTest(log=text), self.assertRaises(ValueError):
                self.collect(text)
            self.assertFalse((self.output / "PBE_ENDPOINT_COLLECTION.json").exists())

    def test_log_parses_scientific_ev_and_last_supported_band_counts(self):
        text = (
            " Occupied electronic states = 3\n NBANDS = 20\n"
            " Occupied electronic states 4\n Number of electronic states (NBANDS) 44\n"
            " #SCF IS CONVERGED#\n !FINAL_ETOT_IS {:.16e} eV\n"
        ).format(BASELINE)
        parsed = endpoint._scf_log(text.encode("ascii"))
        self.assertEqual(parsed["energy_ev"], BASELINE)
        self.assertEqual(parsed["band_counts"], {"occupied": 4, "total": 44})
        self.assertEqual(parsed["energy_record"], "!FINAL_ETOT_IS")

    def test_collect_uses_final_etot_not_rpa_e0_and_preserves_logs(self):
        self.prepare()
        text = log_text() + " Etot_without_rpa(Ha): -900\n RPA E0 -1800\n"
        result = self.collect(text)
        candidate = self.output / "OUT.C_Q1/running_scf.log"
        self.assertEqual(candidate.read_text(), text)
        self.assertEqual(self.baseline_log.read_text(), log_text())
        self.assertEqual(result["candidate_energy_ev"], BASELINE)
        self.assertEqual(result["candidate_log_sha256"], digest(candidate))

    def test_collect_band_counts_are_optional_but_inconsistent_counts_reject(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.collect(log_text().replace("NBANDS) = 44", "NBANDS) = 40"))
        with self.assertRaises(ValueError):
            self.collect(log_text().replace("states = 4", "states = 5"))
        result = self.collect(log_text(counts=False))
        self.assertEqual(result["band_counts"], {"occupied": None, "total": None})
        self.assertEqual(result["band_count_check"], "unavailable")

    def test_collect_rejects_modified_preparation_or_staged_inputs(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.collect(log_text(), preparation_sha256="0" * 64)
        for name in ("INPUT", "STRU", "KPT", "C.upf", "C_original.orb",
                     ".provenance/original_running_scf.log"):
            path = self.output / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"modified")
            with self.subTest(file=name), self.assertRaisesRegex(ValueError, "SHA256"):
                self.collect(log_text())
            path.write_bytes(saved)

    def test_cli_prepare_and_collect(self):
        command = [sys.executable, str(MODULE), "prepare", "--optimizer-result", str(self.result),
                   "--optimizer-result-sha256", digest(self.result), "--original-q1", str(self.original),
                   "--baseline-log-sha256", digest(self.baseline_log), "--output", str(self.output)]
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        prepared = json.loads(completed.stdout)
        self.assertEqual(prepared["preparation_sha256"], digest(self.output / "PBE_ENDPOINT_PREPARED.json"))
        out = self.output / "OUT.C_Q1"
        out.mkdir()
        (out / "running_scf.log").write_text(log_text())
        completed = subprocess.run(
            [sys.executable, str(MODULE), "collect", "--prepared-dir", str(self.output),
             "--preparation-sha256", prepared["preparation_sha256"]],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
        self.assertEqual(json.loads(completed.stdout)["pbe_total_energy_gate"], "pass")


if __name__ == "__main__":
    unittest.main()
