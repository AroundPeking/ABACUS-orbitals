import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "ordinary_sos_validation"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NamedCandidatePreparationTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module(
            "prepare_named_periodic_candidate",
            ROOT / "prepare_named_periodic_candidate.py",
        )

    def test_prepares_two_g_manifest_with_physics_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coefficients = root / "coefficients.txt"
            coefficients.write_text("coefficients\n", encoding="ascii")
            orbital = root / "candidate.orb"
            orbital.write_text(
                "Energy Cutoff(Ry)           100.0\n"
                "Radius Cutoff(a.u.)         10.0\n"
                "Lmax                        4\n"
                "Number of Gorbital-->       2\n",
                encoding="ascii",
            )
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "datasets": [{"selected_iq": 43}],
                        "candidates": [
                            {
                                "label": "joint-two-g",
                                "nu": [3, 3, 2, 1, 2],
                                "ao_count_cell": 94,
                                "coefficients": str(coefficients),
                                "coefficients_sha256": hashlib.sha256(
                                    coefficients.read_bytes()
                                ).hexdigest(),
                                "minimum_occupied_capture": 0.999997,
                                "maximum_overlap_condition": 2.3e7,
                                "global_weighted_relative_trace_log_error": 0.025,
                                "global_weighted_relative_pi_error": 0.091,
                            }
                        ],
                    }
                ),
                encoding="ascii",
            )
            spectrum = root / "spectrum.json"
            spectrum.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "maximum_overlap_condition": 2.3e7,
                        "maximum_eigenvalue_ev": 318.0,
                    }
                ),
                encoding="ascii",
            )
            output = root / "candidate"

            result = self.module.prepare_candidate(
                comparison_path=comparison,
                spectrum_path=spectrum,
                orbital_path=orbital,
                output_directory=output,
                label="joint-two-g",
                occupied_capture_floor=0.999898,
                reference_overlap_condition=2.9e6,
                reference_maximum_eigenvalue_ev=301.0,
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["ao_count_atom"], 47)
            self.assertEqual(result["fixed_nu"], [2, 2, 1, 0, 0])
            self.assertEqual(result["pre_sos_gate"], "pass")
            self.assertEqual(
                hashlib.sha256((output / "C_gga_10au_100Ry_joint_two_g.orb").read_bytes()).hexdigest(),
                result["exported_orbital_sha256"],
            )

    def test_rejects_candidate_above_overlap_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coefficients = root / "coefficients.txt"
            coefficients.write_text("coefficients\n", encoding="ascii")
            orbital = root / "candidate.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "datasets": [{"selected_iq": 43}],
                        "candidates": [
                            {
                                "label": "joint-two-g",
                                "nu": [3, 3, 2, 1, 2],
                                "ao_count_cell": 94,
                                "coefficients": str(coefficients),
                                "coefficients_sha256": hashlib.sha256(
                                    coefficients.read_bytes()
                                ).hexdigest(),
                                "minimum_occupied_capture": 1.0,
                                "maximum_overlap_condition": 3.1e7,
                                "global_weighted_relative_trace_log_error": 0.02,
                                "global_weighted_relative_pi_error": 0.08,
                            }
                        ],
                    }
                ),
                encoding="ascii",
            )
            spectrum = root / "spectrum.json"
            spectrum.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "maximum_overlap_condition": 3.1e7,
                        "maximum_eigenvalue_ev": 310.0,
                    }
                ),
                encoding="ascii",
            )

            with self.assertRaisesRegex(ValueError, "pre-SOS physics gate"):
                self.module.prepare_candidate(
                    comparison_path=comparison,
                    spectrum_path=spectrum,
                    orbital_path=orbital,
                    output_directory=root / "candidate",
                    label="joint-two-g",
                    occupied_capture_floor=0.999898,
                    reference_overlap_condition=2.9e6,
                    reference_maximum_eigenvalue_ev=301.0,
                )


class SosOnlyBindingCollectorTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module(
            "collect_named_candidate_sos_binding",
            VALIDATION / "collect_named_candidate_sos_binding.py",
        )

    def test_collects_atom_solid_binding_against_fixed_delta_reference(self):
        atom = {
            "status": "success",
            "side": "atom",
            "method": "sos",
            "scope": "body_only_no_analytic_headwing",
            "coulomb_kernel": "full_periodic_poisson",
            "selected_orbital_sha256": "a" * 64,
            "frequency_grid_sha256": "b" * 64,
            "naux": "200",
            "reference_ha": "-5.0",
            "ecrpa_ha": "-0.1",
        }
        solid = {
            **atom,
            "side": "solid",
            "frequency_grid_sha256": "c" * 64,
            "naux": "400",
            "reference_ha": "-10.3",
            "ecrpa_ha": "-0.4",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atom_path = root / "atom.txt"
            solid_path = root / "solid.txt"
            for path, values in ((atom_path, atom), (solid_path, solid)):
                path.write_text(
                    "".join(f"{key} {value}\n" for key, value in values.items()),
                    encoding="ascii",
                )
            result = self.module.collect_binding_energy(
                atom_sos=atom_path,
                solid_sos=solid_path,
                delta_reference_ev_per_c=6.902326,
            )

        self.assertAlmostEqual(result["zero_order_binding_ev_per_c"], 4.0817079368982)
        self.assertAlmostEqual(result["correlation_binding_ev_per_c"], 2.7211386245988)
        self.assertAlmostEqual(result["sos_total_binding_ev_per_c"], 6.802846561497)
        self.assertAlmostEqual(result["difference_from_delta_ev_per_c"], -0.099479438503)
        self.assertEqual(result["basis_full_body_gate"], "fail")


class NamedCandidateSlurmContractTest(unittest.TestCase):
    def test_chain_is_sos_only_all_band_and_duplicate_safe(self):
        names = {
            "run_named_candidate_atom_producer_55d25e3c9.slurm": (
                "set_input_key nbands \"$ao_count_atom\"",
                "set_input_key exx_pca_threshold 1e-4",
                "set_input_key rpa_pca_fixed_nu \"$fixed_nu\"",
            ),
            "run_named_candidate_atom_sos_d4810f73.slurm": (
                "task = rpa",
                "nfreq = 6",
                "prefix_coul_full = v1_coulomb_full_iq_",
            ),
            "run_named_candidate_solid_full_bz_55d25e3c9.slurm": (
                "#SBATCH --array=1-64%8",
                "set_input_key nbands \"$ao_count_cell\"",
                "exact_rhs_full_periodic_poisson",
            ),
            "run_named_candidate_solid_sos_d4810f73.slurm": (
                "for ordinal in $(seq 1 64)",
                "n_bands_chi0",
                "v1_coulomb_grid_iq_",
            ),
        }
        for name, required in names.items():
            text = (VALIDATION / name).read_text(encoding="ascii")
            for marker in required:
                self.assertIn(marker, text, f"{name}: missing {marker}")
            self.assertNotIn("sternheimer_rpa", text)
            self.assertNotIn("matched_delta", text)

        release = (VALIDATION / "release_named_candidate_sos_validation.sh").read_text(
            encoding="ascii"
        )
        self.assertIn("flock", release)
        self.assertIn("sbatch --test-only", release)
        self.assertIn("afterok", release)
        self.assertIn("refusing duplicate", release)
        self.assertNotIn("delta", release.lower())

        binding = (VALIDATION / "run_named_candidate_binding_collect.slurm").read_text(
            encoding="ascii"
        )
        self.assertIn("python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python", binding)
        self.assertIn('"$python" "$collector"', binding)
        self.assertIn('"$python" - "$work/RESULT.json"', binding)

    def test_atom_restart_recovery_preserves_the_physical_target(self):
        recovery = (
            VALIDATION / "run_named_candidate_atom_restart_recovery_55d25e3c9.slurm"
        ).read_text(encoding="ascii")

        for marker in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=40",
            "wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube",
            "set_input_key init_wfc file",
            "set_input_key init_chg file",
            "set_input_key nelec 4",
            "set_input_key nspin 2",
            "set_input_key nupdown 2",
            'set_input_key ocp_set "3*1 44*0 1*1 46*0"',
            "set_input_key mixing_beta 0.3",
            "set_input_key mixing_beta_mag 0.3",
            "set_input_key rpa 1",
            "#SCF IS CONVERGED#",
            "initial-nonconverged-snapshot",
            "status=success",
        ):
            self.assertIn(marker, recovery)

        self.assertNotIn("sternheimer", recovery.lower())
        self.assertNotIn("rm -rf", recovery)


if __name__ == "__main__":
    unittest.main()
