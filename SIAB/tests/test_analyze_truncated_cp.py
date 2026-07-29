from pathlib import Path
import sys
import tempfile
import unittest


ANALYZER_DIR = (
    Path(__file__).resolve().parents[1]
    / "example_H_sternheimer/held_out_h2_sos_greedy_full"
)
if str(ANALYZER_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYZER_DIR))

from analyze_truncated_cp import (  # noqa: E402
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_MOL,
    combine_counterpoise,
    parse_abacus_energy,
    parse_librpa_ec,
    parse_unique_float,
    rewrite_truncated_ghost_input,
)


class TruncatedCounterpoiseParserTest(unittest.TestCase):
    def write_file(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_parse_unique_float_requires_exactly_one_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = self.write_file(directory, "missing.out", "unrelated\n")
            duplicate = self.write_file(
                directory,
                "duplicate.out",
                "Total EcRPA: -0.1\nTotal EcRPA: -0.2\n",
            )

            with self.assertRaisesRegex(ValueError, "expected exactly one"):
                parse_unique_float(
                    missing,
                    r"Total EcRPA:\s+(?P<value>[-+0-9.eE]+)",
                    "Total EcRPA",
                )
            with self.assertRaisesRegex(ValueError, "expected exactly one"):
                parse_unique_float(
                    duplicate,
                    r"Total EcRPA:\s+(?P<value>[-+0-9.eE]+)",
                    "Total EcRPA",
                )

    def test_rewrite_ghost_input_preserves_header_and_only_truncates_bands(self):
        source_text = """INPUT_PARAMETERS
suffix old_suffix
nbands 334
nspin 2
nupdown 1
nelec 1
ecutwfc 100
rpa 1
out_librpa_reader_version 1
exx_pca_threshold 10
exx_singularity_correction massidda
rpa_ccp_rmesh_times 5
"""
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_file(directory, "INPUT.source", source_text)
            output = Path(directory) / "INPUT"

            rewrite_truncated_ghost_input(source, output, "truncated_suffix")

            actual = output.read_text(encoding="utf-8")
            self.assertTrue(actual.startswith("INPUT_PARAMETERS\n"))
            self.assertIn("suffix                  truncated_suffix\n", actual)
            self.assertIn("nbands                  160\n", actual)
            self.assertEqual(
                actual.replace(
                    "suffix                  truncated_suffix",
                    "suffix old_suffix",
                ).replace("nbands                  160", "nbands 334"),
                source_text,
            )

    def test_parse_abacus_energy_checks_scf_and_zero_order_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = self.write_file(
                directory,
                "abacus.out",
                "rpa_lcao_exx(Ha): -0.300000000000000\n"
                "etxc(Ha): -0.400000000000000\n"
                "etot(Ha): -1.000000000000000\n"
                "Etot_without_rpa(Ha): -0.900000000000000\n",
            )
            running = self.write_file(
                directory,
                "running_scf.log",
                " !FINAL_ETOT_IS -27.211396000000000 eV\n",
            )

            result = parse_abacus_energy(stdout, running)

        self.assertEqual(
            result,
            {
                "dft_total_ha": -1.0,
                "xc_ha": -0.4,
                "exx_ha": -0.3,
                "zero_order_ha": -0.9,
            },
        )

    def test_parse_abacus_energy_rejects_inconsistent_zero_order(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = self.write_file(
                directory,
                "abacus.out",
                "rpa_lcao_exx(Ha): -0.300000000000000\n"
                "etxc(Ha): -0.400000000000000\n"
                "etot(Ha): -1.000000000000000\n"
                "Etot_without_rpa(Ha): -0.800000000000000\n",
            )
            running = self.write_file(
                directory,
                "running_scf.log",
                f" !FINAL_ETOT_IS {-1.0 * HARTREE_TO_EV:.15f} eV\n",
            )

            with self.assertRaisesRegex(ValueError, "zero-order identity"):
                parse_abacus_energy(stdout, running)

    def test_parse_librpa_ec_reads_the_unique_total(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "librpa.out",
                "header\n| Total EcRPA:       -0.078351140\nfooter\n",
            )
            self.assertEqual(parse_librpa_ec(path), -0.078351140)


class TruncatedCounterpoiseCombinationTest(unittest.TestCase):
    def test_combines_raw_cp_and_bsse_by_energy_component(self):
        h2 = {"zero_order_ha": -2.0, "ec_ha": -0.20}
        isolated_h = {"zero_order_ha": -0.90, "ec_ha": -0.05}
        ghost = {"zero_order_ha": -0.95, "ec_ha": -0.08}

        result = combine_counterpoise(h2, isolated_h, ghost)

        expected_ha = {
            "raw_zero_order": 0.20,
            "raw_correlation": 0.10,
            "raw_total": 0.30,
            "cp_zero_order": 0.10,
            "cp_correlation": 0.04,
            "cp_total": 0.14,
            "bsse_zero_order": 0.10,
            "bsse_correlation": 0.06,
            "bsse_total": 0.16,
        }
        for key, expected in expected_ha.items():
            self.assertAlmostEqual(result[f"{key}_ha"], expected, places=14)
            self.assertAlmostEqual(
                result[f"{key}_kcal_mol"],
                expected * HARTREE_TO_KCAL_MOL,
                places=11,
            )


class TruncatedGhostContractTest(unittest.TestCase):
    def test_uses_physical_spin_full_node_and_only_truncates_bands(self):
        script_path = ANALYZER_DIR / "run_ghost_truncated_cp.slurm"
        self.assertTrue(script_path.is_file(), script_path)
        script = script_path.read_text(encoding="utf-8")

        required = (
            "#SBATCH --partition=normal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=30",
            "#SBATCH --mem=110610M",
            "#SBATCH --time=1-00:00:00",
            "expected_nbands=160",
            "expected_spins=2",
            "prepare-ghost-input",
            "n_bands_chi0 = 120",
            '"nfreq": "16"',
            "sha256sum -c SOURCE_SHA256SUMS",
            'install -m 0644 "$template_dir/INPUT" "$case_dir/INPUT"',
        )
        for value in required:
            self.assertIn(value, script)

        self.assertNotIn("#SBATCH --partition=debug", script)
        self.assertNotIn('"nspin": "1"', script)
        self.assertNotIn("coulomb_cut", script)
        self.assertIn("v1_coulomb_full_iq_1_rank0.dat", script)

        analyzer = (ANALYZER_DIR / "analyze_truncated_cp.py").read_text(
            encoding="utf-8"
        )
        for value in (
            '"nbands": "334"',
            'expected["nbands"] = "160"',
            '"nspin": "2"',
            '"nupdown": "1"',
            '"rpa_ccp_rmesh_times": "5"',
        ):
            self.assertIn(value, analyzer)


if __name__ == "__main__":
    unittest.main()
