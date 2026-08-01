import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "example_H_sternheimer"
    / "projected_pi_loss"
    / "analyze_projected_pi.py"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    return {
        "abacus_commit": "1" * 40,
        "auxiliary_basis_sha256": "a" * 64,
        "cell_bohr": [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0],
        "ecut_ry": 50.0,
        "kernel": "full_coulomb",
        "orbital_sha256": "b" * 64,
        "pseudopotential_sha256": "c" * 64,
        "spin_convention": "occupation_in_metadata",
        "executable_sha256": "d" * 64,
        "exx_pca_thr": 1.0e-4,
        "auxiliary_whitening": "global_full_coulomb_v1",
        "raw_auxiliary_dimension": 2,
        "whitened_auxiliary_rank": 2,
        "discarded_auxiliary_rank": 0,
        "coulomb_relative_threshold": 1.0e-10,
        "coulomb_transform_sha256": "e" * 64,
        "mpi_ranks": 2,
        "omp_threads": 30,
    }


def complex_rows(values):
    return "".join(f"{value.real:.17g} {value.imag:.17g}\n" for value in values)


def write_pair(directory, prefix):
    directory.mkdir(parents=True, exist_ok=True)
    nprimitive = 100
    blocks = (
        "H 0 0 0 25 0\n"
        "H 0 1 -1 25 25\n"
        "H 0 1 0 25 50\n"
        "H 0 1 1 25 75\n"
    )
    overlap = [
        complex(1.0 if row == column else 0.0, 0.0)
        for row in range(nprimitive)
        for column in range(nprimitive)
    ]
    d = [
        complex(
            (channel + 1) * ((primitive % 7) + 1) / 100.0,
            ((primitive % 5) - 2) / 200.0,
        )
        for channel in range(2)
        for primitive in range(nprimitive)
    ]
    q = [
        complex(
            (frequency + 1) * ((primitive % 11) + 1) / 120.0,
            (channel + 1) * ((primitive % 3) - 1) / 90.0,
        )
        for frequency in range(2)
        for channel in range(2)
        for primitive in range(nprimitive)
    ]
    provenance_json = json.dumps(provenance(), separators=(",", ":"))
    source = directory / f"{prefix}_source.dat"
    response = directory / f"{prefix}_response.dat"
    source.write_text(
        "<STERNHEIMER_SIAB_SOURCE_HEADER>\n"
        "format_version 1\n"
        "n_source 2\n"
        "n_primitive 100\n"
        "n_blocks 4\n"
        "grid_volume_bohr3 0.125\n"
        "</STERNHEIMER_SIAB_SOURCE_HEADER>\n"
        "<PRIMITIVE_BLOCKS>\n"
        + blocks
        + "</PRIMITIVE_BLOCKS>\n"
        "<SOURCE_METADATA>\n"
        "0 0 2.0 1.0\n"
        "0 1 2.0 1.0\n"
        "</SOURCE_METADATA>\n"
        "<OVERLAP_D>\n"
        + complex_rows(d)
        + "</OVERLAP_D>\n"
        "<OVERLAP_S>\n"
        + complex_rows(overlap)
        + "</OVERLAP_S>\n"
        "<PROVENANCE_JSON>\n"
        + provenance_json
        + "\n</PROVENANCE_JSON>\n",
        encoding="utf-8",
    )
    response.write_text(
        "<STERNHEIMER_SIAB_HEADER>\n"
        "format_version 1\n"
        "n_reference 4\n"
        "n_primitive 100\n"
        "n_blocks 4\n"
        "grid_volume_bohr3 0.125\n"
        "</STERNHEIMER_SIAB_HEADER>\n"
        "<PRIMITIVE_BLOCKS>\n"
        + blocks
        + "</PRIMITIVE_BLOCKS>\n"
        "<REFERENCE_METADATA>\n"
        "0 0 0.5 2.0 0.3 1.0\n"
        "0 1 0.5 2.0 0.3 1.0\n"
        "0 0 1.5 2.0 0.7 1.0\n"
        "0 1 1.5 2.0 0.7 1.0\n"
        "</REFERENCE_METADATA>\n"
        "<OVERLAP_Q>\n"
        + complex_rows(q)
        + "</OVERLAP_Q>\n"
        "<OVERLAP_S>\n"
        + complex_rows(overlap)
        + "</OVERLAP_S>\n"
        "<PROVENANCE_JSON>\n"
        + provenance_json
        + "\n</PROVENANCE_JSON>\n",
        encoding="utf-8",
    )
    return response, source


def write_coefficients(path):
    columns = []
    for l, zeta, pivot in (
        (0, 1, 0),
        (0, 2, 1),
        (0, 3, 2),
        (1, 1, 0),
        (1, 2, 1),
    ):
        values = [1.0 if index == pivot else 0.0 for index in range(25)]
        columns.append(
            "\tType\tL\tZeta-Orbital\n"
            f"\t  H \t{l}\t    {zeta}\n"
            + "".join(f"\t {value:.14f}\n" for value in values)
        )
    path.write_text(
        "<Coefficient>\n"
        "\t 5 Total number of radial orbitals.\n"
        + "".join(columns)
        + "</Coefficient>\n<Mkb>\nLeft spillage = 0.0\n</Mkb>\n",
        encoding="utf-8",
    )


def write_audit(path, case, passed=True):
    checks = {
        "abacus_finish_marker": passed,
        "charge_grid_exact": passed,
        "final_total_energy_le_1e_12_ha": passed,
        "nbands_exact": passed,
        "new_scf_complete": passed,
        "occupations_le_1e_14": passed,
        "occupied_eigenvalues_le_1e_12_ha": passed,
        "occupied_state_count_exact": passed,
        "old_scf_complete": passed,
        "wavefunction_grid_exact": passed,
    }
    payload = {
        "case": case,
        "checks": checks,
        "eig_occ_comparison": {
            "max_occupation_abs_diff": 0.0,
            "max_occupied_eigenvalue_abs_diff_ha": 1.0e-14,
            "occupied_state_count": 1,
        },
        "files": {
            name: {"path": f"/{case}/{name}", "sha256": digit * 64}
            for name, digit in (
                ("new_eig_occ", "1"),
                ("new_running_scf_log", "2"),
                ("old_eig_occ", "3"),
                ("old_running_scf_log", "4"),
            )
        },
        "format": "sternheimer_siab_zero_order_identity_v1",
        "pass": passed,
        "running_log_comparison": {
            "charge_grid": [180, 180, 180],
            "final_total_energy_abs_diff_ha": 1.0e-15,
            "finish_marker": passed,
            "nbands": 9 if case == "H" else 18,
            "scf_complete": passed,
            "wavefunction_grid": [180, 180, 180],
        },
        "thresholds": {
            "final_total_energy_abs_diff_ha": 1.0e-12,
            "occupation_abs_diff": 1.0e-14,
            "occupied_eigenvalue_abs_diff_ha": 1.0e-12,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class ProjectedPiAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.h_response, self.h_source = write_pair(self.root, "H")
        self.h2_response, self.h2_source = write_pair(self.root, "H2")
        self.h_audit = self.root / "H_audit.json"
        self.h2_audit = self.root / "H2_audit.json"
        write_audit(self.h_audit, "H")
        write_audit(self.h2_audit, "H2")
        self.coefficients = []
        for name in ("initial", "joint", "guarded"):
            path = self.root / f"{name}.txt"
            write_coefficients(path)
            self.coefficients.append(path)

    def command(self, output):
        initial, joint, guarded = self.coefficients
        return [
            sys.executable,
            str(SCRIPT),
            "--h-response",
            str(self.h_response),
            "--h-source",
            str(self.h_source),
            "--h-audit",
            str(self.h_audit),
            "--h2-response",
            str(self.h2_response),
            "--h2-source",
            str(self.h2_source),
            "--h2-audit",
            str(self.h2_audit),
            "--initial",
            str(initial),
            "--joint",
            str(joint),
            "--guarded",
            str(guarded),
            "--output-dir",
            str(output),
        ]

    def test_failed_ranking_writes_complete_diagnostics_and_exits_two(self):
        output = self.root / "output"
        completed = subprocess.run(
            self.command(output),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        expected = {
            "projected_pi_ranking.json",
            "projected_pi_ranking.md",
            "projected_pi_frequency.pdf",
            "projected_pi_frequency.png",
        }
        self.assertEqual({path.name for path in output.iterdir()}, expected)
        payload = json.loads(
            (output / "projected_pi_ranking.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["decision"], "stop_galerkin_required")
        self.assertEqual(payload["thresholds"], [1.0e-10, 1.0e-11, 1.0e-12])
        self.assertEqual(payload["reader_warnings"], {"H": [], "H2": []})
        self.assertEqual(payload["zero_order_audit_sha256"]["H"], sha256(self.h_audit))
        self.assertEqual(len(payload["input_sha256"]), 9)
        self.assertEqual(
            set(payload["results"]), {"1e-10", "1e-11", "1e-12"}
        )
        for threshold_result in payload["results"].values():
            self.assertEqual(
                set(threshold_result),
                {"initial_tzdp", "fixed_dzp_joint", "low_frequency_guarded"},
            )
            for result in threshold_result.values():
                self.assertIn("total_loss", result)
                self.assertEqual(set(result["families"]), {"H", "H2"})
                self.assertIn("reference_rank", result["families"]["H"])
                self.assertIn("frequency_loss", result["families"]["H"])
        self.assertFalse(payload["gates"]["joint_improves_initial"])
        self.assertFalse(payload["gates"]["guarded_improves_initial"])
        self.assertFalse(list(output.glob("*.tmp")))

    def test_rejects_failed_zero_order_audit(self):
        write_audit(self.h_audit, "H", passed=False)
        output = self.root / "bad_audit"
        completed = subprocess.run(
            self.command(output),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("zero-order audit H did not pass", completed.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
