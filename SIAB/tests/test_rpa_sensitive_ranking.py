import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "example_H_sternheimer"
    / "projected_pi_loss"
    / "analyze_rpa_sensitive_ranking.py"
)
BASIS_NU = {
    "two_d": (3, 2, 2, 0, 0),
    "first_f": (3, 2, 2, 1, 0),
    "first_g": (3, 2, 2, 1, 1),
    "second_f": (3, 2, 2, 2, 1),
    "second_g": (3, 2, 2, 1, 2),
}
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)
FAMILIES = ("H", "H2")
RADIAL_SIZE = 25


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    return {
        "abacus_commit": "1" * 40,
        "auxiliary_basis_sha256": "a" * 64,
        "cell_bohr": [
            20.0,
            0.0,
            0.0,
            0.0,
            20.0,
            0.0,
            0.0,
            0.0,
            20.0,
        ],
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


def primitive_blocks():
    blocks = []
    offsets = {}
    offset = 0
    for l_value in range(5):
        for m_value in range(-l_value, l_value + 1):
            blocks.append(
                f"H 0 {l_value} {m_value} {RADIAL_SIZE} {offset}\n"
            )
            offsets[(l_value, m_value)] = offset
            offset += RADIAL_SIZE
    return "".join(blocks), offsets, offset


def write_pair(directory, prefix, q_scale):
    blocks, offsets, n_primitive = primitive_blocks()
    overlap = [
        complex(1.0 if row == column else 0.0, 0.0)
        for row in range(n_primitive)
        for column in range(n_primitive)
    ]
    support = (
        (offsets[(0, 0)], (0.12, 0.03)),
        (offsets[(3, 0)], (0.05, 0.08)),
        (offsets[(4, 0)], (0.07, 0.04)),
    )
    d = [[0.0] * n_primitive for _ in range(2)]
    for index, channel_values in support:
        for channel, value in enumerate(channel_values):
            d[channel][index] = value
    q = []
    for frequency_scale in (0.8, 0.35):
        for channel in range(2):
            q.append(
                [
                    q_scale * frequency_scale * value
                    for value in d[channel]
                ]
            )
    provenance_json = json.dumps(provenance(), separators=(",", ":"))
    source = directory / f"{prefix}_source.dat"
    response = directory / f"{prefix}_response.dat"
    source.write_text(
        "<STERNHEIMER_SIAB_SOURCE_HEADER>\n"
        "format_version 1\n"
        "n_source 2\n"
        f"n_primitive {n_primitive}\n"
        "n_blocks 25\n"
        "grid_volume_bohr3 0.125\n"
        "</STERNHEIMER_SIAB_SOURCE_HEADER>\n"
        "<PRIMITIVE_BLOCKS>\n"
        + blocks
        + "</PRIMITIVE_BLOCKS>\n"
        "<SOURCE_METADATA>\n"
        "0 0 1.0 1.0\n"
        "0 1 1.0 1.0\n"
        "</SOURCE_METADATA>\n"
        "<OVERLAP_D>\n"
        + complex_rows(value for row in d for value in row)
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
        f"n_primitive {n_primitive}\n"
        "n_blocks 25\n"
        "grid_volume_bohr3 0.125\n"
        "</STERNHEIMER_SIAB_HEADER>\n"
        "<PRIMITIVE_BLOCKS>\n"
        + blocks
        + "</PRIMITIVE_BLOCKS>\n"
        "<REFERENCE_METADATA>\n"
        "0 0 0.5 1.0 0.4 1.0\n"
        "0 1 0.5 1.0 0.4 1.0\n"
        "0 0 1.5 1.0 0.6 1.0\n"
        "0 1 1.5 1.0 0.6 1.0\n"
        "</REFERENCE_METADATA>\n"
        "<OVERLAP_Q>\n"
        + complex_rows(value for row in q for value in row)
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


def write_coefficients(path, nu, good=True, first_f_gate=True):
    columns = []
    for l_value, count in enumerate(nu):
        for zeta in range(count):
            pivot = zeta if good else 10 + zeta
            if l_value == 3 and not first_f_gate:
                pivot = 10 + zeta
            values = [
                1.0 if index == pivot else 0.0
                for index in range(RADIAL_SIZE)
            ]
            columns.append(
                "\tType\tL\tZeta-Orbital\n"
                f"\t  H \t{l_value}\t    {zeta + 1}\n"
                + "".join(f"\t {value:.14f}\n" for value in values)
            )
    path.write_text(
        "<Coefficient>\n"
        f"\t {sum(nu)} Total number of radial orbitals.\n"
        + "".join(columns)
        + "</Coefficient>\n<Mkb>\nLeft spillage = 0.0\n</Mkb>\n",
        encoding="utf-8",
    )


def write_audit(path, case):
    checks = {
        "abacus_finish_marker": True,
        "charge_grid_exact": True,
        "final_total_energy_le_1e_12_ha": True,
        "nbands_exact": True,
        "new_scf_complete": True,
        "occupations_le_1e_14": True,
        "occupied_eigenvalues_le_1e_12_ha": True,
        "occupied_state_count_exact": True,
        "old_scf_complete": True,
        "wavefunction_grid_exact": True,
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
        "pass": True,
        "running_log_comparison": {
            "charge_grid": [180, 180, 180],
            "final_total_energy_abs_diff_ha": 1.0e-15,
            "finish_marker": True,
            "nbands": 9 if case == "H" else 18,
            "scf_complete": True,
            "wavefunction_grid": [180, 180, 180],
        },
        "thresholds": {
            "final_total_energy_abs_diff_ha": 1.0e-12,
            "occupation_abs_diff": 1.0e-14,
            "occupied_eigenvalue_abs_diff_ha": 1.0e-12,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class RpaSensitiveRankingTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            SCRIPT.is_file(),
            f"Task 6 analyzer script is absent: {SCRIPT}",
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.h_response, self.h_source = write_pair(self.root, "H", 1.0)
        self.h2_response, self.h2_source = write_pair(self.root, "H2", 1.15)
        self.h_audit = self.root / "H_audit.json"
        self.h2_audit = self.root / "H2_audit.json"
        write_audit(self.h_audit, "H")
        write_audit(self.h2_audit, "H2")
        self.coefficients = {}
        for name, nu in BASIS_NU.items():
            path = self.root / f"{name}.txt"
            write_coefficients(path, nu, good=not name.startswith("second_"))
            self.coefficients[name] = path

    def command(self, output, first_f=None):
        coefficient_paths = dict(self.coefficients)
        if first_f is not None:
            coefficient_paths["first_f"] = first_f
        command = [
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
        ]
        for name in BASIS_NU:
            command.extend((f"--{name.replace('_', '-')}", str(coefficient_paths[name])))
        command.extend(("--output-dir", str(output)))
        return command

    def test_selects_largest_of_multiple_admissible_alphas_and_writes_complete_outputs(self):
        output = self.root / "pass"
        completed = subprocess.run(
            self.command(output),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        expected_files = {
            "rpa_sensitive_ranking.json",
            "rpa_sensitive_ranking.md",
            "rpa_sensitive_frequency.pdf",
            "rpa_sensitive_frequency.png",
        }
        self.assertEqual({path.name for path in output.iterdir()}, expected_files)
        self.assertFalse(list(output.glob("*.tmp*")))
        for plot_name in ("rpa_sensitive_frequency.pdf", "rpa_sensitive_frequency.png"):
            self.assertGreater((output / plot_name).stat().st_size, 1000)

        payload = json.loads(
            (output / "rpa_sensitive_ranking.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["decision"], "pass")
        self.assertEqual(payload["basis_nu"], {name: list(nu) for name, nu in BASIS_NU.items()})
        self.assertEqual(payload["alphas"], list(ALPHAS))
        self.assertGreater(len(payload["admissible_alphas"]), 1)
        self.assertEqual(payload["selected_alpha"], max(payload["admissible_alphas"]))
        self.assertEqual(payload["selected_alpha"], max(ALPHAS))
        self.assertFalse(payload["uses_sos_energy_as_numeric_input"])
        self.assertFalse(payload["uses_ghost_family"])
        self.assertFalse(payload["new_candidate_was_evaluated"])

        self.assertEqual(len(payload["input_sha256"]), 11)
        self.assertEqual(payload["input_sha256"]["H_audit"], sha256(self.h_audit))
        self.assertEqual(payload["input_sha256"]["first_g"], sha256(self.coefficients["first_g"]))
        self.assertEqual(set(payload["zero_order_audits"]), set(FAMILIES))
        for family in FAMILIES:
            audit = payload["zero_order_audits"][family]
            self.assertEqual(audit["case"], family)
            self.assertTrue(audit["passed"])
            self.assertEqual(len(audit["source_file_sha256"]), 4)

        self.assertEqual(len(payload["alpha_results"]), len(ALPHAS))
        for alpha, alpha_result in zip(ALPHAS, payload["alpha_results"]):
            self.assertEqual(alpha_result["alpha"], alpha)
            self.assertEqual(
                set(alpha_result["gates"]),
                {
                    "first_f_improves_two_d",
                    "first_g_improves_first_f",
                    "second_f_not_better",
                    "second_g_not_better",
                },
            )
            self.assertEqual(
                alpha_result["admissible"],
                all(alpha_result["gates"].values()),
            )
            self.assertEqual(set(alpha_result["bases"]), set(BASIS_NU))
            for basis in alpha_result["bases"].values():
                self.assertTrue(math.isfinite(basis["loss"]))
                self.assertTrue(math.isfinite(basis["max_condition"]))
                self.assertEqual(set(basis["families"]), set(FAMILIES))
                for family in basis["families"].values():
                    scalar_fields = (
                        "loss",
                        "base_loss",
                        "sensitivity_loss",
                        "max_candidate_condition",
                    )
                    array_fields = (
                        "frequency_ha",
                        "frequency_weight",
                        "frequency_loss",
                        "frequency_base_loss",
                        "frequency_sensitivity_loss",
                        "trace_log_difference",
                        "minimum_reference_dielectric_eigenvalue",
                        "minimum_candidate_dielectric_eigenvalue",
                    )
                    for field in scalar_fields:
                        self.assertTrue(math.isfinite(family[field]))
                    lengths = {len(family[field]) for field in array_fields}
                    self.assertEqual(lengths, {2})
                    for field in array_fields:
                        self.assertTrue(all(math.isfinite(value) for value in family[field]))
                    self.assertTrue(
                        all(
                            value > 0.0
                            for value in family["minimum_reference_dielectric_eigenvalue"]
                        )
                    )
                    self.assertTrue(
                        all(
                            value > 0.0
                            for value in family["minimum_candidate_dielectric_eigenvalue"]
                        )
                    )

        markdown = (output / "rpa_sensitive_ranking.md").read_text(encoding="utf-8")
        for text in (
            "Decision: `pass`",
            "Selected alpha: `1.0`",
            "Base loss",
            "Sensitivity loss",
            *BASIS_NU,
        ):
            self.assertIn(text, markdown)

    def test_nonadmissible_fixture_exits_two_and_reports_galerkin_stop(self):
        failed_first_f = self.root / "first_f_failed_gate.txt"
        write_coefficients(
            failed_first_f,
            BASIS_NU["first_f"],
            good=True,
            first_f_gate=False,
        )
        output = self.root / "stop"
        completed = subprocess.run(
            self.command(output, first_f=failed_first_f),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        payload = json.loads(
            (output / "rpa_sensitive_ranking.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["decision"], "stop_galerkin_required")
        self.assertIsNone(payload["selected_alpha"])
        self.assertEqual(payload["admissible_alphas"], [])
        self.assertFalse(
            any(result["admissible"] for result in payload["alpha_results"])
        )
        self.assertFalse(list(output.glob("*.tmp*")))


if __name__ == "__main__":
    unittest.main()
