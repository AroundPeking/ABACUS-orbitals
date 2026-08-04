import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


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
EXPECTED_ARTIFACTS = {
    "rpa_sensitive_ranking.json",
    "rpa_sensitive_ranking.md",
    "rpa_sensitive_frequency.pdf",
    "rpa_sensitive_frequency.png",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "decision",
    "basis_nu",
    "alphas",
    "admissible_alphas",
    "selected_alpha",
    "family_power",
    "relative_rank_tolerance",
    "condition_limit",
    "input_sha256",
    "zero_order_audits",
    "reader_warnings",
    "alpha_results",
    "uses_sos_energy_as_numeric_input",
    "uses_ghost_family",
    "new_candidate_was_evaluated",
    "torch_version",
    "python_version",
}
ALPHA_RESULT_KEYS = {"alpha", "admissible", "gates", "bases"}
BASIS_RESULT_KEYS = {
    "loss",
    "max_condition",
    "frequency_ha",
    "frequency_loss",
    "families",
}
FAMILY_RESULT_KEYS = {
    "loss",
    "base_loss",
    "sensitivity_loss",
    "frequency_ha",
    "frequency_weight",
    "frequency_loss",
    "frequency_base_loss",
    "frequency_sensitivity_loss",
    "trace_log_difference",
    "minimum_reference_dielectric_eigenvalue",
    "minimum_candidate_dielectric_eigenvalue",
    "reference_rank",
    "max_candidate_condition",
}
AUDIT_KEYS = {
    "case",
    "passed",
    "occupied_state_count",
    "grid",
    "max_occupation_abs_diff",
    "max_occupied_eigenvalue_abs_diff_ha",
    "final_total_energy_abs_diff_ha",
    "source_file_paths",
    "source_file_sha256",
    "thresholds",
    "audit_file_sha256",
}


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "task6_rpa_sensitive_ranking",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = load_analyzer()


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
            d[channel][index] = q_scale * value
    q = []
    for frequency_scale in (0.8, 0.35):
        for channel in range(2):
            q.append(
                [
                    frequency_scale * value
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


def synthetic_family_result(loss):
    return {
        "loss": loss,
        "base_loss": loss,
        "sensitivity_loss": loss / 2.0,
        "frequency_ha": [0.5, 1.5],
        "frequency_weight": [1.0, 1.0],
        "frequency_loss": [loss, loss / 2.0],
        "frequency_base_loss": [loss, loss / 2.0],
        "frequency_sensitivity_loss": [loss / 2.0, loss / 4.0],
        "trace_log_difference": [loss / 10.0, loss / 20.0],
        "minimum_reference_dielectric_eigenvalue": [0.8, 0.9],
        "minimum_candidate_dielectric_eigenvalue": [0.7, 0.85],
        "reference_rank": 2,
        "max_candidate_condition": 2.0,
    }


def synthetic_alpha_result(alpha=1.0):
    losses = {
        "two_d": 5.0,
        "first_f": 4.0,
        "first_g": 3.0,
        "second_f": 4.0,
        "second_g": 4.5,
    }
    bases = {}
    for basis_name, loss in losses.items():
        bases[basis_name] = {
            "loss": loss,
            "max_condition": 2.0,
            "frequency_ha": [0.5, 1.5],
            "frequency_loss": [loss, loss / 2.0],
            "families": {
                family_name: synthetic_family_result(loss)
                for family_name in FAMILIES
            },
        }
    gates = {
        "first_f_improves_two_d": True,
        "first_g_improves_first_f": True,
        "second_f_not_better": True,
        "second_g_not_better": True,
    }
    return {
        "alpha": alpha,
        "admissible": True,
        "gates": gates,
        "bases": bases,
    }


def synthetic_alpha_results():
    return [synthetic_alpha_result(alpha) for alpha in ALPHAS]


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

    def parsed_arguments(self, output, first_f=None):
        return ANALYZER.parse_arguments(self.command(output, first_f)[2:])

    def replace_option(self, command, option, value):
        option_index = command.index(option)
        command[option_index + 1] = str(value)

    def assert_complete_artifacts(self, output):
        self.assertEqual(
            {path.name for path in output.iterdir()},
            EXPECTED_ARTIFACTS,
        )

    def assert_no_staging(self, output):
        staging_prefix = f".{output.name}.staging-"
        self.assertFalse(
            [
                path
                for path in output.parent.iterdir()
                if path.name.startswith(staging_prefix)
            ]
        )

    def test_each_duplicate_coefficient_option_is_rejected(self):
        for basis_name in BASIS_NU:
            option = f"--{basis_name.replace('_', '-')}"
            output = self.root / f"duplicate_{basis_name}"
            command = self.command(output)
            command.extend((option, str(self.coefficients[basis_name])))
            with self.subTest(option=option):
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertIn(
                    f"coefficient option {option} may be specified only once",
                    completed.stderr,
                )
                self.assertFalse(output.exists())

    def test_missing_coefficient_option_remains_rejected(self):
        output = self.root / "missing"
        command = self.command(output)
        option_index = command.index("--second-g")
        del command[option_index : option_index + 2]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn(
            "the following arguments are required: --second-g",
            completed.stderr,
        )
        self.assertFalse(output.exists())

    def test_family_roles_reject_same_resolved_path_before_publication(self):
        shared_paths = {
            "response": self.h_response,
            "source": self.h_source,
            "audit": self.h_audit,
        }
        for role, shared in shared_paths.items():
            output = self.root / f"same_path_{role}"
            command = self.command(output)
            self.replace_option(command, f"--h-{role}", shared)
            self.replace_option(command, f"--h2-{role}", shared)
            with self.subTest(role=role):
                with mock.patch.object(
                    ANALYZER,
                    "_evaluate",
                    return_value=synthetic_alpha_results(),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"H/H2 family alias for {role}",
                    ):
                        ANALYZER.run(ANALYZER.parse_arguments(command[2:]))
                self.assertFalse(output.exists())

    def test_family_roles_reject_separate_files_with_identical_bytes(self):
        h_paths = {
            "response": self.h_response,
            "source": self.h_source,
            "audit": self.h_audit,
        }
        for role, h_path in h_paths.items():
            h2_path = self.root / f"equal_H2_{role}.dat"
            h2_path.write_bytes(h_path.read_bytes())
            output = self.root / f"equal_content_{role}"
            command = self.command(output)
            self.replace_option(command, f"--h-{role}", h_path)
            self.replace_option(command, f"--h2-{role}", h2_path)
            with self.subTest(role=role):
                with mock.patch.object(
                    ANALYZER,
                    "_evaluate",
                    return_value=synthetic_alpha_results(),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"H/H2 family alias for {role}",
                    ):
                        ANALYZER.run(ANALYZER.parse_arguments(command[2:]))
                self.assertFalse(output.exists())

    def test_selects_largest_of_multiple_admissible_alphas_and_writes_complete_outputs(self):
        output = self.root / "pass"
        completed = subprocess.run(
            self.command(output),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assert_complete_artifacts(output)
        self.assert_no_staging(output)
        for plot_name in ("rpa_sensitive_frequency.pdf", "rpa_sensitive_frequency.png"):
            self.assertGreater((output / plot_name).stat().st_size, 1000)

        payload = json.loads(
            (output / "rpa_sensitive_ranking.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(payload), TOP_LEVEL_KEYS)
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

        self.assertEqual(
            set(payload["input_sha256"]),
            {
                "H_response",
                "H_source",
                "H_audit",
                "H2_response",
                "H2_source",
                "H2_audit",
                *BASIS_NU,
            },
        )
        self.assertEqual(payload["input_sha256"]["H_audit"], sha256(self.h_audit))
        self.assertEqual(payload["input_sha256"]["first_g"], sha256(self.coefficients["first_g"]))
        self.assertEqual(set(payload["zero_order_audits"]), set(FAMILIES))
        self.assertEqual(set(payload["reader_warnings"]), set(FAMILIES))
        for family in FAMILIES:
            audit = payload["zero_order_audits"][family]
            self.assertEqual(set(audit), AUDIT_KEYS)
            self.assertEqual(audit["case"], family)
            self.assertTrue(audit["passed"])
            self.assertEqual(len(audit["source_file_sha256"]), 4)

        self.assertEqual(len(payload["alpha_results"]), len(ALPHAS))
        for alpha, alpha_result in zip(ALPHAS, payload["alpha_results"]):
            self.assertEqual(set(alpha_result), ALPHA_RESULT_KEYS)
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
                self.assertEqual(set(basis), BASIS_RESULT_KEYS)
                self.assertTrue(math.isfinite(basis["loss"]))
                self.assertTrue(math.isfinite(basis["max_condition"]))
                self.assertEqual(set(basis["families"]), set(FAMILIES))
                for family in basis["families"].values():
                    self.assertEqual(set(family), FAMILY_RESULT_KEYS)
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

        time.sleep(1.1)
        repeated_output = self.root / "pass_repeated"
        repeated = subprocess.run(
            self.command(repeated_output),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assert_complete_artifacts(repeated_output)
        for artifact in EXPECTED_ARTIFACTS:
            with self.subTest(deterministic_artifact=artifact):
                self.assertEqual(
                    (output / artifact).read_bytes(),
                    (repeated_output / artifact).read_bytes(),
                )

    def test_late_pdf_failure_cleans_atomic_set_and_allows_retry(self):
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt

        output = self.root / "late_failure"
        arguments = self.parsed_arguments(output)
        original_savefig = Figure.savefig

        def fail_pdf(figure, filename, *args, **kwargs):
            if Path(filename).suffix == ".pdf":
                raise RuntimeError("injected late PDF failure")
            return original_savefig(figure, filename, *args, **kwargs)

        with mock.patch.object(
            ANALYZER,
            "_evaluate",
            return_value=synthetic_alpha_results(),
        ):
            with mock.patch.object(Figure, "savefig", new=fail_pdf):
                with mock.patch.object(
                    plt,
                    "close",
                    wraps=plt.close,
                ) as close_figure:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "injected late PDF failure",
                    ):
                        ANALYZER.run(arguments)
        self.assertFalse(output.exists())
        self.assert_no_staging(output)
        self.assertTrue(close_figure.called)

        with mock.patch.object(
            ANALYZER,
            "_evaluate",
            return_value=synthetic_alpha_results(),
        ):
            self.assertEqual(ANALYZER.run(arguments), 0)
        self.assert_complete_artifacts(output)
        self.assert_no_staging(output)

    def test_existing_output_directory_is_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep\n", encoding="ascii")
        with mock.patch.object(
            ANALYZER,
            "_evaluate",
            return_value=synthetic_alpha_results(),
        ):
            with self.assertRaises(FileExistsError):
                ANALYZER.run(self.parsed_arguments(output))
        self.assertEqual(marker.read_text(encoding="ascii"), "keep\n")
        self.assertEqual({path.name for path in output.iterdir()}, {"keep.txt"})
        self.assert_no_staging(output)

    def test_input_content_mutation_before_publication_is_rejected(self):
        output = self.root / "mutated"

        def mutate_input(*unused_arguments):
            with self.coefficients["two_d"].open("ab") as stream:
                stream.write(b"mutation after read\n")
            return synthetic_alpha_results()

        with mock.patch.object(
            ANALYZER,
            "_evaluate",
            side_effect=mutate_input,
        ):
            with self.assertRaisesRegex(ValueError, "two_d.*content"):
                ANALYZER.run(self.parsed_arguments(output))
        self.assertFalse(output.exists())
        self.assert_no_staging(output)

    def test_input_hashes_are_bound_before_strict_readers(self):
        output = self.root / "mutated_by_reader"
        original_reader = ANALYZER.read_zero_order_audit

        def read_and_mutate(path, expected_case):
            audit = original_reader(path, expected_case)
            if expected_case == "H":
                with Path(path).open("ab") as stream:
                    stream.write(b" \n")
            return audit

        with mock.patch.object(
            ANALYZER,
            "read_zero_order_audit",
            side_effect=read_and_mutate,
        ):
            with mock.patch.object(
                ANALYZER,
                "_evaluate",
                return_value=synthetic_alpha_results(),
            ):
                with self.assertRaisesRegex(ValueError, "H_audit.*content"):
                    ANALYZER.run(self.parsed_arguments(output))
        self.assertFalse(output.exists())
        self.assert_no_staging(output)

    def test_input_symlink_retarget_before_publication_is_rejected(self):
        output = self.root / "retargeted"
        response_link = self.root / "H_response_link.dat"
        os.symlink(self.h_response, response_link)
        command = self.command(output)
        self.replace_option(command, "--h-response", response_link)
        arguments = ANALYZER.parse_arguments(command[2:])

        def retarget_input(*unused_arguments):
            response_link.unlink()
            os.symlink(self.h2_response, response_link)
            return synthetic_alpha_results()

        with mock.patch.object(
            ANALYZER,
            "_evaluate",
            side_effect=retarget_input,
        ):
            with self.assertRaisesRegex(ValueError, "H_response.*identity"):
                ANALYZER.run(arguments)
        self.assertFalse(output.exists())
        self.assert_no_staging(output)

    def test_plot_has_two_family_axes_and_base_sensitivity_series(self):
        from matplotlib.figure import Figure

        plot_dir = self.root / "plot_structure"
        plot_dir.mkdir()
        original_savefig = Figure.savefig
        observed = []

        def inspect_figure(figure, filename, *args, **kwargs):
            if not observed:
                observed.extend(
                    (
                        axis.get_title(),
                        [line.get_label() for line in axis.lines],
                    )
                    for axis in figure.axes
                )
            return original_savefig(figure, filename, *args, **kwargs)

        with mock.patch.object(Figure, "savefig", new=inspect_figure):
            ANALYZER._write_plot(plot_dir, synthetic_alpha_result())

        self.assertEqual([title for title, unused in observed], list(FAMILIES))
        expected_labels = {
            f"{basis_name} {series}"
            for basis_name in BASIS_NU
            for series in ("base", "sensitivity")
        }
        for title, labels in observed:
            with self.subTest(family=title):
                self.assertEqual(len(labels), 10)
                self.assertEqual(set(labels), expected_labels)

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
        self.assert_complete_artifacts(output)
        self.assert_no_staging(output)
        payload = json.loads(
            (output / "rpa_sensitive_ranking.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["decision"], "stop_galerkin_required")
        self.assertIsNone(payload["selected_alpha"])
        self.assertEqual(payload["admissible_alphas"], [])
        self.assertFalse(
            any(result["admissible"] for result in payload["alpha_results"])
        )


if __name__ == "__main__":
    unittest.main()
