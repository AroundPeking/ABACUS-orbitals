import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "galerkin_binding_workflow"
    / "build_c_solid_all_q_candidate.py"
)
AUDIT_SCRIPT = (
    ROOT
    / "galerkin_binding_workflow"
    / "audit_c_solid_qstar_inputs.py"
)
COMPLEMENT_SCRIPT = (
    ROOT
    / "galerkin_binding_workflow"
    / "build_c_solid_qstar_complement.py"
)
REFERENCE_AUDIT_SCRIPT = (
    ROOT
    / "galerkin_binding_workflow"
    / "audit_c_solid_direct_reference.py"
)
CONFIG = (
    ROOT
    / "galerkin_binding_workflow"
    / "c_diamond_solid_q123_reduced.json"
)
STANDARD_CONFIG = (
    ROOT
    / "galerkin_binding_workflow"
    / "c_diamond_solid_fd8_q13_standard.json"
)
STANDARD_Q1_RUNNER = (
    ROOT
    / "galerkin_binding_workflow"
    / "run_c_solid_fd8_q13_standard_q1_df.slurm"
)
STANDARD_QAVG_INPUT = (
    ROOT
    / "galerkin_binding_workflow"
    / "librpa_c_solid_fd8_q13_qavg.in"
)
STANDARD_FREQUENCY_REFERENCE = (
    ROOT
    / "galerkin_binding_workflow"
    / "c_diamond_fd8_nfreq12_reference.tsv"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_c_solid_all_q_candidate",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_c_solid_qstar_inputs",
        AUDIT_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_complement_module():
    spec = importlib.util.spec_from_file_location(
        "build_c_solid_qstar_complement",
        COMPLEMENT_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_c_solid_direct_reference",
        REFERENCE_AUDIT_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CSolidAllQGalerkinTest(unittest.TestCase):
    FULL_CONTRACT = (
        {"label": 1, "selected_iq": 1, "multiplicity": 1},
        {"label": 2, "selected_iq": 22, "multiplicity": 8},
        {"label": 3, "selected_iq": 43, "multiplicity": 4},
        {"label": 6, "selected_iq": 6, "multiplicity": 6},
        {"label": 7, "selected_iq": 27, "multiplicity": 24},
        {"label": 8, "selected_iq": 23, "multiplicity": 12},
        {"label": 11, "selected_iq": 11, "multiplicity": 3},
        {"label": 28, "selected_iq": 55, "multiplicity": 6},
    )
    STANDARD_FD8_CONTRACT = (
        {"label": 1, "selected_iq": 1, "multiplicity": 1},
        {"label": 2, "selected_iq": 2, "multiplicity": 6},
        {"label": 3, "selected_iq": 3, "multiplicity": 3},
        {"label": 6, "selected_iq": 6, "multiplicity": 6},
        {"label": 7, "selected_iq": 7, "multiplicity": 12},
        {"label": 8, "selected_iq": 8, "multiplicity": 6},
        {"label": 11, "selected_iq": 11, "multiplicity": 3},
        {"label": 22, "selected_iq": 22, "multiplicity": 2},
        {"label": 23, "selected_iq": 23, "multiplicity": 6},
        {"label": 24, "selected_iq": 24, "multiplicity": 6},
        {"label": 27, "selected_iq": 27, "multiplicity": 6},
        {"label": 28, "selected_iq": 28, "multiplicity": 6},
        {"label": 43, "selected_iq": 43, "multiplicity": 1},
    )

    def dataset(self, record, **changes):
        values = {
            "abacus_commit": "a" * 40,
            "executable_sha256": "b" * 64,
            "orbital_sha256": "c" * 64,
            "pseudopotential_sha256": "d" * 64,
            "auxiliary_basis_sha256": "e" * 64,
            "primitive_blocks_sha256": "f" * 64,
            "physics_hash": f'{record["label"]:064x}',
            "selected_iq": record["selected_iq"],
            "q_count": 64,
            "qpoint": (0.0, 0.0, 0.0),
            "q_weight": record["multiplicity"] / 64.0,
            "frequency_ha": torch.tensor(
                [0.1, 0.3, 0.8, 2.0, 5.8, 20.8],
                dtype=torch.float64,
            ),
            "frequency_weights_ha": torch.tensor(
                [0.2, 0.3, 0.7, 2.0, 6.7, 30.9],
                dtype=torch.float64,
            ),
        }
        values.update(changes)
        return SimpleNamespace(**values)

    def datasets(self, contract=None):
        contract = self.FULL_CONTRACT if contract is None else tuple(contract)
        return tuple(self.dataset(record) for record in contract)

    def standard_datasets(self):
        frequency = torch.tensor(
            [0.02, 0.07, 0.16, 0.31, 0.55, 0.92, 1.5, 2.5, 4.2, 7.5, 15.0, 40.0],
            dtype=torch.float64,
        )
        weights = torch.tensor(
            [0.03, 0.06, 0.11, 0.18, 0.29, 0.45, 0.70, 1.1, 1.9, 3.8, 10.0, 50.0],
            dtype=torch.float64,
        )
        return tuple(
            self.dataset(
                record,
                frequency_ha=frequency.clone(),
                frequency_weights_ha=weights.clone(),
            )
            for record in self.STANDARD_FD8_CONTRACT
        )

    def test_standard_config_locks_twelve_frequency_pca_and_qavg_contract(self):
        module = load_module()

        config = module.load_config(STANDARD_CONFIG)

        self.assertEqual(config["format_version"], 2)
        self.assertEqual(config["qstar_scheme"], "fd8_discrete_13")
        self.assertEqual(config["frequency_count"], 12)
        self.assertEqual(config["product_pca_threshold"], 1.0e-6)
        self.assertEqual(config["training_coulomb"], "full_periodic")
        self.assertEqual(config["qstar_contract"], list(self.STANDARD_FD8_CONTRACT))
        self.assertEqual(
            config["final_energy_protocol"],
            {
                "coulomb": "full_periodic",
                "frequency_count": 12,
                "option_dielect_func": 3,
                "product_pca_threshold": 1.0e-6,
                "replace_w_head": True,
                "rpa_headwing_body_start": 1,
                "rpa_headwing_mode": "qavg",
                "sqrt_coulomb_threshold": 1.0e-5,
                "use_rpa_gamma": True,
            },
        )

    def test_accepts_standard_fd8_thirteen_q_twelve_frequency_contract(self):
        module = load_module()

        result = module.validate_qstar_datasets(
            self.standard_datasets(),
            qstar_contract=self.STANDARD_FD8_CONTRACT,
            q_count=64,
            coverage="full",
            expected_labels=module.STANDARD_FD8_QSTAR_LABELS,
            expected_frequency_count=12,
        )

        self.assertEqual(result["dataset_contract_gate"], "pass")
        self.assertEqual(result["frequency_count"], 12)
        self.assertEqual(result["multiplicity_sum"], 64)
        self.assertEqual(
            result["logical_qstar_labels"],
            [1, 2, 3, 6, 7, 8, 11, 22, 23, 24, 27, 28, 43],
        )

    def test_standard_contract_rejects_legacy_eight_q_and_six_frequency_data(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "13 standard FD8 q stars"):
            module.validate_qstar_datasets(
                self.datasets(),
                qstar_contract=self.FULL_CONTRACT,
                q_count=64,
                coverage="full",
                expected_labels=module.STANDARD_FD8_QSTAR_LABELS,
                expected_frequency_count=12,
            )
        with self.assertRaisesRegex(ValueError, "twelve-point"):
            module.validate_qstar_datasets(
                self.datasets(self.STANDARD_FD8_CONTRACT),
                qstar_contract=self.STANDARD_FD8_CONTRACT,
                q_count=64,
                coverage="full",
                expected_labels=module.STANDARD_FD8_QSTAR_LABELS,
                expected_frequency_count=12,
            )

    def test_standard_q1_runner_and_final_qavg_input_keep_stage_boundaries(self):
        runner = STANDARD_Q1_RUNNER.read_text(encoding="ascii")
        final_input = STANDARD_QAVG_INPUT.read_text(encoding="ascii")

        self.assertIn("set_input_key sternheimer_nfreq 12", runner)
        self.assertIn("set_input_key exx_pca_threshold 1e-6", runner)
        self.assertIn("set_input_key out_sternheimer_basis_opt 1", runner)
        self.assertIn("set_input_key sternheimer_q_index 1", runner)
        self.assertNotIn("rpa_headwing_mode", runner)
        self.assertNotIn("replace_w_head", runner)
        self.assertIn("nfreq = 12", final_input)
        self.assertIn("use_rpa_gamma = true", final_input)
        self.assertIn("replace_w_head = true", final_input)
        self.assertIn("option_dielect_func = 3", final_input)
        self.assertIn("rpa_headwing_mode = qavg", final_input)
        self.assertIn("rpa_headwing_body_start = 1", final_input)
        self.assertIn("sqrt_coulomb_threshold = 1e-5", final_input)

    def test_standard_frequency_reference_matches_accepted_delta_st_grid(self):
        rows = [
            tuple(float(value) for value in line.split())
            for line in STANDARD_FREQUENCY_REFERENCE.read_text(
                encoding="ascii"
            ).splitlines()
        ]
        runner = STANDARD_Q1_RUNNER.read_text(encoding="ascii")

        self.assertEqual(len(rows), 12)
        self.assertEqual(
            rows[0],
            (1.0, 0.04387042772464342, 0.08917051395758147),
        )
        self.assertEqual(
            rows[-1],
            (12.0, 38.14573704601481, 41.61618619514986),
        )
        self.assertIn("c_diamond_fd8_nfreq12_reference.tsv", runner)
        self.assertIn("frequency grid differs from accepted standard reference", runner)

    def test_accepts_complete_weighted_eight_qstar_contract(self):
        module = load_module()

        result = module.validate_qstar_datasets(
            self.datasets(),
            qstar_contract=self.FULL_CONTRACT,
            q_count=64,
            coverage="full",
        )

        self.assertEqual(result["dataset_contract_gate"], "pass")
        self.assertEqual(result["coverage"], "full")
        self.assertEqual(result["logical_qstar_labels"], [1, 2, 3, 6, 7, 8, 11, 28])
        self.assertEqual(result["selected_iq"], [1, 22, 43, 6, 27, 23, 11, 55])
        self.assertEqual(result["multiplicity_sum"], 64)
        self.assertEqual(result["physical_release_gate"], "pending_candidate")

    def test_reduced_q123_contract_is_permanently_held(self):
        module = load_module()
        contract = self.FULL_CONTRACT[:3]

        result = module.validate_qstar_datasets(
            self.datasets(contract),
            qstar_contract=contract,
            q_count=64,
            coverage="reduced",
        )

        self.assertEqual(result["dataset_contract_gate"], "pass")
        self.assertEqual(result["coverage"], "reduced")
        self.assertEqual(result["multiplicity_sum"], 13)
        self.assertEqual(result["physical_release_gate"], "hold")

    def test_rejects_missing_or_duplicate_full_qstar(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "eight logical q stars"):
            module.validate_qstar_datasets(
                self.datasets(self.FULL_CONTRACT[:-1]),
                qstar_contract=self.FULL_CONTRACT[:-1],
                q_count=64,
                coverage="full",
            )
        duplicated = self.FULL_CONTRACT[:-1] + (dict(self.FULL_CONTRACT[0]),)
        with self.assertRaisesRegex(ValueError, "unique"):
            module.validate_qstar_datasets(
                self.datasets(duplicated),
                qstar_contract=duplicated,
                q_count=64,
                coverage="full",
            )

    def test_rejects_wrong_q_weight(self):
        module = load_module()
        datasets = list(self.datasets())
        datasets[3] = self.dataset(self.FULL_CONTRACT[3], q_weight=0.5)
        with self.assertRaisesRegex(ValueError, "q weight"):
            module.validate_qstar_datasets(
                tuple(datasets),
                qstar_contract=self.FULL_CONTRACT,
                q_count=64,
                coverage="full",
            )

    def test_rejects_frequency_grid_mismatch(self):
        module = load_module()
        datasets = list(self.datasets())
        datasets[4] = self.dataset(
            self.FULL_CONTRACT[4],
            frequency_ha=torch.tensor(
                [0.1, 0.3, 0.8, 2.0, 5.8, 21.0],
                dtype=torch.float64,
            ),
        )
        with self.assertRaisesRegex(ValueError, "frequency grid"):
            module.validate_qstar_datasets(
                tuple(datasets),
                qstar_contract=self.FULL_CONTRACT,
                q_count=64,
                coverage="full",
            )

    def test_accepts_roundoff_level_frequency_grid_differences(self):
        module = load_module()
        datasets = list(self.datasets())
        datasets[1] = self.dataset(
            self.FULL_CONTRACT[1],
            frequency_ha=torch.tensor(
                [0.1, 0.3, 0.8, 2.0, 5.8, 20.8 + 2.0e-13],
                dtype=torch.float64,
            ),
            frequency_weights_ha=torch.tensor(
                [0.2, 0.3, 0.7, 2.0, 6.7, 30.9 - 2.0e-13],
                dtype=torch.float64,
            ),
        )

        result = module.validate_qstar_datasets(
            tuple(datasets),
            qstar_contract=self.FULL_CONTRACT,
            q_count=64,
            coverage="full",
        )

        self.assertEqual(result["dataset_contract_gate"], "pass")

    def test_rejects_shared_provenance_mismatch(self):
        module = load_module()
        datasets = list(self.datasets())
        datasets[5] = self.dataset(
            self.FULL_CONTRACT[5],
            auxiliary_basis_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "shared provenance"):
            module.validate_qstar_datasets(
                tuple(datasets),
                qstar_contract=self.FULL_CONTRACT,
                q_count=64,
                coverage="full",
            )

    def test_reduced_config_and_cli_have_no_atomic_inputs(self):
        module = load_module()
        config = module.load_config(CONFIG)
        self.assertEqual(config["system"], "C_diamond_solid")
        self.assertEqual(config["coverage"], "reduced")
        self.assertEqual(config["q_count"], 64)
        self.assertEqual(config["qstar_contract"], list(self.FULL_CONTRACT[:3]))

        arguments = module.parse_args(
            [
                "--config",
                str(CONFIG),
                "--qstar",
                "1=/tmp/q1",
                "--qstar",
                "2=/tmp/q2",
                "--qstar",
                "3=/tmp/q3",
                "--initial",
                "/tmp/initial.orb",
                "--output-directory",
                "/tmp/output",
                "--source-commit",
                "1" * 40,
            ]
        )
        self.assertEqual([label for label, _ in arguments.qstar], [1, 2, 3])
        self.assertFalse(any("atom" in name for name in vars(arguments)))

    def test_cli_writes_solid_only_candidate_artifacts(self):
        module = load_module()
        contract = self.FULL_CONTRACT[:3]
        datasets = {
            record["label"]: self.dataset(record) for record in contract
        }
        initial = {
            "C": [
                torch.eye(3, dtype=torch.float64),
                torch.eye(3, dtype=torch.float64),
                torch.eye(2, dtype=torch.float64),
                torch.empty((3, 0), dtype=torch.float64),
                torch.empty((3, 0), dtype=torch.float64),
            ]
        }
        calls = {}

        def dataset_reader(path, **options):
            self.assertFalse(options["include_reference_projection"])
            return datasets[int(Path(path).name[1:])]

        def coefficient_reader(path, **options):
            calls["coefficient_reader"] = (Path(path), options)
            return initial

        def gradient_evaluator(values, coefficients, **options):
            calls["gradient"] = (values, coefficients, options)
            return SimpleNamespace(
                family_order=("C_solid",),
                family_losses={"C_solid": 2.0},
                gradient_norms={"C_solid": 0.5},
                gradient_cosines={},
                minimum_occupied_capture=0.99999,
                maximum_overlap_condition=2.0,
            )

        candidate = SimpleNamespace(
            family="C_solid",
            trust_radius=0.01,
            coefficients=initial,
            coefficients_sha256="9" * 64,
        )

        def candidate_builder(result, **options):
            calls["candidate_builder"] = (result, options)
            return candidate

        def candidate_evaluator(result, coefficients):
            self.assertIs(coefficients, initial)
            return {
                "loss": 1.5,
                "family_losses": {"C_solid": 1.5},
                "minimum_occupied_capture": 0.99998,
                "maximum_overlap_condition": 2.1,
            }

        def coefficient_writer(path, coefficients):
            self.assertIs(coefficients, initial)
            Path(path).write_text("candidate\n", encoding="ascii")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(CONFIG.read_text(encoding="ascii"))
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="ascii")
            initial_path = root / "initial.orb"
            initial_path.write_text("initial\n", encoding="ascii")
            for label in (1, 2, 3):
                (root / f"q{label}").mkdir()
            output = root / "output"

            result = module.main(
                [
                    "--config",
                    str(config_path),
                    "--qstar",
                    f"1={root / 'q1'}",
                    "--qstar",
                    f"2={root / 'q2'}",
                    "--qstar",
                    f"3={root / 'q3'}",
                    "--initial",
                    str(initial_path),
                    "--output-directory",
                    str(output),
                    "--source-commit",
                    "1" * 40,
                ],
                dataset_reader=dataset_reader,
                dataset_contract_validator=lambda values: None,
                coefficient_reader=coefficient_reader,
                gradient_evaluator=gradient_evaluator,
                candidate_builder=candidate_builder,
                candidate_evaluator=candidate_evaluator,
                coefficient_writer=coefficient_writer,
            )

            self.assertEqual(
                calls["gradient"][2]["dataset_families"],
                ("C_solid", "C_solid", "C_solid"),
            )
            self.assertEqual(calls["candidate_builder"][1]["family"], "C_solid")
            for name in (
                "STATUS.json",
                "PROVENANCE.json",
                "DATASET_INVENTORY.json",
                "GRADIENT.json",
                "CANDIDATE.json",
                "ORBITAL_RESULTS.txt",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(result["candidate_generation_gate"], "pass")
            self.assertEqual(result["physical_release_gate"], "hold")
            status = json.loads((output / "STATUS.json").read_text(encoding="ascii"))
            self.assertEqual(status["status"], "success")
            self.assertEqual(status["coverage"], "reduced")

    def test_read_only_audit_reports_present_and_missing_qstars(self):
        module = load_audit_module()

        def reader(path):
            label = int(Path(path).name[1:])
            record = next(
                item for item in self.FULL_CONTRACT if item["label"] == label
            )
            return self.dataset(record)

        result = module.audit_qstar_inputs(
            {1: Path("q1"), 2: Path("q2"), 3: Path("q3")},
            dataset_reader=reader,
            direct_reference=None,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["present_logical_qstars"], [1, 2, 3])
        self.assertEqual(result["present_selected_iq"], [1, 22, 43])
        self.assertEqual(result["present_multiplicity_sum"], 13)
        self.assertEqual(result["missing_logical_qstars"], [6, 7, 8, 11, 28])
        self.assertEqual(result["missing_selected_iq"], [6, 27, 23, 11, 55])
        self.assertEqual(result["missing_multiplicity_sum"], 51)
        self.assertEqual(
            result["missing_qstars"],
            [
                {
                    "logical_qstar_label": 6,
                    "selected_iq": 6,
                    "multiplicity": 6,
                    "q_weight": 6.0 / 64.0,
                },
                {
                    "logical_qstar_label": 7,
                    "selected_iq": 27,
                    "multiplicity": 24,
                    "q_weight": 24.0 / 64.0,
                },
                {
                    "logical_qstar_label": 8,
                    "selected_iq": 23,
                    "multiplicity": 12,
                    "q_weight": 12.0 / 64.0,
                },
                {
                    "logical_qstar_label": 11,
                    "selected_iq": 11,
                    "multiplicity": 3,
                    "q_weight": 3.0 / 64.0,
                },
                {
                    "logical_qstar_label": 28,
                    "selected_iq": 55,
                    "multiplicity": 6,
                    "q_weight": 6.0 / 64.0,
                },
            ],
        )
        self.assertFalse(result["direct_solid_reference"]["exists"])
        self.assertEqual(result["direct_solid_reference"]["status"], "missing")
        self.assertEqual(result["complement_submission_gate"], "hold")
        self.assertEqual(result["physical_release_gate"], "hold")

    def test_builds_hash_locked_missing_qstar_complement(self):
        module = load_complement_module()
        inventory = {
            "coverage": "reduced",
            "dataset_contract_gate": "pass",
            "datasets": [
                {
                    "logical_qstar_label": record["label"],
                    "selected_iq": record["selected_iq"],
                    "multiplicity": record["multiplicity"],
                    "q_weight": record["multiplicity"] / 64.0,
                    "physics_hash": f'{record["label"]:064x}',
                }
                for record in self.FULL_CONTRACT[:3]
            ],
            "multiplicity_sum": 13,
            "physical_release_gate": "hold",
            "q_count": 64,
        }

        result = module.build_complement_contract(
            inventory,
            inventory_sha256="1" * 64,
            source_commit="2" * 40,
            measured_q_wall_minutes=[47.5, 52.9, 53.5],
            nodes_per_q=48,
            storage_gib_per_q=29.0,
            direct_reference=None,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["present_multiplicity_sum"], 13)
        self.assertEqual(result["missing_multiplicity_sum"], 51)
        self.assertEqual(result["missing_selected_iq"], [6, 27, 23, 11, 55])
        self.assertEqual(result["resource_estimate"]["missing_q_count"], 5)
        self.assertEqual(result["resource_estimate"]["storage_gib"], 145.0)
        self.assertAlmostEqual(
            result["resource_estimate"]["node_hours_min"],
            190.0,
        )
        self.assertAlmostEqual(
            result["resource_estimate"]["node_hours_max"],
            214.0,
        )
        self.assertEqual(result["direct_solid_reference"]["status"], "missing")
        self.assertEqual(result["physical_submission_gate"], "hold")

    def test_full_bz_coulomb_diagnostic_is_not_a_direct_response_reference(self):
        module = load_reference_audit_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "full-bz-coulomb"
            for index in range(1, 65):
                output = root / f"q{index}" / f"OUT.Q{index}"
                output.mkdir(parents=True)
                (output / "STERNHEIMER_SIAB_STATUS.dat").write_text(
                    "status abfs_diag_only\n"
                    f"sternheimer_q_index {index}\n"
                    "nfreq 6\n",
                    encoding="ascii",
                )
                (output / f"v1_coulomb_full_iq_{index}_rank0.dat").write_text(
                    "coulomb\n",
                    encoding="ascii",
                )

            result = module.audit_direct_reference_candidates(
                candidate_roots=[root],
                response_roots=[],
            )

        record = result["candidate_roots"][0]
        self.assertEqual(record["classification"], "coulomb_only_diagnostic")
        self.assertEqual(record["status_file_count"], 64)
        self.assertEqual(record["status_values"], {"abfs_diag_only": 64})
        self.assertEqual(record["response_dataset_count"], 0)
        self.assertEqual(result["direct_solid_reference"]["status"], "missing")
        self.assertEqual(result["physical_submission_gate"], "hold")

    def test_reduced_converged_response_reports_missing_qstar_weight(self):
        module = load_reference_audit_module()
        selected_iq = (1, 22, 43)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "response-root"
            for index in selected_iq:
                output = root / f"q{index}" / f"OUT.Q{index}"
                dataset = output / "STERNHEIMER_BASIS_OPT_V1"
                dataset.mkdir(parents=True)
                (output / "STERNHEIMER_SIAB_STATUS.dat").write_text(
                    "status success\n"
                    "format basis_opt_v1\n"
                    f"sternheimer_q_index {index}\n"
                    "nfreq 6\n"
                    "all_converged yes\n",
                    encoding="ascii",
                )
                (dataset / "status.dat").write_text(
                    "status success\n",
                    encoding="ascii",
                )
                (dataset / "response_ik_1_ifreq_0.bin").write_bytes(b"response")

            result = module.audit_direct_reference_candidates(
                candidate_roots=[],
                response_roots=[root],
            )

        self.assertEqual(result["available_selected_iq"], [1, 22, 43])
        self.assertEqual(result["available_logical_qstars"], [1, 2, 3])
        self.assertEqual(result["available_multiplicity_sum"], 13)
        self.assertEqual(result["missing_selected_iq"], [6, 27, 23, 11, 55])
        self.assertEqual(result["missing_multiplicity_sum"], 51)
        self.assertEqual(result["direct_solid_reference"]["status"], "missing")
        self.assertEqual(
            result["direct_solid_reference"]["reason"],
            "converged_response_coverage_is_reduced",
        )
        self.assertEqual(result["physical_submission_gate"], "hold")


if __name__ == "__main__":
    unittest.main()
