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
CONFIG = (
    ROOT
    / "galerkin_binding_workflow"
    / "c_diamond_solid_q123_reduced.json"
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


if __name__ == "__main__":
    unittest.main()
