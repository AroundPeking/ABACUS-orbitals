import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "galerkin_binding_workflow"
    / "build_c_solid_all_q_candidate.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_c_solid_all_q_candidate",
        SCRIPT,
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
        {"label": 7, "selected_iq": 7, "multiplicity": 24},
        {"label": 8, "selected_iq": 8, "multiplicity": 12},
        {"label": 11, "selected_iq": 11, "multiplicity": 3},
        {"label": 28, "selected_iq": 28, "multiplicity": 6},
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
        self.assertEqual(result["selected_iq"], [1, 22, 43, 6, 7, 8, 11, 28])
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


if __name__ == "__main__":
    unittest.main()
