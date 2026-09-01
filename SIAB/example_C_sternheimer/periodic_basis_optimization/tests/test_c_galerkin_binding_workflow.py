import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "galerkin_binding_workflow"
SCRIPT = WORKFLOW / "build_c_candidate_bank.py"
CONFIG = WORKFLOW / "c_diamond.json"


def load_module():
    spec = importlib.util.spec_from_file_location("build_c_candidate_bank", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CGalerkinBindingWorkflowTest(unittest.TestCase):
    def test_frozen_c_adapter_contract(self):
        module = load_module()
        config = module.load_config(CONFIG)

        self.assertEqual(config["system"], "C_atom_diamond")
        self.assertEqual(config["element"], "C")
        self.assertEqual(config["candidate_nu"], [3, 3, 2, 0, 0])
        self.assertEqual(config["fixed_nu"], [2, 2, 1, 0, 0])
        self.assertEqual(config["ao_count_per_atom"], 22)
        self.assertEqual(config["family_pair"], ["C_atom", "C_solid"])
        self.assertEqual(config["pareto_weights"], [0.25, 0.5, 0.75])
        self.assertEqual(config["atom_occupations"], {"up": 3, "down": 1})
        self.assertEqual(config["pbe_max_abs_deviation_ev"], 0.01)
        self.assertEqual(config["tail_q_indices"], [2, 6])
        self.assertEqual(config["proxy_q_indices"], [6, 7, 8])
        self.assertEqual(config["q3_maximum_condition_ratio"], 3.0)
        self.assertEqual(config["full_qstar_indices"], [1, 2, 3, 6, 7, 8, 11, 28])
        self.assertEqual(config["n_bands_chi0"], -1)
        self.assertEqual(config["product_pca_threshold"], 1.0e-4)
        self.assertEqual(config["coulomb"], "exact_grid_full_periodic")
        self.assertEqual(config["frequency_count"], 6)
        self.assertEqual(config["librpa_commit"], "d4810f73")
        self.assertEqual(config["reference_binding_ev_per_c"], 6.902326)
        self.assertEqual(config["acceptance_tolerance_ev_per_c"], 0.1)
        self.assertFalse(config["counterpoise"])

    def test_rejects_an_adapter_that_changes_the_frozen_qstar_set(self):
        module = load_module()
        payload = json.loads(CONFIG.read_text(encoding="ascii"))
        payload["full_qstar_indices"] = [1, 2, 3]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "full q-star"):
                module.load_config(path)

    def test_manifest_selects_only_candidates_that_pass_family_tradeoff(self):
        module = load_module()
        payload = module.build_bank_manifest(
            config=module.load_config(CONFIG),
            source_commit="1" * 40,
            input_hashes={"initial": "2" * 64},
            gradient_summary={
                "family_order": ["C_solid", "C_atom"],
                "family_losses": {"C_solid": 2.0, "C_atom": 1.0},
                "gradient_norms": {"C_solid": 4.0, "C_atom": 2.0},
                "gradient_cosines": {"C_solid:C_atom": -0.5},
                "minimum_occupied_capture": 0.999,
                "maximum_overlap_condition": 3.0,
            },
            candidates=[
                {
                    "name": "pareto_w0p25",
                    "weight": 0.25,
                    "trust_radius": 0.02,
                    "coefficients_sha256": "3" * 64,
                    "family_evaluation": {
                        "family_losses": {"C_solid": 1.9, "C_atom": 1.01},
                        "minimum_occupied_capture": 0.999,
                        "maximum_overlap_condition": 3.1,
                    },
                },
                {
                    "name": "pareto_w0p75",
                    "weight": 0.75,
                    "trust_radius": 0.02,
                    "coefficients_sha256": "4" * 64,
                    "family_evaluation": {
                        "family_losses": {"C_solid": 2.2, "C_atom": 0.8},
                        "minimum_occupied_capture": 0.999,
                        "maximum_overlap_condition": 3.2,
                    },
                },
            ],
        )

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["promotable_candidates"], ["pareto_w0p25"])
        self.assertEqual(payload["candidates"][0]["family_tradeoff_gate"], "pass")
        self.assertEqual(payload["candidates"][1]["family_tradeoff_gate"], "fail")


if __name__ == "__main__":
    unittest.main()
