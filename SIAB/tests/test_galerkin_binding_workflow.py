import unittest

import common  # noqa: F401 - configures the optimizer import path
import galerkin_binding_workflow


class GalerkinBindingWorkflowTest(unittest.TestCase):
    def config(self):
        return {
            "candidate_fingerprint": "a" * 64,
            "stages": [
                "galerkin_screen",
                "pbe_gate",
                "tail_gate",
                "proxy_gate",
                "full_q_gate",
            ],
            "reference_binding_ev": 6.902326,
            "acceptance_tolerance_ev": 0.1,
        }

    @staticmethod
    def passed(*stages):
        return {
            stage: {
                "status": "success",
                "gate": "pass",
                "candidate_fingerprint": "a" * 64,
            }
            for stage in stages
        }

    def test_returns_exactly_one_next_stage(self):
        initial = galerkin_binding_workflow.assess_workflow(self.config(), {})
        after_galerkin = galerkin_binding_workflow.assess_workflow(
            self.config(),
            self.passed("galerkin_screen"),
        )

        self.assertEqual(initial["decision"], "continue")
        self.assertEqual(initial["workflow_state"], "galerkin_screen")
        self.assertEqual(initial["next_action"], "submit")
        self.assertEqual(after_galerkin["workflow_state"], "pbe_gate")
        self.assertEqual(after_galerkin["next_action"], "submit")

    def test_gate_failure_is_terminal(self):
        evidence = self.passed("galerkin_screen")
        evidence["pbe_gate"] = {
            "status": "success",
            "gate": "fail",
            "candidate_fingerprint": "a" * 64,
            "failure_reasons": ["binding_shift_exceeds_10_mev"],
        }

        result = galerkin_binding_workflow.assess_workflow(
            self.config(),
            evidence,
        )

        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(result["workflow_state"], "rejected")
        self.assertEqual(result["failed_stage"], "pbe_gate")
        self.assertEqual(result["next_action"], None)

    def test_active_duplicate_waits_and_completed_duplicate_collects(self):
        config = self.config()
        active = galerkin_binding_workflow.assess_workflow(
            config,
            {},
            existing_actions={
                "galerkin_screen": {
                    "candidate_fingerprint": "a" * 64,
                    "status": "running",
                    "job_id": "123",
                }
            },
        )
        completed = galerkin_binding_workflow.assess_workflow(
            config,
            {},
            existing_actions={
                "galerkin_screen": {
                    "candidate_fingerprint": "a" * 64,
                    "status": "completed",
                    "job_id": "124",
                }
            },
        )

        self.assertEqual(active["next_action"], "wait")
        self.assertEqual(active["existing_job_id"], "123")
        self.assertEqual(completed["next_action"], "collect")
        self.assertEqual(completed["existing_job_id"], "124")

    def test_final_binding_applies_reference_tolerance(self):
        stages = self.config()["stages"]
        accepted_evidence = self.passed(*stages)
        accepted_evidence["full_q_gate"]["binding_energy_ev"] = 6.96
        rejected_evidence = self.passed(*stages)
        rejected_evidence["full_q_gate"]["binding_energy_ev"] = 7.05

        accepted = galerkin_binding_workflow.assess_workflow(
            self.config(),
            accepted_evidence,
        )
        rejected = galerkin_binding_workflow.assess_workflow(
            self.config(),
            rejected_evidence,
        )

        self.assertEqual(accepted["decision"], "accepted")
        self.assertAlmostEqual(accepted["binding_error_ev"], 0.057674)
        self.assertEqual(rejected["decision"], "rejected")
        self.assertEqual(rejected["failed_stage"], "full_q_gate")
        self.assertAlmostEqual(rejected["binding_error_ev"], 0.147674)

    def test_rejects_evidence_from_another_candidate(self):
        evidence = self.passed("galerkin_screen")
        evidence["galerkin_screen"]["candidate_fingerprint"] = "b" * 64

        with self.assertRaisesRegex(ValueError, "candidate fingerprint"):
            galerkin_binding_workflow.assess_workflow(self.config(), evidence)


if __name__ == "__main__":
    unittest.main()
