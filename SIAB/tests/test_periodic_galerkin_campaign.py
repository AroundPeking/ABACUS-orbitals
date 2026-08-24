import unittest
from dataclasses import replace

import torch

import common  # noqa: F401 - configures the optimizer import path
from periodic_galerkin_campaign import evaluate_periodic_basis_capacity
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock
from test_periodic_galerkin_sternheimer import PeriodicGalerkinSternheimerTest


class PeriodicGalerkinCampaignTest(unittest.TestCase):
    def test_reports_candidate_and_mother_pi_capacity(self):
        dataset, _, _ = PeriodicGalerkinSternheimerTest().complete_two_level_dataset()
        coefficients = {
            "C": [torch.eye(2, dtype=torch.float64)],
        }

        report = evaluate_periodic_basis_capacity(
            dataset,
            coefficients,
            mother_response_tolerance=1.0e-10,
        )

        self.assertEqual(report["candidate"]["ao_count"], 2)
        self.assertEqual(report["candidate"]["nu"], {"C": [2]})
        self.assertLess(report["candidate"]["relative_pi_error"], 1.0e-14)
        self.assertLess(report["mother"]["relative_pi_error"], 1.0e-14)
        self.assertEqual(report["mother"]["minimum_effective_rank"], 2)
        self.assertEqual(report["mother"]["capacity_gate"], "PASS")

    def test_mother_capacity_gate_rejects_incomplete_primitive_space(self):
        dataset, _, _ = PeriodicGalerkinSternheimerTest().complete_two_level_dataset()
        dataset = dataset.__class__(
            **{
                **dataset.__dict__,
                "reference_response": 1.2 * dataset.reference_response,
            }
        )
        coefficients = {
            "C": [torch.eye(2, dtype=torch.float64)],
        }

        report = evaluate_periodic_basis_capacity(
            dataset,
            coefficients,
            mother_response_tolerance=1.0e-3,
        )

        self.assertGreater(report["mother"]["relative_pi_error"], 1.0e-3)
        self.assertEqual(report["mother"]["capacity_gate"], "FAIL")

    def test_rejects_nonpositive_capacity_tolerance(self):
        dataset, _, _ = PeriodicGalerkinSternheimerTest().complete_two_level_dataset()
        coefficients = {"C": [torch.eye(2, dtype=torch.float64)]}

        with self.assertRaisesRegex(ValueError, "mother_response_tolerance"):
            evaluate_periodic_basis_capacity(
                dataset,
                coefficients,
                mother_response_tolerance=0.0,
            )

    def test_reports_candidate_rank_failure_without_losing_mother_gate(self):
        dataset, delta, _ = PeriodicGalerkinSternheimerTest().complete_two_level_dataset()
        record = replace(
            dataset.kpoints[0],
            overlap=torch.diag(torch.tensor([1.0, 1.0, 0.0])).to(torch.complex128),
            hamiltonian_ha=torch.diag(
                torch.tensor([-0.5, 0.7, 0.0], dtype=torch.float64)
            ).to(torch.complex128),
            occupied_projection=torch.tensor(
                [[1.0, 0.0, 0.0]], dtype=torch.complex128
            ),
            source=torch.tensor([[[0.0, 0.3, 0.0]]], dtype=torch.complex128),
            reference_projection=torch.tensor(
                [[[[0.0, delta.conjugate(), 0.0]]]], dtype=torch.complex128
            ),
        )
        dataset = replace(
            dataset,
            primitive_count=3,
            primitive_blocks=(
                PeriodicGalerkinPrimitiveBlock("C", 0, 0, 0, 3, 0),
            ),
            kpoints=(record,),
        )

        report = evaluate_periodic_basis_capacity(
            dataset,
            {"C": [torch.eye(3, dtype=torch.float64)]},
            mother_response_tolerance=1.0e-10,
        )

        self.assertEqual(report["candidate"]["evaluation_gate"], "FAIL")
        self.assertIn("rank deficient", report["candidate"]["error"])
        self.assertEqual(report["mother"]["capacity_gate"], "PASS")
        self.assertTrue(report["optimization_allowed"])


if __name__ == "__main__":
    unittest.main()
