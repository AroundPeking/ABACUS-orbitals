"""Tests for the real-H Delta-ST response optimization runner."""

import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "example_H_sternheimer" / "delta_st_response_compression"
sys.path.insert(0, str(RUNNER_DIR))

from run_h_response_optimization import (
    _anchor_payload,
    _extension_payload,
    _history_payload,
    _variable_orbitals,
    parse_args,
)


class HDeltaSTResponseOptimizationRunnerTest(unittest.TestCase):
    def test_parser_freezes_the_bounded_full_optimization_defaults(self):
        argv = [
            "run_h_response_optimization.py",
            "reference",
            "primitive.dat",
            "fixed_ao.dat",
            "ORBITAL_RESULTS.txt",
            "H.orb",
            "output",
            "--reference-commit",
            "reference-commit",
            "--sidecar-commit",
            "sidecar-commit",
            "--siab-commit",
            "siab-commit",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.max_steps, 100)
        self.assertEqual(args.initial_step, 0.2)
        self.assertEqual(args.maximum_step, 2.0)
        self.assertEqual(args.relative_rank_tolerance, 1.0e-7)
        self.assertEqual(args.gradient_tolerance, 1.0e-8)
        self.assertEqual(args.relative_loss_tolerance, 1.0e-9)
        self.assertEqual(args.relative_loss_patience, 5)
        self.assertIsNone(args.append_l)
        self.assertTrue(hasattr(args, "include_fixed_ao_virtual"))
        self.assertFalse(args.include_fixed_ao_virtual)

    def test_parser_accepts_an_ordered_sequence_of_shell_extensions(self):
        argv = [
            "run_h_response_optimization.py",
            "reference",
            "primitive.dat",
            "fixed_ao.dat",
            "ORBITAL_RESULTS.txt",
            "H.orb",
            "output",
            "--reference-commit",
            "reference-commit",
            "--sidecar-commit",
            "sidecar-commit",
            "--siab-commit",
            "siab-commit",
            "--append-l",
            "1",
            "2",
            "--include-fixed-ao-virtual",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.append_l, [1, 2])
        self.assertTrue(args.include_fixed_ao_virtual)

    def test_history_payload_keeps_physical_convergence_diagnostics(self):
        record = types.SimpleNamespace(
            iteration=3,
            loss=0.125,
            accepted_step=0.4,
            relative_loss_reduction=0.02,
            raw_fixed_gradient_norm=0.3,
            masked_fixed_gradient_norm=0.0,
            variable_gradient_norm=0.04,
            maximum_frequency_loss=0.8,
            maximum_condition=52.0,
            retained_rank_by_spin=(8, 9),
            dropped_rank_by_spin=(1, 0),
        )

        payload = _history_payload(record)

        self.assertEqual(payload["iteration"], 3)
        self.assertEqual(payload["loss"], 0.125)
        self.assertEqual(payload["retained_rank_by_spin"], [8, 9])
        self.assertEqual(payload["dropped_rank_by_spin"], [1, 0])
        self.assertEqual(payload["masked_fixed_gradient_norm"], 0.0)

    def test_occupied_anchor_payload_records_the_atomic_basis_rotation(self):
        anchor = types.SimpleNamespace(
            occupied_band_index=0,
            omitted_original_s_zeta=1,
            fixed_ao_coefficients=(0.99, 0.1, 0.01),
            maximum_off_s_coefficient=2.0e-15,
            eigenvalue_max_abs_error_ha=3.0e-12,
        )

        payload = _anchor_payload(anchor)

        self.assertEqual(payload["occupied_band_index"], 0)
        self.assertEqual(payload["omitted_original_s_zeta"], 1)
        self.assertEqual(payload["fixed_ao_coefficients"], [0.99, 0.1, 0.01])
        self.assertEqual(payload["maximum_off_s_coefficient"], 2.0e-15)

    def test_extension_payload_records_all_deterministic_candidate_losses(self):
        extension = types.SimpleNamespace(
            element="H",
            l=1,
            selected_mode=4,
            initial_loss=0.12,
            selected_loss=0.08,
            candidate_losses=(0.11, 0.10, 0.09, 0.085, 0.08),
            radial_metric_condition=30.0,
            maximum_magnetic_metric_relative_deviation=4.0e-7,
            maximum_metric_orthogonality=2.0e-15,
            metric_normalization_error=3.0e-15,
        )

        payload = _extension_payload(extension)

        self.assertEqual(payload["element"], "H")
        self.assertEqual(payload["l"], 1)
        self.assertEqual(payload["selected_mode"], 4)
        self.assertEqual(
            payload["candidate_losses"],
            [0.11, 0.10, 0.09, 0.085, 0.08],
        )
        self.assertEqual(payload["selected_loss"], 0.08)
        self.assertEqual(
            payload["maximum_magnetic_metric_relative_deviation"],
            4.0e-7,
        )

    def test_variable_orbitals_follow_the_extended_radial_counts(self):
        self.assertEqual(
            _variable_orbitals((3, 3, 0)),
            ["H/l0/zeta3", "H/l1/zeta2", "H/l1/zeta3"],
        )


if __name__ == "__main__":
    unittest.main()
