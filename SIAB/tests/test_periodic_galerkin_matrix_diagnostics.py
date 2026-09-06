"""Local synthetic matrix diagnostics, without producer or physics jobs."""

from dataclasses import fields, is_dataclass, replace
import hashlib
import importlib
import importlib.util
import json
import math
import struct
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
import periodic_galerkin_rpa as rpa
import test_periodic_galerkin_sternheimer as fixtures


MODULE = "periodic_galerkin_matrix_diagnostics"
SCOPE = "candidate_minus_frozen_reference_not_mother_or_cross_pca"
Q_FIELDS = {
    "selected_iq",
    "q_weight",
    "physics_hash",
    "whitening_sha256",
    "frequency_ha",
    "frequency_weights_ha",
    "residual_squared_norm",
    "reference_squared_norm",
    "weighted_residual_squared_norm",
    "weighted_reference_squared_norm",
    "relative_pi_error",
    "pi_numerator",
    "pi_denominator",
}


class PeriodicGalerkinMatrixDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec(MODULE), "matrix diagnostic API not implemented"
        )
        self.diagnose = importlib.import_module(MODULE).periodic_rpa_matrix_diagnostics

    def fixture(self):
        (
            base,
            _,
            _,
        ) = fixtures.PeriodicGalerkinSternheimerTest().complete_two_level_dataset()
        reference = torch.diag_embed(
            torch.tensor([[-1.0, -2.0], [-0.5, -1.0]], dtype=torch.complex128)
        )
        first = replace(
            base,
            selected_iq=1,
            q_count=16,
            qpoint=(0.25, 0.0, 0.0),
            q_weight=0.25,
            raw_auxiliary_dimension=3,
            whitened_auxiliary_rank=2,
            frequency_ha=torch.tensor([0.2, 0.8], dtype=torch.float64),
            frequency_weights_ha=torch.tensor([0.4, 1.2], dtype=torch.float64),
            coulomb_metric=torch.eye(3, dtype=torch.complex128),
            coulomb_whitening=torch.eye(3, dtype=torch.complex128)[:, :2],
            reference_response=reference,
        )
        second = replace(
            first,
            selected_iq=5,
            qpoint=(0.0, 0.5, 0.0),
            q_weight=0.5,
            physics_hash="8" * 64,
            whitened_auxiliary_rank=1,
            coulomb_whitening=first.coulomb_whitening[:, :1],
            reference_response=torch.tensor(
                [[[-3.0]], [[-1.0]]], dtype=torch.complex128
            ),
        )
        return (first, second), (0.5 * reference, 2.0 * second.reference_response)

    def assert_plain_json(self, value):
        if isinstance(value, dict):
            self.assertIs(type(value), dict)
            for key, item in value.items():
                self.assertIs(type(key), str)
                self.assert_plain_json(item)
        elif isinstance(value, list):
            for item in value:
                self.assert_plain_json(item)
        else:
            self.assertIn(type(value), (str, int, float, bool, type(None)))
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value))
        json.dumps(value, allow_nan=False)

    def test_weights_different_q_ranks_and_global_pi_loss_reconstruction(self):
        datasets, responses = self.fixture()
        report = self.diagnose(datasets, responses)
        self.assertEqual(report["scope"], SCOPE)
        for phrase in ("Coulomb-whitened", "Frobenius", "k", "q", "frequency"):
            self.assertIn(phrase, report["metric_definition"])
        self.assertEqual(len(report["per_q"]), 2)
        expected_norms = (([1.25, 0.3125], [5.0, 1.25]), ([9.0, 1.0], [9.0, 1.0]))
        for dataset, record, (residual, reference) in zip(
            datasets, report["per_q"], expected_norms
        ):
            self.assertEqual(set(record), Q_FIELDS)
            self.assertEqual(record["selected_iq"], dataset.selected_iq)
            self.assertEqual(record["q_weight"], dataset.q_weight)
            self.assertEqual(record["physics_hash"], dataset.physics_hash)
            self.assertEqual(record["frequency_ha"], [0.2, 0.8])
            self.assertEqual(record["frequency_weights_ha"], [0.4, 1.2])
            self.assertEqual(record["residual_squared_norm"], residual)
            self.assertEqual(record["reference_squared_norm"], reference)
            for norm, name in ((residual, "residual"), (reference, "reference")):
                expected = [dataset.q_weight * w * n for w, n in zip([0.4, 1.2], norm)]
                self.assertEqual(record["weighted_" + name + "_squared_norm"], expected)
                total = "pi_numerator" if name == "residual" else "pi_denominator"
                self.assertAlmostEqual(record[total], math.fsum(expected), places=14)
            self.assertEqual(
                record["relative_pi_error"],
                [math.sqrt(r / d) for r, d in zip(residual, reference)],
            )
        for key in ("pi_numerator", "pi_denominator"):
            self.assertAlmostEqual(
                report[key], math.fsum(q[key] for q in report["per_q"]), places=14
            )
        self.assertAlmostEqual(report["pi_numerator"], 2.61875, places=14)
        self.assertAlmostEqual(report["pi_denominator"], 3.275, places=14)
        objective = rpa.periodic_rpa_objective(datasets, responses)
        self.assertAlmostEqual(
            report["pi_relative_squared_error"],
            float(objective.pi_relative_squared_error),
            places=14,
        )
        self.assertAlmostEqual(
            report["pi_relative_squared_error"],
            report["pi_numerator"] / report["pi_denominator"],
            places=14,
        )
        self.assert_plain_json(report)

    def test_same_eigenvalues_different_eigenvectors_are_detected(self):
        datasets, _ = self.fixture()
        dataset = datasets[0]
        swapped = dataset.reference_response.flip((-2, -1))
        torch.testing.assert_close(
            torch.linalg.eigvalsh(swapped),
            torch.linalg.eigvalsh(dataset.reference_response),
        )
        objective = rpa.periodic_rpa_objective((dataset,), (swapped,))
        self.assertEqual(float(objective.trace_log_relative_squared_error), 0.0)
        self.assertEqual(float(objective.energy_relative_squared_error), 0.0)
        report = self.diagnose((dataset,), (swapped,))
        self.assertEqual(report["per_q"][0]["residual_squared_norm"], [2.0, 0.5])
        self.assertGreater(report["pi_relative_squared_error"], 0.0)

    def test_common_complex_unitary_change_preserves_matrix_metric(self):
        datasets, responses = self.fixture()
        dataset, response = datasets[0], responses[0]
        unitary = torch.tensor(
            [[1.0, 1j], [1j, 1.0]], dtype=torch.complex128
        ) / math.sqrt(2.0)
        transformed = replace(
            dataset,
            coulomb_whitening=dataset.coulomb_whitening @ unitary,
            reference_response=unitary.mH @ dataset.reference_response @ unitary,
        )
        before = self.diagnose((dataset,), (response,))
        after = self.diagnose((transformed,), (unitary.mH @ response @ unitary,))
        for key in ("pi_numerator", "pi_denominator", "pi_relative_squared_error"):
            self.assertAlmostEqual(before[key], after[key], places=14)
        for key in Q_FIELDS - {
            "selected_iq",
            "q_weight",
            "physics_hash",
            "whitening_sha256",
            "pi_numerator",
            "pi_denominator",
        }:
            for a, b in zip(before["per_q"][0][key], after["per_q"][0][key]):
                self.assertAlmostEqual(a, b, places=14)
        self.assertNotEqual(
            before["per_q"][0]["whitening_sha256"],
            after["per_q"][0]["whitening_sha256"],
        )

    def test_hash_uses_canonical_complex128_contiguous_little_endian_bytes(self):
        datasets, responses = self.fixture()
        dataset, response = datasets[0], responses[0]
        whitening = dataset.coulomb_whitening
        expected = hashlib.sha256(
            b"".join(
                struct.pack("<dd", complex(value).real, complex(value).imag)
                for row in whitening.tolist()
                for value in row
            )
        ).hexdigest()
        variants = (whitening, whitening.contiguous(), whitening.real)
        for value in variants:
            report = self.diagnose(
                (replace(dataset, coulomb_whitening=value),), (response,)
            )
            self.assertEqual(report["per_q"][0]["whitening_sha256"], expected)
            for phrase in ("SHA-256", "complex128", "C-order", "little-endian"):
                self.assertIn(phrase, report["whitening_sha256_definition"])
        complex_whitening = whitening * 1j
        reports = [
            self.diagnose((replace(dataset, coulomb_whitening=w),), (response,))
            for w in (complex_whitening.conj(), complex_whitening.conj().resolve_conj())
        ]
        self.assertEqual(
            reports[0]["per_q"][0]["whitening_sha256"],
            reports[1]["per_q"][0]["whitening_sha256"],
        )

    def test_no_mutation_and_default_loss_and_gradient_are_identical(self):
        datasets, templates = self.fixture()
        reference = datasets[0].reference_response.detach().clone().requires_grad_()
        datasets = (replace(datasets[0], reference_response=reference), datasets[1])
        parameter = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
        parameter.grad = torch.tensor(7.0, dtype=torch.float64)
        responses = tuple(parameter * t for t in templates)
        snapshots = []

        def snapshot(value):
            if isinstance(value, torch.Tensor):
                snapshots.append(
                    (
                        value,
                        value.detach().clone(),
                        value._version,
                        value.requires_grad,
                        value.grad_fn,
                    )
                )
            elif is_dataclass(value):
                for field in fields(value):
                    snapshot(getattr(value, field.name))
            elif isinstance(value, tuple):
                for item in value:
                    snapshot(item)

        snapshot((datasets, responses))
        defaults = rpa.periodic_rpa_objective.__kwdefaults__.copy()
        before = rpa.periodic_rpa_objective(datasets, responses)
        (gradient_before,) = torch.autograd.grad(
            before.loss, parameter, retain_graph=True
        )
        report = self.diagnose(datasets, responses)
        self.assertTrue(torch.is_grad_enabled())
        self.assert_plain_json(report)
        after = rpa.periodic_rpa_objective(datasets, responses)
        (gradient_after,) = torch.autograd.grad(after.loss, parameter)
        torch.testing.assert_close(before.loss, after.loss, rtol=0, atol=0)
        torch.testing.assert_close(gradient_before, gradient_after, rtol=0, atol=0)
        self.assertEqual(float(parameter.grad), 7.0)
        self.assertIsNone(reference.grad)
        self.assertEqual(
            defaults, {"pi_weight": 1.0, "trace_log_weight": 1.0, "energy_weight": 1.0}
        )
        self.assertEqual(rpa.periodic_rpa_objective.__kwdefaults__, defaults)
        for value, saved, version, requires_grad, grad_fn in snapshots:
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
            self.assertEqual(value._version, version)
            self.assertEqual(value.requires_grad, requires_grad)
            self.assertIs(value.grad_fn, grad_fn)
        report["per_q"][0]["frequency_ha"][0] = 123.0
        self.assertEqual(float(datasets[0].frequency_ha[0]), 0.2)

    def test_invalid_shapes_and_collection_contract_are_rejected(self):
        datasets, responses = self.fixture()
        cases = (
            ((), ()),
            (list(datasets), responses),
            (datasets, list(responses)),
            (datasets, responses[:1]),
            ((object(),), responses[:1]),
            ((datasets[0],), (responses[0][0],)),
            ((datasets[0],), (responses[0][:, :1, :1],)),
            (
                (replace(datasets[0], reference_response=responses[0][:, :1, :1]),),
                responses[:1],
            ),
        )
        for index, (ds, rs) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.diagnose(ds, rs)

    def test_frozen_protocol_frequency_and_q_weight_checks_are_reused(self):
        datasets, responses = self.fixture()
        changes = (
            {"selected_iq": 1},
            {"orbital_sha256": "different"},
            {"auxiliary_basis_sha256": "different"},
            {"q_count": 32},
            {"frequency_ha": datasets[1].frequency_ha.flip(0)},
            {"frequency_weights_ha": datasets[1].frequency_weights_ha * 2},
            {"frequency_weights_ha": datasets[1].frequency_weights_ha[:1]},
            {"q_weight": float("nan")},
            {"q_weight": 1.0},
        )
        for change in changes:
            ds = (datasets[0], replace(datasets[1], **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.diagnose(ds, responses)

    def test_nonfinite_nonhermitian_and_noncausal_candidate_or_reference_rejected(self):
        datasets, responses = self.fixture()
        invalid = [responses[0].clone() for _ in range(5)]
        invalid[0][0, 0, 0] = float("nan")
        invalid[1][0, 0, 0] = float("inf")
        invalid[2][0, 0, 1] = 0.2j
        invalid[3][0, 0, 0] = 0.2
        invalid[4][0, 0, 0] = 1.1
        invalid.append(responses[0].to(torch.complex64))
        for index, value in enumerate(invalid):
            for reference in (False, True):
                ds = (
                    (replace(datasets[0], reference_response=value),)
                    if reference
                    else datasets[:1]
                )
                rs = responses[:1] if reference else (value,)
                with self.subTest(index=index, reference=reference), self.assertRaises(
                    ValueError
                ):
                    self.diagnose(ds, rs)

    def test_zero_local_reference_is_json_finite(self):
        datasets, responses = self.fixture()
        zero = torch.zeros_like(datasets[0].reference_response)
        datasets = (replace(datasets[0], reference_response=zero), datasets[1])
        candidate = zero.clone()
        candidate[0] = responses[0][0]
        report = self.diagnose(datasets, (candidate, responses[1]))
        record = report["per_q"][0]
        self.assertEqual(record["reference_squared_norm"], [0.0, 0.0])
        self.assertEqual(record["relative_pi_error"], [None, 0.0])
        self.assertEqual(record["pi_denominator"], 0.0)
        self.assertGreater(record["pi_numerator"], 0.0)
        self.assert_plain_json(report)

    def test_zero_global_reference_retains_objective_rejection(self):
        datasets, responses = self.fixture()
        datasets = tuple(
            replace(d, reference_response=torch.zeros_like(d.reference_response))
            for d in datasets
        )
        with self.assertRaisesRegex(ValueError, "zero or nonfinite norm"):
            self.diagnose(datasets, responses)

    def test_finite_q_selection_excludes_actual_gamma_and_breaks_ties_by_iq(self):
        datasets, _ = self.fixture()
        first = replace(datasets[0], q_weight=0.1)
        tied = replace(first, selected_iq=3, qpoint=(0.0, 1e-15, 0.0))
        gamma = replace(first, selected_iq=9, qpoint=(0.0, -0.0, 0.0), q_weight=0.2)
        ds = (tied, gamma, first)
        rs = (
            2 * tied.reference_response,
            5 * gamma.reference_response,
            2 * first.reference_response,
        )
        report = self.diagnose(ds, rs)
        self.assertGreater(
            report["per_q"][1]["pi_numerator"], report["per_q"][2]["pi_numerator"]
        )
        self.assertEqual(
            report["largest_weighted_residual_finite_q"], report["per_q"][2]
        )
        reversed_report = self.diagnose(ds[::-1], rs[::-1])
        self.assertEqual(
            reversed_report["largest_weighted_residual_finite_q"]["selected_iq"], 1
        )
        self.assertIsNone(
            self.diagnose((gamma,), (rs[1],))["largest_weighted_residual_finite_q"]
        )

    def test_finite_q_selection_uses_integrated_weighted_not_peak_residual(self):
        datasets, _ = self.fixture()
        first = replace(datasets[0], q_weight=0.1)
        second = replace(first, selected_iq=2, q_weight=0.4)
        a = first.reference_response.clone()
        b = second.reference_response.clone()
        a[0] *= 3
        b[1] *= 3
        report = self.diagnose((first, second), (a, b))
        self.assertGreater(
            report["per_q"][0]["residual_squared_norm"][0],
            report["per_q"][1]["residual_squared_norm"][1],
        )
        self.assertEqual(report["largest_weighted_residual_finite_q"]["selected_iq"], 2)


if __name__ == "__main__":
    unittest.main()
