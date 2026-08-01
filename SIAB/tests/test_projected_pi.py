from dataclasses import replace
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from projected_pi import (
    NormalizedPhysicalFamilyProjectedPi,
    ProjectedPiEvaluator,
)
from sternheimer_data import PrimitiveBlock, SternheimerData
from sternheimer_source_data import SternheimerSourceData
from sternheimer_source_pair import pair_response_and_source


def provenance():
    return {
        "abacus_commit": "1" * 40,
        "auxiliary_basis_sha256": "a" * 64,
        "cell_bohr": [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0],
        "ecut_ry": 25.0,
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


def make_pair():
    block = PrimitiveBlock("H", 0, 0, 0, 3, 0)
    overlap = torch.tensor(
        [
            [2.0 + 0.0j, 0.2 + 0.1j, 0.0 + 0.0j],
            [0.2 - 0.1j, 1.5 + 0.0j, 0.0 + 0.1j],
            [0.0 + 0.0j, 0.0 - 0.1j, 1.2 + 0.0j],
        ],
        dtype=torch.complex128,
    )
    d = torch.tensor(
        [
            [0.8 + 0.1j, 0.1 - 0.2j, 0.2 + 0.3j],
            [0.4 - 0.2j, 0.3 + 0.1j, -0.1 + 0.2j],
        ],
        dtype=torch.complex128,
    )
    q = torch.tensor(
        [
            [
                [0.5 + 0.2j, -0.2 + 0.1j, 0.3 - 0.1j],
                [0.1 - 0.3j, 0.4 + 0.2j, -0.2 + 0.1j],
            ],
            [
                [0.2 - 0.1j, 0.3 + 0.4j, -0.1 + 0.2j],
                [-0.3 + 0.2j, 0.1 - 0.2j, 0.5 + 0.3j],
            ],
        ],
        dtype=torch.complex128,
    ).unsqueeze(1)
    source = SternheimerSourceData(
        format_version=1,
        grid_volume_bohr3=0.125,
        blocks=(block,),
        occupied_state=torch.tensor([0, 0], dtype=torch.int64),
        auxiliary_channel=torch.tensor([0, 1], dtype=torch.int64),
        occupation=torch.tensor([2.0, 2.0], dtype=torch.float64),
        norm=torch.tensor([1.2, 0.8], dtype=torch.float64),
        d=d,
        overlap=overlap,
        provenance=provenance(),
    )
    response = SternheimerData(
        format_version=1,
        grid_volume_bohr3=0.125,
        blocks=(block,),
        occupied_state=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        auxiliary_channel=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        frequency_ha=torch.tensor([0.5, 0.5, 1.5, 1.5], dtype=torch.float64),
        occupation=torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64),
        frequency_weight=torch.tensor([0.3, 0.3, 0.7, 0.7], dtype=torch.float64),
        norm=torch.tensor([1.0, 1.1, 0.9, 1.2], dtype=torch.float64),
        q=q.reshape(4, 3),
        overlap=overlap,
        provenance=provenance(),
    )
    return pair_response_and_source(response, source), d.unsqueeze(0), q


def coefficients(value=None, requires_grad=False):
    if value is None:
        value = torch.tensor(
            [[1.0, 0.1], [0.2, 0.9], [0.3, -0.2]],
            dtype=torch.float64,
        )
    value = value.detach().clone().requires_grad_(requires_grad)
    return {"H": [value]}


def direct_values(d, q, overlap, coefficient):
    c = coefficient.to(torch.complex128)
    g = c.mH @ overlap @ c
    candidate_a = []
    reference_a = []
    s_inverse = torch.linalg.inv(overlap)
    for frequency in range(q.shape[0]):
        candidate = torch.zeros((2, 2), dtype=torch.complex128)
        reference = torch.zeros((2, 2), dtype=torch.complex128)
        for occupied in range(q.shape[1]):
            candidate += (
                2.0
                * (d[occupied] @ c)
                @ torch.linalg.inv(g)
                @ (q[frequency, occupied] @ c).mH
            )
            reference += (
                2.0
                * d[occupied]
                @ s_inverse
                @ q[frequency, occupied].mH
            )
        candidate_a.append(candidate)
        reference_a.append(reference)
    candidate_a = torch.stack(candidate_a)
    reference_a = torch.stack(reference_a)
    candidate_pi = candidate_a + candidate_a.mH
    reference_pi = reference_a + reference_a.mH
    squared_error = torch.sum(
        torch.abs(candidate_pi - reference_pi) ** 2, dim=(1, 2)
    )
    reference_norm = torch.sum(torch.abs(reference_pi) ** 2, dim=(1, 2))
    frequency_loss = squared_error / reference_norm
    weight = torch.tensor([0.3, 0.7], dtype=torch.float64)
    loss = torch.sum(weight * squared_error) / torch.sum(
        weight * reference_norm
    )
    return {
        "candidate_a": candidate_a,
        "reference_a": reference_a,
        "candidate_pi": candidate_pi,
        "reference_pi": reference_pi,
        "frequency_loss": frequency_loss,
        "loss": loss,
    }


class ProjectedPiTest(unittest.TestCase):
    def setUp(self):
        self.pair, self.d, self.q = make_pair()
        self.coefficient = coefficients()["H"][0]

    def test_matches_direct_complex_matrix_formula(self):
        result = ProjectedPiEvaluator(self.pair).evaluate(
            coefficients(self.coefficient)
        )
        expected = direct_values(
            self.d,
            self.q,
            self.pair.source.overlap,
            self.coefficient,
        )

        for field in (
            "candidate_a",
            "reference_a",
            "candidate_pi",
            "reference_pi",
            "frequency_loss",
            "loss",
        ):
            torch.testing.assert_close(
                getattr(result, field),
                expected[field],
                rtol=1.0e-13,
                atol=1.0e-13,
            )
        self.assertEqual(result.reference_rank, 3)
        torch.testing.assert_close(
            result.frequency_ha,
            torch.tensor([0.5, 1.5], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.frequency_weight,
            torch.tensor([0.3, 0.7], dtype=torch.float64),
        )

    def test_pi_is_hermitian_and_uses_a_plus_adjoint(self):
        result = ProjectedPiEvaluator(self.pair).evaluate(
            coefficients(self.coefficient)
        )
        torch.testing.assert_close(
            result.candidate_pi,
            result.candidate_pi.mH,
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertGreater(
            float(torch.max(torch.abs(result.candidate_pi - 2.0 * result.candidate_a))),
            1.0e-3,
        )

    def test_occupation_is_counted_once(self):
        result = ProjectedPiEvaluator(self.pair).evaluate(
            coefficients(self.coefficient)
        )
        expected = direct_values(
            self.d,
            self.q,
            self.pair.source.overlap,
            self.coefficient,
        )
        torch.testing.assert_close(result.reference_a, expected["reference_a"])
        self.assertGreater(
            float(torch.max(torch.abs(result.reference_a - 2.0 * expected["reference_a"]))),
            1.0e-3,
        )

    def test_common_source_response_phase_is_invariant(self):
        phase = torch.exp(torch.tensor(0.37j, dtype=torch.complex128))
        source = replace(self.pair.source, d=self.pair.source.d * phase)
        response = replace(self.pair.response, q=self.pair.response.q * phase)
        phased_pair = pair_response_and_source(response, source)

        original = ProjectedPiEvaluator(self.pair).evaluate(
            coefficients(self.coefficient)
        )
        phased = ProjectedPiEvaluator(phased_pair).evaluate(
            coefficients(self.coefficient)
        )
        for field in (
            "candidate_a",
            "reference_a",
            "candidate_pi",
            "reference_pi",
            "frequency_loss",
            "loss",
        ):
            torch.testing.assert_close(
                getattr(phased, field),
                getattr(original, field),
                rtol=1.0e-13,
                atol=1.0e-13,
            )

    def test_coefficient_gradient_matches_centered_difference(self):
        candidate = coefficients(self.coefficient, requires_grad=True)
        result = ProjectedPiEvaluator(self.pair).evaluate(candidate)
        result.loss.backward()
        analytic = float(candidate["H"][0].grad[1, 0])

        step = 1.0e-6
        plus = self.coefficient.clone()
        minus = self.coefficient.clone()
        plus[1, 0] += step
        minus[1, 0] -= step
        evaluator = ProjectedPiEvaluator(self.pair)
        finite_difference = float(
            (
                evaluator.evaluate(coefficients(plus)).loss
                - evaluator.evaluate(coefficients(minus)).loss
            )
            / (2.0 * step)
        )
        self.assertAlmostEqual(analytic, finite_difference, delta=2.0e-7)

    def test_rejects_singular_candidate_overlap(self):
        singular = torch.tensor(
            [[1.0, 1.0], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float64
        )
        with self.assertRaisesRegex(RuntimeError, "candidate overlap"):
            ProjectedPiEvaluator(self.pair).evaluate(coefficients(singular))

    def test_rejects_incomplete_response_rectangle(self):
        response = replace(
            self.pair.response,
            occupied_state=self.pair.response.occupied_state[:-1],
            auxiliary_channel=self.pair.response.auxiliary_channel[:-1],
            frequency_ha=self.pair.response.frequency_ha[:-1],
            occupation=self.pair.response.occupation[:-1],
            frequency_weight=self.pair.response.frequency_weight[:-1],
            norm=self.pair.response.norm[:-1],
            q=self.pair.response.q[:-1],
        )
        incomplete_pair = pair_response_and_source(response, self.pair.source)
        with self.assertRaisesRegex(ValueError, "complete response rectangle"):
            ProjectedPiEvaluator(incomplete_pair)

    def test_rejects_inconsistent_frequency_weights(self):
        weights = self.pair.response.frequency_weight.clone()
        weights[1] = 0.4
        response = replace(self.pair.response, frequency_weight=weights)
        inconsistent_pair = pair_response_and_source(response, self.pair.source)
        with self.assertRaisesRegex(ValueError, "frequency weights differ"):
            ProjectedPiEvaluator(inconsistent_pair)

    def test_rejects_nonpositive_primitive_reference_norm(self):
        source = replace(
            self.pair.source,
            d=torch.zeros_like(self.pair.source.d),
        )
        zero_pair = pair_response_and_source(self.pair.response, source)
        with self.assertRaisesRegex(RuntimeError, "primitive-reference norm"):
            ProjectedPiEvaluator(zero_pair)

    def test_equal_family_aggregation(self):
        family = NormalizedPhysicalFamilyProjectedPi(
            (("H", self.pair), ("H2", self.pair))
        )
        result = family.evaluate(coefficients(self.coefficient))
        single = ProjectedPiEvaluator(self.pair).evaluate(
            coefficients(self.coefficient)
        )
        torch.testing.assert_close(result.loss, 2.0 * single.loss)
        self.assertEqual(tuple(result.results), ("H", "H2"))

    def test_rejects_empty_or_duplicate_family_names(self):
        for named_pairs in (
            (),
            (("", self.pair),),
            (("H", self.pair), ("H", self.pair)),
        ):
            with self.subTest(named_pairs=named_pairs):
                with self.assertRaisesRegex(ValueError, "physical family names"):
                    NormalizedPhysicalFamilyProjectedPi(named_pairs)


if __name__ == "__main__":
    unittest.main()
