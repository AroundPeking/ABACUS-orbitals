import math
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from response_selection import (
    CandidateGain,
    ResponseTargetFamily,
    append_response_shell,
    borrowing_gap,
    evaluate_response_candidates,
    normalized_family_loss,
    score_candidate,
    select_best_candidate,
)
from sternheimer_data import PrimitiveBlock
from sternheimer_spillage import (
    RadialResidualSpectrum,
    evaluate_spillage_for_columns,
)
from test_sternheimer_spillage import make_sternheimer_data


def one_center_family_data():
    return make_sternheimer_data(
        [PrimitiveBlock("H", 0, 0, 0, 3, 0)],
        [math.sqrt(2.0), math.sqrt(6.0), 0.0],
        norm=10.0,
    )


def fragment_ghost_data():
    blocks = (
        PrimitiveBlock("H", 0, 0, 0, 2, 0),
        PrimitiveBlock("H", 1, 0, 0, 2, 2),
    )
    return make_sternheimer_data(
        blocks,
        [0.0, math.sqrt(0.4), 0.0, math.sqrt(0.4)],
        norm=1.0,
    )


def fixed_dzp_coefficients():
    return {
        "H": [
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        ]
    }


def expanded_coefficients():
    return {"H": [torch.eye(2, dtype=torch.float64)]}


def response_coefficients():
    return {
        "H": [
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            torch.empty((2, 0), dtype=torch.float64),
        ]
    }


def response_spectrum(l, eigenvalue, coefficients):
    coefficients = torch.as_tensor(coefficients, dtype=torch.float64)
    eigenvalues = torch.tensor([eigenvalue], dtype=torch.float64)
    cumulative = torch.tensor(
        [1.0 if eigenvalue > 0.0 else 0.0], dtype=torch.float64
    )
    return RadialResidualSpectrum(
        element="H",
        atom_index=None,
        l=l,
        magnetic_channels=tuple(range(-l, l + 1)),
        numerical_rank=1,
        eigenvalues=eigenvalues,
        cumulative_capture=cumulative,
        coefficients=coefficients,
        overlap_relative_deviation=0.0,
        atom_indices=(0,),
    )


def response_target_families():
    physical_data = make_sternheimer_data(
        [PrimitiveBlock("H", 0, 0, 0, 2, 0)],
        [0.0, 1.0],
        norm=1.0,
    )
    return (
        ResponseTargetFamily("atom", (physical_data,), "physical"),
        ResponseTargetFamily("multicenter", (physical_data,), "physical"),
    )


class ExplicitProjectorTest(unittest.TestCase):
    def test_selected_columns_use_the_full_overlap_metric(self):
        data = fragment_ghost_data()
        coefficients = expanded_coefficients()

        own = evaluate_spillage_for_columns(
            data,
            coefficients,
            include=lambda label: label.atom_index == 0,
        )
        all_centers = evaluate_spillage_for_columns(
            data,
            coefficients,
            include=lambda label: True,
        )

        torch.testing.assert_close(
            own.weighted_residual,
            torch.tensor(0.6, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-13,
        )
        torch.testing.assert_close(
            all_centers.weighted_residual,
            torch.tensor(0.2, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-13,
        )

    def test_rejects_an_empty_selected_projector(self):
        with self.assertRaisesRegex(
            ValueError, "selected projector contains no orbital columns"
        ):
            evaluate_spillage_for_columns(
                fragment_ghost_data(),
                fixed_dzp_coefficients(),
                include=lambda label: False,
            )


class ResponseTargetFamilyTest(unittest.TestCase):
    def test_family_loss_is_normalized_to_fixed_dzp(self):
        data = one_center_family_data()
        family = ResponseTargetFamily("atom", (data,), "physical")
        fixed_dzp = {
            "H": [
                torch.tensor(
                    [[1.0], [0.0], [0.0]], dtype=torch.float64
                )
            ]
        }
        current = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        self.assertAlmostEqual(
            normalized_family_loss(family, current, fixed_dzp),
            0.25,
            places=13,
        )

    def test_borrowing_gap_is_own_minus_all_with_one_dzp_scale(self):
        family = ResponseTargetFamily(
            "fragment_ghost",
            (fragment_ghost_data(),),
            "ghost",
            real_atom_index=0,
        )

        value = borrowing_gap(
            family,
            expanded_coefficients(),
            fixed_dzp_coefficients(),
        )

        self.assertAlmostEqual(value, 0.4, places=13)

    def test_family_contract_rejects_wrong_roles_and_missing_real_atom(self):
        data = fragment_ghost_data()
        with self.assertRaisesRegex(ValueError, "role"):
            ResponseTargetFamily("bad", (data,), "energy")
        with self.assertRaisesRegex(ValueError, "real_atom_index"):
            ResponseTargetFamily("ghost", (data,), "ghost")


class CandidateScoreTest(unittest.TestCase):
    def test_score_is_physical_gain_per_ao_function(self):
        candidate = CandidateGain(
            l=3,
            mode=0,
            atom=0.9,
            multicenter=0.4,
        )

        self.assertAlmostEqual(score_candidate(candidate), 1.3 / 7.0, places=14)
        self.assertFalse(hasattr(candidate, "balance"))

    def test_selector_prefers_more_gain_per_actual_ao_function(self):
        d = CandidateGain(
            l=2,
            mode=0,
            atom=0.50,
            multicenter=0.0,
        )
        f = CandidateGain(
            l=3,
            mode=0,
            atom=0.84,
            multicenter=0.0,
        )

        self.assertEqual(select_best_candidate([d, f]).l, 3)

    def test_tie_break_is_lower_cost_then_l_then_mode(self):
        candidates = [
            CandidateGain(2, 0, 5.0, 0.0),
            CandidateGain(1, 2, 3.0, 0.0),
            CandidateGain(1, 0, 3.0, 0.0),
        ]

        self.assertEqual(select_best_candidate(candidates).key, (1, 0))

    def test_selector_rejects_nonpositive_physical_gain(self):
        values = [
            CandidateGain(0, 0, -0.1, 0.1),
            CandidateGain(1, 0, -0.2, 0.1),
        ]

        with self.assertRaisesRegex(RuntimeError, "no admissible positive-score"):
            select_best_candidate(values)


class CandidateEvaluatorTest(unittest.TestCase):
    def test_append_canonicalizes_mode_without_changing_existing_columns(self):
        current = response_coefficients()
        before = current["H"][0].clone()
        spectrum = response_spectrum(0, 1.0, [[0.0], [-1.0]])

        appended = append_response_shell(current, spectrum, mode=0)

        self.assertTrue(torch.equal(current["H"][0], before))
        self.assertTrue(torch.equal(appended["H"][0][:, :1], before))
        torch.testing.assert_close(
            appended["H"][0][:, 1],
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_evaluator_returns_gains_and_deterministic_rejections(self):
        atom, multicenter = response_target_families()
        current = response_coefficients()
        spectra = (
            response_spectrum(1, 0.0, [[0.0], [1.0]]),
            response_spectrum(0, 1.0, [[0.0], [1.0]]),
        )

        values = evaluate_response_candidates(
            spectra,
            current,
            current,
            atom,
            multicenter,
        )

        self.assertEqual([value.gain.key for value in values], [(0, 0), (1, 0)])
        self.assertTrue(values[0].admissible)
        self.assertAlmostEqual(values[0].gain.atom, 1.0, places=13)
        self.assertAlmostEqual(values[0].gain.multicenter, 1.0, places=13)
        self.assertFalse(hasattr(values[0].gain, "balance"))
        self.assertAlmostEqual(values[0].score, 2.0, places=13)
        self.assertIsNone(values[0].rejection_reason)
        self.assertFalse(values[1].admissible)
        self.assertEqual(
            values[1].rejection_reason,
            "residual eigenvalue is not positive",
        )

    def test_evaluator_scores_representable_gain_above_family_floors(self):
        atom, multicenter = response_target_families()
        current = response_coefficients()

        value = evaluate_response_candidates(
            (response_spectrum(0, 1.0, [[0.0], [1.0]]),),
            current,
            current,
            atom,
            multicenter,
            atom_floor=0.5,
            multicenter_floor=0.0,
        )[0]

        self.assertAlmostEqual(value.gain.atom, 2.0, places=13)
        self.assertAlmostEqual(value.gain.multicenter, 1.0, places=13)
        self.assertAlmostEqual(value.score, 3.0, places=13)


if __name__ == "__main__":
    unittest.main()
