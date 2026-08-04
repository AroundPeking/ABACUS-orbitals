import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from common import info
import optimization_loss
from freeze_orbitals import validate_freeze_orbitals, zero_frozen_gradients
from IO.read_json import read_json
from optimization_loss import (
    LOSS_DEFAULTS,
    compose_loss,
    constraints_satisfied,
    normalize_loss_config,
    selection_component,
)
from opt_orbital_converge import Opt_Orbital_Converge
from opt_orbital_spillage import Opt_Orbital_Spillage
from projected_pi_optimization import ProjectedPiOptimizationResult
from radial_locality import RadialLocalityResult
from sternheimer_spillage import SternheimerLossResult


def make_legacy_spillage(with_linear=False):
    info_stru = [info(Na={"H": 1}, Nb_true=1, weight=torch.tensor([1.0]))]
    info_element = {"H": info(Nl=1, Ne=2, Nu=[1])}
    q = {"H": [torch.tensor([[0.8, 0.6]], dtype=torch.complex128)]}
    s = {
        ("H", "H"): [
            [torch.eye(2, dtype=torch.complex128).reshape(1, 1, 2, 1, 1, 2)]
        ]
    }
    c = {
        "H": [
            torch.tensor(
                [[1.0], [0.0]], dtype=torch.float64, requires_grad=True
            )
        ]
    }
    target = [torch.tensor([1.0], dtype=torch.float64)]
    file_list = {"origin": ["synthetic"]}
    if with_linear:
        file_list["linear"] = [["synthetic-linear"]]

    loss = Opt_Orbital_Spillage(
        info_stru, info_element, {"same_band": True}, "one", file_list
    )
    loss.set_QSVI([q], [s], target)
    if with_linear:
        q_linear = {
            "H": [torch.tensor([[0.1, 0.0]], dtype=torch.complex128)]
        }
        s_linear = {
            ("H", "H"): [
                [torch.zeros((1, 1, 2, 1, 1, 2), dtype=torch.complex128)]
            ]
        }
        loss.set_QSVI_linear(
            [[q_linear]],
            [[s_linear]],
            [[torch.tensor([0.0], dtype=torch.float64)]],
        )
    return loss, c


def make_converge_case(mode=None, max_steps=8, freeze_specs=None):
    info_stru = [info(Na={"H": 1}, Nb_true=1, weight=torch.tensor([1.0]))]
    info_element = {"H": info(Nl=1, Ne=2, Nu=[2])}
    q = {"H": [torch.tensor([[0.8, 0.6]], dtype=torch.complex128)]}
    s = {
        ("H", "H"): [
            [torch.eye(2, dtype=torch.complex128).reshape(1, 1, 2, 1, 1, 2)]
        ]
    }
    c = {
        "H": [
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float64,
                requires_grad=True,
            )
        ]
    }
    stage = {
        "optimizer": "Adam",
        "kwargs": {"lr": 0.05},
        "cal_T": False,
        "norm": "one",
        "max_steps": max_steps,
    }
    if mode is not None:
        stage["loss"] = normalize_loss_config({"mode": mode})
    c_init = {"init_from_file": False}
    if freeze_specs is not None:
        c_init["freeze_orbitals"] = copy.deepcopy(freeze_specs)

    converge = Opt_Orbital_Converge()
    converge.set_info(
        {"origin": ["synthetic"]},
        [stage],
        info_stru,
        c_init,
        {"same_band": True},
    )
    converge.set_info_element(info_element)
    converge.set_QSVI([q], [s], [torch.tensor([1.0], dtype=torch.float64)])
    return converge, c


def make_single_orbital_converge(max_steps=11):
    info_stru = [info(Na={"H": 1}, Nb_true=1, weight=torch.tensor([1.0]))]
    info_element = {"H": info(Nl=1, Ne=2, Nu=[1])}
    q = {"H": [torch.tensor([[0.8, 0.6]], dtype=torch.complex128)]}
    s = {
        ("H", "H"): [
            [torch.eye(2, dtype=torch.complex128).reshape(1, 1, 2, 1, 1, 2)]
        ]
    }
    c = {
        "H": [
            torch.tensor(
                [[1.0], [0.0]], dtype=torch.float64, requires_grad=True
            )
        ]
    }
    stage = {
        "optimizer": "Adam",
        "kwargs": {"lr": 0.1},
        "cal_T": False,
        "norm": "one",
        "max_steps": max_steps,
        "loss": normalize_loss_config(
            {
                "mode": "st_constrained",
                "tau_dft": 0.0,
                "tau_dpsi": 0.0,
                "constraint_penalty_dft": 0.0,
                "constraint_penalty_dpsi": 0.0,
            }
        ),
    }
    converge = Opt_Orbital_Converge()
    converge.set_info(
        {"origin": ["synthetic"]},
        [stage],
        info_stru,
        {"init_from_file": False},
        {"same_band": True},
    )
    converge.set_info_element(info_element)
    converge.set_QSVI([q], [s], [torch.tensor([1.0], dtype=torch.float64)])
    return converge, c


class QuadraticSternheimer:
    def __init__(self, max_condition=2.0):
        self.max_condition = max_condition

    def evaluate(self, c):
        target = torch.tensor(
            [[0.75, 0.5], [0.25, 0.5]], dtype=c["H"][0].dtype
        )
        loss = torch.sum((c["H"][0] - target) ** 2)
        return SternheimerLossResult(
            loss=loss,
            weighted_residual=loss,
            weighted_norm=torch.ones_like(loss),
            max_condition=self.max_condition,
        )


class QuadraticProjectedPi:
    def __init__(self, max_condition=2.0):
        self.max_condition = max_condition

    def evaluate(self, c):
        target = torch.tensor(
            [[0.75, 0.5], [0.25, 0.5]], dtype=c["H"][0].dtype
        )
        loss = torch.sum((c["H"][0] - target) ** 2)
        frequency_loss = torch.stack((loss, 1.5 * loss))
        return ProjectedPiOptimizationResult(
            loss=loss,
            max_condition=self.max_condition,
            frequency_ha=torch.tensor([0.1, 1.0], dtype=loss.dtype),
            frequency_loss=frequency_loss,
            family_results={"H": object(), "H2": object()},
        )


class SingleOrbitalSternheimer:
    def evaluate(self, c):
        target = torch.tensor([[0.0], [1.0]], dtype=c["H"][0].dtype)
        loss = torch.sum((c["H"][0] - target) ** 2)
        return SternheimerLossResult(
            loss=loss,
            weighted_residual=loss,
            weighted_norm=torch.ones_like(loss),
            max_condition=1.0,
        )


class OpposingFrequencySternheimer:
    def evaluate(self, c):
        x = c["H"][0][0, 0]
        loss = x.square()
        low_frequency_loss = 0.4 - 0.2 * x
        one = torch.ones_like(loss).reshape(1)
        return SternheimerLossResult(
            loss=loss,
            weighted_residual=loss,
            weighted_norm=torch.ones_like(loss),
            max_condition=1.0,
            frequency_ha=torch.tensor(
                [0.1], dtype=loss.dtype, device=loss.device
            ),
            frequency_residual=low_frequency_loss.reshape(1),
            frequency_norm=one,
            frequency_loss=low_frequency_loss.reshape(1),
        )


class QuadraticLocality:
    def __init__(self, max_condition=3.0):
        self.max_condition = max_condition

    def evaluate(self, c):
        loss = torch.sum(c["H"][0][:, 1].square())
        return RadialLocalityResult(
            loss=loss,
            max_condition=self.max_condition,
            by_channel={},
        )


class NamedLegacyComponentsTest(unittest.TestCase):
    def test_origin_only_named_components_preserve_total(self):
        loss, c = make_legacy_spillage()

        components = loss.cal_components(c)

        self.assertEqual(set(components), {"dft_origin", "dft_dpsi"})
        self.assertAlmostEqual(components["dft_origin"].item(), 0.2, places=14)
        self.assertEqual(components["dft_dpsi"].item(), 0.0)
        self.assertAlmostEqual(loss.cal_Spillage(c).item(), 0.4, places=14)

    def test_linear_component_and_reconstructed_gradients(self):
        loss, c = make_legacy_spillage(with_linear=True)

        components = loss.cal_components(c)
        reconstructed = 2 * components["dft_origin"] + components["dft_dpsi"]
        total = loss.cal_Spillage(c)
        grad_total = torch.autograd.grad(total, c["H"][0], retain_graph=True)[0]
        grad_reconstructed = torch.autograd.grad(reconstructed, c["H"][0])[0]

        self.assertAlmostEqual(components["dft_origin"].item(), 0.2, places=14)
        self.assertAlmostEqual(components["dft_dpsi"].item(), 0.2, places=14)
        self.assertAlmostEqual(total.item(), 0.6, places=14)
        torch.testing.assert_close(grad_total, grad_reconstructed)


class FreezeOrbitalsTest(unittest.TestCase):
    def test_real_adam_step_freezes_exact_radial_column(self):
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
            ]
        }
        indices = validate_freeze_orbitals(
            [{"element": "H", "l": 0, "zeta": 1}], c
        )
        fixed_before = c["H"][0][:, 0].detach().clone()
        variable_before = c["H"][0][:, 1].detach().clone()
        optimizer = torch.optim.Adam(c["H"], lr=0.1)

        optimizer.zero_grad()
        torch.sum((c["H"][0] - 0.5) ** 2).backward()
        zero_frozen_gradients(c, indices)
        optimizer.step()

        self.assertTrue(torch.equal(c["H"][0][:, 0], fixed_before))
        self.assertFalse(torch.equal(c["H"][0][:, 1], variable_before))

    def test_rejects_invalid_and_duplicate_specs(self):
        c = {"H": [torch.ones((2, 2), dtype=torch.float64, requires_grad=True)]}
        invalid_specs = (
            "not-a-list",
            [{"element": "He", "l": 0, "zeta": 1}],
            [{"element": "H", "l": 1, "zeta": 1}],
            [{"element": "H", "l": 0, "zeta": 0}],
            [{"element": "H", "l": 0, "zeta": 3}],
            [{"element": "H", "l": True, "zeta": 1}],
            [{"element": "H", "l": 0.0, "zeta": 1}],
            [{"element": "H", "l": 0, "zeta": False}],
            [{"element": "H", "l": 0, "zeta": 1, "extra": 2}],
            [{"element": "H", "l": 0}],
            [
                {"element": "H", "l": 0, "zeta": 1},
                {"element": "H", "l": 0, "zeta": 1},
            ],
        )
        for specs in invalid_specs:
            with self.subTest(specs=specs):
                with self.assertRaisesRegex((TypeError, ValueError), "invalid|duplicate"):
                    validate_freeze_orbitals(specs, c)

    def test_rejects_non_string_and_empty_element_labels(self):
        coefficient = torch.ones(
            (2, 1), dtype=torch.float64, requires_grad=True
        )
        c = {"H": [coefficient], 0: [coefficient], "": [coefficient]}

        for element in ([], 0, ""):
            spec = {"element": element, "l": 0, "zeta": 1}
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError) as context:
                    validate_freeze_orbitals([spec], c)
                self.assertIn(repr((element, 0, 1)), str(context.exception))

    def test_rejects_stale_indices_and_missing_gradient(self):
        c = {"H": [torch.ones((2, 1), dtype=torch.float64, requires_grad=True)]}
        with self.assertRaisesRegex(ValueError, "invalid.*H.*0.*1"):
            zero_frozen_gradients(c, frozenset({("H", 0, 1)}))
        with self.assertRaisesRegex(ValueError, "gradient.*H.*0.*0"):
            zero_frozen_gradients(c, frozenset({("H", 0, 0)}))


class OptimizationLossTest(unittest.TestCase):
    def test_pi_dpsi_joint_config_is_mode_specific_and_strict(self):
        for old_mode in ("st_only", "st_constrained", "st_dpsi_joint"):
            with self.subTest(old_mode=old_mode):
                self.assertEqual(
                    normalize_loss_config({"mode": old_mode}),
                    {**LOSS_DEFAULTS, "mode": old_mode},
                )

        config = normalize_loss_config({"mode": "pi_dpsi_joint"})
        self.assertEqual(
            config,
            {
                **LOSS_DEFAULTS,
                "mode": "pi_dpsi_joint",
                "projected_pi_rank_tolerance": 1.0e-12,
            },
        )
        for value in (0.0, 1.0, -1.0, float("nan"), True):
            with self.subTest(rank_tolerance=value):
                with self.assertRaises((TypeError, ValueError)):
                    normalize_loss_config(
                        {
                            "mode": "pi_dpsi_joint",
                            "projected_pi_rank_tolerance": value,
                        }
                    )
        for field, value in (
            ("low_frequency_guard_weight", 1.0),
            ("low_frequency_guard_tolerance", 0.1),
            ("radial_tail_weight", 1.0),
            ("radial_tail_radius", 4.0),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    normalize_loss_config(
                        {"mode": "pi_dpsi_joint", field: value}
                    )

    @staticmethod
    def rpa_sensitive_config(**updates):
        config = {
            "mode": "pi_rpa_sensitive_joint",
            "projected_pi_rank_tolerance": 1.0e-12,
            "projected_pi_sensitivity_alpha": 0.25,
            "joint_dpsi_weight": 0.02,
        }
        config.update(updates)
        return config

    def test_rpa_sensitive_joint_accepts_exact_required_contract(self):
        config = normalize_loss_config(self.rpa_sensitive_config())

        self.assertEqual(config["mode"], "pi_rpa_sensitive_joint")
        self.assertEqual(config["projected_pi_rank_tolerance"], 1.0e-12)
        self.assertEqual(config["projected_pi_sensitivity_alpha"], 0.25)
        self.assertEqual(config["joint_dpsi_weight"], 0.02)

    def test_rpa_sensitive_joint_requires_explicit_rank_tolerance(self):
        config = self.rpa_sensitive_config()
        config.pop("projected_pi_rank_tolerance")

        with self.assertRaisesRegex(
            ValueError,
            "pi_rpa_sensitive_joint requires projected_pi_rank_tolerance",
        ):
            normalize_loss_config(config)

    def test_rpa_sensitive_joint_requires_explicit_alpha(self):
        config = self.rpa_sensitive_config()
        config.pop("projected_pi_sensitivity_alpha")

        with self.assertRaisesRegex(
            ValueError,
            "pi_rpa_sensitive_joint requires projected_pi_sensitivity_alpha",
        ):
            normalize_loss_config(config)

    def test_rpa_sensitive_joint_requires_explicit_dpsi_weight(self):
        config = self.rpa_sensitive_config()
        config.pop("joint_dpsi_weight")

        with self.assertRaisesRegex(
            ValueError,
            "pi_rpa_sensitive_joint requires joint_dpsi_weight",
        ):
            normalize_loss_config(config)

    def test_rpa_sensitive_joint_accepts_alpha_zero(self):
        config = normalize_loss_config(
            self.rpa_sensitive_config(projected_pi_sensitivity_alpha=0.0)
        )

        self.assertEqual(config["projected_pi_sensitivity_alpha"], 0.0)

    def test_rpa_sensitive_joint_accepts_alpha_one(self):
        config = normalize_loss_config(
            self.rpa_sensitive_config(projected_pi_sensitivity_alpha=1.0)
        )

        self.assertEqual(config["projected_pi_sensitivity_alpha"], 1.0)

    def test_rpa_sensitive_joint_rejects_nonfinite_alpha(self):
        for alpha in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    ValueError, "projected_pi_sensitivity_alpha.*finite"
                ):
                    normalize_loss_config(
                        self.rpa_sensitive_config(
                            projected_pi_sensitivity_alpha=alpha
                        )
                    )

    def test_rpa_sensitive_joint_rejects_alpha_below_zero(self):
        with self.assertRaisesRegex(
            ValueError, "projected_pi_sensitivity_alpha"
        ):
            normalize_loss_config(
                self.rpa_sensitive_config(projected_pi_sensitivity_alpha=-0.1)
            )

    def test_rpa_sensitive_joint_rejects_alpha_above_one(self):
        with self.assertRaisesRegex(
            ValueError, "projected_pi_sensitivity_alpha"
        ):
            normalize_loss_config(
                self.rpa_sensitive_config(projected_pi_sensitivity_alpha=1.1)
            )

    def test_legacy_modes_reject_rpa_sensitivity_alpha(self):
        for mode in (
            "st_only",
            "st_constrained",
            "st_dpsi_joint",
            "pi_dpsi_joint",
        ):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    normalize_loss_config(
                        {
                            "mode": mode,
                            "projected_pi_sensitivity_alpha": 0.25,
                        }
                    )

    def test_rpa_sensitive_joint_rejects_other_loss_penalties(self):
        for field, value in (
            ("radial_tail_weight", 1.0),
            ("radial_tail_radius", 4.0),
            ("low_frequency_guard_weight", 1.0),
            ("low_frequency_guard_tolerance", 0.1),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    normalize_loss_config(
                        self.rpa_sensitive_config(**{field: value})
                    )

    def test_pi_dpsi_joint_uses_projected_pi_as_primary(self):
        projected_pi = torch.tensor(
            0.3, dtype=torch.float64, requires_grad=True
        )
        dft = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        dpsi = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
        config = normalize_loss_config({"mode": "pi_dpsi_joint"})

        result = compose_loss(
            "pi_dpsi_joint",
            projected_pi,
            dft,
            dpsi,
            {"dft_origin": 1.0, "dft_dpsi": 1.0},
            config,
        )
        result["total"].backward()

        self.assertNotIn("sternheimer", result)
        self.assertIs(result["projected_pi"], projected_pi)
        self.assertAlmostEqual(result["regularization_dpsi"].item(), 0.9)
        self.assertAlmostEqual(result["total"].item(), 1.2)
        self.assertAlmostEqual(projected_pi.grad.item(), 1.0)

    def test_rpa_sensitive_joint_uses_projected_pi_and_weighted_dpsi(self):
        projected_pi = torch.tensor(
            0.3, dtype=torch.float64, requires_grad=True
        )
        dft = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        dpsi = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
        config = normalize_loss_config(self.rpa_sensitive_config())

        result = compose_loss(
            "pi_rpa_sensitive_joint",
            projected_pi,
            dft,
            dpsi,
            {"dft_origin": 1.0, "dft_dpsi": 1.0},
            config,
        )
        result["total"].backward()

        self.assertNotIn("sternheimer", result)
        self.assertIs(result["projected_pi"], projected_pi)
        self.assertAlmostEqual(result["regularization_dpsi"].item(), 0.018)
        self.assertAlmostEqual(result["total"].item(), 0.318)
        self.assertAlmostEqual(projected_pi.grad.item(), 1.0)
        self.assertAlmostEqual(dpsi.grad.item(), 0.02)

    def test_rpa_sensitive_joint_selection_uses_total(self):
        self.assertEqual(
            selection_component("pi_rpa_sensitive_joint"), "total"
        )

    def test_low_frequency_guard_value_and_gradient(self):
        st = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        low = torch.tensor(0.27, dtype=torch.float64, requires_grad=True)
        config = normalize_loss_config(
            {
                "mode": "st_only",
                "low_frequency_guard_weight": 10.0,
                "low_frequency_guard_tolerance": 0.0,
            }
        )
        baseline = {
            "dft_origin": 1.0,
            "dft_dpsi": 1.0,
            "sternheimer_lowest_frequency": 0.25,
        }

        result = compose_loss(
            "st_only",
            st,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            baseline,
            config,
            st_low_frequency=low,
        )
        result["total"].backward()

        expected = 10.0 * (0.27 / 0.25 - 1.0) ** 2
        expected_gradient = 20.0 * (0.27 / 0.25 - 1.0) / 0.25
        self.assertAlmostEqual(
            result["sternheimer_lowest_frequency"].item(), 0.27
        )
        self.assertAlmostEqual(
            result["regularization_low_frequency"].item(), expected
        )
        self.assertAlmostEqual(result["total"].item(), 0.3 + expected)
        self.assertAlmostEqual(st.grad.item(), 1.0)
        self.assertAlmostEqual(low.grad.item(), expected_gradient)

        improved = compose_loss(
            "st_only",
            torch.tensor(0.3, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            baseline,
            config,
            st_low_frequency=torch.tensor(0.24, dtype=torch.float64),
        )
        self.assertEqual(improved["regularization_low_frequency"].item(), 0.0)

    def test_low_frequency_guard_feasibility_boundary(self):
        config = normalize_loss_config(
            {
                "mode": "st_only",
                "low_frequency_guard_weight": 10.0,
                "low_frequency_guard_tolerance": 0.02,
            }
        )
        baseline = {
            "dft_origin": 0.0,
            "dft_dpsi": 0.0,
            "sternheimer_lowest_frequency": 0.25,
        }

        self.assertTrue(
            optimization_loss.low_frequency_guard_satisfied(
                torch.tensor(0.255, dtype=torch.float64), baseline, config
            )
        )
        self.assertFalse(
            optimization_loss.low_frequency_guard_satisfied(
                torch.tensor(0.255001, dtype=torch.float64), baseline, config
            )
        )

    def test_low_frequency_guard_rejects_invalid_inputs(self):
        invalid = (
            {"low_frequency_guard_weight": -1.0},
            {"low_frequency_guard_weight": True},
            {"low_frequency_guard_weight": float("nan")},
            {"low_frequency_guard_tolerance": -1.0},
            {"low_frequency_guard_tolerance": float("inf")},
        )
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises((TypeError, ValueError)):
                    normalize_loss_config({"mode": "st_only", **options})

        active = normalize_loss_config(
            {"mode": "st_only", "low_frequency_guard_weight": 1.0}
        )
        with self.assertRaisesRegex(ValueError, "must exceed epsilon"):
            compose_loss(
                "st_only",
                torch.tensor(0.3),
                torch.tensor(0.0),
                torch.tensor(0.0),
                {
                    "dft_origin": 0.0,
                    "dft_dpsi": 0.0,
                    "sternheimer_lowest_frequency": 0.0,
                },
                active,
                st_low_frequency=torch.tensor(0.2),
            )

    def test_joint_dpsi_regularization_is_active_inside_hard_gate(self):
        st = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        dft = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        dpsi = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
        config = normalize_loss_config(
            {
                "mode": "st_dpsi_joint",
                "joint_dpsi_weight": 0.5,
                "tau_dft": 0.05,
                "tau_dpsi": 0.10,
            }
        )

        result = compose_loss(
            "st_dpsi_joint",
            st,
            dft,
            dpsi,
            {"dft_origin": 1.0, "dft_dpsi": 1.0},
            config,
        )
        result["total"].backward()

        self.assertEqual(result["constraint_dft"].item(), 0.0)
        self.assertEqual(result["constraint_dpsi"].item(), 0.0)
        self.assertAlmostEqual(result["regularization_dpsi"].item(), 0.45)
        self.assertAlmostEqual(result["total"].item(), 0.75)
        self.assertAlmostEqual(st.grad.item(), 1.0)
        self.assertEqual(dft.grad.item(), 0.0)
        self.assertAlmostEqual(dpsi.grad.item(), 0.5)

    def test_st_only_returns_st_identity_and_zero_constraints(self):
        st = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        result = compose_loss(
            "st_only",
            st,
            torch.tensor(1.2, dtype=torch.float64),
            torch.tensor(1.3, dtype=torch.float64),
            {"dft_origin": 1.0, "dft_dpsi": 1.0},
            normalize_loss_config({"mode": "st_only"}),
        )

        self.assertIs(result["total"], st)
        self.assertEqual(result["constraint_dft"].item(), 0.0)
        self.assertEqual(result["constraint_dpsi"].item(), 0.0)
        self.assertEqual(result["regularization_dpsi"].item(), 0.0)
        self.assertEqual(result["radial_tail"].item(), 0.0)
        self.assertEqual(result["regularization_locality"].item(), 0.0)
        self.assertEqual(
            set(result),
            {
                "dft_origin",
                "dft_dpsi",
                "sternheimer",
                "regularization_dpsi",
                "constraint_dft",
                "constraint_dpsi",
                "radial_tail",
                "regularization_locality",
                "total",
            },
        )

    def test_radial_tail_regularization_contributes_value_and_gradient(self):
        st = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        radial_tail = torch.tensor(
            0.2, dtype=torch.float64, requires_grad=True
        )
        config = normalize_loss_config(
            {
                "mode": "st_dpsi_joint",
                "joint_dpsi_weight": 0.0,
                "radial_tail_weight": 2.5,
                "radial_tail_radius": 4.0,
            }
        )

        result = compose_loss(
            "st_dpsi_joint",
            st,
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            {"dft_origin": 1.0, "dft_dpsi": 1.0},
            config,
            radial_tail=radial_tail,
        )
        result["total"].backward()

        self.assertAlmostEqual(result["radial_tail"].item(), 0.2)
        self.assertAlmostEqual(
            result["regularization_locality"].item(), 0.5
        )
        self.assertAlmostEqual(result["total"].item(), 0.8)
        self.assertAlmostEqual(radial_tail.grad.item(), 2.5)

    def test_positive_radial_tail_weight_requires_radius_and_metric(self):
        with self.assertRaisesRegex(ValueError, "radial_tail_radius"):
            normalize_loss_config(
                {"mode": "st_only", "radial_tail_weight": 1.0}
            )

        config = normalize_loss_config(
            {
                "mode": "st_only",
                "radial_tail_weight": 1.0,
                "radial_tail_radius": 4.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "radial_tail"):
            compose_loss(
                "st_only",
                torch.tensor(0.3),
                torch.tensor(0.0),
                torch.tensor(0.0),
                {"dft_origin": 0.0, "dft_dpsi": 0.0},
                config,
            )

    def test_candidate_selection_matches_loss_mode(self):
        self.assertEqual(selection_component("st_only"), "sternheimer")
        self.assertEqual(selection_component("st_constrained"), "sternheimer")
        self.assertEqual(selection_component("st_dpsi_joint"), "total")
        self.assertEqual(selection_component("pi_dpsi_joint"), "total")

    def test_constrained_hinges_and_gradients(self):
        st = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        dft = torch.tensor(1.20, dtype=torch.float64, requires_grad=True)
        dpsi = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
        config = normalize_loss_config(
            {
                "mode": "st_constrained",
                "tau_dft": 0.05,
                "tau_dpsi": 0.10,
                "constraint_penalty_dft": 10.0,
                "constraint_penalty_dpsi": 10.0,
            }
        )

        result = compose_loss(
            "st_constrained",
            st,
            dft,
            dpsi,
            {"dft_origin": 1.0, "dft_dpsi": 1.0},
            config,
        )
        result["total"].backward()

        self.assertAlmostEqual(result["constraint_dft"].item(), 10 * 0.15**2)
        self.assertAlmostEqual(result["constraint_dpsi"].item(), 10 * 0.15**2)
        self.assertIsNotNone(st.grad)
        self.assertGreater(abs(dft.grad.item()), 0.0)
        self.assertGreater(abs(dpsi.grad.item()), 0.0)

    def test_constraint_boundaries_are_satisfied(self):
        config = normalize_loss_config({"mode": "st_constrained"})
        baseline = {"dft_origin": 2.0, "dft_dpsi": 4.0}
        self.assertTrue(
            constraints_satisfied(
                torch.tensor(2.1), torch.tensor(4.4), baseline, config
            )
        )
        self.assertFalse(
            constraints_satisfied(
                torch.tensor(2.1001), torch.tensor(4.4), baseline, config
            )
        )

    def test_rejects_invalid_config_mode_tensors_and_baseline(self):
        invalid_configs = (
            {},
            {"mode": "legacy"},
            {"mode": "st_only", "epsilon": float("nan")},
            {"mode": "st_only", "condition_limit": 0.5},
            {"mode": "st_only", "tau_dft": -1.0},
            {"mode": "st_only", "constraint_penalty_dpsi": float("inf")},
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    normalize_loss_config(config)

        config = normalize_loss_config({"mode": "st_only"})
        valid = torch.tensor(1.0)
        invalid_tensors = (
            (torch.tensor([1.0]), valid, valid),
            (torch.tensor(float("nan")), valid, valid),
            (torch.tensor(-1.0), valid, valid),
        )
        for st, dft, dpsi in invalid_tensors:
            with self.subTest(st=st):
                with self.assertRaises((TypeError, ValueError)):
                    compose_loss(
                        "st_only",
                        st,
                        dft,
                        dpsi,
                        {"dft_origin": 1.0, "dft_dpsi": 1.0},
                        config,
                    )
        with self.assertRaises((TypeError, ValueError, KeyError)):
            compose_loss(
                "st_only",
                valid,
                valid,
                valid,
                {"dft_origin": -1.0},
                config,
            )

    def test_defaults_are_not_mutated(self):
        result = normalize_loss_config({"mode": "st_only", "tau_dft": 0.2})
        self.assertEqual(result["tau_dft"], 0.2)
        self.assertEqual(LOSS_DEFAULTS["tau_dft"], 0.05)


class ReadJsonCompatibilityTest(unittest.TestCase):
    def write_input(self, value):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with handle:
            json.dump(value, handle)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return handle.name

    @staticmethod
    def minimal_input():
        return {
            "file_list": {"origin": ["synthetic"]},
            "element": {"Nt_all": ["H"], "Nu": {"H": [2]}},
            "weight": {"stru": [1.0]},
        }

    def test_legacy_input_still_returns_seven_values_without_new_defaults(self):
        result = read_json(self.write_input(self.minimal_input()))

        self.assertEqual(len(result), 7)
        self.assertNotIn("loss", result[3][0])
        self.assertNotIn("freeze_orbitals", result[4])

    def test_new_input_propagates_independent_normalized_stage_configs(self):
        value = self.minimal_input()
        value["optimize"] = [{}, {}]
        value["freeze_orbitals"] = [{"element": "H", "l": 0, "zeta": 1}]
        value["loss"] = {"mode": "st_constrained"}

        result = read_json(self.write_input(value))
        stages = result[3]

        self.assertEqual(
            result[4]["freeze_orbitals"],
            [{"element": "H", "l": 0, "zeta": 1}],
        )
        self.assertEqual(stages[0]["loss"]["mode"], "st_constrained")
        self.assertEqual(stages[0]["loss"]["epsilon"], LOSS_DEFAULTS["epsilon"])
        self.assertIsNot(stages[0]["loss"], stages[1]["loss"])

    def test_invalid_loss_is_rejected(self):
        value = self.minimal_input()
        value["loss"] = {"mode": "st_constrained", "tau_dft": -0.1}
        with self.assertRaises(ValueError):
            read_json(self.write_input(value))


class ConvergeIntegrationTest(unittest.TestCase):
    @staticmethod
    def files():
        return io.StringIO(), io.StringIO()

    def test_new_stage_requires_evaluator_and_rejects_kinetic_term(self):
        converge, c = make_converge_case("st_only", max_steps=1)
        with self.assertRaisesRegex(ValueError, "Sternheimer.*evaluator"):
            converge.cal_converge(c, self.files())

        converge, c = make_converge_case("st_only", max_steps=1)
        converge.info_optimize[0]["cal_T"] = True
        converge.set_sternheimer_spillage(QuadraticSternheimer())
        with self.assertRaisesRegex(ValueError, "cal_T"):
            converge.cal_converge(c, self.files())

    def test_st_only_logs_named_header_and_explicit_freeze_takes_precedence(self):
        converge, c = make_converge_case(
            "st_only",
            freeze_specs=[{"element": "H", "l": 0, "zeta": 1}],
        )
        converge.set_C_read_index({("H", 0, 1)})
        evaluator = QuadraticSternheimer()
        converge.set_sternheimer_spillage(evaluator)
        fixed_before = c["H"][0][:, 0].detach().clone()
        variable_before = c["H"][0][:, 1].detach().clone()
        initial_st = evaluator.evaluate(c).loss.item()
        files = self.files()

        result = converge.cal_converge(c, files)

        expected_header = (
            "istep_big\tistep_small\tistep_all\tdft_origin\tdft_dpsi\t"
            "sternheimer\tregularization_dpsi\tconstraint_dft\t"
            "constraint_dpsi\ttotal\tradial_tail\t"
            "regularization_locality\tmax_st_condition\t"
            "max_locality_condition\taccepted"
        )
        self.assertEqual(files[1].getvalue().splitlines()[0], expected_header)
        self.assertTrue(torch.equal(result["C"]["H"][0][:, 0], fixed_before))
        self.assertFalse(torch.equal(result["C"]["H"][0][:, 1], variable_before))
        self.assertLess(result["loss_components"]["sternheimer"], initial_st)
        self.assertEqual(result["loss_mode"], "st_only")
        self.assertEqual(result["Loss"], result["loss_components"]["total"])
        self.assertEqual(result["Spillage"], result["loss_components"]["total"])
        self.assertEqual(result["max_st_condition"], 2.0)
        self.assertEqual(result["max_locality_condition"], 1.0)

    def test_pi_dpsi_joint_uses_objective_specific_schema(self):
        converge, c = make_converge_case(
            "pi_dpsi_joint",
            max_steps=1,
            freeze_specs=[{"element": "H", "l": 0, "zeta": 1}],
        )
        evaluator = QuadraticProjectedPi()
        converge.set_projected_pi_objective(evaluator)
        files = self.files()

        result = converge.cal_converge(c, files)

        expected_header = (
            "istep_big\tistep_small\tistep_all\tdft_origin\tdft_dpsi\t"
            "projected_pi\tprojected_pi_lowest_frequency\t"
            "regularization_dpsi\tconstraint_dft\tconstraint_dpsi\t"
            "total\tmax_projected_pi_condition\taccepted"
        )
        self.assertEqual(files[1].getvalue().splitlines()[0], expected_header)
        self.assertEqual(result["loss_mode"], "pi_dpsi_joint")
        self.assertIn("projected_pi", result["loss_components"])
        self.assertNotIn("sternheimer", result["loss_components"])
        self.assertEqual(result["max_projected_pi_condition"], 2.0)
        self.assertEqual(
            result["projected_pi_diagnostics"]["frequency_ha"], [0.1, 1.0]
        )
        self.assertEqual(
            result["projected_pi_diagnostics"]["family_names"], ["H", "H2"]
        )

    def test_low_frequency_guard_rejects_regressed_candidate(self):
        converge, c = make_converge_case("st_only", max_steps=1)
        converge.info_optimize[0]["loss"].update(
            {
                "low_frequency_guard_weight": 10.0,
                "low_frequency_guard_tolerance": 0.0,
            }
        )
        evaluator = OpposingFrequencySternheimer()
        converge.set_sternheimer_spillage(evaluator)

        class RegressingStep:
            def zero_grad(self):
                c["H"][0].grad = None

            def step(self, closure):
                closure()
                with torch.no_grad():
                    c["H"][0][0, 0] = 0.5

        files = self.files()
        with mock.patch(
            "opt_orbital_converge.optimize.get_optim",
            return_value=RegressingStep(),
        ):
            result = converge.cal_converge(c, files)

        self.assertAlmostEqual(evaluator.evaluate(c).loss.item(), 0.25)
        self.assertAlmostEqual(
            evaluator.evaluate(c).lowest_frequency_loss.item(), 0.30
        )
        self.assertAlmostEqual(result["C"]["H"][0][0, 0].item(), 1.0)
        self.assertAlmostEqual(
            result["loss_baseline"]["sternheimer_lowest_frequency"], 0.20
        )
        self.assertAlmostEqual(
            result["loss_components"]["sternheimer_lowest_frequency"], 0.20
        )
        self.assertEqual(
            result["low_frequency_diagnostics"],
            {
                "lowest_st_frequency_ha": 0.1,
                "initial_lowest_st_loss": 0.2,
                "final_lowest_st_loss": 0.2,
                "low_frequency_guard_tolerance": 0.0,
                "low_frequency_guard_weight": 10.0,
            },
        )

        lines = files[1].getvalue().splitlines()
        header = lines[0].split("\t")
        rows = [
            dict(zip(header, line.split("\t")))
            for line in lines[1:]
            if line.split("\t", 1)[0].lstrip("-").isdigit()
        ]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["accepted"], "true")
        self.assertEqual(rows[1]["accepted"], "false")
        self.assertAlmostEqual(
            float(rows[1]["sternheimer_lowest_frequency"]), 0.30
        )

    def test_locality_evaluator_regularizes_nonfixed_columns(self):
        converge, c = make_converge_case(
            "st_dpsi_joint",
            max_steps=1,
            freeze_specs=[{"element": "H", "l": 0, "zeta": 1}],
        )
        converge.info_optimize[0]["loss"].update(
            {
                "radial_tail_weight": 0.5,
                "radial_tail_radius": 4.0,
                "radial_tail_condition_limit": 10.0,
            }
        )
        converge.set_sternheimer_spillage(QuadraticSternheimer())
        converge.set_radial_locality(QuadraticLocality())
        before = c["H"][0][:, 1].detach().clone()

        result = converge.cal_converge(c, self.files())

        self.assertFalse(torch.equal(c["H"][0][:, 1], before))
        self.assertGreaterEqual(result["loss_components"]["radial_tail"], 0.0)
        self.assertEqual(result["max_locality_condition"], 3.0)

    def test_adam_one_step_selects_post_step_state(self):
        converge, c = make_converge_case("st_only", max_steps=1)
        evaluator = QuadraticSternheimer()
        converge.set_sternheimer_spillage(evaluator)
        initial_st = evaluator.evaluate(c).loss.item()

        result = converge.cal_converge(c, self.files())

        final_st = evaluator.evaluate(c).loss.item()
        self.assertLess(final_st, initial_st)
        self.assertAlmostEqual(
            result["loss_components"]["sternheimer"], final_st, places=14
        )
        self.assertTrue(torch.equal(result["C"]["H"][0], c["H"][0]))

    def test_lbfgs_trial_closure_state_is_not_a_candidate(self):
        converge, c = make_converge_case("st_only", max_steps=1)
        converge.info_optimize[0]["optimizer"] = "LBFGS"
        converge.info_optimize[0]["kwargs"] = {}
        evaluator = QuadraticSternheimer()
        converge.set_sternheimer_spillage(evaluator)
        initial = c["H"][0].detach().clone()
        transient = torch.tensor(
            [[0.75, 0.5], [0.25, 0.5]], dtype=torch.float64
        )
        final = (initial + transient) / 2

        class TrialPointLBFGS:
            def zero_grad(self):
                c["H"][0].grad = None

            def step(self, closure):
                with torch.no_grad():
                    c["H"][0].copy_(transient)
                closure()
                with torch.no_grad():
                    c["H"][0].copy_(final)

        files = self.files()
        with mock.patch(
            "opt_orbital_converge.optimize.get_optim",
            return_value=TrialPointLBFGS(),
        ):
            result = converge.cal_converge(c, files)

        self.assertTrue(torch.equal(result["C"]["H"][0], final))
        self.assertAlmostEqual(
            result["loss_components"]["sternheimer"],
            evaluator.evaluate({"H": [final]}).loss.item(),
            places=14,
        )
        candidate_rows = [
            line
            for line in files[1].getvalue().splitlines()
            if line.split("\t", 1)[0].lstrip("-").isdigit()
        ]
        self.assertEqual(len(candidate_rows), 2)

    def test_explicit_freeze_survives_adam_weight_decay_and_state(self):
        converge, c = make_converge_case(
            "st_only",
            max_steps=3,
            freeze_specs=[{"element": "H", "l": 0, "zeta": 1}],
        )
        converge.info_optimize[0]["kwargs"] = {
            "lr": 0.05,
            "weight_decay": 0.4,
        }
        converge.set_sternheimer_spillage(QuadraticSternheimer())
        fixed = c["H"][0][:, 0].detach().clone()

        result = converge.cal_converge(c, self.files())

        self.assertTrue(torch.equal(c["H"][0][:, 0], fixed))
        self.assertTrue(torch.equal(result["C"]["H"][0][:, 0], fixed))

    def test_condition_limit_rejects_external_evaluator_candidates(self):
        converge, c = make_converge_case("st_only", max_steps=1)
        converge.info_optimize[0]["loss"]["condition_limit"] = 1.5
        converge.set_sternheimer_spillage(
            QuadraticSternheimer(max_condition=2.0)
        )
        files = self.files()

        with self.assertRaisesRegex(RuntimeError, "condition"):
            converge.cal_converge(c, files)

        candidate_rows = [
            line.split("\t")
            for line in files[1].getvalue().splitlines()
            if line.split("\t", 1)[0].lstrip("-").isdigit()
        ]
        self.assertTrue(candidate_rows)
        self.assertTrue(all(row[-1] == "false" for row in candidate_rows))

    def test_zero_step_no_accepted_diagnostic_is_finite(self):
        converge, c = make_converge_case("st_only", max_steps=0)
        converge.info_optimize[0]["loss"]["condition_limit"] = 1.5
        converge.set_sternheimer_spillage(
            QuadraticSternheimer(max_condition=2.0)
        )

        with self.assertRaises(RuntimeError) as context:
            converge.cal_converge(c, self.files())

        message = str(context.exception)
        self.assertNotIn("inf", message.lower())
        self.assertRegex(message, r"dft=.*dpsi=.*condition=")

    def test_stage_norm_baselines_are_independent_and_configs_immutable(self):
        converge, c = make_converge_case()
        stages = [
            {
                "optimizer": "SGD",
                "kwargs": {"lr": 0.0},
                "cal_T": False,
                "norm": norm,
                "max_steps": 1,
                "loss": {
                    "mode": "st_constrained",
                    "tau_dft": 0.0,
                    "tau_dpsi": 0.0,
                },
            }
            for norm in ("element", "one")
        ]
        stages_before = copy.deepcopy(stages)
        converge.info_optimize = stages
        converge.VI_origin = [torch.tensor([2.0], dtype=torch.float64)]
        converge.set_sternheimer_spillage(QuadraticSternheimer())
        files = self.files()

        result = converge.cal_converge(c, files)

        self.assertEqual(stages, stages_before)
        header = (
            "istep_big\tistep_small\tistep_all\tdft_origin\tdft_dpsi\t"
            "sternheimer\tregularization_dpsi\tconstraint_dft\t"
            "constraint_dpsi\ttotal\tradial_tail\t"
            "regularization_locality\tmax_st_condition\t"
            "max_locality_condition\taccepted"
        )
        lines = files[1].getvalue().splitlines()
        self.assertEqual(lines.count(header), 2)
        candidate_rows = [
            line.split("\t")
            for line in lines
            if line.split("\t", 1)[0].lstrip("-").isdigit()
        ]
        self.assertTrue(candidate_rows)
        self.assertTrue(all(row[-1] == "true" for row in candidate_rows))
        self.assertAlmostEqual(result["loss_baseline"]["dft_origin"], 0.5)

    def test_constrained_selection_uses_feasible_st_best_not_total_best(self):
        converge, c = make_single_orbital_converge()
        evaluator = SingleOrbitalSternheimer()
        converge.set_sternheimer_spillage(evaluator)
        files = self.files()

        result = converge.cal_converge(c, files)

        spillage = converge._make_spillage(converge.info_optimize[0])
        selected_dft = spillage.cal_components(result["C"])["dft_origin"]
        final_dft = spillage.cal_components(c)["dft_origin"]
        config = converge.info_optimize[0]["loss"]
        self.assertTrue(
            constraints_satisfied(
                selected_dft,
                torch.zeros_like(selected_dft),
                result["loss_baseline"],
                config,
            )
        )
        self.assertFalse(
            constraints_satisfied(
                final_dft,
                torch.zeros_like(final_dft),
                result["loss_baseline"],
                config,
            )
        )
        final_st = evaluator.evaluate(c).loss.item()
        self.assertGreater(
            result["loss_components"]["sternheimer"], final_st
        )
        candidate_totals = [
            float(row[9])
            for row in (
                line.split("\t") for line in files[1].getvalue().splitlines()
            )
            if row[0].lstrip("-").isdigit()
        ]
        self.assertAlmostEqual(final_st, min(candidate_totals))
        accepted_st = [
            float(row[5])
            for row in (
                line.split("\t") for line in files[1].getvalue().splitlines()
            )
            if row[0].lstrip("-").isdigit() and row[-1] == "true"
        ]
        self.assertAlmostEqual(
            result["loss_components"]["sternheimer"], min(accepted_st)
        )

    def test_mixed_new_then_legacy_stages_emit_explicit_schema_headers(self):
        converge, c = make_converge_case()
        converge.info_optimize = [
            {
                "optimizer": "SGD",
                "kwargs": {"lr": 0.0},
                "cal_T": False,
                "norm": "one",
                "max_steps": 0,
                "loss": {"mode": "st_only"},
            },
            {
                "optimizer": "SGD",
                "kwargs": {"lr": 0.0},
                "cal_T": False,
                "norm": "one",
                "max_steps": 1,
            },
        ]
        converge.set_sternheimer_spillage(QuadraticSternheimer())
        files = self.files()

        converge.cal_converge(c, files)

        lines = files[1].getvalue().splitlines()
        legacy_header = "istep_big\tistep_small\tistep_all\tSpillage"
        self.assertIn(legacy_header, lines)
        legacy_index = lines.index(legacy_header)
        self.assertEqual(len(lines[legacy_index + 1].split("\t")), 4)

    def test_constrained_best_point_obeys_thresholds(self):
        converge, c = make_converge_case("st_constrained", max_steps=4)
        evaluator = QuadraticSternheimer()
        converge.set_sternheimer_spillage(evaluator)

        result = converge.cal_converge(c, self.files())

        spillage = Opt_Orbital_Spillage(
            converge.info_stru,
            converge.info_element,
            converge.info_V,
            "one",
            converge.file_list,
        )
        spillage.set_QSVI(converge.QI, converge.SI, converge.VI_origin)
        components = spillage.cal_components(result["C"])
        self.assertTrue(
            constraints_satisfied(
                components["dft_origin"],
                components["dft_dpsi"],
                result["loss_baseline"],
                converge.info_optimize[0]["loss"],
            )
        )

    def test_legacy_path_retains_task_one_behavior_and_header(self):
        loss, c = make_legacy_spillage()
        converge = Opt_Orbital_Converge()
        converge.set_info(
            {"origin": ["synthetic"]},
            [
                {
                    "optimizer": "Adam",
                    "kwargs": {"lr": 0.01},
                    "cal_T": False,
                    "norm": "one",
                    "max_steps": 1,
                }
            ],
            loss.info_stru,
            {"init_from_file": False},
            loss.info_V,
        )
        converge.set_info_element(loss.info_element)
        converge.set_QSVI(loss.QI, loss.SI, loss.VI_origin)
        files = self.files()

        result = converge.cal_converge(c, files)

        self.assertAlmostEqual(result["Spillage"], 0.4, places=14)
        self.assertNotIn("loss_components", result)
        self.assertNotIn("sternheimer", files[1].getvalue())
        self.assertEqual(files[0].getvalue().splitlines()[1], "istep\tSpillage")


if __name__ == "__main__":
    unittest.main()
