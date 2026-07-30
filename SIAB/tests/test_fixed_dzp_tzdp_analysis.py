import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "SIAB"
    / "example_H_sternheimer"
    / "fixed_dzp_tzdp_sos"
    / "analyze_orbitals.py"
)


def load_analysis_module():
    spec = importlib.util.spec_from_file_location(
        "fixed_dzp_tzdp_analysis", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedDzpTzdpAnalysisTest(unittest.TestCase):
    def test_frequency_resolved_loss_keeps_greenx_weight_out_of_local_ratio(self):
        analysis = load_analysis_module()
        target = analysis.TargetArrays(
            blocks=(
                SimpleNamespace(
                    element="H", l=0, n_primitive=3, offset=0
                ),
            ),
            q=np.array(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
                dtype=np.complex128,
            ),
            overlap=np.eye(3, dtype=np.complex128),
            norm=np.ones(2),
            occupation=np.array([1.0, 2.0]),
            frequency_weight=np.array([0.25, 0.75]),
            frequency_ha=np.array([0.1, 0.4]),
            auxiliary_channel=np.zeros(2, dtype=np.int64),
        )
        coefficients = {("H", 0): np.eye(3)}
        context = analysis.fixed_context(target, coefficients)

        local = analysis.evaluate_extra_orbitals_by_frequency(
            target, coefficients, context
        )

        self.assertEqual(
            local,
            [
                {"frequency_ha": 0.1, "loss": 0.0},
                {"frequency_ha": 0.4, "loss": 0.75},
            ],
        )
        self.assertAlmostEqual(
            analysis.evaluate_extra_orbitals(target, coefficients, context),
            1.125 / 1.75,
        )

    def test_read_orbitals_keeps_l_and_zero_based_zeta(self):
        analysis = load_analysis_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "H.orb"
            path.write_text(
                "\n".join(
                    [
                        "Mesh 3",
                        "dr 0.5",
                        "Type L N",
                        "0 0 0",
                        "1.0 2.0 3.0",
                        "Type L N",
                        "0 1 1",
                        "4.0 5.0 6.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            radius, orbitals = analysis.read_orbitals(path)

        np.testing.assert_allclose(radius, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(orbitals[(0, 0)], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(orbitals[(1, 1)], [4.0, 5.0, 6.0])

    def test_rank_revealing_projection_drops_near_null_direction(self):
        analysis = load_analysis_module()
        overlap = np.diag([2.0, 1.0e-8]).astype(np.complex128)
        q = np.array([2.0, 1.0], dtype=np.complex128)

        result = analysis.rank_revealing_projection(
            overlap, q, relative_rank_tolerance=1.0e-4
        )

        self.assertEqual(result.rank, 1)
        np.testing.assert_allclose(result.coefficients, [1.0, 0.0])
        self.assertAlmostEqual(result.represented_norm, 2.0)

    def test_classify_reference_uses_represented_block_norm(self):
        analysis = load_analysis_module()
        block_overlaps = {
            0: np.eye(2, dtype=np.complex128),
            1: np.eye(2, dtype=np.complex128),
        }
        q_by_l = {
            0: np.array([0.1, 0.0], dtype=np.complex128),
            1: np.array([0.0, 0.8], dtype=np.complex128),
        }

        dominant_l, represented = analysis.classify_reference(
            block_overlaps,
            q_by_l,
            relative_rank_tolerance=1.0e-8,
        )

        self.assertEqual(dominant_l, 1)
        self.assertAlmostEqual(represented[0], 0.01)
        self.assertAlmostEqual(represented[1], 0.64)


if __name__ == "__main__":
    unittest.main()
