#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c_sos_differential_qstar_proxy import assess_differential_proxy


HARTREE_TO_EV = 27.211386245988


def make_rows(*, outlier=0.0):
    rows = []
    features = (
        (-0.020, -0.080, -0.040),
        (-0.024, -0.072, -0.045),
        (-0.018, -0.090, -0.035),
        (-0.027, -0.076, -0.038),
        (-0.016, -0.084, -0.048),
        (-0.022, -0.088, -0.042),
    )
    for index, (q6, q7, q8) in enumerate(features):
        solid = -0.10 + 2.0 * q6 - 3.0 * q7 + 0.5 * q8
        atom = -0.13 - 0.001 * index
        zero_order = 5.0 + 0.01 * index
        total = zero_order + (atom - 0.5 * solid) * HARTREE_TO_EV
        if index == len(features) - 1:
            total += outlier
        rows.append(
            {
                "name": f"candidate-{index}",
                "zero_order_binding_ev_per_c": zero_order,
                "atom_ecrpa_ha": atom,
                "solid_ecrpa_ha": solid,
                "sos_total_binding_ev_per_c": total,
                "qstar_weighted_contributions_ha": {6: q6, 7: q7, 8: q8},
            }
        )
    return rows


class DifferentialQstarProxyTest(unittest.TestCase):
    def test_passes_redundant_affine_data_with_exact_leave_one_out(self):
        result = assess_differential_proxy(
            make_rows(),
            q_indices=(6, 7, 8),
            maximum_loo_error_ev_per_c=0.01,
            minimum_rank_concordance=0.95,
        )

        self.assertEqual(result["proxy_gate"], "pass")
        self.assertEqual(result["full_design_rank"], 4)
        self.assertEqual(result["loo_design_ranks"], [4] * 6)
        self.assertAlmostEqual(result["loo_max_abs_error_ev_per_c"], 0.0, places=10)
        self.assertAlmostEqual(result["loo_rank_concordance"], 1.0)

    def test_rejects_a_proxy_that_misses_the_physical_endpoint(self):
        result = assess_differential_proxy(
            make_rows(outlier=0.05),
            q_indices=(6, 7, 8),
            maximum_loo_error_ev_per_c=0.01,
            minimum_rank_concordance=0.95,
        )

        self.assertEqual(result["proxy_gate"], "fail")
        self.assertIn("loo_error_too_large", result["failure_reasons"])

    def test_rejects_a_rank_deficient_calibration_set(self):
        rows = make_rows()
        for row in rows:
            row["qstar_weighted_contributions_ha"][8] = (
                2.0 * row["qstar_weighted_contributions_ha"][6]
            )

        result = assess_differential_proxy(
            rows,
            q_indices=(6, 7, 8),
            maximum_loo_error_ev_per_c=0.01,
            minimum_rank_concordance=0.95,
        )

        self.assertEqual(result["proxy_gate"], "fail")
        self.assertIn("full_design_rank_deficient", result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
