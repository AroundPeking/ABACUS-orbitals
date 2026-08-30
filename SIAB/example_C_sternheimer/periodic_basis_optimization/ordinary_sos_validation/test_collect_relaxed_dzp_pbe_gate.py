import math
import unittest

from collect_relaxed_dzp_pbe_gate import (
    REFERENCE_ATOM_EV,
    REFERENCE_SOLID_C2_EV,
    collect_pbe_gate,
)


class RelaxedDzpPbeGateTest(unittest.TestCase):
    def test_reference_values_pass(self):
        result = collect_pbe_gate(
            atom_energy_ev=REFERENCE_ATOM_EV,
            solid_c2_energy_ev=REFERENCE_SOLID_C2_EV,
        )
        self.assertEqual(result["pbe_gate"], "pass")
        self.assertTrue(all(result["checks"].values()))
        self.assertAlmostEqual(result["reference_binding_ev_per_c"], 7.4522049445947)

    def test_atom_drift_fails(self):
        result = collect_pbe_gate(
            atom_energy_ev=REFERENCE_ATOM_EV + 0.011,
            solid_c2_energy_ev=REFERENCE_SOLID_C2_EV,
        )
        self.assertEqual(result["pbe_gate"], "fail")
        self.assertFalse(result["checks"]["atom_energy"])

    def test_binding_cancellation_is_checked_separately(self):
        result = collect_pbe_gate(
            atom_energy_ev=REFERENCE_ATOM_EV + 0.009,
            solid_c2_energy_ev=REFERENCE_SOLID_C2_EV - 0.018,
        )
        self.assertTrue(result["checks"]["atom_energy"])
        self.assertTrue(result["checks"]["solid_energy_per_c"])
        self.assertFalse(result["checks"]["binding_energy"])
        self.assertEqual(result["pbe_gate"], "fail")

    def test_nonfinite_energy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            collect_pbe_gate(
                atom_energy_ev=math.nan,
                solid_c2_energy_ev=REFERENCE_SOLID_C2_EV,
            )


if __name__ == "__main__":
    unittest.main()
