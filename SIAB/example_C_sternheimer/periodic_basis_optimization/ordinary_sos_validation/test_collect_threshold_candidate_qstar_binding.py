#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_threshold_candidate_qstar_binding import collect_binding


def write_summary(path, *, side, zero_order, correlation, orbital_sha="a" * 64):
    path.write_text(
        "\n".join(
            (
                "status success",
                f"side {side}",
                "method sos",
                f"selected_orbital_sha256 {orbital_sha}",
                f"reference_ha {zero_order}",
                f"ecrpa_ha {correlation}",
            )
        )
        + "\n",
        encoding="ascii",
    )


class ThresholdCandidateQstarBindingTest(unittest.TestCase):
    def test_accepts_result_within_point_one_ev(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom = root / "atom.txt"
            solid = root / "solid.txt"
            write_summary(atom, side="atom", zero_order=-5.0, correlation=-0.1)
            write_summary(solid, side="solid", zero_order=-10.4, correlation=-0.32)

            result = collect_binding(
                atom_summary=atom,
                solid_summary=solid,
                delta_reference_ev_per_c=6.902326,
                tolerance_ev_per_c=0.1,
            )

            self.assertAlmostEqual(result["sos_total_binding_ev_per_c"], 7.0749604252)
            self.assertAlmostEqual(result["difference_from_delta_ev_per_c"], 0.1726344252)
            self.assertEqual(result["binding_gate"], "fail")

            write_summary(solid, side="solid", zero_order=-10.4, correlation=-0.31)
            result = collect_binding(
                atom_summary=atom,
                solid_summary=solid,
                delta_reference_ev_per_c=6.902326,
                tolerance_ev_per_c=0.1,
            )
            self.assertEqual(result["binding_gate"], "pass")

    def test_rejects_mismatched_orbital(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom = root / "atom.txt"
            solid = root / "solid.txt"
            write_summary(atom, side="atom", zero_order=-5.0, correlation=-0.1)
            write_summary(
                solid,
                side="solid",
                zero_order=-10.4,
                correlation=-0.46,
                orbital_sha="b" * 64,
            )
            with self.assertRaisesRegex(ValueError, "orbital"):
                collect_binding(
                    atom_summary=atom,
                    solid_summary=solid,
                    delta_reference_ev_per_c=6.902326,
                    tolerance_ev_per_c=0.1,
                )


if __name__ == "__main__":
    unittest.main()
