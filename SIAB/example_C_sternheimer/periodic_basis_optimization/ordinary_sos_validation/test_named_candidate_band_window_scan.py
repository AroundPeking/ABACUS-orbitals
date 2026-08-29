#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_named_candidate_band_window_scan import (
    binding_row,
    parse_band_energy_window,
)


class NamedCandidateBandWindowScanTest(unittest.TestCase):
    def test_parses_librpa_band_energy_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            band_out = Path(temporary) / "band_out"
            band_out.write_text(
                "\n".join(
                    [
                        "1",
                        "2",
                        "4",
                        "4",
                        "0.0",
                        "1 1",
                        "1 1.0 -0.50 -13.6",
                        "2 1.0 -0.20 -5.4",
                        "3 0.0 0.10 2.7",
                        "4 0.0 1.00 27.2",
                        "1 2",
                        "1 1.0 -0.40 -10.9",
                        "2 0.0 -0.10 -2.7",
                        "3 0.0 0.20 5.4",
                        "4 0.0 1.20 32.7",
                    ]
                )
                + "\n",
                encoding="ascii",
            )

            result = parse_band_energy_window(band_out)

        self.assertEqual(result["nk"], 1)
        self.assertEqual(result["nspin"], 2)
        self.assertEqual(result["nbands"], 4)
        self.assertAlmostEqual(result["minimax_min_gap_ha"], 0.10)
        self.assertAlmostEqual(result["minimax_max_transition_ha"], 1.70)

    def test_binding_row_uses_atom_minus_half_solid(self) -> None:
        result = binding_row(
            atom_bands=10,
            solid_bands=20,
            atom_e0_ha=-5.0,
            solid_e0_ha=-10.2,
            atom_ec_ha=-0.20,
            solid_ec_ha=-0.50,
        )

        self.assertAlmostEqual(result["zero_order_binding_ha_per_c"], 0.10)
        self.assertAlmostEqual(result["correlation_binding_ha_per_c"], 0.05)
        self.assertAlmostEqual(result["total_binding_ha_per_c"], 0.15)
        self.assertAlmostEqual(
            result["total_binding_ev_per_c"],
            0.15 * 27.211386245988,
        )


if __name__ == "__main__":
    unittest.main()
