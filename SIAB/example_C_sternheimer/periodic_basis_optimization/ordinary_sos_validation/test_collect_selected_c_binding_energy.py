#!/usr/bin/env python3
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from collect_selected_c_binding_energy import collect_binding_energy


ORBITAL_SHA = "3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1"


def write_summary(
    path: Path,
    *,
    side: str,
    method: str,
    e0_ha: float,
    ec_ha: float,
    naux: int,
    frequency_sha: str,
    status: str = "success",
) -> None:
    path.write_text(
        "\n".join(
            [
                f"status {status}",
                f"side {side}",
                f"method {method}",
                "scope body_only_no_analytic_headwing",
                "coulomb_kernel full_periodic_poisson",
                f"selected_orbital_sha256 {ORBITAL_SHA}",
                f"frequency_grid_sha256 {frequency_sha}",
                f"naux {naux}",
                f"reference_ha {e0_ha:.15f}",
                f"ecrpa_ha {ec_ha:.15f}",
            ]
        )
        + "\n",
        encoding="ascii",
    )


class SelectedCarbonBindingEnergyTest(unittest.TestCase):
    def make_endpoints(self, root: Path) -> dict[str, Path]:
        paths = {name: root / f"{name}.txt" for name in (
            "atom_sos", "atom_delta", "solid_sos", "solid_delta"
        )}
        write_summary(
            paths["atom_sos"], side="atom", method="sos",
            e0_ha=-5.0, ec_ha=-0.20, naux=261, frequency_sha="a" * 64,
        )
        write_summary(
            paths["atom_delta"], side="atom", method="delta_st",
            e0_ha=-5.0, ec_ha=-0.21, naux=261, frequency_sha="a" * 64,
        )
        write_summary(
            paths["solid_sos"], side="solid", method="sos",
            e0_ha=-10.2, ec_ha=-0.50, naux=522, frequency_sha="b" * 64,
        )
        write_summary(
            paths["solid_delta"], side="solid", method="delta_st",
            e0_ha=-10.2, ec_ha=-0.52, naux=522, frequency_sha="b" * 64,
        )
        return paths

    def test_two_atom_binding_formula_and_total_energy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = collect_binding_energy(**self.make_endpoints(Path(temporary)))

        self.assertAlmostEqual(result["sos_correlation_binding_ha_per_c"], 0.05)
        self.assertAlmostEqual(result["delta_st_correlation_binding_ha_per_c"], 0.05)
        self.assertAlmostEqual(result["sos_total_binding_ha_per_c"], 0.15)
        self.assertAlmostEqual(result["delta_st_total_binding_ha_per_c"], 0.15)
        self.assertAlmostEqual(result["binding_difference_ha_per_c"], 0.0)
        self.assertEqual(result["basis_full_body_gate"], "pass")

    def test_point_one_kcal_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_endpoints(Path(temporary))
            write_summary(
                paths["solid_delta"], side="solid", method="delta_st",
                e0_ha=-10.2, ec_ha=-0.50, naux=522,
                frequency_sha="b" * 64,
            )
            write_summary(
                paths["atom_delta"], side="atom", method="delta_st",
                e0_ha=-5.0,
                ec_ha=-0.20 + 0.05 / 627.5094740631,
                naux=261,
                frequency_sha="a" * 64,
            )
            result = collect_binding_energy(**paths)
            self.assertEqual(result["basis_full_body_gate"], "pass")

            write_summary(
                paths["atom_delta"], side="atom", method="delta_st",
                e0_ha=-5.0,
                ec_ha=-0.20 + 0.11 / 627.5094740631,
                naux=261,
                frequency_sha="a" * 64,
            )
            result = collect_binding_energy(**paths)
            self.assertEqual(result["basis_full_body_gate"], "fail")
            self.assertTrue(math.isclose(
                abs(result["binding_difference_kcal_mol_per_c"]),
                0.11,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ))

    def test_rejects_mismatched_atom_frequency_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_endpoints(Path(temporary))
            write_summary(
                paths["atom_delta"], side="atom", method="delta_st",
                e0_ha=-5.0, ec_ha=-0.21, naux=261,
                frequency_sha="c" * 64,
            )
            with self.assertRaisesRegex(ValueError, "atom frequency"):
                collect_binding_energy(**paths)

    def test_rejects_old_or_incomplete_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_endpoints(Path(temporary))
            paths["atom_sos"].write_text(
                "status success\nside atom\nmethod sos\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "missing keys"):
                collect_binding_energy(**paths)


if __name__ == "__main__":
    unittest.main()
