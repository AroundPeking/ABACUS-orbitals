import json
import math
import tempfile
import unittest
from pathlib import Path

from c_diamond_counterpoise import (
    HARTREE_TO_EV,
    build_ghost_cluster,
    collect_counterpoise,
    render_stru,
)


class DiamondGhostClusterTests(unittest.TestCase):
    def test_first_shell_contains_four_tetrahedral_neighbors(self):
        cluster = build_ghost_cluster(
            lattice_constant_angstrom=3.6,
            box_angstrom=20.0,
            shell_count=1,
        )

        self.assertEqual(cluster["ghost_count"], 4)
        expected_distance = math.sqrt(3.0) * 3.6 / 4.0
        for row in cluster["ghosts"]:
            self.assertAlmostEqual(row["distance_angstrom"], expected_distance, places=12)
            self.assertTrue(all(0.0 < value < 1.0 for value in row["fractional"]))

    def test_two_shells_contain_four_plus_twelve_neighbors(self):
        cluster = build_ghost_cluster(
            lattice_constant_angstrom=3.6,
            box_angstrom=20.0,
            shell_count=2,
        )

        self.assertEqual(cluster["ghost_count"], 16)
        self.assertEqual(cluster["shell_populations"], [4, 12])
        self.assertAlmostEqual(cluster["shell_distances_angstrom"][1], 3.6 / math.sqrt(2.0), places=12)

    def test_stru_keeps_real_and_ghost_basis_identical(self):
        cluster = build_ghost_cluster(
            lattice_constant_angstrom=3.6,
            box_angstrom=20.0,
            shell_count=1,
        )
        text = render_stru(
            cluster,
            orbital_filename="C_candidate.orb",
            pseudopotential_filename="C_ONCV_PBE-1.0.upf",
        )

        self.assertIn("C_empty 12.011 C_ONCV_PBE-1.0.upf", text)
        self.assertEqual(text.count("C_candidate.orb"), 2)
        self.assertIn("C_empty\n0.0\n4\n", text)


class CounterpoiseCollectionTests(unittest.TestCase):
    def test_adds_ghost_minus_isolated_atom_correction(self):
        raw = {
            "atom_zero_order_ha": -5.0,
            "atom_ecrpa_ha": -0.1,
            "zero_order_binding_ev_per_c": 5.0,
            "correlation_binding_ev_per_c": 2.0,
            "sos_total_binding_ev_per_c": 7.0,
            "delta_st_reference_ev_per_c": 6.9,
            "selected_orbital_sha256": "a" * 64,
        }
        ghost = {
            "reference_ha": -5.002,
            "ecrpa_ha": -0.103,
            "selected_orbital_sha256": "a" * 64,
            "shell_count": 1,
            "ghost_count": 4,
        }

        result = collect_counterpoise(raw, ghost)

        self.assertAlmostEqual(result["zero_order_binding_cp_ev_per_c"], 5.0 - 0.002 * HARTREE_TO_EV)
        self.assertAlmostEqual(result["correlation_binding_cp_ev_per_c"], 2.0 - 0.003 * HARTREE_TO_EV)
        self.assertAlmostEqual(result["sos_total_binding_cp_ev_per_c"], 7.0 - 0.005 * HARTREE_TO_EV)
        self.assertAlmostEqual(result["bsse_total_ev_per_c"], 0.005 * HARTREE_TO_EV)
        self.assertEqual(result["scope"], "first_shell_counterpoise_diagnostic_requires_shell_convergence")

    def test_rejects_mismatched_orbital_hash(self):
        raw = {
            "atom_zero_order_ha": -5.0,
            "atom_ecrpa_ha": -0.1,
            "zero_order_binding_ev_per_c": 5.0,
            "correlation_binding_ev_per_c": 2.0,
            "sos_total_binding_ev_per_c": 7.0,
            "delta_st_reference_ev_per_c": 6.9,
            "selected_orbital_sha256": "a" * 64,
        }
        ghost = {
            "reference_ha": -5.002,
            "ecrpa_ha": -0.103,
            "selected_orbital_sha256": "b" * 64,
            "shell_count": 1,
            "ghost_count": 4,
        }

        with self.assertRaisesRegex(ValueError, "orbital hash"):
            collect_counterpoise(raw, ghost)

    def test_cli_writes_json_and_status(self):
        raw = {
            "atom_zero_order_ha": -5.0,
            "atom_ecrpa_ha": -0.1,
            "zero_order_binding_ev_per_c": 5.0,
            "correlation_binding_ev_per_c": 2.0,
            "sos_total_binding_ev_per_c": 7.0,
            "delta_st_reference_ev_per_c": 6.9,
            "selected_orbital_sha256": "a" * 64,
        }
        ghost = {
            "reference_ha": -5.002,
            "ecrpa_ha": -0.103,
            "selected_orbital_sha256": "a" * 64,
            "shell_count": 1,
            "ghost_count": 4,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw.json").write_text(json.dumps(raw), encoding="ascii")
            (root / "ghost.json").write_text(json.dumps(ghost), encoding="ascii")
            from c_diamond_counterpoise import main

            main(
                [
                    "collect",
                    "--raw-binding",
                    str(root / "raw.json"),
                    "--ghost-summary",
                    str(root / "ghost.json"),
                    "--output-root",
                    str(root / "result"),
                ]
            )
            result = json.loads((root / "result" / "RESULT.json").read_text(encoding="ascii"))
            self.assertEqual(result["status"], "success")
            self.assertEqual((root / "result" / "STATUS").read_text(encoding="ascii"), "success\n")


if __name__ == "__main__":
    unittest.main()
