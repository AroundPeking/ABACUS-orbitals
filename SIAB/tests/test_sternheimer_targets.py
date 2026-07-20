from pathlib import Path
import unittest

import sys

OPT_DIR = Path(__file__).resolve().parents[1] / "opt_orb_pytorch_dpsi"
if str(OPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPT_DIR))

from sternheimer_targets import parse_target_entries


class SternheimerTargetEntryTest(unittest.TestCase):
    def test_wraps_legacy_path_as_default_physical_family(self):
        entries = parse_target_entries(["atom.dat"])

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].path, Path("atom.dat"))
        self.assertEqual(entries[0].family, "default")
        self.assertEqual(entries[0].role, "physical")

    def test_parses_named_target_families(self):
        entries = parse_target_entries(
            [
                {
                    "path": "atom.dat",
                    "family": "atom",
                    "role": "physical",
                },
                {
                    "path": "h3.dat",
                    "family": "multicenter",
                    "role": "physical",
                },
                {
                    "path": "ghost.dat",
                    "family": "fragment_ghost",
                    "role": "ghost",
                },
            ]
        )

        self.assertEqual(
            [(value.family, value.role) for value in entries],
            [
                ("atom", "physical"),
                ("multicenter", "physical"),
                ("fragment_ghost", "ghost"),
            ],
        )

    def test_rejects_energy_fields_in_target_entries(self):
        with self.assertRaisesRegex(
            ValueError, "RPA energy is not a selector input"
        ):
            parse_target_entries(
                [
                    {
                        "path": "h2.dat",
                        "family": "atom",
                        "role": "physical",
                        "rpa_binding": 108.72,
                    }
                ]
            )

    def test_rejects_duplicate_family_role_path(self):
        entry = {
            "path": "atom.dat",
            "family": "atom",
            "role": "physical",
        }
        with self.assertRaisesRegex(ValueError, "duplicate Sternheimer target"):
            parse_target_entries([entry, dict(entry)])


if __name__ == "__main__":
    unittest.main()
