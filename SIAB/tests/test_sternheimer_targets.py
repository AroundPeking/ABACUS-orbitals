from pathlib import Path
import unittest

import torch

import sys

OPT_DIR = Path(__file__).resolve().parents[1] / "opt_orb_pytorch_dpsi"
if str(OPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPT_DIR))

from sternheimer_data import PrimitiveBlock
from sternheimer_targets import apply_target_element_aliases, parse_target_entries
from test_sternheimer_spillage import make_sternheimer_data


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
                    "element_aliases": {"H_empty": "H"},
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
        self.assertEqual(entries[2].element_aliases, (("H_empty", "H"),))

    def test_applies_explicit_ghost_element_alias_without_changing_atom_index(self):
        data = make_sternheimer_data(
            (
                PrimitiveBlock("H", 0, 0, 0, 1, 0),
                PrimitiveBlock("H_empty", 1, 0, 0, 1, 1),
            ),
            torch.tensor([0.2, 0.3], dtype=torch.float64),
        )
        entry = parse_target_entries(
            [
                {
                    "path": "ghost.dat",
                    "family": "fragment_ghost",
                    "role": "ghost",
                    "element_aliases": {"H_empty": "H"},
                }
            ]
        )[0]

        remapped = apply_target_element_aliases(data, entry)

        self.assertEqual(
            [(block.element, block.atom_index) for block in remapped.blocks],
            [("H", 0), ("H", 1)],
        )
        self.assertIs(remapped.q, data.q)
        self.assertIs(remapped.overlap, data.overlap)

    def test_rejects_alias_source_missing_from_target(self):
        data = make_sternheimer_data(
            [PrimitiveBlock("H", 0, 0, 0, 1, 0)],
            [0.2],
        )
        entry = parse_target_entries(
            [
                {
                    "path": "ghost.dat",
                    "family": "fragment_ghost",
                    "role": "ghost",
                    "element_aliases": {"H_empty": "H"},
                }
            ]
        )[0]

        with self.assertRaisesRegex(ValueError, "alias source.*H_empty"):
            apply_target_element_aliases(data, entry)

    def test_rejects_implicit_or_cyclic_element_aliases(self):
        base = {
            "path": "ghost.dat",
            "family": "fragment_ghost",
            "role": "ghost",
        }
        with self.assertRaisesRegex(ValueError, "element_aliases"):
            parse_target_entries([{**base, "element_aliases": {"H": "H"}}])
        with self.assertRaisesRegex(ValueError, "element_aliases"):
            parse_target_entries(
                [
                    {
                        **base,
                        "element_aliases": {
                            "H_empty": "H",
                            "H": "H_empty",
                        },
                    }
                ]
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
