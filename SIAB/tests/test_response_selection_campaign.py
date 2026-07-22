from pathlib import Path
import sys
import tempfile
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path

SELECTOR_DIR = (
    Path(__file__).resolve().parents[1]
    / "example_H_sternheimer/greedy_response_selection"
)
if str(SELECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(SELECTOR_DIR))

from response_selection_campaign import (
    assemble_response_families,
    read_optimizer_coefficients,
    write_optimizer_coefficients,
)
from sternheimer_targets import parse_target_entries
from test_sternheimer_spillage import h_s_block, make_sternheimer_data


def response_data():
    return make_sternheimer_data((h_s_block(2),), [1.0, 0.0])


class OptimizerCoefficientBridgeTest(unittest.TestCase):
    def test_round_trip_preserves_columns_and_empty_channels_exactly(self):
        coefficients = {
            "H": [
                torch.tensor(
                    [[1.0, -0.0], [0.25, -0.5]], dtype=torch.float64
                ),
                torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                torch.empty((2, 0), dtype=torch.float64),
                torch.empty((2, 0), dtype=torch.float64),
                torch.empty((2, 0), dtype=torch.float64),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ORBITAL_RESULTS.txt"
            write_optimizer_coefficients(path, coefficients)
            loaded = read_optimizer_coefficients(
                path,
                element="H",
                radial_rows=2,
                max_l=4,
                expected_nu=(2, 1, 0, 0, 0),
            )

        self.assertEqual(len(loaded["H"]), 5)
        for expected, actual in zip(coefficients["H"], loaded["H"]):
            self.assertTrue(torch.equal(expected, actual))

    def test_rejects_missing_requested_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ORBITAL_RESULTS.txt"
            path.write_text(
                "<Coefficient>\n"
                " 2 Total number of radial orbitals.\n"
                " Type L Zeta-Orbital\n"
                " H 0 1\n"
                " 1.0\n"
                " 0.0\n"
                "</Coefficient>\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing coefficient column"):
                read_optimizer_coefficients(
                    path,
                    element="H",
                    radial_rows=2,
                    max_l=1,
                    expected_nu=(2, 0),
                )


class ResponseFamilyAssemblyTest(unittest.TestCase):
    def test_requires_and_returns_the_three_frozen_families(self):
        data = response_data()
        entries = parse_target_entries(
            [
                {"path": "atom.dat", "family": "atom", "role": "physical"},
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

        atom, multicenter, ghost = assemble_response_families(
            tuple(zip(entries, (data, data, data)))
        )

        self.assertEqual(atom.name, "atom")
        self.assertEqual(multicenter.name, "multicenter")
        self.assertEqual(ghost.name, "fragment_ghost")
        self.assertEqual(ghost.real_atom_index, 0)

    def test_rejects_missing_ghost_family(self):
        data = response_data()
        entries = parse_target_entries(
            [
                {"path": "atom.dat", "family": "atom", "role": "physical"},
                {
                    "path": "h3.dat",
                    "family": "multicenter",
                    "role": "physical",
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "exactly atom, multicenter"):
            assemble_response_families(tuple(zip(entries, (data, data))))


if __name__ == "__main__":
    unittest.main()
