from pathlib import Path
import json
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
    build_response_spectrum_builder,
    optimize_response_step,
    read_optimizer_coefficients,
    resolve_optimizer_template_paths,
    run_response_selection_campaign,
    write_optimizer_coefficients,
)
from run_response_selection import load_initial_coefficients
from sternheimer_targets import parse_target_entries
from test_sternheimer_spillage import h_s_block, make_sternheimer_data


SIAB_ROOT = Path(__file__).resolve().parents[1]
REAL_H_TZDP = (
    SIAB_ROOT.parent
    / "Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/ORBITAL_RESULTS.txt"
)


def response_data():
    return make_sternheimer_data((h_s_block(2),), [1.0, 0.0])


class OptimizerCoefficientBridgeTest(unittest.TestCase):
    def test_initial_loader_keeps_full_tzdp_and_only_validates_fixed_dzp(self):
        fixed = (
            {"element": "H", "l": 0, "zeta": 1},
            {"element": "H", "l": 0, "zeta": 2},
            {"element": "H", "l": 1, "zeta": 1},
        )

        loaded = load_initial_coefficients(
            REAL_H_TZDP,
            element="H",
            radial_rows=25,
            max_l=4,
            fixed_specs=fixed,
        )

        self.assertEqual(
            [channel.shape for channel in loaded["H"]],
            [(25, 3), (25, 2), (25, 0), (25, 0), (25, 0)],
        )

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
    def test_requires_and_returns_exactly_two_physical_families(self):
        data = response_data()
        entries = parse_target_entries(
            [
                {"path": "atom.dat", "family": "atom", "role": "physical"},
                {
                    "path": "h2.dat",
                    "family": "multicenter",
                    "role": "physical",
                },
            ]
        )

        atom, multicenter = assemble_response_families(
            tuple(zip(entries, (data, data)))
        )

        self.assertEqual(atom.name, "atom")
        self.assertEqual(multicenter.name, "multicenter")

    def test_rejects_ghost_family_in_selection(self):
        data = response_data()
        entries = parse_target_entries(
            [
                {"path": "atom.dat", "family": "atom", "role": "physical"},
                {
                    "path": "h2.dat",
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

        with self.assertRaisesRegex(ValueError, "exactly atom, multicenter"):
            assemble_response_families(tuple(zip(entries, (data, data, data))))


class JointOptimizationBridgeTest(unittest.TestCase):
    def test_resolves_all_legacy_matrix_paths_against_template_directory(self):
        template = {
            "file_list": {
                "origin": ["data/a/orb_matrix.0.dat"],
                "linear": [["data/a/orb_matrix.1.dat"]],
            }
        }
        base = Path("/immutable/campaign")

        resolved = resolve_optimizer_template_paths(template, base)

        self.assertEqual(
            resolved["file_list"]["origin"],
            [str(base / "data/a/orb_matrix.0.dat")],
        )
        self.assertEqual(
            resolved["file_list"]["linear"],
            [[str(base / "data/a/orb_matrix.1.dat")]],
        )
        self.assertEqual(template["file_list"]["origin"][0][0:4], "data")

    def test_runs_native_joint_step_and_reads_back_exact_shape(self):
        coefficients = {
            "H": [
                torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
                torch.tensor([[0.25], [0.75]], dtype=torch.float64),
            ]
        }
        fixed = (
            {"element": "H", "l": 0, "zeta": 1},
            {"element": "H", "l": 0, "zeta": 2},
            {"element": "H", "l": 1, "zeta": 1},
        )
        targets = [
            {"path": "atom.dat", "family": "atom", "role": "physical"},
            {
                "path": "h2.dat",
                "family": "multicenter",
                "role": "physical",
            },
        ]
        template = {
            "seed": 1,
            "file_list": {"origin": ["origin.dat"], "linear": [["dpsi.dat"]]},
            "element": {"Nt_all": ["H"], "Nu": {"H": [2, 1]}},
            "C_init_info": {"init_from_file": True, "C_init_file": "old"},
            "freeze_orbitals": list(fixed),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer = root / "optimizer.py"
            optimizer.write_text(
                "import json, shutil\n"
                "from pathlib import Path\n"
                "value = json.loads(Path('INPUT').read_text())\n"
                "shutil.copyfile(value['C_init_info']['C_init_file'], "
                "'ORBITAL_RESULTS.txt')\n"
                "Path('ORBITAL_1U.dat').write_text('orbital\\n')\n"
                "Path('Spillage.dat').write_text('spillage\\n')\n",
                encoding="utf-8",
            )

            result = optimize_response_step(
                step=1,
                coefficients=coefficients,
                template=template,
                targets=targets,
                fixed_specs=fixed,
                seed=20260720,
                output_dir=root / "step_001",
                optimizer=optimizer,
                python=Path(sys.executable),
            )

            input_value = json.loads(result.input_path.read_text(encoding="utf-8"))
            self.assertEqual(input_value["element"]["Nu"]["H"], [2, 1])
            self.assertEqual(input_value["seed"], 20260720)
            self.assertEqual(set(result.artifact_sha256), {
                "ORBITAL_RESULTS.txt", "ORBITAL_1U.dat", "Spillage.dat"
            })
            for expected, actual in zip(
                coefficients["H"], result.coefficients["H"]
            ):
                self.assertTrue(torch.equal(expected, actual))


class PhysicalSpectrumBuilderTest(unittest.TestCase):
    def test_aggregates_atom_and_multicenter_residual_for_each_l(self):
        data = make_sternheimer_data((h_s_block(2),), [0.0, 1.0])
        entries = parse_target_entries(
            [
                {"path": "atom.dat", "family": "atom", "role": "physical"},
                {
                    "path": "h2.dat",
                    "family": "multicenter",
                    "role": "physical",
                },
            ]
        )
        atom, multicenter = assemble_response_families(
            tuple(zip(entries, (data, data)))
        )
        current = {
            "H": [torch.tensor([[1.0], [0.0]], dtype=torch.float64)]
        }
        builder = build_response_spectrum_builder(
            atom,
            multicenter,
            element="H",
            max_l=0,
            relative_rank_tolerance=1.0e-4,
            magnetic_overlap_tolerance=1.0e-4,
            condition_limit=1.0e12,
        )

        spectra = builder(current)

        self.assertEqual(len(spectra), 1)
        self.assertEqual(spectra[0].l, 0)
        self.assertGreater(float(spectra[0].eigenvalues[0]), 0.0)
        torch.testing.assert_close(
            torch.abs(spectra[0].coefficients[:, 0]),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        )

    def test_one_step_campaign_rebuilds_spectrum_and_freezes_manifest(self):
        data = make_sternheimer_data((h_s_block(2),), [0.0, 1.0])
        target_values = [
            {"path": "atom.dat", "family": "atom", "role": "physical"},
            {
                "path": "h2.dat",
                "family": "multicenter",
                "role": "physical",
            },
        ]
        entries = parse_target_entries(target_values)
        families = assemble_response_families(
            tuple(zip(entries, (data, data)))
        )
        initial = {
            "H": [torch.tensor([[1.0], [0.0]], dtype=torch.float64)]
        }
        fixed = ({"element": "H", "l": 0, "zeta": 1},)
        template = {
            "seed": 1,
            "file_list": {"origin": ["origin.dat"], "linear": [["dpsi.dat"]]},
            "element": {"Nt_all": ["H"], "Nu": {"H": [1]}},
            "C_init_info": {"init_from_file": True, "C_init_file": "old"},
            "freeze_orbitals": list(fixed),
        }
        config = {
            "format_version": 1,
            "seed": 20260720,
            "max_l": 0,
            "relative_rank_tolerance": 1.0e-4,
            "magnetic_overlap_tolerance": 1.0e-4,
            "global_capture": 0.999,
            "per_l_residual_limit": 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer = root / "optimizer.py"
            optimizer.write_text(
                "import json, shutil\n"
                "from pathlib import Path\n"
                "value = json.loads(Path('INPUT').read_text())\n"
                "shutil.copyfile(value['C_init_info']['C_init_file'], "
                "'ORBITAL_RESULTS.txt')\n"
                "Path('ORBITAL_1U.dat').write_text('orbital\\n')\n"
                "Path('Spillage.dat').write_text('spillage\\n')\n",
                encoding="utf-8",
            )

            result = run_response_selection_campaign(
                config=config,
                initial=initial,
                fixed_specs=fixed,
                families=families,
                optimizer_template=template,
                targets=target_values,
                output_dir=root / "campaign",
                optimizer=optimizer,
                python=Path(sys.executable),
                condition_limit=1.0e12,
                max_steps=3,
            )

            self.assertEqual(result.selection.status, "converged")
            self.assertEqual(len(result.selection.steps), 1)
            self.assertTrue(result.selection_manifest.is_file())
            manifest = json.loads(
                result.campaign_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "converged")
            self.assertEqual(manifest["steps"], 1)
            self.assertEqual(
                [target["role"] for target in manifest["targets"]],
                ["physical", "physical"],
            )
            self.assertNotIn("h2_energy", result.campaign_manifest.read_text())


if __name__ == "__main__":
    unittest.main()
