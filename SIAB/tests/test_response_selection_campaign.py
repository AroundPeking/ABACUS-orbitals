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
    apply_optimizer_loss_overrides,
    assemble_response_families,
    build_response_spectrum_builder,
    extract_fixed_reference_coefficients,
    optimize_response_step,
    read_optimizer_coefficients,
    read_optimizer_metrics,
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
    def test_fixed_reference_is_extracted_from_full_tzdp_prefix(self):
        initial = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [[1.0, 0.25], [0.0, 0.75]], dtype=torch.float64
                ),
                torch.empty((2, 0), dtype=torch.float64),
            ]
        }
        fixed = (
            {"element": "H", "l": 0, "zeta": 1},
            {"element": "H", "l": 0, "zeta": 2},
            {"element": "H", "l": 1, "zeta": 1},
        )

        reference = extract_fixed_reference_coefficients(initial, fixed)

        self.assertEqual(
            [channel.shape for channel in reference["H"]],
            [(2, 2), (2, 1), (2, 0)],
        )
        torch.testing.assert_close(reference["H"][0], initial["H"][0][:, :2])
        torch.testing.assert_close(reference["H"][1], initial["H"][1][:, :1])
        self.assertEqual(initial["H"][0].shape, (2, 3))
        self.assertEqual(initial["H"][1].shape, (2, 2))

    def test_fixed_reference_rejects_nonprefix_zetas(self):
        initial = {"H": [torch.eye(3, dtype=torch.float64)]}

        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            extract_fixed_reference_coefficients(
                initial,
                ({"element": "H", "l": 0, "zeta": 2},),
            )

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
    def test_compact_config_only_overrides_radial_locality_loss(self):
        template = {
            "loss": {
                "mode": "st_dpsi_joint",
                "joint_dpsi_weight": 0.1,
            }
        }
        config = {
            "optimizer_loss": {
                "radial_tail_weight": 0.3,
                "radial_tail_radius": 4.0,
                "radial_tail_condition_limit": 1.0e10,
            }
        }

        result = apply_optimizer_loss_overrides(template, config)

        self.assertEqual(result["loss"]["mode"], "st_dpsi_joint")
        self.assertEqual(result["loss"]["joint_dpsi_weight"], 0.1)
        self.assertEqual(result["loss"]["radial_tail_weight"], 0.3)
        self.assertNotIn("radial_tail_weight", template["loss"])
        with self.assertRaisesRegex(ValueError, "only radial locality"):
            apply_optimizer_loss_overrides(
                template,
                {"optimizer_loss": {"joint_dpsi_weight": 0.0}},
            )

    def test_reads_named_optimizer_loss_and_condition_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ORBITAL_RESULTS.txt"
            path.write_text(
                "<Mkb>\n"
                "Sternheimer loss = 2.5000000000e-01\n"
                "Radial tail fraction = 8.0000000000e-02\n"
                "Radial locality regularization loss = 4.0000000000e-03\n"
                "Total loss = 3.0000000000e-01\n"
                "Maximum ST overlap condition = 1.2000000000e+01\n"
                "Maximum radial locality condition = 8.0000000000e+00\n"
                "</Mkb>\n",
                encoding="utf-8",
            )

            metrics = read_optimizer_metrics(path)

        self.assertEqual(
            metrics,
            {
                "sternheimer": 0.25,
                "radial_tail": 0.08,
                "regularization_locality": 0.004,
                "total": 0.30,
                "max_st_condition": 12.0,
                "max_locality_condition": 8.0,
            },
        )

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
    def test_builds_candidate_spectrum_from_atom_only(self):
        atom_data = make_sternheimer_data((h_s_block(2),), [0.0, 1.0])
        multicenter_data = make_sternheimer_data(
            (h_s_block(2),), [0.0, 3.0]
        )
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
            tuple(zip(entries, (atom_data, multicenter_data)))
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
        self.assertEqual(float(spectra[0].eigenvalues[0]), 1.0)
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

    def test_compact_campaign_freezes_metrics_at_ao_budget(self):
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
            "selection_mode": "ao_budget_frontier",
            "max_ao_per_atom": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer = root / "optimizer.py"
            optimizer.write_text(
                "import json\n"
                "from pathlib import Path\n"
                "value = json.loads(Path('INPUT').read_text())\n"
                "source = Path(value['C_init_info']['C_init_file']).read_text()\n"
                "metrics = ('Sternheimer loss = 2.5000000000e-01\\n'\n"
                "           'Radial tail fraction = 8.0000000000e-02\\n'\n"
                "           'Radial locality regularization loss = 4.0000000000e-03\\n'\n"
                "           'Total loss = 3.0000000000e-01\\n'\n"
                "           'Maximum ST overlap condition = 1.2000000000e+01\\n'\n"
                "           'Maximum radial locality condition = 8.0000000000e+00\\n')\n"
                "Path('ORBITAL_RESULTS.txt').write_text(source.replace('</Mkb>', metrics + '</Mkb>'))\n"
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

            manifest = json.loads(
                result.selection_manifest.read_text(encoding="utf-8")
            )
            campaign = json.loads(
                result.campaign_manifest.read_text(encoding="utf-8")
            )

        self.assertEqual(result.selection.status, "ao_budget_reached")
        self.assertEqual(len(result.selection.steps), 1)
        self.assertEqual(
            manifest["steps"][0]["optimization_metrics"]["radial_tail"],
            0.08,
        )
        self.assertEqual(
            campaign["optimizer_steps"]["1"]["optimization_metrics"]
            ["max_locality_condition"],
            8.0,
        )


if __name__ == "__main__":
    unittest.main()
