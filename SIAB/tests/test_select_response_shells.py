import json
from pathlib import Path
import struct
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

from response_selection import (
    CandidateEvaluation,
    CandidateGain,
    ResponseTargetFamily,
)
from select_response_shells import (
    FrozenSelectionStep,
    build_step_input,
    freeze_selection_sequence,
    load_coefficients_json,
    run_nested_selection,
    run_joint_optimizer,
    select_one_response_shell,
)
from sternheimer_data import PrimitiveBlock
from sternheimer_spillage import (
    RadialResidualSpectrum,
    radial_residual_spectrum_many,
)
from test_response_selection import (
    response_coefficients,
    response_spectrum,
    response_target_families,
)
from test_sternheimer_spillage import make_sternheimer_data


FIXED_DZP = (
    {"element": "H", "l": 0, "zeta": 1},
    {"element": "H", "l": 0, "zeta": 2},
    {"element": "H", "l": 1, "zeta": 1},
)


def baseline_coefficients():
    return {
        "H": [
            torch.tensor(
                [[1.0, 0.0], [0.0, -0.0], [0.0, 1.0]],
                dtype=torch.float64,
            ),
            torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64),
            torch.empty((3, 0), dtype=torch.float64),
            torch.empty((3, 0), dtype=torch.float64),
            torch.empty((3, 0), dtype=torch.float64),
        ]
    }


def append_column(coefficients, l, column):
    result = {
        element: [value.detach().clone() for value in by_l]
        for element, by_l in coefficients.items()
    }
    result["H"][l] = torch.cat(
        (
            result["H"][l],
            torch.tensor(column, dtype=torch.float64).reshape(-1, 1),
        ),
        dim=1,
    )
    return result


def accepted(l, mode, atom, multicenter):
    gain = CandidateGain(l, mode, atom, multicenter)
    return CandidateEvaluation(
        gain=gain,
        score=(atom + multicenter) / (2 * l + 1),
        admissible=True,
        rejection_reason=None,
    )


def three_steps():
    current = baseline_coefficients()
    result = []
    for l, column in (
        (2, [0.0, 1.0, 0.0]),
        (0, [1.0, 1.0, 0.0]),
        (3, [0.0, 0.0, 1.0]),
    ):
        current = append_column(current, l, column)
        selected = accepted(l, 0, 0.5, 0.25)
        result.append(
            FrozenSelectionStep(
                selected=selected,
                candidates=(selected,),
                coefficients=current,
            )
        )
    return tuple(result)


def float64_bytes(values):
    return b"".join(
        struct.pack("=d", float(value)) for value in values.reshape(-1)
    )


class FrozenSelectionManifestTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_contains_no_energy(self):
        config = {"format_version": 1, "seed": 20260720}
        with tempfile.TemporaryDirectory() as first_dir:
            first = freeze_selection_sequence(
                Path(first_dir),
                config,
                baseline_coefficients(),
                FIXED_DZP,
                three_steps(),
            )
            first_bytes = first.read_bytes()
        with tempfile.TemporaryDirectory() as second_dir:
            second = freeze_selection_sequence(
                Path(second_dir),
                config,
                baseline_coefficients(),
                FIXED_DZP,
                three_steps(),
            )
            second_bytes = second.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        text = first_bytes.decode("utf-8").lower()
        self.assertNotIn("h2_energy", text)
        self.assertNotIn("rpa_binding", text)

    def test_each_step_keeps_fixed_dzp_columns_bitwise_equal(self):
        baseline = baseline_coefficients()
        with tempfile.TemporaryDirectory() as directory:
            manifest = freeze_selection_sequence(
                Path(directory),
                {"format_version": 1, "seed": 20260720},
                baseline,
                FIXED_DZP,
                three_steps(),
            )
            values = json.loads(manifest.read_text(encoding="utf-8"))
            for step in values["steps"]:
                current = load_coefficients_json(
                    Path(directory) / step["coefficients"]
                )
                for spec in FIXED_DZP:
                    l = spec["l"]
                    zeta = spec["zeta"] - 1
                    self.assertEqual(
                        float64_bytes(current["H"][l][:, zeta]),
                        float64_bytes(baseline["H"][l][:, zeta]),
                    )

    def test_energy_fields_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "energy fields"):
                freeze_selection_sequence(
                    Path(directory),
                    {"format_version": 1, "h2_energy": -1.0},
                    baseline_coefficients(),
                    FIXED_DZP,
                    three_steps(),
                )


class OneStepSelectorTest(unittest.TestCase):
    def test_selects_and_appends_best_candidate_with_full_score_table(self):
        atom, multicenter = response_target_families()
        current = response_coefficients()

        step = select_one_response_shell(
            current,
            current,
            (
                response_spectrum(1, 0.0, [[0.0], [1.0]]),
                response_spectrum(0, 1.0, [[0.0], [1.0]]),
            ),
            atom,
            multicenter,
        )

        self.assertEqual(step.selected.gain.key, (0, 0))
        self.assertEqual(len(step.candidates), 2)
        self.assertFalse(step.candidates[1].admissible)
        self.assertEqual(step.coefficients["H"][0].shape, (2, 2))
        self.assertEqual(current["H"][0].shape, (2, 1))

    def test_fails_explicitly_when_no_positive_candidate_remains(self):
        atom, multicenter = response_target_families()
        current = response_coefficients()

        with self.assertRaisesRegex(RuntimeError, "no admissible positive-score"):
            select_one_response_shell(
                current,
                current,
                (response_spectrum(1, 0.0, [[0.0], [1.0]]),),
                atom,
                multicenter,
            )


class NestedSelectorTest(unittest.TestCase):
    def test_rebuilds_spectrum_and_stops_after_three_captured_modes(self):
        data = make_sternheimer_data(
            [PrimitiveBlock("H", 0, 0, 0, 4, 0)],
            torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.complex128,
            ),
            norm=torch.ones(3, dtype=torch.float64),
        )
        atom = ResponseTargetFamily("atom", (data,), "physical")
        multicenter = ResponseTargetFamily("multicenter", (data,), "physical")
        initial = {
            "H": [
                torch.tensor(
                    [[1.0], [0.0], [0.0], [0.0]], dtype=torch.float64
                )
            ]
        }

        def spectra_for(coefficients):
            n_column = coefficients["H"][0].shape[1]
            if n_column == 4:
                return (
                    RadialResidualSpectrum(
                        element="H",
                        atom_index=None,
                        l=0,
                        magnetic_channels=(0,),
                        numerical_rank=1,
                        eigenvalues=torch.zeros(1, dtype=torch.float64),
                        cumulative_capture=torch.zeros(1, dtype=torch.float64),
                        coefficients=torch.zeros((4, 1), dtype=torch.float64),
                        overlap_relative_deviation=0.0,
                        atom_indices=(0,),
                    ),
                )
            projected = [
                {"element": "H", "l": 0, "zeta": zeta}
                for zeta in range(1, n_column + 1)
            ]
            return (
                radial_residual_spectrum_many(
                    (data,), coefficients, projected, "H", 0
                ),
            )

        optimized_steps = []

        def optimize_step(index, coefficients, selected):
            optimized_steps.append((index, selected.gain.key))
            return coefficients

        result = run_nested_selection(
            {
                "global_capture": 0.999,
                "per_l_residual_limit": 0.01,
            },
            initial,
            initial,
            ({"element": "H", "l": 0, "zeta": 1},),
            atom,
            multicenter,
            spectra_for,
            optimize_step,
            max_steps=4,
        )

        self.assertEqual(result.status, "converged")
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(optimized_steps, [(1, (0, 0)), (2, (0, 0)), (3, (0, 0))])
        self.assertGreaterEqual(result.metrics.global_capture, 0.999)
        self.assertLessEqual(result.metrics.per_l_residual_ratio[0], 0.01)
        self.assertFalse(hasattr(result.metrics, "borrowing"))


class JointOptimizerContractTest(unittest.TestCase):
    def test_step_input_keeps_two_physical_targets_and_fixed_dzp(self):
        template = {
            "seed": 1,
            "file_list": {"origin": ["origin.dat"], "linear": [["dpsi.dat"]]},
            "element": {"Nt_all": ["H"], "Nu": {"H": [2, 1]}},
            "C_init_info": {"init_from_file": True, "C_init_file": "old"},
            "freeze_orbitals": [],
        }
        targets = [
            {"path": "atom.dat", "family": "atom", "role": "physical"},
            {
                "path": "h2.dat",
                "family": "multicenter",
                "role": "physical",
            },
        ]

        value = build_step_input(
            template,
            targets,
            Path("initial.txt"),
            {"H": [3, 2, 1, 0, 0]},
            FIXED_DZP,
            seed=20260720,
        )

        self.assertEqual(value["file_list"]["sternheimer"], targets)
        self.assertEqual(
            [target["role"] for target in value["file_list"]["sternheimer"]],
            ["physical", "physical"],
        )
        self.assertEqual(value["element"]["Nu"]["H"], [3, 2, 1, 0, 0])
        self.assertEqual(value["freeze_orbitals"], list(FIXED_DZP))
        self.assertEqual(value["C_init_info"]["C_init_file"], "initial.txt")
        self.assertEqual(value["seed"], 20260720)

    def test_step_input_rejects_ghost_control_target(self):
        template = {
            "seed": 1,
            "file_list": {"origin": ["origin.dat"], "linear": [["dpsi.dat"]]},
            "element": {"Nt_all": ["H"], "Nu": {"H": [2, 1]}},
            "C_init_info": {"init_from_file": True, "C_init_file": "old"},
            "freeze_orbitals": [],
        }
        targets = [
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

        with self.assertRaisesRegex(ValueError, "exactly atom and multicenter"):
            build_step_input(
                template,
                targets,
                Path("initial.txt"),
                {"H": [3, 2, 1, 0, 0]},
                FIXED_DZP,
                seed=20260720,
            )

    def test_optimizer_requires_all_three_output_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer = root / "optimizer.py"
            optimizer.write_text(
                "from pathlib import Path\n"
                "for name in ('ORBITAL_RESULTS.txt', 'ORBITAL_1U.dat', "
                "'Spillage.dat'):\n"
                "    Path(name).write_text(name + '\\n')\n",
                encoding="utf-8",
            )
            output = root / "step"

            artifacts = run_joint_optimizer(
                {"seed": 20260720},
                output,
                optimizer,
                Path(sys.executable),
            )

            self.assertEqual(
                set(artifacts),
                {"ORBITAL_RESULTS.txt", "ORBITAL_1U.dat", "Spillage.dat"},
            )
            self.assertTrue((output / "run.log").is_file())


if __name__ == "__main__":
    unittest.main()
