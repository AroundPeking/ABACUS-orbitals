import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "atomic_occupied_gauge.py"
)
SPEC = importlib.util.spec_from_file_location("atomic_occupied_gauge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_wfs(path, coefficients, eigenvalues, occupations):
    lines = [
        f"{coefficients.shape[1]} (number of bands)",
        f"{coefficients.shape[0]} (number of orbitals)",
    ]
    for band in range(coefficients.shape[1]):
        lines.extend(
            (
                f"{band + 1} (band)",
                f"{eigenvalues[band]:.16e} (Ry)",
                f"{occupations[band]:.16e} (Occupations)",
                " ".join(
                    f"{float(value):.16e}" for value in coefficients[:, band]
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class AtomicOccupiedGaugeTest(unittest.TestCase):
    def test_recovers_orthogonal_rotation_within_occupied_subspace(self):
        response = torch.eye(3, dtype=torch.float64)
        rotation = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.8, -0.6],
                [0.0, -0.6, -0.8],
            ],
            dtype=torch.float64,
        )
        source = response @ rotation
        eigenvalues = [-1.0, -0.4, -0.4]
        occupations = [1.0, 1.0, 1.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_path = root / "response.txt"
            source_path = root / "source.txt"
            write_wfs(response_path, response, eigenvalues, occupations)
            write_wfs(source_path, source, eigenvalues, occupations)

            result = MODULE.derive_occupied_gauge(
                (response_path,),
                (source_path,),
            )

        torch.testing.assert_close(result.transform.real.contiguous(), rotation)
        self.assertLess(result.maximum_subspace_residual, 1.0e-14)
        self.assertLess(result.maximum_unitarity_error, 1.0e-14)
        self.assertEqual(result.occupied_counts, (3,))

    def test_rejects_source_outside_response_occupied_subspace(self):
        response = torch.eye(3, dtype=torch.float64)[:, :2]
        source = torch.tensor(
            [[1.0, 0.0], [0.0, 0.8], [0.0, 0.6]], dtype=torch.float64
        )
        eigenvalues = [-1.0, -0.4]
        occupations = [1.0, 1.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_path = root / "response.txt"
            source_path = root / "source.txt"
            write_wfs(response_path, response, eigenvalues, occupations)
            write_wfs(source_path, source, eigenvalues, occupations)

            with self.assertRaisesRegex(ValueError, "occupied subspace residual"):
                MODULE.derive_occupied_gauge(
                    (response_path,),
                    (source_path,),
                    residual_tolerance=1.0e-10,
                )

    def test_rotates_each_channel_from_source_to_response_gauge(self):
        occupied_state = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        auxiliary_channel = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        source_d = torch.tensor(
            [[1.0], [2.0], [3.0], [4.0]], dtype=torch.complex128
        )
        transform = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128
        )

        aligned = MODULE.rotate_source_rows_to_response_gauge(
            source_d,
            occupied_state,
            auxiliary_channel,
            transform,
        )

        expected = torch.tensor(
            [[3.0], [4.0], [1.0], [2.0]], dtype=torch.complex128
        )
        torch.testing.assert_close(aligned, expected)


if __name__ == "__main__":
    unittest.main()
