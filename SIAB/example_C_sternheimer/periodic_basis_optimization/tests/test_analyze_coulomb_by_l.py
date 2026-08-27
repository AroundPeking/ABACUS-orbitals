import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np

from SIAB.example_C_sternheimer.periodic_basis_optimization.ordinary_sos_validation.analyze_coulomb_by_l import (
    expand_angular_momentum_labels,
    read_auxiliary_angular_momenta,
    read_coulomb_v1,
    summarize_coulomb_pair,
)


class AnalyzeCoulombByLTest(unittest.TestCase):
    def test_expands_radial_channels_over_m_components(self) -> None:
        self.assertEqual(expand_angular_momentum_labels([0, 1]), [0, 1, 1, 1])

    def test_reads_basis_and_resolves_error_by_angular_momentum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            basis = root / "basis_aux_out"
            basis.write_text("1 4 abacus\n1 4\n1 2\n0\n1\n", encoding="ascii")

            native = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)
            grid = np.eye(4, dtype=np.complex128)
            native_path = root / "native.dat"
            grid_path = root / "grid.dat"
            self._write_coulomb(native_path, native)
            self._write_coulomb(grid_path, grid)

            radial_l = read_auxiliary_angular_momenta(basis)
            self.assertEqual(radial_l, [0, 1])
            native_read = read_coulomb_v1(native_path)
            grid_read = read_coulomb_v1(grid_path)
            result = summarize_coulomb_pair(
                native_read,
                grid_read,
                expand_angular_momentum_labels(radial_l),
            )

            by_pair = {
                (entry["l_row"], entry["l_column"]): entry
                for entry in result["angular_momentum_blocks"]
            }
            self.assertAlmostEqual(by_pair[(0, 0)]["error_frobenius"], 0.0)
            self.assertAlmostEqual(by_pair[(1, 1)]["error_frobenius"], np.sqrt(14.0))
            self.assertAlmostEqual(result["minimum_eigenvector_l_weights"]["0"], 1.0)
            self.assertAlmostEqual(result["maximum_eigenvector_l_weights"]["1"], 1.0)

    @staticmethod
    def _write_coulomb(path: Path, matrix: np.ndarray) -> None:
        naux = matrix.shape[0]
        header_bytes = 24 + 4 + 12
        with path.open("wb") as handle:
            handle.write(struct.pack("=6i", -20129433, 1, naux, 1, 1, 1))
            handle.write(struct.pack("=i", naux))
            handle.write(struct.pack("=iq", 0, header_bytes))
            handle.write(np.asarray(matrix, dtype=np.complex128).tobytes())


if __name__ == "__main__":
    unittest.main()
