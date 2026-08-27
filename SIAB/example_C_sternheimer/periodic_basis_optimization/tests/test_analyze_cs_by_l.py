import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np

from SIAB.example_C_sternheimer.periodic_basis_optimization.ordinary_sos_validation.analyze_cs_by_l import (
    read_uniform_basis_metadata,
    summarize_cs_by_l,
)


class AnalyzeCsByLTest(unittest.TestCase):
    def test_streams_reader_v1_coefficients_by_auxiliary_angular_momentum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            basis_wfc = root / "basis_wfc_out"
            basis_aux = root / "basis_aux_out"
            basis_wfc.write_text("1 8 abacus\n1 4\n1 2\n0\n1\n", encoding="ascii")
            basis_aux.write_text("1 8 abacus\n1 4\n1 2\n0\n1\n", encoding="ascii")

            first = np.arange(1.0, 65.0).reshape(16, 4)
            second = -0.5 * first
            cs_path = root / "v1_Cs_data_0.txt"
            self._write_cs(cs_path, [first, second])

            wfc = read_uniform_basis_metadata(basis_wfc)
            aux = read_uniform_basis_metadata(basis_aux)
            result = summarize_cs_by_l(cs_path, wfc, aux, chunk_bytes=64)

            expected = np.concatenate((first, second), axis=0)
            by_l = {entry["l"]: entry for entry in result["angular_momentum_channels"]}
            self.assertEqual(result["number_of_atoms"], 2)
            self.assertEqual(result["number_of_blocks"], 2)
            self.assertEqual(result["wavefunction_basis_per_atom"], 4)
            self.assertEqual(result["auxiliary_basis_per_atom"], 4)
            self.assertEqual(by_l[0]["coefficient_count"], 32)
            self.assertEqual(by_l[1]["coefficient_count"], 96)
            self.assertAlmostEqual(by_l[0]["frobenius"], np.linalg.norm(expected[:, :1]))
            self.assertAlmostEqual(by_l[1]["frobenius"], np.linalg.norm(expected[:, 1:]))
            self.assertAlmostEqual(by_l[1]["maximum_abs"], np.max(np.abs(expected[:, 1:])))
            self.assertAlmostEqual(result["frobenius"], np.linalg.norm(expected))
            self.assertAlmostEqual(
                sum(entry["fraction_of_squared_norm"] for entry in by_l.values()),
                1.0,
            )

    @staticmethod
    def _write_cs(path: Path, blocks: list[np.ndarray]) -> None:
        number_of_blocks = len(blocks)
        header_size = 28 + number_of_blocks * 36
        offsets = []
        offset = header_size
        for block in blocks:
            offsets.append(offset)
            offset += block.size * 8

        with path.open("wb") as handle:
            handle.write(struct.pack("=iiiqq", -10267453, 2, 1, number_of_blocks, number_of_blocks))
            records = ((1, 1, (0, 0, 0)), (2, 1, (1, 0, 0)))
            for block, file_offset, (ia1, ia2, cell) in zip(blocks, offsets, records):
                handle.write(
                    struct.pack(
                        "=5idq",
                        ia1,
                        ia2,
                        *cell,
                        float(np.max(np.abs(block))),
                        file_offset,
                    )
                )
            for block in blocks:
                handle.write(np.asarray(block, dtype=np.float64).tobytes())


if __name__ == "__main__":
    unittest.main()
