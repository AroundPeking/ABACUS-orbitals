#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diamond_qstar_sos_gate import QSTAR_REPRESENTATIVES
from sparse_qstar_sos_gate import collect_sparse_qstar_gate


def write_librpa(path, values):
    lines = [
        "RPA correlation energy (Hartree)",
        "| Weighted contribution from each k:",
    ]
    for index, value in enumerate(values, start=1):
        lines.append(f"| ({index},0,0): ({value:.16e},0.0)")
    lines.extend((f"| Total EcRPA: {sum(values):18.9f}", "libRPA finished successfully"))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class SparseQstarSosGateTest(unittest.TestCase):
    def test_sparse_reconstruction_matches_full_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            multiplicities = dict(QSTAR_REPRESENTATIVES)
            representative_values = {index: -1.0e-4 * index for index in multiplicities}
            expected = sum(
                representative_values[index] * multiplicity
                for index, multiplicity in QSTAR_REPRESENTATIVES
            )
            full = [0.0] * 64
            full[0] = expected
            sparse = [0.0] * 64
            for index, value in representative_values.items():
                sparse[index - 1] = value
            sparse_path = root / "sparse.out"
            full_path = root / "full.out"
            write_librpa(sparse_path, sparse)
            write_librpa(full_path, full)

            result = collect_sparse_qstar_gate(
                outputs=[("allband", sparse_path)],
                references={"allband": full_path},
                binding_tolerance_kcal_mol_per_c=0.1,
            )

            row = result["rows"][0]
            self.assertAlmostEqual(row["qstar_reconstruction_ha"], expected, places=14)
            self.assertAlmostEqual(row["reference_ecrpa_ha"], expected, places=9)
            self.assertEqual(result["sparse_qstar_gate"], "pass")

    def test_rejects_nonzero_inactive_q(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = [0.0] * 64
            values[3] = 1.0e-6
            path = root / "sparse.out"
            write_librpa(path, values)
            with self.assertRaisesRegex(ValueError, "inactive q"):
                collect_sparse_qstar_gate(
                    outputs=[("allband", path)],
                    references={},
                    binding_tolerance_kcal_mol_per_c=0.1,
                )

    def test_rejects_missing_reference_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sparse.out"
            reference = root / "reference.out"
            write_librpa(path, [0.0] * 64)
            write_librpa(reference, [0.0] * 64)
            with self.assertRaisesRegex(ValueError, "reference labels"):
                collect_sparse_qstar_gate(
                    outputs=[("allband", path)],
                    references={"other": reference},
                    binding_tolerance_kcal_mol_per_c=0.1,
                )


if __name__ == "__main__":
    unittest.main()
