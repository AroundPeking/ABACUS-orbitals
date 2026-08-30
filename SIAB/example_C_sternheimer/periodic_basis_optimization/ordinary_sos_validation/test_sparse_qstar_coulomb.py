#!/usr/bin/env python3

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparse_qstar_coulomb import COULOMB_V1_MARKER, write_zero_coulomb_v1


def write_fixture(path, *, marker=COULOMB_V1_MARKER, truncate=False):
    atom_naux = (2, 1)
    pairs = ((0, 0), (0, 1), (1, 1))
    prefix_size = 24 + 4 * len(atom_naux) + 12 * len(pairs)
    offsets = []
    cursor = prefix_size
    for atom_i, atom_j in pairs:
        offsets.append(cursor)
        cursor += 16 * atom_naux[atom_i] * atom_naux[atom_j]
    payload = bytearray(cursor)
    struct.pack_into("<6i", payload, 0, marker, 3, sum(atom_naux), 1, 2, 3)
    struct.pack_into("<2i", payload, 24, *atom_naux)
    table_offset = 32
    for pair_index, offset in enumerate(offsets):
        struct.pack_into("<iq", payload, table_offset, pair_index, offset)
        table_offset += 12
    for offset, (atom_i, atom_j) in zip(offsets, pairs):
        count = atom_naux[atom_i] * atom_naux[atom_j]
        for index in range(count):
            struct.pack_into("<dd", payload, offset + 16 * index, index + 1.0, -index - 0.5)
    if truncate:
        payload = payload[:-8]
    path.write_bytes(payload)
    return offsets, atom_naux


class SparseQstarCoulombTest(unittest.TestCase):
    def test_zero_writer_preserves_layout_and_changes_iq(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dat"
            output = root / "zero.dat"
            offsets, atom_naux = write_fixture(source)

            report = write_zero_coulomb_v1(source, output, iq=17)

            data = output.read_bytes()
            header = struct.unpack_from("<6i", data, 0)
            self.assertEqual(header, (COULOMB_V1_MARKER, 17, 3, 1, 2, 3))
            self.assertEqual(struct.unpack_from("<2i", data, 24), atom_naux)
            self.assertEqual(len(data), len(source.read_bytes()))
            self.assertTrue(all(byte == 0 for byte in data[min(offsets) :]))
            self.assertEqual(report["status"], "success")
            self.assertEqual(report["iq"], 17)
            self.assertEqual(report["zeroed_complex_values"], 7)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dat"
            output = root / "zero.dat"
            write_fixture(source)
            output.write_bytes(b"occupied")
            with self.assertRaises(FileExistsError):
                write_zero_coulomb_v1(source, output, iq=1)

    def test_rejects_bad_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dat"
            write_fixture(source, marker=123)
            with self.assertRaisesRegex(ValueError, "marker"):
                write_zero_coulomb_v1(source, root / "zero.dat", iq=1)

    def test_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dat"
            write_fixture(source, truncate=True)
            with self.assertRaisesRegex(ValueError, "payload"):
                write_zero_coulomb_v1(source, root / "zero.dat", iq=1)

    def test_rejects_invalid_iq(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dat"
            write_fixture(source)
            with self.assertRaisesRegex(ValueError, "q-point"):
                write_zero_coulomb_v1(source, root / "zero.dat", iq=0)


if __name__ == "__main__":
    unittest.main()
