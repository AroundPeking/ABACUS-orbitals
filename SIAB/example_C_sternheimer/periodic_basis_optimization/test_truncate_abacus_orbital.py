#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from truncate_abacus_orbital import truncate_abacus_orbital


FIXTURE = """---------------------------------------------------------------------------
Element                     C
Energy Cutoff(Ry)           100.0
Radius Cutoff(a.u.)         10.0
Lmax                        4
Number of Sorbital-->       1
Number of Porbital-->       1
Number of Dorbital-->       1
Number of Forbital-->       1
Number of Gorbital-->       1
---------------------------------------------------------------------------
SUMMARY  END

Mesh                        3
dr                          0.01
                Type                   L                   N
                   0                   0                   0
1.0  2.0  3.0

                Type                   L                   N
                   0                   1                   0
4.0  5.0  6.0

                Type                   L                   N
                   0                   2                   0
7.0  8.0  9.0

                Type                   L                   N
                   0                   3                   0
10.0  11.0  12.0

                Type                   L                   N
                   0                   4                   0
13.0  14.0  15.0

"""


class TruncateAbacusOrbitalTest(unittest.TestCase):
    def test_drops_channels_above_target_lmax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.orb"
            output = root / "output.orb"
            source.write_text(FIXTURE, encoding="ascii")

            report = truncate_abacus_orbital(source, output, target_lmax=3)

            text = output.read_text(encoding="ascii")
            self.assertIn("Lmax                        3", text)
            self.assertIn("Number of Forbital-->       1", text)
            self.assertNotIn("Number of Gorbital", text)
            self.assertNotIn("13.0  14.0  15.0", text)
            self.assertIn("10.0  11.0  12.0", text)
            self.assertEqual(report["source_nu"], [1, 1, 1, 1, 1])
            self.assertEqual(report["output_nu"], [1, 1, 1, 1])

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.orb"
            output = root / "output.orb"
            source.write_text(FIXTURE, encoding="ascii")
            output.write_text("occupied", encoding="ascii")
            with self.assertRaises(FileExistsError):
                truncate_abacus_orbital(source, output, target_lmax=3)

    def test_rejects_missing_radial_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.orb"
            source.write_text(FIXTURE.replace("13.0  14.0  15.0\n\n", ""), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "radial block"):
                truncate_abacus_orbital(source, root / "output.orb", target_lmax=3)

    def test_rejects_target_not_smaller_than_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.orb"
            source.write_text(FIXTURE, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "smaller"):
                truncate_abacus_orbital(source, root / "output.orb", target_lmax=4)


if __name__ == "__main__":
    unittest.main()
