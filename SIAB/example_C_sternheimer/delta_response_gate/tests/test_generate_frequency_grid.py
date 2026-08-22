import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_frequency_grid import generate_frequency_grid, parse_greenx_frequency_file


def eig_occ_text(shift=0.0):
    return f"""1 # ionic step
 Electronic state energy (eV) and occupations
 Spin number 2
 spin=1 k-point=1/1 Cartesian=0 0 0 (1 plane wave)
 1 {-10.0 + shift} 1.0
 2 {-4.0 + shift} 1.0
 3 {-3.0 + shift} 1.0
 4 {1.0 + shift} 0.0
 5 {8.0 + shift} 0.0

 spin=2 k-point=1/1 Cartesian=0 0 0 (1 plane wave)
 1 {-8.0 + shift} 1.0
 2 {-2.0 + shift} 0.0
 3 {7.0 + shift} 0.0
"""


class FrequencyGridTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.fixed = self.base / "fixed.txt"
        self.free = self.base / "free.txt"
        self.fixed.write_text(eig_occ_text(), encoding="ascii")
        self.free.write_text(eig_occ_text(0.1), encoding="ascii")
        self.greenx = self.base / "greenx"
        self.greenx.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "input=$1\n"
            "dir=${input%inputs.dat}\n"
            "awk 'BEGIN {for(i=1;i<=6;i++) printf \"%.16e %.16e\\n\", i*0.1, i*0.2}' > \"${dir}freq.dat\"\n",
            encoding="ascii",
        )
        self.greenx.chmod(self.greenx.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self.temporary.cleanup()

    def test_generates_six_point_grid_and_manifest_from_union_window(self):
        output = self.base / "frequency"

        manifest = generate_frequency_grid(
            fixed_eig_occ=self.fixed,
            free_eig_occ=self.free,
            greenx_executable=self.greenx,
            output_dir=output,
            nfreq=6,
        )

        grid = output / "fixed_frequency_grid.dat"
        rows = [line for line in grid.read_text(encoding="ascii").splitlines() if line and not line.startswith("#")]
        self.assertEqual(len(rows), 6)
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["nfreq"], 6)
        self.assertEqual(manifest, json.loads((output / "FREQUENCY_MANIFEST.json").read_text(encoding="ascii")))
        self.assertEqual(manifest["grid_sha256"], __import__("hashlib").sha256(grid.read_bytes()).hexdigest())

    def test_rejects_nonpositive_or_unordered_greenx_rows(self):
        bad = self.base / "bad.dat"
        bad.write_text("0.1 0.2\n0.1 0.3\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_greenx_frequency_file(bad, expected_size=2)

        bad.write_text("0.1 -0.2\n0.2 0.3\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_greenx_frequency_file(bad, expected_size=2)

    def test_refuses_existing_output_directory(self):
        output = self.base / "frequency"
        output.mkdir()
        with self.assertRaisesRegex(FileExistsError, "frequency output already exists"):
            generate_frequency_grid(
                fixed_eig_occ=self.fixed,
                free_eig_occ=self.free,
                greenx_executable=self.greenx,
                output_dir=output,
                nfreq=6,
            )


if __name__ == "__main__":
    unittest.main()
