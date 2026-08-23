import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_frequency_grid import generate_frequency_grid


def eig_occ_text():
    return """1 # ionic step
 spin=1 k-point=1/1 Cartesian=0 0 0 (1 plane wave)
 1 -10.0 1.0
 2 -4.0 1.0
 3 -3.0 1.0
 4 1.0 0.0

 spin=2 k-point=1/1 Cartesian=0 0 0 (1 plane wave)
 1 -8.0 1.0
 2 -2.0 0.0
"""


class FrequencyGridTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.eig = self.base / "eig_occ.txt"
        self.eig.write_text(eig_occ_text(), encoding="ascii")
        self.greenx = self.base / "greenx"
        self.greenx.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "text = pathlib.Path(sys.argv[1]).read_text()\n"
            "assert 'n_mesh_points 16' in text\n"
            "out = pathlib.Path(text.splitlines()[0])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'freq.dat').write_text('\\n'.join("
            "f'{i * 0.01:.16e} {i * 0.02:.16e}' for i in range(1, 17)) + '\\n')\n",
            encoding="ascii",
        )
        self.greenx.chmod(self.greenx.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self.temporary.cleanup()

    def test_generates_immutable_sixteen_point_grid(self):
        output = self.base / "frequency"
        manifest = generate_frequency_grid(
            eig_occ=self.eig,
            greenx_executable=self.greenx,
            output_dir=output,
        )

        self.assertEqual(manifest["nfreq"], 16)
        self.assertEqual(manifest["status"], "success")
        grid = output / "fixed_frequency_grid_nfreq16.dat"
        rows = [line for line in grid.read_text(encoding="ascii").splitlines() if line and not line.startswith("#")]
        self.assertEqual(len(rows), 16)
        self.assertTrue((output / "FREQUENCY_MANIFEST.json").is_file())
        recorded = json.loads((output / "FREQUENCY_MANIFEST.json").read_text(encoding="ascii"))
        self.assertEqual(recorded["grid_sha256"], manifest["grid_sha256"])

    def test_refuses_existing_output(self):
        output = self.base / "exists"
        output.mkdir()
        with self.assertRaisesRegex(FileExistsError, "frequency output already exists"):
            generate_frequency_grid(
                eig_occ=self.eig,
                greenx_executable=self.greenx,
                output_dir=output,
            )


if __name__ == "__main__":
    unittest.main()
