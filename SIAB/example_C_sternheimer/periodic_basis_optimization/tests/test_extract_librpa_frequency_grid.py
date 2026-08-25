import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1] / "ordinary_sos_validation"
sys.path.insert(0, str(MODULE_DIR))

from extract_librpa_frequency_grid import extract_frequency_grid


class ExtractLibrpaFrequencyGridTest(unittest.TestCase):
    def _write_output(self, root: Path, rows: list[tuple[float, float]]) -> Path:
        path = root / "librpa.out"
        lines = [
            "Grid type: Minimax time-frequency grids",
            f"Grid size: {len(rows)}",
            "Frequency node & weight:",
        ]
        lines.extend(
            f"{index:2d} {omega:.16f} {weight:.16f}"
            for index, (omega, weight) in enumerate(rows)
        )
        lines.extend(["Time node & weight:", "libRPA finished successfully"])
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path

    def test_exports_one_based_abacus_grid_and_manifest(self):
        rows = [(0.1, 0.2), (0.5, 0.7), (2.0, 3.0)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_output(root, rows)
            grid = root / "selected_sos_frequency_grid.dat"
            manifest = root / "selected_sos_frequency_grid.json"

            result = extract_frequency_grid(
                source=source,
                output=grid,
                manifest=manifest,
                expected_nfreq=3,
            )

            data_rows = [
                line.split()
                for line in grid.read_text(encoding="ascii").splitlines()
                if line and not line.startswith("#")
            ]
            self.assertEqual([int(row[0]) for row in data_rows], [1, 2, 3])
            self.assertEqual(
                [(float(row[1]), float(row[2])) for row in data_rows], rows
            )
            payload = json.loads(manifest.read_text(encoding="ascii"))
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["nfreq"], 3)
            self.assertEqual(payload["source_sha256"], result["source_sha256"])
            self.assertEqual(payload["grid_sha256"], result["grid_sha256"])

    def test_rejects_nonconsecutive_or_nonpositive_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_output(root, [(0.1, 0.2), (0.5, 0.7)])
            text = source.read_text(encoding="ascii").replace(
                " 1 0.5000000000000000", " 2 0.5000000000000000"
            )
            source.write_text(text, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "consecutive"):
                extract_frequency_grid(
                    source=source,
                    output=root / "grid.dat",
                    manifest=root / "manifest.json",
                    expected_nfreq=2,
                )

            source = self._write_output(root, [(0.1, 0.2), (0.5, -0.7)])
            with self.assertRaisesRegex(ValueError, "positive"):
                extract_frequency_grid(
                    source=source,
                    output=root / "grid2.dat",
                    manifest=root / "manifest2.json",
                    expected_nfreq=2,
                )

    def test_rejects_incomplete_or_ambiguous_frequency_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_output(root, [(0.1, 0.2), (0.5, 0.7)])
            source.write_text(
                source.read_text(encoding="ascii")
                + source.read_text(encoding="ascii"),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                extract_frequency_grid(
                    source=source,
                    output=root / "grid.dat",
                    manifest=root / "manifest.json",
                    expected_nfreq=2,
                )

            source.write_text(
                "Grid size: 2\nFrequency node & weight:\n"
                "0 0.1 0.2\nTime node & weight:\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "expected 2"):
                extract_frequency_grid(
                    source=source,
                    output=root / "grid2.dat",
                    manifest=root / "manifest2.json",
                    expected_nfreq=2,
                )


if __name__ == "__main__":
    unittest.main()
