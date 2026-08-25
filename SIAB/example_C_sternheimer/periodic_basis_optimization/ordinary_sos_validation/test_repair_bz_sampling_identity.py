#!/usr/bin/env python3

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOOL = ROOT / "repair_bz_sampling_identity.py"


class RepairBzSamplingIdentityTest(unittest.TestCase):
    def test_replaces_only_reader_labels(self):
        source_text = """4 4 4
3 3
1 0.25 0.0 0.0 0.0 0.0 0.0 0.0 1 1
2 0.50 0.5 0.0 0.0 0.5 0.0 0.0 1 1
3 0.25 0.5 0.5 0.0 0.5 0.5 0.0 2 2
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.dat"
            output = root / "output.dat"
            report = root / "report.json"
            source.write_text(source_text, encoding="ascii")

            subprocess.run(
                [str(TOOL), str(source), str(output), "--report", str(report)],
                check=True,
            )

            source_rows = [line.split() for line in source_text.splitlines()[2:]]
            output_lines = output.read_text(encoding="ascii").splitlines()
            output_rows = [line.split() for line in output_lines[2:]]
            self.assertEqual(output_lines[:2], source_text.splitlines()[:2])
            self.assertEqual(len(output_rows), 3)
            for index, (source_row, output_row) in enumerate(
                zip(source_rows, output_rows), start=1
            ):
                self.assertEqual(output_row[:-2], source_row[:-2])
                self.assertEqual(output_row[-2:], [str(index), str(index)])

            payload = json.loads(report.read_text(encoding="ascii"))
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["irreducible_count"], 3)
            self.assertEqual(payload["original_unique_reader_labels"], 2)
            self.assertEqual(payload["repaired_unique_reader_labels"], 3)
            self.assertTrue(payload["nonlabel_fields_unchanged"])

    def test_rejects_mismatched_row_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.dat"
            output = root / "output.dat"
            source.write_text(
                "4 4 4\n3 3\n1 1.0 0 0 0 0 0 0 1 1\n", encoding="ascii"
            )
            result = subprocess.run(
                [str(TOOL), str(source), str(output)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
