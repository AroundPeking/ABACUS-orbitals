#!/usr/bin/env python3

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from prepare_nested_tzdp_candidate import prepare_candidate


FIXTURE = """---------------------------------------------------------------------------
Element                     C
Lmax                        2
Number of Sorbital-->       3
Number of Porbital-->       3
Number of Dorbital-->       2
---------------------------------------------------------------------------
Mesh                        2
                Type                   L                   N
                   0                   0                   0
1.0 1.1
                Type                   L                   N
                   0                   0                   1
2.0 2.1
                Type                   L                   N
                   0                   0                   2
3.0 3.1
                Type                   L                   N
                   0                   1                   0
4.0 4.1
                Type                   L                   N
                   0                   1                   1
5.0 5.1
                Type                   L                   N
                   0                   1                   2
6.0 6.1
                Type                   L                   N
                   0                   2                   0
7.0 7.1
                Type                   L                   N
                   0                   2                   1
8.0 8.1
"""


class PrepareNestedTzdpCandidateTest(unittest.TestCase):
    def test_writes_immutable_candidate_with_locked_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source.orb"
            root = parent / "candidate"
            source.write_text(FIXTURE, encoding="ascii")

            result = prepare_candidate(source, root, target_nu=[2, 2, 1])

            payload = json.loads((root / "CANDIDATE.json").read_text(encoding="ascii"))
            selection = json.loads((root / "SELECTION.json").read_text(encoding="ascii"))
            orbital = root / payload["orbital_filename"]
            self.assertEqual(payload["profile"], "nested_tzdp_2s2p1d")
            self.assertEqual(payload["nu"], [2, 2, 1])
            self.assertEqual(payload["ao_count_atom"], 13)
            self.assertEqual(payload["orbital_sha256"], hashlib.sha256(orbital.read_bytes()).hexdigest())
            self.assertEqual(result["orbital_sha256"], payload["orbital_sha256"])
            self.assertEqual(selection["output"], str(orbital.resolve()))
            self.assertEqual((root / "STATUS").read_text(encoding="ascii"), "success\n")
            self.assertIn("status=success\n", (root / "provenance.txt").read_text(encoding="ascii"))

    def test_writes_minimal_combined_sp_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source.orb"
            root = parent / "candidate"
            source.write_text(FIXTURE, encoding="ascii")

            result = prepare_candidate(source, root, target_nu=[3, 3, 1])

            self.assertEqual(result["profile"], "nested_tzdp_3s3p1d")
            self.assertEqual(result["nu"], [3, 3, 1])
            self.assertEqual(result["ao_count_atom"], 17)

    def test_writes_d_complete_s_only_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source.orb"
            root = parent / "candidate"
            source.write_text(FIXTURE, encoding="ascii")

            result = prepare_candidate(source, root, target_nu=[3, 2, 2])

            self.assertEqual(result["profile"], "nested_tzdp_3s2p2d")
            self.assertEqual(result["nu"], [3, 2, 2])
            self.assertEqual(result["ao_count_atom"], 19)

    def test_rejects_non_whitelisted_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source.orb"
            source.write_text(FIXTURE, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "approved nested layouts"):
                prepare_candidate(source, parent / "candidate", target_nu=[1, 1, 1])

    def test_refuses_existing_root(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source.orb"
            root = parent / "candidate"
            source.write_text(FIXTURE, encoding="ascii")
            root.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_candidate(source, root, target_nu=[2, 2, 1])


if __name__ == "__main__":
    unittest.main()
