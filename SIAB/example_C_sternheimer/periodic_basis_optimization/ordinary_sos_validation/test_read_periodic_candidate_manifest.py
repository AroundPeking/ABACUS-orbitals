import hashlib
import json
import pathlib
import tempfile
import unittest

from read_periodic_candidate_manifest import read_candidate


class ReadPeriodicCandidateManifestTest(unittest.TestCase):
    def test_reads_relaxed_dzp_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "relaxed.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "relaxed_dzp", "nu": [3, 3, 2, 0, 0], "ao_count_atom": 22, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            result = read_candidate(root)
            self.assertEqual(result["nao_atom"], 22)
            self.assertEqual(result["layout"], "3s3p2d")

    def test_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "relaxed.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "relaxed_dzp", "nu": [3, 3, 2, 0, 0], "ao_count_atom": 22, "orbital_filename": orbital.name, "orbital_sha256": "0" * 64}),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                read_candidate(root)


if __name__ == "__main__":
    unittest.main()
