import hashlib
import json
import pathlib
import tempfile
import unittest

from read_periodic_candidate_manifest import read_candidate


class ReadPeriodicCandidateManifestTest(unittest.TestCase):
    def test_reads_nested_tzdp_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "nested.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "nested_tzdp_2s2p1d", "nu": [2, 2, 1], "ao_count_atom": 13, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            result = read_candidate(root)
            self.assertEqual(result["nao_atom"], 13)
            self.assertEqual(result["layout"], "2s2p1d")

    def test_reads_minimal_combined_sp_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "nested.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "nested_tzdp_3s3p1d", "nu": [3, 3, 1], "ao_count_atom": 17, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            result = read_candidate(root)
            self.assertEqual(result["nao_atom"], 17)
            self.assertEqual(result["layout"], "3s3p1d")

    def test_reads_d_complete_s_only_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "nested.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "nested_tzdp_3s2p2d", "nu": [3, 2, 2], "ao_count_atom": 19, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            result = read_candidate(root)
            self.assertEqual(result["nao_atom"], 19)
            self.assertEqual(result["layout"], "3s2p2d")

    def test_reads_d_complete_p_only_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "nested.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "nested_tzdp_2s3p2d", "nu": [2, 3, 2], "ao_count_atom": 21, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            result = read_candidate(root)
            self.assertEqual(result["nao_atom"], 21)
            self.assertEqual(result["layout"], "2s3p2d")

    def test_reads_interpolated_dzp_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "interpolated.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "interpolated_dzp", "nu": [3, 3, 2, 0, 0], "ao_count_atom": 22, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            result = read_candidate(root)
            self.assertEqual(result["nao_atom"], 22)
            self.assertEqual(result["layout"], "3s3p2d")

    def test_reads_controlled_lowest_f_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "controlled-f.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "controlled_lowest_f", "nu": [3, 3, 2, 1, 0], "ao_count_atom": 29, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            result = read_candidate(root)
            self.assertEqual(result["nao_atom"], 29)
            self.assertEqual(result["layout"], "3s3p2d1f")

    def test_rejects_unapproved_nested_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "nested.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "nested_tzdp_1s1p1d", "nu": [1, 1, 1], "ao_count_atom": 9, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "unsupported staged candidate"):
                read_candidate(root)

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

    def test_reads_fixed_prefix_dzp_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            orbital = root / "fixed.orb"
            orbital.write_text("orbital\n", encoding="ascii")
            digest = hashlib.sha256(orbital.read_bytes()).hexdigest()
            (root / "CANDIDATE.json").write_text(
                json.dumps({"status": "success", "profile": "fixed_dzp", "nu": [3, 3, 2, 0, 0], "ao_count_atom": 22, "orbital_filename": orbital.name, "orbital_sha256": digest}),
                encoding="ascii",
            )
            result = read_candidate(root)
            self.assertEqual(result["nao_atom"], 22)
            self.assertEqual(result["layout"], "3s3p2d")


if __name__ == "__main__":
    unittest.main()
