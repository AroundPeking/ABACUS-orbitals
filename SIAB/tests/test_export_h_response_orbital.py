"""Tests for exporting optimized H response coefficients as ABACUS orbitals."""

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "example_H_sternheimer" / "delta_st_response_compression"
sys.path.insert(0, str(RUNNER_DIR))

from export_h_response_orbital import export_h_response_orbital


REAL_H_TZDP = (
    ROOT.parent
    / "Dojo-NC-SR"
    / "Orbitals_v2.0"
    / "H_TZDP"
    / "info"
    / "8"
    / "ORBITAL_RESULTS.txt"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExportHResponseOrbitalTest(unittest.TestCase):
    def test_exports_orthonormal_tzdp_without_changing_radial_subspaces(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            orbital_path = directory / "H_response_3s2p.orb"
            manifest_path = directory / "export.json"

            manifest = export_h_response_orbital(
                REAL_H_TZDP,
                orbital_path,
                manifest_path,
                nu=(3, 2),
                siab_commit="test-commit",
            )

            text = orbital_path.read_text(encoding="ascii")
            self.assertIn("Lmax                        1", text)
            self.assertIn("Number of Sorbital-->       3", text)
            self.assertIn("Number of Porbital-->       2", text)
            self.assertEqual(text.count("Type                   L                   N"), 5)
            self.assertEqual(manifest["radial_orbitals_by_l"], [3, 2])
            self.assertEqual(manifest["candidate_ao_dimension"], 9)
            self.assertEqual(manifest["source_siab_commit"], "test-commit")
            self.assertEqual(manifest["orbital_sha256"], _sha256(orbital_path))
            for channel in manifest["subspace_validation"]:
                self.assertLess(channel["orthonormality_max_abs_error"], 1.0e-12)
                self.assertLess(channel["maximum_relative_span_residual"], 1.0e-12)
                self.assertLess(channel["serialized_max_abs_error"], 1.0e-13)

            on_disk = json.loads(manifest_path.read_text(encoding="ascii"))
            self.assertEqual(on_disk, manifest)

    def test_rejects_a_coefficient_file_that_does_not_define_every_requested_orbital(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            with self.assertRaisesRegex(ValueError, "does not define every requested AO"):
                export_h_response_orbital(
                    REAL_H_TZDP,
                    directory / "invalid.orb",
                    directory / "invalid.json",
                    nu=(3, 3, 2),
                    siab_commit="test-commit",
                )


if __name__ == "__main__":
    unittest.main()
