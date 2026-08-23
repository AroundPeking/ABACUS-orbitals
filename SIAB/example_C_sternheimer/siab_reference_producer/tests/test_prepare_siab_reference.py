import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_siab_reference import RESTART_NAMES, prepare_siab_reference


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareSiabReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.response = self.base / "response"
        self.case = self.response / "branches/fixed"
        response_restart = self.case / "OUT.C_DELTA_RESPONSE_GATE"
        response_restart.mkdir(parents=True)
        self.source_phase = self.base / "pbe/runs/fixed/fixed_zero_restart"
        source_restart = self.source_phase / "OUT.C_PBE_REFERENCE_GATE"
        source_restart.mkdir(parents=True)
        assets = {
            "STRU": "ATOMIC_SPECIES\nC 12.011 C_ONCV_PBE-1.0.upf\n",
            "KPT": "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n",
            "C_ONCV_PBE-1.0.upf": "pseudo\n",
            "C_gga_10au_100Ry_3s3p2d.orb": "orbital\n",
            "SOURCE_EIG_OCC.txt": "triplet eig occ\n",
        }
        for name, content in assets.items():
            (self.case / name).write_text(content, encoding="ascii")
            if name == "SOURCE_EIG_OCC.txt":
                (source_restart / "eig_occ.txt").write_text(content, encoding="ascii")
            else:
                (self.source_phase / name).write_text(content, encoding="ascii")
        for name in RESTART_NAMES:
            (source_restart / name).write_text(f"original restart {name}\n", encoding="ascii")
            # A completed response run may rewrite these files.  They are not
            # the immutable zero-field PBE source.
            (response_restart / name).write_text(f"response rewrite {name}\n", encoding="ascii")
        files = {
            name: {"sha256": sha256(self.case / name), "size": (self.case / name).stat().st_size}
            for name in assets
        }
        restart_files = {
            name: {"sha256": sha256(source_restart / name), "size": (source_restart / name).stat().st_size}
            for name in RESTART_NAMES
        }
        (self.response / "PREPARATION_MANIFEST.json").write_text(
            json.dumps(
                {
                    "status": "prepared",
                    "branches": {
                        "fixed": {
                            "source_phase": "runs/fixed/fixed_zero_restart",
                            "source_phase_absolute": str(self.source_phase),
                            "files": files,
                            "restart_files": restart_files,
                        }
                    },
                }
            ),
            encoding="ascii",
        )
        (self.response / "DELTA_RESPONSE_GATE_RESULT.txt").write_text(
            "status=DELTA_RESPONSE_GATE_PASSED\nblocked_on=None\n",
            encoding="ascii",
        )
        self.frequency = self.base / "nfreq16.dat"
        self.frequency.write_text(
            "# index omega_Ha weight_Ha\n"
            + "\n".join(
                f"{index} {index * 0.01:.16e} {index * 0.02:.16e}"
                for index in range(1, 17)
            )
            + "\n",
            encoding="ascii",
        )
        self.abfs = self.base / "C_10au_3s3p2d1f1g_pca1e-4.abfs"
        self.abfs.write_text("ABFS\n", encoding="ascii")

    def tearDown(self):
        self.temporary.cleanup()

    def test_stages_one_immutable_fixed_zero_field_target(self):
        root = self.base / "siab"
        manifest = prepare_siab_reference(
            root, self.response, self.frequency, self.abfs
        )

        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["source_branch"], "fixed")
        self.assertEqual(
            manifest["source_phase"], "runs/fixed/fixed_zero_restart"
        )
        self.assertEqual(manifest["frequency_count"], 16)
        self.assertEqual(manifest["abfs"]["sha256"], sha256(self.abfs))
        self.assertEqual(manifest["response_gate_status"], "DELTA_RESPONSE_GATE_PASSED")

        for name in (
            "INPUT",
            "STRU",
            "KPT",
            "C_ONCV_PBE-1.0.upf",
            "C_gga_10au_100Ry_3s3p2d.orb",
            "SOURCE_EIG_OCC.txt",
            "fixed_frequency_grid_nfreq16.dat",
            self.abfs.name,
            "PREPARATION_MANIFEST.json",
        ):
            path = root / name
            self.assertTrue(path.is_file(), name)
            self.assertFalse(path.is_symlink(), name)
        for name in RESTART_NAMES:
            staged = root / "OUT.C_SIAB_REFERENCE" / name
            source = self.source_phase / "OUT.C_PBE_REFERENCE_GATE" / name
            self.assertEqual(staged.read_bytes(), source.read_bytes())

    def test_refuses_unpassed_response_or_wrong_frequency_count(self):
        result = self.response / "DELTA_RESPONSE_GATE_RESULT.txt"
        result.write_text("status=DELTA_RESPONSE_GATE_FAILED\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "response gate has not passed"):
            prepare_siab_reference(
                self.base / "failed", self.response, self.frequency, self.abfs
            )

        result.write_text(
            "status=DELTA_RESPONSE_GATE_PASSED\nblocked_on=None\n",
            encoding="ascii",
        )
        lines = self.frequency.read_text(encoding="ascii").splitlines()
        self.frequency.write_text("\n".join(lines[:-1]) + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "expected 16 frequency rows"):
            prepare_siab_reference(
                self.base / "nfreq15", self.response, self.frequency, self.abfs
            )

    def test_refuses_tampered_source_and_existing_target(self):
        (self.source_phase / "STRU").write_text("tampered\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            prepare_siab_reference(
                self.base / "tampered", self.response, self.frequency, self.abfs
            )

        root = self.base / "exists"
        root.mkdir()
        with self.assertRaisesRegex(FileExistsError, "target already exists"):
            prepare_siab_reference(root, self.response, self.frequency, self.abfs)


if __name__ == "__main__":
    unittest.main()
