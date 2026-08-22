import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_response_gate import prepare_response_gate


RESTART_NAMES = ("wfs1_nao.txt", "wfs2_nao.txt", "chgs1.cube", "chgs2.cube")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eig_occ_text():
    return """1 # ionic step
 Electronic state energy (eV) and occupations
 Spin number 2
 spin=1 k-point=1/1 Cartesian=0 0 0 (1 plane wave)
 1 -10.0 1.0
 2 -4.0 1.0
 3 -3.0 1.0
 4 1.0 0.0

 spin=2 k-point=1/1 Cartesian=0 0 0 (1 plane wave)
 1 -8.0 1.0
 2 -2.0 0.0
"""


class PrepareResponseGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.pbe = self.base / "pbe"
        phases = {
            "runs/fixed/fixed_zero_restart": "fixed",
            "runs/dir0/free_restart2": "free",
        }
        result_phases = {}
        for relative, label in phases.items():
            phase = self.pbe / relative
            output = phase / "OUT.C_PBE_REFERENCE_GATE"
            output.mkdir(parents=True)
            (phase / "STRU").write_text("STRU\n", encoding="ascii")
            (phase / "KPT").write_text("KPT\n", encoding="ascii")
            (phase / "C_ONCV_PBE-1.0.upf").write_text("pseudo\n", encoding="ascii")
            (phase / "C_gga_10au_100Ry_3s3p2d.orb").write_text("orbital\n", encoding="ascii")
            for name in RESTART_NAMES:
                (output / name).write_text(f"{label}-{name}\n", encoding="ascii")
            eig = output / "eig_occ.txt"
            eig.write_text(eig_occ_text(), encoding="ascii")
            result_phases[relative] = {
                "file_sha256": {"eig_occ.txt": sha256(eig)},
                "integer_occupations": True,
            }
        (self.pbe / "RESULT_SUMMARY.json").write_text(
            json.dumps(
                {
                    "status": "PBE_GATE_PASSED",
                    "zero_field_comparison_status": "ZERO_FIELD_COMPARISON_PASSED",
                    "blocked_on": None,
                    "phases": result_phases,
                }
            ),
            encoding="ascii",
        )
        self.frequency = self.base / "fixed_frequency_grid.dat"
        self.frequency.write_text(
            "# index omega_Ha weight_Ha\n"
            + "\n".join(f"{index} {0.1 * index:.16e} {0.2 * index:.16e}" for index in range(1, 7))
            + "\n",
            encoding="ascii",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_prepares_two_real_branch_copies_with_exact_restart_sources(self):
        root = self.base / "response"

        manifest = prepare_response_gate(root, self.pbe, self.frequency)

        self.assertEqual(set(manifest["branches"]), {"fixed", "free"})
        for branch in ("fixed", "free"):
            case = root / "branches" / branch
            restart_dir = case / "OUT.C_DELTA_RESPONSE_GATE"
            self.assertTrue(case.is_dir())
            self.assertFalse(case.is_symlink())
            for name in (
                "INPUT",
                "STRU",
                "KPT",
                "C_ONCV_PBE-1.0.upf",
                "C_gga_10au_100Ry_3s3p2d.orb",
                "fixed_frequency_grid.dat",
                "SOURCE_EIG_OCC.txt",
            ):
                path = case / name
                self.assertTrue(path.is_file(), name)
                self.assertFalse(path.is_symlink(), name)
            self.assertTrue(restart_dir.is_dir())
            self.assertFalse(restart_dir.is_symlink())
            for name in RESTART_NAMES:
                path = restart_dir / name
                self.assertTrue(path.is_file(), name)
                self.assertFalse(path.is_symlink(), name)
            input_text = (case / "INPUT").read_text(encoding="ascii")
            self.assertIn(f"ocp {'1' if branch == 'fixed' else '0'}\n", input_text)
            self.assertIn("sternheimer_fd_order 8\n", input_text)
            self.assertIn("exx_pca_threshold 1e-4\n", input_text)
            self.assertIn("sternheimer_frequency_grid_file fixed_frequency_grid.dat\n", input_text)
            source_phase = self.pbe / manifest["branches"][branch]["source_phase"]
            output = source_phase / "OUT.C_PBE_REFERENCE_GATE"
            for name in RESTART_NAMES:
                self.assertEqual((restart_dir / name).read_bytes(), (output / name).read_bytes())

    def test_common_physical_assets_and_frequency_are_byte_identical(self):
        root = self.base / "response"
        manifest = prepare_response_gate(root, self.pbe, self.frequency)

        for name in (
            "STRU",
            "KPT",
            "C_ONCV_PBE-1.0.upf",
            "C_gga_10au_100Ry_3s3p2d.orb",
            "fixed_frequency_grid.dat",
        ):
            fixed = root / "branches/fixed" / name
            free = root / "branches/free" / name
            self.assertEqual(sha256(fixed), sha256(free))
            self.assertEqual(manifest["common_files"][name]["sha256"], sha256(fixed))

    def test_refuses_to_overwrite_existing_root(self):
        root = self.base / "response"
        root.mkdir()

        with self.assertRaisesRegex(FileExistsError, "response gate root already exists"):
            prepare_response_gate(root, self.pbe, self.frequency)

    def test_rejects_unpassed_or_tampered_pbe_source(self):
        result_path = self.pbe / "RESULT_SUMMARY.json"
        result = json.loads(result_path.read_text(encoding="ascii"))
        result["status"] = "PBE_GATE_FAILED"
        result_path.write_text(json.dumps(result), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "PBE gate has not passed"):
            prepare_response_gate(self.base / "failed", self.pbe, self.frequency)

        result["status"] = "PBE_GATE_PASSED"
        result_path.write_text(json.dumps(result), encoding="ascii")
        eig = self.pbe / "runs/fixed/fixed_zero_restart/OUT.C_PBE_REFERENCE_GATE/eig_occ.txt"
        eig.write_text(eig.read_text(encoding="ascii") + "# tamper\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "eig_occ hash mismatch"):
            prepare_response_gate(self.base / "tampered", self.pbe, self.frequency)


if __name__ == "__main__":
    unittest.main()
