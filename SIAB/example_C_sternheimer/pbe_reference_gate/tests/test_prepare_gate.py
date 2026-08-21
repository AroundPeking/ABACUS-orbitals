import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MODULE_ROOT))

import prepare_gate
from prepare_gate import BRANCHES, prepare_branch


EXPECTED_BRANCHES = {
    "fixed": ("fixed", None),
    "dir0": ("field", 0),
    "dir1": ("field", 1),
    "dir2": ("field", 2),
}
EXPECTED_KPT = "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "gate"
        self.assets = self.base / "assets"
        self.assets.mkdir()
        self.pseudo = self.assets / "C_ONCV_PBE-1.0.upf"
        self.orbital = self.assets / "C_gga_10au_100Ry_3s3p2d.orb"
        self.pseudo.write_bytes(b"pseudo-content\n")
        self.orbital.write_bytes(b"orbital-content\n")

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, branch="fixed"):
        return prepare_branch(
            self.root,
            branch=branch,
            pseudo=self.pseudo,
            orbital=self.orbital,
        )

    def test_branch_mapping_is_exact(self):
        self.assertEqual(BRANCHES, EXPECTED_BRANCHES)

    def test_prepares_only_the_initial_phase_for_every_branch(self):
        for branch, (mode, field_dir) in EXPECTED_BRANCHES.items():
            with self.subTest(branch=branch):
                branch_root = self.prepare(branch)
                expected_phase = "fixed_cold" if branch == "fixed" else "field_seed"

                self.assertEqual(branch_root, self.root / "runs" / branch)
                self.assertEqual(
                    {path.name for path in branch_root.iterdir()},
                    {expected_phase, "BRANCH_PROVENANCE.json"},
                )
                phase = branch_root / expected_phase
                self.assertEqual(
                    {path.name for path in phase.iterdir()},
                    {
                        "INPUT",
                        "STRU",
                        "KPT",
                        self.pseudo.name,
                        self.orbital.name,
                    },
                )
                input_text = (phase / "INPUT").read_text()
                self.assertIn(f"ocp {'1' if mode == 'fixed' else '0'}\n", input_text)
                if mode == "field":
                    self.assertIn(f"efield_dir {field_dir}\n", input_text)
                else:
                    self.assertNotIn("efield_dir", input_text)

    def test_stru_is_exact_twenty_angstrom_centered_carbon_cell(self):
        phase = self.prepare() / "fixed_cold"
        text = (phase / "STRU").read_text()

        self.assertIn(
            f"ATOMIC_SPECIES\nC 12.011 {self.pseudo.name}\n",
            text,
        )
        self.assertIn(
            "LATTICE_CONSTANT\n37.79452249150619\n\n"
            "LATTICE_VECTORS\n"
            "1 0 0\n0 1 0\n0 0 1\n",
            text,
        )
        self.assertIn(
            "ATOMIC_POSITIONS\nDirect\n\nC\n0.0\n1\n"
            "0.5 0.5 0.5 0 0 0 mag 2.0\n",
            text,
        )
        self.assertIn(
            f"NUMERICAL_ORBITAL\n{self.orbital.name}\n",
            text,
        )

    def test_kpt_is_explicit_gamma_only(self):
        phase = self.prepare("dir1") / "field_seed"
        self.assertEqual((phase / "KPT").read_text(), EXPECTED_KPT)

    def test_assets_are_real_copies_and_survive_source_removal(self):
        phase = self.prepare() / "fixed_cold"
        pseudo_copy = phase / self.pseudo.name
        orbital_copy = phase / self.orbital.name

        self.assertTrue(pseudo_copy.is_file())
        self.assertTrue(orbital_copy.is_file())
        self.assertFalse(pseudo_copy.is_symlink())
        self.assertFalse(orbital_copy.is_symlink())
        self.pseudo.unlink()
        self.orbital.unlink()
        self.assertEqual(pseudo_copy.read_bytes(), b"pseudo-content\n")
        self.assertEqual(orbital_copy.read_bytes(), b"orbital-content\n")

    def test_provenance_records_actual_sources_protocol_hashes_and_sizes(self):
        branch_root = self.prepare("dir2")
        provenance_path = branch_root / "BRANCH_PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text())
        phase = branch_root / "field_seed"

        self.assertEqual(provenance["schema"], "c-pbe-reference-gate-branch")
        self.assertEqual(provenance["version"], 1)
        self.assertEqual(provenance["branch"], "dir2")
        self.assertEqual(provenance["mode"], "field")
        self.assertEqual(provenance["field_dir"], 2)
        self.assertEqual(provenance["box_angstrom"], 20.0)
        self.assertEqual(provenance["atom_direct"], [0.5, 0.5, 0.5])
        self.assertEqual(
            provenance["sources"]["pseudo"]["absolute_path"],
            str(self.pseudo.resolve()),
        )
        self.assertEqual(
            provenance["sources"]["orbital"]["absolute_path"],
            str(self.orbital.resolve()),
        )
        self.assertEqual(
            provenance["sources"]["pseudo"]["sha256"], sha256(self.pseudo)
        )
        self.assertEqual(
            provenance["sources"]["orbital"]["sha256"], sha256(self.orbital)
        )
        self.assertEqual(
            provenance["renderer"],
            {
                "function": "gate_contract.render_input",
                "mode": "field",
                "field_dir": 2,
                "restart": False,
            },
        )
        self.assertEqual(provenance["frozen_protocol"]["nx"], "135")
        self.assertEqual(provenance["phase"]["relative_path"], "runs/dir2/field_seed")

        expected_files = {
            "INPUT",
            "STRU",
            "KPT",
            self.pseudo.name,
            self.orbital.name,
        }
        self.assertEqual(set(provenance["phase"]["files"]), expected_files)
        for name in expected_files:
            with self.subTest(name=name):
                file_path = phase / name
                record = provenance["phase"]["files"][name]
                self.assertEqual(record["relative_path"], f"runs/dir2/field_seed/{name}")
                self.assertEqual(record["sha256"], sha256(file_path))
                self.assertEqual(record["size"], file_path.stat().st_size)

        serialized = provenance_path.read_text()
        for forbidden in ("git", "commit", "server", "hostname", "abacus_binary"):
            self.assertNotIn(f'"{forbidden}"', serialized.lower())

    def test_preexisting_branch_directory_is_never_modified(self):
        target = self.root / "runs" / "fixed"
        target.mkdir(parents=True)
        marker = target / "keep"
        marker.write_text("old data")

        with self.assertRaises(FileExistsError):
            self.prepare()

        self.assertEqual(marker.read_text(), "old data")

    def test_preexisting_branch_symlink_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        runs = self.root / "runs"
        runs.mkdir(parents=True)
        (runs / "fixed").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(FileExistsError):
            self.prepare()

        self.assertEqual(list(outside.iterdir()), [])

    def test_repeated_prepare_is_rejected(self):
        self.prepare("dir0")
        with self.assertRaises(FileExistsError):
            self.prepare("dir0")

    def test_rejects_symlink_directory_and_missing_sources(self):
        symlink = self.assets / "pseudo-link.upf"
        symlink.symlink_to(self.pseudo)
        directory = self.assets / "orbital-directory"
        directory.mkdir()
        missing = self.assets / "missing.orb"

        cases = (
            (symlink, self.orbital),
            (self.pseudo, symlink),
            (directory, self.orbital),
            (self.pseudo, directory),
            (missing, self.orbital),
            (self.pseudo, missing),
        )
        for pseudo, orbital in cases:
            with self.subTest(pseudo=pseudo, orbital=orbital):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    prepare_branch(
                        self.root,
                        branch="fixed",
                        pseudo=pseudo,
                        orbital=orbital,
                    )
                self.assertFalse(
                    os.path.lexists(self.root / "runs" / "fixed")
                )

    def test_rejects_asset_basename_collision(self):
        second_directory = self.base / "other"
        second_directory.mkdir()
        same_name = second_directory / self.pseudo.name
        same_name.write_text("orbital")

        with self.assertRaisesRegex(ValueError, "basenames must be distinct"):
            prepare_branch(
                self.root,
                branch="fixed",
                pseudo=self.pseudo,
                orbital=same_name,
            )

    def test_rejects_reserved_asset_basenames(self):
        for reserved in ("INPUT", "STRU", "KPT", "BRANCH_PROVENANCE.json"):
            with self.subTest(reserved=reserved):
                bad = self.assets / reserved
                bad.write_text("asset")
                with self.assertRaisesRegex(ValueError, "reserved"):
                    prepare_branch(
                        self.root,
                        branch="fixed",
                        pseudo=bad,
                        orbital=self.orbital,
                    )

    def test_failure_after_exclusive_mkdir_removes_partial_branch(self):
        target = self.root / "runs" / "dir1"
        with mock.patch.object(
            prepare_gate,
            "_write_text_file",
            side_effect=OSError("injected write failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected write failure"):
                self.prepare("dir1")

        self.assertFalse(os.path.lexists(target))

    def test_rejects_unknown_branch_before_writing(self):
        with self.assertRaisesRegex(ValueError, "unsupported branch"):
            self.prepare("other")
        self.assertFalse((self.root / "runs").exists())


class PrepareGateCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "gate"
        self.pseudo = self.base / "C.upf"
        self.orbital = self.base / "C.orb"
        self.pseudo.write_text("pseudo")
        self.orbital.write_text("orbital")
        self.script = MODULE_ROOT / "prepare_gate.py"

    def tearDown(self):
        self.temporary.cleanup()

    def run_direct(self, *arguments):
        return subprocess.run(
            [sys.executable, str(self.script), *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def run_module(self, *arguments):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "SIAB.example_C_sternheimer.pbe_reference_gate.prepare_gate",
                *arguments,
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_direct_cli_prepare(self):
        result = self.run_direct(
            "prepare",
            "--root",
            str(self.root),
            "--branch",
            "dir0",
            "--pseudo",
            str(self.pseudo),
            "--orbital",
            str(self.orbital),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.root / "runs" / "dir0"))
        self.assertTrue((self.root / "runs" / "dir0" / "field_seed" / "INPUT").is_file())

    def test_module_cli_prepare(self):
        result = self.run_module(
            "prepare",
            "--root",
            str(self.root),
            "--branch",
            "fixed",
            "--pseudo",
            str(self.pseudo),
            "--orbital",
            str(self.orbital),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "runs" / "fixed" / "fixed_cold" / "STRU").is_file())

    def test_direct_cli_renders_fixed_restart(self):
        result = self.run_direct("render", "--mode", "fixed", "--restart")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ocp 1\n", result.stdout)
        self.assertIn("init_wfc file\n", result.stdout)
        self.assertIn("init_chg file\n", result.stdout)

    def test_module_cli_renders_free_restart(self):
        result = self.run_module(
            "render",
            "--mode",
            "free",
            "--field-dir",
            "2",
            "--restart",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ocp 0\n", result.stdout)
        self.assertIn("efield_flag 0\n", result.stdout)
        self.assertNotIn("efield_dir", result.stdout)

    def test_cli_invalid_renderer_combination_is_nonzero(self):
        result = self.run_direct(
            "render", "--mode", "free", "--field-dir", "1"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("free mode requires restart=True", result.stderr)


if __name__ == "__main__":
    unittest.main()
