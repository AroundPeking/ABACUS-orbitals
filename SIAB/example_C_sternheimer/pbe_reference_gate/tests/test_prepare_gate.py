import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MODULE_ROOT))

import prepare_gate
from prepare_gate import BRANCHES, prepare_branch


EXPECTED_BRANCHES = {
    "fixed": ("fixed_field", 0),
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

    def assert_control_template_tamper_rejected(
        self, name, original_token, replacement_token
    ):
        root = self.base / f"template-tamper-{name.lower()}"
        prepared = prepare_branch(
            root,
            branch="fixed",
            pseudo=self.pseudo,
            orbital=self.orbital,
        )
        staged = prepared / "fixed_field_seed" / name
        content = staged.read_bytes()
        self.assertIn(original_token, content)
        staged.write_bytes(content.replace(original_token, replacement_token))

        provenance_path = prepared / "BRANCH_PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text())
        record = provenance["phase"]["files"][name]
        record["sha256"] = sha256(staged)
        record["size"] = staged.stat().st_size
        provenance_path.write_text(json.dumps(provenance))

        with self.assertRaisesRegex(ValueError, f"{name}.*template"):
            prepare_branch(
                root,
                branch="dir0",
                pseudo=self.pseudo,
                orbital=self.orbital,
            )
        self.assertFalse(os.path.lexists(root / "runs" / "dir0"))

    def test_branch_mapping_is_exact(self):
        self.assertEqual(BRANCHES, EXPECTED_BRANCHES)

    def test_prepares_only_the_initial_phase_for_every_branch(self):
        for branch, (mode, field_dir) in EXPECTED_BRANCHES.items():
            with self.subTest(branch=branch):
                branch_root = self.prepare(branch)
                expected_phase = (
                    "fixed_field_seed" if branch == "fixed" else "field_seed"
                )

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
                self.assertEqual(
                    input_text,
                    prepare_gate.render_input(
                        mode=mode,
                        field_dir=field_dir,
                        restart=False,
                    ),
                )
                self.assertIn(
                    f"ocp {'1' if mode == 'fixed_field' else '0'}\n",
                    input_text,
                )
                if mode in {"fixed_field", "field"}:
                    self.assertIn(f"efield_dir {field_dir}\n", input_text)
                else:
                    self.assertNotIn("efield_dir", input_text)

    def test_stru_is_exact_twenty_angstrom_centered_carbon_cell(self):
        phase = self.prepare() / "fixed_field_seed"
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
            "ATOMIC_POSITIONS\nDirect\n\nC\n0.0\n1\n" "0.5 0.5 0.5 0 0 0 mag 2.0\n",
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
        phase = self.prepare() / "fixed_field_seed"
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
        self.assertEqual(provenance["sources"]["pseudo"]["sha256"], sha256(self.pseudo))
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
                self.assertEqual(
                    record["relative_path"], f"runs/dir2/field_seed/{name}"
                )
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
                self.assertFalse(os.path.lexists(self.root / "runs" / "fixed"))

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

    def test_rejects_whitespace_and_control_characters_in_asset_basenames(self):
        for unsafe_name in (
            "C pseudo.upf",
            "C\tpseudo.upf",
            "C\npseudo.upf",
            "C\x01pseudo.upf",
        ):
            with self.subTest(unsafe_name=repr(unsafe_name)):
                unsafe = self.assets / unsafe_name
                unsafe.write_text("asset")
                with self.assertRaisesRegex(ValueError, "single safe token"):
                    prepare_branch(
                        self.root,
                        branch="fixed",
                        pseudo=unsafe,
                        orbital=self.orbital,
                    )

    def test_changed_pseudo_content_is_rejected_across_branches(self):
        self.prepare("fixed")
        self.pseudo.write_bytes(b"changed-pseudo\n")

        with self.assertRaisesRegex(ValueError, "pseudo.*does not match"):
            self.prepare("dir0")

        self.assertFalse(os.path.lexists(self.root / "runs" / "dir0"))

    def test_changed_orbital_content_is_rejected_across_branches(self):
        self.prepare("fixed")
        self.orbital.write_bytes(b"changed-orbital\n")

        with self.assertRaisesRegex(ValueError, "orbital.*does not match"):
            self.prepare("dir1")

        self.assertFalse(os.path.lexists(self.root / "runs" / "dir1"))

    def test_same_assets_from_different_absolute_paths_are_accepted(self):
        self.prepare("fixed")
        alternate = self.base / "alternate-assets"
        alternate.mkdir()
        alternate_pseudo = alternate / self.pseudo.name
        alternate_orbital = alternate / self.orbital.name
        alternate_pseudo.write_bytes(self.pseudo.read_bytes())
        alternate_orbital.write_bytes(self.orbital.read_bytes())

        prepared = prepare_branch(
            self.root,
            branch="dir2",
            pseudo=alternate_pseudo,
            orbital=alternate_orbital,
        )

        provenance = json.loads((prepared / "BRANCH_PROVENANCE.json").read_text())
        self.assertEqual(
            provenance["sources"]["pseudo"]["absolute_path"],
            str(alternate_pseudo.resolve()),
        )

    def test_missing_corrupt_or_incomplete_existing_provenance_is_rejected(self):
        mutations = (
            lambda path: path.unlink(),
            lambda path: path.write_text("not-json"),
            lambda path: path.write_text(json.dumps({"schema": "wrong"})),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                root = self.base / f"gate-{index}"
                prepare_branch(
                    root,
                    branch="fixed",
                    pseudo=self.pseudo,
                    orbital=self.orbital,
                )
                mutate(root / "runs" / "fixed" / "BRANCH_PROVENANCE.json")

                with self.assertRaisesRegex(ValueError, "provenance"):
                    prepare_branch(
                        root,
                        branch="dir0",
                        pseudo=self.pseudo,
                        orbital=self.orbital,
                    )
                self.assertFalse(os.path.lexists(root / "runs" / "dir0"))

    def test_frozen_protocol_mismatch_in_existing_provenance_is_rejected(self):
        self.prepare("fixed")
        provenance_path = self.root / "runs" / "fixed" / "BRANCH_PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["frozen_protocol"]["nx"] = "999"
        provenance_path.write_text(json.dumps(provenance))

        with self.assertRaisesRegex(ValueError, "frozen_protocol"):
            self.prepare("dir0")

    def test_existing_input_is_bound_to_renderer_template(self):
        self.assert_control_template_tamper_rejected(
            "INPUT", b"ecutwfc 30\n", b"ecutwfc 31\n"
        )

    def test_existing_stru_is_bound_to_twenty_angstrom_template(self):
        self.assert_control_template_tamper_rejected(
            "STRU",
            b"37.79452249150619\n",
            b"18.897261245753095\n",
        )

    def test_existing_kpt_is_bound_to_gamma_template(self):
        self.assert_control_template_tamper_rejected(
            "KPT", b"1 1 1 0 0 0\n", b"2 1 1 0 0 0\n"
        )

    def test_staged_asset_and_phase_record_cannot_diverge_from_source_record(self):
        for label, source in (("pseudo", self.pseudo), ("orbital", self.orbital)):
            with self.subTest(label=label):
                root = self.base / f"phase-tamper-{label}"
                prepared = prepare_branch(
                    root,
                    branch="fixed",
                    pseudo=self.pseudo,
                    orbital=self.orbital,
                )
                provenance_path = prepared / "BRANCH_PROVENANCE.json"
                provenance = json.loads(provenance_path.read_text())
                staged = prepared / "fixed_field_seed" / source.name
                staged.write_bytes(f"tampered-{label}\n".encode())
                phase_record = provenance["phase"]["files"][source.name]
                phase_record["sha256"] = sha256(staged)
                phase_record["size"] = staged.stat().st_size
                provenance_path.write_text(json.dumps(provenance))

                with self.assertRaisesRegex(ValueError, f"{label}.*provenance chain"):
                    prepare_branch(
                        root,
                        branch="dir0",
                        pseudo=self.pseudo,
                        orbital=self.orbital,
                    )
                self.assertFalse(os.path.lexists(root / "runs" / "dir0"))

    def test_source_record_cannot_diverge_from_phase_asset_record(self):
        for label, source in (("pseudo", self.pseudo), ("orbital", self.orbital)):
            with self.subTest(label=label):
                root = self.base / f"source-tamper-{label}"
                prepared = prepare_branch(
                    root,
                    branch="fixed",
                    pseudo=self.pseudo,
                    orbital=self.orbital,
                )
                alternate_dir = self.base / f"alternate-{label}"
                alternate_dir.mkdir()
                alternate = alternate_dir / source.name
                alternate.write_bytes(f"alternate-{label}\n".encode())
                provenance_path = prepared / "BRANCH_PROVENANCE.json"
                provenance = json.loads(provenance_path.read_text())
                source_record = provenance["sources"][label]
                source_record["sha256"] = sha256(alternate)
                source_record["size"] = alternate.stat().st_size
                provenance_path.write_text(json.dumps(provenance))

                pseudo = alternate if label == "pseudo" else self.pseudo
                orbital = alternate if label == "orbital" else self.orbital
                with self.assertRaisesRegex(ValueError, f"{label}.*provenance chain"):
                    prepare_branch(
                        root,
                        branch="dir0",
                        pseudo=pseudo,
                        orbital=orbital,
                    )
                self.assertFalse(os.path.lexists(root / "runs" / "dir0"))

    def test_source_replaced_after_read_is_rejected_and_records_no_branch(self):
        replacement = self.assets / "replacement.upf"
        replacement.write_bytes(b"replacement-content\n")

        def replace_after_read(path, label):
            if label == "pseudo":
                os.replace(replacement, path)

        with mock.patch.object(
            prepare_gate,
            "_after_source_read",
            side_effect=replace_after_read,
        ):
            with self.assertRaisesRegex(ValueError, "changed while being read"):
                self.prepare("fixed")

        self.assertFalse(os.path.lexists(self.root / "runs" / "fixed"))

    def test_provenance_path_and_identity_describe_the_read_source(self):
        real_assets = self.base / "real-assets"
        real_assets.mkdir()
        real_pseudo = real_assets / self.pseudo.name
        real_orbital = real_assets / self.orbital.name
        real_pseudo.write_bytes(self.pseudo.read_bytes())
        real_orbital.write_bytes(self.orbital.read_bytes())
        alias = self.base / "asset-alias"
        alias.symlink_to(real_assets, target_is_directory=True)

        prepared = prepare_branch(
            self.root,
            branch="fixed",
            pseudo=alias / real_pseudo.name,
            orbital=alias / real_orbital.name,
        )
        provenance = json.loads((prepared / "BRANCH_PROVENANCE.json").read_text())

        pseudo_record = provenance["sources"]["pseudo"]
        self.assertEqual(pseudo_record["absolute_path"], str(real_pseudo.resolve()))
        self.assertEqual(pseudo_record["device"], real_pseudo.stat().st_dev)
        self.assertEqual(pseudo_record["inode"], real_pseudo.stat().st_ino)

    def test_exception_cleans_hidden_partial_and_locks_not_formal_branch(self):
        runs = self.root / "runs"
        with mock.patch.object(
            prepare_gate,
            "_before_publish",
            side_effect=OSError("injected pre-publish failure"),
        ):
            with self.assertRaisesRegex(OSError, "pre-publish failure"):
                self.prepare("dir1")

        self.assertFalse(os.path.lexists(runs / "dir1"))
        self.assertEqual(list(runs.iterdir()), [])

    def test_temporary_open_failure_removes_new_directory_and_all_locks(self):
        original_open = prepare_gate._open_directory_at

        def fail_temporary_open(parent_fd, name, label):
            if label == "temporary preparation":
                raise OSError("injected temporary open failure")
            return original_open(parent_fd, name, label)

        with mock.patch.object(
            prepare_gate,
            "_open_directory_at",
            side_effect=fail_temporary_open,
        ):
            with self.assertRaisesRegex(OSError, "temporary open failure"):
                self.prepare("fixed")

        runs = self.root / "runs"
        self.assertEqual(list(runs.iterdir()), [])
        self.assertEqual(self.prepare("fixed"), runs / "fixed")

    def test_stale_hidden_preparation_is_reported_not_published(self):
        runs = self.root / "runs"
        runs.mkdir(parents=True)
        (runs / ".dir1.prepare-deadbeef").mkdir()
        (runs / ".dir1.prepare.lock").write_text("stale")

        with self.assertRaisesRegex(RuntimeError, "stale preparation"):
            self.prepare("dir1")

        self.assertFalse(os.path.lexists(runs / "dir1"))
        self.assertTrue((runs / ".dir1.prepare-deadbeef").is_dir())

    def test_stale_lock_from_another_branch_blocks_preparation(self):
        runs = self.root / "runs"
        runs.mkdir(parents=True)
        (runs / ".dir0.prepare.lock").write_text("stale")

        with self.assertRaisesRegex(RuntimeError, "stale preparation lock"):
            self.prepare("dir2")

        self.assertFalse(os.path.lexists(runs / "dir2"))

    def test_two_concurrent_preparations_do_not_overwrite(self):
        lock_acquired = threading.Event()
        release = threading.Event()

        def hold_first(branch):
            lock_acquired.set()
            self.assertTrue(release.wait(timeout=5))

        with mock.patch.object(
            prepare_gate,
            "_after_preparation_lock_acquired",
            side_effect=hold_first,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(self.prepare, "fixed")
                self.assertTrue(lock_acquired.wait(timeout=5))
                second = executor.submit(self.prepare, "dir0")
                with self.assertRaisesRegex(RuntimeError, "preparation lock"):
                    second.result(timeout=5)
                release.set()
                self.assertEqual(first.result(timeout=5), self.root / "runs" / "fixed")

        self.assertTrue((self.root / "runs" / "fixed").is_dir())
        self.assertFalse(os.path.lexists(self.root / "runs" / "dir0"))

    def test_concurrent_formal_branch_created_before_publish_is_not_overwritten(self):
        marker_text = "external branch"

        def occupy_target(runs_path, branch):
            target = runs_path / branch
            target.mkdir()
            (target / "marker").write_text(marker_text)

        with mock.patch.object(
            prepare_gate,
            "_before_publish",
            side_effect=occupy_target,
        ):
            with self.assertRaises(FileExistsError):
                self.prepare("dir2")

        target = self.root / "runs" / "dir2"
        self.assertEqual((target / "marker").read_text(), marker_text)

    def test_cleanup_does_not_delete_external_replacement_after_publish(self):
        target = self.root / "runs" / "dir2"
        displaced = self.root / "runs" / ".displaced-owned-branch"

        def replace_published(_runs_fd, _branch, _staged_fd):
            target.rename(displaced)
            target.mkdir()
            (target / "marker").write_text("external replacement")
            raise RuntimeError("published identity replaced")

        with mock.patch.object(
            prepare_gate,
            "_verify_published_identity",
            side_effect=replace_published,
        ):
            with self.assertRaisesRegex(RuntimeError, "identity replaced"):
                self.prepare("dir2")

        self.assertEqual((target / "marker").read_text(), "external replacement")

    def test_root_replaced_by_symlink_after_open_is_rejected(self):
        outside = self.base / "outside-root"
        outside.mkdir()
        moved = self.base / "original-root"

        def replace_root(root_path, _runs_path):
            root_path.rename(moved)
            root_path.symlink_to(outside, target_is_directory=True)

        with mock.patch.object(
            prepare_gate,
            "_after_directory_fds_opened",
            side_effect=replace_root,
        ):
            with self.assertRaisesRegex(ValueError, "root directory identity"):
                self.prepare("fixed")

        self.assertEqual(list(outside.iterdir()), [])

    def test_runs_replaced_by_symlink_after_open_is_rejected(self):
        outside = self.base / "outside-runs"
        outside.mkdir()
        moved = self.root / "original-runs"

        def replace_runs(_root_path, runs_path):
            runs_path.rename(moved)
            runs_path.symlink_to(outside, target_is_directory=True)

        with mock.patch.object(
            prepare_gate,
            "_after_directory_fds_opened",
            side_effect=replace_runs,
        ):
            with self.assertRaisesRegex(ValueError, "runs directory identity"):
                self.prepare("fixed")

        self.assertEqual(list(outside.iterdir()), [])

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
        self.assertTrue(
            (self.root / "runs" / "dir0" / "field_seed" / "INPUT").is_file()
        )

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
        self.assertTrue(
            (
                self.root
                / "runs"
                / "fixed"
                / "fixed_field_seed"
                / "STRU"
            ).is_file()
        )

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
        result = self.run_direct("render", "--mode", "free", "--field-dir", "1")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("free mode requires restart=True", result.stderr)


if __name__ == "__main__":
    unittest.main()
