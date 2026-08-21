import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gate_contract import (
    DRIFT_TOL_KCAL,
    ENERGY_TOL_HA,
    HA_TO_KCAL_MOL,
    INTEGER_TOL,
    audit_phase,
    compare_zero_field_results,
)


ENERGY_EV = -192.190008196889


def input_text(mode):
    ocp = 1 if mode == "fixed" else 0
    lines = [
        "INPUT_PARAMETERS",
        "nspin 2",
        "nelec 4",
        "nupdown 2",
        "gamma_only 1",
        f"ocp {ocp}",
        "efield_flag 0",
        "efield_amp 0",
    ]
    if mode == "fixed":
        lines.append("ocp_set 3*1 19*0 1*1 21*0")
    return "\n".join(lines) + "\n"


def log_text(energy_ev=ENERGY_EV, *, converged=True, extra_energy=False):
    lines = ["ABACUS running log"]
    if converged:
        lines.append(" #SCF IS CONVERGED#")
    lines.append(f" !FINAL_ETOT_IS {energy_ev} eV")
    if extra_energy:
        lines.append(f" !FINAL_ETOT_IS {energy_ev + 0.1} eV")
    return "\n".join(lines) + "\n"


def eig_occ_text(
    spin1=(1.0, 1.0, 1.0, 0.0),
    spin2=(1.0, 0.0, 0.0, 0.0),
    *,
    ionic_steps=1,
    spin1_kpoint="1/1",
):
    def rows(occupations, offset):
        return [
            f" {index} {-51.5 + offset + index:.14f} {occupation:.15f}"
            for index, occupation in enumerate(occupations, start=1)
        ]

    blocks = []
    for step in range(1, ionic_steps + 1):
        blocks.extend(
            [
                f"{step}     # ionic step",
                " Electronic state energy (eV) and occupations",
                " Spin number 2",
                (
                    " spin=1 k-point="
                    f"{spin1_kpoint} Cartesian=0.0000000 0.0000000 0.0000000 "
                    "(123 plane wave)"
                ),
                *rows(spin1, 0.0),
                "",
                (
                    " spin=2 k-point=1/1 Cartesian=0.0000000 0.0000000 0.0000000 "
                    "(123 plane wave)"
                ),
                *rows(spin2, 0.2),
                "",
            ]
        )
    return "\n".join(blocks)


def write_phase(
    path,
    *,
    mode="fixed",
    energy_ev=ENERGY_EV,
    converged=True,
    extra_energy=False,
    spin1=(1.0, 1.0, 1.0, 0.0),
    spin2=(1.0, 0.0, 0.0, 0.0),
    ionic_steps=1,
    spin1_kpoint="1/1",
):
    path.mkdir(parents=True)
    (path / "INPUT").write_text(input_text(mode))
    out = path / "OUT.C_PBE_REFERENCE_GATE"
    out.mkdir()
    (out / "running_scf.log").write_text(
        log_text(
            energy_ev,
            converged=converged,
            extra_energy=extra_energy,
        )
    )
    (out / "eig_occ.txt").write_text(
        eig_occ_text(
            spin1,
            spin2,
            ionic_steps=ionic_steps,
            spin1_kpoint=spin1_kpoint,
        )
    )


class PhaseAuditTests(unittest.TestCase):
    def test_constants_are_frozen(self):
        self.assertEqual(HA_TO_KCAL_MOL, 627.5094740631)
        self.assertEqual(INTEGER_TOL, 1e-10)
        self.assertEqual(DRIFT_TOL_KCAL, 0.001)
        self.assertEqual(ENERGY_TOL_HA, 1e-5)

    def test_parses_real_abacus_spin_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "fixed"
            write_phase(case)

            phase = audit_phase(case, expected_mode="fixed")

            self.assertEqual(dict(phase.spin_counts), {1: 3.0, 2: 1.0})
            self.assertTrue(phase.integer_occupations)
            self.assertAlmostEqual(phase.energy_ev, ENERGY_EV)
            self.assertEqual(set(phase.file_hashes), {
                "INPUT",
                "running_scf.log",
                "eig_occ.txt",
            })
            self.assertEqual(len(phase.stage_hash), 64)
            with self.assertRaises(TypeError):
                phase.spin_counts[1] = 2.0

    def test_rejects_fractional_occupation(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "free"
            write_phase(case, mode="free", spin1=(1.0, 1.0, 0.5, 0.0))
            with self.assertRaisesRegex(ValueError, "fractional occupation"):
                audit_phase(case, expected_mode="free")

    def test_rejects_missing_convergence_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "fixed"
            write_phase(case, converged=False)
            with self.assertRaisesRegex(ValueError, "SCF convergence"):
                audit_phase(case, expected_mode="fixed")

    def test_rejects_multiple_final_energies(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "fixed"
            write_phase(case, extra_energy=True)
            with self.assertRaisesRegex(ValueError, "exactly one final total energy"):
                audit_phase(case, expected_mode="fixed")

    def test_rejects_an_extra_malformed_final_energy_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "fixed"
            write_phase(case)
            log_path = case / "OUT.C_PBE_REFERENCE_GATE/running_scf.log"
            log_path.write_text(
                log_path.read_text() + " !FINAL_ETOT_IS malformed output\n"
            )
            with self.assertRaisesRegex(ValueError, "exactly one final total energy"):
                audit_phase(case, expected_mode="fixed")

    def test_rejects_nonfinite_final_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "fixed"
            write_phase(case, energy_ev="nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                audit_phase(case, expected_mode="fixed")

    def test_rejects_wrong_zero_field_or_ocp_contract(self):
        mutations = (
            ("fixed", "efield_flag 0", "efield_flag 1", "zero-field"),
            ("free", "efield_amp 0", "efield_amp 1e-4", "zero-field"),
            ("fixed", "ocp 1", "ocp 0", "ocp=1"),
            ("free", "ocp 0", "ocp 1", "ocp=0"),
        )
        for mode, old, new, message in mutations:
            with self.subTest(mode=mode, mutation=new):
                with tempfile.TemporaryDirectory() as tmp:
                    case = Path(tmp) / mode
                    write_phase(case, mode=mode)
                    input_path = case / "INPUT"
                    input_path.write_text(input_path.read_text().replace(old, new))
                    with self.assertRaisesRegex(ValueError, message):
                        audit_phase(case, expected_mode=mode)

    def test_rejects_wrong_spin_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "free"
            write_phase(case, mode="free", spin1=(1.0, 1.0, 0.0, 0.0))
            with self.assertRaisesRegex(ValueError, "spin electron counts"):
                audit_phase(case, expected_mode="free")

    def test_rejects_multiple_ionic_steps_or_kpoints(self):
        cases = (
            ({"ionic_steps": 2}, "exactly one ionic step"),
            ({"spin1_kpoint": "1/2"}, "exactly one k-point per spin"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with tempfile.TemporaryDirectory() as tmp:
                    case = Path(tmp) / "fixed"
                    write_phase(case, **kwargs)
                    with self.assertRaisesRegex(ValueError, message):
                        audit_phase(case, expected_mode="fixed")

    def test_rejects_missing_or_ambiguous_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "fixed"
            write_phase(case)
            (case / "OUT.C_PBE_REFERENCE_GATE" / "eig_occ.txt").unlink()
            with self.assertRaisesRegex(ValueError, "missing eig_occ.txt"):
                audit_phase(case, expected_mode="fixed")

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "fixed"
            write_phase(case)
            (case / "running_scf.log").write_text(log_text())
            with self.assertRaisesRegex(ValueError, "ambiguous running_scf.log"):
                audit_phase(case, expected_mode="fixed")


class GateComparisonTests(unittest.TestCase):
    def valid_arguments(self):
        return {
            "fixed_energy_ha": -5.419,
            "free_energies_ha": {
                0: -5.419004,
                1: -5.419003,
                2: -5.419002,
            },
            "fixed_drift_kcal": 0.0002,
            "free_drifts_kcal": {0: 0.0003, 1: 0.0002, 2: 0.0004},
        }

    def test_accepts_equivalent_zero_field_results(self):
        summary = compare_zero_field_results(**self.valid_arguments())
        self.assertEqual(summary["status"], "PBE_GATE_PASSED")
        self.assertEqual(set(summary["free_pair_differences_ha"]), {
            "0-1", "0-2", "1-2"
        })

    def test_rejects_bad_drift(self):
        args = self.valid_arguments()
        args["free_drifts_kcal"][1] = DRIFT_TOL_KCAL
        with self.assertRaisesRegex(ValueError, "free direction 1 drift"):
            compare_zero_field_results(**args)

    def test_rejects_fixed_free_energy_difference(self):
        args = self.valid_arguments()
        args["free_energies_ha"][0] = -5.418
        with self.assertRaisesRegex(ValueError, "fixed/free energy"):
            compare_zero_field_results(**args)

    def test_rejects_free_pair_energy_difference(self):
        args = self.valid_arguments()
        args["free_energies_ha"] = {
            0: -5.419,
            1: -5.419 + 0.9 * ENERGY_TOL_HA,
            2: -5.419 - 0.9 * ENERGY_TOL_HA,
        }
        with self.assertRaisesRegex(ValueError, "free-direction energy"):
            compare_zero_field_results(**args)

    def test_rejects_missing_direction_or_nonfinite_value(self):
        args = self.valid_arguments()
        args["free_energies_ha"].pop(2)
        with self.assertRaisesRegex(ValueError, "exactly directions"):
            compare_zero_field_results(**args)

        args = self.valid_arguments()
        args["fixed_energy_ha"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_zero_field_results(**args)

        args = self.valid_arguments()
        args["free_energies_ha"] = {
            False: -5.419004,
            1: -5.419003,
            2: -5.419002,
        }
        with self.assertRaisesRegex(ValueError, "exactly directions"):
            compare_zero_field_results(**args)


class AuditCliTests(unittest.TestCase):
    def populate_gate(self, root):
        energies = {
            "runs/fixed/fixed_cold": ("fixed", ENERGY_EV + 1e-7),
            "runs/fixed/fixed_restart": ("fixed", ENERGY_EV),
        }
        for direction in range(3):
            energies[f"runs/dir{direction}/free_restart1"] = (
                "free", ENERGY_EV + direction * 1e-7
            )
            energies[f"runs/dir{direction}/free_restart2"] = (
                "free", ENERGY_EV + direction * 1e-7 + 1e-8
            )
        for relative, (mode, energy) in energies.items():
            write_phase(root / relative, mode=mode, energy_ev=energy)

    def test_cli_writes_complete_atomic_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.populate_gate(root)

            completed = subprocess.run(
                [sys.executable, str(ROOT / "audit_gate.py"), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((root / "RESULT_SUMMARY.json").read_text())
            text = (root / "RESULT_SUMMARY.txt").read_text()
            self.assertEqual(summary["status"], "PBE_GATE_PASSED")
            self.assertEqual(len(summary["phases"]), 8)
            for phase in summary["phases"].values():
                self.assertIn("energy_ev", phase)
                self.assertIn("energy_ha", phase)
                self.assertEqual(phase["spin_counts"], {"1": 3.0, "2": 1.0})
                self.assertEqual(
                    phase["occupations"],
                    {
                        "1": [1.0, 1.0, 1.0, 0.0],
                        "2": [1.0, 0.0, 0.0, 0.0],
                    },
                )
                self.assertTrue(phase["integer_occupations"])
                self.assertEqual(len(phase["stage_sha256"]), 64)
                self.assertEqual(
                    set(phase["file_sha256"]),
                    {"INPUT", "running_scf.log", "eig_occ.txt"},
                )
                for digest in phase["file_sha256"].values():
                    self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertIn("status=PBE_GATE_PASSED", text)
            self.assertIn("fixed_drift_kcal=", text)
            self.assertIn("free_direction_2_drift_kcal=", text)
            phase_lines = [line for line in text.splitlines() if line.startswith("phase=")]
            self.assertEqual(len(phase_lines), 8)
            for line in phase_lines:
                self.assertIn("spin1_occupations=1,1,1,0", line)
                self.assertIn("spin2_occupations=1,0,0,0", line)
                self.assertRegex(line, r"\bINPUT_sha256=[0-9a-f]{64}\b")
                self.assertRegex(
                    line, r"\brunning_scf\.log_sha256=[0-9a-f]{64}\b"
                )
                self.assertRegex(
                    line, r"\beig_occ\.txt_sha256=[0-9a-f]{64}\b"
                )
                self.assertRegex(line, r"\bstage_sha256=[0-9a-f]{64}\b")
            self.assertFalse(list(root.glob(".RESULT_SUMMARY.*.tmp")))

    def test_cli_failure_cannot_leave_pass_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.populate_gate(root)
            bad_input = root / "runs/dir1/free_restart2/INPUT"
            bad_input.write_text(bad_input.read_text().replace("ocp 0", "ocp 1"))
            (root / "RESULT_SUMMARY.txt").write_text("status=PBE_GATE_PASSED\n")

            completed = subprocess.run(
                [sys.executable, str(ROOT / "audit_gate.py"), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn(
                "PBE_GATE_PASSED",
                (root / "RESULT_SUMMARY.txt").read_text(),
            )
            failure = json.loads((root / "RESULT_SUMMARY.json").read_text())
            self.assertEqual(failure["status"], "PBE_GATE_FAILED")
            self.assertIn("ocp=0", failure["error"])


if __name__ == "__main__":
    unittest.main()
