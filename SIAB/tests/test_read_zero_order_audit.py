import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


OPT_DIR = Path(__file__).resolve().parents[1] / "opt_orb_pytorch_dpsi"
if str(OPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPT_DIR))

from IO.read_zero_order_audit import read_zero_order_audit


def valid_payload(case="H"):
    return {
        "format": "sternheimer_siab_zero_order_identity_v1",
        "case": case,
        "pass": True,
        "checks": {
            "abacus_finish_marker": True,
            "charge_grid_exact": True,
            "final_total_energy_le_1e_12_ha": True,
            "nbands_exact": True,
            "new_scf_complete": True,
            "occupations_le_1e_14": True,
            "occupied_eigenvalues_le_1e_12_ha": True,
            "occupied_state_count_exact": True,
            "old_scf_complete": True,
            "wavefunction_grid_exact": True,
        },
        "eig_occ_comparison": {
            "max_occupation_abs_diff": 5.0e-15,
            "max_occupied_eigenvalue_abs_diff_ha": 7.0e-13,
            "occupied_state_count": 1,
        },
        "running_log_comparison": {
            "charge_grid": [180, 180, 180],
            "final_total_energy_abs_diff_ha": 8.0e-13,
            "finish_marker": True,
            "nbands": 9,
            "scf_complete": True,
            "wavefunction_grid": [180, 180, 180],
        },
        "files": {
            name: {"path": f"/{case}/{name}", "sha256": digit * 64}
            for name, digit in (
                ("new_eig_occ", "1"),
                ("new_running_scf_log", "2"),
                ("old_eig_occ", "3"),
                ("old_running_scf_log", "4"),
            )
        },
        "thresholds": {
            "final_total_energy_abs_diff_ha": 1.0e-12,
            "occupation_abs_diff": 1.0e-14,
            "occupied_eigenvalue_abs_diff_ha": 1.0e-12,
        },
    }


class ReadZeroOrderAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "audit.json"

    def write(self, payload):
        self.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def test_reads_valid_audit_into_frozen_validated_values(self):
        self.write(valid_payload())

        audit = read_zero_order_audit(self.path, "H")

        self.assertEqual(audit.case, "H")
        self.assertTrue(audit.passed)
        self.assertEqual(audit.occupied_state_count, 1)
        self.assertEqual(audit.grid, (180, 180, 180))
        self.assertEqual(audit.max_occupation_abs_diff, 5.0e-15)
        self.assertEqual(
            audit.max_occupied_eigenvalue_abs_diff_ha, 7.0e-13
        )
        self.assertEqual(audit.final_total_energy_abs_diff_ha, 8.0e-13)
        self.assertEqual(
            dict(audit.source_file_sha256),
            {
                "new_eig_occ": "1" * 64,
                "new_running_scf_log": "2" * 64,
                "old_eig_occ": "3" * 64,
                "old_running_scf_log": "4" * 64,
            },
        )
        with self.assertRaisesRegex(Exception, "cannot assign"):
            audit.case = "H2"

    def test_rejects_false_check_and_false_top_level_pass(self):
        for mutate, message in (
            (
                lambda payload: payload["checks"].__setitem__(
                    "charge_grid_exact", False
                ),
                "check failed: charge_grid_exact",
            ),
            (
                lambda payload: payload.__setitem__("pass", False),
                "did not pass",
            ),
        ):
            with self.subTest(message=message):
                payload = valid_payload()
                mutate(payload)
                self.write(payload)
                with self.assertRaisesRegex(ValueError, message):
                    read_zero_order_audit(self.path, "H")

    def test_rejects_wrong_case_and_loose_threshold(self):
        self.write(valid_payload(case="H2"))
        with self.assertRaisesRegex(ValueError, "wrong case"):
            read_zero_order_audit(self.path, "H")

        payload = valid_payload()
        payload["thresholds"]["occupation_abs_diff"] = 1.1e-14
        self.write(payload)
        with self.assertRaisesRegex(ValueError, "threshold is too loose"):
            read_zero_order_audit(self.path, "H")

    def test_rejects_missing_sha256_and_nonidentical_grids(self):
        payload = valid_payload()
        del payload["files"]["new_eig_occ"]["sha256"]
        self.write(payload)
        with self.assertRaisesRegex(ValueError, "file record is invalid"):
            read_zero_order_audit(self.path, "H")

        payload = valid_payload()
        payload["running_log_comparison"]["wavefunction_grid"] = [181, 180, 180]
        self.write(payload)
        with self.assertRaisesRegex(ValueError, "grids differ"):
            read_zero_order_audit(self.path, "H")

    def test_rejects_nonfinite_differences(self):
        for section, key in (
            ("eig_occ_comparison", "max_occupation_abs_diff"),
            ("eig_occ_comparison", "max_occupied_eigenvalue_abs_diff_ha"),
            ("running_log_comparison", "final_total_energy_abs_diff_ha"),
        ):
            with self.subTest(key=key):
                payload = valid_payload()
                payload[section][key] = math.inf
                self.write(payload)
                with self.assertRaisesRegex(ValueError, "finite number"):
                    read_zero_order_audit(self.path, "H")


if __name__ == "__main__":
    unittest.main()
