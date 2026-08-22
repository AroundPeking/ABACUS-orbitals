import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_response_gate import (  # noqa: E402
    COULOMB_MARKER,
    RESPONSE_MARKER,
    audit_response_gate,
    read_coulomb_v1,
    read_response_v1,
)


def write_one_atom_matrix(path, marker, matrix, ifreq=None, omega=None, weight=None):
    matrix = np.asarray(matrix, dtype=np.complex128)
    naux = matrix.shape[0]
    if marker == COULOMB_MARKER:
        header = struct.pack("=6i", marker, 1, naux, 1, 1, 1)
    else:
        header = struct.pack(
            "=6i2di", marker, 1, ifreq, naux, 1, 1, omega, weight, 1
        )
    table_end = len(header) + 4 + 12
    path.write_bytes(
        header
        + struct.pack("=i", naux)
        + struct.pack("=iq", 0, table_end)
        + matrix.tobytes(order="C")
    )


def write_branch(root, branch, perturbation=0.0):
    case = root / "branches" / branch
    case.mkdir(parents=True)
    coulomb = np.diag([4.0, 9.0])
    write_one_atom_matrix(
        case / "v1_coulomb_full_iq_1_rank0.dat", COULOMB_MARKER, coulomb
    )
    for ifreq in range(1, 7):
        response = np.diag(
            [-0.4 * ifreq * (1.0 + perturbation), -0.9 * ifreq]
        )
        write_one_atom_matrix(
            case / f"v1_sternheimer_chi0_iq_1_ifreq_{ifreq}.dat",
            RESPONSE_MARKER,
            response,
            ifreq=ifreq,
            omega=0.1 * ifreq,
            weight=0.05 * ifreq,
        )
    (case / "RESPONSE_COMPLETE.json").write_text(
        json.dumps({"status": "RESPONSE_COMPLETE", "branch": branch}) + "\n",
        encoding="ascii",
    )
    librpa = case / "librpa"
    librpa.mkdir()
    rows = [
        {
            "ifreq": ifreq,
            "values": [0.1 * ifreq, 0.05 * ifreq, -0.01 * ifreq, 0.0, -0.001, 0.0],
        }
        for ifreq in range(1, 7)
    ]
    (librpa / "LIBRPA_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "LIBRPA_COMPLETE",
                "branch": branch,
                "coulomb_kernel": "full",
                "sqrt_coulomb_threshold": 1.0e-5,
                "trace_log_rows": rows,
                "ecrpa_ha": -0.006,
            }
        )
        + "\n",
        encoding="ascii",
    )


class AuditResponseGateTests(unittest.TestCase):
    def test_reader_reconstructs_coulomb_and_response_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = np.array([[2.0, 0.2j], [-0.2j, 3.0]])
            coulomb = root / "coulomb.dat"
            response = root / "response.dat"
            write_one_atom_matrix(coulomb, COULOMB_MARKER, matrix)
            write_one_atom_matrix(
                response,
                RESPONSE_MARKER,
                -matrix,
                ifreq=2,
                omega=0.4,
                weight=0.7,
            )
            coulomb_record = read_coulomb_v1([coulomb])
            response_record = read_response_v1(response)
            np.testing.assert_allclose(coulomb_record.matrix, matrix)
            np.testing.assert_allclose(response_record.matrix, -matrix)
            self.assertEqual(response_record.ifreq, 2)
            self.assertAlmostEqual(response_record.omega, 0.4)
            self.assertAlmostEqual(response_record.weight, 0.7)

    def test_gate_passes_equivalent_zero_field_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_branch(root, "fixed", perturbation=0.0)
            write_branch(root, "free", perturbation=1.0e-6)
            result = audit_response_gate(root)
            self.assertEqual(result["status"], "DELTA_RESPONSE_GATE_PASSED")
            self.assertLess(result["max_pi_spectrum_relative_difference"], 1.0e-3)
            self.assertLess(result["ecrpa_difference_kcal_per_mol"], 0.1)

    def test_gate_blocks_a_physically_different_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_branch(root, "fixed", perturbation=0.0)
            write_branch(root, "free", perturbation=0.1)
            result = audit_response_gate(root)
            self.assertEqual(result["status"], "DELTA_RESPONSE_GATE_BLOCKED")
            self.assertIn("pi_spectrum", result["blocked_on"])


if __name__ == "__main__":
    unittest.main()
