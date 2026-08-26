import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "opt_orb_pytorch_dpsi" / "IO" / "read_istate.py"


def load_reader():
    spec = importlib.util.spec_from_file_location("siab_read_istate", READER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CurrentAbacusBandWeightTest(unittest.TestCase):
    def test_reads_gamma_only_eig_occ_for_two_spins(self):
        content = """1     # ionic step
 Electronic state energy (eV) and occupations
 Spin number 2
 spin=1 k-point=1/1 Cartesian=0.0 0.0 0.0 (100 plane wave)
 1 -10.0 1.0
 2 -2.0 1.0
 3 1.0 0.0

 spin=2 k-point=1/1 Cartesian=0.0 0.0 0.0 (100 plane wave)
 1 -9.0 1.0
 2 -1.0 0.0
 3 2.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eig_occ.txt"
            path.write_text(content)
            occupations = load_reader().read_istate(path)

        self.assertEqual(len(occupations), 2)
        self.assertEqual(occupations[0].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(occupations[1].tolist(), [1.0, 0.0, 0.0])

    def test_preserves_legacy_istate_support(self):
        content = """legacy istate
1 -1.0 2.0
BAND Kpoint = 1
1 -10.0 1.0
2 1.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "istate.info"
            path.write_text(content)
            occupations = load_reader().read_istate(path)

        self.assertEqual(len(occupations), 1)
        self.assertEqual(occupations[0].tolist(), [1.0, 0.0])

    def test_auto_weight_source_uses_current_abacus_output(self):
        source = (ROOT / "SIAB.py").read_text()
        self.assertIn('pwDataPath_STRU[STRUname][iBL]+"/eig_occ.txt"', source)


if __name__ == "__main__":
    unittest.main()
