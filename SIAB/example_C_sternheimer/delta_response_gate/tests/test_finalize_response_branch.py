import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finalize_response_branch import require_scf_converged  # noqa: E402


class FinalizeResponseBranchTests(unittest.TestCase):
    def test_accepts_the_actual_abacus_scf_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "running_scf.log"
            log.write_text("setup\n #SCF IS CONVERGED#\nresponse\n", encoding="ascii")
            require_scf_converged(log)

    def test_rejects_an_old_stdout_phrase_without_the_actual_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "running_scf.log"
            log.write_text("convergence has been achieved\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "SCF convergence marker"):
                require_scf_converged(log)


if __name__ == "__main__":
    unittest.main()
