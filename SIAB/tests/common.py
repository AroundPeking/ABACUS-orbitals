from pathlib import Path
import sys

OPT_DIR = Path(__file__).resolve().parents[1] / "opt_orb_pytorch_dpsi"
if str(OPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPT_DIR))

import util


def info(**values):
    result = util.Info()
    result.__dict__.update(values)
    return result
