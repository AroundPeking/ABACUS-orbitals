#!/usr/bin/env python3
import json
import os
from pathlib import Path
import platform
import runpy
import sys

import matplotlib
import torch


def configure_threads():
    expected = int(os.environ["SLURM_CPUS_PER_TASK"])
    torch.set_num_threads(expected)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != expected:
        raise RuntimeError("Torch intra-op thread count does not match allocation")
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch inter-op thread count is not one")


def runtime_record():
    return {
        "executable": sys.executable,
        "matplotlib_version": matplotlib.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_config_parallel_info": torch.__config__.parallel_info(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }


def main():
    configure_threads()
    if sys.argv[1:] == ["--versions"]:
        print(json.dumps(runtime_record(), indent=2, sort_keys=True))
        return 0
    if len(sys.argv) < 2:
        raise SystemExit("usage: task7_torch_launcher.py ANALYZER [ARG ...]")
    analyzer = Path(sys.argv[1]).resolve(strict=True)
    sys.argv = [str(analyzer), *sys.argv[2:]]
    runpy.run_path(str(analyzer), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
