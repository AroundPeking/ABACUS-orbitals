import itertools
import os
import re

import torch


# occ[ik][ib] after flattening spin blocks in ascending spin order
# Legacy SIAB expects paths ending with istate.info. Newer ABACUS writes eig.txt instead.
def read_istate(file_name):
    resolved = resolve_istate_file(file_name)
    if os.path.basename(resolved) == "eig.txt":
        return read_eig(resolved)
    return read_legacy_istate(resolved)


def resolve_istate_file(file_name):
    if os.path.exists(file_name):
        return file_name
    sibling = os.path.join(os.path.dirname(file_name), "eig.txt")
    if os.path.exists(sibling):
        return sibling
    raise FileNotFoundError(file_name)


def read_legacy_istate(file_name):
    nspin0 = get_nspin0_legacy(file_name)
    occ = [[] for _ in range(nspin0)]
    with open(file_name, "r") as file:
        content = file.read().split("BAND")
        for content_k in content[1:]:
            content_k = content_k.split("\n")
            _ = get_k_legacy(content_k[0])
            for ispin in range(nspin0):
                occ[ispin].append([])
            for line in content_k[1:]:
                line = line.strip()
                if not line:
                    continue
                fields = line.split()
                if nspin0 == 1:
                    occ[0][-1].append(float(fields[2]))
                elif nspin0 == 2:
                    occ[0][-1].append(float(fields[2]))
                    occ[1][-1].append(float(fields[4]))
            for ispin in range(nspin0):
                occ[ispin][-1] = torch.Tensor(occ[ispin][-1])
    return list(itertools.chain(*occ))


def read_eig(file_name):
    spin_blocks = {}
    current_occ = None
    with open(file_name, "r") as file:
        for raw in file:
            line = raw.strip()
            if not line:
                continue
            match = re.match(r"spin=(\d+)\s+k-point=(\d+)/(\d+)", line)
            if match:
                spin = int(match.group(1))
                spin_blocks.setdefault(spin, []).append([])
                current_occ = spin_blocks[spin][-1]
                continue
            if current_occ is None:
                continue
            fields = line.split()
            if len(fields) >= 3 and fields[0].isdigit():
                current_occ.append(float(fields[2]))

    occ = []
    for spin in sorted(spin_blocks):
        occ.extend(torch.Tensor(k_occ) for k_occ in spin_blocks[spin])
    return occ


def get_k_legacy(line):
    return int(re.compile(r"Kpoint\s*=\s*(\d+)").search(line).group(1))


def get_nspin0_legacy(file_name):
    with open(file_name, "r") as file:
        file.readline()
        line = file.readline()
        lens = len(line.split())
        if lens == 3:
            return 1
        if lens == 5:
            return 2
        raise ValueError(f"Unsupported istate.info header format in {file_name}: {line.rstrip()}")
