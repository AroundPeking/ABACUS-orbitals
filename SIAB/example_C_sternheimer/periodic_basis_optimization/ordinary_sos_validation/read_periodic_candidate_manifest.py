#!/usr/bin/env python3
"""Read either a staged candidate or legacy truncation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STAGED_LAYOUTS = {
    "relaxed_dzp": ([3, 3, 2, 0, 0], 22, "3s3p2d"),
    "fixed_dzp": ([3, 3, 2, 0, 0], 22, "3s3p2d"),
    "nested_tzdp_2s2p1d": ([2, 2, 1], 13, "2s2p1d"),
    "nested_tzdp_3s2p1d": ([3, 2, 1], 14, "3s2p1d"),
    "nested_tzdp_2s3p1d": ([2, 3, 1], 16, "2s3p1d"),
    "nested_tzdp_2s2p2d": ([2, 2, 2], 18, "2s2p2d"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_candidate(root: Path) -> dict:
    root = Path(root).resolve(strict=True)
    staged = root / "CANDIDATE.json"
    legacy = root / "TRUNCATION.json"
    if staged.is_file() and not legacy.exists():
        payload = json.loads(staged.read_text(encoding="ascii"))
        profile = payload.get("profile")
        if payload.get("status") != "success" or profile not in STAGED_LAYOUTS:
            raise ValueError("unsupported staged candidate")
        expected_nu, nao, layout = STAGED_LAYOUTS[profile]
        if payload.get("nu") != expected_nu or payload.get("ao_count_atom") != nao:
            raise ValueError("staged candidate layout mismatch")
        orbital = (root / payload["orbital_filename"]).resolve(strict=True)
        expected_sha = payload["orbital_sha256"]
        manifest = staged
    elif legacy.is_file() and not staged.exists():
        payload = json.loads(legacy.read_text(encoding="ascii"))
        if payload.get("source_nu") != [3, 3, 2, 1, 1]:
            raise ValueError("legacy source layout mismatch")
        if payload.get("output_nu") != [3, 3, 2, 1] or payload.get("output_nao") != 29:
            raise ValueError("legacy truncated layout mismatch")
        orbital = Path(payload["output"]).resolve(strict=True)
        expected_sha = sha256(orbital)
        layout = "3s3p2d1f"
        manifest = legacy
        nao = 29
    else:
        raise ValueError("candidate root must contain exactly one supported manifest")
    if orbital.parent != root:
        raise ValueError("candidate orbital must be inside the immutable root")
    actual_sha = sha256(orbital)
    if actual_sha != expected_sha:
        raise ValueError("candidate orbital SHA256 mismatch")
    return {
        "manifest": str(manifest),
        "orbital": str(orbital),
        "orbital_name": orbital.name,
        "orbital_sha256": actual_sha,
        "nao_atom": nao,
        "layout": layout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    args = parser.parse_args()
    result = read_candidate(args.candidate_root)
    for key in ("manifest", "orbital_name", "orbital_sha256", "nao_atom", "layout"):
        print(result[key])


if __name__ == "__main__":
    main()
