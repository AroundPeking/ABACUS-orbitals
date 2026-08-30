#!/usr/bin/env python3
"""Stage an immutable per-channel prefix selected from the original C TZDP orbital."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from truncate_abacus_orbital import select_abacus_orbital_channels


APPROVED_LAYOUTS = {
    (2, 2, 1): ("nested_tzdp_2s2p1d", "2s2p1d", 13),
    (3, 2, 1): ("nested_tzdp_3s2p1d", "3s2p1d", 14),
    (2, 3, 1): ("nested_tzdp_2s3p1d", "2s3p1d", 16),
    (2, 2, 2): ("nested_tzdp_2s2p2d", "2s2p2d", 18),
    (3, 3, 1): ("nested_tzdp_3s3p1d", "3s3p1d", 17),
    (3, 2, 2): ("nested_tzdp_3s2p2d", "3s2p2d", 19),
    (2, 3, 2): ("nested_tzdp_2s3p2d", "2s3p2d", 21),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_candidate(source: Path, root: Path, *, target_nu) -> dict:
    source = Path(source).resolve(strict=True)
    root = Path(root).resolve()
    key = tuple(int(count) for count in target_nu)
    if key not in APPROVED_LAYOUTS:
        raise ValueError("target_nu is not one of the approved nested layouts")
    if root.exists():
        raise FileExistsError(root)
    if not root.parent.is_dir():
        raise ValueError("candidate parent directory does not exist")

    profile, layout, nao = APPROVED_LAYOUTS[key]
    temporary = Path(tempfile.mkdtemp(prefix=root.name + ".tmp-", dir=root.parent))
    try:
        orbital_name = "C_gga_10au_100Ry_{}.orb".format(layout)
        orbital = temporary / orbital_name
        selection = select_abacus_orbital_channels(source, orbital, target_nu=key)
        selection["output"] = str((root / orbital_name).resolve())
        if selection["output_nao"] != nao:
            raise ValueError("selected orbital AO count does not match approved layout")
        selection_path = temporary / "SELECTION.json"
        selection_path.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        candidate = {
            "ao_count_atom": nao,
            "orbital_filename": orbital_name,
            "orbital_sha256": sha256(orbital),
            "profile": profile,
            "selection": "per_channel_radial_prefix_from_original_sg15_tzdp",
            "source_orbital": str(source),
            "source_orbital_sha256": sha256(source),
            "status": "success",
            "nu": list(key),
        }
        candidate_path = temporary / "CANDIDATE.json"
        candidate_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        provenance = temporary / "provenance.txt"
        provenance.write_text(
            "status=success\n"
            "purpose=nested_tzdp_single_channel_causal_scan\n"
            "profile={}\n"
            "layout={}\n"
            "source_orbital_sha256={}\n"
            "selected_orbital_sha256={}\n"
            "candidate_manifest_sha256={}\n"
            "selection_manifest_sha256={}\n".format(
                profile,
                layout,
                candidate["source_orbital_sha256"],
                candidate["orbital_sha256"],
                sha256(candidate_path),
                sha256(selection_path),
            ),
            encoding="ascii",
        )
        (temporary / "STATUS").write_text("success\n", encoding="ascii")
        os.replace(temporary, root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return candidate


def _parse_target_nu(value: str):
    try:
        return [int(token) for token in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("target-nu must be comma-separated integers") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-nu", type=_parse_target_nu, required=True)
    args = parser.parse_args()
    result = prepare_candidate(args.source, args.root, target_nu=args.target_nu)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
