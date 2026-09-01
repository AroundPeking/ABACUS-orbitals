#!/usr/bin/env python3
"""Freeze the q3-selected C Galerkin candidate for physical validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


HERE = Path(__file__).resolve().parent
PERIODIC_ROOT = HERE.parent
SIAB_ROOT = PERIODIC_ROOT.parents[1]
sys.path.insert(0, str(PERIODIC_ROOT))
sys.path.insert(0, str(SIAB_ROOT / "opt_orb_pytorch_dpsi"))

from export_periodic_orbitals import write_abacus_orbital  # noqa: E402
from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
)


NU = (3, 3, 2, 0, 0)
AO_COUNT_ATOM = 22
PROFILE = "galerkin_pareto_dzp"
COEFFICIENT_NAME = "C_3s3p2d_galerkin_pareto.txt"
ORBITAL_NAME = "C_gga_10au_100Ry_3s3p2d_galerkin_pareto.orb"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path, name):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"{name} must be a nonempty regular file")
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _require_success(payload, name):
    if payload.get("status") != "success":
        raise ValueError(f"{name} status must be success")


def _require_sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _inside(path, root, name):
    path = Path(path).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside the candidate bank root") from error
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"{name} must be a nonempty regular file")
    return path


def stage_selected_candidate(*, bank_root, q3_root, output_directory, source_commit):
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git hash")
    bank_root = Path(bank_root).resolve(strict=True)
    q3_root = Path(q3_root).resolve(strict=True)
    output = Path(output_directory).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("candidate output parent directory does not exist")

    bank_path = bank_root / "CANDIDATE_BANK.json"
    bank = _read_json(bank_path, "candidate bank")
    bank_status = _read_json(bank_root / "STATUS.json", "candidate bank status")
    bank_provenance = _read_json(
        bank_root / "PROVENANCE.json",
        "candidate bank provenance",
    )
    _require_success(bank, "candidate bank")
    _require_success(bank_status, "candidate bank status")
    _require_success(bank_provenance, "candidate bank provenance")
    if bank.get("format_version") != 1 or bank.get("candidate_bank_gate") != "pass":
        raise ValueError("candidate bank must pass the version-1 gate")
    bank_sha = sha256(bank_path)
    for payload in (bank_status, bank_provenance):
        if payload.get("candidate_bank_sha256") != bank_sha:
            raise ValueError("candidate bank provenance hash differs")

    selection_path = q3_root / "SELECTION_RESULT.json"
    selection = _read_json(selection_path, "q3 selection")
    selection_status = _read_json(q3_root / "STATUS.json", "q3 selection status")
    selection_provenance = _read_json(
        q3_root / "PROVENANCE.json",
        "q3 selection provenance",
    )
    _require_success(selection, "q3 selection")
    _require_success(selection_status, "q3 selection status")
    _require_success(selection_provenance, "q3 selection provenance")
    if selection.get("format_version") != 1 or selection.get("gate") != "pass":
        raise ValueError("q3 selection gate must pass")
    selection_sha = sha256(selection_path)
    for payload in (selection_status, selection_provenance):
        if payload.get("selection_sha256") != selection_sha:
            raise ValueError("q3 selection provenance hash differs")

    selected_name = selection.get("selected_candidate")
    candidates = bank.get("candidates")
    if not isinstance(selected_name, str) or not isinstance(candidates, list):
        raise ValueError("selected candidate is missing from the bank")
    matches = [record for record in candidates if record.get("name") == selected_name]
    if len(matches) != 1 or matches[0].get("family_tradeoff_gate") != "pass":
        raise ValueError("selected candidate is not a unique promotable bank entry")
    record = matches[0]
    bank_candidate_sha = _require_sha(
        record.get("orbital_sha256"),
        "candidate bank orbital hash",
    )
    selected_sha = _require_sha(
        selection.get("selected_orbital_sha256"),
        "selected orbital hash",
    )
    if selected_sha != bank_candidate_sha:
        raise ValueError("selection hash differs from the candidate bank")
    source_coefficients = _inside(
        record.get("orbital_file"),
        bank_root,
        "selected coefficients",
    )
    if sha256(source_coefficients) != selected_sha:
        raise ValueError("selected coefficient file SHA256 mismatch")

    coefficients = read_periodic_optimizer_coefficients(
        source_coefficients,
        element="C",
        radial_rows=31,
        expected_nu=NU,
    )
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        coefficient_path = temporary / COEFFICIENT_NAME
        orbital_path = temporary / ORBITAL_NAME
        shutil.copyfile(source_coefficients, coefficient_path)
        write_abacus_orbital(
            orbital_path,
            coefficients,
            element="C",
            ecut_ry=100.0,
            rcut_bohr=10.0,
            dr_bohr=0.01,
            smoothing_sigma_bohr=0.1,
        )
        payload = {
            "ao_count_atom": AO_COUNT_ATOM,
            "candidate_bank_sha256": bank_sha,
            "coefficients_filename": COEFFICIENT_NAME,
            "coefficients_sha256": sha256(coefficient_path),
            "nu": list(NU),
            "orbital_filename": ORBITAL_NAME,
            "orbital_sha256": sha256(orbital_path),
            "profile": PROFILE,
            "q3_selection_sha256": selection_sha,
            "selected_candidate": selected_name,
            "source_coefficients_sha256": selected_sha,
            "source_commit": source_commit,
            "status": "success",
        }
        manifest = temporary / "CANDIDATE.json"
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        provenance = {
            "status": "success",
            "source_commit": source_commit,
            "script_sha256": sha256(Path(__file__)),
            "candidate_bank_sha256": bank_sha,
            "q3_selection_sha256": selection_sha,
            "source_coefficients_sha256": selected_sha,
            "candidate_manifest_sha256": sha256(manifest),
            "orbital_sha256": payload["orbital_sha256"],
        }
        (temporary / "PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        (temporary / "STATUS.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "selected_candidate": selected_name,
                    "orbital_sha256": payload["orbital_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        (temporary / "STATUS").write_text("success\n", encoding="ascii")
        (temporary / "provenance.txt").write_text(
            "status=success\n"
            "purpose=galerkin_pareto_candidate_physical_validation\n"
            f"source_commit={source_commit}\n"
            f"selected_candidate={selected_name}\n"
            f"candidate_bank_sha256={bank_sha}\n"
            f"q3_selection_sha256={selection_sha}\n"
            f"source_coefficients_sha256={selected_sha}\n"
            f"selected_orbital_sha256={payload['orbital_sha256']}\n"
            f"candidate_manifest_sha256={sha256(manifest)}\n",
            encoding="ascii",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--q3-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = stage_selected_candidate(
        bank_root=args.bank_root,
        q3_root=args.q3_root,
        output_directory=args.output_directory,
        source_commit=args.source_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
