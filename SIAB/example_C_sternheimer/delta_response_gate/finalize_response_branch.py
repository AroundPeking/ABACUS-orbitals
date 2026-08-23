#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from response_contract import parse_eig_occ


EXPECTED_INPUT = {
    "sternheimer_nfreq": "6",
    "sternheimer_frequency_grid_file": "fixed_frequency_grid.dat",
    "sternheimer_frequency_mpi": "1",
    "sternheimer_fd_order": "8",
    "sternheimer_delta": "1",
    "sternheimer_delta_max_states": "0",
    "exx_pca_threshold": "1e-4",
    "exx_singularity_correction": "massidda",
    "exx_ccp_rmesh_times": "1",
    "rpa_ccp_rmesh_times": "1",
    "init_wfc": "file",
    "init_chg": "file",
}
RESTART_WFC_MESSAGES = (
    "Read NAO wave functions from OUT.C_DELTA_RESPONSE_GATE/wfs1_nao.txt",
    "Read NAO wave functions from OUT.C_DELTA_RESPONSE_GATE/wfs2_nao.txt",
)
RESTART_DENSITY_MESSAGES = (
    "Read electron density from file: OUT.C_DELTA_RESPONSE_GATE/chgs1.cube",
    "Read electron density from file: OUT.C_DELTA_RESPONSE_GATE/chgs2.cube",
)
BASIS_METADATA = ("stru_out", "bz_sampling_out", "basis_out", "basis_wfc_out", "basis_aux_out")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path, root: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {
        "path": str(path.relative_to(root.resolve(strict=True))),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def require_nonempty(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    return path


def require_scf_converged(path: Path) -> None:
    text = require_nonempty(path, "SCF log").read_text(encoding="ascii", errors="replace")
    if not re.search(r"^\s*#SCF IS CONVERGED#\s*$", text, flags=re.MULTILINE):
        raise ValueError(f"SCF convergence marker is absent from {path}")


def parse_input(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in require_nonempty(path, "INPUT").read_text(encoding="ascii").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line == "INPUT_PARAMETERS":
            continue
        fields = line.split(None, 1)
        values[fields[0]] = fields[1].strip() if len(fields) == 2 else ""
    return values


def parse_report(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in require_nonempty(path, "Sternheimer report").read_text(
        encoding="ascii", errors="replace"
    ).splitlines():
        fields = raw.split(None, 1)
        if len(fields) == 2:
            values[fields[0]] = fields[1].strip()
    return values


def require_manifest(root: Path, branch: str) -> dict:
    manifest_path = require_nonempty(root / "PREPARATION_MANIFEST.json", "preparation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if branch not in manifest.get("branches", {}):
        raise ValueError(f"branch {branch!r} is absent from the preparation manifest")
    pbe_result = Path(manifest["pbe_gate_root"]) / "RESULT_SUMMARY.json"
    pbe = json.loads(require_nonempty(pbe_result, "PBE gate result").read_text(encoding="ascii"))
    if (
        pbe.get("status") != "PBE_GATE_PASSED"
        or pbe.get("zero_field_comparison_status") != "ZERO_FIELD_COMPARISON_PASSED"
        or pbe.get("blocked_on") is not None
    ):
        raise ValueError("the zero-field PBE equivalence gate has not passed")
    if sha256(pbe_result) != manifest.get("pbe_result_sha256"):
        raise ValueError("the accepted PBE result hash differs from the preparation manifest")
    return manifest


def unique_report(case: Path) -> Path:
    reports = sorted(case.glob("OUT.*/STERNHEIMER_CHI0.dat"))
    if len(reports) != 1:
        raise ValueError(f"expected one Sternheimer report, found {len(reports)}")
    return reports[0]


def finalize_response_branch(
    root: Path,
    branch: str,
    expected_abacus_sha: str,
    producer_source_commit: str,
    finalizer_source_commit: str,
) -> dict:
    root = root.resolve(strict=True)
    if branch not in {"fixed", "free"}:
        raise ValueError(f"unsupported branch: {branch}")
    require_manifest(root, branch)
    source_commit = require_nonempty(root / "SOURCE_COMMIT.txt", "producer source commit").read_text(
        encoding="ascii"
    ).strip()
    if source_commit != producer_source_commit:
        raise ValueError("producer source commit differs from the immutable campaign record")

    case = (root / "branches" / branch).resolve(strict=True)
    completion = case / "RESPONSE_COMPLETE.json"
    if completion.exists() or completion.is_symlink():
        raise FileExistsError(f"completion record already exists: {completion}")

    inputs = parse_input(case / "INPUT")
    for key, expected in EXPECTED_INPUT.items():
        if inputs.get(key) != expected:
            raise ValueError(f"INPUT mismatch for {key}: {inputs.get(key)!r} != {expected!r}")
    expected_ocp = "1" if branch == "fixed" else "0"
    if inputs.get("ocp") != expected_ocp:
        raise ValueError(f"unexpected occupation mode for {branch}")
    if branch == "fixed" and inputs.get("ocp_set") != "3*1 19*0 1*1 21*0":
        raise ValueError("fixed branch occupation string changed")
    if branch == "free" and "ocp_set" in inputs:
        raise ValueError("free branch must not contain ocp_set")

    stdout = require_nonempty(case / "abacus.out", "ABACUS stdout").read_text(
        encoding="ascii", errors="replace"
    )
    if "!!SCF IS NOT CONVERGED!!" in stdout:
        raise ValueError("ABACUS reported an unconverged SCF calculation")
    for message in RESTART_WFC_MESSAGES:
        if message not in stdout:
            raise ValueError(f"missing wave-function restart evidence: {message}")

    provenance = require_nonempty(case / "RESPONSE_PROVENANCE.txt", "response provenance").read_text(
        encoding="ascii", errors="replace"
    )
    if f"abacus_sha256={expected_abacus_sha}" not in provenance:
        raise ValueError("ABACUS executable hash is absent from response provenance")
    if f"workflow_source_commit={producer_source_commit}" not in provenance:
        raise ValueError("producer commit is absent from response provenance")

    report_path = unique_report(case)
    out_dir = report_path.parent
    scf_log = out_dir / "running_scf.log"
    require_scf_converged(scf_log)
    scf_text = scf_log.read_text(encoding="ascii", errors="replace")
    for message in RESTART_DENSITY_MESSAGES:
        if scf_text.count(message) != 1:
            raise ValueError(f"expected exactly one density restart record: {message}")

    report = parse_report(report_path)
    expected_report = {
        "status": "success",
        "frequency_grid_source": "file",
        "nfreq": "6",
        "sternheimer_fd_order": "8",
        "sternheimer_delta": "yes",
        "pca_threshold": "0.0001",
        "ccp_rmesh_times": "1",
        "sternheimer_frequency_mpi": "yes",
        "mpi_ranks": "6",
        "all_converged": "yes",
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise ValueError(f"Sternheimer report mismatch for {key}: {report.get(key)!r}")
    if report.get("format") not in {"v1", "v1_partial"}:
        raise ValueError(f"unsupported Sternheimer output format: {report.get('format')!r}")

    try:
        channels = int(report["abfs_channels"])
        occupied = int(report["occupied_bands"])
        solved = int(report["solved_equations"])
        residual = float(report["max_solver_relative_residual"])
    except (KeyError, ValueError) as error:
        raise ValueError("invalid equation counts or residual in Sternheimer report") from error
    if channels <= 0 or occupied != 4 or solved != 6 * occupied * channels:
        raise ValueError("Sternheimer equation count is inconsistent with frequencies, states, or channels")
    if not math.isfinite(residual) or residual > 1.0e-6:
        raise ValueError(f"Sternheimer residual exceeds 1e-6: {residual}")

    eig_occ = parse_eig_occ(require_nonempty(out_dir / "eig_occ.txt", "response eig_occ"))
    if eig_occ.spin_counts != {1: 3, 2: 1}:
        raise ValueError("response occupations are not the C triplet")

    response_files = sorted(case.glob("v1_sternheimer_chi0_iq_1_ifreq_*.dat"))
    if len(response_files) != 6 or any(not path.is_file() or path.stat().st_size <= 0 for path in response_files):
        raise ValueError("expected exactly six nonempty Sternheimer response files")
    coulomb = require_nonempty(case / "v1_coulomb_full_iq_1_rank0.dat", "full Coulomb matrix")
    for name in BASIS_METADATA:
        require_nonempty(case / name, name)

    payload = {
        "status": "RESPONSE_COMPLETE",
        "branch": branch,
        "abacus_sha256": expected_abacus_sha,
        "workflow_source_commit": producer_source_commit,
        "producer_source_commit": producer_source_commit,
        "finalizer_source_commit": finalizer_source_commit,
        "frequency_grid_source": "file",
        "nfreq": 6,
        "finite_difference_order": 8,
        "sternheimer_delta": True,
        "virtual_source": "all_ks_bands_implicit_feature_branch",
        "pca_threshold": 1.0e-4,
        "perturbation_hartree_kernel": "abfs_ccp_fock_alpha_1_singularity_limits",
        "response_metric": "full_coulomb_reader_v1",
        "librpa_coulomb_kernel": "full",
        "solver_tolerance": 1.0e-6,
        "max_solver_relative_residual": residual,
        "abfs_channels": channels,
        "solved_equations": solved,
        "spin_counts": {str(key): value for key, value in eig_occ.spin_counts.items()},
        "report": file_record(report_path, case),
        "response_files": [file_record(path, case) for path in response_files],
        "full_coulomb": file_record(coulomb, case),
    }
    temporary = completion.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    temporary.replace(completion)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--branch", required=True, choices=("fixed", "free"))
    parser.add_argument("--expected-abacus-sha", required=True)
    parser.add_argument("--producer-source-commit", required=True)
    parser.add_argument("--finalizer-source-commit", required=True)
    args = parser.parse_args()
    payload = finalize_response_branch(
        root=args.root,
        branch=args.branch,
        expected_abacus_sha=args.expected_abacus_sha,
        producer_source_commit=args.producer_source_commit,
        finalizer_source_commit=args.finalizer_source_commit,
    )
    print(
        "RESPONSE_OK "
        f"branch={payload['branch']} channels={payload['abfs_channels']} "
        f"residual={payload['max_solver_relative_residual']}"
    )


if __name__ == "__main__":
    main()
