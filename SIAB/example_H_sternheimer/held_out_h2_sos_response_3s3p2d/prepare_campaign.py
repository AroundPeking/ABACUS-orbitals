#!/usr/bin/env python3
"""Prepare matched TZDP and response-optimized H/H2/ghost SOS cases."""

import argparse
import hashlib
import json
import pathlib
import re
import shutil


LANES = {
    "baseline_tzdp": (3, 2),
    "optimized_3s3p2d": (3, 3, 2),
}
CASES = {
    "H": {"atom_count": 1, "asset_entries": 1, "nspin": 2, "nelec": 1},
    "H2": {"atom_count": 2, "asset_entries": 1, "nspin": 1, "nelec": 2},
    "H_ghost": {
        "atom_count": 2,
        "asset_entries": 2,
        "nspin": 2,
        "nelec": 1,
    },
}
ASSET_KEYS = (
    "baseline_orbital",
    "optimized_orbital",
    "pseudopotential",
    "auxiliary_basis",
)


def _sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(label, path, expected):
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 expected {expected}, got {actual}")
    return actual


def _orbital_radial_counts(path):
    text = pathlib.Path(path).read_text(encoding="ascii")
    lmax_matches = re.findall(r"^Lmax\s+(\d+)\s*$", text, flags=re.MULTILINE)
    if len(lmax_matches) != 1:
        raise ValueError(f"{path}: expected exactly one Lmax")
    counts = [
        int(value)
        for value in re.findall(
            r"^Number of [A-Z]orbital-->\s+(\d+)\s*$",
            text,
            flags=re.MULTILINE,
        )
    ]
    if len(counts) != int(lmax_matches[0]) + 1 or any(value <= 0 for value in counts):
        raise ValueError(f"{path}: radial orbital counts do not match Lmax")
    return tuple(counts)


def _parse_input(path):
    rows = []
    values = {}
    for line in pathlib.Path(path).read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields or fields[0] == "INPUT_PARAMETERS":
            rows.append((None, line))
            continue
        key = fields[0]
        if key in values:
            raise ValueError(f"{path}: duplicate INPUT key {key}")
        values[key] = " ".join(fields[1:])
        rows.append((key, line))
    return rows, values


def _rewrite_input(source, destination, *, suffix, nbands):
    rows, values = _parse_input(source)
    required = {
        "ecutwfc": "100",
        "rpa": "1",
        "out_librpa_reader_version": "1",
        "exx_pca_threshold": "10",
        "exx_singularity_correction": "massidda",
        "rpa_ccp_rmesh_times": "5",
        "scf_thr": "1e-8",
    }
    for key, expected in required.items():
        if values.get(key) != expected:
            raise ValueError(
                f"{source}: INPUT {key} expected {expected}, got {values.get(key)}"
            )
    replacements = {"suffix": suffix, "nbands": str(nbands)}
    counts = {key: 0 for key in replacements}
    output = []
    for key, line in rows:
        if key in replacements:
            output.append(f"{key:<24}{replacements[key]}")
            counts[key] += 1
        else:
            output.append(line)
    if counts != {"suffix": 1, "nbands": 1}:
        raise ValueError(f"{source}: INPUT rewrite count mismatch {counts}")
    pathlib.Path(destination).write_text("\n".join(output) + "\n", encoding="ascii")


def _validate_librpa(path):
    text = pathlib.Path(path).read_text(encoding="ascii")
    for token in (
        "task = rpa",
        "nfreq = 16",
        "prefix_coul_full = v1_coulomb_full_iq_",
        "version_coul_reader = 1",
        "version_lri_reader = 1",
        "vq_threshold = 0",
        "sqrt_coulomb_threshold = 0",
    ):
        if token not in text:
            raise ValueError(f"{path}: missing {token}")


def prepare_campaign(
    template_root,
    campaign_root,
    baseline_orbital,
    optimized_orbital,
    pseudopotential,
    auxiliary_basis,
    *,
    source_commit,
    expected_sha256,
):
    """Create six immutable full-band cases with a shared physical protocol."""
    if not source_commit:
        raise ValueError("source_commit must be nonempty")
    if set(expected_sha256) != set(ASSET_KEYS):
        raise ValueError(f"expected_sha256 must contain exactly {ASSET_KEYS}")
    assets = {
        "baseline_orbital": pathlib.Path(baseline_orbital).resolve(),
        "optimized_orbital": pathlib.Path(optimized_orbital).resolve(),
        "pseudopotential": pathlib.Path(pseudopotential).resolve(),
        "auxiliary_basis": pathlib.Path(auxiliary_basis).resolve(),
    }
    hashes = {
        label: _require_hash(label, path, expected_sha256[label])
        for label, path in assets.items()
    }
    orbital_counts = {
        "baseline_tzdp": _orbital_radial_counts(assets["baseline_orbital"]),
        "optimized_3s3p2d": _orbital_radial_counts(assets["optimized_orbital"]),
    }
    if orbital_counts != LANES:
        raise ValueError(
            f"orbital radial counts expected {LANES}, got {orbital_counts}"
        )

    template_root = pathlib.Path(template_root).resolve()
    campaign_root = pathlib.Path(campaign_root).resolve()
    if campaign_root.exists():
        raise ValueError(f"campaign root already exists: {campaign_root}")
    campaign_root.mkdir(parents=True)

    manifest_cases = []
    for lane, radial_counts in LANES.items():
        orbital_key = (
            "baseline_orbital" if lane == "baseline_tzdp" else "optimized_orbital"
        )
        orbital = assets[orbital_key]
        ao_per_h = sum((2 * l + 1) * count for l, count in enumerate(radial_counts))
        for case_name, case in CASES.items():
            template = template_root / case_name
            for name in ("INPUT", "STRU", "KPT", "librpa.in"):
                if not (template / name).is_file():
                    raise ValueError(f"missing template file {template / name}")
            _validate_librpa(template / "librpa.in")

            case_dir = campaign_root / lane / case_name
            case_dir.mkdir(parents=True)
            bands = ao_per_h * case["atom_count"]
            suffix = f"H2_RESPONSE_GATE_{lane}_{case_name}_BOX20A_PCA1e4_RMESH5"
            _rewrite_input(
                template / "INPUT", case_dir / "INPUT", suffix=suffix, nbands=bands
            )
            shutil.copy2(template / "KPT", case_dir / "KPT")
            shutil.copy2(template / "librpa.in", case_dir / "librpa.in")

            stru = (template / "STRU").read_text(encoding="ascii")
            old_orbital = "H_gga_8au_100Ry_13s11p10d5f4g.orb"
            expected_entries = case["asset_entries"]
            if stru.count(old_orbital) != expected_entries:
                raise ValueError(f"{template / 'STRU'}: orbital entry count mismatch")
            old_auxiliary = "H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs"
            if stru.count(old_auxiliary) != expected_entries:
                raise ValueError(f"{template / 'STRU'}: ABFS entry count mismatch")
            stru = stru.replace(old_orbital, orbital.name)
            stru = stru.replace(old_auxiliary, assets["auxiliary_basis"].name)
            (case_dir / "STRU").write_text(stru, encoding="ascii")

            for asset in (orbital, assets["pseudopotential"], assets["auxiliary_basis"]):
                shutil.copy2(asset, case_dir / asset.name)
            manifest_cases.append(
                {
                    "lane": lane,
                    "case": case_name,
                    "directory": str(case_dir),
                    "radial_orbitals_by_l": list(radial_counts),
                    "ao_per_h": ao_per_h,
                    "nbands": bands,
                    "nspin": case["nspin"],
                    "nelec": case["nelec"],
                    "orbital_asset_key": orbital_key,
                    "orbital_filename": orbital.name,
                    "orbital_sha256": hashes[orbital_key],
                    "input_sha256": {
                        name: _sha256(case_dir / name)
                        for name in ("INPUT", "STRU", "KPT", "librpa.in")
                    },
                }
            )

    manifest = {
        "schema_version": 1,
        "method": "matched_full_band_h2_sos_gate",
        "source_commit": source_commit,
        "asset_sha256": hashes,
        "asset_filenames": {
            label: path.name for label, path in assets.items()
        },
        "physics": {
            "cell_angstrom": 20.0,
            "h2_bond_angstrom": 0.74085,
            "ecutwfc_ry": 100,
            "nfreq": 16,
            "coulomb_kernel": "full",
            "rpa_ccp_rmesh_times": 5,
            "auxiliary_basis_pca_threshold": 1.0e-4,
            "explicit_abfs": True,
            "abacus_exx_pca_threshold": 10,
        },
        "cases": manifest_cases,
    }
    (campaign_root / "campaign.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_root", type=pathlib.Path)
    parser.add_argument("campaign_root", type=pathlib.Path)
    parser.add_argument("baseline_orbital", type=pathlib.Path)
    parser.add_argument("optimized_orbital", type=pathlib.Path)
    parser.add_argument("pseudopotential", type=pathlib.Path)
    parser.add_argument("auxiliary_basis", type=pathlib.Path)
    parser.add_argument("--source-commit", required=True)
    for key in ASSET_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}-sha256", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    expected = {
        key: getattr(args, f"{key}_sha256")
        for key in ASSET_KEYS
    }
    manifest = prepare_campaign(
        args.template_root,
        args.campaign_root,
        args.baseline_orbital,
        args.optimized_orbital,
        args.pseudopotential,
        args.auxiliary_basis,
        source_commit=args.source_commit,
        expected_sha256=expected,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
