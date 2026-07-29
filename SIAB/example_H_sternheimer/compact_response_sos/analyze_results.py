#!/usr/bin/env python3
"""Validate and summarize the matched compact-basis H2/H/H+ghost campaign."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


COMMON_ANALYZER = (
    Path(__file__).resolve().parents[1]
    / "held_out_h2_sos_greedy_full"
)
if str(COMMON_ANALYZER) not in sys.path:
    sys.path.insert(0, str(COMMON_ANALYZER))

from analyze_truncated_cp import (  # noqa: E402
    combine_counterpoise,
    parse_abacus_energy,
    parse_librpa_ec,
)


DEFAULT_LANES = ("tail_0p00", "tail_0p10", "tail_0p30")
EXPECTED_WEIGHTS = {
    "tail_0p00": 0.0,
    "tail_0p10": 0.1,
    "tail_0p30": 0.3,
}
CASES = ("H2", "H", "H_ghost")


def _one(path, pattern, label):
    matches = tuple(Path(path).glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected exactly one {label}, found {len(matches)}"
        )
    return matches[0]


def _load_selection_contract(path, lane):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "lane_weight",
        "selection_status",
        "selection_steps",
        "ao_function_count",
        "nu",
        "orbital_sha256",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path}: missing selection keys {sorted(missing)}")
    if payload["format_version"] != 1:
        raise ValueError(f"{path}: unsupported selection format")
    if payload["selection_status"] != "ao_budget_reached":
        raise ValueError(f"{path}: selection did not reach the AO budget")
    if payload["ao_function_count"] != 48:
        raise ValueError(f"{path}: expected 48 AO/H")
    if not isinstance(payload["selection_steps"], int) or payload["selection_steps"] <= 0:
        raise ValueError(f"{path}: invalid selection step count")
    if not isinstance(payload["nu"], list) or len(payload["nu"]) != 5:
        raise ValueError(f"{path}: invalid compact basis multiplicities")
    expected_weight = EXPECTED_WEIGHTS.get(lane)
    if expected_weight is None:
        raise ValueError(f"unsupported compact-response lane {lane}")
    if not math.isclose(
        float(payload["lane_weight"]),
        expected_weight,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{path}: radial-tail lane mismatch")
    digest = payload["orbital_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{path}: invalid orbital SHA256")
    return payload


def _verify_checksums(case_dir):
    checksum_path = Path(case_dir) / "PRODUCTION_OUTPUTS.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{checksum_path}: empty checksum manifest")
    for line in lines:
        fields = line.split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"{checksum_path}: invalid checksum row {line!r}")
        relative = fields[1].lstrip("* ")
        target = (Path(case_dir) / relative).resolve()
        try:
            target.relative_to(Path(case_dir).resolve())
        except ValueError as exc:
            raise ValueError(f"{checksum_path}: path escapes case directory") from exc
        if not target.is_file():
            raise ValueError(f"{checksum_path}: missing output {relative}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != fields[0]:
            raise ValueError(f"{checksum_path}: checksum mismatch for {relative}")


def _case_result(case_dir, lane):
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise ValueError(f"{case_dir}: missing production case")
    selection = _load_selection_contract(
        case_dir / "selection_contract.json", lane
    )
    abacus_stdout = _one(case_dir, "abacus.*.out", "ABACUS stdout")
    librpa_stdout = _one(case_dir, "librpa.*.out", "LibRPA stdout")
    output_dir = _one(case_dir, "OUT.*", "ABACUS output directory")
    librpa_text = librpa_stdout.read_text(encoding="utf-8")
    if librpa_text.count("libRPA finished successfully") != 1:
        raise ValueError(f"{librpa_stdout}: LibRPA did not finish successfully")
    producer = parse_abacus_energy(
        abacus_stdout, output_dir / "running_scf.log"
    )
    ec = parse_librpa_ec(librpa_stdout)
    _verify_checksums(case_dir)
    return {
        "selection": selection,
        "producer": producer,
        "ec_ha": ec,
    }


def summarize_campaign(campaign_root, lanes=DEFAULT_LANES):
    campaign_root = Path(campaign_root)
    rows = []
    for lane in lanes:
        cases = {
            case: _case_result(campaign_root / lane / case, lane)
            for case in CASES
        }
        selection = cases["H2"]["selection"]
        for case in CASES[1:]:
            if cases[case]["selection"] != selection:
                raise ValueError(
                    f"{lane}: {case} did not use the same frozen orbital contract"
                )
        energies = {
            case: {
                "zero_order_ha": value["producer"]["zero_order_ha"],
                "ec_ha": value["ec_ha"],
            }
            for case, value in cases.items()
        }
        rows.append(
            {
                "lane": lane,
                "selection": selection,
                "cases": {
                    case: {
                        **value["producer"],
                        "ec_ha": value["ec_ha"],
                    }
                    for case, value in cases.items()
                },
                "binding": combine_counterpoise(
                    energies["H2"], energies["H"], energies["H_ghost"]
                ),
            }
        )
    return {"format_version": 1, "lanes": rows}


def _markdown(payload):
    lines = [
        "| lane | basis | ST loss | tail loss | D raw | D CP | BSSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["lanes"]:
        selection = row["selection"]
        metrics = selection.get("optimization_metrics", {})
        binding = row["binding"]
        basis = "".join(
            f"{count}{label}"
            for count, label in zip(selection["nu"], "spdfg")
            if count
        )
        lines.append(
            "| {lane} | {basis} | {st:.8f} | {tail:.8f} | "
            "{raw:.6f} | {cp:.6f} | {bsse:.6f} |".format(
                lane=row["lane"],
                basis=basis,
                st=float(metrics.get("sternheimer_loss", float("nan"))),
                tail=float(metrics.get("radial_tail_loss", float("nan"))),
                raw=binding["raw_total_kcal_mol"],
                cp=binding["cp_total_kcal_mol"],
                bsse=binding["bsse_total_kcal_mol"],
            )
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = summarize_campaign(arguments.campaign_root)
    arguments.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    arguments.markdown_output.write_text(
        _markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
