#!/usr/bin/env python3
"""Combine matched H2, H, and H+ghost SOS-RPA counterpoise energies."""

import argparse
import json
import math
from pathlib import Path
import re
import sys


HARTREE_TO_EV = 27.211396
HARTREE_TO_KCAL_MOL = 627.5094740631


def _parse_abacus_input(path):
    rows = []
    values = {}
    header_seen = False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            rows.append((None, raw))
            continue
        if line == "INPUT_PARAMETERS":
            if header_seen or values:
                raise ValueError(f"{path}: misplaced or duplicate INPUT_PARAMETERS")
            header_seen = True
            rows.append((None, raw))
            continue
        if not header_seen:
            raise ValueError(f"{path}: missing INPUT_PARAMETERS before {line}")
        fields = line.split(None, 1)
        if len(fields) != 2 or fields[0] in values:
            raise ValueError(f"{path}: invalid or duplicate INPUT row: {raw}")
        values[fields[0]] = fields[1].strip()
        rows.append((fields[0], raw))
    if not header_seen:
        raise ValueError(f"{path}: missing INPUT_PARAMETERS")
    return rows, values


def rewrite_truncated_ghost_input(source_path, output_path, suffix):
    rows, source = _parse_abacus_input(source_path)
    source_expected = {
        "nbands": "334",
        "nspin": "2",
        "nupdown": "1",
        "nelec": "1",
        "ecutwfc": "100",
        "rpa": "1",
        "out_librpa_reader_version": "1",
        "exx_pca_threshold": "10",
        "exx_singularity_correction": "massidda",
        "rpa_ccp_rmesh_times": "5",
    }
    for key, expected in source_expected.items():
        if source.get(key) != expected:
            raise ValueError(
                f"source INPUT {key} expected {expected}, got {source.get(key)}"
            )

    result = []
    for key, raw in rows:
        if key == "suffix":
            result.append(f"suffix                  {suffix}")
        elif key == "nbands":
            result.append("nbands                  160")
        else:
            result.append(raw)
    Path(output_path).write_text("\n".join(result) + "\n", encoding="utf-8")

    _, actual = _parse_abacus_input(output_path)
    changed = {key for key in source if source[key] != actual.get(key)}
    if changed != {"suffix", "nbands"}:
        raise ValueError(f"unexpected INPUT changes: {sorted(changed)}")
    expected = dict(source_expected)
    expected["nbands"] = "160"
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"generated INPUT {key} expected {value}, got {actual.get(key)}"
            )
    if actual.get("suffix") != suffix:
        raise ValueError(
            f"generated INPUT suffix expected {suffix}, got {actual.get('suffix')}"
        )
    return actual


def parse_unique_float(path, pattern, label):
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="strict")
    matches = tuple(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected exactly one {label} marker, found {len(matches)}"
        )
    try:
        value = float(matches[0].group("value"))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {label} value") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: {label} must be finite")
    return value


def _abacus_value(path, label):
    pattern = rf"^\s*{re.escape(label)}\s*:\s*(?P<value>[-+0-9.eE]+)\s*$"
    return parse_unique_float(path, pattern, label)


def parse_abacus_energy(abacus_stdout, running_scf):
    exx = _abacus_value(abacus_stdout, "rpa_lcao_exx(Ha)")
    xc = _abacus_value(abacus_stdout, "etxc(Ha)")
    dft_total = _abacus_value(abacus_stdout, "etot(Ha)")
    zero_order = _abacus_value(abacus_stdout, "Etot_without_rpa(Ha)")
    final_ev = parse_unique_float(
        running_scf,
        r"^\s*!FINAL_ETOT_IS\s+(?P<value>[-+0-9.eE]+)\s+eV\s*$",
        "!FINAL_ETOT_IS",
    )

    expected_zero_order = dft_total - xc + exx
    if not math.isclose(zero_order, expected_zero_order, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(
            "ABACUS zero-order identity failed: "
            f"reported {zero_order:.16g}, etot-etxc+exx {expected_zero_order:.16g}"
        )
    expected_final_ev = dft_total * HARTREE_TO_EV
    if not math.isclose(final_ev, expected_final_ev, rel_tol=0.0, abs_tol=5.0e-7):
        raise ValueError(
            "ABACUS SCF total mismatch: "
            f"running_scf {final_ev:.16g} eV, stdout {expected_final_ev:.16g} eV"
        )
    return {
        "dft_total_ha": dft_total,
        "xc_ha": xc,
        "exx_ha": exx,
        "zero_order_ha": zero_order,
    }


def parse_librpa_ec(path):
    return parse_unique_float(
        path,
        r"^\s*\|?\s*Total EcRPA:\s*(?P<value>[-+0-9.eE]+)\s*$",
        "Total EcRPA",
    )


def _require_energy(value, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} energy must be a dictionary")
    required = {"zero_order_ha", "ec_ha"}
    if set(value) != required:
        raise ValueError(f"{name} energy requires exactly {sorted(required)}")
    result = {}
    for key in sorted(required):
        number = float(value[key])
        if not math.isfinite(number):
            raise ValueError(f"{name} {key} must be finite")
        result[key] = number
    return result


def combine_counterpoise(
    h2,
    isolated_h,
    ghost,
    hartree_to_kcal=HARTREE_TO_KCAL_MOL,
):
    h2 = _require_energy(h2, "H2")
    isolated_h = _require_energy(isolated_h, "isolated H")
    ghost = _require_energy(ghost, "H+ghost")
    if not math.isfinite(hartree_to_kcal) or hartree_to_kcal <= 0.0:
        raise ValueError("hartree_to_kcal must be finite and positive")

    values = {}
    for label, key in (
        ("zero_order", "zero_order_ha"),
        ("correlation", "ec_ha"),
    ):
        values[f"raw_{label}"] = 2.0 * isolated_h[key] - h2[key]
        values[f"cp_{label}"] = 2.0 * ghost[key] - h2[key]
    values["raw_total"] = values["raw_zero_order"] + values["raw_correlation"]
    values["cp_total"] = values["cp_zero_order"] + values["cp_correlation"]
    for label in ("zero_order", "correlation", "total"):
        values[f"bsse_{label}"] = values[f"raw_{label}"] - values[f"cp_{label}"]

    result = {}
    for key, value in values.items():
        result[f"{key}_ha"] = value
        result[f"{key}_kcal_mol"] = value * hartree_to_kcal
    return result


def _parse_response(value):
    fields = value.split(":")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "response must be H2_BAND:H2_LOG:H_BAND:H_LOG:GHOST_LOG"
        )
    try:
        h2_band = int(fields[0])
        h_band = int(fields[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("response band counts must be integers") from exc
    if h2_band <= 0 or h_band <= 0:
        raise argparse.ArgumentTypeError("response band counts must be positive")
    return {
        "h2_band": h2_band,
        "h2_librpa": Path(fields[1]),
        "h_band": h_band,
        "h_librpa": Path(fields[3]),
        "ghost_librpa": Path(fields[4]),
    }


def _producer_arguments(parser, prefix):
    parser.add_argument(f"--{prefix}-abacus-stdout", type=Path, required=True)
    parser.add_argument(f"--{prefix}-running-scf", type=Path, required=True)


def _producer_energy(arguments, prefix):
    return parse_abacus_energy(
        getattr(arguments, f"{prefix}_abacus_stdout"),
        getattr(arguments, f"{prefix}_running_scf"),
    )


def _markdown(rows):
    lines = [
        "| H2/ghost bands | H bands | D0 CP | Dc CP | Dtotal CP | Dtotal raw | BSSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = row["binding"]
        lines.append(
            "| {h2_band} | {h_band} | {cp_zero_order:.6f} | "
            "{cp_correlation:.6f} | {cp_total:.6f} | {raw_total:.6f} | "
            "{bsse_total:.6f} |".format(
                h2_band=row["h2_band"],
                h_band=row["h_band"],
                **{
                    key: value[f"{key}_kcal_mol"]
                    for key in (
                        "cp_zero_order",
                        "cp_correlation",
                        "cp_total",
                        "raw_total",
                        "bsse_total",
                    )
                },
            )
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("h2", "h", "ghost"):
        _producer_arguments(parser, prefix)
    parser.add_argument("--response", action="append", type=_parse_response, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    producers = {
        prefix: _producer_energy(arguments, prefix)
        for prefix in ("h2", "h", "ghost")
    }
    rows = []
    seen = set()
    for response in arguments.response:
        key = (response["h2_band"], response["h_band"])
        if key in seen:
            raise ValueError(f"duplicate response cutoff pair {key}")
        seen.add(key)
        h2 = {
            "zero_order_ha": producers["h2"]["zero_order_ha"],
            "ec_ha": parse_librpa_ec(response["h2_librpa"]),
        }
        isolated_h = {
            "zero_order_ha": producers["h"]["zero_order_ha"],
            "ec_ha": parse_librpa_ec(response["h_librpa"]),
        }
        ghost = {
            "zero_order_ha": producers["ghost"]["zero_order_ha"],
            "ec_ha": parse_librpa_ec(response["ghost_librpa"]),
        }
        rows.append(
            {
                "h2_band": response["h2_band"],
                "h_band": response["h_band"],
                "binding": combine_counterpoise(h2, isolated_h, ghost),
            }
        )
    rows.sort(key=lambda value: (value["h2_band"], value["h_band"]))
    payload = {"format_version": 1, "producers": producers, "rows": rows}
    arguments.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments.markdown_output.write_text(_markdown(rows), encoding="utf-8")


def cli(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["prepare-ghost-input"]:
        parser = argparse.ArgumentParser(
            description="Prepare the physical 160-band H+ghost ABACUS input."
        )
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--suffix", required=True)
        arguments = parser.parse_args(argv[1:])
        rewrite_truncated_ghost_input(
            arguments.source,
            arguments.output,
            arguments.suffix,
        )
        return
    main(argv)


if __name__ == "__main__":
    cli()
