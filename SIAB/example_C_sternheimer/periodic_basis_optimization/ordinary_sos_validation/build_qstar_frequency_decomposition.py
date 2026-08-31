#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Optional


QSTAR_MULTIPLICITIES = {
    1: 1,
    2: 8,
    3: 4,
    6: 6,
    7: 24,
    8: 12,
    11: 3,
    28: 6,
}
FREQUENCY_COUNT = 6
Q_COUNT = 64
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex(match, prefix):
    # Named groups inside repeated regex fragments are renamed by the caller.
    return complex(float(match.group(f"{prefix}_real")), float(match.group(f"{prefix}_imag")))


def _compile_patterns():
    number = FLOAT
    vector = rf"(?P<q_text>\((?P<q1>{number}),(?P<q2>{number}),(?P<q3>{number})\))"

    def value(name):
        return rf"\((?P<{name}_real>{number}),(?P<{name}_imag>{number})\)"

    normal = re.compile(
        rf"^RPA normal split ifreq=(?P<ifreq>\d+) freq=(?P<freq_text>{number}) "
        rf"q={vector} trace_pi={value('trace')} logdet={value('logdet')} raw={value('raw')}$"
    )
    freqdiag = re.compile(
        rf"^RPA freqdiag ifreq=(?P<ifreq>\d+) freq=(?P<freq_text>{number}) "
        rf"q={vector} raw={value('raw')} freq_weight=(?P<freq_weight>{number}) "
        rf"qweight=(?P<qweight>{number}) weighted={value('weighted')}$"
    )
    return normal, freqdiag


NORMAL_PATTERN, FREQDIAG_PATTERN = _compile_patterns()


def _parse(path, pattern, kind):
    records = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid {kind} record at {path}:{line_number}")
        record = {
            "ifreq": int(match.group("ifreq")),
            "frequency": float(match.group("freq_text")),
            "frequency_text": match.group("freq_text"),
            "q_text": match.group("q_text"),
            "q": tuple(float(match.group(name)) for name in ("q1", "q2", "q3")),
            "raw": _complex(match, "raw"),
            "line_number": line_number,
        }
        if kind == "normal split":
            record["trace"] = _complex(match, "trace")
            record["logdet"] = _complex(match, "logdet")
        else:
            record["frequency_weight"] = float(match.group("freq_weight"))
            record["qweight"] = float(match.group("qweight"))
            record["weighted"] = _complex(match, "weighted")
        records.append(record)
    return records


def _close(actual: float, expected: float, tolerance: float, label: str):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label}: actual={actual:.17e} expected={expected:.17e}")


def _write_atomic(path: Path, text: str):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite {temporary}")
    temporary.write_text(text, encoding="ascii")
    temporary.replace(path)


def build_decomposition(
    *,
    normal_path: Path,
    freqdiag_path: Path,
    name: str,
    output_tsv: Path,
    output_json: Path,
    expected_sparse_ecrpa: Optional[float] = None,
    expected_star_ecrpa: Optional[float] = None,
    tolerance: float = 5.0e-10,
    star_tolerance: float = 5.0e-6,
    qstar_indices=None,
) -> dict:
    normal_path = Path(normal_path).resolve(strict=True)
    freqdiag_path = Path(freqdiag_path).resolve(strict=True)
    output_tsv = Path(output_tsv).resolve()
    output_json = Path(output_json).resolve()
    if not name or any(character.isspace() for character in name):
        raise ValueError("name must be nonempty and contain no whitespace")
    if output_tsv == output_json:
        raise ValueError("TSV and JSON outputs must differ")
    for output_path in (output_tsv, output_json):
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
    if qstar_indices is None:
        expected_q_indices = list(QSTAR_MULTIPLICITIES)
    else:
        expected_q_indices = [int(value) for value in qstar_indices]
        if (
            not expected_q_indices
            or expected_q_indices != sorted(set(expected_q_indices))
            or any(value not in QSTAR_MULTIPLICITIES for value in expected_q_indices)
        ):
            raise ValueError("qstar_indices must be a nonempty sorted subset of the eight representatives")

    normal = _parse(normal_path, NORMAL_PATTERN, "normal split")
    freqdiag = _parse(freqdiag_path, FREQDIAG_PATTERN, "frequency diagnostic")
    expected_records = Q_COUNT * FREQUENCY_COUNT
    if len(normal) != expected_records or len(freqdiag) != expected_records:
        raise ValueError(
            f"expected {expected_records} records in each input; "
            f"got normal={len(normal)} freqdiag={len(freqdiag)}"
        )

    normal_by_key = {}
    q_order = []
    q_seen = set()
    for record in normal:
        key = (record["q_text"], record["ifreq"])
        if key in normal_by_key:
            raise ValueError(f"duplicate normal split key {key}")
        normal_by_key[key] = record
        if record["q_text"] not in q_seen:
            q_seen.add(record["q_text"])
            q_order.append(record["q_text"])
    if len(q_order) != Q_COUNT:
        raise ValueError(f"expected {Q_COUNT} q vectors, found {len(q_order)}")

    by_q = {q_text: [] for q_text in q_order}
    sparse_total = 0.0
    for record in freqdiag:
        key = (record["q_text"], record["ifreq"])
        reference = normal_by_key.get(key)
        if reference is None:
            raise ValueError(f"frequency diagnostic has no normal split match for {key}")
        _close(record["frequency"], reference["frequency"], tolerance, f"frequency mismatch for {key}")
        _close(record["raw"].real, reference["raw"].real, tolerance, f"raw real mismatch for {key}")
        _close(record["raw"].imag, reference["raw"].imag, tolerance, f"raw imag mismatch for {key}")
        _close(
            (reference["trace"] + reference["logdet"]).real,
            reference["raw"].real,
            tolerance,
            f"trace/logdet real reconstruction for {key}",
        )
        _close(
            (reference["trace"] + reference["logdet"]).imag,
            reference["raw"].imag,
            tolerance,
            f"trace/logdet imag reconstruction for {key}",
        )
        expected_weighted = (
            record["raw"] * record["frequency_weight"] * record["qweight"] / (2.0 * math.pi)
        )
        _close(record["weighted"].real, expected_weighted.real, tolerance, f"weighted real for {key}")
        _close(record["weighted"].imag, expected_weighted.imag, tolerance, f"weighted imag for {key}")
        for label, value in (
            ("trace", reference["trace"]),
            ("logdet", reference["logdet"]),
            ("raw", reference["raw"]),
            ("weighted", record["weighted"]),
        ):
            if abs(value.imag) > 1.0e-10:
                raise ValueError(f"non-negligible imaginary {label} for {key}: {value.imag:.17e}")
        by_q[record["q_text"]].append((reference, record))
        sparse_total += record["weighted"].real

    selected_q_indices = []
    for ordinal, q_text in enumerate(q_order, start=1):
        records = by_q[q_text]
        if sorted(record[1]["ifreq"] for record in records) != list(range(FREQUENCY_COUNT)):
            raise ValueError(f"incomplete frequencies for q index {ordinal}")
        if any(abs(record[1]["raw"]) > 1.0e-18 for record in records):
            selected_q_indices.append(ordinal)
    if selected_q_indices != expected_q_indices:
        raise ValueError(
            f"nonzero q indices are {selected_q_indices}; expected {expected_q_indices}"
        )

    lines = []
    q_metrics = {}
    star_total = 0.0
    for q_index in selected_q_indices:
        q_text = q_order[q_index - 1]
        multiplicity = QSTAR_MULTIPLICITIES[q_index]
        values = sorted(by_q[q_text], key=lambda pair: pair[1]["ifreq"])
        weighted_raw_values = []
        for reference, record in values:
            factor = record["frequency_weight"] * record["qweight"] * multiplicity / (2.0 * math.pi)
            weighted_trace = reference["trace"].real * factor
            weighted_logdet = reference["logdet"].real * factor
            weighted_raw = reference["raw"].real * factor
            _close(
                weighted_raw,
                record["weighted"].real * multiplicity,
                tolerance,
                f"star-weighted raw for q{q_index} frequency {record['ifreq']}",
            )
            lines.append(
                "\t".join(
                    (
                        "RECORD",
                        name,
                        str(q_index),
                        str(multiplicity),
                        str(record["ifreq"]),
                        record["frequency_text"],
                        f"q={q_text}",
                        f"{reference['trace'].real:.17e}",
                        f"{reference['logdet'].real:.17e}",
                        f"{reference['raw'].real:.17e}",
                        f"{weighted_trace:.17e}",
                        f"{weighted_logdet:.17e}",
                        f"{weighted_raw:.17e}",
                    )
                )
            )
            weighted_raw_values.append(weighted_raw)
            star_total += weighted_raw
        highest_reference, _ = values[-1]
        trace_scale = abs(highest_reference["trace"].real)
        absolute_sum = sum(abs(value) for value in weighted_raw_values)
        q_metrics[f"q{q_index}"] = {
            "multiplicity": multiplicity,
            "q": list(highest_reference["q"]),
            "star_contribution_ha": sum(weighted_raw_values),
            "highest_frequency_cancellation_ratio": (
                abs(highest_reference["raw"].real) / trace_scale if trace_scale else math.inf
            ),
            "high_frequency_tail_fraction": (
                abs(weighted_raw_values[-1]) / absolute_sum if absolute_sum else math.inf
            ),
            "highest_frequency_trace": highest_reference["trace"].real,
            "highest_frequency_logdet": highest_reference["logdet"].real,
            "highest_frequency_raw": highest_reference["raw"].real,
        }

    if expected_sparse_ecrpa is not None:
        _close(sparse_total, expected_sparse_ecrpa, tolerance, "sparse EcRPA reconstruction")
    if expected_star_ecrpa is not None:
        _close(star_total, expected_star_ecrpa, star_tolerance, "q-star EcRPA reconstruction")

    tsv_text = "\n".join(lines) + "\n"
    _write_atomic(output_tsv, tsv_text)
    result = {
        "status": "success",
        "name": name,
        "scope": (
            "full_qstar_reconstruction"
            if selected_q_indices == list(QSTAR_MULTIPLICITIES)
            else "partial_qstar_screen"
        ),
        "normal_path": str(normal_path),
        "normal_sha256": _sha256(normal_path),
        "freqdiag_path": str(freqdiag_path),
        "freqdiag_sha256": _sha256(freqdiag_path),
        "decomposition_path": str(output_tsv),
        "decomposition_sha256": _sha256(output_tsv),
        "normal_record_count": len(normal),
        "freqdiag_record_count": len(freqdiag),
        "decomposition_record_count": len(lines),
        "q_count": len(q_order),
        "frequency_count": FREQUENCY_COUNT,
        "selected_q_indices": selected_q_indices,
        "qstar_multiplicity_sum": sum(
            QSTAR_MULTIPLICITIES[index] for index in selected_q_indices
        ),
        "sparse_ecrpa_ha": sparse_total,
        "weighted_selected_ecrpa_ha": star_total,
        "star_reconstructed_ecrpa_ha": star_total,
        "q_metrics": q_metrics,
    }
    if expected_sparse_ecrpa is not None:
        result["expected_sparse_ecrpa_ha"] = expected_sparse_ecrpa
        result["sparse_reconstruction_difference_ha"] = sparse_total - expected_sparse_ecrpa
    if expected_star_ecrpa is not None:
        result["expected_printed_qstar_ecrpa_ha"] = expected_star_ecrpa
        result["printed_qstar_difference_ha"] = star_total - expected_star_ecrpa
        result["printed_qstar_tolerance_ha"] = star_tolerance
    _write_atomic(
        output_json,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build a strict 4x4x4 eight-q-star frequency decomposition from LibRPA debug output."
    )
    parser.add_argument("--normal-split", type=Path, required=True)
    parser.add_argument("--freqdiag", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-sparse-ecrpa", type=float)
    parser.add_argument("--expected-star-ecrpa", type=float)
    parser.add_argument("--tolerance", type=float, default=5.0e-10)
    parser.add_argument("--star-tolerance", type=float, default=5.0e-6)
    parser.add_argument(
        "--qstar-indices",
        default=",".join(str(value) for value in QSTAR_MULTIPLICITIES),
        help="comma-separated sorted subset of 1,2,3,6,7,8,11,28",
    )
    args = parser.parse_args()
    qstar_indices = tuple(int(value) for value in args.qstar_indices.split(",") if value)
    result = build_decomposition(
        normal_path=args.normal_split,
        freqdiag_path=args.freqdiag,
        name=args.name,
        output_tsv=args.output_tsv,
        output_json=args.output_json,
        expected_sparse_ecrpa=args.expected_sparse_ecrpa,
        expected_star_ecrpa=args.expected_star_ecrpa,
        tolerance=args.tolerance,
        star_tolerance=args.star_tolerance,
        qstar_indices=qstar_indices,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
