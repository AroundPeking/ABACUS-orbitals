#!/usr/bin/env python3
"""Reconstruct diamond 4x4x4 solid EcRPA from eight q-star representatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path


HARTREE_TO_KCAL_MOL = 627.5094740631
QSTAR_REPRESENTATIVES = (
    (1, 1),
    (2, 8),
    (3, 4),
    (6, 6),
    (7, 24),
    (8, 12),
    (11, 3),
    (28, 6),
)
EXPECTED_Q_COUNT = 64
MAX_IMAGINARY_HA = 1.0e-10
PRINTED_SUM_TOLERANCE_HA = 1.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_librpa_q_contributions(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    text = path.read_text(encoding="utf-8", errors="strict")
    if "libRPA finished successfully" not in text:
        raise ValueError(f"missing LibRPA success marker: {path}")

    header = "| Weighted contribution from each k:"
    if text.count(header) != 1:
        raise ValueError(f"expected one weighted-q block: {path}")
    block = text.split(header, maxsplit=1)[1]
    total_match = re.search(
        r"^\| Total EcRPA:\s+([-+0-9.eE]+)\s*$",
        block,
        flags=re.MULTILINE,
    )
    if total_match is None:
        raise ValueError(f"missing Total EcRPA after weighted-q block: {path}")
    q_block = block[: total_match.start()]
    q_pattern = re.compile(
        r"^\|\s*\([^)]*\):\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*$",
        flags=re.MULTILINE,
    )
    values = [complex(float(real), float(imag)) for real, imag in q_pattern.findall(q_block)]
    if len(values) != EXPECTED_Q_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_Q_COUNT} weighted q contributions, got {len(values)}: {path}"
        )
    if not all(math.isfinite(value.real) and math.isfinite(value.imag) for value in values):
        raise ValueError(f"non-finite weighted q contribution: {path}")
    max_imaginary = max(abs(value.imag) for value in values)
    if max_imaginary > MAX_IMAGINARY_HA:
        raise ValueError(f"weighted q contribution has a large imaginary part: {path}")

    reported = float(total_match.group(1))
    exact_sum = sum(value.real for value in values)
    if not math.isfinite(reported):
        raise ValueError(f"non-finite Total EcRPA: {path}")
    if not math.isclose(
        exact_sum,
        reported,
        rel_tol=0.0,
        abs_tol=PRINTED_SUM_TOLERANCE_HA,
    ):
        raise ValueError(
            f"weighted q sum {exact_sum:.12g} disagrees with Total EcRPA {reported:.12g}: {path}"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "reported_ecrpa_ha": reported,
        "exact_q_sum_ha": exact_sum,
        "max_abs_imaginary_ha": max_imaginary,
        "q_contributions_ha": values,
    }


def collect_qstar_gate(
    *,
    outputs: list[tuple[str, Path]],
    binding_tolerance_kcal_mol_per_c: float,
) -> dict:
    tolerance = float(binding_tolerance_kcal_mol_per_c)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("binding tolerance must be finite and positive")
    labels = [label for label, _ in outputs]
    if not labels or len(labels) != len(set(labels)) or any(not label.strip() for label in labels):
        raise ValueError("output labels must be nonempty and unique")
    if sum(weight for _, weight in QSTAR_REPRESENTATIVES) != EXPECTED_Q_COUNT:
        raise AssertionError("q-star multiplicities do not cover the 4x4x4 grid")

    rows = []
    for label, path in outputs:
        parsed = parse_librpa_q_contributions(path)
        values = parsed.pop("q_contributions_ha")
        reconstruction = sum(
            values[index - 1].real * weight
            for index, weight in QSTAR_REPRESENTATIVES
        )
        error_ha = reconstruction - parsed["exact_q_sum_ha"]
        induced_binding_error = -0.5 * error_ha * HARTREE_TO_KCAL_MOL
        rows.append(
            {
                "label": label,
                **parsed,
                "qstar_reconstruction_ha": reconstruction,
                "reconstruction_error_ha": error_ha,
                "induced_binding_error_kcal_mol_per_c": induced_binding_error,
            }
        )

    maximum_error = max(
        abs(row["induced_binding_error_kcal_mol_per_c"])
        for row in rows
    )
    return {
        "status": "success",
        "quantity": "diamond_4x4x4_qstar_ecrpa_reconstruction",
        "q_count": EXPECTED_Q_COUNT,
        "qstar_representatives_one_based": [
            {"q_index": index, "multiplicity": weight}
            for index, weight in QSTAR_REPRESENTATIVES
        ],
        "binding_tolerance_kcal_mol_per_c": tolerance,
        "maximum_abs_induced_binding_error_kcal_mol_per_c": maximum_error,
        "qstar_reconstruction_gate": "pass" if maximum_error < tolerance else "fail",
        "rows": rows,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_gate_artifacts(
    *,
    output_root: Path,
    result: dict,
    provenance: dict[str, str],
) -> None:
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    _atomic_write_text(
        output_root / "RESULT.json",
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    header = (
        "label\treported_ecrpa_ha\texact_q_sum_ha\tqstar_reconstruction_ha\t"
        "reconstruction_error_ha\tinduced_binding_error_kcal_mol_per_c\tsha256\tpath"
    )
    lines = [header]
    for row in result["rows"]:
        lines.append(
            "\t".join(
                [
                    str(row["label"]),
                    f'{row["reported_ecrpa_ha"]:.17g}',
                    f'{row["exact_q_sum_ha"]:.17g}',
                    f'{row["qstar_reconstruction_ha"]:.17g}',
                    f'{row["reconstruction_error_ha"]:.17g}',
                    f'{row["induced_binding_error_kcal_mol_per_c"]:.17g}',
                    str(row["sha256"]),
                    str(row["path"]),
                ]
            )
        )
    _atomic_write_text(output_root / "RESULT.tsv", "\n".join(lines) + "\n")

    provenance_lines = [f"{key} {value}" for key, value in sorted(provenance.items())]
    for row in result["rows"]:
        provenance_lines.append(f'input_{row["label"]}_sha256 {row["sha256"]}')
        provenance_lines.append(f'input_{row["label"]}_path {row["path"]}')
    _atomic_write_text(output_root / "provenance.txt", "\n".join(provenance_lines) + "\n")
    _atomic_write_text(output_root / "STATUS", "success\n")


def _labelled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def _key_value(value: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key or not item:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    return key, item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solid-output", action="append", type=_labelled_path, required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--binding-tolerance-kcal-mol-per-c", type=float, default=0.1)
    parser.add_argument("--provenance", action="append", type=_key_value, default=[])
    args = parser.parse_args()

    result = collect_qstar_gate(
        outputs=args.solid_output,
        binding_tolerance_kcal_mol_per_c=args.binding_tolerance_kcal_mol_per_c,
    )
    write_gate_artifacts(
        output_root=args.output_root,
        result=result,
        provenance=dict(args.provenance),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
