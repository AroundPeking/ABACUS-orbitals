#!/usr/bin/env python3
"""Validate an eight-representative sparse-q LibRPA SOS reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from diamond_qstar_sos_gate import (
    EXPECTED_Q_COUNT,
    HARTREE_TO_KCAL_MOL,
    QSTAR_REPRESENTATIVES,
    parse_librpa_q_contributions,
)


INACTIVE_Q_TOLERANCE_HA = 1.0e-12


def collect_sparse_qstar_gate(
    *,
    outputs: list[tuple[str, Path]],
    references: dict[str, Path],
    binding_tolerance_kcal_mol_per_c: float,
) -> dict:
    tolerance = float(binding_tolerance_kcal_mol_per_c)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("binding tolerance must be finite and positive")
    labels = [label for label, _ in outputs]
    if not labels or len(labels) != len(set(labels)) or any(not label.strip() for label in labels):
        raise ValueError("output labels must be nonempty and unique")
    if references and set(references) != set(labels):
        raise ValueError("reference labels must exactly match sparse output labels")

    representative_indices = {index - 1 for index, _ in QSTAR_REPRESENTATIVES}
    if sum(weight for _, weight in QSTAR_REPRESENTATIVES) != EXPECTED_Q_COUNT:
        raise AssertionError("q-star multiplicities do not cover the 4x4x4 grid")

    rows = []
    for label, path in outputs:
        sparse = parse_librpa_q_contributions(path)
        values = sparse.pop("q_contributions_ha")
        inactive_maximum = max(
            abs(value)
            for index, value in enumerate(values)
            if index not in representative_indices
        )
        if inactive_maximum > INACTIVE_Q_TOLERANCE_HA:
            raise ValueError(
                f"sparse output contains a nonzero inactive q contribution: {path}"
            )
        reconstruction = sum(
            values[index - 1].real * multiplicity
            for index, multiplicity in QSTAR_REPRESENTATIVES
        )
        row = {
            "label": label,
            **sparse,
            "maximum_abs_inactive_q_contribution_ha": inactive_maximum,
            "qstar_reconstruction_ha": reconstruction,
        }
        if label in references:
            reference = parse_librpa_q_contributions(references[label])
            reference.pop("q_contributions_ha")
            difference = reconstruction - reference["exact_q_sum_ha"]
            induced_binding_error = -0.5 * difference * HARTREE_TO_KCAL_MOL
            row.update(
                {
                    "reference_path": reference["path"],
                    "reference_sha256": reference["sha256"],
                    "reference_ecrpa_ha": reference["exact_q_sum_ha"],
                    "reconstruction_error_ha": difference,
                    "induced_binding_error_kcal_mol_per_c": induced_binding_error,
                }
            )
        rows.append(row)

    maximum_error = max(
        (
            abs(row["induced_binding_error_kcal_mol_per_c"])
            for row in rows
            if "induced_binding_error_kcal_mol_per_c" in row
        ),
        default=0.0,
    )
    return {
        "status": "success",
        "quantity": "diamond_4x4x4_sparse_qstar_ecrpa_reconstruction",
        "q_count": EXPECTED_Q_COUNT,
        "qstar_representatives_one_based": [
            {"q_index": index, "multiplicity": weight}
            for index, weight in QSTAR_REPRESENTATIVES
        ],
        "inactive_q_tolerance_ha": INACTIVE_Q_TOLERANCE_HA,
        "binding_tolerance_kcal_mol_per_c": tolerance,
        "maximum_abs_induced_binding_error_kcal_mol_per_c": maximum_error,
        "sparse_qstar_gate": "pass" if maximum_error < tolerance else "fail",
        "rows": rows,
    }


def _atomic_write(path: Path, text: str) -> None:
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


def write_artifacts(output_root: Path, result: dict, provenance: dict[str, str]) -> None:
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    _atomic_write(
        output_root / "RESULT.json",
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    lines = [
        "label\tsparse_total_ha\tqstar_reconstruction_ha\treference_ecrpa_ha\t"
        "induced_binding_error_kcal_mol_per_c"
    ]
    for row in result["rows"]:
        lines.append(
            "\t".join(
                (
                    row["label"],
                    f'{row["exact_q_sum_ha"]:.17g}',
                    f'{row["qstar_reconstruction_ha"]:.17g}',
                    f'{row.get("reference_ecrpa_ha", float("nan")):.17g}',
                    f'{row.get("induced_binding_error_kcal_mol_per_c", float("nan")):.17g}',
                )
            )
        )
    _atomic_write(output_root / "RESULT.tsv", "\n".join(lines) + "\n")
    provenance_lines = [f"{key} {value}" for key, value in sorted(provenance.items())]
    for row in result["rows"]:
        provenance_lines.append(f'input_{row["label"]}_sha256 {row["sha256"]}')
        if "reference_sha256" in row:
            provenance_lines.append(
                f'reference_{row["label"]}_sha256 {row["reference_sha256"]}'
            )
    _atomic_write(output_root / "provenance.txt", "\n".join(provenance_lines) + "\n")
    _atomic_write(output_root / "STATUS", "success\n")


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
    parser.add_argument("--sparse-output", action="append", type=_labelled_path, required=True)
    parser.add_argument("--reference-output", action="append", type=_labelled_path, default=[])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--binding-tolerance-kcal-mol-per-c", type=float, default=0.1)
    parser.add_argument("--provenance", action="append", type=_key_value, default=[])
    args = parser.parse_args()
    references = dict(args.reference_output)
    if len(references) != len(args.reference_output):
        raise ValueError("reference labels must be unique")
    result = collect_sparse_qstar_gate(
        outputs=args.sparse_output,
        references=references,
        binding_tolerance_kcal_mol_per_c=args.binding_tolerance_kcal_mol_per_c,
    )
    write_artifacts(args.output_root, result, dict(args.provenance))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
