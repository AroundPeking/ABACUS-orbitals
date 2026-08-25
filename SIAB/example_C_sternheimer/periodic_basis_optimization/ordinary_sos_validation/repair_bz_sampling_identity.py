#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair ABACUS reader-v1 BZ labels when every stored irreducible q "
            "has its own emitted Coulomb data."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    source = arguments.source.resolve()
    output = arguments.output.resolve()
    if source == output:
        raise ValueError("source and output must be different files")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)

    lines = source.read_text(encoding="ascii").splitlines()
    if len(lines) < 2:
        raise ValueError("BZ sampling file must contain two header lines")
    mesh = lines[0].split()
    counts = lines[1].split()
    if len(mesh) != 3 or any(int(value) <= 0 for value in mesh):
        raise ValueError("invalid BZ mesh header")
    if len(counts) != 2:
        raise ValueError("invalid irreducible-count header")
    stored_count, coulomb_count = map(int, counts)
    if stored_count <= 0 or stored_count != coulomb_count:
        raise ValueError("stored and Coulomb irreducible counts must match")
    rows = [line.split() for line in lines[2:]]
    if len(rows) != stored_count:
        raise ValueError(
            f"expected {stored_count} BZ rows but found {len(rows)}"
        )

    original_labels = []
    repaired_rows = []
    for index, row in enumerate(rows, start=1):
        if len(row) < 10:
            raise ValueError(f"BZ row {index} has too few fields")
        if int(row[0]) != index:
            raise ValueError(f"BZ row id {row[0]} is not the expected {index}")
        original_labels.append((int(row[-2]), int(row[-1])))
        repaired_rows.append(row[:-2] + [str(index), str(index)])

    output.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n".join(
        lines[:2] + [" ".join(row) for row in repaired_rows]
    ) + "\n"
    output.write_text(output_text, encoding="ascii")

    reread_rows = [line.split() for line in output.read_text(encoding="ascii").splitlines()[2:]]
    nonlabel_fields_unchanged = all(
        original[:-2] == repaired[:-2]
        for original, repaired in zip(rows, reread_rows)
    )
    repaired_labels = [(int(row[-2]), int(row[-1])) for row in reread_rows]
    if not nonlabel_fields_unchanged:
        raise RuntimeError("non-label BZ fields changed during repair")
    if repaired_labels != [(index, index) for index in range(1, stored_count + 1)]:
        raise RuntimeError("repaired BZ labels are not an identity mapping")

    report = {
        "status": "success",
        "mesh": [int(value) for value in mesh],
        "irreducible_count": stored_count,
        "original_unique_reader_labels": len(set(original_labels)),
        "repaired_unique_reader_labels": len(set(repaired_labels)),
        "nonlabel_fields_unchanged": nonlabel_fields_unchanged,
        "source_sha256": file_sha256(source),
        "output_sha256": file_sha256(output),
    }
    if arguments.report is not None:
        report_path = arguments.report.resolve()
        if report_path.exists():
            raise FileExistsError(report_path)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
