#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diamond_qstar_sos_gate import (
    HARTREE_TO_KCAL_MOL,
    QSTAR_REPRESENTATIVES,
    collect_qstar_gate,
    parse_librpa_q_contributions,
    write_gate_artifacts,
)


def _write_librpa_output(path: Path, values: list[complex]) -> None:
    lines = ["| Weighted contribution from each k:"]
    for index, value in enumerate(values):
        lines.append(
            "| ( {0:9.6f}, {1:9.6f}, {2:9.6f}): ({3:.12g},{4:.12g})".format(
                index / 64.0,
                0.0,
                0.0,
                value.real,
                value.imag,
            )
        )
    lines.append(f"| Total EcRPA:       {sum(value.real for value in values):.12g}")
    lines.append("libRPA finished successfully")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class DiamondQstarSosGateTest(unittest.TestCase):
    def test_parses_exactly_64_finite_q_contributions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "librpa.out"
            values = [complex(-0.01 - index * 1.0e-5, 1.0e-16) for index in range(64)]
            _write_librpa_output(output, values)

            parsed = parse_librpa_q_contributions(output)

        self.assertEqual(len(parsed["q_contributions_ha"]), 64)
        self.assertAlmostEqual(parsed["reported_ecrpa_ha"], sum(v.real for v in values))

    def test_rejects_truncated_q_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "librpa.out"
            _write_librpa_output(output, [complex(-0.01, 0.0)] * 63)

            with self.assertRaisesRegex(ValueError, "exactly 64"):
                parse_librpa_q_contributions(output)

    def test_accepts_microhartree_roundoff_in_printed_q_sum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "librpa.out"
            values = [complex(-0.05185222, 0.0)] * 64
            _write_librpa_output(output, values)
            text = output.read_text(encoding="ascii")
            reported = sum(value.real for value in values) + 2.5e-6
            output.write_text(
                text.replace(
                    f"| Total EcRPA:       {sum(value.real for value in values):.12g}",
                    f"| Total EcRPA:       {reported:.12g}",
                ),
                encoding="ascii",
            )

            parsed = parse_librpa_q_contributions(output)

        self.assertAlmostEqual(parsed["reported_ecrpa_ha"], reported)

    def test_rejects_q_sum_mismatch_above_printing_roundoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "librpa.out"
            values = [complex(-0.05185222, 0.0)] * 64
            _write_librpa_output(output, values)
            text = output.read_text(encoding="ascii")
            reported = sum(value.real for value in values) + 6.0e-6
            output.write_text(
                text.replace(
                    f"| Total EcRPA:       {sum(value.real for value in values):.12g}",
                    f"| Total EcRPA:       {reported:.12g}",
                ),
                encoding="ascii",
            )

            with self.assertRaisesRegex(ValueError, "weighted q sum"):
                parse_librpa_q_contributions(output)

    def test_collects_exact_symmetric_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "librpa.out"
            _write_librpa_output(output, [complex(-0.01, 0.0)] * 64)

            result = collect_qstar_gate(
                outputs=[("allband", output)],
                binding_tolerance_kcal_mol_per_c=0.1,
            )

        self.assertEqual(sum(weight for _, weight in QSTAR_REPRESENTATIVES), 64)
        self.assertAlmostEqual(result["rows"][0]["exact_q_sum_ha"], -0.64)
        self.assertAlmostEqual(result["rows"][0]["qstar_reconstruction_ha"], -0.64)
        self.assertEqual(result["qstar_reconstruction_gate"], "pass")

    def test_binding_error_uses_minus_one_half_solid_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "librpa.out"
            values = [complex(-0.01, 0.0)] * 64
            values[63] += 1.0e-3
            _write_librpa_output(output, values)

            result = collect_qstar_gate(
                outputs=[("window", output)],
                binding_tolerance_kcal_mol_per_c=0.1,
            )

        expected = 0.5e-3 * HARTREE_TO_KCAL_MOL
        self.assertAlmostEqual(
            abs(result["rows"][0]["induced_binding_error_kcal_mol_per_c"]),
            expected,
        )
        self.assertEqual(result["qstar_reconstruction_gate"], "fail")

    def test_writes_immutable_result_and_provenance_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "librpa.out"
            _write_librpa_output(output, [complex(-0.01, 0.0)] * 64)
            result = collect_qstar_gate(
                outputs=[("allband", output)],
                binding_tolerance_kcal_mol_per_c=0.1,
            )

            artifact_root = root / "gate"
            write_gate_artifacts(
                output_root=artifact_root,
                result=result,
                provenance={"source_commit": "abc123", "purpose": "unit-test"},
            )

            payload = json.loads((artifact_root / "RESULT.json").read_text(encoding="ascii"))
            status = (artifact_root / "STATUS").read_text(encoding="ascii")
            provenance = (artifact_root / "provenance.txt").read_text(encoding="ascii")
            has_tsv = (artifact_root / "RESULT.tsv").is_file()

        self.assertEqual(payload["qstar_reconstruction_gate"], "pass")
        self.assertEqual(status, "success\n")
        self.assertIn("source_commit abc123\n", provenance)
        self.assertTrue(has_tsv)


if __name__ == "__main__":
    unittest.main()
