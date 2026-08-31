#!/usr/bin/env python3

import json
import math
import tempfile
import unittest
from pathlib import Path

from build_qstar_frequency_decomposition import build_decomposition
from c_sos_trust_region import parse_frequency_decomposition


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


class QStarFrequencyDecompositionTest(unittest.TestCase):
    def _write_inputs(self, directory: Path):
        normal_path = directory / "normal_split.dat"
        freqdiag_path = directory / "freqdiag.dat"
        normal_lines = []
        freqdiag_lines = []
        sparse_total = 0.0
        star_total = 0.0
        for q_index in range(1, 65):
            q_text = f"({q_index:.17e},0.00000000000000000e+00,0.00000000000000000e+00)"
            multiplicity = QSTAR_MULTIPLICITIES.get(q_index, 0)
            for ifreq in range(6):
                frequency = 0.25 * (ifreq + 1)
                trace = -10.0 / (ifreq + 1) if multiplicity else 0.0
                raw = -(q_index + ifreq + 1) * 1.0e-3 if multiplicity else 0.0
                logdet = raw - trace
                frequency_weight = 0.1 * (ifreq + 1)
                qweight = 1.0 / 64.0
                weighted = raw * frequency_weight * qweight / (2.0 * math.pi)
                sparse_total += weighted
                star_total += multiplicity * weighted
                normal_lines.append(
                    "RPA normal split "
                    f"ifreq={ifreq} freq={frequency:.17e} q={q_text} "
                    f"trace_pi=({trace:.17e},0.00000000000000000e+00) "
                    f"logdet=({logdet:.17e},0.00000000000000000e+00) "
                    f"raw=({raw:.17e},0.00000000000000000e+00)"
                )
                freqdiag_lines.append(
                    "RPA freqdiag "
                    f"ifreq={ifreq} freq={frequency:.17e} q={q_text} "
                    f"raw=({raw:.17e},0.00000000000000000e+00) "
                    f"freq_weight={frequency_weight:.17e} qweight={qweight:.17e} "
                    f"weighted=({weighted:.17e},0.00000000000000000e+00)"
                )
        normal_path.write_text("\n".join(normal_lines) + "\n", encoding="ascii")
        freqdiag_path.write_text("\n".join(freqdiag_lines) + "\n", encoding="ascii")
        return normal_path, freqdiag_path, sparse_total, star_total

    def test_builds_strict_eight_qstar_decomposition(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            normal_path, freqdiag_path, sparse_total, star_total = self._write_inputs(directory)
            output_tsv = directory / "decomposition.tsv"
            output_json = directory / "summary.json"

            result = build_decomposition(
                normal_path=normal_path,
                freqdiag_path=freqdiag_path,
                name="candidate",
                output_tsv=output_tsv,
                output_json=output_json,
                expected_sparse_ecrpa=sparse_total,
                expected_star_ecrpa=star_total,
                tolerance=5.0e-12,
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["normal_record_count"], 384)
            self.assertEqual(result["freqdiag_record_count"], 384)
            self.assertEqual(result["selected_q_indices"], list(QSTAR_MULTIPLICITIES))
            self.assertEqual(result["decomposition_record_count"], 48)
            self.assertAlmostEqual(result["sparse_ecrpa_ha"], sparse_total)
            self.assertAlmostEqual(result["star_reconstructed_ecrpa_ha"], star_total)
            self.assertEqual(len(output_tsv.read_text(encoding="ascii").splitlines()), 48)
            self.assertEqual(json.loads(output_json.read_text(encoding="ascii"))["status"], "success")
            self.assertIn("q2", result["q_metrics"])
            self.assertIn("high_frequency_tail_fraction", result["q_metrics"]["q6"])
            trust_region_metrics = parse_frequency_decomposition(output_tsv)
            self.assertAlmostEqual(
                trust_region_metrics["q2"]["highest_frequency_cancellation_ratio"],
                result["q_metrics"]["q2"]["highest_frequency_cancellation_ratio"],
            )
            self.assertAlmostEqual(
                trust_region_metrics["q6"]["high_frequency_tail_fraction"],
                result["q_metrics"]["q6"]["high_frequency_tail_fraction"],
            )

    def test_rejects_unexpected_nonzero_q(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            normal_path, freqdiag_path, _, _ = self._write_inputs(directory)
            marker = "q=(4.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00)"
            normal_text = normal_path.read_text(encoding="ascii").replace(
                marker
                + " trace_pi=(0.00000000000000000e+00,0.00000000000000000e+00)"
                + " logdet=(0.00000000000000000e+00,0.00000000000000000e+00)"
                + " raw=(0.00000000000000000e+00,0.00000000000000000e+00)",
                marker
                + " trace_pi=(0.00000000000000000e+00,0.00000000000000000e+00)"
                + " logdet=(-1.00000000000000002e-03,0.00000000000000000e+00)"
                + " raw=(-1.00000000000000002e-03,0.00000000000000000e+00)",
                1,
            )
            normal_path.write_text(normal_text, encoding="ascii")
            weighted = -1.0e-3 * 0.1 / 64.0 / (2.0 * math.pi)
            freqdiag_text = freqdiag_path.read_text(encoding="ascii").replace(
                marker
                + " raw=(0.00000000000000000e+00,0.00000000000000000e+00)"
                + " freq_weight=1.00000000000000006e-01 qweight=1.56250000000000000e-02"
                + " weighted=(0.00000000000000000e+00,0.00000000000000000e+00)",
                marker
                + " raw=(-1.00000000000000002e-03,0.00000000000000000e+00)"
                + " freq_weight=1.00000000000000006e-01 qweight=1.56250000000000000e-02"
                + f" weighted=({weighted:.17e},0.00000000000000000e+00)",
                1,
            )
            freqdiag_path.write_text(freqdiag_text, encoding="ascii")

            with self.assertRaisesRegex(ValueError, "nonzero q indices"):
                build_decomposition(
                    normal_path=normal_path,
                    freqdiag_path=freqdiag_path,
                    name="bad",
                    output_tsv=directory / "decomposition.tsv",
                    output_json=directory / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()
