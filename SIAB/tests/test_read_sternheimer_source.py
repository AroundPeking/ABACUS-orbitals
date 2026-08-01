from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from IO.read_sternheimer_source import read_sternheimer_source


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "sternheimer_source_v1.dat"
)


def replace_section_body(text, section, transform):
    lines = text.splitlines()
    start = lines.index(f"<{section}>")
    end = lines.index(f"</{section}>")
    body = transform(lines[start + 1 : end])
    return "\n".join(lines[: start + 1] + body + lines[end:]) + "\n"


class ReadSternheimerSourceTest(unittest.TestCase):
    def setUp(self):
        self.fixture_text = FIXTURE.read_text(encoding="utf-8")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def write_variant(self, text):
        path = Path(self.temp_dir.name) / "sternheimer_source.dat"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_version_one_source_data(self):
        data = read_sternheimer_source(FIXTURE)

        self.assertEqual(data.format_version, 1)
        self.assertEqual(data.grid_volume_bohr3, 0.125)
        self.assertEqual(data.d.shape, (2, 4))
        self.assertEqual(data.overlap.shape, (4, 4))
        self.assertEqual(data.blocks[1].key, ("H", 0, 1, 0))
        self.assertEqual(data.occupied_state.tolist(), [0, 0])
        self.assertEqual(data.auxiliary_channel.tolist(), [0, 1])
        self.assertEqual(data.occupation.tolist(), [2.0, 2.0])
        self.assertEqual(data.norm.tolist(), [1.2, 0.8])
        self.assertEqual(data.occupied_state.dtype, torch.int64)
        self.assertEqual(data.auxiliary_channel.dtype, torch.int64)
        self.assertEqual(data.occupation.dtype, torch.float64)
        self.assertEqual(data.norm.dtype, torch.float64)
        self.assertEqual(data.d.dtype, torch.complex128)
        self.assertEqual(data.overlap.dtype, torch.complex128)
        self.assertEqual(data.d.device.type, "cpu")
        self.assertEqual(data.provenance["kernel"], "full_coulomb")

    def test_rejects_unsupported_version(self):
        path = self.write_variant(
            self.fixture_text.replace("format_version 1", "format_version 2")
        )
        with self.assertRaisesRegex(ValueError, "unsupported format_version 2"):
            read_sternheimer_source(path)

    def test_rejects_duplicate_source_key(self):
        path = self.write_variant(
            self.fixture_text.replace("0 1 2.0 0.8", "0 0 2.0 0.8")
        )
        with self.assertRaisesRegex(ValueError, "duplicate source key"):
            read_sternheimer_source(path)

    def test_rejects_negative_source_indices(self):
        for original, replacement, message in (
            ("0 0 2.0 1.2", "-1 0 2.0 1.2", "occupied_state"),
            ("0 0 2.0 1.2", "0 -1 2.0 1.2", "auxiliary_channel"),
        ):
            with self.subTest(field=message):
                path = self.write_variant(
                    self.fixture_text.replace(original, replacement, 1)
                )
                with self.assertRaisesRegex(ValueError, message):
                    read_sternheimer_source(path)

    def test_rejects_nonpositive_norm(self):
        for value in ("0.0", "-0.1"):
            with self.subTest(value=value):
                path = self.write_variant(
                    self.fixture_text.replace("2.0 1.2", f"2.0 {value}", 1)
                )
                with self.assertRaisesRegex(ValueError, "norm.*positive"):
                    read_sternheimer_source(path)

    def test_rejects_nonpositive_occupation(self):
        for value in ("0.0", "-0.1"):
            with self.subTest(value=value):
                path = self.write_variant(
                    self.fixture_text.replace("0 0 2.0 1.2", f"0 0 {value} 1.2", 1)
                )
                with self.assertRaisesRegex(ValueError, "occupation.*positive"):
                    read_sternheimer_source(path)

    def test_rejects_short_overlap_d(self):
        path = self.write_variant(
            replace_section_body(
                self.fixture_text,
                "OVERLAP_D",
                lambda lines: lines[:-1],
            )
        )
        with self.assertRaisesRegex(
            ValueError, "OVERLAP_D expected 8 complex values"
        ):
            read_sternheimer_source(path)

    def test_rejects_nonfinite_overlap_d(self):
        def make_nonfinite(lines):
            lines = list(lines)
            lines[1] = "nan 0.0"
            return lines

        path = self.write_variant(
            replace_section_body(self.fixture_text, "OVERLAP_D", make_nonfinite)
        )
        with self.assertRaisesRegex(ValueError, r"OVERLAP_D\[0\]\.real must be finite"):
            read_sternheimer_source(path)

    def test_direct_data_rejects_nonfinite_overlap_d(self):
        data = read_sternheimer_source(FIXTURE)
        d = data.d.clone()
        d[0, 0] = complex(float("nan"), 0.0)
        with self.assertRaisesRegex(ValueError, "d must contain only finite values"):
            replace(data, d=d)

    def test_rejects_invalid_block_offset(self):
        path = self.write_variant(
            self.fixture_text.replace("H 0 1 0 2 2", "H 0 1 0 2 3")
        )
        with self.assertRaisesRegex(ValueError, r"blocks\[1\]\.offset expected 2"):
            read_sternheimer_source(path)

    def test_rejects_nonhermitian_overlap_s(self):
        def make_nonhermitian(lines):
            lines = list(lines)
            lines[2] = "0.25 0.0"
            return lines

        path = self.write_variant(
            replace_section_body(self.fixture_text, "OVERLAP_S", make_nonhermitian)
        )
        with self.assertRaisesRegex(ValueError, "OVERLAP_S is not Hermitian"):
            read_sternheimer_source(path)

    def test_rejects_missing_required_provenance(self):
        def remove_kernel(lines):
            provenance = json.loads(lines[0])
            del provenance["kernel"]
            return [json.dumps(provenance, separators=(",", ":"))]

        path = self.write_variant(
            replace_section_body(
                self.fixture_text, "PROVENANCE_JSON", remove_kernel
            )
        )
        with self.assertRaisesRegex(ValueError, "missing provenance key: kernel"):
            read_sternheimer_source(path)


if __name__ == "__main__":
    unittest.main()
