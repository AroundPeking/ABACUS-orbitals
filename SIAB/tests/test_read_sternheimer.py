from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from IO.read_sternheimer import read_sternheimer
from sternheimer_data import PrimitiveBlock


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sternheimer_matrix_v1.dat"


def replace_section_body(text, section, transform):
    lines = text.splitlines()
    start = lines.index(f"<{section}>")
    end = lines.index(f"</{section}>")
    body = transform(lines[start + 1 : end])
    return "\n".join(lines[: start + 1] + body + lines[end:]) + "\n"


class ReadSternheimerTest(unittest.TestCase):
    def setUp(self):
        self.fixture_text = FIXTURE.read_text(encoding="utf-8")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def write_variant(self, text):
        path = Path(self.temp_dir.name) / "sternheimer.dat"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_version_one_data(self):
        data = read_sternheimer(FIXTURE)

        self.assertEqual(data.format_version, 1)
        self.assertEqual(data.q.shape, (2, 4))
        self.assertEqual(data.overlap.shape, (4, 4))
        self.assertEqual(data.norm.tolist(), [1.2, 0.8])
        torch.testing.assert_close(
            data.effective_weight,
            torch.tensor([0.6, 1.4], dtype=torch.float64),
        )
        self.assertEqual(data.blocks[1].key, ("H", 0, 1, 0))
        self.assertEqual(data.provenance["kernel"], "full_coulomb")
        self.assertEqual(data.occupied_state.dtype, torch.int64)
        self.assertEqual(data.auxiliary_channel.dtype, torch.int64)
        for field in (
            "frequency_ha",
            "occupation",
            "frequency_weight",
            "norm",
        ):
            self.assertEqual(getattr(data, field).dtype, torch.float64)
        self.assertEqual(data.q.dtype, torch.complex128)
        self.assertEqual(data.overlap.dtype, torch.complex128)

    def test_direct_data_rejects_non_finite_frequency(self):
        data = read_sternheimer(FIXTURE)
        frequency_ha = data.frequency_ha.clone()
        frequency_ha[0] = float("nan")

        with self.assertRaisesRegex(
            ValueError, "frequency_ha must contain only finite values"
        ):
            replace(data, frequency_ha=frequency_ha)

    def test_direct_data_rejects_non_finite_complex_q(self):
        data = read_sternheimer(FIXTURE)
        q = data.q.clone()
        q[0, 0] = complex(float("nan"), 0.0)

        with self.assertRaisesRegex(
            ValueError, "q must contain only finite values"
        ):
            replace(data, q=q)

    def test_rejects_unsupported_version(self):
        path = self.write_variant(
            self.fixture_text.replace("format_version 1", "format_version 2")
        )

        with self.assertRaisesRegex(ValueError, "unsupported format_version 2"):
            read_sternheimer(path)

    def test_rejects_short_overlap_q(self):
        path = self.write_variant(
            replace_section_body(
                self.fixture_text,
                "OVERLAP_Q",
                lambda lines: lines[:-1],
            )
        )

        with self.assertRaisesRegex(
            ValueError, "OVERLAP_Q expected 8 complex values"
        ):
            read_sternheimer(path)

    def test_rejects_non_finite_frequency(self):
        path = self.write_variant(
            self.fixture_text.replace(
                "0 0 0.5 2.0 0.3 1.2",
                "0 0 nan 2.0 0.3 1.2",
                1,
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            r"REFERENCE_METADATA\[0\]\.frequency_ha must be finite",
        ):
            read_sternheimer(path)

    def test_rejects_non_finite_overlap_q(self):
        def make_non_finite(lines):
            lines = list(lines)
            lines[1] = "inf 0.0"
            return lines

        path = self.write_variant(
            replace_section_body(
                self.fixture_text,
                "OVERLAP_Q",
                make_non_finite,
            )
        )

        with self.assertRaisesRegex(
            ValueError, r"OVERLAP_Q\[0\]\.real must be finite"
        ):
            read_sternheimer(path)

    def test_rejects_non_hermitian_overlap_s(self):
        def make_non_hermitian(lines):
            lines = list(lines)
            lines[2] = "0.25 0.0"
            return lines

        path = self.write_variant(
            replace_section_body(
                self.fixture_text,
                "OVERLAP_S",
                make_non_hermitian,
            )
        )

        with self.assertRaisesRegex(ValueError, "OVERLAP_S is not Hermitian"):
            read_sternheimer(path)

    def test_rejects_large_overlap_s_with_small_relative_mismatch(self):
        def make_non_hermitian(lines):
            lines = list(lines)
            lines[2] = "10000000000.0 3.0"
            lines[5] = "10000000000.5 -3.0"
            return lines

        path = self.write_variant(
            replace_section_body(
                self.fixture_text,
                "OVERLAP_S",
                make_non_hermitian,
            )
        )

        with self.assertRaisesRegex(ValueError, "OVERLAP_S is not Hermitian"):
            read_sternheimer(path)

    def test_rejects_missing_kernel_provenance(self):
        def remove_kernel(lines):
            provenance = json.loads(lines[0])
            del provenance["kernel"]
            return [json.dumps(provenance, separators=(",", ":"))]

        path = self.write_variant(
            replace_section_body(
                self.fixture_text,
                "PROVENANCE_JSON",
                remove_kernel,
            )
        )

        with self.assertRaisesRegex(ValueError, "missing provenance key: kernel"):
            read_sternheimer(path)

    def test_rejects_duplicate_provenance_key(self):
        path = self.write_variant(
            self.fixture_text.replace(
                '"kernel":"full_coulomb"',
                '"kernel":"full_coulomb","kernel":"screened"',
                1,
            )
        )

        with self.assertRaisesRegex(
            ValueError, "PROVENANCE_JSON duplicate key: kernel"
        ):
            read_sternheimer(path)

    def test_rejects_non_finite_provenance_constants(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                path = self.write_variant(
                    self.fixture_text.replace(
                        '"ecut_ry":25.0', f'"ecut_ry":{constant}', 1
                    )
                )

                with self.assertRaises(ValueError) as context:
                    read_sternheimer(path)
                self.assertIn(
                    f"PROVENANCE_JSON invalid constant: {constant}",
                    str(context.exception),
                )

    def test_preserves_hash_in_provenance_string(self):
        path = self.write_variant(
            self.fixture_text.replace(
                "occupation_in_metadata", "occupation#in_metadata", 1
            )
        )

        data = read_sternheimer(path)

        self.assertEqual(
            data.provenance["spin_convention"], "occupation#in_metadata"
        )

    def test_rejects_empty_primitive_block_element(self):
        with self.assertRaisesRegex(ValueError, "element must be nonempty"):
            PrimitiveBlock(
                element="",
                atom_index=0,
                l=0,
                m=0,
                n_primitive=1,
                offset=0,
            )

    def test_rejects_negative_atom_index(self):
        path = self.write_variant(
            self.fixture_text.replace(
                "H 0 0 0 2 0",
                "H -1 0 0 2 0",
                1,
            )
        )

        with self.assertRaisesRegex(ValueError, "atom_index must be nonnegative"):
            read_sternheimer(path)

    def test_rejects_negative_angular_momentum(self):
        path = self.write_variant(
            self.fixture_text.replace(
                "H 0 0 0 2 0",
                "H 0 -1 0 2 0",
                1,
            )
        )

        with self.assertRaisesRegex(ValueError, "l must be nonnegative"):
            read_sternheimer(path)

    def test_rejects_m_outside_angular_momentum_range(self):
        path = self.write_variant(
            self.fixture_text.replace(
                "H 0 0 0 2 0",
                "H 0 0 1 2 0",
                1,
            )
        )

        with self.assertRaisesRegex(
            ValueError, "m must satisfy -l <= m <= l"
        ):
            read_sternheimer(path)

    def test_rejects_negative_occupied_state(self):
        path = self.write_variant(
            self.fixture_text.replace(
                "0 0 0.5 2.0 0.3 1.2",
                "-1 0 0.5 2.0 0.3 1.2",
                1,
            )
        )

        with self.assertRaisesRegex(
            ValueError, "occupied_state must contain only nonnegative values"
        ):
            read_sternheimer(path)

    def test_rejects_negative_auxiliary_channel(self):
        path = self.write_variant(
            self.fixture_text.replace(
                "0 0 0.5 2.0 0.3 1.2",
                "0 -1 0.5 2.0 0.3 1.2",
                1,
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "auxiliary_channel must contain only nonnegative values",
        ):
            read_sternheimer(path)

    def test_rejects_mismatched_closing_tag(self):
        path = self.write_variant(
            self.fixture_text.replace("</OVERLAP_Q>", "</OVERLAP_S>", 1)
        )

        with self.assertRaisesRegex(ValueError, "mismatched closing tag"):
            read_sternheimer(path)


if __name__ == "__main__":
    unittest.main()
