import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "extract_sternheimer_frequency_subset.py"
)
SPEC = importlib.util.spec_from_file_location(
    "extract_sternheimer_frequency_subset", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIXTURE = """<STERNHEIMER_SIAB_HEADER>
format_version 1
n_reference 4
n_primitive 2
n_blocks 1
grid_volume_bohr3 1.0
</STERNHEIMER_SIAB_HEADER>
<PRIMITIVE_BLOCKS>
# element atom_index l m n_primitive offset
C 0 0 0 2 0
</PRIMITIVE_BLOCKS>
<REFERENCE_METADATA>
# occupied_state auxiliary_channel frequency_ha occupation frequency_weight norm
0 0 0.1 1.0 0.2 1.0
0 0 0.4 1.0 0.5 1.0
0 1 0.1 1.0 0.2 1.0
0 1 0.4 1.0 0.5 1.0
</REFERENCE_METADATA>
<OVERLAP_Q>
# row-major
1 0
2 0
3 0
4 0
5 0
6 0
7 0
8 0
</OVERLAP_Q>
<OVERLAP_S>
# row-major
1 0
0 0
0 0
1 0
</OVERLAP_S>
<PROVENANCE_JSON>
{"frequency_ha":[0.1,0.4]}
</PROVENANCE_JSON>
"""


class ExtractSternheimerFrequencySubsetTest(unittest.TestCase):
    def test_extracts_one_frequency_without_reordering_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "full.dat"
            output = root / "subset.dat"
            source.write_text(FIXTURE, encoding="ascii")

            report = MODULE.extract_frequency_subset(
                source,
                output,
                frequency_index=0,
            )

            self.assertEqual(report["selected_frequency_ha"], 0.1)
            self.assertEqual(report["selected_rows"], [0, 2])
            text = output.read_text(encoding="ascii")
            self.assertIn("n_reference 2\n", text)
            self.assertIn("0 0 0.1 1.0 0.2 1.0\n", text)
            self.assertIn("0 1 0.1 1.0 0.2 1.0\n", text)
            self.assertNotIn("0 0 0.4 1.0 0.5 1.0\n", text)
            self.assertIn("1 0\n2 0\n5 0\n6 0\n", text)
            self.assertNotIn("3 0\n4 0\n", text)
            self.assertIn('{"frequency_ha":[0.1,0.4]}', text)

    def test_refuses_existing_output_and_bad_frequency_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "full.dat"
            output = root / "subset.dat"
            source.write_text(FIXTURE, encoding="ascii")
            output.write_text("occupied\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "output already exists"):
                MODULE.extract_frequency_subset(source, output, frequency_index=0)
            output.unlink()
            with self.assertRaisesRegex(ValueError, "frequency index"):
                MODULE.extract_frequency_subset(source, output, frequency_index=2)


if __name__ == "__main__":
    unittest.main()
