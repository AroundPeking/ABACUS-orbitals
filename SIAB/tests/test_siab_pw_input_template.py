import ast
import textwrap
import unittest
from pathlib import Path


SIAB_SOURCE = Path(__file__).resolve().parents[1] / "SIAB.py"


def load_input_generator():
    tree = ast.parse(SIAB_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_input_INPUT"
    )
    namespace = {"textwrap": textwrap}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SIAB_SOURCE), "exec"), namespace)
    return namespace["get_input_INPUT"]


class SiabPwInputTemplateTest(unittest.TestCase):
    def test_spillage_controls_are_inline_for_current_abacus(self):
        generate = load_input_generator()
        input_text = generate(
            "Si-STRU2-8-1.9",
            "/tmp/siab",
            1,
            2,
            10,
            100,
            8,
            0.01,
        )

        self.assertNotIn("wannier_card", input_text)
        self.assertIn("out_spillage         2", input_text)
        self.assertIn(
            "spillage_outdir      OUT.Si-STRU2-8-1.9",
            input_text,
        )


if __name__ == "__main__":
    unittest.main()
