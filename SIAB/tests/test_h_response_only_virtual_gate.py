import importlib.util
import pathlib
import unittest


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "example_H_sternheimer"
    / "delta_st_response_compression"
    / "run_h_response_only_virtual_gate.py"
)


class HResponseOnlyVirtualGateTest(unittest.TestCase):
    def test_provides_reproducible_h_gate_driver(self):
        self.assertTrue(SCRIPT.is_file())

    def test_driver_exposes_main_entrypoint(self):
        spec = importlib.util.spec_from_file_location("h_response_only_gate", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(getattr(module, "main", None)))


if __name__ == "__main__":
    unittest.main()
