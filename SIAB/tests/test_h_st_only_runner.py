import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "example_H_sternheimer/run_st_only.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("h_st_only_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HStOnlyRunnerTest(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def test_build_input_uses_absolute_campaign_files(self):
        template = {
            "seed": 20260718,
            "file_list": {"sternheimer": ["old-target"]},
            "C_init_info": {
                "init_from_file": True,
                "C_init_file": "old-coefficients",
            },
            "loss": {"mode": "st_only"},
        }
        target = Path("/tmp/target.dat")
        coefficients = Path("/tmp/coefficients.dat")

        result = self.runner.build_input(template, target, coefficients)

        self.assertEqual(
            result["file_list"],
            {"sternheimer": [str(target.resolve())]},
        )
        self.assertEqual(
            result["C_init_info"]["C_init_file"], str(coefficients.resolve())
        )
        self.assertEqual(result["seed"], 20260718)
        self.assertEqual(result["loss"]["mode"], "st_only")

    def test_read_coefficient_selects_fixed_level1_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ORBITAL_RESULTS.txt"
            path.write_text(
                "<Coefficient>\n"
                " 2 Total number of radial orbitals.\n"
                " Type L Zeta-Orbital\n"
                " H 0 1\n"
                " 0.25\n"
                " -0.5\n"
                " Type L Zeta-Orbital\n"
                " H 0 2\n"
                " 0.75\n"
                " 1.0\n"
                "</Coefficient>\n",
                encoding="utf-8",
            )

            value = self.runner.read_coefficient(path, "H", 0, 1)

        self.assertEqual(value, (0.25, -0.5))

    def test_summarize_requires_bitwise_fixed_column_and_reports_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sternheimer_matrix.dat"
            initial = root / "initial.txt"
            output = root / "output"
            output.mkdir()
            target.write_text("target\n", encoding="utf-8")
            initial.write_text(self._coefficient_text(0.25), encoding="utf-8")
            (output / "ORBITAL_RESULTS.txt").write_text(
                self._coefficient_text(0.25, loss=0.2), encoding="utf-8"
            )
            (output / "Spillage.dat").write_text(
                "istep_big istep_small istep_all dft_origin dft_dpsi "
                "sternheimer constraint_dft constraint_dpsi total "
                "max_st_condition accepted\n"
                "-1 0 0 0 0 0.5 0 0 0.5 12 true\n"
                "0 1 1 0 0 0.3 0 0 0.3 15 true\n",
                encoding="utf-8",
            )
            (output / "INPUT").write_text(
                json.dumps({"seed": 20260718}), encoding="utf-8"
            )
            (output / "run.log").write_text("Time (PyTorch): 1.25\n")

            report = self.runner.summarize_campaign(
                target=target,
                initial_coefficients=initial,
                output_dir=output,
                elapsed_seconds=1.5,
            )

        self.assertTrue(report["fixed_level1_bitwise_equal"])
        self.assertEqual(report["initial_sternheimer_loss"], 0.5)
        self.assertEqual(report["final_sternheimer_loss"], 0.2)
        self.assertEqual(report["loss_ratio"], 0.4)

    @staticmethod
    def _coefficient_text(value, loss=None):
        metadata = ""
        if loss is not None:
            metadata = (
                "<Mkb>\n"
                f"Left spillage = {loss}\n"
                "Mode = st_only\n"
                "DFT origin loss = 0.0\n"
                "DFT dpsi loss = 0.0\n"
                f"Sternheimer loss = {loss}\n"
                "DFT constraint loss = 0.0\n"
                "dpsi constraint loss = 0.0\n"
                f"Total loss = {loss}\n"
                "</Mkb>\n"
            )
        return (
            "<Coefficient>\n"
            " 1 Total number of radial orbitals.\n"
            " Type L Zeta-Orbital\n"
            " H 0 1\n"
            f" {value}\n"
            "</Coefficient>\n"
            + metadata
        )


if __name__ == "__main__":
    unittest.main()
