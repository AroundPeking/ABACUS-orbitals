import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gate_contract import render_input


SHARED_PROTOCOL = {
    "suffix": "C_PBE_REFERENCE_GATE",
    "calculation": "scf",
    "ntype": "1",
    "nelec": "4",
    "nspin": "2",
    "nupdown": "2",
    "nbands": "22",
    "basis_type": "lcao",
    "ecutwfc": "30",
    "lcao_ecut": "100",
    "nx": "135",
    "ny": "135",
    "nz": "135",
    "ks_solver": "genelpa",
    "dft_functional": "pbe",
    "symmetry": "0",
    "kpar": "1",
    "pseudo_dir": "./",
    "orbital_dir": "./",
    "scf_thr": "1e-10",
    "scf_nmax": "300",
    "mixing_type": "broyden",
    "mixing_beta": "0.3",
    "mixing_beta_mag": "0.3",
    "smearing_method": "fixed",
    "out_chg": "1",
    "out_wfc_lcao": "2",
    "out_mul": "1",
}


def parse_input(text):
    lines = text.splitlines()
    if not lines or lines[0] != "INPUT_PARAMETERS":
        raise AssertionError("rendered input must start with INPUT_PARAMETERS")

    values = {}
    for line in lines[1:]:
        key, value = line.split(maxsplit=1)
        if key in values:
            raise AssertionError(f"duplicate input key: {key}")
        values[key] = value
    return values


class InputContractTests(unittest.TestCase):
    def test_shared_protocol_is_frozen_in_every_mode(self):
        cases = (
            {"mode": "fixed", "restart": False},
            {"mode": "field", "field_dir": 1, "restart": False},
            {"mode": "free", "field_dir": 2, "restart": True},
        )

        for case in cases:
            with self.subTest(case=case):
                values = parse_input(render_input(**case))
                self.assertEqual(
                    {key: values[key] for key in SHARED_PROTOCOL},
                    SHARED_PROTOCOL,
                )

    def test_fixed_zero_field_contract(self):
        values = parse_input(render_input(mode="fixed", restart=False))

        self.assertEqual(values["nspin"], "2")
        self.assertEqual(values["nelec"], "4")
        self.assertEqual(values["nupdown"], "2")
        self.assertEqual(values["ocp"], "1")
        self.assertEqual(values["ocp_set"], "3*1 19*0 1*1 21*0")
        self.assertEqual(values["efield_flag"], "0")
        self.assertEqual(values["efield_amp"], "0")
        self.assertNotIn("init_wfc", values)
        self.assertNotIn("init_chg", values)

    def test_field_seed_is_not_fixed(self):
        values = parse_input(
            render_input(mode="field", field_dir=1, restart=False)
        )

        self.assertEqual(values["ocp"], "0")
        self.assertNotIn("ocp_set", values)
        self.assertEqual(values["efield_flag"], "1")
        self.assertEqual(values["dip_cor_flag"], "0")
        self.assertEqual(values["efield_dir"], "1")
        self.assertEqual(values["efield_pos_max"], "0.8")
        self.assertEqual(values["efield_pos_dec"], "0.1")
        self.assertEqual(values["efield_amp"], "1e-4")

    def test_free_restart_removes_field_and_fixed_occupation(self):
        values = parse_input(
            render_input(mode="free", field_dir=2, restart=True)
        )

        self.assertEqual(values["ocp"], "0")
        self.assertNotIn("ocp_set", values)
        self.assertEqual(values["efield_flag"], "0")
        self.assertEqual(values["efield_amp"], "0")
        self.assertNotIn("efield_dir", values)
        self.assertEqual(values["init_wfc"], "file")
        self.assertEqual(values["init_chg"], "file")

    def test_rejects_invalid_field_directions(self):
        for mode in ("field", "free"):
            for field_dir in (None, -1, 3, "1"):
                with self.subTest(mode=mode, field_dir=field_dir):
                    with self.assertRaisesRegex(ValueError, "field_dir"):
                        render_input(
                            mode=mode,
                            field_dir=field_dir,
                            restart=mode == "free",
                        )


if __name__ == "__main__":
    unittest.main()
