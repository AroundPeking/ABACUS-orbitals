import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from response_contract import (
    ACCEPTED_PHASES,
    EV_PER_HA,
    parse_eig_occ,
    render_response_input,
    union_transition_window,
)


def eig_occ_text(spin1, spin2):
    lines = [
        "1 # ionic step",
        " Electronic state energy (eV) and occupations",
        " Spin number 2",
    ]
    for spin, rows in ((1, spin1), (2, spin2)):
        lines.append(
            f" spin={spin} k-point=1/1 Cartesian=0.0 0.0 0.0 (1 plane wave)"
        )
        lines.extend(
            f" {index} {energy:.16f} {occupation:.16f}"
            for index, (energy, occupation) in enumerate(rows, 1)
        )
        lines.append("")
    return "\n".join(lines)


def parse_input(text):
    lines = text.splitlines()
    if not lines or lines[0] != "INPUT_PARAMETERS":
        raise AssertionError("missing INPUT_PARAMETERS header")
    values = {}
    for line in lines[1:]:
        key, value = line.split(maxsplit=1)
        if key in values:
            raise AssertionError(f"duplicate key {key}")
        values[key] = value
    return values


class ResponseContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_eig_occ(self, name, spin1, spin2):
        path = self.root / name
        path.write_text(eig_occ_text(spin1, spin2), encoding="ascii")
        return path

    def test_accepted_sources_are_final_zero_field_phases(self):
        self.assertEqual(
            ACCEPTED_PHASES,
            {
                "fixed": "runs/fixed/fixed_zero_restart",
                "free": "runs/dir0/free_restart2",
            },
        )

    def test_parses_integer_triplet_and_transition_window(self):
        path = self.write_eig_occ(
            "fixed.txt",
            [(-10.0, 1.0), (-4.0, 1.0), (-3.0, 1.0), (1.0, 0.0), (8.0, 0.0)],
            [(-8.0, 1.0), (-2.0, 0.0), (7.0, 0.0)],
        )

        record = parse_eig_occ(path)

        self.assertEqual(record.spin_counts, {1: 3, 2: 1})
        self.assertEqual(record.nbands_by_spin, {1: 5, 2: 3})
        self.assertAlmostEqual(record.transition_min_ha, 4.0 / EV_PER_HA)
        self.assertAlmostEqual(record.transition_max_ha, 18.0 / EV_PER_HA)

    def test_union_window_covers_both_zero_field_states(self):
        fixed = parse_eig_occ(
            self.write_eig_occ(
                "fixed.txt",
                [(-10.0, 1.0), (-4.0, 1.0), (-3.0, 1.0), (1.0, 0.0), (8.0, 0.0)],
                [(-8.0, 1.0), (-2.0, 0.0), (7.0, 0.0)],
            )
        )
        free = parse_eig_occ(
            self.write_eig_occ(
                "free.txt",
                [(-11.0, 1.0), (-4.0, 1.0), (-3.5, 1.0), (-0.5, 0.0), (9.0, 0.0)],
                [(-9.0, 1.0), (-1.0, 0.0), (10.0, 0.0)],
            )
        )

        minimum, maximum = union_transition_window([fixed, free])

        self.assertAlmostEqual(minimum, 3.0 / EV_PER_HA)
        self.assertAlmostEqual(maximum, 20.0 / EV_PER_HA)

    def test_rejects_fractional_or_wrong_spin_occupations(self):
        fractional = self.write_eig_occ(
            "fractional.txt",
            [(-10.0, 1.0), (-4.0, 1.0), (-3.0, 0.5), (1.0, 0.5)],
            [(-8.0, 1.0), (-2.0, 0.0)],
        )
        wrong_spin = self.write_eig_occ(
            "wrong-spin.txt",
            [(-10.0, 1.0), (-4.0, 1.0), (1.0, 0.0)],
            [(-8.0, 1.0), (-2.0, 1.0), (7.0, 0.0)],
        )

        with self.assertRaisesRegex(ValueError, "integer"):
            parse_eig_occ(fractional)
        with self.assertRaisesRegex(ValueError, "triplet"):
            parse_eig_occ(wrong_spin)

    def test_response_input_freezes_physics_and_changes_only_ocp(self):
        fixed = parse_input(render_response_input("fixed"))
        free = parse_input(render_response_input("free"))

        common = {
            "suffix": "C_DELTA_RESPONSE_GATE",
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
            "gamma_only": "1",
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
            "out_wfc_lcao": "1",
            "out_app_flag": "1",
            "out_mul": "1",
            "efield_flag": "0",
            "efield_amp": "0",
            "init_wfc": "file",
            "init_chg": "file",
            "rpa": "1",
            "out_librpa_reader_version": "1",
            "exx_pca_threshold": "1e-4",
            "exx_singularity_correction": "massidda",
            "exx_ccp_rmesh_times": "1",
            "rpa_ccp_rmesh_times": "1",
            "out_sternheimer_librpa": "1",
            "sternheimer_nfreq": "6",
            "sternheimer_frequency_grid_file": "fixed_frequency_grid.dat",
            "sternheimer_frequency_mpi": "1",
            "sternheimer_delta": "1",
            "sternheimer_fd_order": "8",
            "sternheimer_delta_max_states": "0",
            "sternheimer_delta_norm_tol": "1e-10",
        }
        self.assertEqual({key: fixed[key] for key in common}, common)
        self.assertEqual({key: free[key] for key in common}, common)
        self.assertEqual(fixed["ocp"], "1")
        self.assertEqual(fixed["ocp_set"], "3*1 19*0 1*1 21*0")
        self.assertEqual(free["ocp"], "0")
        self.assertNotIn("ocp_set", free)

    def test_rejects_unknown_response_branch(self):
        with self.assertRaisesRegex(ValueError, "unsupported response branch"):
            render_response_input("dir1")


if __name__ == "__main__":
    unittest.main()
