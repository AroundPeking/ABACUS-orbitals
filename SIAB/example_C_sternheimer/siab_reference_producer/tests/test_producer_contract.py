import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from producer_contract import render_siab_input, render_siab_stru


class ProducerContractTests(unittest.TestCase):
    def test_input_follows_the_accepted_h_reference_protocol(self):
        text = render_siab_input()
        required = (
            "nelec 4\n",
            "nspin 2\n",
            "nupdown 2\n",
            "nbands 22\n",
            "nx 135\n",
            "ny 135\n",
            "nz 135\n",
            "ocp 1\n",
            "ocp_set 3*1 19*0 1*1 21*0\n",
            "init_wfc file\n",
            "init_chg file\n",
            "bessel_nao_ecut 100\n",
            "bessel_nao_rcut 10\n",
            "bessel_nao_smooth 1\n",
            "bessel_nao_sigma 0.1\n",
            "bessel_nao_tolerence 1e-12\n",
            "out_sternheimer_librpa 0\n",
            "out_sternheimer_siab 1\n",
            "sternheimer_siab_coulomb_threshold 1e-10\n",
            "sternheimer_siab_lmax 4\n",
            "sternheimer_nfreq 16\n",
            "sternheimer_frequency_grid_file fixed_frequency_grid_nfreq16.dat\n",
            "sternheimer_frequency_mpi 1\n",
            "sternheimer_channel_mpi 1\n",
            "sternheimer_mpi_layout global_equation\n",
            "sternheimer_delta 1\n",
            "sternheimer_fd_order 8\n",
            "sternheimer_delta_max_states 0\n",
            "sternheimer_delta_norm_tol 1e-10\n",
            "exx_pca_threshold 10\n",
            "exx_singularity_correction massidda\n",
            "exx_ccp_rmesh_times 1\n",
            "rpa_ccp_rmesh_times 1\n",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("out_sternheimer_librpa 1", text)
        self.assertNotIn("efield_flag 1", text)
        self.assertNotIn("exx_pca_threshold 1e-4", text)

    def test_stru_adds_exactly_one_explicit_abfs_section(self):
        source = """ATOMIC_SPECIES
C 12.011 C_ONCV_PBE-1.0.upf

NUMERICAL_ORBITAL
C_gga_10au_100Ry_3s3p2d.orb
"""
        result = render_siab_stru(source, "C_10au_3s3p2d1f1g_pca1e-4.abfs")
        self.assertEqual(result.count("ABFS_ORBITAL"), 1)
        self.assertTrue(
            result.endswith(
                "ABFS_ORBITAL\nC_10au_3s3p2d1f1g_pca1e-4.abfs\n"
            )
        )

    def test_stru_rejects_an_existing_abfs_section(self):
        with self.assertRaisesRegex(ValueError, "already contains ABFS_ORBITAL"):
            render_siab_stru("ABFS_ORBITAL\nold.abfs\n", "new.abfs")


if __name__ == "__main__":
    unittest.main()
