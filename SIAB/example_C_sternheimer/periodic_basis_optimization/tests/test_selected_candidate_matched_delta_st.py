from pathlib import Path
import unittest


class SelectedCandidateMatchedDeltaStContractTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.validation = self.root / "ordinary_sos_validation"

    def test_response_uses_selected_basis_exact_sos_grid_and_nested_mpi(self):
        text = (
            self.validation
            / "run_selected_candidate_matched_delta_response.slurm"
        ).read_text(encoding="ascii")

        self.assertIn("#SBATCH --partition=p1", text)
        self.assertIn("#SBATCH --nodes=48", text)
        self.assertIn("#SBATCH --ntasks=48", text)
        self.assertIn("#SBATCH --ntasks-per-node=1", text)
        self.assertIn("#SBATCH --cpus-per-task=40", text)
        self.assertIn("#SBATCH --mem=190000", text)
        self.assertIn("C_gga_10au_100Ry_selected_product_pca.orb", text)
        self.assertIn("SELECTED_SOS_FREQUENCY_GRID.dat", text)
        self.assertIn('set_input_key sternheimer_frequency_grid_file "$frequency_name"', text)
        self.assertIn("set_input_key out_sternheimer_librpa 1", text)
        self.assertIn("set_input_key out_sternheimer_basis_opt 0", text)
        self.assertIn("set_input_key symmetry 1", text)
        self.assertIn("set_input_key kpar 8", text)
        self.assertIn("set_input_key sternheimer_frequency_mpi 1", text)
        self.assertIn("set_input_key sternheimer_mpi_layout frequency_grouped", text)
        self.assertIn("set_input_key sternheimer_fd_order 8", text)
        self.assertIn("set_input_key exx_pca_threshold 1e-4", text)
        self.assertIn("frequency_grid_source file", text)
        self.assertIn("all_converged yes", text)
        self.assertNotIn("ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=", text)

    def test_release_reads_q1_canonical_set_and_has_duplicate_guard(self):
        text = (
            self.validation
            / "release_selected_candidate_matched_delta_after_q1.sh"
        ).read_text(encoding="ascii")

        self.assertIn("sternheimer_canonical_q_indices", text)
        self.assertIn("SELECTED_SOS_FREQUENCY_GRID.dat", text)
        self.assertIn("MATCHED_DELTA_CHAIN.txt", text)
        self.assertIn("sacct -n -X -j", text)
        self.assertIn("runtime_gate_hours=18", text)
        self.assertIn("abacus.time", text)
        self.assertIn("--array=", text)
        self.assertIn("--dependency=afterok:", text)
        self.assertNotIn("1,22,43,6,27,23,11,55", text)
        self.assertNotIn("1,2,3,6,7,8,11,22,23,24,27,28,43", text)

    def test_consumer_merges_symmetry_routes_and_uses_all_grid_matrices(self):
        text = (
            self.validation
            / "run_selected_candidate_matched_delta_reader_d4810f73.slurm"
        ).read_text(encoding="ascii")

        self.assertIn("d4810f73aab20c36e69b1c353c945b77f40931c9", text)
        self.assertIn("merge_sternheimer_symmetry_manifests.py", text)
        self.assertIn("for ordinal in $(seq 1 64)", text)
        self.assertIn('test "$grid_coulomb_file_count" -eq 64', text)
        self.assertIn("task = sternheimer_rpa", text)
        self.assertIn("prefix_coul_full = v1_coulomb_grid_iq_", text)
        self.assertIn("fn_sternheimer_qpoints = v1_sternheimer_qpoints.dat", text)
        self.assertIn("fn_sternheimer_qstar_routes = v1_sternheimer_qstar_routes.dat", text)
        self.assertIn("use_symmetry_rpa = true", text)
        self.assertIn("replace_w_head = false", text)
        self.assertIn("frequency_grid_source=selected_candidate_sos", text)
        self.assertIn("scope body_only_no_analytic_headwing", text)
        self.assertNotIn("v1_coulomb_full_iq_", text)


if __name__ == "__main__":
    unittest.main()
