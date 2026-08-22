import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HpcContractTests(unittest.TestCase):
    def read(self, name):
        return (ROOT / name).read_text(encoding="ascii")

    def test_response_runner_uses_one_node_per_frequency_and_full_node_resources(self):
        text = self.read("run_response_branch_server66.slurm")
        required = (
            "#SBATCH --partition=640",
            "#SBATCH --array=0-1%2",
            "#SBATCH --nodes=6",
            "#SBATCH --ntasks=6",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=48",
            "#SBATCH --mem=180000M",
            "#SBATCH --time=1-00:00:00",
            "#SBATCH --exclusive",
            "ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-6",
            "ABACUS_STERNHEIMER_FD_ST_MAX_ITER=300",
            'mpirun -np "$SLURM_NTASKS" -ppn 1',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("--partition=debug", text)
        self.assertNotIn("SOLVER_TOL=1e-8", text)

    def test_response_runner_checks_the_full_physical_contract(self):
        text = self.read("run_response_branch_server66.slurm")
        for token in (
            "RESULT_SUMMARY.json",
            "PREPARATION_MANIFEST.json",
            "Read NAO wave functions from OUT.C_DELTA_RESPONSE_GATE/wfs1_nao.txt",
            "Read NAO wave functions from OUT.C_DELTA_RESPONSE_GATE/wfs2_nao.txt",
            "Read electron density from file: OUT.C_DELTA_RESPONSE_GATE/chgs1.cube",
            "Read electron density from file: OUT.C_DELTA_RESPONSE_GATE/chgs2.cube",
            "frequency_grid_source file",
            "sternheimer_fd_order 8",
            "sternheimer_delta yes",
            "all_converged yes",
            "max_solver_relative_residual",
            "perturbation_coulomb_kernel full_periodic_poisson",
            "workflow_source_commit",
            "v1_sternheimer_chi0_iq_1_ifreq_*_rank*.dat",
            "v1_coulomb_full_iq_1_rank0.dat",
            "RESPONSE_COMPLETE.json",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_librpa_runner_uses_full_coulomb_and_records_each_frequency(self):
        text = self.read("run_librpa_branch_server66.slurm")
        for token in (
            "#SBATCH --partition=640",
            "#SBATCH --nodes=1",
            "#SBATCH --cpus-per-task=48",
            "#SBATCH --mem=180000M",
            "task = sternheimer_rpa",
            "nfreq = 6",
            "sqrt_coulomb_threshold = 1e-5",
            "prefix_coul_full = v1_coulomb_full_iq_",
            "prefix_sternheimer_chi0 = v1_sternheimer_chi0_iq_",
            "use_rpa_gamma = true",
            "replace_w_head = false",
            "Total Sternheimer EcRPA",
            "LIBRPA_COMPLETE.json",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_submitter_has_stable_duplicate_guard_and_dependencies(self):
        text = self.read("submit_response_gate_server66.sh")
        for token in (
            "squeue",
            "sacct",
            ".submission-claim",
            "RESPONSE_JOB_ID.txt",
            "LIBRPA_JOB_ID.txt",
            "--dependency=afterok:",
            "--job-name=",
            "--array=0-1%2",
            "DELTA_GATE_SOURCE_COMMIT",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
