import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HpcContractTests(unittest.TestCase):
    def read(self, name):
        return (ROOT / name).read_text(encoding="ascii")

    def test_runner_uses_full_server66_nodes_and_validates_siab_outputs(self):
        text = self.read("run_siab_reference_server66.slurm")
        required = (
            "#SBATCH --partition=640",
            "#SBATCH --nodes=10",
            "#SBATCH --ntasks=10",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=48",
            "#SBATCH --mem=180000M",
            "#SBATCH --time=1-00:00:00",
            "#SBATCH --exclusive",
            "ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-6",
            "ABACUS_STERNHEIMER_FD_ST_MAX_ITER=300",
            'mpirun -np "$SLURM_NTASKS" -ppn 1',
            "sternheimer_mpi_layout",
            "global_equation",
            "sternheimer_matrix.dat",
            "STERNHEIMER_SIAB_STATUS.dat",
            "STERNHEIMER_SIAB_MEMORY.dat",
            "STERNHEIMER_ABFS_CHANNELS.dat",
            "STERNHEIMER_SIAB_COULOMB_WHITENING.dat",
            "SIAB_REFERENCE_COMPLETE.json",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("--partition=debug", text)
        self.assertNotIn("SOLVER_TOL=1e-8", text)

    def test_submitter_has_duplicate_guard_and_preflight_mode(self):
        text = self.read("submit_siab_reference_server66.sh")
        for token in (
            "squeue",
            "sacct",
            ".submission-claim",
            "SIAB_REFERENCE_JOB_ID.txt",
            "sbatch --test-only",
            "ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1",
            "SIAB_REFERENCE_SOURCE_COMMIT",
            "sternheimer_mpi_layout[[:space:]]+global_equation",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
