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
            "#SBATCH --time=UNLIMITED",
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

    def test_df_runner_uses_sixteen_global_ranks_and_explicit_batch_environment(self):
        text = self.read("run_siab_reference_df.slurm")
        required = (
            "#SBATCH --partition=p1",
            "#SBATCH --nodes=16",
            "#SBATCH --ntasks=16",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=40",
            "#SBATCH --mem=190000M",
            "#SBATCH --time=UNLIMITED",
            "#SBATCH --exclusive",
            "module load oneapi/2024.2",
            "I_MPI_FABRICS=shm:ofi",
            "ABACUS_STERNHEIMER_CHANNEL_THREADS=40",
            "ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-6",
            "8cf890e8c09cc4d09bf8aca246158f5fca27d7f1",
            'mpirun -ppn 1 -np "$SLURM_NTASKS"',
            "SIAB_REFERENCE_COMPLETE.json",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("source ~/.bashrc", text)
        self.assertNotIn("--partition=debug", text)

    def test_df_pilot_preserves_full_physics_and_writes_runtime_evidence(self):
        text = self.read("run_siab_reference_pilot_df.slurm")
        for token in (
            "#SBATCH --partition=p1",
            "#SBATCH --nodes=16",
            "#SBATCH --cpus-per-task=40",
            "timeout --signal=TERM --kill-after=120s 30m",
            "STERNHEIMER_SIAB_PROGRESS_rank*.dat",
            "SIAB_REFERENCE_PILOT.json",
            "sternheimer_nfreq[[:space:]]+16",
            "sternheimer_mpi_layout[[:space:]]+global_equation",
            '"mpi_ranks": 16',
            '"omp_threads_per_rank": 40',
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_df_submitter_has_observable_duplicate_guard_and_test_only_mode(self):
        text = self.read("submit_siab_reference_df.sh")
        for token in (
            "squeue",
            "sacct",
            ".submission-claim-df",
            "DF_DIAG_JOB_ID.txt",
            "DF_PILOT_JOB_ID.txt",
            "DF_REFERENCE_JOB_ID.txt",
            "sbatch --test-only",
            "run_siab_abfs_diag_df.slurm",
            "run_siab_reference_pilot_df.slurm",
            "run_siab_reference_df.slurm",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_df_recovery_requires_completed_precompute_and_cancelled_formal_job(self):
        text = self.read("submit_siab_reference_df_recovery.sh")
        for token in (
            "runtime_gate_failed",
            'pilot["abacus_exit_code"] == 124',
            'pilot["completed_equations"] == 0',
            "channel_workers_ready",
            "STERNHEIMER_CHI0_FAILURE_rank*.dat",
            "CANCELLED",
            "00:00:00",
            ".submission-claim-df-recovery",
            "DF_REFERENCE_RECOVERY_JOB_ID.txt",
            "squeue",
            "sacct",
            "sbatch --test-only",
            "run_siab_reference_df.slurm",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
