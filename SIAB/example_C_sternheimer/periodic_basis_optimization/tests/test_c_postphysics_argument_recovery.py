import importlib.util
from pathlib import Path
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / "galerkin_binding_workflow"
SCRIPT = WORKFLOW / "validate_c_postphysics_argument_failure.py"


class PostPhysicsArgumentRecoveryTests(unittest.TestCase):
    def check(self, **overrides):
        self.assertTrue(SCRIPT.is_file(), "missing strict post-physics failure gate")
        spec = importlib.util.spec_from_file_location("argument_failure", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = dict(
            job_id="21875804_4",
            accounting=(
                "JobID|State|ExitCode|Elapsed|MaxRSS\n"
                "21875804_4|FAILED|2:0|01:58:55|\n"
                "21875804_4.batch|FAILED|2:0|01:58:55|100K\n"
                "21875804_4.extern|COMPLETED|0:0|01:58:55|100K\n"
                "21875804_4.0|COMPLETED|0:0|01:55:48|6000000K\n"
            ),
            stderr=("validate_c_solid_fd8_q_dataset.py: error: argument "
                    "--frequency-grid-source: invalid choice: "
                    "'frozen_df_q1_greenx_minimax' (choose from 'greenx_minimax', "
                    "'frozen_q1_greenx_minimax')\n"),
            abacus_output=" FINISH Time  : done\n TOTAL  Time  : 6947\n",
            abacus_time="\tExit status: 0\n",
        )
        data.update(overrides)
        return module.validate_completion(**data)

    def test_completed_physics_with_exact_argument_failure_is_recoverable(self):
        result = self.check()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["completed_mpi_steps"], 1)
        self.assertEqual(result["source_scheduler_exit_code"], "2:0")

    def test_node_failure_or_failed_mpi_step_is_rejected(self):
        for state in ("NODE_FAIL", "RUNNING"):
            with self.assertRaisesRegex(ValueError, "parent"):
                self.check(accounting="JobID|State|ExitCode\n21875804_4|" + state + "|2:0\n")
        with self.assertRaisesRegex(ValueError, "step"):
            self.check(accounting=("JobID|State|ExitCode\n21875804_4|FAILED|2:0\n"
                                  "21875804_4.0|FAILED|5:0\n"))

    def test_unknown_argument_or_transport_failure_is_rejected(self):
        for message in ("unrecognized argument: --wrong", "Transport retry count exceeded"):
            with self.assertRaisesRegex(ValueError, "argument"):
                self.check(stderr=message)

    def test_missing_finish_or_nonzero_abacus_exit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finish"):
            self.check(abacus_output="still running\n")
        with self.assertRaisesRegex(ValueError, "exit"):
            self.check(abacus_time="Exit status: 127\n")

    def test_missing_mpi_steps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "MPI"):
            self.check(accounting="JobID|State|ExitCode\n21875804_4|FAILED|2:0\n")

    def test_producer_uses_supported_frequency_source(self):
        script = (WORKFLOW / "run_c_solid_fd8_q13_standard_dfdcu.slurm").read_text()
        self.assertNotIn("--frequency-grid-source frozen_df_q1_greenx_minimax", script)
        self.assertIn("--frequency-grid-source frozen_q1_greenx_minimax", script)


if __name__ == "__main__":
    unittest.main()
