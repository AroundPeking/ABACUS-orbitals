import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"


class PbeLauncherContractTests(unittest.TestCase):
    def test_single_node_pmi2_launcher_is_shared_with_preflight(self):
        script = (WORKFLOW / "run_c_all_radial_pbe_endpoint.slurm").read_text()
        self.assertIn('test "${SLURM_JOB_NUM_NODES:?}" = 1', script)
        self.assertIn('launcher=(srun --mpi=pmi2 --cpu-bind=none -n 4)', script)
        self.assertIn('export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi2.so', script)
        self.assertNotIn('launcher=(mpirun', script)
        probe = '"${launcher[@]}" "$stage/mpi_preflight"'
        physics = '"${launcher[@]}" "$abacus"'
        self.assertIn(probe, script)
        self.assertIn(physics, script)
        self.assertLess(script.index(probe), script.index(physics))
        self.assertIn("grep -qx 'MPI_PREFLIGHT_OK ranks=4'", script)

    def test_preflight_source_is_hash_locked_and_checks_collective(self):
        script = (WORKFLOW / "run_c_all_radial_pbe_endpoint.slurm").read_text()
        path = WORKFLOW / "c_single_node_mpi_preflight.c"
        self.assertTrue(path.is_file(), "MPI collective preflight is missing")
        source = path.read_text()
        self.assertIn('C_MPI_PREFLIGHT_SHA256', script)
        self.assertIn('MPI_Allreduce', source)
        self.assertIn('size != 4', source)
        self.assertIn('sum != 6', source)
        self.assertIn('MPI_PREFLIGHT_OK ranks=4', source)


if __name__ == "__main__":
    unittest.main()
