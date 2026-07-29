"""Static contracts for the H2 response-shell producer family."""

import json
from pathlib import Path
import unittest


SIAB_ROOT = Path(__file__).resolve().parents[1]
GREEDY = SIAB_ROOT / "example_H_sternheimer" / "greedy_response_selection"


def input_values(path):
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) >= 2 and fields[0] != "INPUT_PARAMETERS":
            values[fields[0]] = " ".join(fields[1:])
    return values


class H2GreedyResponseTemplatesTest(unittest.TestCase):
    def test_compact_response_sos_uses_frozen_48_ao_frontiers(self):
        campaign = SIAB_ROOT / "example_H_sternheimer" / (
            "compact_response_sos"
        )
        runner = (campaign / "run_sos.slurm").read_text(encoding="utf-8")
        readme = (campaign / "README.md").read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=normal", runner)
        self.assertIn("#SBATCH --array=0-8", runner)
        self.assertIn("#SBATCH --cpus-per-task=30", runner)
        self.assertIn("#SBATCH --mem=110610M", runner)
        self.assertIn("#SBATCH --time=1-00:00:00", runner)
        self.assertIn(
            "lanes=(tail_0p00 tail_0p00 tail_0p00 tail_0p10 tail_0p10 "
            "tail_0p10 tail_0p30 tail_0p30 tail_0p30)",
            runner,
        )
        self.assertIn(
            "case_names=(H2 H H_ghost H2 H H_ghost H2 H H_ghost)",
            runner,
        )
        self.assertIn("nbands=(96 48 96 96 48 96 96 48 96)", runner)
        self.assertIn("expected_spins=(1 2 2 1 2 2 1 2 2)", runner)
        self.assertIn("expected_electrons=(2 1 1 2 1 1 2 1 1)", runner)
        self.assertIn("ao_budget_reached", runner)
        self.assertIn('last["ao_function_count"] != 48', runner)
        self.assertIn('config["optimizer_loss"]["radial_tail_weight"]', runner)
        self.assertIn("ORBITAL_1U.dat", runner)
        self.assertIn("optimizer_steps", runner)
        self.assertIn("held_out_h2_sos_greedy_full/cases", runner)
        self.assertIn("exx_pca_threshold", runner)
        self.assertIn("H_empty", runner)
        self.assertIn("fixed_abs", runner)
        self.assertIn(
            "d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f",
            runner,
        )
        self.assertIn("v1_coulomb_full_iq_1_rank0.dat", runner)
        self.assertIn("libRPA finished successfully", runner)
        self.assertIn("version_exception=historical_exact_sos_binary", runner)

        self.assertIn("H2/H/H+ghost", readme)
        self.assertIn("48 AO/H", readme)
        self.assertIn("96/48/96", readme)
        self.assertIn("does not feed back", readme)

    def test_full_greedy_basis_sos_uses_all_bands_and_fixed_abs(self):
        campaign = SIAB_ROOT / "example_H_sternheimer" / (
            "held_out_h2_sos_greedy_full"
        )
        runner = (campaign / "run_sos_cp.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --partition=normal", runner)
        self.assertIn("#SBATCH --array=0-2", runner)
        self.assertIn("#SBATCH --cpus-per-task=30", runner)
        self.assertIn("#SBATCH --mem=110610M", runner)
        self.assertIn("#SBATCH --time=1-00:00:00", runner)
        self.assertIn(
            "d518c5997f667249554ab19af1a36d12f17439ffb7f0ecaf5977c1e829be7b00",
            runner,
        )
        self.assertIn("nbands=334", runner)
        self.assertIn("nbands=167", runner)
        self.assertIn("expected_electrons=2", runner)
        self.assertIn("expected_electrons=1", runner)
        self.assertIn("H_empty", runner)
        self.assertIn("fixed_abs", runner)
        self.assertIn(
            "2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d",
            runner,
        )
        self.assertIn(
            "defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c",
            runner,
        )

        expected = {
            "H2": (334, 1, 2),
            "H": (167, 2, 1),
            "H_ghost": (334, 2, 1),
        }
        for case, (nbands, nspin, nelec) in expected.items():
            case_dir = campaign / "cases" / case
            values = input_values(case_dir / "INPUT")
            self.assertEqual(values["nbands"], str(nbands))
            self.assertEqual(values.get("nspin", "1"), str(nspin))
            self.assertEqual(values["nelec"], str(nelec))
            self.assertEqual(values["ecutwfc"], "100")
            self.assertEqual(values["rpa"], "1")
            self.assertEqual(values["exx_pca_threshold"], "10")
            self.assertEqual(
                values["exx_singularity_correction"], "massidda"
            )
            self.assertEqual(values["rpa_ccp_rmesh_times"], "5")

            librpa = (case_dir / "librpa.in").read_text(encoding="utf-8")
            self.assertIn("prefix_coul_full = v1_coulomb_full_iq_", librpa)
            self.assertIn("nfreq = 16", librpa)
            self.assertIn("vq_threshold = 0", librpa)
            self.assertIn("sqrt_coulomb_threshold = 0", librpa)

            stru = (case_dir / "STRU").read_text(encoding="utf-8")
            self.assertIn("37.79452292169073", stru)
            self.assertIn("H_gga_8au_100Ry_13s11p10d5f4g.orb", stru)
            self.assertIn(
                "H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs", stru
            )
        ghost_stru = (campaign / "cases/H_ghost/STRU").read_text(
            encoding="utf-8"
        )
        self.assertIn("H_empty", ghost_stru)

    def test_full_greedy_ghost_retry_distributes_reader_v1_over_two_nodes(self):
        campaign = SIAB_ROOT / "example_H_sternheimer" / (
            "held_out_h2_sos_greedy_full"
        )
        runner = (campaign / "run_ghost_mpi_cp.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --partition=normal", runner)
        self.assertIn("#SBATCH --nodes=2", runner)
        self.assertIn("#SBATCH --ntasks=2", runner)
        self.assertIn("#SBATCH --ntasks-per-node=1", runner)
        self.assertIn("#SBATCH --cpus-per-task=30", runner)
        self.assertIn("#SBATCH --mem=110610M", runner)
        self.assertIn("#SBATCH --time=1-00:00:00", runner)
        self.assertIn('mpirun -np "$SLURM_NTASKS" -ppn 1', runner)
        self.assertIn("v1_Cs_data_*.txt", runner)
        self.assertIn("v1_coulomb_full_iq_1_rank*.dat", runner)
        self.assertIn("expected_reader_ranks=2", runner)
        self.assertIn("mpirun -np 1 -ppn 1", runner)
        self.assertIn("libRPA finished successfully", runner)

    def test_selection_tolerates_measured_atomic_l4_grid_anisotropy(self):
        config = json.loads(
            (GREEDY / "selection_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["magnetic_overlap_tolerance"], 2.0e-4)

    def test_step30_resume_uses_corrected_floor_contract(self):
        config = json.loads(
            (GREEDY / "selection_config_resume_step30.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["global_capture"], 0.999)
        self.assertEqual(config["magnetic_overlap_tolerance"], 3.0e-4)
        self.assertEqual(config["max_l"], 4)

        runner = (
            GREEDY / "run_selection_resume_step30.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=normal", runner)
        self.assertIn("#SBATCH --cpus-per-task=30", runner)
        self.assertIn("#SBATCH --mem=110610M", runner)
        self.assertIn("#SBATCH --time=1-00:00:00", runner)
        self.assertIn(
            "siab_greedy_selection_source_h_h2_floor_fixed_resume_v6_20260728",
            runner,
        )
        self.assertIn(
            "resume_root=/work1/ghj/sternheimer_abacus_tests/"
            "siab_greedy_selection_campaign_h_h2_physical_only_prod_v5_20260727",
            runner,
        )
        self.assertIn(
            "baseline=$resume_root/work/step_030/optimizer/ORBITAL_RESULTS.txt",
            runner,
        )
        self.assertIn("selection_config_resume_step30.json", runner)
        self.assertIn("--max-steps 16", runner)
        self.assertIn("torch.equal", runner)
        self.assertNotIn("ghost_target", runner)
        self.assertNotIn("producer_h2_fragment_ghost", runner)

    def test_h2_and_h2_ghost_replace_the_h3_target_contract(self):
        h2 = GREEDY / "producer_h2"
        ghost = GREEDY / "producer_h2_fragment_ghost"
        self.assertTrue(h2.is_dir())
        self.assertTrue(ghost.is_dir())

        h2_input = input_values(h2 / "INPUT")
        self.assertEqual(h2_input["suffix"], "H2_SIAB_GREEDY_R074085_WFULL_NF16_E50")
        self.assertEqual(h2_input["ntype"], "1")
        self.assertEqual(h2_input["nelec"], "2")
        self.assertEqual(h2_input["nspin"], "1")
        self.assertEqual(h2_input["ecutwfc"], "50")
        self.assertEqual(h2_input["sternheimer_nfreq"], "16")
        self.assertEqual(h2_input["sternheimer_frequency_mpi"], "1")
        self.assertEqual(h2_input["sternheimer_channel_mpi"], "1")
        self.assertEqual(h2_input["rpa_ccp_rmesh_times"], "5")

        ghost_input = input_values(ghost / "INPUT")
        self.assertEqual(ghost_input["suffix"], "H_GHOST1_SIAB_GREEDY_R074085_WFULL_NF16_E50")
        self.assertEqual(ghost_input["ntype"], "2")
        self.assertEqual(ghost_input["nelec"], "1")
        self.assertEqual(ghost_input["nspin"], "2")
        self.assertEqual(ghost_input["sternheimer_channel_mpi"], "1")

        atom_input = input_values((GREEDY / "producer_atom") / "INPUT")
        self.assertEqual(atom_input["sternheimer_channel_mpi"], "1")

        h2_stru = (h2 / "STRU").read_text(encoding="utf-8")
        ghost_stru = (ghost / "STRU").read_text(encoding="utf-8")
        self.assertIn("0.48147879757009904 0.5 0.5", h2_stru)
        self.assertIn("0.518521202429901 0.5 0.5", h2_stru)
        self.assertIn("H_empty", ghost_stru)
        self.assertIn("0.48147879757009904 0.5 0.5", ghost_stru)
        self.assertIn("0.518521202429901 0.5 0.5", ghost_stru)

        runner = (GREEDY / "run_targets.slurm").read_text(encoding="utf-8")
        self.assertIn(
            "case_names=(producer_atom producer_h2 producer_h2_fragment_ghost)",
            runner,
        )
        self.assertIn("expected_atoms=(1 2 2)", runner)
        self.assertIn("expected_auxiliary_channels=(214 428 428)", runner)
        self.assertIn("expected_solved_equations=(3424 6848 6848)", runner)
        self.assertIn("#SBATCH --nodes=32", runner)
        self.assertIn("#SBATCH --ntasks=32", runner)
        self.assertIn("ABACUS_STERNHEIMER_THREADS:-30", runner)
        self.assertIn('mpirun -np "$SLURM_NTASKS" -ppn 1', runner)
        self.assertIn("sternheimer_channel_mpi yes", runner)
        self.assertIn("frequency_group_size 2", runner)
        self.assertIn("mpi_ranks 32", runner)
        self.assertIn(
            "abacus_source_commit=c273b4ee7051138293d9988c3eb79bee36c0af10",
            runner,
        )
        self.assertIn(
            "abacus_sha256=ff38348fbad89fde4a985c13f97b59ffc94353c22c7098e19b373c1ef7e76fee",
            runner,
        )

        selection = (GREEDY / "run_selection.slurm").read_text(encoding="utf-8")
        self.assertIn("producer_h2", selection)
        self.assertNotIn("producer_h2_fragment_ghost", selection)
        self.assertNotIn("ghost_target", selection)
        self.assertNotIn("ghost_validation", selection)
        self.assertNotIn("--ghost-target", selection)
        self.assertNotIn("producer_h3", selection)
        self.assertNotIn("$h3_target", selection)
        self.assertIn('len(payload.get("frequencies", ())) != 16', selection)
        self.assertIn(
            "source_root=/work1/ghj/sternheimer_abacus_tests/"
            "siab_greedy_selection_source_h_h2_physical_only_prod_v5_20260727",
            selection,
        )
        self.assertIn(
            "target_root=/work1/ghj/sternheimer_abacus_tests/"
            "siab_greedy_targets_h2_channel_mpi_prod_v1_20260726",
            selection,
        )
        self.assertIn(
            "output_root=/work1/ghj/sternheimer_abacus_tests/"
            "siab_greedy_selection_campaign_h_h2_physical_only_prod_v5_20260727",
            selection,
        )
        self.assertIn("STERNHEIMER_SIAB_STATUS.dat", selection)
        self.assertIn("sternheimer_channel_mpi yes", selection)
        self.assertIn("frequency_group_size 2", selection)
        self.assertIn("mpi_ranks 32", selection)
        self.assertIn("all_converged yes", selection)
        self.assertIn("test -s SOURCE_COMMIT", selection)

    def test_h2_target_openmp_benchmark_uses_the_production_operator(self):
        benchmark = (
            GREEDY / "run_h2_target_openmp_ab.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-1", benchmark)
        self.assertIn("thread_counts=(16 24)", benchmark)
        self.assertIn("producer_h2", benchmark)
        self.assertIn("ecutwfc 50", benchmark)
        self.assertIn("sternheimer_nfreq 1", benchmark)
        self.assertIn("H2_SIAB_OMP${threads}_NFREQ1_E50", benchmark)
        self.assertIn("global_full_coulomb", benchmark)
        self.assertIn("validate_targets.py", benchmark)
        self.assertIn("compare_sternheimer_targets.py", benchmark)

        comparator = GREEDY / "compare_sternheimer_targets.py"
        self.assertTrue(comparator.is_file())

    def test_h2_channel_mpi_gate_uses_full_outputs_at_10_ry(self):
        gate = (
            GREEDY / "run_h2_target_channel_mpi_gate.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn('mode=${SIAB_CHANNEL_MPI_MODE:?}', gate)
        self.assertIn("serial|channel_mpi", gate)
        self.assertIn("ecutwfc 10", gate)
        self.assertIn("sternheimer_nfreq 1", gate)
        self.assertIn("sternheimer_channel_mpi", gate)
        self.assertIn('mpirun -np "$SLURM_NTASKS" -ppn 1', gate)
        self.assertIn("validate_targets.py", gate)
        self.assertIn("solved_equations 428", gate)


if __name__ == "__main__":
    unittest.main()
