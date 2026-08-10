"""Tests for the matched TZDP/response-optimized H2 SOS campaign."""

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / "example_H_sternheimer" / "held_out_h2_sos_response_3s3p2d"
sys.path.insert(0, str(CAMPAIGN_DIR))

from prepare_campaign import prepare_campaign


TEMPLATE_ROOT = (
    ROOT
    / "example_H_sternheimer"
    / "held_out_h2_sos_greedy_full"
    / "cases"
)
BASELINE_ORBITAL = (
    ROOT.parent
    / "Dojo-NC-SR"
    / "Orbitals_v2.0"
    / "H_TZDP"
    / "H_gga_8au_100Ry_3s2p.orb"
)
PSEUDO = ROOT.parent / "SG15_v1.0" / "Pseudopotential" / "H_ONCV_PBE-1.0.upf"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_values(path):
    return {
        fields[0]: fields[1]
        for line in path.read_text(encoding="ascii").splitlines()
        if len(fields := line.split()) >= 2 and fields[0] != "INPUT_PARAMETERS"
    }


def _minimal_orbital(radial_counts):
    labels = ("S", "P", "D", "F", "G")
    return "\n".join(
        [
            "---------------------------------------------------------------------------",
            "Element                     H",
            "Energy Cutoff(Ry)           100",
            "Radius Cutoff(a.u.)         8.0",
            f"Lmax                        {len(radial_counts) - 1}",
            *[
                f"Number of {labels[l]}orbital-->       {count}"
                for l, count in enumerate(radial_counts)
            ],
            "---------------------------------------------------------------------------",
            "SUMMARY  END",
            "",
            "Mesh                        801",
            "dr                          0.01",
            "",
        ]
    )


class H2ResponseSOSCampaignTest(unittest.TestCase):
    def test_server66_runner_uses_mpi_with_one_thread_per_rank(self):
        case_runner = (CAMPAIGN_DIR / "run_case_66.sh").read_text(encoding="ascii")
        campaign_runner = (CAMPAIGN_DIR / "run_campaign_66.sh").read_text(
            encoding="ascii"
        )

        self.assertNotIn("#SBATCH", case_runner)
        self.assertLess(
            case_runner.index('source "$HOME/.bashrc"'),
            case_runner.index("set -euo pipefail"),
        )
        self.assertIn('mpirun -np "$mpi_ranks" -ppn "$mpi_ranks"', case_runner)
        self.assertIn("OMP_NUM_THREADS=1", case_runner)
        self.assertNotIn("OMP_NUM_THREADS=$threads", case_runner)
        self.assertIn("dcf5e649bd68d31e7a57d150a50c65c05694b91361ba277ebbe9f228242e7d4b", case_runner)
        self.assertIn("00db48f2d90db43828826a4a4bdb6e9f666e7c92ad4f197247283e83cbf94f40", case_runner)
        self.assertIn("libRPA finished successfully", case_runner)
        self.assertIn("Total EcRPA:", case_runner)
        self.assertIn("mpi_ranks=${2:-8}", campaign_runner)
        self.assertIn("max_parallel=${3:-4}", campaign_runner)
        self.assertIn("max_parallel * mpi_ranks", campaign_runner)
        self.assertEqual(campaign_runner.count('run_case_66.sh"'), 1)
        self.assertIn("baseline_tzdp:H", campaign_runner)
        self.assertIn("optimized_3s3p2d:H_ghost", campaign_runner)
        self.assertIn("&", campaign_runner)

    def test_prepares_six_matched_full_band_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            optimized = directory / "H_response_3s3p2d.orb"
            optimized.write_text(_minimal_orbital((3, 3, 2)), encoding="ascii")
            auxiliary = directory / "H_pca1e-4.abfs"
            auxiliary.write_text("fixed auxiliary basis\n", encoding="ascii")
            campaign = directory / "campaign"

            manifest = prepare_campaign(
                TEMPLATE_ROOT,
                campaign,
                BASELINE_ORBITAL,
                optimized,
                PSEUDO,
                auxiliary,
                source_commit="test-source",
                expected_sha256={
                    "baseline_orbital": _sha256(BASELINE_ORBITAL),
                    "optimized_orbital": _sha256(optimized),
                    "pseudopotential": _sha256(PSEUDO),
                    "auxiliary_basis": _sha256(auxiliary),
                },
            )

            self.assertEqual(manifest["physics"]["nfreq"], 16)
            self.assertEqual(manifest["physics"]["ecutwfc_ry"], 100)
            self.assertEqual(manifest["physics"]["coulomb_kernel"], "full")
            self.assertEqual(manifest["physics"]["auxiliary_basis_pca_threshold"], 1.0e-4)
            self.assertEqual(len(manifest["cases"]), 6)
            expected_bands = {
                ("baseline_tzdp", "H"): 9,
                ("baseline_tzdp", "H2"): 18,
                ("baseline_tzdp", "H_ghost"): 18,
                ("optimized_3s3p2d", "H"): 22,
                ("optimized_3s3p2d", "H2"): 44,
                ("optimized_3s3p2d", "H_ghost"): 44,
            }
            for (lane, case), bands in expected_bands.items():
                case_dir = campaign / lane / case
                values = _input_values(case_dir / "INPUT")
                self.assertEqual(int(values["nbands"]), bands)
                self.assertEqual(values["ecutwfc"], "100")
                self.assertEqual(values["exx_pca_threshold"], "10")
                self.assertEqual(values["rpa_ccp_rmesh_times"], "5")
                librpa = (case_dir / "librpa.in").read_text(encoding="ascii")
                self.assertIn("nfreq = 16", librpa)
                self.assertIn("prefix_coul_full = v1_coulomb_full_iq_", librpa)
                self.assertIn("vq_threshold = 0", librpa)

            on_disk = json.loads(
                (campaign / "campaign.json").read_text(encoding="ascii")
            )
            self.assertEqual(on_disk, manifest)

    def test_rejects_an_asset_hash_mismatch_before_creating_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            optimized = directory / "H_response_3s3p2d.orb"
            optimized.write_text(_minimal_orbital((3, 3, 2)), encoding="ascii")
            auxiliary = directory / "H_pca1e-4.abfs"
            auxiliary.write_text("fixed auxiliary basis\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "optimized_orbital SHA256"):
                prepare_campaign(
                    TEMPLATE_ROOT,
                    directory / "campaign",
                    BASELINE_ORBITAL,
                    optimized,
                    PSEUDO,
                    auxiliary,
                    source_commit="test-source",
                    expected_sha256={
                        "baseline_orbital": _sha256(BASELINE_ORBITAL),
                        "optimized_orbital": "0" * 64,
                        "pseudopotential": _sha256(PSEUDO),
                        "auxiliary_basis": _sha256(auxiliary),
                    },
                )


if __name__ == "__main__":
    unittest.main()
