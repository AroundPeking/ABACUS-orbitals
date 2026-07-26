"""Regression tests for OpenMP Sternheimer target comparisons."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest import mock

import common  # noqa: F401 - configures the optimizer import path
from IO.read_sternheimer import read_sternheimer
import torch


GREEDY = (
    Path(__file__).resolve().parents[1]
    / "example_H_sternheimer"
    / "greedy_response_selection"
)
if str(GREEDY) not in sys.path:
    sys.path.insert(0, str(GREEDY))

import compare_sternheimer_targets as compare_targets


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sternheimer_matrix_v1.dat"


def with_provenance(data, **values):
    provenance = dict(data.provenance)
    provenance.update(values)
    return replace(data, provenance=provenance)


class CompareSternheimerTargetsTest(unittest.TestCase):
    def setUp(self):
        self.data = read_sternheimer(FIXTURE)

    def test_allows_only_omp_thread_count_to_differ(self):
        reference = with_provenance(self.data, omp_threads=16)
        candidate = with_provenance(self.data, omp_threads=24)

        with mock.patch.object(
            compare_targets,
            "read_sternheimer",
            side_effect=(reference, candidate),
        ):
            result = compare_targets.compare("omp16.dat", "omp24.dat")

        self.assertEqual(
            result["provenance"],
            {"reference_omp_threads": 16, "candidate_omp_threads": 24},
        )
        self.assertEqual(result["q"]["max_abs"], 0.0)
        self.assertEqual(result["overlap"]["relative_frobenius"], 0.0)

    def test_rejects_any_physical_provenance_difference(self):
        reference = with_provenance(self.data, omp_threads=16)
        candidate = with_provenance(self.data, omp_threads=24, ecut_ry=50.0)

        with mock.patch.object(
            compare_targets,
            "read_sternheimer",
            side_effect=(reference, candidate),
        ):
            with self.assertRaisesRegex(ValueError, "ecut_ry"):
                compare_targets.compare("omp16.dat", "omp24.dat")

    def test_rejects_mpi_rank_count_difference_by_default(self):
        reference = with_provenance(self.data, mpi_ranks=1)
        candidate = with_provenance(self.data, mpi_ranks=2)

        with mock.patch.object(
            compare_targets,
            "read_sternheimer",
            side_effect=(reference, candidate),
        ):
            with self.assertRaisesRegex(ValueError, "mpi_ranks"):
                compare_targets.compare("serial.dat", "channel_mpi.dat")

    def test_allows_mpi_rank_count_difference_only_when_requested(self):
        reference = with_provenance(self.data, mpi_ranks=1)
        candidate = with_provenance(self.data, mpi_ranks=2)

        with mock.patch.object(
            compare_targets,
            "read_sternheimer",
            side_effect=(reference, candidate),
        ):
            result = compare_targets.compare(
                "serial.dat",
                "channel_mpi.dat",
                allow_mpi_ranks_differ=True,
            )

        self.assertEqual(result["provenance"]["reference_mpi_ranks"], 1)
        self.assertEqual(result["provenance"]["candidate_mpi_ranks"], 2)
        self.assertEqual(result["q"]["max_abs"], 0.0)

    def test_aligns_one_common_phase_for_each_occupied_state(self):
        occupied_state = torch.tensor([0, 1], dtype=torch.int64)
        reference = replace(self.data, occupied_state=occupied_state)
        candidate_q = reference.q.clone()
        candidate_q[0] *= -1.0
        candidate_q[1] *= 1.0j
        candidate = replace(reference, q=candidate_q)

        with mock.patch.object(
            compare_targets,
            "read_sternheimer",
            side_effect=(reference, candidate),
        ):
            result = compare_targets.compare(
                "serial.dat",
                "channel_mpi.dat",
                align_occupied_state_phase=True,
            )

        self.assertLess(result["q"]["max_abs"], 1.0e-15)
        self.assertLess(result["q"]["relative_frobenius"], 1.0e-15)
        self.assertEqual(
            [entry["occupied_state"] for entry in result["q_phase_alignment"]],
            [0, 1],
        )

    def test_does_not_hide_a_single_channel_phase_error(self):
        candidate_q = -self.data.q.clone()
        candidate_q[0] *= -1.0
        candidate = replace(self.data, q=candidate_q)

        with mock.patch.object(
            compare_targets,
            "read_sternheimer",
            side_effect=(self.data, candidate),
        ):
            result = compare_targets.compare(
                "serial.dat",
                "channel_mpi.dat",
                align_occupied_state_phase=True,
            )

        self.assertGreater(result["q"]["max_abs"], 0.1)
        self.assertGreater(result["q"]["relative_frobenius"], 0.1)


if __name__ == "__main__":
    unittest.main()
