from dataclasses import replace
from pathlib import Path
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from IO.read_sternheimer import read_sternheimer
from IO.read_sternheimer_source import read_sternheimer_source
from sternheimer_data import PrimitiveBlock
from sternheimer_source_pair import pair_response_and_source


FIXTURES = Path(__file__).resolve().parent / "fixtures"
RESPONSE_FIXTURE = FIXTURES / "sternheimer_matrix_v1.dat"
SOURCE_FIXTURE = FIXTURES / "sternheimer_source_v1.dat"


class SternheimerSourcePairTest(unittest.TestCase):
    def setUp(self):
        source = read_sternheimer_source(SOURCE_FIXTURE)
        response = read_sternheimer(RESPONSE_FIXTURE)
        self.source = source
        self.response = replace(
            response,
            auxiliary_channel=torch.tensor([0, 1], dtype=torch.int64),
            provenance=dict(source.provenance),
        )

    def test_pairs_every_unique_response_key(self):
        pair = pair_response_and_source(self.response, self.source)

        self.assertIs(pair.response, self.response)
        self.assertIs(pair.source, self.source)
        self.assertEqual(
            pair.source_row_for_response_key,
            {(0, 0): 0, (0, 1): 1},
        )
        self.assertEqual(pair.provenance_warnings, ())

    def test_rejects_primitive_block_structure_mismatch(self):
        response = replace(
            self.response,
            blocks=(
                PrimitiveBlock("H", 0, 0, 0, 1, 0),
                PrimitiveBlock("H", 0, 1, 0, 3, 1),
            ),
        )
        with self.assertRaisesRegex(ValueError, "primitive blocks differ"):
            pair_response_and_source(response, self.source)

    def test_rejects_grid_volume_difference(self):
        source = replace(self.source, grid_volume_bohr3=0.25)
        with self.assertRaisesRegex(ValueError, "grid_volume_bohr3 differ"):
            pair_response_and_source(self.response, source)

    def test_rejects_overlap_maximum_absolute_difference(self):
        overlap = self.response.overlap.clone()
        overlap[0, 0] += 2.0e-13
        response = replace(self.response, overlap=overlap)

        with self.assertRaisesRegex(ValueError, "maximum absolute difference"):
            pair_response_and_source(response, self.source)

    def test_accepts_roundoff_scaled_to_large_overlap(self):
        source_overlap = self.source.overlap.clone() * 50.0
        response_overlap = source_overlap.clone()
        response_overlap[0, 0] += 7.0e-14

        pair_response_and_source(
            replace(self.response, overlap=response_overlap),
            replace(self.source, overlap=source_overlap),
        )

    def test_rejects_overlap_relative_difference_independently(self):
        source_overlap = torch.eye(4, dtype=torch.complex128) * 1.0e-2
        response_overlap = source_overlap.clone()
        response_overlap[0, 0] += 5.0e-15
        source = replace(self.source, overlap=source_overlap)
        response = replace(self.response, overlap=response_overlap)

        with self.assertRaisesRegex(ValueError, "relative Frobenius difference"):
            pair_response_and_source(response, source)

    def test_rejects_each_physical_provenance_difference(self):
        replacements = {
            "cell_bohr": [21.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0],
            "ecut_ry": 50.0,
            "kernel": "cut_coulomb",
            "orbital_sha256": "f" * 64,
            "pseudopotential_sha256": "0" * 64,
            "spin_convention": "different_spin_convention",
            "exx_pca_thr": 1.0e-6,
            "auxiliary_whitening": "different_whitening",
            "raw_auxiliary_dimension": 3,
            "whitened_auxiliary_rank": 3,
            "discarded_auxiliary_rank": 1,
            "coulomb_relative_threshold": 1.0e-8,
            "coulomb_transform_sha256": "1" * 64,
        }
        for key, value in replacements.items():
            with self.subTest(key=key):
                provenance = dict(self.source.provenance)
                provenance[key] = value
                source = replace(self.source, provenance=provenance)
                with self.assertRaisesRegex(ValueError, key):
                    pair_response_and_source(self.response, source)

    def test_accepts_legacy_auxiliary_hash_for_identical_whitened_space(self):
        provenance = dict(self.source.provenance)
        provenance["auxiliary_basis_sha256"] = "e" * 64
        pair = pair_response_and_source(
            self.response,
            replace(self.source, provenance=provenance),
        )

        self.assertEqual(len(pair.provenance_warnings), 1)
        self.assertIn("auxiliary_basis_sha256", pair.provenance_warnings[0])

    def test_rejects_auxiliary_hash_and_whitened_space_difference(self):
        provenance = dict(self.source.provenance)
        provenance["auxiliary_basis_sha256"] = "e" * 64
        provenance["coulomb_transform_sha256"] = "f" * 64

        with self.assertRaisesRegex(ValueError, "auxiliary_basis_sha256"):
            pair_response_and_source(
                self.response,
                replace(self.source, provenance=provenance),
            )

    def test_rejects_missing_physical_provenance(self):
        provenance = dict(self.source.provenance)
        del provenance["coulomb_transform_sha256"]
        source = replace(self.source, provenance=provenance)

        with self.assertRaisesRegex(
            ValueError, "missing physical provenance.*coulomb_transform_sha256"
        ):
            pair_response_and_source(self.response, source)

    def test_warns_for_execution_provenance_differences_only(self):
        for key, value in (
            ("abacus_commit", "2" * 40),
            ("executable_sha256", "3" * 64),
            ("mpi_ranks", 8),
            ("omp_threads", 16),
        ):
            with self.subTest(key=key):
                provenance = dict(self.source.provenance)
                provenance[key] = value
                source = replace(self.source, provenance=provenance)
                pair = pair_response_and_source(self.response, source)
                self.assertEqual(len(pair.provenance_warnings), 1)
                self.assertIn(key, pair.provenance_warnings[0])

    def test_rejects_unlisted_provenance_difference(self):
        response_provenance = dict(self.response.provenance)
        source_provenance = dict(self.source.provenance)
        response_provenance["sternheimer_nfreq"] = 16
        source_provenance["sternheimer_nfreq"] = 6

        with self.assertRaisesRegex(ValueError, "sternheimer_nfreq"):
            pair_response_and_source(
                replace(self.response, provenance=response_provenance),
                replace(self.source, provenance=source_provenance),
            )

    def test_rejects_missing_or_extra_source_key(self):
        missing_source = replace(
            self.source,
            occupied_state=self.source.occupied_state[:1],
            auxiliary_channel=self.source.auxiliary_channel[:1],
            occupation=self.source.occupation[:1],
            norm=self.source.norm[:1],
            d=self.source.d[:1],
        )
        with self.assertRaisesRegex(ValueError, "source keys differ"):
            pair_response_and_source(self.response, missing_source)

        extra_source = replace(
            self.source,
            occupied_state=torch.tensor([0, 0, 0], dtype=torch.int64),
            auxiliary_channel=torch.tensor([0, 1, 2], dtype=torch.int64),
            occupation=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
            norm=torch.tensor([1.2, 0.8, 0.7], dtype=torch.float64),
            d=torch.cat((self.source.d, self.source.d[:1]), dim=0),
        )
        with self.assertRaisesRegex(ValueError, "source keys differ"):
            pair_response_and_source(self.response, extra_source)

    def test_enforces_occupation_tolerance(self):
        within = self.source.occupation.clone()
        within[0] += 8.0e-15
        pair_response_and_source(
            self.response, replace(self.source, occupation=within)
        )

        outside = self.source.occupation.clone()
        outside[0] += 2.0e-14
        with self.assertRaisesRegex(ValueError, "occupation differs"):
            pair_response_and_source(
                self.response, replace(self.source, occupation=outside)
            )

    def test_rejects_noncontiguous_response_channels(self):
        response = replace(
            self.response,
            auxiliary_channel=torch.tensor([0, 2], dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "response channel IDs"):
            pair_response_and_source(response, self.source)


if __name__ == "__main__":
    unittest.main()
