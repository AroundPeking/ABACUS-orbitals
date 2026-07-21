from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from sternheimer_data import PrimitiveBlock
from test_sternheimer_spillage import make_sternheimer_data


EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "example_H_sternheimer"
    / "greedy_response_selection"
)
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from validate_targets import (  # noqa: E402
    parse_abfs_channels,
    parse_sternheimer_status,
    validate_response_target,
)


def make_complete_target():
    blocks = []
    offset = 0
    for l in range(3):
        for m in range(-l, l + 1):
            blocks.append(PrimitiveBlock("H", 0, l, m, 2, offset))
            offset += 2

    nfreq = 2
    naux = 3
    nreference = nfreq * naux
    data = make_sternheimer_data(
        blocks,
        torch.zeros((nreference, offset), dtype=torch.complex128),
    )
    return replace(
        data,
        occupied_state=torch.zeros(nreference, dtype=torch.int64),
        auxiliary_channel=torch.arange(naux, dtype=torch.int64).repeat_interleave(
            nfreq
        ),
        frequency_ha=torch.tensor([0.25, 0.75], dtype=torch.float64).repeat(
            naux
        ),
        provenance={
            **data.provenance,
            "kernel": "full_coulomb",
            "auxiliary_basis_sha256": "a" * 64,
            "auxiliary_whitening": "global_full_coulomb_v1",
            "raw_auxiliary_dimension": 4,
            "whitened_auxiliary_rank": 3,
            "discarded_auxiliary_rank": 1,
            "coulomb_relative_threshold": 1.0e-10,
            "coulomb_eigenvalues": [1.0e-12, 1.0, 2.0, 4.0],
            "coulomb_max_orthonormality_error": 1.0e-14,
            "coulomb_transform_sha256": "d" * 64,
        },
    )


def complete_channel_text():
    lines = [
        "# ABACUS Sternheimer ABFS channel diagnostic",
        "# channel atom atom_local type l radial m label max_abs",
    ]
    channel = 0
    for l in range(2):
        for m in range(2 * l + 1):
            lines.append(
                f"{channel} 0 {channel} 0 {l} 0 {m} H0-L{l}-N0-M{m} 1.0"
            )
            channel += 1
    return "\n".join(lines) + "\n"


def complete_status():
    return {
        "status": "success",
        "format": "siab_v1",
        "abfs_source": "explicit_abfs",
        "abfs_channels": "4",
        "response_channels": "3",
        "raw_auxiliary_dimension": "4",
        "whitened_auxiliary_rank": "3",
        "discarded_auxiliary_rank": "1",
        "coulomb_transform_sha256": "d" * 64,
        "auxiliary_basis_sha256": "a" * 64,
        "primitive_representation": "serial_reciprocal_pw_v1",
        "primitive_count": "18",
        "primitive_reciprocal_count": "97",
        "estimated_dense_memory_bytes": "4096",
        "slurm_memory_per_node_bytes": "8192",
        "memory_diagnostic": "STERNHEIMER_SIAB_MEMORY.dat",
        "target_file": "sternheimer_matrix.dat",
    }


class ResponseTargetValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_accepts_complete_primitive_frequency_and_explicit_abfs_contract(self):
        channels = parse_abfs_channels(
            self.write("STERNHEIMER_ABFS_CHANNELS.dat", complete_channel_text())
        )
        status = parse_sternheimer_status(
            self.write(
                "STERNHEIMER_ABACUS_STATUS.dat",
                "\n".join(f"{key} {value}" for key, value in complete_status().items())
                + "\n",
            )
        )

        summary = validate_response_target(
            make_complete_target(),
            channels,
            status,
            expected_atoms=1,
            primitive_lmax=2,
            radial_primitives=2,
            expected_nfreq=2,
            auxiliary_radial_counts=(1, 1),
            expected_auxiliary_sha256="a" * 64,
        )

        self.assertEqual(summary.primitive_columns, 18)
        self.assertEqual(summary.auxiliary_channels, 4)
        self.assertEqual(summary.response_channels, 3)
        self.assertEqual(summary.frequencies, (0.25, 0.75))
        self.assertEqual(summary.abfs_source, "explicit_abfs")

    def test_rejects_auxiliary_basis_hash_mismatch(self):
        channels = parse_abfs_channels(
            self.write("channels.dat", complete_channel_text())
        )

        with self.assertRaisesRegex(ValueError, "auxiliary basis SHA256"):
            validate_response_target(
                make_complete_target(),
                channels,
                complete_status(),
                expected_atoms=1,
                primitive_lmax=2,
                radial_primitives=2,
                expected_nfreq=2,
                auxiliary_radial_counts=(1, 1),
                expected_auxiliary_sha256="b" * 64,
            )

    def test_rejects_incompatible_primitive_representation(self):
        channels = parse_abfs_channels(
            self.write("channels.dat", complete_channel_text())
        )
        bad_status = {
            **complete_status(),
            "primitive_representation": "dense_real_grid_v0",
        }

        with self.assertRaisesRegex(ValueError, "reciprocal PW primitive"):
            validate_response_target(
                make_complete_target(),
                channels,
                bad_status,
                expected_atoms=1,
                primitive_lmax=2,
                radial_primitives=2,
                expected_nfreq=2,
                auxiliary_radial_counts=(1, 1),
            )

    def test_rejects_product_pca_status_even_when_dimensions_match(self):
        channels = parse_abfs_channels(
            self.write("channels.dat", complete_channel_text())
        )
        bad_status = {**complete_status(), "abfs_source": "product_pca"}
        status = parse_sternheimer_status(
            self.write(
                "status.dat",
                "\n".join(f"{key} {value}" for key, value in bad_status.items())
                + "\n",
            )
        )

        with self.assertRaisesRegex(ValueError, "explicit_abfs"):
            validate_response_target(
                make_complete_target(),
                channels,
                status,
                expected_atoms=1,
                primitive_lmax=2,
                radial_primitives=2,
                expected_nfreq=2,
                auxiliary_radial_counts=(1, 1),
            )

    def test_rejects_missing_g_like_primitive_multiplet(self):
        data = make_complete_target()
        shortened = tuple(
            block
            for block in data.blocks
            if not (block.l == 2 and block.m == 2)
        )
        repaired = []
        offset = 0
        for block in shortened:
            repaired.append(replace(block, offset=offset))
            offset += block.n_primitive
        data = replace(
            data,
            blocks=tuple(repaired),
            q=data.q[:, :offset],
            overlap=data.overlap[:offset, :offset],
        )
        channels = parse_abfs_channels(
            self.write("channels.dat", complete_channel_text())
        )
        status = complete_status()

        with self.assertRaisesRegex(ValueError, "incomplete primitive m group"):
            validate_response_target(
                data,
                channels,
                status,
                expected_atoms=1,
                primitive_lmax=2,
                radial_primitives=2,
                expected_nfreq=2,
                auxiliary_radial_counts=(1, 1),
            )

    def test_rejects_incomplete_frequency_cartesian_product(self):
        data = make_complete_target()
        data = replace(
            data,
            auxiliary_channel=data.auxiliary_channel.clone(),
            frequency_ha=data.frequency_ha.clone(),
        )
        data.auxiliary_channel[-1] = 1
        channels = parse_abfs_channels(
            self.write("channels.dat", complete_channel_text())
        )

        with self.assertRaisesRegex(ValueError, "reference Cartesian product"):
            validate_response_target(
                data,
                channels,
                complete_status(),
                expected_atoms=1,
                primitive_lmax=2,
                radial_primitives=2,
                expected_nfreq=2,
                auxiliary_radial_counts=(1, 1),
            )

    def test_rejects_zero_high_l_auxiliary_potential(self):
        text = complete_channel_text().replace(
            "H0-L1-N0-M1 1.0", "H0-L1-N0-M1 0.0"
        )
        channels = parse_abfs_channels(self.write("channels.dat", text))

        with self.assertRaisesRegex(ValueError, "nonzero"):
            validate_response_target(
                make_complete_target(),
                channels,
                complete_status(),
                expected_atoms=1,
                primitive_lmax=2,
                radial_primitives=2,
                expected_nfreq=2,
                auxiliary_radial_counts=(1, 1),
            )


if __name__ == "__main__":
    unittest.main()
