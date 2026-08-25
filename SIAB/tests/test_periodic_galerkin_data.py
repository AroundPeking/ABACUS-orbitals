import hashlib
import os
import struct
import tempfile
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from periodic_galerkin_data import read_periodic_galerkin_dataset


HEADER = struct.Struct("<16sIIiiiQQ")
MAGIC = b"ABACUS_STBOPT_V1"


class PeriodicGalerkinDataTest(unittest.TestCase):
    def test_reads_complete_complex_periodic_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_fixture(directory)

            data = read_periodic_galerkin_dataset(directory)

            self.assertEqual(data.abacus_commit, "1" * 40)
            self.assertEqual(data.physics_hash, "6" * 64)
            self.assertEqual(data.selected_iq, 1)
            self.assertEqual(data.primitive_count, 2)
            self.assertEqual(data.raw_auxiliary_dimension, 1)
            self.assertEqual(data.whitened_auxiliary_rank, 1)
            self.assertEqual(len(data.primitive_blocks), 1)
            self.assertEqual(data.primitive_blocks[0].element, "C")
            self.assertEqual(data.primitive_blocks[0].n_primitive, 2)
            torch.testing.assert_close(
                data.frequency_ha,
                torch.tensor([0.25], dtype=torch.float64),
            )
            torch.testing.assert_close(
                data.reference_response,
                torch.tensor([[[-0.4 + 0.0j]]], dtype=torch.complex128),
            )

            record = data.kpoints[0]
            self.assertEqual(record.source_ik, 1)
            self.assertEqual(record.target_ik, 1)
            self.assertEqual(record.reciprocal_shift, (0, 0, 0))
            self.assertEqual(record.k_weight, 2.0)
            torch.testing.assert_close(
                record.source_eigenvalue_ha,
                torch.tensor([-0.5], dtype=torch.float64),
            )
            torch.testing.assert_close(
                record.occupation,
                torch.tensor([1.0], dtype=torch.float64),
            )
            self.assertEqual(record.overlap.shape, (2, 2))
            self.assertEqual(record.hamiltonian_ha.shape, (2, 2))
            self.assertEqual(record.occupied_projection.shape, (1, 2))
            self.assertEqual(record.source.shape, (1, 1, 2))
            self.assertEqual(record.reference_projection.shape, (1, 1, 1, 2))
            torch.testing.assert_close(
                record.hamiltonian_ha,
                torch.tensor(
                    [[-0.5 + 0.0j, 0.05 + 0.1j],
                     [0.05 - 0.1j, 0.7 + 0.0j]],
                    dtype=torch.complex128,
                ),
            )

    def test_rejects_dataset_without_exact_reference_response(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_fixture(directory)
            os.remove(os.path.join(directory, "reference_response_ifreq_0.bin"))

            with self.assertRaisesRegex(RuntimeError, "missing chunk"):
                read_periodic_galerkin_dataset(directory)

    def test_can_validate_without_retaining_reference_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_fixture(directory)

            data = read_periodic_galerkin_dataset(
                directory, include_reference_projection=False
            )

            self.assertEqual(data.kpoints[0].reference_projection.numel(), 0)
            torch.testing.assert_close(
                data.reference_response,
                torch.tensor([[[-0.4 + 0.0j]]], dtype=torch.complex128),
            )

            path = os.path.join(directory, "response_ik_1_ifreq_0.bin")
            with open(path, "r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                handle.write(b"\x01")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                read_periodic_galerkin_dataset(
                    directory, include_reference_projection=False
                )

    def write_fixture(self, directory):
        chunks = {
            "coulomb_metric.bin": (4, 0, -1, 1, 1, [1.0 + 0.0j]),
            "coulomb_whitening.bin": (5, 0, -1, 1, 1, [1.0 + 0.0j]),
            "reference_response_ifreq_0.bin": (8, 0, 0, 1, 1, [-0.4 + 0.0j]),
            "overlap_ik_1.bin": (
                1,
                1,
                -1,
                2,
                2,
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ),
            "hamiltonian_ik_1.bin": (
                6,
                1,
                -1,
                2,
                2,
                [-1.0 + 0.0j, 0.1 + 0.2j, 0.1 - 0.2j, 1.4 + 0.0j],
            ),
            "occupied_projection_ik_1.bin": (
                7,
                1,
                -1,
                1,
                2,
                [1.0 + 0.0j, 0.0 + 0.0j],
            ),
            "source_ik_1.bin": (
                2,
                1,
                -1,
                1,
                2,
                [0.0 + 0.0j, 0.3 + 0.1j],
            ),
            "response_ik_1_ifreq_0.bin": (
                3,
                1,
                0,
                1,
                2,
                [0.0 + 0.0j, -0.2 + 0.05j],
            ),
        }
        entries = []
        for path, (kind, ik, ifrequency, rows, columns, values) in chunks.items():
            full_path = os.path.join(directory, path)
            with open(full_path, "wb") as handle:
                handle.write(
                    HEADER.pack(
                        MAGIC,
                        1,
                        kind,
                        1,
                        ik,
                        ifrequency,
                        rows,
                        columns,
                    )
                )
                for value in values:
                    handle.write(struct.pack("<dd", value.real, value.imag))
            digest = self.sha256(full_path)
            q_weight = 1.0
            k_weight = 1.0 if ik == 0 else 2.0
            frequency = 0.25 if ifrequency == 0 else -1.0
            entries.append(
                f"entry\t{kind}\t1\t{ik}\t{ifrequency}\t{rows}\t{columns}"
                f"\t{q_weight:.17e}\t{k_weight:.17e}\t{frequency:.17e}"
                f"\t{path}\t{digest}"
            )

        primitive_path = os.path.join(directory, "primitive_blocks.dat")
        with open(primitive_path, "w", encoding="ascii") as handle:
            handle.write("ABACUS_STERNHEIMER_BASIS_OPT_PRIMITIVES_V1\n")
            handle.write("# element atom_index l m n_primitive offset\n")
            handle.write("C 0 0 0 2 0\n")

        manifest = [
            "ABACUS_STERNHEIMER_BASIS_OPT_MANIFEST_V1",
            "abacus_commit " + "1" * 40,
            "executable_sha256 " + "2" * 64,
            "orbital_sha256 " + "3" * 64,
            "pseudopotential_sha256 " + "4" * 64,
            "auxiliary_basis_sha256 " + "5" * 64,
            "primitive_blocks_sha256 " + self.sha256(primitive_path),
            "physics_hash " + "6" * 64,
            "kernel full_coulomb",
            "q_count 1",
            "selected_iq 1",
            "k_count 1",
            "frequency_count 1",
            "raw_auxiliary_dimension 1",
            "whitened_auxiliary_rank 1",
            "discarded_auxiliary_rank 0",
            "coulomb_relative_threshold 1.0e-10",
            "coulomb_max_orthonormality_error 0.0",
            "coulomb_transform_sha256 " + "8" * 64,
            "primitive_count 2",
            f"entry_count {len(entries)}",
            "qpoint 0.0 0.0 0.0",
            "q_weight 1.0",
            "frequency 0 2.5e-1 1.0",
            "kpoint 1 1 0.0 0.0 0.0 0.0 0.0 0.0 0 0 0 2.0 1 1.0",
            "eigenvalues_ry 1 1 -1.0",
            *entries,
        ]
        with open(os.path.join(directory, "manifest.dat"), "w", encoding="ascii") as handle:
            handle.write("\n".join(manifest) + "\n")
        with open(os.path.join(directory, "status.dat"), "w", encoding="ascii") as handle:
            handle.write("status success\n")
            handle.write("all_converged yes\n")
            handle.write("physics_hash " + "6" * 64 + "\n")

    @staticmethod
    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read())
        return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
