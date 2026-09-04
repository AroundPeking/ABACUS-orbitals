import hashlib
import importlib.util
import io
import json
import math
import struct
import sys
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    ROOT
    / "example_C_sternheimer"
    / "periodic_basis_optimization"
    / "galerkin_binding_workflow"
    / "validate_periodic_basis_opt_streaming.py"
)
SPEC = importlib.util.spec_from_file_location("validate_periodic_basis_opt_streaming", VALIDATOR_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
sys.modules["validate_periodic_basis_opt_streaming"] = VALIDATOR

ACCEPTOR_PATH = (
    ROOT
    / "example_C_sternheimer"
    / "periodic_basis_optimization"
    / "galerkin_binding_workflow"
    / "accept_periodic_basis_opt_streaming_validation.py"
)
ACCEPTOR_SPEC = importlib.util.spec_from_file_location(
    "accept_periodic_basis_opt_streaming_validation", ACCEPTOR_PATH
)
ACCEPTOR = importlib.util.module_from_spec(ACCEPTOR_SPEC)
ACCEPTOR_SPEC.loader.exec_module(ACCEPTOR)

SIGNATURE_PATH = (
    ROOT
    / "example_C_sternheimer"
    / "periodic_basis_optimization"
    / "galerkin_binding_workflow"
    / "basis_opt_response_signature.py"
)
SIGNATURE_SPEC = importlib.util.spec_from_file_location(
    "basis_opt_response_signature", SIGNATURE_PATH
)
SIGNATURE = importlib.util.module_from_spec(SIGNATURE_SPEC)
SIGNATURE_SPEC.loader.exec_module(SIGNATURE)


HEADER = struct.Struct("<16sIIiiiQQ")
MAGIC = b"ABACUS_STBOPT_V1"


def write_chunk(path, kind, iq, ik, ifrequency, rows, columns, values):
    path = Path(path)
    with path.open("wb") as handle:
        handle.write(HEADER.pack(MAGIC, 1, kind, iq, ik, ifrequency, rows, columns))
        for value in values:
            handle.write(struct.pack("<dd", value.real, value.imag))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StreamingChunkTest(unittest.TestCase):
    def test_large_chunk_scan_has_bounded_python_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.bin"
            value_count = 1_000_000
            with path.open("wb") as handle:
                handle.write(HEADER.pack(MAGIC, 1, 3, 1, 1, 0, 1000, 1000))
                block = struct.pack("<dd", 0.25, -0.5) * 4096
                remaining = value_count
                while remaining:
                    count = min(remaining, 4096)
                    handle.write(block[: count * 16])
                    remaining -= count
            expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

            tracemalloc.start()
            chunk = VALIDATOR.scan_chunk(path, expected_sha256=expected_sha)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.assertEqual(chunk["rows"], 1000)
            self.assertEqual(chunk["columns"], 1000)
            self.assertNotIn("values", chunk)
            self.assertLess(peak, 8 * 1024 * 1024)

    def test_non_finite_value_is_rejected_across_scan_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonfinite.bin"
            values = [0j] * 65537
            values[-1] = complex(math.nan, 0.0)
            digest = write_chunk(path, 3, 1, 1, 0, 1, len(values), values)
            with self.assertRaisesRegex(RuntimeError, "non-finite payload"):
                VALIDATOR.scan_chunk(path, expected_sha256=digest, block_bytes=4096)


class StreamingDatasetTest(unittest.TestCase):
    def _make_dataset(self, root):
        root = Path(root)
        chunks = [
            (4, 0, -1, 2, 2, [1 + 0j, 0j, 0j, 1 + 0j], "metric.bin"),
            (5, 0, -1, 2, 2, [1 + 0j, 0j, 0j, 1 + 0j], "transform.bin"),
            (8, 0, 0, 2, 2, [-0.2 + 0j, 0.01j, -0.01j, -0.1 + 0j], "reference.bin"),
            (1, 1, -1, 2, 2, [1 + 0j, 0j, 0j, 1 + 0j], "overlap.bin"),
            (2, 1, -1, 2, 2, [1 + 0j, 0j, 0j, 1 + 0j], "source.bin"),
            (6, 1, -1, 2, 2, [-1 + 0j, 0j, 0j, 2 + 0j], "hamiltonian.bin"),
            (7, 1, -1, 1, 2, [1 + 0j, 0j], "occupied.bin"),
            (3, 1, 0, 2, 2, [0.5 + 0j, 0j, 0j, 0.25 + 0j], "response.bin"),
        ]
        entries = []
        for kind, ik, ifrequency, rows, columns, values, name in chunks:
            digest = write_chunk(root / name, kind, 1, ik, ifrequency, rows, columns, values)
            entries.append(
                f"entry {kind} 1 {ik} {ifrequency} {rows} {columns} 1.0 2.0 0.5 {name} {digest}"
            )
        manifest = [
            "ABACUS_STERNHEIMER_BASIS_OPT_MANIFEST_V1",
            "abacus_commit test-commit",
            "executable_sha256 test-executable",
            "orbital_sha256 test-orbital",
            "pseudopotential_sha256 test-pseudopotential",
            "auxiliary_basis_source product_pca",
            "auxiliary_basis_sha256 test-auxiliary",
            "primitive_blocks_sha256 test-primitives",
            "kernel full_coulomb",
            "q_count 1",
            "selected_iq 1",
            "qpoint 0.0 0.0 0.0",
            "q_weight 1.0",
            "entry_count 8",
            "raw_auxiliary_dimension 2",
            "whitened_auxiliary_rank 2",
            "primitive_count 2",
            "coulomb_max_orthonormality_error 0.0",
            "physics_hash test-physics",
            "frequency 0 0.5 1.0",
            "kpoint 1 1 0 0 0 0 0 0 0 0 0 2.0 1 1.0",
            "eigenvalues_ry 1 1 -0.5",
            *entries,
        ]
        (root / "manifest.dat").write_text("\n".join(manifest) + "\n", encoding="ascii")
        (root / "status.dat").write_text(
            "status success\nall_converged yes\nphysics_hash test-physics\n",
            encoding="ascii",
        )

    def test_main_matches_legacy_validation_fields_on_small_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_dataset(tmp)
            output = io.StringIO()
            with redirect_stdout(output):
                payload = VALIDATOR.main([tmp, "--commit", "test-commit"])
            printed = json.loads(output.getvalue())

            self.assertEqual(payload, printed)
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["entries"], 8)
            self.assertEqual(payload["kpoints"], 1)
            self.assertEqual(payload["frequencies"], 1)
            self.assertEqual(payload["raw_auxiliary_dimension"], 2)
            self.assertEqual(payload["whitened_auxiliary_rank"], 2)
            self.assertEqual(payload["primitive_count"], 2)
            self.assertEqual(payload["metric_hermitian_relative_error"], 0.0)
            self.assertEqual(payload["sampled_whitening_max_error"], 0.0)
            self.assertEqual(payload["reference_response_hermitian_relative_error"], 0.0)
            self.assertEqual(payload["overlap_hermitian_relative_error"], 0.0)
            self.assertEqual(payload["hamiltonian_hermitian_relative_error"], 0.0)


class RunnerContractTest(unittest.TestCase):
    def test_standard_runners_use_the_repository_streaming_validator(self):
        workflow = VALIDATOR_PATH.parent
        runners = (
            "run_c_solid_fd8_q13_standard_q1_df.slurm",
            "run_c_solid_fd8_q13_standard_remaining_df.slurm",
            "run_c_solid_fd8_q13_standard_dfdcu.slurm",
        )
        for name in runners:
            text = (workflow / name).read_text(encoding="ascii")
            self.assertIn("validate_periodic_basis_opt_streaming.py", text, name)
            self.assertNotIn("$abacus_root/validate_periodic_basis_opt.py", text, name)
            self.assertNotIn(
                "$C_SOLID_DFDCU_STAGE_ROOT/validate_periodic_basis_opt.py",
                text,
                name,
            )
            self.assertIn(
                'expected_auxiliary_sha=$("$validator_python" -', text, name
            )
            self.assertIn('\n"$validator_python" "$validator"', text, name)
            self.assertNotIn("python3", text, name)
        dfdcu = (workflow / runners[-1]).read_text(encoding="ascii")
        self.assertIn("BASIS_OPT_VALIDATOR_SHA256", dfdcu)
        self.assertIn('sha256sum "$basis_opt_validator"', dfdcu)
        self.assertIn("DF_Q1_RESPONSE_SIGNATURE_SHA256", dfdcu)
        self.assertIn("DFDCU_Q1_RESPONSE_SIGNATURE_SHA256", dfdcu)
        self.assertIn("Q1_CROSS_HOST_RESPONSE_COMPARISON_SHA256", dfdcu)
        self.assertIn('sha256sum "$q1_root/DF_Q1_RESPONSE_SIGNATURE.json"', dfdcu)
        self.assertIn('sha256sum "$q1_root/DFDCU_Q1_RESPONSE_SIGNATURE.json"', dfdcu)
        self.assertIn(
            'sha256sum "$q1_root/Q1_CROSS_HOST_RESPONSE_COMPARISON.json"',
            dfdcu,
        )
        self.assertIn("#SBATCH --array=0-6%2", dfdcu)
        self.assertIn("q_labels=(2 3 6 7 8 11 28)", dfdcu)
        self.assertIn("selected_iq=(22 43 6 27 23 11 55)", dfdcu)
        self.assertIn("q_multiplicities=(8 4 6 24 12 3 6)", dfdcu)
        self.assertIn('case "$SLURM_JOB_PARTITION" in', dfdcu)
        self.assertIn("normal|long)", dfdcu)
        self.assertIn('echo scheduler_partition=$SLURM_JOB_PARTITION', dfdcu)
        self.assertNotIn('test "$SLURM_JOB_PARTITION" = normal', dfdcu)
        self.assertIn('set_input_key sternheimer_q_index "$iq"', dfdcu)
        self.assertIn('--expected-q-weight "$expected_q_weight"', dfdcu)
        self.assertNotIn("--expected-q-weight 0.015625", dfdcu)

    def test_recovery_runner_is_read_only_and_memory_gated(self):
        runner = (
            VALIDATOR_PATH.parent
            / "recover_c_solid_fd8_q_dataset_validation.slurm"
        ).read_text(encoding="ascii")
        self.assertIn("validate_periodic_basis_opt_streaming.py", runner)
        self.assertIn("REFERENCE_VALIDATION_JSON", runner)
        self.assertIn("MAX_VALIDATOR_RSS_KB", runner)
        self.assertNotIn("mpirun", runner)
        self.assertNotIn('"$abacus"', runner)

    def test_completed_scan_acceptor_runner_does_not_rescan_or_run_physics(self):
        runner = (
            VALIDATOR_PATH.parent
            / "accept_c_solid_fd8_q_streaming_validation_recovery.slurm"
        ).read_text(encoding="ascii")
        self.assertIn("accept_periodic_basis_opt_streaming_validation.py", runner)
        self.assertIn("BASIS_OPT_VALIDATION_JSON", runner)
        self.assertIn("VALIDATOR_TIME_FILE", runner)
        self.assertIn("EXPECTED_MANIFEST_SHA256", runner)
        self.assertIn("EXPECTED_STATUS_SHA256", runner)
        self.assertNotIn("validate_periodic_basis_opt_streaming.py\" \"$DATASET_ROOT", runner)
        self.assertNotIn("mpirun", runner)
        self.assertNotIn('"$abacus"', runner)

    def test_standard_protocol_recovery_is_read_only_and_uses_frozen_python(self):
        runner = (
            VALIDATOR_PATH.parent
            / "recover_c_solid_fd8_q_standard_validation_dfdcu.slurm"
        ).read_text(encoding="ascii")
        self.assertIn("validate_c_solid_fd8_q_dataset.py", runner)
        self.assertIn("STANDARD_PROTOCOL_VALIDATION.json", runner)
        self.assertIn("STANDARD_PROTOCOL_RECOVERY_ACCEPTANCE.json", runner)
        self.assertIn('expected_auxiliary_sha=$("$validator_python" -', runner)
        self.assertIn('\n"$validator_python" "$validator"', runner)
        self.assertIn("SOURCE_SCHEDULER_STATE", runner)
        self.assertIn("SOURCE_SCHEDULER_EXIT_CODE", runner)
        self.assertIn("BASIS_OPT_VALIDATION_JSON", runner)
        self.assertNotIn("mpirun", runner)
        self.assertNotIn('"$abacus"', runner)
        self.assertNotIn("python3", runner)


class CompletedStreamingValidationAcceptanceTest(unittest.TestCase):
    def _validation(self):
        return {
            "status": "success",
            "entries": 1038,
            "kpoints": 64,
            "frequencies": 12,
            "raw_auxiliary_dimension": 320,
            "whitened_auxiliary_rank": 301,
            "primitive_count": 1550,
            "metric_hermitian_relative_error": 9.0e-15,
            "declared_whitening_max_error": 6.2e-9,
            "sampled_whitening_max_error": 4.2e-8,
            "sampled_whitening_limit": 1.9e-6,
            "reference_response_hermitian_relative_error": 0.0,
            "overlap_hermitian_relative_error": 0.0,
            "hamiltonian_hermitian_relative_error": 1.8e-14,
        }

    def test_independent_host_roundoff_is_not_exact_parity_failure(self):
        reference = self._validation()
        actual = self._validation()
        actual.update(
            {
                "metric_hermitian_relative_error": 1.01e-14,
                "declared_whitening_max_error": 5.03e-9,
                "sampled_whitening_max_error": 4.05e-8,
                "sampled_whitening_limit": 1.51e-6,
                "overlap_hermitian_relative_error": 1.86e-16,
                "hamiltonian_hermitian_relative_error": 1.77e-14,
            }
        )

        result = ACCEPTOR.accept_validation(
            actual,
            reference,
            max_rss_kb=32488,
            max_rss_limit_kb=4194304,
        )

        self.assertEqual(
            result["status"],
            "success_recovered_from_completed_streaming_validation",
        )
        self.assertEqual(result["dimension_parity"], "pass")
        self.assertEqual(result["numerical_gate"], "pass")
        self.assertEqual(result["memory_gate"], "pass")
        self.assertNotEqual(
            result["cross_host_diagnostic_differences"][
                "declared_whitening_max_error"
            ],
            0.0,
        )

    def test_acceptor_rejects_dimension_mismatch(self):
        reference = self._validation()
        actual = self._validation()
        actual["primitive_count"] += 1
        with self.assertRaisesRegex(RuntimeError, "dimension mismatch"):
            ACCEPTOR.accept_validation(actual, reference, 1000, 4194304)

    def test_acceptor_rejects_failed_numerical_or_memory_gate(self):
        reference = self._validation()
        actual = self._validation()
        actual["sampled_whitening_max_error"] = 2.0e-6
        with self.assertRaisesRegex(RuntimeError, "numerical gate"):
            ACCEPTOR.accept_validation(actual, reference, 1000, 4194304)
        actual = self._validation()
        with self.assertRaisesRegex(RuntimeError, "memory gate"):
            ACCEPTOR.accept_validation(actual, reference, 5000, 4096)


@unittest.skipIf(VALIDATOR.np is None, "NumPy is required for response signatures")
class ResponseSignatureTest(unittest.TestCase):
    def _signature(self, eigenvalues):
        return {
            "status": "success",
            "protocol": {
                "abacus_commit": "test-commit",
                "orbital_sha256": "test-orbital",
                "pseudopotential_sha256": "test-pseudopotential",
                "auxiliary_basis_source": "product_pca",
                "auxiliary_basis_sha256": "test-auxiliary",
                "primitive_blocks_sha256": "test-primitives",
                "kernel": "full_coulomb",
                "q_count": 1,
                "selected_iq": 1,
                "qpoint": [0.0, 0.0, 0.0],
                "q_weight": 1.0,
                "raw_auxiliary_dimension": 2,
                "whitened_auxiliary_rank": 2,
                "primitive_count": 2,
                "kpoints": 1,
                "frequencies": [[0, 0.5, 1.0]],
            },
            "reference_response": [
                {
                    "ifrequency": 0,
                    "frequency": 0.5,
                    "weight": 1.0,
                    "trace_real": sum(eigenvalues),
                    "trace_imag": 0.0,
                    "frobenius_norm": math.sqrt(
                        sum(value * value for value in eigenvalues)
                    ),
                    "eigenvalues": list(eigenvalues),
                }
            ],
        }

    def test_unitary_invariant_response_signature_comparison_passes(self):
        reference = self._signature([-0.2, -0.1])
        candidate = self._signature([-0.2 * (1.0 + 1.0e-10), -0.1])
        result = SIGNATURE.compare_signatures(
            candidate, reference, relative_tolerance=1.0e-8
        )
        self.assertEqual(result["status"], "success")
        self.assertLess(result["max_response_spectrum_relative_error"], 1.0e-8)

    def test_response_signature_records_cross_build_physics_hash_difference(self):
        reference = self._signature([-0.2, -0.1])
        candidate = self._signature([-0.2, -0.1])
        reference["protocol"]["physics_hash"] = "reference-build"
        candidate["protocol"]["physics_hash"] = "candidate-build"
        result = SIGNATURE.compare_signatures(
            candidate, reference, relative_tolerance=1.0e-8
        )
        self.assertEqual(result["gate"], "pass")
        self.assertFalse(result["physics_hash_match"])
        self.assertEqual(result["actual_physics_hash"], "candidate-build")
        self.assertEqual(result["reference_physics_hash"], "reference-build")

    def test_response_signature_rejects_immutable_input_difference(self):
        reference = self._signature([-0.2, -0.1])
        candidate = self._signature([-0.2, -0.1])
        candidate["protocol"]["orbital_sha256"] = "different-orbital"
        with self.assertRaisesRegex(RuntimeError, "orbital_sha256"):
            SIGNATURE.compare_signatures(
                candidate, reference, relative_tolerance=1.0e-8
            )

    def test_compare_cli_binds_input_signature_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual_path = Path(tmp) / "actual.json"
            reference_path = Path(tmp) / "reference.json"
            actual_path.write_text(
                json.dumps(self._signature([-0.2, -0.1])), encoding="ascii"
            )
            reference_path.write_text(
                json.dumps(self._signature([-0.2, -0.1])), encoding="ascii"
            )
            actual_sha256 = hashlib.sha256(actual_path.read_bytes()).hexdigest()
            reference_sha256 = hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest()
            output = io.StringIO()
            with redirect_stdout(output):
                payload = SIGNATURE.main(
                    ["compare", str(actual_path), str(reference_path)]
                )
        self.assertEqual(payload, json.loads(output.getvalue()))
        self.assertEqual(
            payload["actual_signature_sha256"],
            actual_sha256,
        )
        self.assertEqual(
            payload["reference_signature_sha256"],
            reference_sha256,
        )

    def test_response_signature_comparison_rejects_spectrum_change(self):
        reference = self._signature([-0.2, -0.1])
        candidate = self._signature([-0.25, -0.1])
        with self.assertRaisesRegex(RuntimeError, "response spectrum mismatch"):
            SIGNATURE.compare_signatures(
                candidate, reference, relative_tolerance=1.0e-8
            )

    def test_small_dataset_signature_contains_reference_spectrum(self):
        dataset_test = StreamingDatasetTest()
        with tempfile.TemporaryDirectory() as tmp:
            dataset_test._make_dataset(tmp)
            result = SIGNATURE.build_signature(Path(tmp), "test-commit")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["protocol"]["frequencies"], [[0, 0.5, 1.0]])
        self.assertEqual(len(result["reference_response"]), 1)
        self.assertAlmostEqual(
            result["reference_response"][0]["trace_real"], -0.3
        )
        self.assertEqual(
            len(result["reference_response"][0]["eigenvalues"]), 2
        )


if __name__ == "__main__":
    unittest.main()
