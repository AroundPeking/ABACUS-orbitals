"""Exact, source-bound persistent active datasets without pickle or chunk rereads."""

from dataclasses import fields, is_dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import common  # noqa: F401
import periodic_galerkin_data as reader
from periodic_galerkin_basis import prepare_periodic_block_contraction_record
from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
from periodic_galerkin_rpa import periodic_rpa_objective
from test_periodic_galerkin_streaming_reduction import write_multik_fixture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PeriodicGalerkinDatasetCacheTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec("periodic_galerkin_dataset_cache")
        self.assertIsNotNone(spec, "persistent dataset cache module is missing")
        self.cache = importlib.import_module("periodic_galerkin_dataset_cache")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.coefficients = write_multik_fixture(self.source)
        self.dataset = reader.read_periodic_galerkin_dataset(
            self.source, include_reference_projection=False,
            active_coefficients=self.coefficients,
        )
        self.path = self.root / "cache"
        self.binding = dict(
            source_directory=self.source, manifest_sha256=sha(self.source / "manifest.dat"),
            status_sha256=sha(self.source / "status.dat"),
            identifiers={"freeze_sha256": "a" * 64, "acceptance_sha256": "b" * 64,
                         "mapping_sha256": self.dataset.active_primitive_reduction.mapping_sha256},
        )

    def write(self, dataset=None, path=None, **options):
        self.cache_sha256 = self.cache.write_periodic_galerkin_dataset_cache(
            self.path if path is None else path, self.dataset if dataset is None else dataset,
            **dict(self.binding, **options)
        )
        return self.cache_sha256

    def read(self, path=None, **options):
        return self.cache.read_periodic_galerkin_dataset_cache(
            self.path if path is None else path,
            **dict(self.binding, cache_sha256=self.cache_sha256, **options)
        )

    def assert_exact(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.resolve_conj().resolve_neg().numpy().tobytes(),
                             expected.detach().resolve_conj().resolve_neg().numpy().tobytes())
            if actual.numel():
                self.assertNotEqual(actual.data_ptr(), expected.data_ptr())
        elif is_dataclass(expected):
            for field in fields(expected):
                self.assert_exact(getattr(actual, field.name), getattr(expected, field.name))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_exact(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_exact(a, b)
        else:
            self.assertEqual(actual, expected)

    def metadata(self):
        return json.loads((self.path / "dataset.json").read_text())

    def update_metadata(self, metadata):
        path = self.path / "dataset.json"
        path.write_text(json.dumps(metadata))
        completion = json.loads((self.path / "COMPLETE.json").read_text())
        completion["metadata_sha256"] = sha(path)
        (self.path / "COMPLETE.json").write_text(json.dumps(completion))
        self.cache_sha256 = sha(self.path / "COMPLETE.json")

    def test_roundtrip_all_fields_complex_and_full_mother_normalization(self):
        self.write()
        actual = self.read(active_coefficients=self.coefficients)
        self.assert_exact(actual, self.dataset)
        self.assertEqual(sum(k.k_weight for k in actual.kpoints), 2.0)
        self.assertTrue(all(torch.equal(k.occupation, torch.ones_like(k.occupation))
                            for k in actual.kpoints))
        self.assertEqual(actual.active_primitive_reduction.original_primitive_count, 27)
        self.assertEqual(actual.primitive_count, 18)
        self.assertIsNotNone(actual.kpoints[0].occupied_projection_normalization)
        actual.kpoints[0].source.zero_()
        self.assert_exact(self.read(), self.dataset)

    def test_nonempty_block_contraction_cache_rejected_not_silently_dropped(self):
        prepared = replace(self.dataset, kpoints=tuple(
            prepare_periodic_block_contraction_record(k, self.dataset.primitive_blocks, self.coefficients)
            for k in self.dataset.kpoints
        ))
        with self.assertRaisesRegex(ValueError, "block_contraction_cache"):
            self.write(prepared)
        self.assertFalse((self.path / "COMPLETE.json").exists())

    def test_multi_q_objective_and_all_column_gradients_equal_after_roundtrip(self):
        self.write()
        snapshots = []
        for dataset in (self.dataset, self.read()):
            coefficients = {"C": [c.clone().requires_grad_() for c in self.coefficients["C"]]}
            family = tuple(replace(dataset, q_count=2, selected_iq=iq, q_weight=weight)
                           for iq, weight in ((1, .25), (2, .75)))
            responses = tuple(evaluate_periodic_galerkin_coefficient_response(
                q, coefficients, occupied_capture_tolerance=.01
            ).response for q in family)
            result = periodic_rpa_objective(family, responses)
            result.loss.backward()
            snapshots.append((result.loss.detach(), tuple(r.detach() for r in responses),
                              tuple(c.grad for c in coefficients["C"] if c.numel())))
        for a, b in zip(snapshots[0], snapshots[1]):
            if isinstance(a, tuple):
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_reload_never_calls_reader_or_opens_producer_binary_chunks(self):
        self.write()
        for path in self.source.glob("*.bin"):
            path.unlink()
        with mock.patch.object(reader, "read_periodic_galerkin_dataset", side_effect=AssertionError), \
                mock.patch.object(reader, "_read_chunk", side_effect=AssertionError):
            self.assert_exact(self.read(), self.dataset)

    def test_standalone_reload_uses_explicit_derivative_and_source_hash_binding(self):
        self.write()
        self.assertEqual(self.cache_sha256, sha(self.path / "COMPLETE.json"))
        for path in self.source.iterdir():
            path.unlink()
        self.assert_exact(self.read(source_directory=None), self.dataset)
        self.cache_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.read(source_directory=None)

    def test_changed_source_files_and_caller_hash_identifiers_rejected(self):
        self.write()
        for filename in ("manifest.dat", "status.dat", "primitive_blocks.dat"):
            path = self.source / filename
            before = path.read_bytes()
            path.write_bytes(before + b"changed\n")
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "source|SHA256"):
                self.read()
            path.write_bytes(before)
        for options in ({"manifest_sha256": "0" * 64}, {"status_sha256": "0" * 64},
                        {"identifiers": {"freeze_sha256": "c" * 64}}, {"identifiers": None},
                        {"identifiers": dict(self.binding["identifiers"], mapping_sha256="d" * 64)}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "source|binding|SHA256"):
                self.read(**options)

    def test_mismatched_dataset_metadata_not_published(self):
        variants = ({"physics_hash": "f" * 64}, {"selected_iq": 2}, {"q_weight": .5},
                    {"frequency_ha": self.dataset.frequency_ha * 2},
                    {"orbital_sha256": "f" * 64},
                    {"kpoints": (replace(self.dataset.kpoints[0], k_weight=.5),) + self.dataset.kpoints[1:]})
        for i, change in enumerate(variants):
            path = self.root / ("wrong" + str(i))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "source|metadata"):
                self.write(replace(self.dataset, **change), path=path)
            self.assertFalse((path / "COMPLETE.json").exists())

    def test_corrupt_missing_array_and_metadata_rejected(self):
        self.write()
        array = next(self.path.glob("array_*.npy"))
        saved = array.read_bytes()
        array.write_bytes(saved[:-1] + bytes([saved[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.read()
        array.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.read()
        array.write_bytes(saved)
        path = self.path / "dataset.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.read()

    def test_incomplete_cache_no_overwrite_and_interrupted_write_never_published(self):
        self.write()
        before = {p.name: p.read_bytes() for p in self.path.iterdir()}
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.iterdir()})
        (self.path / "COMPLETE.json").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.read()
        partial = self.root / "partial"
        with mock.patch.object(np, "save", side_effect=OSError("interrupted write")):
            with self.assertRaises(OSError):
                self.write(path=partial)
        self.assertFalse((partial / "COMPLETE.json").exists())

    def test_completion_published_last_and_numpy_never_uses_pickle(self):
        save = np.save
        def checked_save(*args, **kwargs):
            self.assertFalse((self.path / "COMPLETE.json").exists())
            self.assertIs(kwargs.get("allow_pickle"), False)
            return save(*args, **kwargs)
        with mock.patch.object(np, "save", side_effect=checked_save):
            self.write()
        with mock.patch.object(np, "load", wraps=np.load) as load:
            self.read()
        self.assertTrue(load.call_count)
        self.assertTrue(all(call[1].get("allow_pickle") is False for call in load.call_args_list))

    def test_pickle_payload_rejected_even_if_array_hash_was_replaced(self):
        self.write()
        metadata = self.metadata()
        array = self.path / metadata["arrays"][0]["file"]
        np.save(str(array), np.array([{"not": "numeric"}], dtype=object), allow_pickle=True)
        metadata["arrays"][0]["sha256"] = sha(array)
        self.update_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "allow_pickle|Object arrays"):
            self.read()

    def test_source_or_frozen_tensor_mutation_during_write_prevents_publication(self):
        saved_manifest = (self.source / "manifest.dat").read_bytes()
        saved_frequency = self.dataset.frequency_ha.clone()
        for what in ("source", "tensor"):
            path = self.root / ("mutated_" + what)
            original_save = np.save
            changed = [False]

            def changed_save(*args, **kwargs):
                original_save(*args, **kwargs)
                if not changed[0]:
                    if what == "source":
                        (self.source / "manifest.dat").write_bytes(saved_manifest + b"changed\n")
                    else:
                        self.dataset.frequency_ha.mul_(2)
                    changed[0] = True

            with mock.patch.object(np, "save", side_effect=changed_save):
                with self.subTest(what=what), self.assertRaisesRegex(ValueError, "SHA256|mutated"):
                    self.write(path=path)
            self.assertFalse((path / "COMPLETE.json").exists())
            (self.source / "manifest.dat").write_bytes(saved_manifest)
            self.dataset.frequency_ha.copy_(saved_frequency)

    def test_wrong_mapping_identifier_rejected_before_publication(self):
        identifiers = dict(self.binding["identifiers"], mapping_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.write(identifiers=identifiers)
        self.assertFalse((self.path / "COMPLETE.json").exists())

    def test_unknown_classes_fields_schema_and_unsafe_paths_rejected(self):
        self.write()
        original = self.metadata()
        mutations = [lambda m: m.update(format_version=999),
                     lambda m: m["dataset"].update(type="subprocess.Popen"),
                     lambda m: m["dataset"]["fields"].pop("active_primitive_reduction"),
                     lambda m: m["arrays"][0].update(file="../outside.npy"),
                     lambda m: m["arrays"][0].update(dtype="|O"),
                     lambda m: m["arrays"][0].update(shape=[999999])]
        for mutation in mutations:
            metadata = json.loads(json.dumps(original))
            mutation(metadata)
            self.update_metadata(metadata)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.read()

    def test_rejects_unknown_python_objects_and_gradient_inputs(self):
        for index, record in enumerate((replace(self.dataset.kpoints[0], block_contraction_cache=object()),
                                       replace(self.dataset.kpoints[0], source=self.dataset.kpoints[0].source.clone().requires_grad_()))):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "supported|frozen|gradient"):
                self.write(replace(self.dataset, kpoints=(record,) + self.dataset.kpoints[1:]),
                           path=self.root / ("unsupported" + str(index)))

    def test_changed_active_coefficient_profile_rejected(self):
        self.write()
        different = {"C": list(self.coefficients["C"])}
        different["C"][1] = torch.ones((3, 1), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "profile"):
            self.read(active_coefficients=different)

    def test_noncontiguous_conjugate_and_complex64_tensor_values_preserved(self):
        record = self.dataset.kpoints[0]
        record = replace(record, overlap=record.overlap.T.conj(), source=record.source.to(torch.complex64))
        dataset = replace(self.dataset, kpoints=(record,) + self.dataset.kpoints[1:])
        self.write(dataset)
        self.assert_exact(self.read(), dataset)


if __name__ == "__main__":
    unittest.main()
