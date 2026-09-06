"""Prepared reference reuse must preserve the frozen full-q RPA objective."""

from dataclasses import FrozenInstanceError, fields, replace
import unittest
from unittest import mock

import torch

import common  # noqa: F401
import periodic_galerkin_rpa as rpa
import test_periodic_galerkin_rpa as fixtures


class PeriodicGalerkinRpaCacheTest(unittest.TestCase):
    def fixture(self, complete=True):
        dataset, _ = fixtures.PeriodicGalerkinRpaTest().fixture()
        weights = (0.15, 0.25, 0.6) if complete else (0.05, 0.15, 0.3)
        return tuple(
            replace(
                dataset,
                selected_iq=iq + 1,
                qpoint=(0.125 * iq, 0.0, 0.0),
                q_weight=weight,
                whitened_auxiliary_rank=rank,
                frequency_ha=dataset.frequency_ha.clone(),
                frequency_weights_ha=dataset.frequency_weights_ha.clone(),
                reference_response=(
                    dataset.reference_response[:, :rank, :rank] * (1.0 + 0.2 * iq)
                ).clone(),
            )
            for iq, (weight, rank) in enumerate(zip(weights, (3, 2, 1)))
        )

    def prepare(self, datasets):
        prepare = getattr(rpa, "prepare_periodic_rpa_reference", None)
        self.assertTrue(callable(prepare), "prepared RPA reference API is missing")
        return prepare(datasets)

    def responses(self, datasets, scale=0.83, requires_grad=False):
        return tuple(
            (d.reference_response.detach() * scale).requires_grad_(requires_grad)
            for d in datasets
        )

    def assert_same_result(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        for field in fields(expected):
            left, right = getattr(actual, field.name), getattr(expected, field.name)
            if isinstance(right, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual(left.requires_grad, right.requires_grad)
            elif field.name == "q_records":
                self.assertEqual(len(left), len(right))
                for cached_record, plain_record in zip(left, right):
                    self.assert_same_result(cached_record, plain_record)
            else:
                self.assertEqual(left, right)

    def assert_stale(self, datasets, cache):
        with self.assertRaisesRegex(ValueError, "reference cache"):
            rpa.periodic_rpa_objective(
                datasets, self.responses(datasets), reference_cache=cache
            )

    def test_cached_multi_q_ranks_and_weights_match_every_result_field(self):
        for complete in (False, True):
            datasets = self.fixture(complete)
            cache = self.prepare(datasets)
            for scale in (0.6, 1.0, 1.2):
                with self.subTest(complete=complete, scale=scale):
                    responses = self.responses(datasets, scale)
                    expected = rpa.periodic_rpa_objective(datasets, responses)
                    actual = rpa.periodic_rpa_objective(
                        datasets, responses, reference_cache=cache
                    )
                    self.assert_same_result(actual, expected)
                    self.assertEqual(actual.complete_q_weight, complete)
                    self.assertEqual(actual.q_weight_coverage, 1.0 if complete else 0.5)

    def test_cached_complex_gradients_match_for_all_objective_weights(self):
        datasets = self.fixture()
        cache = self.prepare(datasets)
        for weights in ((1.0, 1.0, 1.0), (2.5, 0.7, 0.0), (0.2, 3.0, 4.0)):
            kwargs = dict(zip(("pi_weight", "trace_log_weight", "energy_weight"), weights))
            with self.subTest(weights=weights):
                expected_responses = self.responses(datasets, requires_grad=True)
                actual_responses = self.responses(datasets, requires_grad=True)
                expected = rpa.periodic_rpa_objective(datasets, expected_responses, **kwargs)
                actual = rpa.periodic_rpa_objective(
                    datasets, actual_responses, reference_cache=cache, **kwargs
                )
                self.assert_same_result(actual, expected)
                expected.loss.backward()
                actual.loss.backward()
                for cached, plain in zip(actual_responses, expected_responses):
                    self.assertTrue(bool(torch.isfinite(cached.grad).all()))
                    torch.testing.assert_close(cached.grad, plain.grad, rtol=0, atol=0)
                self.assertGreater(float(actual_responses[0].grad.imag.abs().max()), 0)

    def test_reference_graph_is_detached_and_cache_supports_repeated_backward(self):
        datasets = self.fixture()
        for dataset in datasets:
            dataset.reference_response.requires_grad_()
            dataset.frequency_weights_ha.requires_grad_()
        cache = self.prepare(datasets)
        for scale in (0.7, 0.9):
            responses = self.responses(datasets, scale, requires_grad=True)
            result = rpa.periodic_rpa_objective(datasets, responses, reference_cache=cache)
            result.loss.backward()
            self.assertFalse(result.reference_energy_ha.requires_grad)
            for dataset, response, record in zip(datasets, responses, result.q_records):
                self.assertIsNone(dataset.reference_response.grad)
                self.assertIsNone(dataset.frequency_weights_ha.grad)
                self.assertIsNotNone(response.grad)
                self.assertFalse(record.reference_raw.requires_grad)
                self.assertFalse(record.reference_contributions_ha.requires_grad)

    def test_reference_trace_log_and_normalizers_are_not_recomputed(self):
        datasets = self.fixture()
        with mock.patch.object(rpa, "rpa_trace_log", wraps=rpa.rpa_trace_log) as trace_log:
            cache = self.prepare(datasets)
            self.assertEqual(trace_log.call_count, len(datasets))
        for scale in (0.7, 0.9):
            responses = self.responses(datasets, scale)
            with mock.patch.object(rpa, "rpa_trace_log", wraps=rpa.rpa_trace_log) as trace_log, \
                    mock.patch.object(torch, "sum", wraps=torch.sum) as summed, \
                    mock.patch.object(torch, "dot", wraps=torch.dot) as dotted:
                rpa.periodic_rpa_objective(datasets, responses, reference_cache=cache)
            self.assertEqual(trace_log.call_count, len(datasets))
            self.assertEqual(summed.call_count, len(datasets))
            self.assertEqual(dotted.call_count, len(datasets))

    def test_default_and_explicit_none_keep_uncached_path(self):
        datasets = self.fixture()
        self.prepare(datasets)
        responses = self.responses(datasets)
        with mock.patch.object(rpa, "prepare_periodic_rpa_reference", side_effect=AssertionError), \
                mock.patch.object(rpa, "rpa_trace_log", wraps=rpa.rpa_trace_log) as trace_log:
            expected = rpa.periodic_rpa_objective(datasets, responses)
            actual = rpa.periodic_rpa_objective(datasets, responses, reference_cache=None)
        self.assertEqual(trace_log.call_count, 4 * len(datasets))
        self.assert_same_result(actual, expected)
        self.assertEqual(
            tuple(field.name for field in fields(actual)),
            ("loss", "pi_relative_squared_error", "trace_log_relative_squared_error",
             "energy_relative_squared_error", "candidate_energy_ha", "reference_energy_ha",
             "q_weight_coverage", "complete_q_weight", "q_records"),
        )

    def test_cache_accepts_a_new_tuple_of_the_same_dataset_objects(self):
        datasets = self.fixture()
        cache = self.prepare(datasets)
        same_datasets = tuple(list(datasets))
        self.assertIsNot(same_datasets, datasets)
        responses = self.responses(datasets)
        self.assert_same_result(
            rpa.periodic_rpa_objective(same_datasets, responses, reference_cache=cache),
            rpa.periodic_rpa_objective(datasets, responses),
        )

    def test_cache_rejects_different_dataset_objects_q_order_or_subset(self):
        datasets = self.fixture()
        cache = self.prepare(datasets)
        cases = (tuple(replace(d) for d in datasets), datasets[::-1], datasets[:1])
        for changed in cases:
            with self.subTest(count=len(changed)):
                self.assert_stale(changed, cache)

    def test_cache_rejects_changed_q_frequency_reference_or_protocol(self):
        datasets = self.fixture()
        cache = self.prepare(datasets)
        dataset = datasets[0]
        changes = (
            {"selected_iq": 7}, {"qpoint": (0.5, 0.5, 0.5)}, {"q_weight": 0.1},
            {"physics_hash": "different"}, {"orbital_sha256": "different"},
            {"frequency_ha": dataset.frequency_ha.clone()},
            {"frequency_weights_ha": dataset.frequency_weights_ha.clone()},
            {"reference_response": dataset.reference_response.clone()},
            {"reference_response": dataset.reference_response.detach()},
        )
        for change in changes:
            with self.subTest(field=next(iter(change))):
                self.assert_stale((replace(dataset, **change),) + datasets[1:], cache)

    def test_cache_rejects_in_place_mutation_of_every_frozen_input(self):
        for name in ("reference_response", "frequency_ha", "frequency_weights_ha"):
            for alias_kind in ("original", "detach", "view"):
                with self.subTest(field=name, alias=alias_kind):
                    datasets = self.fixture()
                    source = getattr(datasets[0], name)
                    alias = source if alias_kind == "original" else (
                        source.detach() if alias_kind == "detach" else source.view(-1)
                    )
                    cache = self.prepare(datasets)
                    alias.mul_(1.001)
                    self.assert_stale(datasets, cache)

    def test_cache_rejects_mutation_through_owner_of_a_detached_input(self):
        for name in ("reference_response", "frequency_ha", "frequency_weights_ha"):
            with self.subTest(field=name):
                datasets = self.fixture()
                owner = getattr(datasets[0], name).clone().requires_grad_()
                datasets = (replace(datasets[0], **{name: owner.detach()}),) + datasets[1:]
                cache = self.prepare(datasets)
                with torch.no_grad():
                    owner.mul_(1.001)
                self.assert_stale(datasets, cache)

    def test_mutating_returned_records_cannot_corrupt_cache_or_inputs(self):
        datasets = self.fixture()
        originals = tuple(
            tuple(getattr(d, name).clone() for name in (
                "reference_response", "frequency_ha", "frequency_weights_ha"
            )) for d in datasets
        )
        responses = self.responses(datasets)
        expected = rpa.periodic_rpa_objective(datasets, responses)
        cache = self.prepare(datasets)
        result = rpa.periodic_rpa_objective(datasets, responses, reference_cache=cache)
        result.reference_energy_ha.zero_()
        for record in result.q_records:
            for name in ("reference_raw", "reference_contributions_ha",
                         "frequency_ha", "frequency_weights_ha"):
                getattr(record, name).zero_()
        for dataset, original in zip(datasets, originals):
            for name, value in zip(
                ("reference_response", "frequency_ha", "frequency_weights_ha"), original
            ):
                torch.testing.assert_close(getattr(dataset, name), value, rtol=0, atol=0)
        self.assert_same_result(
            rpa.periodic_rpa_objective(datasets, responses, reference_cache=cache), expected
        )

    def test_prepared_cache_is_frozen(self):
        cache = self.prepare(self.fixture())
        with self.assertRaises(FrozenInstanceError):
            cache.unexpected_attribute = None

    def test_invalid_cache_type_is_rejected(self):
        datasets = self.fixture()
        self.prepare(datasets)
        for invalid in (False, {}, object()):
            with self.subTest(cache=invalid):
                self.assert_stale(datasets, invalid)

    def test_cached_path_still_checks_candidate_physics_and_dimensions(self):
        datasets = self.fixture()
        cache = self.prepare(datasets)
        responses = self.responses(datasets)
        for invalid in (-responses[0], responses[0][:, :1, :1],
                        torch.full_like(responses[0], float("nan"))):
            with self.subTest(shape=tuple(invalid.shape)), self.assertRaises(ValueError):
                rpa.periodic_rpa_objective(
                    datasets, (invalid,) + responses[1:], reference_cache=cache
                )
        with self.assertRaises(ValueError):
            rpa.periodic_rpa_objective(datasets, responses[:1], reference_cache=cache)

    def test_cached_path_preserves_objective_weight_guards(self):
        datasets = self.fixture()
        cache = self.prepare(datasets)
        for name in ("pi_weight", "trace_log_weight", "energy_weight"):
            invalid = (-1.0, float("nan"), float("inf"), True, "1")
            if name != "energy_weight":
                invalid += (0.0,)
            for value in invalid:
                with self.subTest(weight=name, value=value), self.assertRaises(ValueError):
                    rpa.periodic_rpa_objective(
                        datasets, self.responses(datasets), reference_cache=cache,
                        **{name: value}
                    )

    def test_prepare_rejects_invalid_datasets_grids_and_references(self):
        datasets = self.fixture()
        self.prepare(datasets)
        dataset = datasets[0]
        invalid = (
            (), list(datasets), (object(),), (dataset, dataset),
            (replace(dataset, reference_response=-dataset.reference_response),),
            (replace(dataset, reference_response=torch.zeros_like(dataset.reference_response)),),
            (replace(dataset, frequency_ha=dataset.frequency_ha.flip(0)),),
            (replace(dataset, frequency_weights_ha=-dataset.frequency_weights_ha),),
            (replace(dataset, q_weight=float("nan")),),
        )
        for case in invalid:
            with self.subTest(count=len(case)), self.assertRaises(ValueError):
                self.prepare(case)


if __name__ == "__main__":
    unittest.main()
