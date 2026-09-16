from dataclasses import dataclass
import types
import unittest

import torch

import common  # noqa: F401
import c_jy_full_q_gradient as gradient
from periodic_galerkin_radial_diagnostics import radial_gradient_report


PROFILE = (4, 4, 3, 2, 0)
INDICES = (1, 22, 43, 6, 27, 23, 11, 55)
MULTIPLICITIES = (1, 8, 4, 6, 24, 12, 3, 6)


def coefficients(radial_rows=48):
    result = []
    for count in PROFILE:
        value = torch.zeros((radial_rows, count), dtype=torch.float64)
        if count:
            value[:count, :] = torch.eye(count, dtype=torch.float64)
        result.append(value)
    return {"C": result}


class FullQGradientTest(unittest.TestCase):
    def test_row_range_gradient_fractions_separate_legacy_and_new_blocks(self):
        report = {"channels": [
            {"horizontal_gradient": [1.] * 31 + [2.] * 17 + [3.] * 52},
            {"horizontal_gradient": [0.] * 100},
        ]}

        result = gradient.row_range_gradient_squared_fractions(
            report, radial_rows=100, boundaries=(31, 48))

        total = 31 * 1.**2 + 17 * 2.**2 + 52 * 3.**2
        self.assertEqual(
            [(row["start_row"], row["stop_row"]) for row in result],
            [(1, 31), (32, 48), (49, 100)])
        self.assertAlmostEqual(result[0]["squared_fraction"], 31 / total)
        self.assertAlmostEqual(result[1]["squared_fraction"], 68 / total)
        self.assertAlmostEqual(result[2]["squared_fraction"], 468 / total)

    def test_one_q_uses_the_actual_weighted_energy_gradient(self):
        @dataclass(frozen=True)
        class Dataset:
            frequency_ha: torch.Tensor
            primitive_blocks: tuple
            kpoints: tuple
            selected_iq: int
            q_weight: float

        dataset = Dataset(torch.arange(1, 13, dtype=torch.float64),
                          ("blocks",), ("k1", "k2"), 22, 8/64)
        spec = dict(profile=list(PROFILE), ao_per_C=45,
                    coefficients_path="/tmp/c", coefficients_sha256="a"*64)
        prepared = []

        def prepare(record, blocks, current):
            self.assertEqual(blocks, ("blocks",))
            prepared.append(record)
            return "prepared-" + record

        def evaluate(current_dataset, current, **controls):
            self.assertEqual(current_dataset.kpoints,
                             ("prepared-k1", "prepared-k2"))
            self.assertEqual(controls["frequency_batch_size"], 12)
            value = sum(channel[31:, :].sum()
                        for channel in current["C"] if channel.shape[1])
            return types.SimpleNamespace(
                response=value.reshape(1, 1, 1),
                minimum_occupied_capture=.999999,
                maximum_overlap_condition=17.,
                minimum_candidate_rank=90)

        def objective(datasets, responses, **weights):
            self.assertEqual(weights, dict(pi_weight=1.,
                                           trace_log_weight=1.,
                                           energy_weight=0.))
            energy = datasets[0].q_weight * responses[0].sum()
            record = types.SimpleNamespace(
                candidate_contributions_ha=energy.repeat(12)/12,
                reference_contributions_ha=torch.full(
                    (12,), -.2/12, dtype=torch.float64),
                candidate_raw=torch.full((12,), -.1, dtype=torch.float64),
                reference_raw=torch.full((12,), -.2, dtype=torch.float64))
            return types.SimpleNamespace(
                candidate_energy_ha=energy,
                reference_energy_ha=torch.tensor(-.2, dtype=torch.float64),
                q_records=(record,))

        result = gradient.evaluate_q_energy_gradient(
            dataset, spec,
            read_coefficients=lambda *args, **kwargs: coefficients(),
            evaluate_response=evaluate, evaluate_objective=objective,
            radial_gradient_report=radial_gradient_report,
            prepare_block_cache=prepare)
        self.assertEqual(prepared, ["k1", "k2"])
        self.assertEqual(result["selected_iq"], 22)
        self.assertEqual(result["q_weight"], 8/64)
        self.assertEqual(result["frequency_count"], 12)
        self.assertEqual(result["radial_rows"], 48)
        self.assertEqual(result["candidate_energy_ha"], 0.)
        report = result["energy_gradient"]
        self.assertGreater(report["horizontal_gradient_norm"], 0.)
        raw = torch.tensor(report["channels"][0]["raw_gradient"])
        torch.testing.assert_close(raw[:31], torch.zeros_like(raw[:31]))
        torch.testing.assert_close(raw[31:],
                                   torch.full_like(raw[31:], 8/64))

    def test_one_q_accepts_a_hundred_row_mother_at_fixed_output_rank(self):
        dataset = types.SimpleNamespace(
            frequency_ha=torch.arange(1, 13, dtype=torch.float64),
            primitive_blocks=("blocks",), kpoints=("k1",),
            selected_iq=1, q_weight=1/64)
        spec = dict(profile=list(PROFILE), ao_per_C=45,
                    coefficients_path="/tmp/c", coefficients_sha256="a"*64)

        def evaluate(current_dataset, current, **controls):
            value = sum(channel[31:, :].sum()
                        for channel in current["C"] if channel.shape[1])
            return types.SimpleNamespace(
                response=value.reshape(1, 1, 1),
                minimum_occupied_capture=.999999,
                maximum_overlap_condition=17., minimum_candidate_rank=90)

        def objective(datasets, responses, **weights):
            energy = datasets[0].q_weight * responses[0].sum()
            record = types.SimpleNamespace(
                candidate_contributions_ha=energy.repeat(12)/12,
                reference_contributions_ha=torch.full(
                    (12,), -.2/12, dtype=torch.float64))
            return types.SimpleNamespace(
                candidate_energy_ha=energy,
                reference_energy_ha=torch.tensor(-.2, dtype=torch.float64),
                q_records=(record,))

        result = gradient.evaluate_q_energy_gradient(
            dataset, spec,
            read_coefficients=lambda *args, **kwargs: coefficients(100),
            evaluate_response=evaluate, evaluate_objective=objective,
            radial_gradient_report=radial_gradient_report,
            radial_rows=100)

        self.assertEqual(result["radial_rows"], 100)
        self.assertEqual(
            result["energy_gradient"]["channels"][0]["shape"], [100, 4])

    def artifact(self, slot, scale=1.):
        raw = {"C": []}
        for count in PROFILE:
            value = torch.zeros((48, count), dtype=torch.float64)
            if count:
                value[31:, :] = scale
            raw["C"].append(value)
        return dict(
            status="success",
            scope="compressed_shared_radial_full_q_energy_gradient",
            q_slot=slot,
            selected_iq=INDICES[slot],
            q_weight=MULTIPLICITIES[slot]/64,
            frequency_count=12,
            profile=list(PROFILE),
            ao_per_C=45,
            radial_rows=48,
            coefficients_sha256="a"*64,
            candidate_energy_ha=-.01*(slot+1),
            reference_energy_ha=-.02*(slot+1),
            minimum_occupied_capture=.999999,
            maximum_overlap_condition=20.+slot,
            minimum_candidate_rank=90,
            expanded_mother_anchor_gate="pass",
            energy_gradient=radial_gradient_report(coefficients(), raw),
            physical_release_gate="hold")

    def test_reduce_reports_complete_energy_and_new_row_share(self):
        artifacts = [self.artifact(slot, slot+1.) for slot in range(8)]
        result = gradient.reduce_full_q_energy_gradients(
            list(reversed(artifacts)), coefficients(),
            coefficient_sha256="a"*64)
        self.assertTrue(result["complete_q_weight"])
        self.assertAlmostEqual(result["q_weight_coverage"], 1.)
        self.assertAlmostEqual(result["candidate_energy_ha"], -.36)
        self.assertAlmostEqual(result["reference_energy_ha"], -.72)
        self.assertEqual([row["q_slot"] for row in result["per_q"]],
                         list(range(8)))
        self.assertAlmostEqual(
            result["new_row_horizontal_gradient_squared_fraction"], 1.)
        self.assertGreater(
            result["energy_gradient"]["horizontal_gradient_norm"], 0.)

    def test_reducer_rejects_incomplete_or_mixed_candidates(self):
        artifacts = [self.artifact(slot) for slot in range(8)]
        with self.assertRaisesRegex(ValueError, "complete eight-q"):
            gradient.reduce_full_q_energy_gradients(
                artifacts[:-1], coefficients(), coefficient_sha256="a"*64)
        artifacts[3]["coefficients_sha256"] = "b"*64
        with self.assertRaisesRegex(ValueError, "coefficient"):
            gradient.reduce_full_q_energy_gradients(
                artifacts, coefficients(), coefficient_sha256="a"*64)

    def test_full_q_descent_proposal_preserves_fixed_output_rank(self):
        artifacts = [self.artifact(slot, slot+1.) for slot in range(8)]
        reduced = gradient.reduce_full_q_energy_gradients(
            artifacts, coefficients(), coefficient_sha256="a"*64)
        proposal = gradient.propose_full_q_ec_step(
            coefficients(), reduced["energy_gradient"], radius=.02)
        self.assertAlmostEqual(proposal["radius"], .02)
        self.assertLess(proposal["predicted_ec_delta_ha_per_cell"], 0.)
        self.assertEqual([list(value.shape)
                          for value in proposal["coefficients"]["C"]],
                         [[48, 4], [48, 4], [48, 3], [48, 2], [48, 0]])
        for value in proposal["coefficients"]["C"]:
            torch.testing.assert_close(
                value.T @ value,
                torch.eye(value.shape[1], dtype=torch.float64),
                atol=1e-12, rtol=0)
        with self.assertRaises(ValueError):
            gradient.propose_full_q_ec_step(
                coefficients(), reduced["energy_gradient"], radius=.2)


if __name__ == "__main__":
    unittest.main()
