import importlib.util
from pathlib import Path
import unittest


RUNNER = (Path(__file__).resolve().parents[1] / 'example_C_sternheimer'
          / 'periodic_basis_optimization/galerkin_binding_workflow'
          / 'collect_c_jy_full_q_gradient.py')
spec = importlib.util.spec_from_file_location('collect_c_jy_full_q_gradient',
                                              RUNNER)
collector = importlib.util.module_from_spec(spec) if spec else None
if spec:
    spec.loader.exec_module(collector)


class CollectFullQGradientTest(unittest.TestCase):
    def test_exact_zero_padding_baseline_is_required_before_proposal(self):
        self.assertIsNotNone(collector, 'full-q gradient collector missing')
        result = collector.validate_baseline_reproduction(
            dict(candidate_energy_ha=-.46276046596549875,
                 reference_energy_ha=-.5144842779929139),
            expected_candidate_energy_ha=-.46276046596549875,
            expected_reference_energy_ha=-.5144842779929139,
            tolerance_ha=1e-12)
        self.assertEqual(result['zero_padding_baseline_gate'], 'pass')
        self.assertEqual(result['candidate_energy_difference_ha'], 0.)
        with self.assertRaisesRegex(ValueError, 'zero-padding baseline'):
            collector.validate_baseline_reproduction(
                dict(candidate_energy_ha=-.4627,
                     reference_energy_ha=-.5144842779929139),
                expected_candidate_energy_ha=-.46276046596549875,
                expected_reference_energy_ha=-.5144842779929139,
                tolerance_ha=1e-12)

    def test_serialized_proposal_excludes_live_tensors_and_locks_hashes(self):
        proposal = dict(status='prepared', scope='expanded_45ao_full_q_ec_step',
            radius=.02, coefficients=object(), direction=object(),
            predicted_ec_delta_ha_per_cell=-.001,
            actual_full_q_energy='pending', physical_release_gate='hold')
        result = collector.serialize_proposal(
            proposal, gradient_sha256='a'*64,
            input_coefficient_sha256='b'*64,
            output_coefficient_sha256='c'*64)
        self.assertNotIn('coefficients', result)
        self.assertNotIn('direction', result)
        self.assertEqual(result['gradient_sha256'], 'a'*64)
        self.assertEqual(result['input_coefficient_sha256'], 'b'*64)
        self.assertEqual(result['output_coefficient_sha256'], 'c'*64)


if __name__ == '__main__':
    unittest.main()
