import importlib.util
from pathlib import Path
import unittest

import numpy as np

import common  # noqa: F401


RUNNER = (Path(__file__).resolve().parents[1] / 'example_C_sternheimer'
          / 'periodic_basis_optimization/galerkin_binding_workflow'
          / 'collect_c_jy_compressed_rank_ladder.py')
spec = importlib.util.spec_from_file_location('compressed_rank_collector', RUNNER)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CompressedRankCollectorTest(unittest.TestCase):
    def records(self):
        qpoints = ((1, 1, 1), (2, 22, 8), (3, 43, 4), (6, 6, 6),
                   (7, 27, 24), (8, 23, 12), (11, 11, 3), (28, 55, 6))
        profiles = ((3, 3, 2, 1, 0), (4, 4, 3, 2, 0),
                    (5, 5, 4, 3, 0), (6, 6, 5, 4, 0))
        totals = (-.500, -.505, -.508, -.509)
        rows = []
        for slot, (label, iq, multiplicity) in enumerate(qpoints):
            weight = multiplicity / 64
            rows.append(dict(status='success', q_slot=slot, label=label,
                selected_iq=iq, multiplicity=multiplicity, q_weight=weight,
                lmax=3, relative_rank_tolerance=1e-10,
                frequencies=np.arange(12, dtype=float),
                weights=np.arange(12, dtype=float)+.5,
                per_profile=[dict(profile=list(profile),
                    ao_per_C=sum((2*l+1)*count
                                 for l, count in enumerate(profile)),
                    coefficients_sha256=str(index)*64,
                    candidate_energy_ha=energy*weight,
                    reference_energy_ha=-.5144842779929139*weight,
                    minimum_occupied_capture=.9999995,
                    maximum_overlap_condition=10.+index,
                    minimum_candidate_rank=20+index)
                    for index, (profile, energy)
                    in enumerate(zip(profiles, totals), 1)]))
        return rows

    def test_smallest_passing_rank_is_selected_for_ordinary_sos(self):
        result = collector.finalize_collection(
            self.records(), reference_energy_ha=-.5144842779929139,
            spdf_mother_energy_ha=-.5091040675004793)
        self.assertEqual(result['candidate_selection_gate'], 'pass')
        self.assertEqual(result['selected_profile'], [5, 5, 4, 3, 0])
        self.assertEqual(result['selected_ao_per_C'], 61)
        self.assertEqual(result['ordinary_sos_release_gate'], 'pass')
        self.assertFalse(result['ordinary_sos_validated'])

    def test_no_rank_is_released_when_all_miss_the_energy_gate(self):
        rows = self.records()
        for row in rows:
            for profile in row['per_profile']:
                profile['candidate_energy_ha'] = -.49 * row['q_weight']
        result = collector.finalize_collection(
            rows, reference_energy_ha=-.5144842779929139,
            spdf_mother_energy_ha=-.5091040675004793)
        self.assertEqual(result['candidate_selection_gate'], 'fail')
        self.assertIsNone(result['selected_profile'])
        self.assertEqual(result['ordinary_sos_release_gate'], 'hold')


if __name__ == '__main__':
    unittest.main()
