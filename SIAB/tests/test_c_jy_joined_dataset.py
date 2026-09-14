import unittest
from pathlib import Path
import sys

import numpy as np

WF = Path(__file__).resolve().parents[1] / 'opt_orb_pytorch_dpsi'
sys.path.insert(0, str(WF))
from c_jy_joined_dataset import canonical_q_contract, join_q_records
import c_jy_joined_dataset as module


class JoinedDatasetContractTest(unittest.TestCase):
    def rank_records(self):
        profiles = ((3, 3, 2, 1, 0), (4, 4, 3, 2, 0),
                    (5, 5, 4, 3, 0), (6, 6, 5, 4, 0))
        totals = (-.500, -.505, -.508, -.509)
        rows = []
        for q in self.records():
            weight = q['multiplicity'] / 64
            rows.append(dict(q, status='success', q_weight=weight,
                source_commit='a' * 40, radial_fit_result_sha256='b' * 64,
                per_profile=[dict(profile=list(profile),
                    ao_per_C=sum((2*l+1)*n for l, n in enumerate(profile)),
                    coefficients_sha256=(str(i) * 64),
                    candidate_energy_ha=total * weight,
                    reference_energy_ha=-.5144842779929139 * weight,
                    minimum_occupied_capture=.999999,
                    maximum_overlap_condition=10.+i,
                    minimum_candidate_rank=2*sum((2*l+1)*n
                                                 for l, n in enumerate(profile)))
                    for i, (profile, total) in enumerate(zip(profiles, totals), 1)]))
        return rows

    def test_compressed_rank_collector_separates_mother_and_compression_error(self):
        self.assertTrue(hasattr(module, 'collect_compressed_rank_ladder'))
        reference = -.5144842779929139
        mother = -.5091040675004793
        result = module.collect_compressed_rank_ladder(
            self.rank_records(), reference_energy_ha=reference,
            spdf_mother_energy_ha=mother)
        self.assertEqual([row['ao_per_C'] for row in result['per_profile']],
                         [29, 45, 61, 77])
        final = result['per_profile'][-1]
        self.assertAlmostEqual(final['candidate_energy_ha'], -.509)
        self.assertAlmostEqual(final['compression_error_ev_per_C'],
                               (-.509-mother)*27.211386245988/2)
        self.assertAlmostEqual(final['mother_error_ev_per_C'],
                               (mother-reference)*27.211386245988/2)
        self.assertAlmostEqual(final['total_error_ev_per_C'],
                               (-.509-reference)*27.211386245988/2)
        self.assertTrue(final['energy_gate'])
        self.assertEqual(result['physical_release_gate'], 'hold')

    def test_compressed_rank_collector_rejects_cross_q_candidate_mixing(self):
        rows = self.rank_records()
        rows[3]['per_profile'][1]['coefficients_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'coefficient'):
            module.collect_compressed_rank_ladder(rows, spdf_mother_energy_ha=-.5091040675004793)

    def test_energy_collector_sums_already_weighted_q_energies_once(self):
        self.assertTrue(hasattr(module,'collect_energies'))
        rows = self.records()
        reference = -.5144842779929139
        for r in rows:
            r.update(status='success',lmax=3,relative_rank_tolerance=1e-10,
                q_weight=r['multiplicity']/64,
                candidate_energy_ha=(reference+.003)*r['multiplicity']/64,
                reference_energy_ha=reference*r['multiplicity']/64)
        d = module.collect_energies(rows,reference)
        self.assertAlmostEqual(d['candidate_energy_ha'],reference+.003)
        self.assertAlmostEqual(d['signed_error_ev_per_C'],.003*27.211386245988/2)
        self.assertTrue(d['mother_energy_gate'])
        self.assertEqual(d['physical_release_gate'],'hold')
        rows[2]['lmax'] = 4
        with self.assertRaises(ValueError):
            module.collect_energies(rows,reference)
    def records(self):
        labels = ((1, 1, 1), (2, 22, 8), (3, 43, 4), (6, 6, 6),
                  (7, 27, 24), (8, 23, 12), (11, 11, 3), (28, 55, 6))
        return [dict(label=l, selected_iq=iq, multiplicity=m,
                     frequencies=np.arange(12, dtype=float),
                     weights=np.arange(12, dtype=float) + .5,
                     payload=l) for l, iq, m in labels]

    def test_join_preserves_canonical_order_and_weights(self):
        joined = join_q_records(self.records()[::-1])
        self.assertEqual([r['label'] for r in joined], [1, 2, 3, 6, 7, 8, 11, 28])
        self.assertEqual([r['selected_iq'] for r in joined], [1, 22, 43, 6, 27, 23, 11, 55])
        self.assertEqual([r['q_weight'] for r in joined], [1/64, 8/64, 4/64, 6/64,
                                                            24/64, 12/64, 3/64, 6/64])

    def test_join_requires_exact_frequency_grid(self):
        rows = self.records()
        rows[2]['frequencies'] = rows[2]['frequencies'].copy()
        rows[2]['frequencies'][4] += 1e-12
        with self.assertRaisesRegex(ValueError, 'frequency'):
            join_q_records(rows)

    def test_contract_rejects_missing_or_duplicate_q(self):
        rows = self.records()[:-1]
        with self.assertRaisesRegex(ValueError, 'complete'):
            canonical_q_contract(rows)

    def test_rejects_weight_changes_short_grids_and_tiny_grid_changes(self):
        rows = self.records()
        rows[3]['multiplicity'] = 12
        with self.assertRaises(ValueError):
            join_q_records(rows)
        rows = self.records()
        for r in rows:
            r['frequencies'] = r['frequencies'][:6]
            r['weights'] = r['weights'][:6]
        with self.assertRaises(ValueError):
            join_q_records(rows)
        rows = self.records()
        rows[2]['weights'][0] = np.nextafter(.5, 1.)
        with self.assertRaises(ValueError):
            join_q_records(rows)
        rows = self.records() + [self.records()[0]]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            canonical_q_contract(rows)


if __name__ == '__main__':
    unittest.main()
