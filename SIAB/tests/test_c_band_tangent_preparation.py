import importlib
import importlib.util
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'))


class BandPreparationTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('prepare_c_band_tangent_probes'),
                             'bounded same-center preparation missing')
        return importlib.import_module('prepare_c_band_tangent_probes')

    def test_band_branch_evidence_keeps_ties_and_detects_switches(self):
        m=self.module()
        def detail(values):
            return dict(maximum_target_band_change_ev=max(map(abs,values)),target_band_details=[
                dict(source_ik=1,target_ik=1,band_indices=list(range(1,len(values)+1)),signed_changes_ev=values)],
                target_band_identity='sorted_eigenvalue_index_not_eigenvector_tracking')
        center=detail([.048,.048,.001])
        probe=detail([.04801,.048011,.002])
        r=m.summarize_target_branches(center,[probe])
        self.assertEqual(r['center_near_active_indices'],[[1,1],[1,2]])
        self.assertTrue(r['probe_maxima_within_center_near_active_set'])
        self.assertEqual(r['smoothness_gate'],'unproven_sorted_eigenvalue_branches')
        r=m.summarize_target_branches(center,[detail([.047,.047,.049])])
        self.assertFalse(r['probe_maxima_within_center_near_active_set'])
        for bad in (dict(probe,maximum_target_band_change_ev=.1),detail([float('nan'),.02])):
            with self.assertRaises(ValueError): m.summarize_target_branches(center,[bad])

    def test_no_scf_response_and_only_existing_eight_probe_definition(self):
        m=self.module()
        source=Path(m.__file__).read_text()
        self.assertIn('export_probes(',source)
        self.assertIn('load_accepted_screen(',source)
        self.assertIn('diagnostics=True',source)
        self.assertIn('actual_scf_count=0',source)
        self.assertNotIn('subprocess',source)
        self.assertNotIn('evaluate_radial_gradients(',source)
        self.assertNotIn('periodic_rpa_objective(',source)
        self.assertIn('FileExistsError',source)


if __name__=='__main__': unittest.main()
