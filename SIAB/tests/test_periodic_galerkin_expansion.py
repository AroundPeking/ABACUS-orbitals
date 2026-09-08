import unittest
from dataclasses import replace
import torch
import common  # noqa: F401
from test_periodic_galerkin_pbe_guard import FrozenBandGuardTest
from test_periodic_galerkin_reduction import ActivePrimitiveReductionTest
from periodic_galerkin_pbe_guard import prepare_frozen_band_guard
from periodic_galerkin_fit import CandidateGuardError
import periodic_galerkin_reduction as reduction
import periodic_galerkin_basis as basis


class ExpansionTest(unittest.TestCase):
    def test_expanded_spectrum_is_opt_in_and_preserves_target_guard(self):
        d,c = FrozenBandGuardTest().fixture()
        big={'C':[torch.eye(3,dtype=torch.float64)]}
        with self.assertRaises(CandidateGuardError): prepare_frozen_band_guard(d,c)(big)
        guard=prepare_frozen_band_guard(d,c,extra_virtual_bands=1,allow_basis_expansion=True)
        actual=guard(big,diagnostics=True,include_unprotected_bands=True)
        self.assertTrue(actual['gate'])
        self.assertEqual(actual['target_band_details'][0]['band_indices'],[1,2])
        self.assertEqual(actual['added_band_details'][0]['band_indices'],[3])
        moved=replace(d,kpoints=tuple(replace(r,hamiltonian_ha=r.hamiltonian_ha.clone()) for r in d.kpoints))
        strict=prepare_frozen_band_guard(moved,c,extra_virtual_bands=1,allow_basis_expansion=True)
        for r in moved.kpoints: r.hamiltonian_ha[1,1]+=.1
        with self.assertRaises(CandidateGuardError): strict(big)
        with self.assertRaises(ValueError): prepare_frozen_band_guard(d,c,allow_basis_expansion='yes')

    def test_reprofile_keeps_exact_operators_and_rejects_missing_angular_data(self):
        d,c=ActivePrimitiveReductionTest().fixture()
        reduced=reduction.reduce_periodic_active_primitives(d,c)
        big={'C':[x.clone() for x in c['C']]}; big['C'][0]=torch.eye(3,dtype=torch.float64)
        view=reduction.reprofile_active_primitives(reduced,big)
        self.assertIsNot(view,reduced)
        self.assertEqual(view.active_primitive_reduction.source_indices,reduced.active_primitive_reduction.source_indices)
        self.assertNotEqual(view.active_primitive_reduction.mapping_sha256,reduced.active_primitive_reduction.mapping_sha256)
        for a,b in zip(view.kpoints,reduced.kpoints):
            self.assertIs(a.overlap,b.overlap); self.assertIs(a.source,b.source)
        full=basis.contract_periodic_candidate_operators(d.kpoints[0],d.primitive_blocks,big)
        compact=basis.contract_periodic_candidate_operators(view.kpoints[0],view.primitive_blocks,big)
        torch.testing.assert_close(full.hamiltonian_ha,compact.hamiltonian_ha,rtol=0,atol=0)
        torch.testing.assert_close(full.overlap,compact.overlap,rtol=0,atol=0)
        self.assertIs(reduction.reprofile_active_primitives(reduced,c),reduced)
        big['C'][1]=torch.ones((3,1),dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'angular support'): reduction.reprofile_active_primitives(reduced,big)

    def test_lowest_resolved_complement_preserves_original_columns(self):
        from periodic_galerkin_expansion import append_smooth_complement
        c={'C':[torch.eye(6,dtype=torch.float64)[:,:2],torch.empty((6,0),dtype=torch.float64)]}
        big,meta=append_smooth_complement(c,'C',0,max_index=5)
        self.assertEqual(meta['primitive_index'],2)
        torch.testing.assert_close(big['C'][0][:,:2],c['C'][0],rtol=0,atol=0)
        torch.testing.assert_close(big['C'][0].T@big['C'][0],torch.eye(3,dtype=torch.float64),rtol=0,atol=1e-14)
        self.assertEqual(c['C'][0].shape[1],2)
        with self.assertRaisesRegex(ValueError,'resolved complement'):
            append_smooth_complement(c,'C',0,max_index=2)


if __name__=='__main__': unittest.main()
