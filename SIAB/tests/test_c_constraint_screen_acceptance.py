import copy
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"))


class ConstraintAcceptanceTest(unittest.TestCase):
    def fixture(self):
        guard=dict(gate=True,scope="frozen_h_band_screen_not_scf_energy",scf_pbe_gate="pending",
            occupied_band_sum_change_ev_per_atom=.0001,maximum_target_band_change_ev=.048,
            minimum_gap_ev=4.3,occupied_band_sum_limit_ev_per_atom=.01,target_band_change_limit_ev=.05,
            k_weight_sum=2.,k_weight_convention="ABACUS_spin_included_no_renormalization")
        rows=[]
        targets=[]
        for l,n in enumerate([3,3,2]):
            for z in range(1,n+1):
                stencils=[]
                for radius in (1e-4,5e-5):
                    a=dict(guard,occupied_band_sum_change_ev_per_atom=.0001-radius*.1,
                           maximum_target_band_change_ev=.048+radius*.37)
                    b=dict(guard,occupied_band_sum_change_ev_per_atom=.0001+radius*.1,
                           maximum_target_band_change_ev=.048-radius*.37)
                    stencils.append(dict(radius=radius,minus=a,plus=b,band_sum_derivative_ev_per_atom=.1))
                rows.append(dict(element="C",l=l,zeta=z,status="success",stencils=stencils,two_size_derivative_difference=0.))
                targets.append(dict(element="C",l=l,zeta=z,stencils=[dict(radius=r,maximum_band_slope_ev=-.37) for r in (1e-4,5e-5)],two_radius_slope_difference_ev=0.,active_band_identity="unmeasured"))
        return dict(status="success",scope="current_ec_step_frozen_constraint_sensitivity_not_pbe",
            center_band_screen=guard,guard_sensitivity=dict(scope="frozen_band_sensitivity_not_actual_PBE_derivative",
            direction="unit_negative_horizontal_gradient_per_saved_radial",radials=rows,
            actual_pbe_sensitivity="unmeasured",physical_release_gate="hold"),target_band_slopes=targets,
            cache_loads=1,guard_evaluations=33,rpa_evaluations=0,backward_passes=0,actual_scf_count=0,
            candidate_count=0,optimizer_steps=0,coefficient_update="none",exported_candidate="none",
            actual_pbe_direction_derivative="unmeasured",physical_release_gate="hold")

    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec("check_c_ec_constraint_screen"),
                             "independent saved-stencil acceptance is required")
        return importlib.import_module("check_c_ec_constraint_screen")

    def test_live_stencil_reconstruction_and_identity(self):
        m=self.module()
        data=self.fixture()
        result=m.analyze_stencils(data)
        self.assertEqual(len(result),8)
        self.assertAlmostEqual(result[4]["target_slope_ev"],-.37)
        self.assertEqual(result[4]["active_band_identity"],"unmeasured")
        for mutation in ("slope","order","nonfinite","false_gate","guard_limit"):
            r=copy.deepcopy(data)
            if mutation=="slope": r["guard_sensitivity"]["radials"][0]["stencils"][0]["band_sum_derivative_ev_per_atom"]+=.01
            if mutation=="order": r["guard_sensitivity"]["radials"].reverse()
            if mutation=="nonfinite": r["guard_sensitivity"]["radials"][0]["stencils"][0]["plus"]["minimum_gap_ev"]=float("nan")
            if mutation=="false_gate": r["guard_sensitivity"]["radials"][0]["stencils"][0]["plus"]["gate"]=False
            if mutation=="guard_limit": r["center_band_screen"]["target_band_change_limit_ev"]=.1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): m.analyze_stencils(r)

    def test_accounting_does_not_invent_equal_memory_snapshots(self):
        m=self.module()
        r=dict(cache_load_seconds=57.,screen_seconds=78.,total_seconds=148.,peak_rss_kib=10827792)
        self.assertEqual(m.validate_timing(r,10834876)["process_peak_rss_kib"],10834876)
        for peak in (1,0):
            with self.assertRaises(ValueError): m.validate_timing(r,peak)

    def test_fixed_scope_and_zero_physics(self):
        m=self.module()
        r=self.fixture()
        m.validate_scope(r)
        for key in ("rpa_evaluations","backward_passes","actual_scf_count","candidate_count"):
            changed=dict(r,**{key:1})
            with self.assertRaises(ValueError): m.validate_scope(changed)


if __name__=="__main__": unittest.main()
