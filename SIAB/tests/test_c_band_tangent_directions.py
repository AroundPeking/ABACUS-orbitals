import copy
import importlib
import importlib.util
import unittest
import torch
import common
from test_periodic_galerkin_direction_calibration import DirectionCalibrationTest


class BandTangentTest(unittest.TestCase):
    def test_two_axes_obey_both_frozen_constraints_without_freezing_radials(self):
        self.assertIsNotNone(importlib.util.find_spec('periodic_galerkin_band_tangent'),
                             'two-constraint direction constructor missing')
        m=importlib.import_module('periodic_galerkin_band_tangent')
        f=DirectionCalibrationTest()
        f.setUp()
        b=[.4,-.2,-.01,.94,-.37,-.013,.73,1.55]
        rows=[dict(element=r['element'],l=r['l'],zeta=r['zeta'],
                   occupied_slope_ev_per_c=r['stencils'][1]['band_sum_derivative_ev_per_atom'],
                   target_slope_ev=x,occupied_two_radius_difference=0.,target_two_radius_difference=0.,
                   active_band_identity='unmeasured') for r,x in zip(f.sensitivity['radials'],b)]
        a=m.build_band_tangent(f.c,f.report,rows)
        self.assertEqual(a['scope'],'two_direction_band_tangent_pbe_calibration')
        self.assertEqual(a['signed_radii'],[-.001,.001,-.002,.002])
        self.assertEqual(a['physical_release_gate'],'hold')
        for name in ('T','N'):
            self.assertAlmostEqual(a['directions'][name]['maximum_band_slope_ev'],0.,places=12)
            self.assertEqual(len(a['directions'][name]['radial_weights']),8)
        self.assertAlmostEqual(a['directions']['T']['frozen_band_slope_ev_per_c'],0.,places=12)
        self.assertLess(a['directions']['T']['ec_slope_ha_per_cell'],0.)
        self.assertGreater(a['tangent_descent_fraction'],0.)
        roundoff=copy.deepcopy(f.report)
        roundoff['channels'][0]['horizontal_residual_norm']+=1e-18
        m.build_band_tangent(f.c,roundoff,rows)
        changed=copy.deepcopy(rows)
        changed[0]['occupied_two_radius_difference']=5e-6
        with self.assertRaises(ValueError): m.build_band_tangent(f.c,f.report,changed)
        for malformed in (rows[::-1],[dict(x,target_slope_ev=float('nan')) for x in rows],
                          [dict(x,target_slope_ev=0.) for x in rows]):
            with self.assertRaises(ValueError): m.build_band_tangent(f.c,f.report,malformed)


if __name__=='__main__': unittest.main()
