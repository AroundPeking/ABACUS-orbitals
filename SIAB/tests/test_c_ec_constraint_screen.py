"""Constraint screening orchestration has no SCF or response evaluation."""

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

WF = Path(__file__).resolve().parents[1] / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
sys.path.insert(0, str(WF))


class ConstraintScreen(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("screen_c_ec_step_constraints"))
        self.m = importlib.import_module("screen_c_ec_step_constraints")

    def test_band_reproduction_tolerates_only_roundoff_not_changed_contract(self):
        original=dict(gate=True, maximum_target_band_change_ev=.04837416187687653, scope="frozen")
        self.m.compare_center_guard(dict(original,maximum_target_band_change_ev=original["maximum_target_band_change_ev"]+1e-15), original)
        for value in (dict(original,gate=False),dict(original,scope="SCF"),
                dict(original,maximum_target_band_change_ev=.049),dict(original,extra=1)):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.m.compare_center_guard(value,original)

    def test_derived_band_slopes_preserve_two_radii_and_do_not_claim_pbe(self):
        rows = []
        for l,n in enumerate([3,3,2]):
            for z in range(n):
                stencils=[]
                for r in (1e-4,5e-5):
                    stencils.append(dict(radius=r,
                        minus=dict(gate=True, maximum_target_band_change_ev=.048-2*r),
                        plus=dict(gate=True, maximum_target_band_change_ev=.048+2*r)))
                rows.append(dict(element="C", l=l, zeta=z+1, status="success", stencils=stencils))
        s=dict(scope="frozen_band_sensitivity_not_actual_PBE_derivative", radials=rows)
        r=self.m.target_band_slopes(s)
        self.assertEqual(len(r),8)
        for row in r:
            self.assertAlmostEqual(row["stencils"][0]["maximum_band_slope_ev"],2.)
            self.assertLess(row["two_radius_slope_difference_ev"],1e-10)
            self.assertEqual(row["active_band_identity"],"unmeasured")
        for change in ("identity","status","nan"):
            bad=copy.deepcopy(s)
            if change=="identity": bad["radials"][0]["l"]=2
            if change=="status": bad["radials"][0]["status"]="guard_rejected"
            if change=="nan": bad["radials"][0]["stencils"][0]["plus"]["maximum_target_band_change_ev"]=float("nan")
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.m.target_band_slopes(bad)

    def test_one_cache_existing_gradient_no_new_gradient_or_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)/"new"; stage.mkdir()
            parent=Path(tmp)/"old"; parent.mkdir()
            center=dict(coefficient_path=parent/"C", original_coefficient_path=parent/"ORIGINAL",
                freeze_path=parent/"F",active_cache_index_path=parent/"I",
                result=dict(freeze_sha256="f"*64,active_cache_index_sha256="a"*64,
                    candidate=dict(coefficient_guard={"gate":True})))
            gradient=dict(coefficient_sha256="c"*64,orbital_sha256="d"*64,
                energy_gradient={"record":"existing"}, accepted_result_sha256="e"*64)
            guard=mock.Mock(return_value={"gate":True})
            rt=SimpleNamespace(torch=SimpleNamespace(set_num_threads=mock.Mock(),no_grad=mock.MagicMock()),
                read_coefficients=mock.Mock(return_value={"C":[]}),
                load_frozen_c=mock.Mock(return_value=([],[],guard)),
                radial_guard_sensitivity=mock.Mock(return_value={"sensitivity":"existing_kernel"}))
            options=dict(stage=stage,source_commit="b"*40,deployment_sha256="1"*64,
                gradient_stage=parent,gradient_result_sha256="2"*64,gradient_acceptance_sha256="3"*64)
            with mock.patch.object(self.m,"verify_source"),mock.patch.object(self.m,"admit_gradient",return_value=(center,gradient)),mock.patch.object(self.m,"_runtime",return_value=rt),mock.patch.object(self.m.common,"validate_dataset_extent"),mock.patch.object(self.m.checks,"_same_json"),mock.patch.object(self.m,"target_band_slopes",return_value=[]):
                r=self.m.screen(**options)
                for key in ("actual_scf_count","rpa_evaluations","backward_passes","candidate_count","optimizer_steps"):
                    self.assertEqual(r[key],0)
                self.assertEqual(r["cache_loads"],1)
                self.assertEqual(r["actual_pbe_direction_derivative"],"unmeasured")
                self.assertEqual(r["physical_release_gate"],"hold")
                rt.radial_guard_sensitivity.assert_called_once()
                self.assertEqual(json.loads((stage/"result/CONSTRAINT_SCREEN.json").read_text()),r)
                with self.assertRaises(FileExistsError):
                    self.m.screen(**options)

    def test_output_must_not_overlap_gradient_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for stage in (p,p/"child"):
                stage.mkdir(exist_ok=True)
                with self.subTest(stage=stage),self.assertRaises(ValueError):
                    self.m.screen(stage=stage,gradient_stage=p,source_commit="a"*40,
                        deployment_sha256="b"*64,gradient_result_sha256="c"*64,
                        gradient_acceptance_sha256="d"*64)


if __name__=="__main__":
    unittest.main()
