"""Calibrating true PBE must not claim a new RPA optimization result."""

import copy
import importlib
import importlib.util
import math
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest

import torch
import common  # noqa: F401
from periodic_galerkin_radial_diagnostics import radial_gradient_report, radial_guard_sensitivity


class DirectionCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("periodic_galerkin_direction_calibration"),
                             "direction calibration module missing")
        self.m = importlib.import_module("periodic_galerkin_direction_calibration")
        self.c = {"C": [torch.eye(31, dtype=torch.float64)[:, :n].clone()
                         for n in (3, 3, 2, 0, 0)]}
        gradient = {"C": [torch.zeros_like(c) for c in self.c["C"]]}
        i = 0
        for l, c in enumerate(self.c["C"]):
            for z in range(c.shape[1]):
                gradient["C"][l][10+z, z] = i+1.
                i += 1
        self.report = radial_gradient_report(self.c, gradient)
        def guard(c):
            value = sum(float(v[10+z, z]) * (-1 if z % 2 else 1)
                        for v in c["C"] for z in range(v.shape[1]))
            return dict(gate=True, occupied_band_sum_change_ev_per_atom=value,
                        maximum_target_band_change_ev=abs(value), minimum_gap_ev=2.)
        self.sensitivity = radial_guard_sensitivity(self.c, self.report, guard)

    def build(self):
        return self.m.build_directions(self.c, self.report, self.sensitivity)

    def test_two_unit_orthogonal_directions_preserve_all_radials_and_parent(self):
        before = copy.deepcopy(self.c)
        artifact = self.build()
        self.assertEqual(len(artifact["radials"]), 8)
        self.assertEqual(artifact["physical_release_gate"], "hold")
        self.assertEqual(artifact["actual_pbe_direction_derivatives"], "unmeasured")
        directions = [self.m.read_direction(self.c, artifact, k) for k in ("T", "N")]
        flat = [torch.cat([v.flatten() for v in d["C"]]) for d in directions]
        for v in flat:
            self.assertAlmostEqual(float(v.norm()), 1., places=12)
        self.assertAlmostEqual(float(flat[0] @ flat[1]), 0., places=12)
        self.assertAlmostEqual(artifact["directions"]["T"]["frozen_band_slope_ev_per_c"], 0., places=12)
        self.assertLess(artifact["directions"]["T"]["ec_slope_ha_per_cell"], 0.)
        for l, c in enumerate(self.c["C"]):
            torch.testing.assert_close(c, before["C"][l], atol=0, rtol=0)
            for d in directions:
                torch.testing.assert_close(c.T @ d["C"][l], torch.zeros((c.shape[1], c.shape[1]), dtype=c.dtype), atol=1e-12, rtol=0)

    def test_exactly_eight_bounded_signed_qr_probes(self):
        artifact = self.build()
        probes = self.m.signed_probes(self.c, artifact)
        self.assertEqual([(p["direction_name"], p["signed_radius"]) for p in probes],
                         [(name, r) for name in ("T", "N") for r in (-.001, .001, -.002, .002)])
        for p in probes:
            for c in p["coefficients"]["C"]:
                torch.testing.assert_close(c.T @ c, torch.eye(c.shape[1], dtype=c.dtype), atol=1e-12, rtol=0)
            self.assertNotIn("candidate_energy_ha", p)
        self.assertEqual(len({tuple(torch.cat([c.flatten() for c in p["coefficients"]["C"]]).tolist()) for p in probes}), 8)

    def test_invalid_radial_reports_or_unresolved_stencils_stop(self):
        mutations = [lambda r, s: r["channels"][0]["radials"].pop(),
                     lambda r, s: s["radials"].reverse(),
                     lambda r, s: s["radials"][0].update(status="guard_rejected"),
                     lambda r, s: s["radials"][0]["stencils"][0].update(band_sum_derivative_ev_per_atom=math.nan),
                     lambda r, s: r["channels"][0]["horizontal_gradient"][0].__setitem__(0, 1.),
                     lambda r, s: s["radials"][0]["stencils"][0].update(band_sum_derivative_ev_per_atom=5.)]
        for mutate in mutations:
            r, s = copy.deepcopy(self.report), copy.deepcopy(self.sensitivity)
            mutate(r, s)
            with self.assertRaises(ValueError):
                self.m.build_directions(self.c, r, s)

    def test_tampered_saved_direction_rejected(self):
        a = self.build()
        a["directions"]["T"]["matrices"]["C"][0][0][0] = .5
        with self.assertRaises(ValueError):
            self.m.read_direction(self.c, a, "T")

    def test_true_pbe_two_radius_derivatives_and_curvature_not_mixed(self):
        center = -309.854
        samples = [dict(direction_name=name, signed_radius=r, scf_log_gate="pass",
                        candidate_energy_ev=center+2*(slope*r+.5*curvature*r*r))
                   for name, slope, curvature in (("T", .02, 3.), ("N", -.1, 7.))
                   for r in (-.001, .001, -.002, .002)]
        result = self.m.analyze_pbe_axes(center, samples)
        self.assertEqual(result["consistency_gate"], "pass")
        self.assertEqual(result["mixed_curvature"], "unmeasured")
        self.assertEqual(result["physical_release_gate"], "hold")
        for name, slope, curvature in (("T", .02, 3.), ("N", -.1, 7.)):
            for row in result["axes"][name]:
                self.assertAlmostEqual(row["derivative_ev_per_c"], slope, places=8)
                self.assertAlmostEqual(row["curvature_ev_per_c"], curvature, places=7)
        samples[0]["candidate_energy_ev"] += .001
        self.assertEqual(self.m.analyze_pbe_axes(center, samples)["consistency_gate"], "fail")

    def test_failed_scf_duplicate_or_missing_axis_is_not_a_derivative(self):
        rows = [dict(direction_name=name, signed_radius=r, scf_log_gate="pass", candidate_energy_ev=-309.)
                for name in ("T", "N") for r in (-.001, .001, -.002, .002)]
        for data in (rows[:-1], rows+[rows[0]], [dict(rows[0], scf_log_gate="fail")]+rows[1:],
                     [dict(rows[0], candidate_energy_ev=math.inf)]+rows[1:]):
            with self.assertRaises(ValueError):
                self.m.analyze_pbe_axes(-309., data)

    def test_probe_export_is_hash_bound_guarded_and_never_overwrites(self):
        workflow = Path(__file__).resolve().parents[1] / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
        sys.path.insert(0, str(workflow))
        self.assertIsNotNone(importlib.util.find_spec("prepare_c_pbe_direction_calibration"), "probe exporter missing")
        runner = importlib.import_module("prepare_c_pbe_direction_calibration")
        center = dict(coefficient_sha256="b"*64, orbital_sha256="c"*64)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"probes"
            def screen(c):
                self.assertFalse(torch.is_grad_enabled())
                return dict(gate=True, minimum_occupied_capture=.99999)
            out = runner.export_probes(self.c, self.build(), center=center,
                center_result_sha256="a"*64, output=root, screen=screen)
            self.assertEqual(len(out), 8)
            for row in out:
                path = root/row["path"]
                p = json.loads(path.read_text())
                self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(p["scope"], "direction_calibration_probe")
                self.assertEqual(p["center_result_sha256"], "a"*64)
                self.assertEqual(p["physical_release_gate"], "hold")
                self.assertNotIn("candidate_energy_ha", p)
                for key in ("coefficient", "orbital"):
                    self.assertEqual(p[key+"_sha256"], hashlib.sha256((path.parent/p[key+"_filename"]).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                runner.export_probes(self.c, self.build(), center=center,
                    center_result_sha256="a"*64, output=root, screen=screen)
            rejected = runner.export_probes(self.c, self.build(), center=center,
                center_result_sha256="a"*64, output=Path(temp)/"rejected", screen=lambda c: dict(gate=False))
            self.assertTrue(all(p["status"] == "rejected" for p in rejected))
            self.assertFalse(list((Path(temp)/"rejected").rglob("*.orb")))

    def test_runner_binds_parent_diagnostic_and_only_uses_cheap_screens(self):
        import test_periodic_galerkin_c_backoff as fixtures
        import test_c_optimized_pbe as pbe
        fixture = fixtures.CBackoffTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.assertIsNotNone(importlib.util.find_spec("prepare_c_pbe_direction_calibration"), "calibration runner missing")
        runner = importlib.import_module("prepare_c_pbe_direction_calibration")
        candidate = fixture.produce()
        source = fixture.output/"RESULT.json"
        fixture.fixture.prepare(optimizer_result=source, optimizer_result_sha256=fixtures.sha(source))
        collection = fixture.fixture.collect(pbe.log_text())
        c = fixture.backoff.read_periodic_optimizer_coefficients(fixture.output/"INTERPOLATED_COEFFICIENTS.txt",
            element="C", radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
        raw = {"C": [torch.zeros_like(v) for v in c["C"]]}
        for l, v in enumerate(raw["C"]):
            for z in range(v.shape[1]):
                v[10+z, z] = 1.+z+l
        report = radial_gradient_report(c, raw)
        def guard(c):
            return dict(gate=True, occupied_band_sum_change_ev_per_atom=float(c["C"][0][10, 0]),
                        maximum_target_band_change_ev=0., minimum_gap_ev=2.)
        d = dict(status="success", coefficient_sha256=candidate["coefficient_sha256"],
            orbital_sha256=candidate["orbital_sha256"], candidate_result_sha256=fixtures.sha(source),
            pbe_collection_sha256=fixtures.sha(fixture.fixture.output/pbe.endpoint.COLLECTION),
            freeze_sha256=candidate["freeze_sha256"], active_cache_index_sha256=candidate["active_cache_index_sha256"],
            nu=[3, 3, 2, 0, 0], fixed_nu=[0]*5, optimizer_steps=0,
            energy_gradient=report, guard_sensitivity=dict(energy_gradient=radial_guard_sensitivity(c, report, guard)))
        diagnostic = fixture.root/"diagnostic.json"
        diagnostic.write_text(json.dumps(d))
        options = dict(center_result=source, center_result_sha256=fixtures.sha(source),
            pbe_collection=fixture.fixture.output/pbe.endpoint.COLLECTION,
            pbe_collection_sha256=d["pbe_collection_sha256"], diagnostic=diagnostic,
            diagnostic_sha256=fixtures.sha(diagnostic), output=fixture.root/"calibration", source_commit="e"*40)
        with mock.patch.object(runner, "load_frozen_c", return_value=((), [], guard)), \
                mock.patch.object(runner, "_minimum_occupied_capture", return_value=.99999) as capture:
            result = runner.prepare_calibration(**options)
        self.assertEqual(capture.call_count, 8)
        self.assertEqual(len(result["probes"]), 8)
        self.assertEqual(result["rpa_evaluations"], 0)
        self.assertEqual(result["center_result_sha256"], fixtures.sha(source))
        self.assertEqual(result["physical_release_gate"], "hold")
        with mock.patch.object(runner, "load_frozen_c") as load, self.assertRaises(ValueError):
            runner.prepare_calibration(**dict(options, diagnostic_sha256="f"*64, output=fixture.root/"bad"))
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
