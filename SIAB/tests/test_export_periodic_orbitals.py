import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "SIAB"
    / "example_C_sternheimer"
    / "periodic_basis_optimization"
    / "export_periodic_orbitals.py"
)


def load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_periodic_orbitals", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_orbital_values(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    mesh = next(int(line.split()[1]) for line in lines if line.startswith("Mesh"))
    dr = next(float(line.split()[1]) for line in lines if line.startswith("dr"))
    orbitals = {}
    index = 0
    while index < len(lines):
        if lines[index].split() != ["Type", "L", "N"]:
            index += 1
            continue
        fields = lines[index + 1].split()
        key = (int(fields[1]), int(fields[2]))
        values = []
        index += 2
        while index < len(lines) and len(values) < mesh:
            if lines[index].split() == ["Type", "L", "N"]:
                break
            values.extend(float(value) for value in lines[index].split())
            index += 1
        orbitals[key] = np.asarray(values, dtype=float)
    return mesh, dr, orbitals


def integrate_simpson(values, dx):
    if values.size % 2 != 1:
        raise ValueError("test integration requires an odd mesh")
    return dx / 3.0 * (
        values[0]
        + values[-1]
        + 4.0 * np.sum(values[1:-1:2])
        + 2.0 * np.sum(values[2:-1:2])
    )


class ExportPeriodicOrbitalsTest(unittest.TestCase):
    def test_original_c_tzdp_round_trip_matches_released_orbital(self):
        export = load_export_module()
        coefficient_path = (
            ROOT
            / "SG15_v1.0"
            / "Orbitals_v2.0"
            / "C_TZDP"
            / "info"
            / "10"
            / "ORBITAL_RESULTS.txt"
        )
        reference_path = coefficient_path.parents[2] / "C_gga_10au_100Ry_3s3p2d.orb"
        coefficients = export.read_periodic_optimizer_coefficients(
            coefficient_path,
            element="C",
            radial_rows=31,
            expected_nu=(3, 3, 2, 0, 0),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "C_export.orb"
            export.write_abacus_orbital(
                output,
                coefficients,
                element="C",
                ecut_ry=100.0,
                rcut_bohr=10.0,
                dr_bohr=0.01,
                smoothing_sigma_bohr=0.1,
            )
            mesh, dr, actual = read_orbital_values(output)
        reference_mesh, reference_dr, reference = read_orbital_values(
            reference_path
        )

        self.assertEqual((mesh, dr, set(actual)), (
            reference_mesh,
            reference_dr,
            set(reference),
        ))
        difference = np.concatenate(
            [actual[key] - reference[key] for key in sorted(actual)]
        )
        reference_values = np.concatenate(
            [reference[key] for key in sorted(reference)]
        )
        self.assertLess(np.max(np.abs(difference)), 1.0e-10)
        self.assertLess(
            np.linalg.norm(difference) / np.linalg.norm(reference_values),
            1.0e-11,
        )

    def test_spherical_bessel_roots_match_reference_values(self):
        export = load_export_module()
        reference = {
            0: math.pi,
            1: 4.493409457909064,
            2: 5.763459196894550,
            3: 6.987932000500520,
            4: 8.182561452571240,
        }

        for l, expected in reference.items():
            root = export.spherical_bessel_roots(l, 1)[0]
            self.assertAlmostEqual(root, expected, places=12)
            self.assertLess(abs(export.spherical_bessel_j(l, root)), 1.0e-12)

    def test_rejects_radial_rows_inconsistent_with_bessel_contract(self):
        export = load_export_module()

        with self.assertRaisesRegex(ValueError, "primitive count"):
            export.validate_bessel_contract(
                radial_rows=4,
                ecut_ry=25.0,
                rcut_bohr=2.0,
            )

    def test_writes_finite_orthonormal_abacus_orbitals(self):
        export = load_export_module()
        coefficients = {
            "C": [
                torch.tensor(
                    [[1.0, 0.2], [0.1, 1.0], [0.3, -0.4]],
                    dtype=torch.float64,
                ),
                torch.tensor([[0.2], [0.8], [0.5]], dtype=torch.float64),
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "C_2s1p.orb"
            export.write_abacus_orbital(
                path,
                coefficients,
                element="C",
                ecut_ry=25.0,
                rcut_bohr=2.0,
                dr_bohr=0.02,
                smoothing_sigma_bohr=0.1,
            )
            mesh, dr, orbitals = read_orbital_values(path)
            text = path.read_text(encoding="ascii")

        self.assertEqual(mesh, 101)
        self.assertEqual(dr, 0.02)
        self.assertIn("Energy Cutoff(Ry)           25.0", text)
        self.assertIn("Radius Cutoff(a.u.)         2.0", text)
        self.assertIn("Lmax                        1", text)
        self.assertIn("Number of Sorbital-->       2", text)
        self.assertIn("Number of Porbital-->       1", text)
        self.assertEqual(set(orbitals), {(0, 0), (0, 1), (1, 0)})
        radius = np.arange(mesh, dtype=float) * dr
        for values in orbitals.values():
            self.assertEqual(values.shape, (mesh,))
            self.assertTrue(np.isfinite(values).all())
            self.assertAlmostEqual(values[-1], 0.0, places=13)
            norm = integrate_simpson(values * values * radius * radius, dr)
            self.assertAlmostEqual(norm, 1.0, places=10)
        overlap = integrate_simpson(
            orbitals[(0, 0)] * orbitals[(0, 1)] * radius * radius,
            dr,
        )
        self.assertLess(abs(overlap), 1.0e-10)

    def test_export_is_deterministic(self):
        export = load_export_module()
        coefficients = {
            "C": [torch.tensor([[1.0], [0.2], [-0.1]], dtype=torch.float64)]
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.orb"
            second = Path(directory) / "second.orb"
            kwargs = dict(
                element="C",
                ecut_ry=25.0,
                rcut_bohr=2.0,
                dr_bohr=0.02,
                smoothing_sigma_bohr=0.1,
            )
            export.write_abacus_orbital(first, coefficients, **kwargs)
            export.write_abacus_orbital(second, coefficients, **kwargs)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
