import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from common import info


class _AddictDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __missing__(self, key):
        value = type(self)()
        self[key] = value
        return value


if "addict" not in sys.modules:
    sys.modules["addict"] = types.SimpleNamespace(Dict=_AddictDict)


from IO.func_C import read_C_init, write_C
from IO.read_sternheimer import read_sternheimer
import main as siab_main
import main_each as siab_main_each
from sternheimer_data import PrimitiveBlock, SternheimerData
from sternheimer_spillage import OrbitalColumn


COMPONENTS = {
    "total": 0.6,
    "constraint_dpsi": 0.05,
    "sternheimer": 0.25,
    "dft_dpsi": 0.2,
    "constraint_dft": 0.1,
    "dft_origin": 0.3,
}


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def provenance():
    return {
        "abacus_commit": "a" * 40,
        "auxiliary_basis_sha256": "b" * 64,
        "cell_bohr": [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0],
        "ecut_ry": 25.0,
        "kernel": "full_coulomb",
        "orbital_sha256": "c" * 64,
        "pseudopotential_sha256": "d" * 64,
        "spin_convention": "occupation_in_metadata",
    }


def make_block_data():
    blocks = []
    offset = 0
    for atom_index in (0, 1):
        for m in (-1, 0, 1):
            blocks.append(PrimitiveBlock("H", atom_index, 1, m, 2, offset))
            offset += 2
    return SternheimerData(
        format_version=1,
        grid_volume_bohr3=0.125,
        blocks=tuple(blocks),
        occupied_state=torch.tensor([0], dtype=torch.int64),
        auxiliary_channel=torch.tensor([0], dtype=torch.int64),
        frequency_ha=torch.tensor([0.5], dtype=torch.float64),
        occupation=torch.tensor([2.0], dtype=torch.float64),
        frequency_weight=torch.tensor([1.0], dtype=torch.float64),
        norm=torch.tensor([1.0], dtype=torch.float64),
        q=torch.zeros((1, offset), dtype=torch.complex128),
        overlap=torch.eye(offset, dtype=torch.complex128),
        provenance=provenance(),
    )


def coefficient_text():
    return (
        "<Coefficient>\n"
        "\t 2 Total number of radial orbitals.\n"
        "\tType\tL\tZeta-Orbital\n"
        "\t  H \t0\t    1\n"
        "\t   1.00000000000000\n"
        "\t   0.00000000000000\n"
        "\tType\tL\tZeta-Orbital\n"
        "\t  H \t0\t    2\n"
        "\t   0.00000000000000\n"
        "\t   1.00000000000000\n"
        "</Coefficient>\n"
    )


def write_initial_coefficient(path, n_primitive=3):
    lines = [
        "<Coefficient>",
        "\t 2 Total number of radial orbitals.",
        "\tType\tL\tZeta-Orbital",
        "\t  H \t0\t    1",
    ]
    lines.extend(
        "\t %18.14f" % value
        for value in ([1.0] + [0.0] * (n_primitive - 1))
    )
    lines.extend(["\tType\tL\tZeta-Orbital", "\t  H \t0\t    2"])
    lines.extend(
        "\t %18.14f" % value
        for value in ([0.0, 1.0] + [0.0] * (n_primitive - 2))
    )
    lines.extend(["</Coefficient>", "<Mkb>", "Left spillage = 0", "</Mkb>"])
    path.write_text("\n".join(lines) + "\n")


def write_legacy_origin(path):
    lines = [
        "header 1",
        "header 2",
        "header 3",
        "header 4",
        "1 ntype",
        "H",
        "1 natom",
        "0.0 0.0 0.0",
        "ignored 1",
        "ignored 2",
        "ignored 3",
        "ignored 4",
        "ignored 5",
        "ignored 6",
        "0 lmax",
        "1 nks",
        "1 nbands",
        "basis header",
        "3 nprimitive",
        "<WEIGHT_OF_KPOINTS>",
        "0.0 0.0 0.0 1.0",
        "</WEIGHT_OF_KPOINTS>",
        "<OVERLAP_Q>",
        "0.8 0.0 0.2 0.0 0.4 0.0",
        "</OVERLAP_Q>",
        "<OVERLAP_Sq>",
    ]
    for row in range(3):
        for column in range(3):
            lines.append(f"{float(row == column):.1f} 0.0")
    lines.extend(["</OVERLAP_Sq>"])
    path.write_text("\n".join(lines) + "\n")


def write_sternheimer(path):
    overlap = []
    for row in range(3):
        for column in range(3):
            overlap.append(f"{float(row == column):.1f} 0.0")
    path.write_text(
        "\n".join(
            [
                "<STERNHEIMER_SIAB_HEADER>",
                "format_version 1",
                "n_reference 1",
                "n_primitive 3",
                "n_blocks 1",
                "grid_volume_bohr3 0.125",
                "</STERNHEIMER_SIAB_HEADER>",
                "<PRIMITIVE_BLOCKS>",
                "H 0 0 0 3 0",
                "</PRIMITIVE_BLOCKS>",
                "<REFERENCE_METADATA>",
                "0 0 0.5 2.0 1.0 0.5",
                "</REFERENCE_METADATA>",
                "<OVERLAP_Q>",
                "0.4 0.0",
                "0.3 0.0",
                "0.5 0.0",
                "</OVERLAP_Q>",
                "<OVERLAP_S>",
                *overlap,
                "</OVERLAP_S>",
                "<PROVENANCE_JSON>",
                json.dumps(provenance(), sort_keys=True),
                "</PROVENANCE_JSON>",
            ]
        )
        + "\n"
    )


def input_value(origin="missing-origin.dat", sternheimer_marker=True, loss=True):
    value = {
        "file_list": {"origin": [origin]},
        "element": {"Nt_all": ["H"], "Nu": {"H": [2]}},
        "weight": {"stru": [1.0], "bands_range": [1]},
        "optimize": [
            {
                "optimizer": "Adam",
                "kwargs": {"lr": 0.05},
                "cal_T": False,
                "norm": "one",
                "max_steps": 1,
            }
        ],
        "C_init_info": {
            "init_from_file": True,
            "C_init_file": "C_init.dat",
        },
        "V_info": {"init_from_file": False, "same_band": True},
        "radial": {
            "Ecut": 25.0,
            "Rcut": 6.0,
            "dr": 0.1,
            "smearing_sigma": 0.0,
        },
        "freeze_orbitals": [{"element": "H", "l": 0, "zeta": 1}],
    }
    if sternheimer_marker is not False:
        value["file_list"]["sternheimer"] = sternheimer_marker
    if loss:
        value["loss"] = {"mode": "st_only"}
    return value


class FixedOrbitalExpansionTest(unittest.TestCase):
    def setUp(self):
        self.c = {
            "H": [
                torch.ones((1, 1), dtype=torch.float64),
                torch.eye(2, dtype=torch.float64),
            ]
        }
        self.data = make_block_data()

    def test_expands_radial_spec_to_every_atom_and_m_block(self):
        result = siab_main._expand_fixed_orbitals(
            self.data,
            self.c,
            [{"element": "H", "l": 1, "zeta": 1}],
        )

        self.assertEqual(
            result,
            tuple(
                OrbitalColumn("H", atom_index, 1, m, 1)
                for atom_index in (0, 1)
                for m in (-1, 0, 1)
            ),
        )

    def test_rejects_unmapped_and_duplicate_radial_specs(self):
        with self.assertRaisesRegex(ValueError, "maps to no primitive blocks"):
            siab_main._expand_fixed_orbitals(
                self.data,
                self.c,
                [{"element": "H", "l": 0, "zeta": 1}],
            )
        duplicate = [
            {"element": "H", "l": 1, "zeta": 1},
            {"element": "H", "l": 1, "zeta": 1},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            siab_main._expand_fixed_orbitals(self.data, self.c, duplicate)


class MainRoutingTest(unittest.TestCase):
    def run_input(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "INPUT").write_text(json.dumps(value))
            with working_directory(path):
                return siab_main.main()

    def test_requires_exactly_one_sternheimer_file(self):
        message = "the first SIAB Sternheimer implementation requires exactly one data file"
        for marker in ("st.dat", [], ["one", "two"]):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, f"^{message}$"):
                    self.run_input(input_value(sternheimer_marker=marker))

    def test_rejects_missing_or_unused_sternheimer_data(self):
        with self.assertRaisesRegex(ValueError, "requires sternheimer data"):
            self.run_input(input_value(sternheimer_marker=False, loss=True))
        with self.assertRaisesRegex(ValueError, "requires a Sternheimer loss stage"):
            self.run_input(input_value(sternheimer_marker=["st.dat"], loss=False))

    def test_main_each_rejects_sternheimer_immediately(self):
        returned = (
            {"origin": ["origin"], "sternheimer": ["st"]},
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )
        with mock.patch.object(
            siab_main_each.IO.read_json, "read_json", return_value=returned
        ):
            with self.assertRaisesRegex(
                ValueError, "^sternheimer input is supported by main.py only$"
            ):
                siab_main_each.main()


class WriteCoefficientMetadataTest(unittest.TestCase):
    def setUp(self):
        self.c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
                )
            ]
        }

    def test_legacy_output_is_byte_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "legacy.dat"
            write_C(output, self.c, 0.25)
            expected = (
                coefficient_text()
                + "<Mkb>\nLeft spillage = 2.5000000000e-01\n</Mkb>\n"
            )
            self.assertEqual(output.read_bytes(), expected.encode())

    def test_metadata_is_stable_inside_mkb_and_coefficients_still_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.dat"
            write_C(
                output,
                self.c,
                0.6,
                loss_components=COMPONENTS,
                mode="st_only",
            )
            text = output.read_text()

            coefficient_end = text.index("</Coefficient>")
            mkb_start = text.index("<Mkb>")
            mkb_end = text.index("</Mkb>")
            self.assertLess(coefficient_end, mkb_start)
            metadata = text[mkb_start:mkb_end]
            expected_lines = (
                "Mode = st_only",
                "DFT origin loss = 3.0000000000e-01",
                "DFT dpsi loss = 2.0000000000e-01",
                "Sternheimer loss = 2.5000000000e-01",
                "DFT constraint loss = 1.0000000000e-01",
                "dpsi constraint loss = 5.0000000000e-02",
                "Total loss = 6.0000000000e-01",
            )
            positions = [metadata.index(line) for line in expected_lines]
            self.assertEqual(positions, sorted(positions))

            parsed, indices = read_C_init(
                output, {"H": info(Nl=1, Ne=2, Nu=[2])}
            )
            torch.testing.assert_close(parsed["H"][0], self.c["H"][0])
            self.assertEqual(indices, {("H", 0, 0), ("H", 0, 1)})

    def test_rejects_incomplete_or_malformed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad.dat"
            invalid = dict(COMPONENTS)
            invalid.pop("total")
            with self.assertRaisesRegex(ValueError, "loss_components"):
                write_C(output, self.c, 0.5, invalid, "st_only")
            with self.assertRaisesRegex(ValueError, "mode"):
                write_C(output, self.c, 0.5, COMPONENTS, "legacy")
            invalid = dict(COMPONENTS, sternheimer=np.inf)
            with self.assertRaisesRegex(ValueError, "sternheimer"):
                write_C(output, self.c, 0.5, invalid, "st_only")


class MainIntegrationTest(unittest.TestCase):
    def test_real_main_reads_once_writes_metadata_and_freezes_level1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            origin = path / "origin.dat"
            st_path = path / "sternheimer.dat"
            write_legacy_origin(origin)
            write_sternheimer(st_path)
            write_initial_coefficient(path / "C_init.dat")
            value = input_value(
                origin=str(origin), sternheimer_marker=[str(st_path)], loss=True
            )
            (path / "INPUT").write_text(json.dumps(value))

            with working_directory(path), mock.patch.object(
                siab_main.IO.read_sternheimer,
                "read_sternheimer",
                wraps=read_sternheimer,
            ) as reader, mock.patch.object(
                siab_main.orbital, "normalize", return_value=None
            ), mock.patch.object(
                siab_main.orbital, "generate_orbital", return_value={"H": [[]]}
            ), mock.patch.object(
                siab_main.orbital, "orth", return_value=None
            ), mock.patch.object(
                siab_main.IO.print_orbital, "print_orbital", return_value=None
            ), mock.patch.object(
                siab_main.IO.print_orbital, "plot_orbital", return_value=None
            ):
                siab_main.main()

            self.assertEqual(reader.call_count, 1)
            header = (path / "Spillage.dat").read_text().splitlines()[0]
            self.assertIn("sternheimer", header.split())
            self.assertIn("constraint_dft", header.split())

            result_text = (path / "ORBITAL_RESULTS.txt").read_text()
            self.assertIn("Mode = st_only", result_text)
            self.assertIn("Sternheimer loss =", result_text)
            result_c, _ = read_C_init(
                path / "ORBITAL_RESULTS.txt",
                {"H": info(Nl=1, Ne=3, Nu=[2])},
            )
            self.assertTrue(
                torch.equal(
                    result_c["H"][0][:, 0],
                    torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                )
            )


if __name__ == "__main__":
    unittest.main()
