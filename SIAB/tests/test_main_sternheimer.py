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
from optimization_loss import LOSS_DEFAULTS
from sternheimer_data import PrimitiveBlock, SternheimerData
from sternheimer_spillage import OrbitalColumn
from test_sternheimer_spillage import make_sternheimer_data


COMPONENTS = {
    "total": 0.6,
    "constraint_dpsi": 0.05,
    "sternheimer": 0.25,
    "regularization_dpsi": 0.0,
    "radial_tail": 0.02,
    "regularization_locality": 0.0,
    "dft_dpsi": 0.2,
    "constraint_dft": 0.1,
    "dft_origin": 0.3,
}

GUARDED_COMPONENTS = dict(
    COMPONENTS,
    sternheimer_lowest_frequency=0.23,
    regularization_low_frequency=0.04,
)

GUARDED_DIAGNOSTICS = {
    "max_st_condition": 12.0,
    "max_locality_condition": 8.0,
    "lowest_st_frequency_ha": 0.068706555678,
    "initial_lowest_st_loss": 0.247384,
    "final_lowest_st_loss": 0.247300,
    "low_frequency_guard_tolerance": 0.0,
    "low_frequency_guard_weight": 10.0,
}

PROJECTED_PI_COMPONENTS = {
    "dft_origin": 0.3,
    "dft_dpsi": 0.2,
    "projected_pi": 0.214,
    "regularization_dpsi": 0.2,
    "constraint_dft": 0.0,
    "constraint_dpsi": 0.0,
    "total": 0.414,
}

PROJECTED_PI_DIAGNOSTICS = {
    "max_projected_pi_condition": 5562.0,
    "lowest_projected_pi_frequency_ha": 0.0687,
    "lowest_projected_pi_loss": 0.19,
    "projected_pi_rank_tolerance": 1.0e-12,
}


class RoutingConstructionObserved(RuntimeError):
    pass


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

    def test_sternheimer_info_allows_zero_unselected_channels(self):
        blocks = [PrimitiveBlock("H", 0, 0, 0, 2, 0)]
        for index, m in enumerate((-1, 0, 1)):
            blocks.append(PrimitiveBlock("H", 0, 1, m, 2, 2 + 2 * index))
        data = make_sternheimer_data(blocks, torch.zeros(8))

        result = siab_main._sternheimer_info_element(
            data,
            info(Nt_all=["H"], Nu={"H": [1, 0]}),
        )

        self.assertEqual(result["H"].Nu, [1, 0])


class MainRoutingTest(unittest.TestCase):
    def run_input(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "INPUT").write_text(json.dumps(value))
            with working_directory(path):
                return siab_main.main()

    def test_rejects_nonlist_or_empty_sternheimer_targets(self):
        message = "SIAB Sternheimer targets require a nonempty list"
        for marker in ("st.dat", []):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, f"^{message}$"):
                    self.run_input(input_value(sternheimer_marker=marker))

    def test_accepts_one_named_physical_target(self):
        target = {
            "path": "st.dat",
            "family": "atom",
            "role": "physical",
        }
        with mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            return_value="loaded",
        ) as reader:
            targets, stages = siab_main._load_sternheimer_data(
                {"sternheimer": [target]},
                [{"loss": {"mode": "st_only"}}],
            )

        self.assertEqual(targets.entries[0].family, "atom")
        self.assertEqual(targets.families[0].name, "atom")
        self.assertEqual(targets.families[0].data, ("loaded",))
        self.assertEqual(stages[0]["mode"], "st_only")
        reader.assert_called_once_with(Path("st.dat"))

    def test_groups_multiple_physical_targets_and_does_not_load_ghost(self):
        targets = [
            {"path": "atom-a.dat", "family": "atom", "role": "physical"},
            {"path": "atom-b.dat", "family": "atom", "role": "physical"},
            {
                "path": "h3.dat",
                "family": "multicenter",
                "role": "physical",
            },
            {
                "path": "ghost.dat",
                "family": "fragment_ghost",
                "role": "ghost",
            },
        ]
        with mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            side_effect=("atom-a", "atom-b", "h3"),
        ) as reader:
            loaded, _ = siab_main._load_sternheimer_data(
                {"sternheimer": targets},
                [{"loss": {"mode": "st_dpsi_joint"}}],
            )

        self.assertEqual(
            [(family.name, family.data) for family in loaded.families],
            [("atom", ("atom-a", "atom-b")), ("multicenter", ("h3",))],
        )
        self.assertEqual(reader.call_count, 3)
        self.assertEqual(len(loaded.entries), 4)

    @staticmethod
    def projected_pi_targets():
        return [
            {
                "path": "H_response.dat",
                "source_path": "H_source.dat",
                "zero_order_audit_path": "H_audit.json",
                "family": "H",
                "role": "physical",
            },
            {
                "path": "H2_response.dat",
                "source_path": "H2_source.dat",
                "zero_order_audit_path": "H2_audit.json",
                "family": "H2",
                "role": "physical",
            },
        ]

    @staticmethod
    def rpa_sensitive_loss():
        return {
            "mode": "pi_rpa_sensitive_joint",
            "projected_pi_rank_tolerance": 1.0e-12,
            "projected_pi_sensitivity_alpha": 0.25,
            "joint_dpsi_weight": 0.02,
        }

    def test_loads_one_strict_source_response_audit_pair_per_family(self):
        targets = self.projected_pi_targets()
        responses = ("H response", "H2 response")
        sources = ("H source", "H2 source")
        pairs = ("H pair", "H2 pair")
        audits = ("H audit", "H2 audit")
        with mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            side_effect=responses,
        ) as response_reader, mock.patch.object(
            siab_main.IO.read_sternheimer_source,
            "read_sternheimer_source",
            side_effect=sources,
        ) as source_reader, mock.patch.object(
            siab_main,
            "apply_target_element_aliases",
            side_effect=lambda data, entry: data,
        ) as aliases, mock.patch.object(
            siab_main,
            "pair_response_and_source",
            side_effect=pairs,
        ) as pairer, mock.patch.object(
            siab_main,
            "read_zero_order_audit",
            side_effect=audits,
        ) as audit_reader:
            loaded, stages = siab_main._load_sternheimer_data(
                {"sternheimer": targets},
                [{"loss": {"mode": "pi_dpsi_joint"}}],
            )

        self.assertEqual(stages[0]["mode"], "pi_dpsi_joint")
        self.assertEqual(loaded.projected_pi_pairs, tuple(zip(("H", "H2"), pairs)))
        self.assertEqual(loaded.zero_order_audits, tuple(zip(("H", "H2"), audits)))
        self.assertEqual(response_reader.call_count, 2)
        self.assertEqual(source_reader.call_count, 2)
        self.assertEqual(aliases.call_count, 4)
        pairer.assert_has_calls(
            [mock.call(responses[0], sources[0]), mock.call(responses[1], sources[1])]
        )
        audit_reader.assert_has_calls(
            [
                mock.call(Path("H_audit.json"), "H"),
                mock.call(Path("H2_audit.json"), "H2"),
            ]
        )

    def test_projected_pi_rejects_incomplete_or_nonphysical_targets(self):
        targets = self.projected_pi_targets()
        invalid_campaigns = (
            ([{key: value for key, value in targets[0].items() if key != "source_path"}, targets[1]], "source_path"),
            ([{key: value for key, value in targets[0].items() if key != "zero_order_audit_path"}, targets[1]], "zero_order_audit_path"),
            ([targets[0]], "one H and one H2"),
            ([targets[0], {**targets[1], "family": "H"}], "one H and one H2"),
            ([targets[0], {**targets[1], "role": "ghost", "source_path": None, "zero_order_audit_path": None}], "ghost"),
        )
        for campaign, message in invalid_campaigns:
            campaign = [
                {key: value for key, value in target.items() if value is not None}
                for target in campaign
            ]
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    siab_main._load_sternheimer_data(
                        {"sternheimer": campaign},
                        [{"loss": {"mode": "pi_dpsi_joint"}}],
                    )

    def test_rejects_mixed_projected_pi_and_legacy_stages(self):
        with self.assertRaisesRegex(ValueError, "cannot mix.*pi_dpsi_joint"):
            siab_main._load_sternheimer_data(
                {"sternheimer": self.projected_pi_targets()},
                [
                    {"loss": {"mode": "pi_dpsi_joint"}},
                    {"loss": {"mode": "st_dpsi_joint"}},
                ],
            )

    def test_rpa_sensitive_loader_builds_one_physical_pair_per_family(self):
        targets = self.projected_pi_targets()
        responses = ("H response", "H2 response")
        sources = ("H source", "H2 source")
        pairs = ("H pair", "H2 pair")
        audits = ("H audit", "H2 audit")
        with mock.patch.object(
            siab_main,
            "normalize_loss_config",
            side_effect=lambda config: config,
        ), mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            side_effect=responses,
        ) as response_reader, mock.patch.object(
            siab_main.IO.read_sternheimer_source,
            "read_sternheimer_source",
            side_effect=sources,
        ) as source_reader, mock.patch.object(
            siab_main,
            "apply_target_element_aliases",
            side_effect=lambda data, entry: data,
        ), mock.patch.object(
            siab_main,
            "pair_response_and_source",
            side_effect=pairs,
        ) as pairer, mock.patch.object(
            siab_main,
            "read_zero_order_audit",
            side_effect=audits,
        ) as audit_reader:
            loaded, stages = siab_main._load_sternheimer_data(
                {"sternheimer": targets},
                [{"loss": self.rpa_sensitive_loss()}],
            )

        self.assertEqual(stages[0]["mode"], "pi_rpa_sensitive_joint")
        self.assertEqual(loaded.projected_pi_pairs, tuple(zip(("H", "H2"), pairs)))
        self.assertEqual(loaded.zero_order_audits, tuple(zip(("H", "H2"), audits)))
        self.assertEqual(response_reader.call_count, 2)
        self.assertEqual(source_reader.call_count, 2)
        pairer.assert_has_calls(
            [mock.call(responses[0], sources[0]), mock.call(responses[1], sources[1])]
        )
        audit_reader.assert_has_calls(
            [
                mock.call(Path("H_audit.json"), "H"),
                mock.call(Path("H2_audit.json"), "H2"),
            ]
        )

    def test_rpa_sensitive_loader_requires_exactly_h_and_h2(self):
        with mock.patch.object(
            siab_main,
            "normalize_loss_config",
            side_effect=lambda config: config,
        ), mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            return_value="loaded",
        ), self.assertRaisesRegex(ValueError, "exactly one H and one H2"):
            siab_main._load_sternheimer_data(
                {"sternheimer": [self.projected_pi_targets()[0]]},
                [{"loss": self.rpa_sensitive_loss()}],
            )

    def test_rpa_sensitive_loader_rejects_duplicate_h(self):
        targets = self.projected_pi_targets()
        targets[1] = {**targets[1], "family": "H"}
        with mock.patch.object(
            siab_main,
            "normalize_loss_config",
            side_effect=lambda config: config,
        ), mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            return_value="loaded",
        ), self.assertRaisesRegex(ValueError, "exactly one H and one H2"):
            siab_main._load_sternheimer_data(
                {"sternheimer": targets},
                [{"loss": self.rpa_sensitive_loss()}],
            )

    def test_rpa_sensitive_loader_rejects_duplicate_h2(self):
        targets = self.projected_pi_targets()
        targets[0] = {**targets[0], "family": "H2"}
        with mock.patch.object(
            siab_main,
            "normalize_loss_config",
            side_effect=lambda config: config,
        ), mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            return_value="loaded",
        ), self.assertRaisesRegex(ValueError, "exactly one H and one H2"):
            siab_main._load_sternheimer_data(
                {"sternheimer": targets},
                [{"loss": self.rpa_sensitive_loss()}],
            )

    def test_rpa_sensitive_loader_rejects_ghost_target(self):
        targets = self.projected_pi_targets()
        targets[1] = {
            key: value
            for key, value in {**targets[1], "role": "ghost"}.items()
            if key not in {"source_path", "zero_order_audit_path"}
        }
        with mock.patch.object(
            siab_main,
            "normalize_loss_config",
            side_effect=lambda config: config,
        ), mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            return_value="loaded",
        ), self.assertRaisesRegex(ValueError, "ghost"):
            siab_main._load_sternheimer_data(
                {"sternheimer": targets},
                [{"loss": self.rpa_sensitive_loss()}],
            )

    def test_rpa_sensitive_loader_requires_source_path(self):
        targets = self.projected_pi_targets()
        targets[0] = {
            key: value
            for key, value in targets[0].items()
            if key != "source_path"
        }
        with mock.patch.object(
            siab_main,
            "normalize_loss_config",
            side_effect=lambda config: config,
        ), mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            return_value="loaded",
        ), self.assertRaisesRegex(ValueError, "source_path"):
            siab_main._load_sternheimer_data(
                {"sternheimer": targets},
                [{"loss": self.rpa_sensitive_loss()}],
            )

    def assert_mixed_rpa_sensitive_stage_rejected(self, legacy_mode):
        with mock.patch.object(
            siab_main,
            "normalize_loss_config",
            side_effect=lambda config: config,
        ), mock.patch.object(
            siab_main.IO.read_sternheimer,
            "read_sternheimer",
            return_value="loaded",
        ), self.assertRaisesRegex(
            ValueError, "cannot mix.*pi_rpa_sensitive_joint"
        ):
            siab_main._load_sternheimer_data(
                {"sternheimer": self.projected_pi_targets()},
                [
                    {"loss": self.rpa_sensitive_loss()},
                    {"loss": {"mode": legacy_mode}},
                ],
            )

    def test_rejects_mixed_rpa_sensitive_and_st_only_stages(self):
        self.assert_mixed_rpa_sensitive_stage_rejected("st_only")

    def test_rejects_mixed_rpa_sensitive_and_st_constrained_stages(self):
        self.assert_mixed_rpa_sensitive_stage_rejected("st_constrained")

    def test_rejects_mixed_rpa_sensitive_and_st_dpsi_joint_stages(self):
        self.assert_mixed_rpa_sensitive_stage_rejected("st_dpsi_joint")

    def test_rejects_mixed_rpa_sensitive_and_pi_dpsi_joint_stages(self):
        self.assert_mixed_rpa_sensitive_stage_rejected("pi_dpsi_joint")

    def _run_until_projected_pi_construction(self, stage):
        pairs = (("H", object()), ("H2", object()))
        targets = siab_main.LoadedSternheimerTargets(
            (),
            (types.SimpleNamespace(data=(object(),)),),
            pairs,
            (),
        )
        file_list = {"origin": ["origin"], "linear": ["dpsi"]}
        info_element = {"H": info(Nl=1, Ne=2, Nu=[2])}
        coefficient = {
            "H": [
                torch.eye(2, dtype=torch.float64, requires_grad=True)
            ]
        }
        constructor = mock.Mock(side_effect=RoutingConstructionObserved)
        legacy_constructor = mock.Mock(
            side_effect=RoutingConstructionObserved
        )
        returned = (
            file_list,
            info(Nt_all=["H"], Nu={"H": [2]}),
            {},
            [{"loss": stage}],
            {
                "init_from_file": False,
                "freeze_orbitals": [
                    {"element": "H", "l": 0, "zeta": 1}
                ],
                "seed": 0,
            },
            {"same_band": True},
            {"Rcut": 6.0},
        )
        with mock.patch.object(
            siab_main.IO.read_json, "read_json", return_value=returned
        ), mock.patch.object(
            siab_main,
            "_load_sternheimer_data",
            return_value=(targets, [stage]),
        ), mock.patch.object(
            siab_main.IO.cal_weight, "cal_weight", return_value="weight"
        ), mock.patch.object(
            siab_main.IO.read_QSV, "read_file_head", return_value="kst"
        ), mock.patch.object(
            siab_main.IO.change_info,
            "change_info",
            return_value=(["structure"], info_element),
        ), mock.patch.object(
            siab_main.IO.read_QSV,
            "read_QSV",
            return_value=("q", "s", "v"),
        ), mock.patch.object(
            siab_main.IO.func_C,
            "random_C_init",
            return_value=coefficient,
        ), mock.patch.object(
            siab_main.orbital, "set_E", return_value="energy"
        ), mock.patch.object(
            siab_main,
            "_normalize_initial_coefficients",
            return_value=None,
        ), mock.patch.object(
            siab_main,
            "NormalizedPhysicalFamilyProjectedPiOptimization",
            constructor,
        ), mock.patch.object(
            siab_main,
            "NormalizedPhysicalFamilySpillage",
            legacy_constructor,
        ), self.assertRaises(RoutingConstructionObserved):
            siab_main.main()
        return pairs, constructor, legacy_constructor

    def test_rpa_sensitive_routes_alpha_rank_and_fourth_order_power(self):
        stage = {
            **LOSS_DEFAULTS,
            **self.rpa_sensitive_loss(),
            "condition_limit": 7.0e8,
        }
        pairs, constructor, legacy_constructor = (
            self._run_until_projected_pi_construction(stage)
        )

        constructor.assert_called_once_with(
            *pairs,
            relative_rank_tolerance=1.0e-12,
            condition_limit=7.0e8,
            sensitivity_alpha=0.25,
            family_power=4,
        )
        legacy_constructor.assert_not_called()

    def test_pi_dpsi_joint_adapter_construction_remains_unchanged(self):
        stage = {
            **LOSS_DEFAULTS,
            "mode": "pi_dpsi_joint",
            "projected_pi_rank_tolerance": 1.0e-12,
            "condition_limit": 7.0e8,
        }
        pairs, constructor, legacy_constructor = (
            self._run_until_projected_pi_construction(stage)
        )

        constructor.assert_called_once_with(
            *pairs,
            relative_rank_tolerance=1.0e-12,
            condition_limit=7.0e8,
        )
        legacy_constructor.assert_not_called()

    def test_rpa_sensitive_main_requires_origin(self):
        stage = {**LOSS_DEFAULTS, **self.rpa_sensitive_loss()}
        targets = siab_main.LoadedSternheimerTargets(
            (), (types.SimpleNamespace(data=(object(),)),), (), ()
        )
        returned = (
            {"sternheimer": ["target"]},
            object(),
            object(),
            [{"loss": stage}],
            {"seed": 0},
            object(),
            object(),
        )
        with mock.patch.object(
            siab_main.IO.read_json, "read_json", return_value=returned
        ), mock.patch.object(
            siab_main,
            "_load_sternheimer_data",
            return_value=(targets, [stage]),
        ), self.assertRaisesRegex(
            ValueError, "pi_rpa_sensitive_joint requires origin and dpsi data"
        ):
            siab_main.main()

    def test_rpa_sensitive_main_requires_linear_dpsi(self):
        stage = {**LOSS_DEFAULTS, **self.rpa_sensitive_loss()}
        targets = siab_main.LoadedSternheimerTargets(
            (), (types.SimpleNamespace(data=(object(),)),), (), ()
        )
        returned = (
            {"origin": ["origin"], "sternheimer": ["target"]},
            object(),
            object(),
            [{"loss": stage}],
            {"seed": 0},
            {"same_band": True},
            object(),
        )
        with mock.patch.object(
            siab_main.IO.read_json, "read_json", return_value=returned
        ), mock.patch.object(
            siab_main,
            "_load_sternheimer_data",
            return_value=(targets, [stage]),
        ), mock.patch.object(
            siab_main.IO.cal_weight,
            "cal_weight",
            side_effect=AssertionError(
                "new-mode routing did not reject missing dpsi"
            ),
        ), self.assertRaisesRegex(
            ValueError, "pi_rpa_sensitive_joint requires linear dpsi data"
        ):
            siab_main.main()

    def test_projected_pi_json_records_full_training_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            pairs = []
            audits = []
            for family in ("H", "H2"):
                response = root / f"{family}_response.dat"
                source = root / f"{family}_source.dat"
                audit_path = root / f"{family}_audit.json"
                response.write_text(f"{family} response\n")
                source.write_text(f"{family} source\n")
                audit_path.write_text(f"{family} audit\n")
                entries.append(
                    types.SimpleNamespace(
                        family=family,
                        path=response,
                        source_path=source,
                        zero_order_audit_path=audit_path,
                    )
                )
                pairs.append(
                    (
                        family,
                        types.SimpleNamespace(
                            response=types.SimpleNamespace(
                                provenance={"kernel": "full_coulomb"}
                            ),
                            source=types.SimpleNamespace(
                                provenance={"kernel": "full_coulomb"}
                            ),
                            provenance_warnings=(),
                        ),
                    )
                )
                audits.append(
                    (
                        family,
                        types.SimpleNamespace(
                            passed=True,
                            occupied_state_count=1,
                            grid=(180, 180, 180),
                            max_occupation_abs_diff=0.0,
                            max_occupied_eigenvalue_abs_diff_ha=0.0,
                            final_total_energy_abs_diff_ha=0.0,
                            source_file_sha256=(("old_eig_occ", "1" * 64),),
                        ),
                    )
                )
            targets = siab_main.LoadedSternheimerTargets(
                tuple(entries), (), tuple(pairs), tuple(audits)
            )
            output = root / "PROJECTED_PI_METADATA.json"
            diagnostics = {
                "frequency_ha": [0.1, 1.0],
                "frequency_loss": [0.2, 0.1],
                "lowest_frequency_ha": 0.1,
                "lowest_frequency_loss": 0.2,
                "max_condition": 10.0,
                "rank_tolerance": 1.0e-12,
                "family_names": ["H", "H2"],
                "families": {"H": {"loss": 0.1}, "H2": {"loss": 0.11}},
            }

            siab_main._write_projected_pi_metadata(
                output,
                targets,
                {
                    "loss_components": PROJECTED_PI_COMPONENTS,
                    "projected_pi_diagnostics": diagnostics,
                },
            )

            payload = json.loads(output.read_text())
            self.assertEqual(payload["mode"], "pi_dpsi_joint")
            self.assertEqual(payload["projected_pi"], diagnostics)
            self.assertFalse(payload["uses_sos_energy"])
            self.assertFalse(payload["uses_ghost_family"])
            for family in ("H", "H2"):
                record = payload["inputs"][family]
                self.assertEqual(len(record["response_sha256"]), 64)
                self.assertEqual(len(record["source_sha256"]), 64)
                self.assertEqual(len(record["zero_order_audit_sha256"]), 64)
                self.assertEqual(record["response_provenance"]["kernel"], "full_coulomb")
                self.assertTrue(record["zero_order_identity"]["passed"])

    def test_rejects_targets_without_a_physical_family(self):
        target = {
            "path": "ghost.dat",
            "family": "fragment_ghost",
            "role": "ghost",
        }
        with self.assertRaisesRegex(
            ValueError, "SIAB optimization requires a physical Sternheimer target"
        ):
            siab_main._load_sternheimer_data(
                {"sternheimer": [target]},
                [{"loss": {"mode": "st_only"}}],
            )

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
                diagnostics={
                    "max_st_condition": 12.0,
                    "max_locality_condition": 8.0,
                },
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
                "dpsi regularization loss = 0.0000000000e+00",
                "DFT constraint loss = 1.0000000000e-01",
                "dpsi constraint loss = 5.0000000000e-02",
                "Radial tail fraction = 2.0000000000e-02",
                "Radial locality regularization loss = 0.0000000000e+00",
                "Total loss = 6.0000000000e-01",
                "Maximum ST overlap condition = 1.2000000000e+01",
                "Maximum radial locality condition = 8.0000000000e+00",
            )
            positions = [metadata.index(line) for line in expected_lines]
            self.assertEqual(positions, sorted(positions))

            parsed, indices = read_C_init(
                output, {"H": info(Nl=1, Ne=2, Nu=[2])}
            )
            torch.testing.assert_close(parsed["H"][0], self.c["H"][0])
            self.assertEqual(indices, {("H", 0, 0), ("H", 0, 1)})

    def test_guarded_metadata_uses_complete_explicit_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "guarded.dat"
            write_C(
                output,
                self.c,
                0.6,
                loss_components=GUARDED_COMPONENTS,
                mode="st_dpsi_joint",
                diagnostics=GUARDED_DIAGNOSTICS,
            )
            text = output.read_text()
            expected_lines = (
                "Sternheimer loss = 2.5000000000e-01",
                "Lowest-frequency ST loss = 2.3000000000e-01",
                "Low-frequency ST regularization loss = 4.0000000000e-02",
                "dpsi regularization loss = 0.0000000000e+00",
                "Lowest ST frequency (Ha) = 6.8706555678e-02",
                "Initial lowest-frequency ST loss = 2.4738400000e-01",
                "Final lowest-frequency ST loss = 2.4730000000e-01",
                "Low-frequency guard tolerance = 0.0000000000e+00",
                "Low-frequency guard weight = 1.0000000000e+01",
            )
            positions = [text.index(line) for line in expected_lines]
            self.assertEqual(positions, sorted(positions))

    def test_projected_pi_metadata_has_separate_explicit_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "projected_pi.dat"
            write_C(
                output,
                self.c,
                PROJECTED_PI_COMPONENTS["total"],
                loss_components=PROJECTED_PI_COMPONENTS,
                mode="pi_dpsi_joint",
                diagnostics=PROJECTED_PI_DIAGNOSTICS,
            )
            text = output.read_text()

            expected_lines = (
                "Mode = pi_dpsi_joint",
                "Projected Pi loss = 2.1400000000e-01",
                "Total loss = 4.1400000000e-01",
                "Lowest projected Pi frequency (Ha) = 6.8700000000e-02",
                "Lowest-frequency projected Pi loss = 1.9000000000e-01",
                "Maximum projected Pi overlap condition = 5.5620000000e+03",
                "Projected Pi rank tolerance = 1.0000000000e-12",
            )
            positions = [text.index(line) for line in expected_lines]
            self.assertEqual(positions, sorted(positions))
            self.assertNotIn("Sternheimer loss =", text)

            parsed, indices = read_C_init(
                output, {"H": info(Nl=1, Ne=2, Nu=[2])}
            )
            torch.testing.assert_close(parsed["H"][0], self.c["H"][0])
            self.assertEqual(indices, {("H", 0, 0), ("H", 0, 1)})

    def test_rejects_partial_or_mismatched_guarded_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad_guarded.dat"
            partial_components = dict(
                COMPONENTS, sternheimer_lowest_frequency=0.23
            )
            with self.assertRaisesRegex(ValueError, "guarded loss components"):
                write_C(
                    output,
                    self.c,
                    0.6,
                    loss_components=partial_components,
                    mode="st_only",
                )

            partial_diagnostics = dict(GUARDED_DIAGNOSTICS)
            partial_diagnostics.pop("final_lowest_st_loss")
            with self.assertRaisesRegex(ValueError, "guarded loss diagnostics"):
                write_C(
                    output,
                    self.c,
                    0.6,
                    loss_components=GUARDED_COMPONENTS,
                    mode="st_only",
                    diagnostics=partial_diagnostics,
                )

            with self.assertRaisesRegex(ValueError, "guarded diagnostics"):
                write_C(
                    output,
                    self.c,
                    0.6,
                    loss_components=COMPONENTS,
                    mode="st_only",
                    diagnostics=GUARDED_DIAGNOSTICS,
                )

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
    def test_main_measures_zero_weight_radial_control_without_changing_total(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            st_path = path / "sternheimer.dat"
            write_sternheimer(st_path)
            write_initial_coefficient(path / "C_init.dat")
            value = input_value(
                sternheimer_marker=[str(st_path)], loss=True
            )
            value["file_list"] = {"sternheimer": [str(st_path)]}
            value["weight"] = {}
            value["loss"].update(
                {
                    "radial_tail_weight": 0.0,
                    "radial_tail_radius": 3.0,
                    "radial_tail_condition_limit": 1.0e10,
                }
            )
            (path / "INPUT").write_text(json.dumps(value))

            with working_directory(path), mock.patch.object(
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

            metadata = {}
            for line in (path / "ORBITAL_RESULTS.txt").read_text().splitlines():
                if " = " in line:
                    label, raw = line.split(" = ", 1)
                    try:
                        metadata[label] = float(raw)
                    except ValueError:
                        pass
            self.assertGreater(metadata["Radial tail fraction"], 0.0)
            self.assertEqual(
                metadata["Radial locality regularization loss"], 0.0
            )
            self.assertEqual(
                metadata["Sternheimer loss"], metadata["Total loss"]
            )

    def test_main_builds_and_records_positive_radial_locality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            st_path = path / "sternheimer.dat"
            write_sternheimer(st_path)
            write_initial_coefficient(path / "C_init.dat")
            value = input_value(
                sternheimer_marker=[str(st_path)], loss=True
            )
            value["file_list"] = {"sternheimer": [str(st_path)]}
            value["weight"] = {}
            value["loss"].update(
                {
                    "radial_tail_weight": 0.5,
                    "radial_tail_radius": 3.0,
                    "radial_tail_condition_limit": 1.0e10,
                }
            )
            (path / "INPUT").write_text(json.dumps(value))

            with working_directory(path), mock.patch.object(
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

            header = (path / "Spillage.dat").read_text().splitlines()[0].split()
            self.assertIn("radial_tail", header)
            self.assertIn("regularization_locality", header)
            self.assertIn("max_locality_condition", header)
            result_text = (path / "ORBITAL_RESULTS.txt").read_text()
            self.assertIn("Radial tail fraction =", result_text)
            self.assertIn(
                "Radial locality regularization loss =", result_text
            )

    def test_st_only_main_needs_no_origin_or_linear_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            st_path = path / "sternheimer.dat"
            write_sternheimer(st_path)
            write_initial_coefficient(path / "C_init.dat")
            value = input_value(
                sternheimer_marker=[str(st_path)], loss=True
            )
            value["file_list"] = {"sternheimer": [str(st_path)]}
            value["weight"] = {}
            (path / "INPUT").write_text(json.dumps(value))

            with working_directory(path), mock.patch.object(
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

            result_text = (path / "ORBITAL_RESULTS.txt").read_text()
            self.assertIn("Mode = st_only", result_text)
            self.assertIn("DFT origin loss = 0.0000000000e+00", result_text)
            self.assertIn("DFT dpsi loss = 0.0000000000e+00", result_text)

    def test_main_persists_complete_low_frequency_guard_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            st_path = path / "sternheimer.dat"
            write_sternheimer(st_path)
            write_initial_coefficient(path / "C_init.dat")
            value = input_value(
                sternheimer_marker=[str(st_path)], loss=True
            )
            value["file_list"] = {"sternheimer": [str(st_path)]}
            value["weight"] = {}
            value["loss"].update(
                {
                    "low_frequency_guard_weight": 10.0,
                    "low_frequency_guard_tolerance": 0.0,
                }
            )
            (path / "INPUT").write_text(json.dumps(value))

            with working_directory(path), mock.patch.object(
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

            header = (path / "Spillage.dat").read_text().splitlines()[0].split()
            self.assertIn("sternheimer_lowest_frequency", header)
            self.assertIn("regularization_low_frequency", header)

            metadata = {}
            for line in (path / "ORBITAL_RESULTS.txt").read_text().splitlines():
                if " = " in line:
                    label, raw = line.split(" = ", 1)
                    try:
                        metadata[label] = float(raw)
                    except ValueError:
                        pass
            self.assertEqual(metadata["Lowest ST frequency (Ha)"], 0.5)
            self.assertEqual(metadata["Low-frequency guard weight"], 10.0)
            self.assertEqual(metadata["Low-frequency guard tolerance"], 0.0)
            self.assertLessEqual(
                metadata["Final lowest-frequency ST loss"],
                metadata["Initial lowest-frequency ST loss"] * (1.0 + 1.0e-12),
            )

    def test_constrained_main_still_requires_legacy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            st_path = path / "sternheimer.dat"
            write_sternheimer(st_path)
            write_initial_coefficient(path / "C_init.dat")
            value = input_value(
                sternheimer_marker=[str(st_path)], loss=True
            )
            value["file_list"] = {"sternheimer": [str(st_path)]}
            value["weight"] = {}
            value["loss"]["mode"] = "st_constrained"
            (path / "INPUT").write_text(json.dumps(value))

            with working_directory(path):
                with self.assertRaisesRegex(
                    ValueError, "st_constrained and st_dpsi_joint require origin"
                ):
                    siab_main.main()

    def test_joint_main_requires_linear_dpsi_data(self):
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
            value["loss"]["mode"] = "st_dpsi_joint"
            (path / "INPUT").write_text(json.dumps(value))

            with working_directory(path):
                with self.assertRaisesRegex(
                    ValueError, "st_dpsi_joint requires linear dpsi data"
                ):
                    siab_main.main()

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
