import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from common import info


from IO.func_C import read_C_init
from IO.read_json import read_json
from attribute_dict import AttributeDict
import main as siab_main
import orbital as siab_orbital
from opt_orbital_converge import Opt_Orbital_Converge
from sternheimer_data import PrimitiveBlock, SternheimerData
from sternheimer_spillage import OrbitalColumn, SternheimerSpillage


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example_H_sternheimer"
LEGACY_PRODUCER = EXAMPLE / "legacy_dpsi_producer"
REAL_H_TZDP = (
    ROOT.parent
    / "Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/ORBITAL_RESULTS.txt"
)
SEED = 20260718
LOSS_DEFAULTS = {
    "epsilon": 1e-14,
    "condition_limit": 1e12,
    "tau_dft": 0.05,
    "tau_dpsi": 0.10,
    "constraint_penalty_dft": 10.0,
    "constraint_penalty_dpsi": 10.0,
    "joint_dpsi_weight": 1.0,
}
DZP_FREEZE_SPECS = [
    {"element": "H", "l": 0, "zeta": 1},
    {"element": "H", "l": 0, "zeta": 2},
    {"element": "H", "l": 1, "zeta": 1},
]


class AttributeDictTest(unittest.TestCase):
    def test_nested_key_and_attribute_access_share_storage(self):
        value = AttributeDict()
        value["H"].Nu = [3, 2]
        value.H.Ne = 25

        self.assertEqual(value.H.Nu, [3, 2])
        self.assertEqual(value["H"]["Ne"], 25)


def _minimal_input():
    return {
        "file_list": {"origin": ["synthetic"]},
        "element": {"Nt_all": ["H"], "Nu": {"H": [3]}},
        "weight": {"stru": [1.0]},
    }


def _read_real_h_tzdp():
    info_element = {"H": info(Nl=2, Ne=25, Nu=[3, 2])}
    info_radial = {
        "Rcut": {"H": 8},
        "dr": {"H": 0.01},
        "Ecut": {"H": 100},
        "smearing_sigma": {"H": 0.0},
    }
    c, _ = read_C_init(REAL_H_TZDP, info_element)
    e = siab_orbital.set_E(info_element, info_radial["Rcut"])
    return c, info_element, info_radial, e


def _make_real_h_sternheimer_data():
    q = torch.zeros((2, 25), dtype=torch.complex128)
    q[0, :4] = torch.tensor(
        [0.20, 0.10, -0.05, 0.15], dtype=torch.complex128
    )
    q[1, :4] = torch.tensor(
        [0.05, -0.15, 0.20, 0.10], dtype=torch.complex128
    )
    return SternheimerData(
        format_version=1,
        grid_volume_bohr3=1.0,
        blocks=(PrimitiveBlock("H", 0, 0, 0, 25, 0),),
        occupied_state=torch.tensor([0, 0], dtype=torch.int64),
        auxiliary_channel=torch.tensor([0, 1], dtype=torch.int64),
        frequency_ha=torch.tensor([0.1, 0.4], dtype=torch.float64),
        occupation=torch.tensor([2.0, 2.0], dtype=torch.float64),
        frequency_weight=torch.tensor([0.4, 0.6], dtype=torch.float64),
        norm=torch.tensor([1.0, 1.0], dtype=torch.float64),
        q=q,
        overlap=torch.eye(25, dtype=torch.complex128),
        provenance=_provenance(),
    )


def _write_input(value):
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    with handle:
        json.dump(value, handle)
    return Path(handle.name)


def _provenance():
    return {
        "abacus_commit": "synthetic-smoke",
        "auxiliary_basis_sha256": "synthetic-smoke",
        "cell_bohr": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "ecut_ry": 1.0,
        "kernel": "none",
        "orbital_sha256": "synthetic-smoke",
        "pseudopotential_sha256": "synthetic-smoke",
        "spin_convention": "unit_test",
    }


def _make_sternheimer_data():
    blocks = [PrimitiveBlock("H", 0, 0, 0, 4, 0)]
    blocks.extend(
        PrimitiveBlock("H", 0, 1, m, 4, 4 + 4 * (m + 1))
        for m in (-1, 0, 1)
    )
    q = torch.zeros((7, 16), dtype=torch.complex128)
    q[0, 3] = 1.0
    row = 1
    for block in blocks[1:]:
        q[row, block.offset + 2] = 1.0
        q[row + 1, block.offset + 3] = 1.0
        row += 2
    return SternheimerData(
        format_version=1,
        grid_volume_bohr3=1.0,
        blocks=tuple(blocks),
        occupied_state=torch.zeros(7, dtype=torch.int64),
        auxiliary_channel=torch.arange(7, dtype=torch.int64),
        frequency_ha=torch.linspace(0.1, 0.7, 7, dtype=torch.float64),
        occupation=torch.full((7,), 2.0, dtype=torch.float64),
        frequency_weight=torch.ones(7, dtype=torch.float64),
        norm=torch.ones(7, dtype=torch.float64),
        q=q,
        overlap=torch.eye(16, dtype=torch.complex128),
        provenance=_provenance(),
    )


def _make_sternheimer_data_with_d():
    blocks = list(_make_sternheimer_data().blocks)
    offset = 16
    for m in range(-2, 3):
        blocks.append(PrimitiveBlock("H", 0, 2, m, 4, offset))
        offset += 4
    q = torch.zeros((1, offset), dtype=torch.complex128)
    q[0, 18] = 1.0
    return SternheimerData(
        format_version=1,
        grid_volume_bohr3=1.0,
        blocks=tuple(blocks),
        occupied_state=torch.zeros(1, dtype=torch.int64),
        auxiliary_channel=torch.zeros(1, dtype=torch.int64),
        frequency_ha=torch.tensor([0.1], dtype=torch.float64),
        occupation=torch.tensor([2.0], dtype=torch.float64),
        frequency_weight=torch.ones(1, dtype=torch.float64),
        norm=torch.ones(1, dtype=torch.float64),
        q=q,
        overlap=torch.eye(offset, dtype=torch.complex128),
        provenance=_provenance(),
    )


def _make_legacy_data():
    info_stru = [
        info(Na={"H": 1}, Nb_true=2, weight=torch.tensor([0.5, 0.5]))
    ]
    info_element = {"H": info(Nl=2, Ne=4, Nu=[3, 2])}
    q_origin = {
        "H": [
            torch.tensor(
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=torch.complex128,
            ),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.complex128,
            ),
        ]
    }
    overlap_by_l = [[None, None], [None, None]]
    zero_overlap_by_l = [[None, None], [None, None]]
    for l1 in range(2):
        for l2 in range(2):
            nm1 = 2 * l1 + 1
            nm2 = 2 * l2 + 1
            shape = (1, nm1, 4, 1, nm2, 4)
            value = torch.zeros(shape, dtype=torch.complex128)
            if l1 == l2:
                for m in range(nm1):
                    for primitive in range(4):
                        value[0, m, primitive, 0, m, primitive] = 1.0
            overlap_by_l[l1][l2] = value
            zero_overlap_by_l[l1][l2] = torch.zeros(
                shape, dtype=torch.complex128
            )
    overlap = {("H", "H"): overlap_by_l}
    zero_overlap = {("H", "H"): zero_overlap_by_l}
    q_linear = {"H": [value.clone() for value in q_origin["H"]]}
    return {
        "info_stru": info_stru,
        "info_element": info_element,
        "q_origin": [q_origin],
        "s_origin": [overlap],
        "v_origin": [torch.tensor([1.0, 1.0], dtype=torch.float64)],
        "q_linear": [[q_linear]],
        "s_linear": [[zero_overlap]],
        "v_linear": [[torch.tensor([1.99, 1.99], dtype=torch.float64)]],
    }


def _initial_c():
    return {
        "H": [
            torch.tensor(
                [
                    [1.00, 0.00, 0.00],
                    [0.00, 1.00, 0.00],
                    [0.00, 0.00, 1.00],
                    [0.00, 0.15, 0.00],
                ],
                dtype=torch.float64,
                requires_grad=True,
            ),
            torch.tensor(
                [
                    [1.00, 0.00],
                    [0.00, 1.00],
                    [0.15, 0.00],
                    [0.00, 0.15],
                ],
                dtype=torch.float64,
                requires_grad=True,
            ),
        ]
    }


def _run_smoke(mode):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    c = _initial_c()
    c0 = tuple(value.detach().clone() for value in c["H"])
    legacy = _make_legacy_data()
    stage = {
        "optimizer": "Adam",
        "kwargs": {"lr": 0.003},
        "cal_T": False,
        "norm": "one",
        "max_steps": 10,
        "loss": {"mode": mode, **LOSS_DEFAULTS},
    }
    converge = Opt_Orbital_Converge()
    converge.set_info(
        {"origin": ["synthetic"], "linear": [["synthetic-linear"]]},
        [stage],
        legacy["info_stru"],
        {
            "init_from_file": True,
            "freeze_orbitals": [{"element": "H", "l": 0, "zeta": 1}],
        },
        {"same_band": True},
    )
    converge.set_info_element(legacy["info_element"])
    converge.set_QSVI(
        legacy["q_origin"], legacy["s_origin"], legacy["v_origin"]
    )
    converge.set_QSVI_linear(
        legacy["q_linear"], legacy["s_linear"], legacy["v_linear"]
    )
    evaluator = SternheimerSpillage(
        _make_sternheimer_data(),
        c,
        [OrbitalColumn("H", 0, 0, 0, 1)],
    )
    converge.set_sternheimer_spillage(evaluator)

    initial_st = evaluator.evaluate(c).loss.item()
    spillage = converge._make_spillage(stage)
    baseline = spillage.cal_components(c)
    detail = io.StringIO()
    result = converge.cal_converge(c, (io.StringIO(), detail))
    final_c = tuple(
        value.detach().clone() for value in result["C"]["H"]
    )
    final_components = spillage.cal_components(result["C"])
    final_st = evaluator.evaluate(result["C"]).loss.item()
    lines = detail.getvalue().splitlines()
    header = lines[0].split("\t")
    trajectory = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(header) or not fields[0].lstrip("-").isdigit():
            continue
        row = {}
        for name, value in zip(header, fields):
            if name == "accepted":
                row[name] = value == "true"
            elif name in ("istep_big", "istep_small", "istep_all"):
                row[name] = int(value)
            else:
                row[name] = float(value)
        trajectory.append(row)
    return {
        "initial_c": c0,
        "final_c": final_c,
        "initial_1s": c0[0][:, 0].clone(),
        "final_1s": final_c[0][:, 0].clone(),
        "initial_st": initial_st,
        "final_st": final_st,
        "baseline_dft": baseline["dft_origin"].item(),
        "baseline_dpsi": baseline["dft_dpsi"].item(),
        "final_dft": final_components["dft_origin"].item(),
        "final_dpsi": final_components["dft_dpsi"].item(),
        "trajectory": trajectory,
    }


class SeedConfigurationTest(unittest.TestCase):
    def test_read_json_propagates_valid_seed_without_changing_return_count(self):
        value = _minimal_input()
        value["seed"] = SEED
        path = _write_input(value)
        self.addCleanup(path.unlink, missing_ok=True)

        result = read_json(path)

        self.assertEqual(len(result), 7)
        self.assertEqual(result[4]["seed"], SEED)

    def test_read_json_rejects_invalid_seed_values(self):
        invalid = (True, False, -1, 2**32, 1.0, "20260718", None)
        for seed in invalid:
            with self.subTest(seed=seed):
                value = _minimal_input()
                value["seed"] = seed
                path = _write_input(value)
                self.addCleanup(path.unlink, missing_ok=True)
                with self.assertRaisesRegex((TypeError, ValueError), "seed"):
                    read_json(path)

    def test_read_json_rejects_nested_seed_with_or_without_top_level_seed(self):
        for nested_seed, top_level in (
            (True, False),
            (SEED, False),
            (True, True),
            (SEED, True),
        ):
            with self.subTest(nested_seed=nested_seed, top_level=top_level):
                value = _minimal_input()
                value["C_init_info"] = {
                    "init_from_file": False,
                    "seed": nested_seed,
                }
                if top_level:
                    value["seed"] = SEED
                path = _write_input(value)
                self.addCleanup(path.unlink, missing_ok=True)
                with self.assertRaisesRegex(
                    ValueError, "seed.*top-level|top-level.*seed"
                ):
                    read_json(path)

    def test_main_sets_and_prints_numpy_and_torch_seed(self):
        output = io.StringIO()
        with mock.patch.object(np.random, "seed") as numpy_seed, mock.patch.object(
            torch, "manual_seed"
        ) as torch_seed, mock.patch("sys.stdout", output):
            actual = siab_main._set_random_seed({"seed": SEED})

        self.assertEqual(actual, SEED)
        numpy_seed.assert_called_once_with(SEED)
        torch_seed.assert_called_once_with(SEED)
        self.assertIn(f"numpy seed: {SEED}", output.getvalue())
        self.assertIn(f"torch seed: {SEED}", output.getvalue())

    def test_main_preserves_legacy_time_based_seed(self):
        with mock.patch.object(siab_main.time, "time", return_value=1234.56789):
            with mock.patch.object(np.random, "seed") as numpy_seed, mock.patch.object(
                torch, "manual_seed"
            ) as torch_seed:
                actual = siab_main._set_random_seed({})

        expected = int(1000 * 1234.56789) % (2**32)
        self.assertEqual(actual, expected)
        numpy_seed.assert_called_once_with(expected)
        torch_seed.assert_called_once_with(expected)

    def test_main_rejects_invalid_programmatic_seed_values(self):
        for seed in (True, False, -1, 2**32, 1.0, "20260718", None):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex((TypeError, ValueError), "seed"):
                    siab_main._set_random_seed({"seed": seed})

    def test_main_seed_resets_real_numpy_and_torch_sequences(self):
        with mock.patch("sys.stdout", io.StringIO()):
            siab_main._set_random_seed({"seed": SEED})
            numpy_first = np.random.random(5)
            torch_first = torch.rand(5)
            siab_main._set_random_seed({"seed": SEED})
            numpy_second = np.random.random(5)
            torch_second = torch.rand(5)

        self.assertTrue(np.array_equal(numpy_first, numpy_second))
        self.assertTrue(torch.equal(torch_first, torch_second))


class ExplicitFreezeInitializationTest(unittest.TestCase):
    freeze_specs = [{"element": "H", "l": 0, "zeta": 1}]

    def test_real_h_1s_is_restored_bitwise_while_other_columns_normalize(self):
        c, info_element, info_radial, e = _read_real_h_tzdp()
        raw_fixed = c["H"][0][:, 0].detach().clone()
        raw_unfrozen = {
            (l, zeta): c["H"][l][:, zeta].detach().clone()
            for l in range(2)
            for zeta in range(c["H"][l].shape[1])
            if (l, zeta) != (0, 0)
        }

        siab_main._normalize_initial_coefficients(
            c, info_element, info_radial, e, self.freeze_specs
        )

        self.assertTrue(torch.equal(c["H"][0][:, 0], raw_fixed))
        self.assertTrue(
            any(
                not torch.equal(c["H"][l][:, zeta], raw)
                for (l, zeta), raw in raw_unfrozen.items()
            )
        )
        normalized_orbitals = siab_orbital.generate_orbital(
            info_element, info_radial, c, e
        )
        for (l, zeta) in raw_unfrozen:
            norm = np.sqrt(
                siab_orbital.inner_product(
                    normalized_orbitals["H"][l][zeta],
                    normalized_orbitals["H"][l][zeta],
                    info_radial["dr"]["H"],
                )
            )
            self.assertAlmostEqual(norm, 1.0, places=13)

    def test_legacy_initialization_matches_the_old_normalization_bitwise(self):
        actual, info_element, info_radial, e = _read_real_h_tzdp()
        expected, _, _, _ = _read_real_h_tzdp()
        siab_orbital.normalize(
            siab_orbital.generate_orbital(
                info_element, info_radial, expected, e
            ),
            info_radial["dr"],
            expected,
            flag_norm_C=True,
        )

        siab_main._normalize_initial_coefficients(
            actual, info_element, info_radial, e, None
        )

        for l in range(2):
            self.assertTrue(torch.equal(actual["H"][l], expected["H"][l]))

    def test_real_evaluator_and_converge_start_from_restored_raw_1s(self):
        c, info_element, info_radial, e = _read_real_h_tzdp()
        raw_fixed = c["H"][0][:, 0].detach().clone()
        siab_main._normalize_initial_coefficients(
            c, info_element, info_radial, e, self.freeze_specs
        )
        c_s = {"H": [c["H"][0]]}
        data = _make_real_h_sternheimer_data()
        evaluator = SternheimerSpillage(
            data,
            c_s,
            [OrbitalColumn("H", 0, 0, 0, 1)],
        )

        self.assertTrue(
            torch.equal(evaluator._a0[:, 0].real, raw_fixed)
        )
        info_s = {"H": info(Nl=1, Ne=25, Nu=[3])}
        info_stru = [
            info(Na={"H": 1}, Nb_true=1, weight=torch.tensor([1.0]))
        ]
        q = {"H": [torch.linspace(0.01, 0.25, 25).reshape(1, 25).to(torch.complex128)]}
        s = {
            ("H", "H"): [[
                torch.eye(25, dtype=torch.complex128).reshape(
                    1, 1, 25, 1, 1, 25
                )
            ]]
        }
        stage = {
            "optimizer": "Adam",
            "kwargs": {"lr": 0.003},
            "cal_T": False,
            "norm": "one",
            "max_steps": 0,
            "loss": {"mode": "st_only", **LOSS_DEFAULTS},
        }
        converge = Opt_Orbital_Converge()
        converge.set_info(
            {"origin": ["synthetic"]},
            [stage],
            info_stru,
            {"init_from_file": True, "freeze_orbitals": self.freeze_specs},
            {"same_band": True},
        )
        converge.set_info_element(info_s)
        converge.set_QSVI(
            [q], [s], [torch.tensor([1.0], dtype=torch.float64)]
        )
        converge.set_sternheimer_spillage(evaluator)

        result = converge.cal_converge(
            c_s, (io.StringIO(), io.StringIO())
        )

        self.assertTrue(torch.equal(result["C"]["H"][0][:, 0], raw_fixed))


class ExampleInputTest(unittest.TestCase):
    def test_inputs_match_exact_campaign_contract(self):
        st_only = json.loads((EXAMPLE / "INPUT.st_only").read_text())
        expanded = json.loads((EXAMPLE / "INPUT.st_response_4s3p").read_text())
        constrained = json.loads((EXAMPLE / "INPUT.st_constrained").read_text())
        joint = json.loads((EXAMPLE / "INPUT.st_dpsi_joint").read_text())
        self.assertEqual(st_only["file_list"].keys(), {"sternheimer"})
        self.assertNotIn("origin", st_only["file_list"])
        self.assertNotIn("linear", st_only["file_list"])
        self.assertIn("origin", constrained["file_list"])
        self.assertIn("linear", constrained["file_list"])
        self.assertEqual(st_only["loss"]["mode"], "st_only")
        self.assertEqual(constrained["loss"]["mode"], "st_constrained")

        self.assertEqual(joint["loss"]["mode"], "st_dpsi_joint")
        self.assertEqual(joint["loss"]["joint_dpsi_weight"], 0.1)
        self.assertEqual(st_only["seed"], SEED)
        self.assertEqual(st_only["element"]["Nu"], {"H": [3, 2]})
        self.assertEqual(expanded["element"]["Nu"], {"H": [4, 3]})
        self.assertEqual(expanded["freeze_orbitals"], DZP_FREEZE_SPECS)
        self.assertEqual(expanded["loss"]["mode"], "st_only")
        self.assertEqual(
            expanded["C_init_info"]["C_init_file"],
            st_only["C_init_info"]["C_init_file"],
        )
        self.assertEqual(st_only["radial"]["Rcut"], 8)
        self.assertEqual(st_only["radial"]["dr"], 0.01)
        self.assertEqual(st_only["radial"]["Ecut"], 100)
        self.assertEqual(st_only["radial"]["smearing_sigma"], 0.1)
        self.assertEqual(
            st_only["optimize"],
            [
                {
                    "optimizer": "Adam",
                    "kwargs": {"lr": 0.003},
                    "cal_T": False,
                    "norm": "element",
                    "max_steps": 3000,
                }
            ],
        )
        self.assertEqual(
            st_only["freeze_orbitals"],
            DZP_FREEZE_SPECS,
        )
        self.assertEqual(constrained["freeze_orbitals"], DZP_FREEZE_SPECS)
        self.assertEqual(joint["freeze_orbitals"], DZP_FREEZE_SPECS)
        self.assertEqual(joint["file_list"], constrained["file_list"])
        self.assertEqual(joint["element"], constrained["element"])
        self.assertEqual(
            st_only["C_init_info"],
            {
                "init_from_file": True,
                "C_init_file": (
                    "../../Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/"
                    "ORBITAL_RESULTS.txt"
                ),
                "opt_C_read": False,
            },
        )
        self.assertEqual(
            st_only["file_list"],
            {
                "sternheimer": [
                    "data/OUT.H-atom-ST/sternheimer_matrix.dat"
                ],
            },
        )
        self.assertEqual(
            st_only["loss"], {"mode": "st_only", **LOSS_DEFAULTS}
        )
        self.assertEqual(constrained["loss"]["mode"], "st_constrained")

    def test_legacy_dpsi_producer_matches_original_h3_contract(self):
        cases = {
            "0.7": ("0.606221", "0.350000"),
            "0.9": ("0.779427", "0.450000"),
            "1.3": ("1.125839", "0.650000"),
        }
        for bond, (triangle_y, triangle_z) in cases.items():
            with self.subTest(bond=bond):
                case = LEGACY_PRODUCER / "cases" / f"H-STRU2-8-{bond}"
                input_text = (case / "INPUT").read_text()
                stru_text = (case / "STRU").read_text()
                inputw_text = (case / "INPUTw").read_text()

                self.assertIn(f"suffix H-STRU2-8-{bond}", input_text)
                self.assertIn("nspin 1", input_text)
                self.assertIn("nbands 10", input_text)
                self.assertIn("ecutwfc 100", input_text)
                self.assertIn("bessel_nao_ecut 100", input_text)
                self.assertIn("bessel_nao_rcut 8", input_text)
                self.assertIn("bessel_nao_smooth 1", input_text)
                self.assertIn("bessel_nao_sigma 0.1", input_text)
                self.assertNotIn("wannier_card", input_text)
                self.assertIn("out_spillage 2", input_text)
                self.assertIn(
                    f"spillage_outdir OUT.H-STRU2-8-{bond}", input_text
                )
                self.assertIn("Cartesian_angstrom", stru_text)
                self.assertIn(f"0.000000 0.000000 {bond}", stru_text)
                self.assertIn(
                    f"0.000000 {triangle_y} {triangle_z}", stru_text
                )
                self.assertIn("out_spillage 2", inputw_text)
                self.assertIn(
                    f"spillage_outdir OUT.H-STRU2-8-{bond}", inputw_text
                )

        self.assertTrue((LEGACY_PRODUCER / "KPT").is_file())
        run_script = (LEGACY_PRODUCER / "run_abacus.slurm").read_text()
        self.assertIn('script_dir=${SLURM_SUBMIT_DIR:?}', run_script)
        self.assertNotIn("BASH_SOURCE", run_script)


class AppendedResponseShellTest(unittest.TestCase):
    @staticmethod
    def _expanded_h_info():
        return {"H": info(Nl=2, Ne=25, Nu=[4, 3])}

    def _read_expanded(self, seed):
        np.random.seed(seed)
        return read_C_init(
            REAL_H_TZDP,
            self._expanded_h_info(),
            return_metadata=True,
        )

    def test_appended_same_l_shells_are_reported_and_seeded(self):
        c1, metadata1 = self._read_expanded(SEED)
        c2, metadata2 = self._read_expanded(SEED)
        c3, metadata3 = self._read_expanded(SEED + 1)
        expected_loaded = frozenset(
            {
                ("H", 0, 0),
                ("H", 0, 1),
                ("H", 0, 2),
                ("H", 1, 0),
                ("H", 1, 1),
            }
        )
        expected_appended = frozenset({("H", 0, 3), ("H", 1, 2)})

        self.assertEqual(metadata1.loaded_indices, expected_loaded)
        self.assertEqual(metadata1.appended_indices, expected_appended)
        self.assertEqual(metadata1, metadata2)
        self.assertEqual(metadata1, metadata3)
        for element, l, zeta in expected_appended:
            value1 = c1[element][l][:, zeta]
            value2 = c2[element][l][:, zeta]
            value3 = c3[element][l][:, zeta]
            self.assertTrue(torch.all(torch.isfinite(value1)))
            self.assertGreater(torch.linalg.vector_norm(value1).item(), 0.0)
            self.assertTrue(torch.equal(value1, value2))
            self.assertFalse(torch.equal(value1, value3))

    def test_new_angular_channel_requires_matching_target_blocks(self):
        requested = info(Nt_all=["H"], Nu={"H": [3, 2, 1]})
        with self.assertRaisesRegex(ValueError, r"H/2.*\[\]"):
            siab_main._sternheimer_info_element(
                _make_sternheimer_data(), requested
            )

        actual = siab_main._sternheimer_info_element(
            _make_sternheimer_data_with_d(), requested
        )

        self.assertEqual(actual["H"].Nl, 3)
        self.assertEqual(actual["H"].Nu, [3, 2, 1])
        self.assertEqual(actual["H"].Ne, 4)

        c = _initial_c()
        c["H"].append(
            torch.tensor(
                [[0.0], [0.0], [1.0], [0.0]],
                dtype=torch.float64,
                requires_grad=True,
            )
        )
        fixed_columns = [
            OrbitalColumn("H", 0, 0, 0, 1),
            OrbitalColumn("H", 0, 0, 0, 2),
            *(OrbitalColumn("H", 0, 1, m, 1) for m in (-1, 0, 1)),
        ]
        result = SternheimerSpillage(
            _make_sternheimer_data_with_d(), c, fixed_columns
        ).evaluate(c)
        self.assertTrue(torch.isfinite(result.loss))

    def test_initialization_rejects_duplicate_and_out_of_range_columns(self):
        info_element = {"H": info(Nl=1, Ne=2, Nu=[2])}
        cases = (
            (
                "duplicate",
                (("H", 0, 1, (0.1, 0.2)), ("H", 0, 1, (0.3, 0.4))),
                "duplicate coefficient column",
            ),
            (
                "out-of-range",
                (("H", 1, 1, (0.1, 0.2)),),
                "outside requested Nu",
            ),
        )
        for name, columns, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "ORBITAL_RESULTS.txt"
                lines = ["<Coefficient>", " 2 Total number of radial orbitals."]
                for element, l, zeta, values in columns:
                    lines.extend(
                        (
                            " Type L Zeta-Orbital",
                            f" {element} {l} {zeta}",
                            *(f" {value}" for value in values),
                        )
                    )
                lines.append("</Coefficient>")
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, message):
                    read_C_init(path, info_element, return_metadata=True)


class DeterministicOptimizationSmokeTest(unittest.TestCase):
    def test_dzp_core_is_fixed_while_3s_and_2p_reduce_st_loss(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        c = _initial_c()
        with torch.no_grad():
            c["H"][0][3, 2] = 0.05
        initial = tuple(value.detach().clone() for value in c["H"])
        data = _make_sternheimer_data()
        fixed_columns = [OrbitalColumn("H", 0, 0, 0, 1)]
        fixed_columns.append(OrbitalColumn("H", 0, 0, 0, 2))
        fixed_columns.extend(
            OrbitalColumn("H", 0, 1, m, 1) for m in (-1, 0, 1)
        )
        evaluator = SternheimerSpillage(data, c, fixed_columns)
        initial_st = evaluator.evaluate(c).loss.item()
        stage = {
            "optimizer": "Adam",
            "kwargs": {"lr": 0.003},
            "cal_T": False,
            "norm": "one",
            "max_steps": 30,
            "loss": {"mode": "st_only", **LOSS_DEFAULTS},
        }
        converge = Opt_Orbital_Converge()
        converge.set_info(
            {"sternheimer": ["synthetic"]},
            [stage],
            [],
            {
                "init_from_file": True,
                "freeze_orbitals": DZP_FREEZE_SPECS,
            },
            {"same_band": True},
        )
        converge.set_info_element(
            {"H": info(Nl=2, Ne=4, Nu=[3, 2])}
        )
        converge.set_sternheimer_spillage(evaluator)

        result = converge.cal_converge(
            c, (io.StringIO(), io.StringIO())
        )["C"]["H"]
        final_st = evaluator.evaluate({"H": result}).loss.item()

        self.assertTrue(torch.equal(initial[0][:, :2], result[0][:, :2]))
        self.assertTrue(torch.equal(initial[1][:, :1], result[1][:, :1]))
        self.assertFalse(torch.equal(initial[0][:, 2], result[0][:, 2]))
        self.assertFalse(torch.equal(initial[1][:, 1], result[1][:, 1]))
        self.assertLess(final_st, initial_st)

    def test_both_modes_are_repeatable_and_constraints_change_the_trajectory(self):
        initial = _initial_c()
        self.assertEqual(initial["H"][0].shape, (4, 3))
        self.assertEqual(initial["H"][1].shape, (4, 2))
        self.assertEqual(
            tuple(block.key for block in _make_sternheimer_data().blocks),
            (
                ("H", 0, 0, 0),
                ("H", 0, 1, -1),
                ("H", 0, 1, 0),
                ("H", 0, 1, 1),
            ),
        )
        results = {}
        for mode in ("st_only", "st_constrained"):
            with self.subTest(mode=mode):
                run1 = _run_smoke(mode)
                run2 = _run_smoke(mode)
                for final1, final2 in zip(run1["final_c"], run2["final_c"]):
                    torch.testing.assert_close(
                        final1, final2, rtol=0.0, atol=0.0
                    )
                self.assertEqual(run1["trajectory"], run2["trajectory"])
                self.assertEqual(len(run1["trajectory"]), 11)
                self.assertLess(run1["final_st"], run1["initial_st"])
                self.assertTrue(
                    torch.equal(run1["initial_1s"], run1["final_1s"])
                )
                self.assertFalse(
                    torch.equal(
                        run1["initial_c"][0][:, 1:],
                        run1["final_c"][0][:, 1:],
                    )
                )
                self.assertFalse(
                    torch.equal(run1["initial_c"][1], run1["final_c"][1])
                )
                self.assertGreater(run1["baseline_dft"], 0.0)
                self.assertGreater(run1["baseline_dpsi"], 0.0)
                results[mode] = run1

        constrained = results["st_constrained"]
        self.assertTrue(
            any(
                row["constraint_dft"] > 0.0
                or row["constraint_dpsi"] > 0.0
                for row in constrained["trajectory"]
            )
        )
        self.assertLessEqual(
            constrained["final_dft"] / constrained["baseline_dft"],
            1.05 + 1e-12,
        )
        self.assertLessEqual(
            constrained["final_dpsi"] / constrained["baseline_dpsi"],
            1.10 + 1e-12,
        )
        st_only = results["st_only"]
        self.assertTrue(
            st_only["final_dft"] / st_only["baseline_dft"] > 1.05 + 1e-12
            or st_only["final_dpsi"] / st_only["baseline_dpsi"]
            > 1.10 + 1e-12
        )
        self.assertTrue(
            any(
                not torch.equal(st_value, constrained_value)
                for st_value, constrained_value in zip(
                    st_only["final_c"], constrained["final_c"]
                )
            )
        )
        self.assertNotEqual(
            [row["total"] for row in st_only["trajectory"]],
            [row["total"] for row in constrained["trajectory"]],
        )


if __name__ == "__main__":
    unittest.main()
