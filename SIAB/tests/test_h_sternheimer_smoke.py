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
GREEDY_RESPONSE = EXAMPLE / "greedy_response_selection"
LEGACY_PRODUCER = EXAMPLE / "legacy_dpsi_producer"
HELD_OUT_SOS = EXAMPLE / "held_out_h2_sos"
FIXED_DZP_TZDP_SOS = EXAMPLE / "fixed_dzp_tzdp_sos"
PROJECTED_PI_LOSS = EXAMPLE / "projected_pi_loss"
HELD_OUT_SOS_4S3P = EXAMPLE / "held_out_h2_sos_4s3p"
HELD_OUT_SOS_4S3P3D = EXAMPLE / "held_out_h2_sos_4s3p3d"
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


class ProjectedPiCampaignContractTest(unittest.TestCase):
    def test_optimizer_input_freezes_exact_physical_contract(self):
        value = json.loads(
            (PROJECTED_PI_LOSS / "INPUT.pi_dpsi_joint").read_text()
        )
        entries = value["file_list"]["sternheimer"]

        self.assertEqual(value["seed"], 20260718)
        self.assertEqual(value["element"]["Nu"], {"H": [3, 2, 0, 0, 0]})
        self.assertEqual(value["radial"]["Rcut"], 8)
        self.assertEqual(value["radial"]["Ecut"], 100)
        self.assertEqual(value["radial"]["dr"], 0.01)
        self.assertEqual(
            value["freeze_orbitals"],
            [
                {"element": "H", "l": 0, "zeta": 1},
                {"element": "H", "l": 0, "zeta": 2},
                {"element": "H", "l": 1, "zeta": 1},
            ],
        )
        self.assertEqual({entry["family"] for entry in entries}, {"H", "H2"})
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertEqual(entry["role"], "physical")
            self.assertTrue(entry["path"].endswith("sternheimer_matrix.dat"))
            self.assertTrue(
                entry["source_path"].endswith("STERNHEIMER_SIAB_SOURCE_V1.dat")
            )
            self.assertTrue(
                entry["zero_order_audit_path"].endswith("zero_order_identity.json")
            )
        self.assertEqual(len(value["file_list"]["origin"]), 3)
        self.assertEqual(len(value["file_list"]["linear"]), 1)
        self.assertEqual(len(value["file_list"]["linear"][0]), 3)
        self.assertIn(
            "fixed_dzp_joint_ORBITAL_RESULTS.txt",
            value["C_init_info"]["C_init_file"],
        )
        self.assertEqual(value["loss"]["mode"], "pi_dpsi_joint")
        self.assertEqual(value["loss"]["projected_pi_rank_tolerance"], 1.0e-12)
        self.assertEqual(value["loss"]["joint_dpsi_weight"], 1.0)
        self.assertEqual(value["loss"].get("radial_tail_weight", 0.0), 0.0)
        self.assertEqual(
            value["loss"].get("low_frequency_guard_weight", 0.0), 0.0
        )
        self.assertEqual(value["optimize"][0]["optimizer"], "Adam")
        self.assertEqual(value["optimize"][0]["kwargs"], {"lr": 0.003})
        self.assertEqual(value["optimize"][0]["max_steps"], 3000)

    def test_one_d_input_expands_only_the_response_space(self):
        value = json.loads(
            (PROJECTED_PI_LOSS / "INPUT.pi_dpsi_joint_3s2p1d").read_text()
        )

        self.assertEqual(value["seed"], 20260718)
        self.assertEqual(value["element"]["Nu"], {"H": [3, 2, 1, 0, 0]})
        self.assertEqual(value["freeze_orbitals"], DZP_FREEZE_SPECS)
        self.assertEqual(value["radial"], {
            "Rcut": 8,
            "dr": 0.01,
            "Ecut": 100,
            "smearing_sigma": 0.1,
        })
        self.assertEqual(
            value["C_init_info"]["C_init_file"],
            "../inputs/projected_pi_joint_3s2p_plus_smooth_1d_ORBITAL_RESULTS.txt",
        )
        self.assertEqual(value["loss"]["mode"], "pi_dpsi_joint")
        self.assertEqual(value["loss"]["projected_pi_rank_tolerance"], 1.0e-12)
        self.assertEqual(value["loss"]["joint_dpsi_weight"], 0.02)
        self.assertEqual(value["loss"].get("radial_tail_weight", 0.0), 0.0)
        self.assertEqual(
            value["loss"].get("low_frequency_guard_weight", 0.0), 0.0
        )
        for paths in (
            value["file_list"]["origin"],
            value["file_list"]["linear"][0],
        ):
            self.assertEqual(len(paths), 3)
            self.assertTrue(all("../inputs/l2/" in path for path in paths))
        self.assertEqual(
            value["file_list"]["sternheimer"],
            json.loads(
                (PROJECTED_PI_LOSS / "INPUT.pi_dpsi_joint").read_text()
            )["file_list"]["sternheimer"],
        )

    def test_two_d_input_appends_only_the_next_response_shell(self):
        value = json.loads(
            (PROJECTED_PI_LOSS / "INPUT.pi_dpsi_joint_3s2p2d").read_text()
        )

        self.assertEqual(value["seed"], 20260718)
        self.assertEqual(value["element"]["Nu"], {"H": [3, 2, 2, 0, 0]})
        self.assertEqual(value["freeze_orbitals"], DZP_FREEZE_SPECS)
        self.assertEqual(
            value["C_init_info"]["C_init_file"],
            "../inputs/projected_pi_joint_3s2p1d_w002_plus_smooth_2d_ORBITAL_RESULTS.txt",
        )
        self.assertEqual(value["loss"]["mode"], "pi_dpsi_joint")
        self.assertEqual(value["loss"]["projected_pi_rank_tolerance"], 1.0e-12)
        self.assertEqual(value["loss"]["joint_dpsi_weight"], 0.02)
        self.assertEqual(value["loss"].get("radial_tail_weight", 0.0), 0.0)
        self.assertEqual(
            value["loss"].get("low_frequency_guard_weight", 0.0), 0.0
        )
        self.assertEqual(
            value["file_list"]["sternheimer"],
            json.loads(
                (PROJECTED_PI_LOSS / "INPUT.pi_dpsi_joint_3s2p1d").read_text()
            )["file_list"]["sternheimer"],
        )

    def test_three_d_input_appends_only_the_next_response_shell(self):
        value = json.loads(
            (PROJECTED_PI_LOSS / "INPUT.pi_dpsi_joint_3s2p3d").read_text()
        )

        self.assertEqual(value["seed"], 20260718)
        self.assertEqual(value["element"]["Nu"], {"H": [3, 2, 3, 0, 0]})
        self.assertEqual(value["freeze_orbitals"], DZP_FREEZE_SPECS)
        self.assertEqual(
            value["C_init_info"]["C_init_file"],
            "../inputs/projected_pi_joint_3s2p2d_w002_plus_smooth_3d_ORBITAL_RESULTS.txt",
        )
        self.assertEqual(value["loss"]["mode"], "pi_dpsi_joint")
        self.assertEqual(value["loss"]["projected_pi_rank_tolerance"], 1.0e-12)
        self.assertEqual(value["loss"]["joint_dpsi_weight"], 0.02)
        self.assertEqual(value["loss"].get("radial_tail_weight", 0.0), 0.0)
        self.assertEqual(
            value["loss"].get("low_frequency_guard_weight", 0.0), 0.0
        )
        self.assertEqual(
            value["file_list"]["sternheimer"],
            json.loads(
                (PROJECTED_PI_LOSS / "INPUT.pi_dpsi_joint_3s2p2d").read_text()
            )["file_list"]["sternheimer"],
        )

    def test_optimizer_slurm_uses_full_normal_node_and_hash_preflight(self):
        script = (PROJECTED_PI_LOSS / "run_pi_dpsi_joint.slurm").read_text()
        required = (
            "#SBATCH --partition=normal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=30",
            "#SBATCH --mem=110610M",
            "#SBATCH --time=1-00:00:00",
            "/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python",
            "/work1/ghj/runtime/siab-projected-pi-mpl-20260801",
            "sha256sum -c",
            "SOURCE_COMMIT",
            "SOURCE_MANIFEST.sha256",
            "INPUTS.sha256",
            "INPUT_OVERRIDE",
            "/usr/bin/time -v",
            "OMP_NUM_THREADS=30",
            "PROJECTED_PI_METADATA.json",
            "Mode = pi_dpsi_joint",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, script)
        self.assertNotIn("--partition=debug", script)
        self.assertNotIn("git -C", script)

    def test_projected_pi_sos_slurm_matches_independent_cp_contract(self):
        script_path = PROJECTED_PI_LOSS / "run_pi_dpsi_joint_sos.slurm"
        self.assertTrue(script_path.is_file())
        script = script_path.read_text()
        required = (
            "#SBATCH --partition=normal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=30",
            "#SBATCH --mem=110610M",
            "#SBATCH --time=1-00:00:00",
            "#SBATCH --array=0-2",
            "PI_TRAIN_ROOT",
            "PI_SOS_ROOT",
            "PI_ORBITAL_SHA256",
            "case_names=(H2 H H_ghost)",
            "orbitals_per_h=",
            'nbands=("$((2 * orbitals_per_h))" "$orbitals_per_h" "$((2 * orbitals_per_h))")',
            "Number of Sorbital-->",
            "Number of Gorbital-->",
            "expected_orbital_entries=(1 1 2)",
            '"ecutwfc": "100"',
            '"rpa_ccp_rmesh_times": "5"',
            '"nfreq = 16"',
            '"prefix_coul_full = v1_coulomb_full_iq_"',
            "libRPA finished successfully",
            "PRODUCTION_OUTPUTS.sha256",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, script)
        self.assertNotIn("--partition=debug", script)


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


def _make_legacy_data(include_d=False):
    nl = 3 if include_d else 2
    info_stru = [
        info(Na={"H": 1}, Nb_true=2, weight=torch.tensor([0.5, 0.5]))
    ]
    nu = [3, 2, 1] if include_d else [3, 2]
    info_element = {"H": info(Nl=nl, Ne=4, Nu=nu)}
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
    if include_d:
        q_origin["H"].append(
            torch.tensor(
                [
                    [0.20, 0.10, 0.30, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.10, 0.25, 0.00, 0.15],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                ],
                dtype=torch.complex128,
            )
        )
    overlap_by_l = [[None for _ in range(nl)] for _ in range(nl)]
    zero_overlap_by_l = [[None for _ in range(nl)] for _ in range(nl)]
    for l1 in range(nl):
        for l2 in range(nl):
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


def _initial_c_with_d():
    c = _initial_c()
    c["H"].append(
        torch.tensor(
            [[0.20], [0.10], [0.70], [0.15]],
            dtype=torch.float64,
            requires_grad=True,
        )
    )
    return c


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
    def test_low_frequency_guard_changes_only_guard_options(self):
        joint = json.loads((EXAMPLE / "INPUT.st_dpsi_joint").read_text())
        guarded = json.loads(
            (EXAMPLE / "INPUT.st_dpsi_joint_low_frequency_guard").read_text()
        )
        expected = copy.deepcopy(joint)
        expected["loss"].update(
            {
                "low_frequency_guard_weight": 10.0,
                "low_frequency_guard_tolerance": 0.0,
            }
        )

        self.assertEqual(guarded, expected)
        self.assertEqual(guarded["seed"], SEED)
        self.assertEqual(guarded["element"]["Nu"], {"H": [3, 2]})
        self.assertEqual(guarded["freeze_orbitals"], DZP_FREEZE_SPECS)
        self.assertEqual(guarded["radial"]["Rcut"], 8)
        self.assertEqual(guarded["radial"]["Ecut"], 100)
        self.assertEqual(guarded["radial"]["smearing_sigma"], 0.1)

    def test_guarded_joint_runner_uses_full_normal_node_and_checks_guard(self):
        run_script = (
            EXAMPLE / "run_joint_low_frequency_guard.slurm"
        ).read_text()

        self.assertIn("#SBATCH -p normal", run_script)
        self.assertIn("#SBATCH -N 1", run_script)
        self.assertIn("#SBATCH --ntasks=1", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn("#SBATCH -t 1-00:00:00", run_script)
        self.assertIn("export OMP_NUM_THREADS=30", run_script)
        self.assertIn("Low-frequency guard weight", run_script)
        self.assertIn("Low-frequency guard tolerance", run_script)
        self.assertIn("Final lowest-frequency ST loss", run_script)
        self.assertIn("Initial lowest-frequency ST loss", run_script)

    def test_inputs_match_exact_campaign_contract(self):
        st_only = json.loads((EXAMPLE / "INPUT.st_only").read_text())
        expanded = json.loads((EXAMPLE / "INPUT.st_response_4s3p").read_text())
        constrained = json.loads((EXAMPLE / "INPUT.st_constrained").read_text())
        joint = json.loads((EXAMPLE / "INPUT.st_dpsi_joint").read_text())
        joint_expanded = json.loads(
            (EXAMPLE / "INPUT.st_dpsi_joint_4s3p").read_text()
        )
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
        self.assertEqual(joint_expanded["element"]["Nu"], {"H": [4, 3]})
        self.assertEqual(
            joint_expanded["freeze_orbitals"], DZP_FREEZE_SPECS
        )
        for key in joint:
            if key != "element":
                self.assertEqual(joint_expanded[key], joint[key])
        self.assertEqual(
            joint_expanded["element"]["Nt_all"], joint["element"]["Nt_all"]
        )
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

    def test_joint_campaign_uses_full_normal_node_and_checks_outputs(self):
        run_script = (EXAMPLE / "run_joint.slurm").read_text()

        self.assertIn("#SBATCH -p normal", run_script)
        self.assertIn("#SBATCH --ntasks=1", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn("module load apps/PyTorch/2.1.0", run_script)
        self.assertIn("export OMP_NUM_THREADS=30", run_script)
        self.assertIn("sternheimer_matrix.dat", run_script)
        self.assertIn("orb_matrix.0.dat", run_script)
        self.assertIn("orb_matrix.1.dat", run_script)
        self.assertIn("Mode = st_dpsi_joint", run_script)

    def test_held_out_h2_sos_recomputes_full_coulomb_pipeline(self):
        for case_name, nbands in (("H2", 18), ("H", 9)):
            with self.subTest(case=case_name):
                case = HELD_OUT_SOS / "cases" / case_name
                input_text = (case / "INPUT").read_text()
                librpa_text = (case / "librpa.in").read_text()
                stru_text = (case / "STRU").read_text()
                self.assertIn(f"nbands                  {nbands}", input_text)
                self.assertIn("ecutwfc                 100", input_text)
                self.assertIn("rpa                     1", input_text)
                self.assertIn("exx_pca_threshold       1e-4", input_text)
                self.assertIn("rpa_ccp_rmesh_times     5", input_text)
                self.assertIn(
                    "exx_singularity_correction massidda", input_text
                )
                self.assertIn("nfreq = 16", librpa_text)
                self.assertIn("prefix_coul_full = v1_coulomb_full_iq_", librpa_text)
                self.assertIn("vq_threshold = 0", librpa_text)
                self.assertIn("sqrt_coulomb_threshold = 0", librpa_text)
                self.assertIn("37.79452292169073", stru_text)
                self.assertIn("H_gga_8au_100Ry_3s2p.orb", stru_text)

        run_script = (HELD_OUT_SOS / "run_sos.slurm").read_text()
        self.assertIn("#SBATCH -p normal", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn("#SBATCH --array=0-1", run_script)
        self.assertIn("EXPECTED_ORBITAL_SHA256", run_script)
        self.assertIn(
            "30b7e5e3d80b59778b0fee836fcd0315c0cfd827621806eb3f2c9e659b8118a7",
            run_script,
        )
        self.assertIn("libRPA finished successfully", run_script)

    def test_fixed_dzp_tzdp_sos_is_a_same_size_fixed_abs_cp_comparison(self):
        run_script = (FIXED_DZP_TZDP_SOS / "run_sos_cp.slurm").read_text()

        self.assertIn("#SBATCH -p normal", run_script)
        self.assertIn("#SBATCH --array=0-5", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn(
            "lanes=(initial_tzdp initial_tzdp initial_tzdp "
            "fixed_dzp_joint fixed_dzp_joint fixed_dzp_joint)",
            run_script,
        )
        self.assertIn("case_names=(H2 H H_ghost H2 H H_ghost)", run_script)
        self.assertIn("nbands=(18 9 18 18 9 18)", run_script)
        self.assertIn("H_gga_8au_100Ry_3s2p.orb", run_script)
        self.assertIn("H_gga_8au_100Ry_fixed_dzp_joint_3s2p.orb", run_script)
        self.assertIn(
            "python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python",
            run_script,
        )
        self.assertIn(
            "H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs", run_script
        )
        self.assertIn("exx_pca_threshold", run_script)
        self.assertIn('"10"', run_script)
        self.assertIn("nfreq = 16", run_script)
        self.assertIn("prefix_coul_full = v1_coulomb_full_iq_", run_script)
        self.assertIn("libRPA finished successfully", run_script)

    def test_guarded_fixed_dzp_sos_reuses_exact_three_case_physics_contract(self):
        run_script = (
            FIXED_DZP_TZDP_SOS / "run_guarded_sos_cp.slurm"
        ).read_text()

        self.assertIn("#SBATCH -p normal", run_script)
        self.assertIn("#SBATCH --array=0-2", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn("case_names=(H2 H H_ghost)", run_script)
        self.assertIn("nbands=(18 9 18)", run_script)
        self.assertIn("expected_spins=(1 2 2)", run_script)
        self.assertIn("expected_electrons=(2 1 1)", run_script)
        self.assertIn(
            "H_gga_8au_100Ry_guarded_fixed_dzp_joint_3s2p.orb",
            run_script,
        )
        self.assertIn(
            "81c3f21a817d30d9d6802529650c4f177e27a62b39bea5d9d9de8f6c425d5330",
            run_script,
        )
        self.assertIn(
            "H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs", run_script
        )
        self.assertIn("exx_pca_threshold", run_script)
        self.assertIn('"10"', run_script)
        self.assertIn("nfreq = 16", run_script)
        self.assertIn("prefix_coul_full = v1_coulomb_full_iq_", run_script)
        self.assertIn("libRPA finished successfully", run_script)

    def test_expanded_held_out_uses_every_4s3p_band(self):
        for case_name, nbands in (("H2", 26), ("H", 13)):
            with self.subTest(case=case_name):
                case = HELD_OUT_SOS_4S3P / "cases" / case_name
                input_text = (case / "INPUT").read_text()
                stru_text = (case / "STRU").read_text()
                librpa_text = (case / "librpa.in").read_text()
                self.assertIn(f"nbands                  {nbands}", input_text)
                self.assertIn("ecutwfc                 100", input_text)
                self.assertIn("exx_pca_threshold       1e-4", input_text)
                self.assertIn("rpa_ccp_rmesh_times     5", input_text)
                self.assertIn("H_gga_8au_100Ry_4s3p.orb", stru_text)
                self.assertIn("nfreq = 16", librpa_text)
                self.assertIn(
                    "prefix_coul_full = v1_coulomb_full_iq_", librpa_text
                )

        run_script = (HELD_OUT_SOS_4S3P / "run_sos.slurm").read_text()
        self.assertIn("#SBATCH -p normal", run_script)
        self.assertIn("#SBATCH --array=0-1", run_script)
        self.assertIn(
            "b394bb7329754e38341050ca4beb3b242b78e4be50c418b8764c98226bc8f033",
            run_script,
        )
        self.assertIn("libRPA finished successfully", run_script)

    def test_d_response_held_out_predeclares_regenerated_and_fixed_abs_lanes(self):
        lanes = {
            "regenerated_4s3p3d": {
                "orbital": "H_gga_8au_100Ry_4s3p3d.orb",
                "nbands": {"H2": 56, "H": 28},
                "pca": "1e-4",
                "explicit_abs": False,
            },
            "fixed_4s3p": {
                "orbital": "H_gga_8au_100Ry_4s3p.orb",
                "nbands": {"H2": 26, "H": 13},
                "pca": "10",
                "explicit_abs": True,
            },
            "fixed_4s3p3d": {
                "orbital": "H_gga_8au_100Ry_4s3p3d.orb",
                "nbands": {"H2": 56, "H": 28},
                "pca": "10",
                "explicit_abs": True,
            },
        }

        for lane_name, lane in lanes.items():
            for case_name in ("H2", "H"):
                with self.subTest(lane=lane_name, case=case_name):
                    case = HELD_OUT_SOS_4S3P3D / "cases" / lane_name / case_name
                    input_text = (case / "INPUT").read_text()
                    stru_text = (case / "STRU").read_text()
                    librpa_text = (case / "librpa.in").read_text()
                    self.assertIn(
                        f"nbands                  {lane['nbands'][case_name]}",
                        input_text,
                    )
                    self.assertIn("ecutwfc                 100", input_text)
                    self.assertIn("rpa                     1", input_text)
                    self.assertIn(
                        f"exx_pca_threshold       {lane['pca']}", input_text
                    )
                    self.assertIn("rpa_ccp_rmesh_times     5", input_text)
                    self.assertIn(
                        "exx_singularity_correction massidda", input_text
                    )
                    self.assertIn("37.79452292169073", stru_text)
                    self.assertIn(lane["orbital"], stru_text)
                    if lane["explicit_abs"]:
                        self.assertIn("ABFS_ORBITAL", stru_text)
                        self.assertIn(
                            "H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs", stru_text
                        )
                    else:
                        self.assertNotIn("ABFS_ORBITAL", stru_text)
                    self.assertIn("nfreq = 16", librpa_text)
                    self.assertIn(
                        "prefix_coul_full = v1_coulomb_full_iq_", librpa_text
                    )
                    self.assertIn("vq_threshold = 0", librpa_text)
                    self.assertIn("sqrt_coulomb_threshold = 0", librpa_text)

        run_script = (HELD_OUT_SOS_4S3P3D / "run_sos.slurm").read_text()
        self.assertIn("#SBATCH -p normal", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn("#SBATCH --array=0-5", run_script)
        self.assertIn("libRPA finished successfully", run_script)

    def test_d_response_bsse_diagnostic_uses_the_full_dimer_basis(self):
        diagnostic = HELD_OUT_SOS_4S3P3D / "bsse_diagnostic"
        lanes = {
            "fixed_4s3p": ("H_gga_8au_100Ry_4s3p.orb", 26),
            "fixed_4s3p3d": ("H_gga_8au_100Ry_4s3p3d.orb", 56),
        }

        for lane_name, (orbital, nbands) in lanes.items():
            with self.subTest(lane=lane_name):
                case = diagnostic / "cases" / lane_name / "H_ghost"
                input_text = (case / "INPUT").read_text()
                stru_text = (case / "STRU").read_text()
                librpa_text = (case / "librpa.in").read_text()
                self.assertIn("ntype                   2", input_text)
                self.assertIn(f"nbands                  {nbands}", input_text)
                self.assertIn("nelec                   1", input_text)
                self.assertIn("nspin                   2", input_text)
                self.assertIn("exx_pca_threshold       10", input_text)
                self.assertIn("rpa_ccp_rmesh_times     5", input_text)
                self.assertIn("H_empty", stru_text)
                self.assertEqual(stru_text.count(orbital), 2)
                self.assertEqual(
                    stru_text.count("H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs"), 2
                )
                self.assertIn("0.48147879757009904", stru_text)
                self.assertIn("0.518521202429901", stru_text)
                self.assertIn("nfreq = 16", librpa_text)
                self.assertIn(
                    "prefix_coul_full = v1_coulomb_full_iq_", librpa_text
                )

        run_script = (diagnostic / "run_bsse.slurm").read_text()
        self.assertIn("#SBATCH -p normal", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn("#SBATCH --array=0-1", run_script)
        self.assertIn("libRPA finished successfully", run_script)

    def test_greedy_response_producers_fix_high_l_sternheimer_contract(self):
        for producer_name in (
            "producer_atom",
            "producer_h2",
            "producer_h2_fragment_ghost",
        ):
            with self.subTest(producer=producer_name):
                producer = GREEDY_RESPONSE / producer_name
                input_text = (producer / "INPUT").read_text()
                stru_text = (producer / "STRU").read_text()

                self.assertIn("out_sternheimer_siab    1", input_text)
                self.assertIn("out_sternheimer_librpa  0", input_text)
                self.assertIn(
                    "sternheimer_siab_coulomb_threshold  1e-10", input_text
                )
                self.assertIn("sternheimer_siab_lmax  4", input_text)
                self.assertIn("sternheimer_nfreq       16", input_text)
                self.assertIn("sternheimer_frequency_mpi 1", input_text)
                self.assertIn("sternheimer_channel_mpi   1", input_text)
                self.assertIn("exx_pca_threshold       10", input_text)
                self.assertIn("rpa_ccp_rmesh_times     5", input_text)
                self.assertIn("ABFS_ORBITAL", stru_text)
                self.assertIn(
                    "H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs", stru_text
                )

    def test_greedy_response_target_runner_is_immutable_and_complete(self):
        run_script = (GREEDY_RESPONSE / "run_targets.slurm").read_text()

        self.assertIn("#SBATCH --partition=normal", run_script)
        self.assertIn("#SBATCH --nodes=32", run_script)
        self.assertIn("#SBATCH --ntasks=32", run_script)
        self.assertIn("#SBATCH --ntasks-per-node=1", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn("#SBATCH --time=1-00:00:00", run_script)
        self.assertIn("#SBATCH --array=0-2", run_script)
        self.assertIn("for required in INPUT STRU KPT; do", run_script)
        self.assertIn('test -s "$template_dir/$required"', run_script)
        self.assertIn("test ! -e \"$output_dir\"", run_script)
        self.assertIn('mpirun -np "$SLURM_NTASKS" -ppn 1', run_script)
        self.assertIn(
            "d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f",
            run_script,
        )
        self.assertIn("expected_primitive_columns=(625 1250 1250)", run_script)
        self.assertIn("expected_solved_equations=(3424 6848 6848)", run_script)
        self.assertIn("validate_targets.py", run_script)
        self.assertIn("abacus_source_commit=", run_script)
        self.assertIn("abacus_sha256=", run_script)
        self.assertIn(
            "abacus_source_commit=c273b4ee7051138293d9988c3eb79bee36c0af10",
            run_script,
        )
        self.assertIn(
            "abacus_sha256=ff38348fbad89fde4a985c13f97b59ffc94353c22c7098e19b373c1ef7e76fee",
            run_script,
        )
        self.assertIn(
            "siab_channel_mpi_exact_c273b4ee7_20260726/artifacts/job21389808/abacus_3p",
            run_script,
        )
        self.assertIn("siab_greedy_targets_source_h2_channel_mpi_prod_v1_20260726", run_script)
        self.assertIn(
            "siab_greedy_targets_h2_channel_mpi_prod_v1_20260726", run_script
        )
        self.assertIn("sternheimer_channel_mpi yes", run_script)
        self.assertIn("frequency_group_size 2", run_script)
        self.assertIn("mpi_ranks 32", run_script)
        self.assertNotIn(
            "ABACUS_STERNHEIMER_FD_ST_ORBITAL_FILES", run_script
        )
        self.assertNotIn("PENDING_GLOBAL_WHITENING", run_script)

    def test_greedy_selection_runner_uses_validated_targets_and_full_node(self):
        run_script = (GREEDY_RESPONSE / "run_selection.slurm").read_text()
        driver = (GREEDY_RESPONSE / "run_response_selection.py").read_text()
        template = json.loads(
            (GREEDY_RESPONSE / "optimizer_template.json").read_text()
        )

        self.assertIn("#SBATCH --partition=normal", run_script)
        self.assertIn("#SBATCH --nodes=1", run_script)
        self.assertIn("#SBATCH --ntasks=1", run_script)
        self.assertIn("#SBATCH --cpus-per-task=30", run_script)
        self.assertIn("#SBATCH --mem=110610M", run_script)
        self.assertIn("#SBATCH --time=1-00:00:00", run_script)
        self.assertEqual(run_script.count("target_validation.json"), 2)
        self.assertNotIn("fragment_ghost", run_script)
        self.assertNotIn("--ghost-target", driver)
        self.assertIn("campaign_manifest.json", run_script)
        self.assertIn("run_response_selection.py", run_script)
        self.assertIn(
            "siab_greedy_selection_source_h_h2_physical_only_prod_v5_20260727",
            run_script,
        )
        self.assertIn(
            "siab_greedy_targets_h2_channel_mpi_prod_v1_20260726", run_script
        )
        self.assertIn(
            "siab_greedy_selection_campaign_h_h2_physical_only_prod_v5_20260727",
            run_script,
        )
        self.assertNotIn("rpa_binding", driver.lower())
        self.assertNotIn("h2_energy", driver.lower())
        self.assertEqual(template["loss"]["mode"], "st_dpsi_joint")
        self.assertEqual(template["element"]["Nu"]["H"], [2, 1, 0, 0, 0])
        self.assertEqual(len(template["file_list"]["origin"]), 3)
        self.assertEqual(len(template["file_list"]["linear"][0]), 3)


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
    def test_joint_d_channel_receives_st_and_dft_dpsi_gradients(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        c = _initial_c_with_d()
        initial = tuple(value.detach().clone() for value in c["H"])
        legacy = _make_legacy_data(include_d=True)
        stage = {
            "optimizer": "Adam",
            "kwargs": {"lr": 0.003},
            "cal_T": False,
            "norm": "one",
            "max_steps": 5,
            "loss": {
                "mode": "st_dpsi_joint",
                **LOSS_DEFAULTS,
                "joint_dpsi_weight": 0.1,
            },
        }
        converge = Opt_Orbital_Converge()
        converge.set_info(
            {"origin": ["synthetic"], "linear": [["synthetic-linear"]]},
            [stage],
            legacy["info_stru"],
            {
                "init_from_file": True,
                "freeze_orbitals": DZP_FREEZE_SPECS,
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
        fixed_columns = [
            OrbitalColumn("H", 0, 0, 0, 1),
            OrbitalColumn("H", 0, 0, 0, 2),
            *(OrbitalColumn("H", 0, 1, m, 1) for m in (-1, 0, 1)),
        ]
        evaluator = SternheimerSpillage(
            _make_sternheimer_data_with_d(), c, fixed_columns
        )
        converge.set_sternheimer_spillage(evaluator)

        components = converge._make_spillage(stage).cal_components(c)
        legacy_loss = components["dft_origin"] + 0.1 * components["dft_dpsi"]
        legacy_d_gradient = torch.autograd.grad(
            legacy_loss, c["H"][2], retain_graph=True
        )[0]
        st_d_gradient = torch.autograd.grad(
            evaluator.evaluate(c).loss, c["H"][2]
        )[0]
        self.assertGreater(torch.linalg.vector_norm(legacy_d_gradient).item(), 0.0)
        self.assertGreater(torch.linalg.vector_norm(st_d_gradient).item(), 0.0)

        result = converge.cal_converge(
            c, (io.StringIO(), io.StringIO())
        )["C"]["H"]
        self.assertTrue(torch.equal(initial[0][:, :2], result[0][:, :2]))
        self.assertTrue(torch.equal(initial[1][:, :1], result[1][:, :1]))
        self.assertFalse(torch.equal(initial[2], result[2]))

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
