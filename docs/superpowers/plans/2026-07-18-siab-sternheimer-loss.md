# SIAB Sternheimer Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend SIAB to optimize the upper H-TZDP orbitals against compact Delta-Sternheimer first-order-wavefunction data, while keeping the fixed `1s` orbital exact and optionally constraining the original DFT and dpsi losses.

**Architecture:** A strict versioned reader converts one tagged ABACUS file into a typed `SternheimerData` object. A separate loss module assembles molecular-orbital columns from SIAB radial coefficients, removes the fixed level1 space with a Schur complement, and evaluates the weighted ST trace loss through Hermitian Cholesky solves. The existing optimizer receives named loss components and an explicit freeze mask without changing legacy inputs or the seven-value return signature of `read_json()`.

**Tech Stack:** Python 3, PyTorch complex128 autograd, NumPy, `unittest`, existing SIAB JSON and coefficient formats.

---

## File Map

Create:

- `SIAB/tests/common.py`: deterministic import path and synthetic SIAB objects.
- `SIAB/tests/test_legacy_spillage.py`: pins the current origin-only numerical behavior.
- `SIAB/opt_orb_pytorch_dpsi/sternheimer_data.py`: immutable typed data model and validation.
- `SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer.py`: strict tagged v1 parser.
- `SIAB/tests/fixtures/sternheimer_matrix_v1.dat`: smallest valid parser/loss fixture.
- `SIAB/tests/test_read_sternheimer.py`: valid and malformed-file parser tests.
- `SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py`: primitive-column assembly and level1-projected ST loss.
- `SIAB/tests/test_sternheimer_spillage.py`: analytic, projection-invariance, conditioning, and gradient tests.
- `SIAB/opt_orb_pytorch_dpsi/freeze_orbitals.py`: explicit `(element,l,zeta)` validation and gradient masking.
- `SIAB/opt_orb_pytorch_dpsi/optimization_loss.py`: named legacy/ST/constrained loss composition.
- `SIAB/tests/test_loss_and_freeze.py`: legacy identity, hinges, and bitwise-freeze tests.
- `SIAB/example_H_sternheimer/INPUT.st_only`: deterministic H-TZDP ST-only optimization.
- `SIAB/example_H_sternheimer/INPUT.st_constrained`: deterministic constrained optimization.
- `SIAB/example_H_sternheimer/README.md`: provenance and run/acceptance commands.
- `SIAB/tests/test_h_sternheimer_smoke.py`: short end-to-end synthetic optimization.

Modify:

- `SIAB/opt_orb_pytorch_dpsi/IO/read_json.py`: copy opt-in ST/freeze/loss settings into existing returned dictionaries.
- `SIAB/opt_orb_pytorch_dpsi/opt_orbital_spillage.py`: expose `dft_origin` and `dft_dpsi` while preserving their exact legacy sum.
- `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py`: compose named losses, log diagnostics, and mask frozen gradients.
- `SIAB/opt_orb_pytorch_dpsi/main.py`: load ST data, pass it to the optimizer, and write loss metadata.
- `SIAB/opt_orb_pytorch_dpsi/main_each.py`: reject the unsupported ST path explicitly instead of ignoring it.
- `SIAB/opt_orb_pytorch_dpsi/IO/func_C.py`: include named final losses and mode in `ORBITAL_RESULTS.txt` without changing coefficient syntax.
- `SIAB/README.md`: document the new opt-in path and backward compatibility.

## Fixed Conventions

The v1 producer file uses these sections and meanings:

```text
<STERNHEIMER_SIAB_HEADER>
format_version 1
n_reference 2
n_primitive 4
n_blocks 2
grid_volume_bohr3 0.125
</STERNHEIMER_SIAB_HEADER>
<PRIMITIVE_BLOCKS>
# element atom_index l m n_primitive offset
H 0 0 0 2 0
H 0 1 0 2 2
</PRIMITIVE_BLOCKS>
<REFERENCE_METADATA>
# occupied_state auxiliary_channel frequency_ha occupation frequency_weight norm
0 0 0.5 2.0 0.3 1.2
0 0 1.5 2.0 0.7 0.8
</REFERENCE_METADATA>
<OVERLAP_Q>
# row-major <Y_rho|B_e>, real imag
0.8 0.0
0.1 0.0
0.2 0.0
0.0 0.0
0.4 0.0
0.3 0.0
0.0 0.0
0.1 0.0
</OVERLAP_Q>
<OVERLAP_S>
# row-major <B_e|B_ep>, real imag
1.0 0.0
0.0 0.0
0.0 0.0
0.0 0.0
0.0 0.0
1.0 0.0
0.0 0.0
0.0 0.0
0.0 0.0
0.0 0.0
1.0 0.0
0.0 0.0
0.0 0.0
0.0 0.0
0.0 0.0
1.0 0.0
</OVERLAP_S>
<PROVENANCE_JSON>
{"abacus_commit":"19ab21e01d02cc805604ed77a6e269af698fdd1d","auxiliary_basis_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","cell_bohr":[20.0,0.0,0.0,0.0,20.0,0.0,0.0,0.0,20.0],"ecut_ry":25.0,"kernel":"full_coulomb","orbital_sha256":"7e398340398306a6baf1c61ea68944d81ed43667473fbcc290d6541c4a661d1c","pseudopotential_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","spin_convention":"occupation_in_metadata"}
</PROVENANCE_JSON>
```

`Q[rho,e]` is always `conj(Y_rho) * B_e` integrated with the uniform-grid volume element. `S_B[e,e']` is `conj(B_e) * B_e'` with the same grid and volume element. `n_rho` already contains that volume element and is recorded in `REFERENCE_METADATA`. The effective row weight is `occupation * frequency_weight`; frequency itself is metadata and is not multiplied into the loss.

### Task 1: Pin Legacy Spillage Before Refactoring

**Files:**
- Create: `SIAB/tests/common.py`
- Create: `SIAB/tests/test_legacy_spillage.py`

- [ ] **Step 1: Add deterministic SIAB test imports and object builders**

```python
# SIAB/tests/common.py
from pathlib import Path
import sys

OPT_DIR = Path(__file__).resolve().parents[1] / "opt_orb_pytorch_dpsi"
if str(OPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPT_DIR))

import util


def info(**values):
    result = util.Info()
    result.__dict__.update(values)
    return result
```

- [ ] **Step 2: Write the exact current-behavior regression**

```python
# SIAB/tests/test_legacy_spillage.py
import unittest
import torch

from common import info
from opt_orbital_spillage import Opt_Orbital_Spillage


class LegacySpillageTest(unittest.TestCase):
    def test_origin_only_loss_remains_point_four(self):
        info_stru = [info(Na={"H": 1}, Nb_true=1, weight=torch.tensor([1.0]))]
        info_element = {"H": info(Nl=1, Ne=2, Nu=[1])}
        q = {"H": [torch.tensor([[0.8, 0.6]], dtype=torch.complex128)]}
        s = {
            ("H", "H"): [[torch.eye(2, dtype=torch.complex128).reshape(1, 1, 2, 1, 1, 2)]]
        }
        c = {"H": [torch.tensor([[1.0], [0.0]], dtype=torch.float64, requires_grad=True)]}
        target = [torch.tensor([1.0], dtype=torch.float64)]
        loss = Opt_Orbital_Spillage(
            info_stru, info_element, {"same_band": True}, "one", {"origin": ["synthetic"]}
        )
        loss.set_QSVI([q], [s], target)
        self.assertAlmostEqual(loss.cal_Spillage(c).item(), 0.4, places=14)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the regression in the PyTorch environment**

Run from repository root on `df_dcu`:

```bash
python3 -c 'import sys,torch; print(sys.version); print(torch.__version__)'
python3 -m unittest discover -s SIAB/tests -p 'test_legacy_spillage.py' -v
```

Expected: Python and PyTorch versions are printed and `test_origin_only_loss_remains_point_four` passes. On the Mac, run only `python3 -m compileall -q SIAB`; local numerical tests are not accepted because PyTorch is absent.

- [ ] **Step 4: Commit the regression**

```bash
git add SIAB/tests/common.py SIAB/tests/test_legacy_spillage.py
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'test(siab): pin legacy spillage behavior'
```

Verify: `git show -s --format='%an <%ae>%n%cn <%ce>' HEAD` prints Codex as author and AroundPeking as committer; set the committer environment specified in the repository workflow when committing.

### Task 2: Add the Versioned Sternheimer Reader

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/sternheimer_data.py`
- Create: `SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer.py`
- Create: `SIAB/tests/fixtures/sternheimer_matrix_v1.dat`
- Create: `SIAB/tests/test_read_sternheimer.py`

- [ ] **Step 1: Write parser acceptance and rejection tests**

The tests must assert:

```python
data = read_sternheimer(FIXTURE)
self.assertEqual(data.format_version, 1)
self.assertEqual(data.q.shape, (2, 4))
self.assertEqual(data.overlap.shape, (4, 4))
self.assertEqual(data.norm.tolist(), [1.2, 0.8])
self.assertEqual(data.effective_weight.tolist(), [0.6, 1.4])
self.assertEqual(data.blocks[1].key, ("H", 0, 1, 0))
self.assertEqual(data.provenance["kernel"], "full_coulomb")
```

Create four temporary variants and assert exact errors:

```python
with self.assertRaisesRegex(ValueError, "unsupported format_version 2"):
    read_sternheimer(version_two)
with self.assertRaisesRegex(ValueError, "OVERLAP_Q expected 8 complex values"):
    read_sternheimer(short_q)
with self.assertRaisesRegex(ValueError, "OVERLAP_S is not Hermitian"):
    read_sternheimer(nonhermitian_s)
with self.assertRaisesRegex(ValueError, "missing provenance key: kernel"):
    read_sternheimer(no_kernel)
```

- [ ] **Step 2: Run the reader tests and confirm they fail**

```bash
python3 -m unittest discover -s SIAB/tests -p 'test_read_sternheimer.py' -v
```

Expected: import failure for `IO.read_sternheimer`.

- [ ] **Step 3: Implement the typed model**

Use these public types and names exactly:

```python
from dataclasses import dataclass
from typing import Dict, Tuple
import torch


@dataclass(frozen=True)
class PrimitiveBlock:
    element: str
    atom_index: int
    l: int
    m: int
    n_primitive: int
    offset: int

    @property
    def key(self):
        return (self.element, self.atom_index, self.l, self.m)


@dataclass(frozen=True)
class SternheimerData:
    format_version: int
    grid_volume_bohr3: float
    blocks: Tuple[PrimitiveBlock, ...]
    occupied_state: torch.Tensor
    auxiliary_channel: torch.Tensor
    frequency_ha: torch.Tensor
    occupation: torch.Tensor
    frequency_weight: torch.Tensor
    norm: torch.Tensor
    q: torch.Tensor
    overlap: torch.Tensor
    provenance: Dict[str, object]

    @property
    def effective_weight(self):
        return self.occupation * self.frequency_weight
```

Validation in `SternheimerData` must require complex128 `q/overlap`, float64 reference arrays, contiguous non-overlapping block offsets covering exactly `n_primitive`, positive norms, nonnegative weights, Hermitian `overlap` within `1e-10`, and these provenance keys: `abacus_commit`, `auxiliary_basis_sha256`, `cell_bohr`, `ecut_ry`, `kernel`, `orbital_sha256`, `pseudopotential_sha256`, `spin_convention`. `cell_bohr` is exactly nine finite row-major lattice-vector components and must define a nonsingular cell; zero and negative components are valid.

- [ ] **Step 4: Implement a strict section parser**

`read_sternheimer(path: str) -> SternheimerData` must:

1. read line-by-line, stripping comments after `#` and blank lines;
2. allow each required tag exactly once and reject text outside tags;
3. parse header keys into a dictionary and reject unknown or repeated keys;
4. parse `n_reference` metadata rows, `n_reference*n_primitive` Q pairs, and `n_primitive*n_primitive` S pairs;
5. parse one JSON object from `PROVENANCE_JSON`;
6. create CPU tensors with `torch.float64` or `torch.complex128` and call model validation.

Do not use a cross-section regular expression. Expose only:

```python
def read_sternheimer(path):
    sections = _read_tagged_sections(path)
    header = _parse_header(sections["STERNHEIMER_SIAB_HEADER"])
    _require_version_one(header)
    blocks = _parse_blocks(sections["PRIMITIVE_BLOCKS"], header)
    metadata = _parse_metadata(sections["REFERENCE_METADATA"], header)
    q = _parse_complex_matrix(sections["OVERLAP_Q"], header["n_reference"], header["n_primitive"], "OVERLAP_Q")
    overlap = _parse_complex_matrix(sections["OVERLAP_S"], header["n_primitive"], header["n_primitive"], "OVERLAP_S")
    provenance = _parse_provenance(sections["PROVENANCE_JSON"])
    return _build_data(header, blocks, metadata, q, overlap, provenance)
```

- [ ] **Step 5: Run reader and legacy tests**

```bash
python3 -m unittest discover -s SIAB/tests -v
```

Expected: all reader tests and the legacy `0.4` regression pass.

- [ ] **Step 6: Commit the reader stage**

```bash
git add SIAB/opt_orb_pytorch_dpsi/sternheimer_data.py \
        SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer.py \
        SIAB/tests/fixtures/sternheimer_matrix_v1.dat \
        SIAB/tests/test_read_sternheimer.py
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'feat(siab): read versioned Sternheimer targets'
```

### Task 3: Implement the Level1-Projected ST Loss

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py`
- Create: `SIAB/tests/test_sternheimer_spillage.py`

- [ ] **Step 1: Write four numerical tests before implementation**

Use float64/complex128 and assert:

1. `S_B=I`, fixed `C0=e0`, variable `C1=e1`, `Q=[0.6,0.64,0.48]`, `n=1` gives `nbar=0.64`, residual `0.2304`, and normalized loss `0.36`.
2. Changing the fixed-space amplitude from `0.6` to `0.6j` leaves the projected loss unchanged because its norm and all components outside fixed `C0` are unchanged.
3. `C1=[0,1,alpha]` at `alpha=0.2` has an autograd derivative agreeing with central finite difference `(L(alpha+1e-6)-L(alpha-1e-6))/(2e-6)` to relative tolerance `1e-6` and absolute tolerance `1e-8`.
4. Linearly dependent variable columns raise `RuntimeError` containing `variable overlap is not positive definite` and the block description.

- [ ] **Step 2: Run the ST tests and confirm import failure**

```bash
python3 -m unittest discover -s SIAB/tests -p 'test_sternheimer_spillage.py' -v
```

Expected: import failure for `sternheimer_spillage`.

- [ ] **Step 3: Implement coefficient-column assembly**

Use these public names:

```python
@dataclass(frozen=True)
class OrbitalColumn:
    element: str
    atom_index: int
    l: int
    m: int
    zeta: int


def assemble_orbital_coefficients(data, c):
    columns = []
    labels = []
    for block in data.blocks:
        radial = c[block.element][block.l]
        if radial.shape[0] != block.n_primitive:
            raise ValueError(
                f"primitive count mismatch for {block.key}: "
                f"file={block.n_primitive}, C={radial.shape[0]}"
            )
        for zeta in range(radial.shape[1]):
            column = torch.zeros(data.q.shape[1], dtype=torch.complex128)
            start = block.offset
            column[start:start + block.n_primitive] = radial[:, zeta].to(torch.complex128)
            columns.append(column)
            labels.append(OrbitalColumn(block.element, block.atom_index, block.l, block.m, zeta + 1))
    return torch.stack(columns, dim=1), tuple(labels)
```

For H, one radial p coefficient column is repeated in the three `m=-1,0,1` blocks supplied by the producer; the code must not invent missing `m` blocks.

- [ ] **Step 4: Implement the projected loss with Cholesky solves**

Expose:

```python
@dataclass(frozen=True)
class SternheimerLossResult:
    loss: torch.Tensor
    weighted_residual: torch.Tensor
    weighted_norm: torch.Tensor
    max_condition: float


def sternheimer_spillage(data, c, fixed_orbitals, condition_limit=1.0e12):
    a, labels = assemble_orbital_coefficients(data, c)
    fixed = [index for index, label in enumerate(labels) if label in fixed_orbitals]
    variable = [index for index, label in enumerate(labels) if label not in fixed_orbitals]
    if not fixed or not variable:
        raise ValueError("Sternheimer loss requires nonempty fixed and variable orbital spaces")
    a0 = a[:, fixed]
    a1 = a[:, variable]
    s00 = a0.conj().T @ data.overlap @ a0
    s01 = a0.conj().T @ data.overlap @ a1
    s11 = a1.conj().T @ data.overlap @ a1
    q0 = data.q @ a0
    q1 = data.q @ a1
    solve00_s01, cond00 = _hermitian_solve(s00, s01, "fixed overlap", condition_limit)
    solve00_q0h, _ = _hermitian_solve(s00, q0.conj().T, "fixed overlap", condition_limit)
    nbar = data.norm - torch.sum(q0 * solve00_q0h.T, dim=1).real
    qbar = q1 - q0 @ solve00_s01
    sbar = s11 - s01.conj().T @ solve00_s01
    solvebar, condbar = _hermitian_solve(sbar, qbar.conj().T, "variable overlap", condition_limit)
    represented = torch.sum(qbar * solvebar.T, dim=1).real
    residual = nbar - represented
    scale = torch.maximum(data.norm.max(), torch.tensor(1.0, dtype=torch.float64))
    if torch.any(residual < -1.0e-10 * scale):
        raise RuntimeError(f"negative projected residual: {residual.min().item():.6e}")
    residual = torch.clamp(residual, min=0.0)
    weight = data.effective_weight
    weighted_norm = torch.sum(weight * nbar)
    weighted_residual = torch.sum(weight * residual)
    if weighted_norm <= 0:
        raise RuntimeError("non-positive weighted projected Sternheimer norm")
    return SternheimerLossResult(weighted_residual / weighted_norm, weighted_residual, weighted_norm, max(cond00, condbar))
```

The displayed function pins the algebra. The optimizer-facing implementation must be a `SternheimerSpillage(data, c0, fixed_orbitals, condition_limit)` object whose constructor evaluates and stores detached `a0`, `s00`, its Cholesky factor, `q0`, `S00^-1 q0^H`, and `nbar` once from the unmodified `C0`. Its `evaluate(c)` method verifies that orbital labels are unchanged, forms the current `s01`, solves it with the cached `s00` Cholesky factor, and evaluates the variable-space terms. This enforces the design requirement that the fixed level1 projector is precomputed rather than rebuilt during every optimizer closure.

`_hermitian_solve` must symmetrize as `(S+S.H)/2`, use `torch.linalg.cholesky_ex`, reject any nonzero `info`, compute `torch.linalg.cond(S).item()`, reject a value above `condition_limit`, and solve with `torch.cholesky_solve`. Do not use `torch.inverse`.

- [ ] **Step 5: Run all numerical tests**

```bash
python3 -m unittest discover -s SIAB/tests -v
```

Expected: analytic loss is `0.36`, projection invariance and gradient checks pass, singular overlap is rejected, and legacy loss remains `0.4`.

- [ ] **Step 6: Commit the ST loss stage**

```bash
git add SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py \
        SIAB/tests/test_sternheimer_spillage.py
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'feat(siab): add projected Sternheimer loss'
```

### Task 4: Add Named Components, Constraints, and Explicit Freeze Masks

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/freeze_orbitals.py`
- Create: `SIAB/opt_orb_pytorch_dpsi/optimization_loss.py`
- Create: `SIAB/tests/test_loss_and_freeze.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/IO/read_json.py:11-48`
- Modify: `SIAB/opt_orb_pytorch_dpsi/opt_orbital_spillage.py:33-63`
- Modify: `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py:11-111`

- [ ] **Step 1: Write input, component, hinge, and freeze tests**

Tests must verify:

```python
self.assertEqual(spillage.cal_Spillage(c), 2.0 * components["dft_origin"] + components["dft_dpsi"])
self.assertEqual(compose_loss("st_only", st=st, dft=dft, dpsi=dpsi, baseline=baseline, config=config)["total"], st)
self.assertEqual(constrained["constraint_dft"].item(), 10.0 * (1.20 - 1.05) ** 2)
self.assertEqual(constrained["constraint_dpsi"].item(), 10.0 * (1.25 - 1.10) ** 2)
```

An Adam step with `freeze_orbitals=[{"element":"H","l":0,"zeta":1}]` must leave `C["H"][0][:,0]` bitwise equal to its clone while changing at least one value in `C["H"][0][:,1]`. Invalid element, l, and zeta each raise a `ValueError` that includes the invalid tuple.

The JSON regression must call `read_json()` on an old input and assert seven returned values. For a new input, it must assert:

```python
self.assertEqual(c_init["freeze_orbitals"], [{"element": "H", "l": 0, "zeta": 1}])
self.assertEqual(optimize[0]["loss"]["mode"], "st_constrained")
```

- [ ] **Step 2: Run tests and confirm missing APIs**

```bash
python3 -m unittest discover -s SIAB/tests -p 'test_loss_and_freeze.py' -v
```

Expected: failures for `cal_components`, `compose_loss`, and freeze-mask imports.

- [ ] **Step 3: Split the exact legacy loss into named components**

`Opt_Orbital_Spillage.cal_components(C)` returns tensors:

```python
{
    "dft_origin": origin_without_the_existing_factor_two,
    "dft_dpsi": sum_of_all_linear_terms_or_zero,
}
```

Keep legacy identity exactly:

```python
def cal_Spillage(self, C):
    components = self.cal_components(C)
    return 2 * components["dft_origin"] + components["dft_dpsi"]
```

- [ ] **Step 4: Add explicit freeze validation and masking**

Use one-based `zeta` in JSON and labels:

```python
def validate_freeze_orbitals(specs, c):
    indices = set()
    for spec in specs:
        key = (spec["element"], int(spec["l"]), int(spec["zeta"]))
        element, l_value, zeta = key
        if element not in c or l_value < 0 or l_value >= len(c[element]) or zeta < 1 or zeta > c[element][l_value].shape[1]:
            raise ValueError(f"invalid frozen orbital {key}")
        indices.add((element, l_value, zeta - 1))
    return indices


def zero_frozen_gradients(c, indices):
    for element, l_value, zero_based_zeta in indices:
        if c[element][l_value].grad is None:
            raise RuntimeError(f"missing gradient for frozen orbital {(element, l_value, zero_based_zeta + 1)}")
        c[element][l_value].grad[:, zero_based_zeta] = 0.0
```

When explicit freeze entries exist, apply exactly these indices after every `backward()`. When absent, retain the old `C_read_index` path unchanged.

- [ ] **Step 5: Add loss composition with baseline-normalized hinges**

Expose `compose_loss(mode, st, dft, dpsi, baseline, config)`. It returns all six keys:

```python
normalized_dft = dft / max(baseline["dft_origin"], config["epsilon"])
normalized_dpsi = dpsi / max(baseline["dft_dpsi"], config["epsilon"])
constraint_dft = config["constraint_penalty_dft"] * torch.relu(normalized_dft - 1.0 - config["tau_dft"]) ** 2
constraint_dpsi = config["constraint_penalty_dpsi"] * torch.relu(normalized_dpsi - 1.0 - config["tau_dpsi"]) ** 2
```

For `st_only`, both constraint tensors are zero and `total=st`. For `st_constrained`, `total=st+constraint_dft+constraint_dpsi`. Reject every other mode. Store baseline values from an unmodified clone of `C0` before the first optimizer step.

- [ ] **Step 6: Preserve the seven-value JSON API**

Before the existing return in `read_json.py`, add behavior equivalent to:

```python
if "freeze_orbitals" in input:
    input["C_init_info"]["freeze_orbitals"] = input["freeze_orbitals"]
if "loss" in input:
    for stage in input["optimize"]:
        stage["loss"] = input["loss"]
```

Defaults for the opt-in ST path are `epsilon=1e-14`, `condition_limit=1e12`, `tau_dft=0.05`, `tau_dpsi=0.10`, and both penalties `10.0`. Do not add ST defaults to old inputs.

- [ ] **Step 7: Integrate six-column logging and accepted-point selection**

Each `Spillage.dat` row must contain:

```text
istep_big istep_small istep_all dft_origin dft_dpsi sternheimer constraint_dft constraint_dpsi total max_st_condition accepted
```

For constrained mode, an accepted best point satisfies both normalized constraints and lowers `sternheimer` relative to the previous accepted point. If no point satisfies both constraints, fail after optimization with the smallest observed violations in the message; do not silently return the unconstrained minimum.

- [ ] **Step 8: Run the full suite and commit stage 4**

```bash
python3 -m unittest discover -s SIAB/tests -v
python3 -m compileall -q SIAB
git add SIAB/opt_orb_pytorch_dpsi/IO/read_json.py \
        SIAB/opt_orb_pytorch_dpsi/opt_orbital_spillage.py \
        SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py \
        SIAB/opt_orb_pytorch_dpsi/freeze_orbitals.py \
        SIAB/opt_orb_pytorch_dpsi/optimization_loss.py \
        SIAB/tests/test_loss_and_freeze.py
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'feat(siab): constrain Sternheimer basis optimization'
```

Expected: all tests pass; fixed-column equality is bitwise, not tolerance-based.

### Task 5: Integrate the Reader Into `main.py`

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/main.py:24-65`
- Modify: `SIAB/opt_orb_pytorch_dpsi/IO/func_C.py:58-80`
- Create: `SIAB/tests/test_main_sternheimer.py`

- [ ] **Step 1: Write a main-path integration test**

The test uses a temporary INPUT, one synthetic legacy origin file copied from a test fixture, the ST v1 fixture, and an initial coefficient file. It runs one Adam step and asserts:

```python
self.assertIn("sternheimer", header.split())
self.assertIn("constraint_dft", header.split())
self.assertIn("Mode = st_only", orbital_results)
self.assertIn("Sternheimer loss =", orbital_results)
```

The test also asserts that `main_each.py` exits with `ValueError("sternheimer input is supported by main.py only")` instead of ignoring the new data.

- [ ] **Step 2: Run and confirm the integration test fails**

```bash
python3 -m unittest discover -s SIAB/tests -p 'test_main_sternheimer.py' -v
```

- [ ] **Step 3: Load and pass the ST data exactly once**

In `main.py`, after legacy Q/S/V reads:

```python
sternheimer_data = None
if "sternheimer" in file_list:
    if len(file_list["sternheimer"]) != 1:
        raise ValueError("the first SIAB Sternheimer implementation requires exactly one data file")
    sternheimer_data = IO.read_sternheimer.read_sternheimer(file_list["sternheimer"][0])
```

Add `set_sternheimer_data(sternheimer_data)` to `Opt_Orbital_Converge`. Require ST data for `st_only` and `st_constrained`; reject ST data with a legacy mode so it cannot be silently unused.

- [ ] **Step 4: Extend result metadata without changing coefficient parsing**

Change `write_C` to accept optional `loss_components=None` and `mode=None`. Write new lines only inside `<Mkb>` after the existing `Left spillage` line. Keep `<Coefficient>` byte-compatible so `read_C_init` remains valid.

- [ ] **Step 5: Run all tests and commit the integration**

```bash
python3 -m unittest discover -s SIAB/tests -v
python3 -m compileall -q SIAB
git add SIAB/opt_orb_pytorch_dpsi/main.py \
        SIAB/opt_orb_pytorch_dpsi/main_each.py \
        SIAB/opt_orb_pytorch_dpsi/IO/func_C.py \
        SIAB/tests/test_main_sternheimer.py
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'feat(siab): integrate Sternheimer optimization mode'
```

### Task 6: Add Deterministic H-TZDP Examples and Smoke Test

**Files:**
- Create: `SIAB/example_H_sternheimer/INPUT.st_only`
- Create: `SIAB/example_H_sternheimer/INPUT.st_constrained`
- Create: `SIAB/example_H_sternheimer/README.md`
- Create: `SIAB/tests/test_h_sternheimer_smoke.py`
- Modify: `SIAB/README.md`

- [ ] **Step 1: Add two inputs sharing the same initial basis and data**

Both inputs must contain:

```json
"freeze_orbitals": [{"element": "H", "l": 0, "zeta": 1}],
"C_init_info": {
  "init_from_file": true,
  "C_init_file": "../../Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/ORBITAL_RESULTS.txt",
  "opt_C_read": false
},
"loss": {
  "mode": "st_only",
  "epsilon": 1e-14,
  "condition_limit": 1e12,
  "tau_dft": 0.05,
  "tau_dpsi": 0.10,
  "constraint_penalty_dft": 10.0,
  "constraint_penalty_dpsi": 10.0
}
```

Both inputs use these exact data locations relative to `SIAB/example_H_sternheimer`:

```json
"file_list": {
  "origin": [
    "data/OUT.H-STRU2-8-0.7/orb_matrix.0.dat",
    "data/OUT.H-STRU2-8-0.9/orb_matrix.0.dat",
    "data/OUT.H-STRU2-8-1.3/orb_matrix.0.dat"
  ],
  "linear": [[
    "data/OUT.H-STRU2-8-0.7/orb_matrix.1.dat",
    "data/OUT.H-STRU2-8-0.9/orb_matrix.1.dat",
    "data/OUT.H-STRU2-8-1.3/orb_matrix.1.dat"
  ]],
  "sternheimer": ["data/OUT.H-atom-ST/sternheimer_matrix.dat"]
}
```

The constrained file changes only `mode` to `st_constrained`. Both use H `Nu=[3,2]`, `Rcut=8`, `dr=0.01`, `Ecut=100`, Adam `lr=0.003`, `max_steps=3000`, and the same producer data paths. Set and print a fixed NumPy/Torch seed `20260718`.

- [ ] **Step 2: Add a deterministic ten-step synthetic smoke test**

The test runs both modes twice from identical `C0` and asserts:

```python
torch.testing.assert_close(run1.final_c, run2.final_c, rtol=0.0, atol=0.0)
self.assertLess(run1.final_st, run1.initial_st)
self.assertTrue(torch.equal(run1.initial_1s, run1.final_1s))
self.assertLessEqual(constrained.final_dft / constrained.baseline_dft, 1.05 + 1e-12)
self.assertLessEqual(constrained.final_dpsi / constrained.baseline_dpsi, 1.10 + 1e-12)
```

- [ ] **Step 3: Document provenance and run commands**

The README must state that H atom ST is the only new supervision, the constrained lane reuses historical H dimer/trimer DFT+dpsi data, H2 RPA is held out, and the result cannot be called validated until the ABACUS producer hashes and H/H2 table are populated.

- [ ] **Step 4: Run all tests and commit the example stage**

```bash
python3 -m unittest discover -s SIAB/tests -v
python3 -m compileall -q SIAB
git add SIAB/example_H_sternheimer SIAB/tests/test_h_sternheimer_smoke.py SIAB/README.md
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'test(siab): add H Sternheimer optimization smoke'
```

Expected: all unit tests pass and two deterministic runs are bitwise identical on the same host/PyTorch build.

## SIAB Completion Gate

Before starting production optimization:

```bash
git status --short
git log --format='%h %s' daffb2ce..HEAD
python3 -m unittest discover -s SIAB/tests -v
python3 -m compileall -q SIAB
```

Required evidence:

1. clean worktree;
2. one focused commit for each stage;
3. parser rejects malformed dimensions/provenance;
4. finite-difference ST gradient passes;
5. old loss remains exactly `0.4` in the pinned fixture;
6. fixed H `1s` remains bitwise identical;
7. constrained synthetic run satisfies `1.05` and `1.10` ratios.
8. the first H experiment performs no PCA or reference-row truncation.
