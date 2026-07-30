# SIAB Low-Frequency Sternheimer Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current integrated Sternheimer objective while reporting every frequency-local loss and preventing selection of a fixed-DZP `3s2p` basis that regresses at the lowest imaginary frequency.

**Architecture:** `SternheimerSpillage` owns frequency grouping and differentiable local losses. `optimization_loss` owns the optional hinge penalty, while `Opt_Orbital_Converge` captures the initial low-frequency baseline, applies the hard acceptance gate, and persists diagnostics. Existing inputs keep a zero guard weight and therefore retain their previous optimization behavior.

**Tech Stack:** Python 3, PyTorch 2.1, SIAB JSON inputs, `unittest`, SLURM `normal` on `df_dcu`, ABACUS/LibRPA held-out SOS-RPA workflow.

---

## File Map

Modify:

- `SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py`: frequency-resolved result tensors and evaluator grouping.
- `SIAB/opt_orb_pytorch_dpsi/optimization_loss.py`: guard defaults, validation, penalty, and feasibility helper.
- `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py`: baseline capture, hard gate, violation report, log columns, and saved diagnostics.
- `SIAB/opt_orb_pytorch_dpsi/IO/func_C.py`: optional guarded result-file labels while preserving the exact zero-guard schema.
- `SIAB/tests/test_sternheimer_spillage.py`: direct two-frequency formulas, ordering, and complex phase invariance.
- `SIAB/tests/test_loss_and_freeze.py`: loss composition, default compatibility, hard acceptance, and output schema.
- `SIAB/tests/test_main_sternheimer.py`: persisted metadata contract.
- `SIAB/tests/test_h_sternheimer_smoke.py`: example-input defaults and fixed-orbital regression.
- `SIAB/example_H_sternheimer/INPUT.st_dpsi_joint_low_frequency_guard`: guarded fixed-DZP input.
- `SIAB/example_H_sternheimer/run_joint_low_frequency_guard.slurm`: one-node optimizer runner with strict provenance.
- `SIAB/example_H_sternheimer/fixed_dzp_tzdp_sos/README.md`: A/B commands and acceptance thresholds.
- `sternheimer_siab_project/main.tex`: verified implementation and physical A/B result after computation finishes.

No ABACUS or LibRPA source file changes in this plan.

### Task 1: Add Frequency-Resolved Sternheimer Results

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py`
- Test: `SIAB/tests/test_sternheimer_spillage.py`

- [ ] **Step 1: Write a failing two-frequency formula test**

Extend `make_sternheimer_data` with an optional `frequency_ha` argument and add a test whose rows are deliberately interleaved:

```python
data = make_sternheimer_data(
    [h_s_block(3)],
    [[0.8, 0.0, 0.0], [0.0, 0.5, 0.0],
     [0.6j, 0.0, 0.0], [0.0, -0.2j, 0.0]],
    norm=[1.0, 2.0, 1.5, 0.5],
    occupation=[1.0, 2.0, 1.0, 2.0],
    frequency_weight=[0.25, 0.75, 0.25, 0.75],
    frequency_ha=[0.1, 0.4, 0.1, 0.4],
)
result = SternheimerSpillage(
    data,
    {"H": [coefficient]},
    [OrbitalColumn("H", 0, 0, 0, 1)],
).evaluate({"H": [coefficient]})

torch.testing.assert_close(
    result.frequency_ha,
    torch.tensor([0.1, 0.4], dtype=torch.float64),
)
torch.testing.assert_close(result.frequency_norm, expected_norm)
torch.testing.assert_close(result.frequency_residual, expected_residual)
torch.testing.assert_close(
    result.frequency_loss, expected_residual / expected_norm
)
```

Compute `expected_norm` and `expected_residual` directly from the fixed-space and combined-space projectors already used by the surrounding tests. Also assert that the original scalar loss equals

```python
torch.sum(weight_by_frequency * expected_residual) / \
torch.sum(weight_by_frequency * expected_norm)
```

so the implementation cannot silently replace the current objective.

- [ ] **Step 2: Run the focused test on `df_dcu` and verify RED**

Run with the server PyTorch module:

```bash
source /etc/profile.d/modules.sh
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load apps/PyTorch/2.1.0/pytorch-2.1.0-dtk2310
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest -v \
  test_sternheimer_spillage.SternheimerSpillageTest.test_reports_frequency_local_losses
```

Expected: failure because `SternheimerLossResult` has no frequency fields.

- [ ] **Step 3: Implement sorted differentiable frequency grouping**

Extend the result type:

```python
@dataclass(frozen=True)
class SternheimerLossResult:
    loss: torch.Tensor
    weighted_residual: torch.Tensor
    weighted_norm: torch.Tensor
    max_condition: float
    frequency_ha: torch.Tensor
    frequency_residual: torch.Tensor
    frequency_norm: torch.Tensor
    frequency_loss: torch.Tensor

    @property
    def lowest_frequency_ha(self):
        positive = torch.nonzero(self.frequency_ha > 0.0, as_tuple=False)
        if positive.numel() == 0:
            raise RuntimeError("low-frequency guard requires a positive frequency")
        return self.frequency_ha[positive[0, 0]]

    @property
    def lowest_frequency_loss(self):
        positive = torch.nonzero(self.frequency_ha > 0.0, as_tuple=False)
        if positive.numel() == 0:
            raise RuntimeError("low-frequency guard requires a positive frequency")
        return self.frequency_loss[positive[0, 0]]
```

Add one pure helper:

```python
def _frequency_resolved_loss(data, norm, residual):
    frequencies, inverse = torch.unique(
        data.frequency_ha, sorted=True, return_inverse=True
    )
    occupation = data.occupation
    by_frequency_norm = torch.zeros_like(frequencies).scatter_add_(
        0, inverse, occupation * norm
    )
    by_frequency_residual = torch.zeros_like(frequencies).scatter_add_(
        0, inverse, occupation * residual
    )
    if not bool(torch.all(by_frequency_norm > 0.0)):
        raise RuntimeError("frequency-resolved projected norm must be positive")
    return (
        frequencies,
        by_frequency_residual,
        by_frequency_norm,
        by_frequency_residual / by_frequency_norm,
    )
```

Use it in all three public result-producing paths so every
`SternheimerLossResult` is complete. Keep `data.effective_weight` for the
existing scalar integrated loss.

- [ ] **Step 4: Add phase and gradient regressions**

Add tests that multiply all rows at one frequency by `exp(0.37j)` and assert
all frequency fields are unchanged to `1e-14`. Backpropagate
`result.lowest_frequency_loss` and compare one coefficient derivative against
a centered finite difference with the existing `1e-6` step.

- [ ] **Step 5: Run the focused module and commit**

```bash
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest -v test_sternheimer_spillage
```

Expected: all `test_sternheimer_spillage` tests pass.

```bash
git add SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py \
        SIAB/tests/test_sternheimer_spillage.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): report frequency-resolved ST loss'
```

### Task 2: Compose And Validate The Low-Frequency Penalty

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/optimization_loss.py`
- Test: `SIAB/tests/test_loss_and_freeze.py`

- [ ] **Step 1: Write failing configuration and hinge tests**

Assert the normalized defaults contain:

```python
"low_frequency_guard_weight": 0.0,
"low_frequency_guard_tolerance": 0.0,
```

Add a test with current low loss `0.27`, baseline `0.25`, weight `10`, and
tolerance `0.0`. It must produce:

```python
expected = 10.0 * (0.27 / 0.25 - 1.0) ** 2
self.assertAlmostEqual(
    result["regularization_low_frequency"].item(), expected
)
```

and a nonzero finite-difference-matching gradient. At `0.24`, the penalty is
exactly zero. Negative, boolean, NaN, and infinite options must fail.

- [ ] **Step 2: Verify RED on `df_dcu`**

```bash
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest -v \
  test_loss_and_freeze.OptimizationLossTest.test_low_frequency_guard_value_and_gradient
```

Expected: unknown loss configuration key or missing component.

- [ ] **Step 3: Add config defaults and a single feasibility helper**

Add:

```python
def low_frequency_guard_satisfied(current, baseline, config):
    normalized = normalize_loss_config(config)
    _validate_loss_tensor("sternheimer_lowest_frequency", current)
    reference = _baseline_tensor(
        "sternheimer_lowest_frequency", baseline, current
    )
    if normalized["low_frequency_guard_weight"] == 0.0:
        return True
    if not bool(reference > normalized["epsilon"]):
        raise ValueError(
            "baseline sternheimer_lowest_frequency must exceed epsilon "
            "when the low-frequency guard is active"
        )
    limit = (1.0 + normalized["low_frequency_guard_tolerance"]) * reference
    allowance = 1.0e-12 * torch.maximum(limit.abs(), torch.ones_like(limit))
    return bool(current <= limit + allowance)
```

Extend `compose_loss` with keyword-only `st_low_frequency=None`. When the guard
is active, validate it and the baseline, then compute the squared hinge. Add
`sternheimer_lowest_frequency` and `regularization_low_frequency` to the
component dictionary before `total` only in this active path. When inactive,
return the exact previous key set without requiring a low-frequency argument.

- [ ] **Step 4: Prove default numerical compatibility**

Update the existing `st_only` identity test to assert `result["total"] is st`
when the guard and radial locality are both inactive, and assert both guarded
component names are absent. This protects the exact legacy behavior.

- [ ] **Step 5: Run loss tests and commit**

```bash
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest -v test_loss_and_freeze.OptimizationLossTest
```

Expected: all optimization-loss tests pass.

```bash
git add SIAB/opt_orb_pytorch_dpsi/optimization_loss.py \
        SIAB/tests/test_loss_and_freeze.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): add low-frequency ST guard loss'
```

### Task 3: Enforce Candidate Feasibility And Persist Diagnostics

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/IO/func_C.py`
- Test: `SIAB/tests/test_loss_and_freeze.py`
- Test: `SIAB/tests/test_main_sternheimer.py`

- [ ] **Step 1: Write a failing hard-gate regression**

Add a synthetic evaluator whose integrated loss decreases as its single
coefficient changes while `lowest_frequency_loss` increases. Configure a
positive guard weight and assert:

```python
self.assertLess(final_integrated_loss, initial_integrated_loss)
self.assertGreater(final_low_loss, initial_low_loss)
self.assertLessEqual(
    result["sternheimer_lowest_frequency"],
    result["loss_baseline"]["sternheimer_lowest_frequency"],
)
self.assertTrue(all(not row["accepted"] for row in regressed_rows))
```

The selected result must be the best feasible point, not the final or lowest
integrated-loss point.

- [ ] **Step 2: Verify RED**

```bash
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest -v \
  test_loss_and_freeze.ConvergeIntegrationTest.test_low_frequency_guard_rejects_regressed_candidate
```

Expected: candidate remains accepted or baseline field is absent.

- [ ] **Step 3: Capture baseline and extend acceptance**

At stage initialization evaluate `C_initial` once and store:

```python
baseline_st = evaluator.evaluate(C_initial)
loss_baselines[stage_index]["sternheimer_lowest_frequency"] = (
    baseline_st.lowest_frequency_loss.detach().clone()
)
```

Pass `st_result.lowest_frequency_loss` to `compose_loss`. Define
`low_frequency_ok` using `low_frequency_guard_satisfied`, include it in
`accepted`, and add its normalized excess to `violation_key` and the terminal
error report.

When the guard is active, add these `Spillage.dat` columns immediately after
`sternheimer`:

```text
sternheimer_lowest_frequency
regularization_low_frequency
```

Save the selected frequency value and selected local loss in `data_transmit`
only for an active guard. Assert that the complete zero-guard header is
byte-for-byte identical to the pre-feature header.

- [ ] **Step 4: Extend result metadata with one explicit guarded schema**

Split the fixed label definitions into the existing base schema and one guarded
extension. The guarded extension contains the same two loss components and
these diagnostic fields:

```python
("lowest_st_frequency_ha", "Lowest ST frequency (Ha)"),
("initial_lowest_st_loss", "Initial lowest-frequency ST loss"),
("final_lowest_st_loss", "Final lowest-frequency ST loss"),
("low_frequency_guard_tolerance", "Low-frequency guard tolerance"),
("low_frequency_guard_weight", "Low-frequency guard weight"),
```

Always write finite values in the guarded schema. `_validate_loss_metadata`
accepts exactly the base set or exactly the base plus both guarded components;
it rejects a partial guarded set. `_validate_loss_diagnostics` similarly
accepts the existing two-condition set or that set plus all five guarded
diagnostics. A zero-guard run must reproduce the old result text exactly.

- [ ] **Step 5: Run convergence and output tests, then commit**

```bash
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest -v test_loss_and_freeze test_main_sternheimer
```

Expected: both modules pass with the guarded schema and the unchanged base
schema.

```bash
git add SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py \
        SIAB/opt_orb_pytorch_dpsi/IO/func_C.py \
        SIAB/tests/test_loss_and_freeze.py \
        SIAB/tests/test_main_sternheimer.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): reject low-frequency ST regressions'
```

### Task 4: Add The Fixed-DZP Guarded Example

**Files:**
- Create: `SIAB/example_H_sternheimer/INPUT.st_dpsi_joint_low_frequency_guard`
- Create: `SIAB/example_H_sternheimer/run_joint_low_frequency_guard.slurm`
- Modify: `SIAB/tests/test_h_sternheimer_smoke.py`
- Modify: `SIAB/example_H_sternheimer/fixed_dzp_tzdp_sos/README.md`

- [ ] **Step 1: Write failing example-contract tests**

Require the new input to match `INPUT.st_dpsi_joint` except for:

```json
"low_frequency_guard_weight": 10.0,
"low_frequency_guard_tolerance": 0.0
```

The test must assert identical seed, files, `Nu={"H":[3,2]}`, fixed
`1s,2s,1p`, `Rcut=8`, `Ecut=100`, smearing, optimizer, and DFT/dpsi settings.

- [ ] **Step 2: Verify RED and add the example files**

```bash
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest -v \
  test_h_sternheimer_smoke.ExampleInputTest.test_low_frequency_guard_changes_only_guard_options
```

Expected: missing input file.

Copy the existing one-node runner contract but use a separate output directory
and require these lines after optimization:

```bash
grep -q '^Low-frequency guard weight = 1.0000000000e+01$' ORBITAL_RESULTS.txt
grep -q '^Low-frequency guard tolerance = 0.0000000000e+00$' ORBITAL_RESULTS.txt
python - <<'PY'
from pathlib import Path
values = {}
for line in Path("ORBITAL_RESULTS.txt").read_text().splitlines():
    if " = " in line:
        key, value = line.split(" = ", 1)
        try:
            values[key] = float(value)
        except ValueError:
            pass
assert values["Final lowest-frequency ST loss"] \
    <= values["Initial lowest-frequency ST loss"] * (1.0 + 1.0e-12)
PY
```

- [ ] **Step 3: Run example tests and commit**

```bash
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest -v test_h_sternheimer_smoke.ExampleInputTest
```

Expected: all example-input tests pass.

```bash
git add SIAB/example_H_sternheimer/INPUT.st_dpsi_joint_low_frequency_guard \
        SIAB/example_H_sternheimer/run_joint_low_frequency_guard.slurm \
        SIAB/example_H_sternheimer/fixed_dzp_tzdp_sos/README.md \
        SIAB/tests/test_h_sternheimer_smoke.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'test(siab): add guarded fixed-DZP optimization'
```

### Task 5: Run The Full Software Gate On `df_dcu`

**Files:**
- No source edits unless a failing test reveals a defect.

- [ ] **Step 1: Stage an immutable source snapshot**

Create a clean server directory named with the exact source commit, record
`git diff --exit-code`, and generate `SOURCE_MANIFEST.sha256` for all files used
by the optimizer. Do not reuse a directory that another job can overwrite.

- [ ] **Step 2: Run the complete SIAB regression**

```bash
source /etc/profile.d/modules.sh
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load apps/PyTorch/2.1.0/pytorch-2.1.0-dtk2310
cd "$SOURCE_ROOT/SIAB/tests"
python -m unittest discover -v
```

Acceptance: every test passes; report the exact passed count and elapsed time.

- [ ] **Step 3: Run one zero-guard compatibility optimization**

Use the existing synthetic smoke and current fixed-DZP input with guard weight
zero. Require the integrated initial loss to remain `0.4428607140` within
`1e-10` and the current optimized loss to remain `0.4054568603` within
`1e-8` when the same seed and optimizer are used.

- [ ] **Step 4: Record test provenance**

Archive source commit, module versions, Python and PyTorch versions, test log,
input hashes, and output hashes under the immutable campaign directory. Commit
only source/test fixes; do not commit generated orbital or scheduler files.

### Task 6: Run The Guarded Optimization And Held-Out H2 Gate

**Files:**
- Modify after verified results: `SIAB/example_H_sternheimer/fixed_dzp_tzdp_sos/README.md`
- Modify after verified results: `/Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project/main.tex`

- [ ] **Step 1: Submit the guarded optimizer on `normal`**

Use exactly one node, one task, 30 CPUs, `110610M`, and 24 hours:

```text
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH -t 1-00:00:00
```

Do not use `debug`. Record job ID, source commit, input hash, target hash,
initial/final integrated loss, all 16 local losses, DFT/dpsi ratios, fixed-
column differences, and elapsed time.

- [ ] **Step 2: Apply the pre-SOS numerical gates**

Require:

```text
max fixed 1s/2s/1p coefficient difference <= 1e-12
final lowest-frequency loss <= initial lowest-frequency loss * (1 + 1e-12)
final integrated ST loss <= 0.4095114289
DFT ratio <= 1.05
dpsi ratio <= 1.10
same orbital count = 3s2p
same Rcut = 8 bohr
```

If any gate fails, stop before the H2 calculation and report which constraint
was active at the selected point.

- [ ] **Step 3: Run the held-out SOS-RPA comparison**

Use the existing `fixed_dzp_tzdp_sos/run_sos_cp.slurm` workflow with a new
immutable candidate label. Keep H2, H, and H+ghost inputs identical except for
the orbital file. Require the same 20-A box, 0.74085-A bond, full Coulomb,
16-frequency file, ABFS, `exx_pca_threshold`, ABACUS executable, and LibRPA
executable as the control.

- [ ] **Step 4: Compute and report physical differences**

Report for initial TZDP, current fixed-DZP joint, and guarded fixed-DZP joint:

```text
EcRPA(H2), EcRPA(H), EcRPA(H+ghost), raw binding,
CP-corrected binding, RPAc BSSE, distance from Delta-ST/FHI-aims reference
```

Do not call the basis improved unless the guarded CP-corrected binding moves
toward the reference and all pre-SOS gates pass.

- [ ] **Step 5: Update the TeX note and verify the PDF**

Add the formulas, a 16-frequency before/after loss table, optimization
provenance, and held-out binding-energy comparison before the existing H2
conclusion. Mark software gates and physical outcome separately. Then run:

```bash
cd /Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Render and inspect the new pages. Require no new unresolved references,
overfull tables, clipped figures, or equations outside the page.

- [ ] **Step 6: Commit verified results**

Commit the README and TeX/PDF result update only after numerical reproduction
and visual inspection. Use Codex as author and AroundPeking as committer, then
verify `git log -1 --format=fuller` in each repository.
