# SIAB `pi_dpsi_joint` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a source-aware `pi_dpsi_joint` SIAB mode that minimizes the full-Coulomb projected response error for equal-weight H and H2 families while keeping the fixed-DZP, DFT, and legacy dpsi constraints, then validate the resulting `3s2p` basis with an independent SOS/CP calculation.

**Architecture:** Reuse the validated source-v1/response-v1 readers, strict pairing layer, and differentiable projected-Pi evaluator. Extend target entries with explicit source and zero-order-audit paths, load one strict pair per physical family, and route only `pi_dpsi_joint` stages to a projected-Pi optimization adapter. Existing Sternheimer spillage modes and output schemas remain unchanged; the new mode gets explicit projected-Pi diagnostics and cannot consume SOS energies or ghost targets during training.

**Tech Stack:** Python 3.10, PyTorch 2.1 complex128/float64, `unittest`, ABACUS source-v1/response-v1, GreenX frequency weights, SLURM `normal` on `df_dcu`, ABACUS plus LibRPA for held-out SOS/CP.

---

## Frozen Physical Definition And Stop Rules

For physical family `X` in `{H,H2}`, primitive functions `B_e`, occupied state
`i`, whitened full-Coulomb auxiliary channels `a,b`, and frequency `omega_j`,
the paired files define

```text
D_i[a,e]     = <s_ia|B_e>
Q_ji[b,e]    = <delta_psi_ib(i omega_j)|B_e>
S[e,e']      = <B_e|B_e'>
Phi_alpha    = sum_e B_e C[e,alpha]
G_C          = C^H S C
```

The candidate and primitive-reference responses are

```text
A_Xj(C) = sum_i f_i (D_i C) G_C^-1 (Q_ji C)^H
Pi_Xj(C) = A_Xj(C) + A_Xj(C)^H
A_Xj(B) = sum_i f_i D_i S^+ Q_ji^H
Pi_Xj(B) = A_Xj(B) + A_Xj(B)^H
```

`f_i` appears exactly once. `A+A^H` supplies the complex-conjugate term; no
extra factor of two is allowed. The auxiliary channels are already whitened
with the same full-Coulomb kernel, so neither expression receives another
`V`, `V^(1/2)`, or `V^(-1/2)` factor.

The family and training losses are

```text
L_X(C) = sum_j w_j ||Pi_Xj(C)-Pi_Xj(B)||_F^2
         / sum_j w_j ||Pi_Xj(B)||_F^2

L_Pi(C) = L_H(C) + L_H2(C)

L_total(C) = L_Pi(C)
             + lambda_dpsi L_dpsi(C)/L_dpsi(C0)
             + lambda_DFT [max(0,L_DFT(C)/L_DFT(C0)-1-tau_DFT)]^2
             + lambda_dpsi_gate [max(0,L_dpsi(C)/L_dpsi(C0)-1-tau_dpsi)]^2
```

Use the validated defaults `lambda_dpsi=1`, `lambda_DFT=10`,
`lambda_dpsi_gate=10`, `tau_DFT=0.05`, and `tau_dpsi=0.10`. Keep radial-tail
and old Sternheimer low-frequency penalties at zero: the source-aware metric
already measures all 16 response matrices, and diffuse response is physical
information rather than an error to suppress.

The first production run keeps H `3s2p`, 25 radial primitives, and 8-bohr
cutoff. It fixes `1s,2s,1p` exactly and starts from the current best
fixed-DZP joint coefficients. Training must stop without a held-out run if
any of these fail:

```text
strict source/response pairing and zero-order audits pass
fixed 1s,2s,1p coefficient difference <= 1e-12
finite projected-Pi loss and gradients
candidate overlap condition <= 1e12
final L_Pi < initial L_Pi
DFT ratio <= 1.05
dpsi ratio <= 1.10
```

SOS/CP is never an optimizer input. After the training gates pass, the new
basis must use the same H2/H/H+ghost, full-Coulomb, explicit 214-ABS-per-H,
20-Angstrom, 100-Ry, 16-frequency, all-band contract as jobs `21438483` and
`21440627`. The physical promotion gate is

```text
D_CP(new) > 105.853882 kcal/mol
BSSE(new) <= 1.082171 kcal/mol
```

where the BSSE limit allows at most `0.05 kcal/mol` regression from the
current joint value `1.032171`. Passing this gate means the new direction is
useful; it does not mean the final `0.1 kcal/mol` accuracy target has been met.

## File Map

Create:

- `SIAB/opt_orb_pytorch_dpsi/zero_order_audit.py`: immutable validated audit model.
- `SIAB/opt_orb_pytorch_dpsi/IO/read_zero_order_audit.py`: reusable production audit reader.
- `SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py`: one-pair-per-family optimizer adapter.
- `SIAB/tests/test_read_zero_order_audit.py`: audit schema and tolerance tests.
- `SIAB/tests/test_projected_pi_optimization.py`: adapter value, gradient, and family tests.
- `SIAB/example_H_sternheimer/projected_pi_loss/INPUT.pi_dpsi_joint`: frozen first-run input.
- `SIAB/example_H_sternheimer/projected_pi_loss/run_pi_dpsi_joint.slurm`: full-resource optimizer runner.
- `SIAB/example_H_sternheimer/projected_pi_loss/run_pi_dpsi_joint_sos.slurm`: held-out H2/H/H+ghost runner.

Modify:

- `SIAB/opt_orb_pytorch_dpsi/sternheimer_targets.py`: optional `source_path` and `zero_order_audit_path` fields.
- `SIAB/opt_orb_pytorch_dpsi/main.py`: objective-specific loading and strict pairing.
- `SIAB/opt_orb_pytorch_dpsi/optimization_loss.py`: add `pi_dpsi_joint` without changing old modes.
- `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py`: explicit projected-Pi logging and selection.
- `SIAB/tests/test_sternheimer_targets.py`: new-field compatibility and rejection tests.
- `SIAB/tests/test_main_sternheimer.py`: route and producer-audit tests.
- `SIAB/tests/test_loss_and_freeze.py`: loss composition and optimizer integration tests.
- `SIAB/tests/test_projected_pi_analysis.py`: consume the shared audit reader.
- `SIAB/example_H_sternheimer/projected_pi_loss/README.md`: commands and final gate result.
- `sternheimer_siab_project/main.tex`: implementation, training, and held-out physics result.

Do not modify ABACUS source files, the source-v1/response-v1 formats, the
existing `st_only`, `st_constrained`, or `st_dpsi_joint` numerical paths, or
the existing SOS/CP reference outputs.

### Task 1: Extract The Zero-Order Audit Reader

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/zero_order_audit.py`
- Create: `SIAB/opt_orb_pytorch_dpsi/IO/read_zero_order_audit.py`
- Create: `SIAB/tests/test_read_zero_order_audit.py`
- Modify: `SIAB/example_H_sternheimer/projected_pi_loss/analyze_projected_pi.py`
- Modify: `SIAB/tests/test_projected_pi_analysis.py`

- [ ] **Step 1: Write the failing shared-reader test**

Build one valid `sternheimer_siab_zero_order_identity_v1` fixture and assert
the returned frozen object exposes `case`, `occupied_state_count`, exact grid,
three maximum differences, source file hashes, and `passed=True`. Add separate
failures for a false check, wrong case, loose threshold, missing SHA256,
nonidentical charge/wavefunction grids, and non-finite differences.

- [ ] **Step 2: Verify RED**

```bash
cd SIAB/tests
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_read_zero_order_audit
```

Expected: import failure for `IO.read_zero_order_audit`.

- [ ] **Step 3: Implement the immutable model and reader**

Use a frozen dataclass with only validated values. The public function is:

```python
def read_zero_order_audit(path, expected_case):
    """Return ZeroOrderAudit or raise ValueError before target loading."""
```

Enforce occupation `<=1e-14`, occupied eigenvalue and total-energy differences
`<=1e-12 Ha`, every existing boolean check, positive state count, exact grids,
and 64-character lowercase SHA256 values.

- [ ] **Step 4: Replace the analysis-private parser**

Delete `_read_zero_order_audit` and its duplicated constants from
`analyze_projected_pi.py`. Import the shared reader and serialize its
`passed` field as the existing JSON status. Do not change the Task 5 output
schema or numerical values.

- [ ] **Step 5: Pass audit and analysis tests**

```bash
cd SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_read_zero_order_audit test_projected_pi_analysis
```

Expected: all tests pass and the synthetic failed ranking still exits `2`.

- [ ] **Step 6: Commit**

```bash
git add SIAB/opt_orb_pytorch_dpsi/zero_order_audit.py \
        SIAB/opt_orb_pytorch_dpsi/IO/read_zero_order_audit.py \
        SIAB/tests/test_read_zero_order_audit.py \
        SIAB/example_H_sternheimer/projected_pi_loss/analyze_projected_pi.py \
        SIAB/tests/test_projected_pi_analysis.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'refactor(siab): share zero-order audit validation'
```

### Task 2: Extend Physical Target Entries With Source Provenance

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/sternheimer_targets.py`
- Modify: `SIAB/tests/test_sternheimer_targets.py`

- [ ] **Step 1: Write failing parser tests**

Require this entry to parse with both paths retained as `Path` objects:

```json
{
  "path": "H/sternheimer_matrix.dat",
  "source_path": "H/STERNHEIMER_SIAB_SOURCE_V1.dat",
  "zero_order_audit_path": "H_zero_order_identity.json",
  "family": "H",
  "role": "physical"
}
```

Prove legacy path-only and existing named entries remain bitwise equal. Reject
empty new paths, energy fields, and either source/audit field on a ghost entry.

- [ ] **Step 2: Verify RED**

```bash
cd SIAB/tests
python -m unittest -v test_sternheimer_targets
```

Expected: new fields are currently unknown.

- [ ] **Step 3: Extend the frozen entry**

Add `source_path: Path | None` and `zero_order_audit_path: Path | None`, both
defaulting to `None`. Keep the existing duplicate key check, now including the
new paths. Do not make these fields mandatory in the parser because old
Sternheimer modes remain valid without them.

- [ ] **Step 4: Pass target tests and commit**

```bash
cd SIAB/tests
python -m unittest -v test_sternheimer_targets
```

```bash
git add SIAB/opt_orb_pytorch_dpsi/sternheimer_targets.py \
        SIAB/tests/test_sternheimer_targets.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): attach source provenance to response targets'
```

### Task 3: Add The Projected-Pi Optimization Adapter

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py`
- Create: `SIAB/tests/test_projected_pi_optimization.py`

- [ ] **Step 1: Write a failing two-family value test**

Use the existing synthetic complex H/H2 pairs. Construct
`NormalizedPhysicalFamilyProjectedPiOptimization(("H",pair_h),("H2",pair_h2))`
at rank tolerance `1e-12`. Assert its result contains scalar `loss`,
`max_condition`, common `frequency_ha`, equal-family `frequency_loss`, and
per-family results. Assert `loss=L_H+L_H2` and frequency diagnostics are the
arithmetic mean of the two normalized family losses.

- [ ] **Step 2: Write the failing gradient and contract tests**

Compare one coefficient derivative with centered finite difference. Reject
duplicate family names, unequal frequency grids or weights, more than one pair
in a family, a ghost family, non-finite loss, and condition above the supplied
limit.

- [ ] **Step 3: Verify RED**

```bash
cd SIAB/tests
python -m unittest -v test_projected_pi_optimization
```

Expected: adapter module is absent.

- [ ] **Step 4: Implement the adapter without changing the algebra**

Wrap `NormalizedPhysicalFamilyProjectedPi`. Return a frozen result with:

```python
@dataclass(frozen=True)
class ProjectedPiOptimizationResult:
    loss: torch.Tensor
    max_condition: float
    frequency_ha: torch.Tensor
    frequency_loss: torch.Tensor
    family_results: dict

    @property
    def lowest_frequency_ha(self):
        return self.frequency_ha[0]

    @property
    def lowest_frequency_loss(self):
        return self.frequency_loss[0]
```

Do not detach `loss` or `frequency_loss`. Preserve complex128/float64 CPU
calculation and the existing `A+A^H` implementation.

- [ ] **Step 5: Pass algebra, adapter, and finite-difference tests**

```bash
cd SIAB/tests
python -m unittest -v test_projected_pi test_projected_pi_optimization
```

- [ ] **Step 6: Commit**

```bash
git add SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py \
        SIAB/tests/test_projected_pi_optimization.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): adapt projected Pi for optimization'
```

### Task 4: Route `pi_dpsi_joint` Without Changing Legacy Modes

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/main.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/optimization_loss.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py`
- Modify: `SIAB/tests/test_main_sternheimer.py`
- Modify: `SIAB/tests/test_loss_and_freeze.py`

- [ ] **Step 1: Add failing configuration tests**

Assert `normalize_loss_config` accepts `mode="pi_dpsi_joint"` and
`projected_pi_rank_tolerance=1e-12`, rejects tolerance outside `(0,1)`, and
preserves the exact normalized dictionaries for all three old modes. Reject
nonzero old low-frequency guard in the new mode.

- [ ] **Step 2: Add failing routing tests**

For `pi_dpsi_joint`, require exactly one physical response/source/audit triple
per family, all audits passing, all strict pairs valid, origin data present,
and linear dpsi data present. Reject ghost targets, mixed source-aware and
source-free physical entries, duplicate families, and mixed projected-Pi and
legacy Sternheimer stages in one input.

- [ ] **Step 3: Add the loss mode**

Extend `_LOSS_MODES` with `pi_dpsi_joint` and add default
`projected_pi_rank_tolerance=1e-12`. In `compose_loss`, use the supplied primary
scalar exactly where `st` is used today, but expose it as `projected_pi` for
the new mode. Keep the same dpsi regularization and DFT/dpsi hinges as
`st_dpsi_joint`. `selection_component("pi_dpsi_joint")` returns `total`.

- [ ] **Step 4: Load the source-aware evaluator in `main.py`**

Read each response and source, apply the same element aliases to both, call
`pair_response_and_source`, and validate the named zero-order audit before
constructing the adapter. Require one pair each for `H` and `H2` in the first
production input; report any allowed execution-provenance warning explicitly.

- [ ] **Step 5: Add objective-specific convergence logging**

Add a `set_projected_pi_objective` method. The new stage must write columns
named `projected_pi`, `projected_pi_lowest_frequency`, and
`max_projected_pi_condition`; old stage headers remain byte-for-byte unchanged.
Candidate acceptance requires the existing DFT/dpsi and condition gates.
The final metadata records family losses, all 16 frequency losses, rank
tolerance, retained primitive ranks, source/response/audit hashes, and no SOS
or ghost input.

- [ ] **Step 6: Pass focused integration tests**

```bash
cd SIAB/tests
python -m unittest -v \
  test_main_sternheimer test_loss_and_freeze \
  test_projected_pi_optimization test_sternheimer_targets
```

- [ ] **Step 7: Commit**

```bash
git add SIAB/opt_orb_pytorch_dpsi/main.py \
        SIAB/opt_orb_pytorch_dpsi/optimization_loss.py \
        SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py \
        SIAB/tests/test_main_sternheimer.py \
        SIAB/tests/test_loss_and_freeze.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): optimize source-aware projected Pi'
```

### Task 5: Freeze A Reproducible H/H2 Training Campaign

**Files:**
- Create: `SIAB/example_H_sternheimer/projected_pi_loss/INPUT.pi_dpsi_joint`
- Create: `SIAB/example_H_sternheimer/projected_pi_loss/run_pi_dpsi_joint.slurm`
- Modify: `SIAB/tests/test_h_sternheimer_smoke.py`
- Modify: `SIAB/example_H_sternheimer/projected_pi_loss/README.md`

- [ ] **Step 1: Write the failing campaign-contract test**

Require `normal`, one node, 30 CPUs, 110610M, 24 h, fixed Python runtime,
isolated Matplotlib path, immutable source commit, and SHA256 preflight. Parse
the input and require H/H2 physical families, paired paths, audits,
`pi_dpsi_joint`, fixed `1s,2s,1p`, H `Nu=[3,2,0,0,0]`, 25 primitives, 8-bohr
cutoff, rank tolerance `1e-12`, and no ghost/SOS fields.

- [ ] **Step 2: Verify RED and add the files**

Start from `fixed_dzp_joint_ORBITAL_RESULTS.txt`, use the exact paired producer
under `siab_projected_pi_paired_h_h2_20260731`, and use the two validated
zero-order audits. Set a fixed seed and preserve the current Adam stage
parameters except for the new mode.

- [ ] **Step 3: Run one-step and short deterministic smoke tests**

Use one optimizer step to prove the gradient reaches only `3s,2p`. Run the
same 10-step input twice and require byte-identical coefficients, losses, and
metadata. The fixed columns must be exactly unchanged.

- [ ] **Step 4: Pass the complete SIAB Python suite on `df_dcu`**

```bash
cd ABACUS-orbitals
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest discover \
  -s SIAB/tests -p 'test_*.py' -v
```

Record the exact test count, wall time, and maximum RSS.

- [ ] **Step 5: Commit**

```bash
git add SIAB/example_H_sternheimer/projected_pi_loss/INPUT.pi_dpsi_joint \
        SIAB/example_H_sternheimer/projected_pi_loss/run_pi_dpsi_joint.slurm \
        SIAB/example_H_sternheimer/projected_pi_loss/README.md \
        SIAB/tests/test_h_sternheimer_smoke.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'test(siab): freeze projected-Pi H training campaign'
```

### Task 6: Run Training And The Independent SOS/CP Gate

**Files:**
- Create remotely: `/work1/ghj/sternheimer_abacus_tests/siab_pi_dpsi_joint_h_20260801/`
- Create: `SIAB/example_H_sternheimer/projected_pi_loss/run_pi_dpsi_joint_sos.slurm`
- Modify: `SIAB/example_H_sternheimer/projected_pi_loss/README.md`
- Modify: `sternheimer_siab_project/main.tex`

- [ ] **Step 1: Submit the optimizer only on `normal`**

Use one node, 30 threads, 110610M, and 24 h. Preserve source commit, input
hashes, Python/PyTorch versions, node, `/usr/bin/time -v`, accepted/rejected
candidate counts, and the full frequency diagnostics. Do not submit to
`debug` and do not overwrite an existing result directory.

- [ ] **Step 2: Enforce the training stop rules**

Run a separate read-only validator over the final coefficient file and JSON
metadata. If any fixed-column, loss, DFT/dpsi, condition, rank, provenance, or
determinism gate fails, document the failure and do not stage SOS.

- [ ] **Step 3: Run the held-out SOS/CP array**

Stage H2, H, and H+ghost with the same explicit ABS and full-Coulomb files as
the current joint control. Use all `18/9/18` bands, 20 Angstrom, 0.74085
Angstrom, 100 Ry, 16 minimax frequencies, and
`rpa_ccp_rmesh_times=5`. Require unique ABACUS/LibRPA completion markers and
byte-identical non-orbital physical inputs.

- [ ] **Step 4: Apply the physical promotion gate**

Compute `D_raw`, `D_CP`, total BSSE, zero-order CP contribution, and RPAc CP
contribution with the existing tested parser. Compare against initial TZDP,
current joint, guarded, and Delta-ST/FHI-aims references. Promote the basis
only if the two physical inequalities in the frozen stop rules pass.

- [ ] **Step 5: Update and visually verify the TeX report**

Record formulas, source hashes, optimizer commit, resources, all training
gates, projected-Pi curves, SOS/CP table, and the explicit pass/fail decision.
Compile `main.pdf`, verify inserted numbers with `pdftotext`, render affected
pages, and inspect every equation, table, caption, and legend for overflow.

- [ ] **Step 6: Commit the physical result**

```bash
git add SIAB/example_H_sternheimer/projected_pi_loss \
        SIAB/example_H_sternheimer/fixed_dzp_tzdp_sos/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'analysis(siab): validate pi-dpsi joint basis'
git log -1 --format='%h%nAuthor: %an <%ae>%nCommitter: %cn <%ce>%n%s'
```

The final report must distinguish three gates: software tests, optimizer
training constraints, and independent SOS/CP physics. Passing either of the
first two is not a physical basis validation.
