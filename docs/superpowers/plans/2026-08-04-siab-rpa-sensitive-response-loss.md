# SIAB RPA-Sensitive Response Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (<code>- [ ]</code>) syntax for tracking.

**Goal:** Add a backward-compatible <code>pi_rpa_sensitive_joint</code> SIAB mode, prove that its frequency/channel metric reproduces the five archived H-basis promotion decisions, retrain the fixed-size H <code>3s2p2d1f1g</code> basis, and run the equilibrium all-band full-Coulomb SOS/CP gate.

**Architecture:** Keep <code>pi_dpsi_joint</code> numerically unchanged. Extend the per-family projected-Pi evaluator with an optional RPA-sensitivity calculation, aggregate H and H2 through the frozen fourth-order family norm only in the new mode, and enforce isolated-H plus total-family improvement during checkpoint selection. A separate offline script selects the largest admissible blend from the frozen alpha set before any new orbital is optimized; H+ghost and LibRPA energies remain outside the gradient.

**Tech Stack:** Python 3.10, PyTorch 2.1 complex128/float64, <code>unittest</code>, ABACUS source-v1/response-v1, GreenX minimax weights, Slurm <code>normal</code> on df_dcu, ABACUS plus LibRPA full Coulomb.

---

## Scope and stop rules

This plan implements the method and the equilibrium iteration gate. The
four-bond-length release campaign is a separate follow-up plan because it
requires four new Delta-ST reference calculations and twelve independent
SOS/CP calculations. It may be written only if the equilibrium candidate
passes both strict improvements:

~~~text
D_CP(new) > 107.888474 kcal/mol
BSSE(new) < 1.070332 kcal/mol
~~~

The implementation stops before optimization if the historical metric scan
finds no admissible alpha. It stops before SOS if training fails the
fixed-DZP, finite-gradient, condition-number, DFT/dpsi, isolated-H, or
fourth-order family-improvement gate. Raw binding alone never promotes a
candidate.

The new loss mode is deliberately named <code>pi_rpa_sensitive_joint</code>.
The existing <code>pi_dpsi_joint</code> mode, output schema, and archived
results remain unchanged when the new mode is not selected.

## File map

Create:

- <code>SIAB/example_H_sternheimer/projected_pi_loss/analyze_rpa_sensitive_ranking.py</code>: five-basis alpha scan and decision.
- <code>SIAB/tests/test_rpa_sensitive_ranking.py</code>: synthetic CLI and output tests.
- <code>SIAB/example_H_sternheimer/projected_pi_loss/INPUT.pi_rpa_sensitive_joint_3s2p2d1f1g</code>: fixed-size production input, created only after alpha freezes.
- <code>SIAB/example_H_sternheimer/projected_pi_loss/run_rpa_sensitive_joint.slurm</code>: immutable one-node training runner.
- <code>SIAB/example_H_sternheimer/projected_pi_loss/results/rpa_sensitive_historical_ranking.json</code>: real five-basis decision.
- <code>SIAB/example_H_sternheimer/projected_pi_loss/results/rpa_sensitive_historical_ranking.md</code>: human-readable ranking.

Modify:

- <code>SIAB/opt_orb_pytorch_dpsi/projected_pi.py</code>: optional positive RPA-sensitivity metric and trace-log diagnostics.
- <code>SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py</code>: optional alpha and fourth-order H/H2 aggregation.
- <code>SIAB/opt_orb_pytorch_dpsi/optimization_loss.py</code>: strict new-mode configuration.
- <code>SIAB/opt_orb_pytorch_dpsi/main.py</code>: new-mode routing and metadata.
- <code>SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py</code>: atomic/family baselines, acceptance, logging, and diagnostics.
- <code>SIAB/opt_orb_pytorch_dpsi/IO/func_C.py</code>: new output schema without changing the old schema.
- <code>SIAB/tests/test_projected_pi.py</code>: analytic sensitivity and invalid-spectrum tests.
- <code>SIAB/tests/test_projected_pi_optimization.py</code>: fourth-order aggregation tests.
- <code>SIAB/tests/test_loss_and_freeze.py</code>: configuration, acceptance, and optimizer tests.
- <code>SIAB/tests/test_main_sternheimer.py</code>: loader, routing, metadata, and output tests.
- <code>SIAB/tests/test_h_sternheimer_smoke.py</code>: fixed campaign and Slurm contract.
- <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>: every RED/GREEN and physical gate.
- <code>sternheimer_siab_project/main.tex</code>: the same staged evidence.

Do not modify ABACUS, LibRPA, source-v1/response-v1 formats, legacy SIAB modes,
existing coefficient files, or the PCA-<code>1e-4</code> SOS contract.

## Remote test contract

All tests importing NumPy or PyTorch run on df_dcu. Reuse:

~~~text
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801
test_root=/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804
~~~

Check the ControlMaster before each remote batch:

~~~bash
/Users/ghj/.codex/skills/ghj-otp-ssh-controlmaster/scripts/otp_ssh_cm.sh check
~~~

For every RED/GREEN commit, push the branch and update a detached checkout to
the exact remote head:

~~~bash
cm=/Users/ghj/.codex/skills/ghj-otp-ssh-controlmaster/scripts/otp_ssh_cm.sh
$cm run 'bash -lc '"'"'
set -euo pipefail
root=/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804
if test ! -d "$root/code/.git"; then
  mkdir -p "$root"
  git clone --branch codex/sternheimer-siab-h \
    git@github.com:AroundPeking/ABACUS-orbitals.git "$root/code"
fi
git -C "$root/code" fetch origin codex/sternheimer-siab-h
git -C "$root/code" checkout --detach origin/codex/sternheimer-siab-h
git -C "$root/code" diff --quiet
git -C "$root/code" diff --cached --quiet
'"'"''
~~~

Every Slurm calculation uses <code>normal</code>, one full node, 30 CPUs,
110610 MB, and the 24-hour limit.

### Task 1: Define the per-family RPA sensitivity with RED tests

**Files:**
- Modify: <code>SIAB/tests/test_projected_pi.py</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>
- Modify: <code>sternheimer_siab_project/main.tex</code>

- [ ] **Step 1: Add a safely scaled physical-pair helper**

~~~python
def scaled_pair(pair, scale=1.0e-2):
    source = replace(pair.source, d=pair.source.d * scale)
    response = replace(pair.response, q=pair.response.q * scale)
    return pair_response_and_source(response, source)
~~~

- [ ] **Step 2: Add the analytic sensitivity test**

Call the evaluator with <code>sensitivity_alpha=0.25</code>. Independently
diagonalize each reference Pi, construct
<code>g=abs(1-1/(1-lambda))</code> and <code>W^(1/2)</code>, and assert:

~~~python
result = ProjectedPiEvaluator(
    scaled_pair(self.pair),
    sensitivity_alpha=0.25,
).evaluate(coefficients(self.coefficient))
torch.testing.assert_close(
    result.loss,
    0.25 * result.base_loss + 0.75 * result.sensitivity_loss,
    rtol=1e-13,
    atol=1e-13,
)
torch.testing.assert_close(
    result.frequency_loss,
    0.25 * result.frequency_base_loss
    + 0.75 * result.frequency_sensitivity_loss,
    rtol=1e-13,
    atol=1e-13,
)
~~~

The independent calculation must also compare the complete sensitivity loss,
per-frequency trace-log difference, and positive dielectric minima.

Use a real scalar perturbation of one free coefficient and compare the
autograd derivative against a centered finite difference for both the pure
sensitivity loss and the alpha-blended loss:

~~~python
epsilon = 1.0e-6
autograd_gradient = torch.autograd.grad(result.sensitivity_loss, coefficient)[0]
finite_difference = (
    sensitivity_loss(coefficient + epsilon * direction)
    - sensitivity_loss(coefficient - epsilon * direction)
) / (2.0 * epsilon)
torch.testing.assert_close(
    torch.sum(autograd_gradient * direction),
    finite_difference,
    rtol=2.0e-5,
    atol=2.0e-7,
)
~~~

Repeat the directional check for <code>result.loss</code>. The test must use a
nondegenerate reference spectrum so the eigenderivative is well defined.

- [ ] **Step 3: Add invariance and failure tests**

Prove separately:

~~~text
common source/response phase leaves all losses unchanged
one common auxiliary-channel permutation leaves all losses unchanged
lambda_max(Pi_ref) >= 1-tolerance raises "reference I-Pi is not positive"
lambda_max(Pi_candidate) >= 1-tolerance raises "candidate I-Pi is not positive"
zero max(g) raises "RPA sensitivity is numerically zero"
non-finite Pi raises an explicit finite-value error
~~~

- [ ] **Step 4: Verify RED on df_dcu**

~~~bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_projected_pi
~~~

Expected: feature-specific failures because the evaluator does not accept
<code>sensitivity_alpha</code>. Import or fixture errors are not acceptable.

- [ ] **Step 5: Record and commit RED**

Append the commit, command, failing names, and expected failure to README and
TeX. State that no implementation or physical result exists.

~~~bash
git add SIAB/tests/test_projected_pi.py \
  SIAB/example_H_sternheimer/projected_pi_loss/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'test(siab): define RPA-sensitive projected-Pi metric'
git push origin HEAD:codex/sternheimer-siab-h
~~~

### Task 2: Implement the per-family sensitivity metric

**Files:**
- Modify: <code>SIAB/opt_orb_pytorch_dpsi/projected_pi.py</code>
- Modify: <code>SIAB/tests/test_projected_pi.py</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>

- [ ] **Step 1: Append optional result fields**

Keep old constructors valid:

~~~python
base_loss: torch.Tensor | None = None
sensitivity_loss: torch.Tensor | None = None
frequency_base_loss: torch.Tensor | None = None
frequency_sensitivity_loss: torch.Tensor | None = None
trace_log_difference: torch.Tensor | None = None
minimum_reference_dielectric_eigenvalue: torch.Tensor | None = None
minimum_candidate_dielectric_eigenvalue: torch.Tensor | None = None
sensitivity_alpha: float | None = None
~~~

- [ ] **Step 2: Add one testable helper**

~~~python
def evaluate_rpa_sensitivity(
    reference_pi,
    candidate_pi,
    frequency_weight,
    relative_tolerance,
):
    """Return positive weighted errors and per-frequency trace-log diagnostics."""
~~~

For each frequency:

~~~python
reference_eigenvalue, reference_eigenvector = torch.linalg.eigh(reference)
candidate_eigenvalue = torch.linalg.eigvalsh(candidate)
reference_dielectric = 1.0 - reference_eigenvalue
candidate_dielectric = 1.0 - candidate_eigenvalue
if float(reference_dielectric.min()) <= relative_tolerance:
    raise RuntimeError("reference I-Pi is not positive")
if float(candidate_dielectric.min()) <= relative_tolerance:
    raise RuntimeError("candidate I-Pi is not positive")
g = torch.abs(1.0 - 1.0 / reference_dielectric)
if float(g.max()) <= relative_tolerance:
    raise RuntimeError("RPA sensitivity is numerically zero")
weight_sqrt = (
    reference_eigenvector
    @ torch.diag(torch.sqrt(g / g.max())).to(torch.complex128)
    @ reference_eigenvector.mH
)
~~~

Compute weighted candidate/reference norms and:

~~~python
trace_log_reference = torch.sum(
    torch.log(reference_dielectric) + reference_eigenvalue
)
trace_log_candidate = torch.sum(
    torch.log(candidate_dielectric) + candidate_eigenvalue
)
~~~

Reject non-finite input and a Hermitian error above
<code>10 * tolerance * max(1, norm)</code>; do not symmetrize invalid input.

- [ ] **Step 3: Preserve the old path**

Add <code>sensitivity_alpha=None</code>. When absent, execute the old
equations and return old loss/frequency values unchanged. When present,
validate <code>0 <= alpha <= 1</code> and use:

~~~python
loss = alpha * base_loss + (1.0 - alpha) * sensitivity_loss
frequency_loss = (
    alpha * frequency_base_loss
    + (1.0 - alpha) * frequency_sensitivity_loss
)
~~~

- [ ] **Step 4: Verify GREEN and legacy stability**

~~~bash
python -m unittest -v test_projected_pi test_projected_pi_optimization
~~~

Expected: all pass; the old direct complex formula keeps
<code>rtol=atol=1e-13</code>.

- [ ] **Step 5: Commit GREEN**

~~~bash
git add SIAB/opt_orb_pytorch_dpsi/projected_pi.py \
  SIAB/tests/test_projected_pi.py \
  SIAB/example_H_sternheimer/projected_pi_loss/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): add RPA-sensitive projected-Pi metric'
git push origin HEAD:codex/sternheimer-siab-h
~~~

### Task 3: Define the new mode and fourth-order aggregation with RED tests

**Files:**
- Modify: <code>SIAB/tests/test_projected_pi_optimization.py</code>
- Modify: <code>SIAB/tests/test_loss_and_freeze.py</code>
- Modify: <code>SIAB/tests/test_main_sternheimer.py</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>

- [ ] **Step 1: Define adapter behavior**

~~~python
adapter = NormalizedPhysicalFamilyProjectedPiOptimization(
    ("H", pair_h),
    ("H2", pair_h2),
    sensitivity_alpha=0.25,
    family_power=4,
)
result = adapter.evaluate(coefficients())
expected = (
    result.family_results["H"].loss.pow(4)
    + result.family_results["H2"].loss.pow(4)
).pow(0.25)
torch.testing.assert_close(result.loss, expected)
self.assertEqual(result.sensitivity_alpha, 0.25)
self.assertEqual(result.family_power, 4)
~~~

Also prove the default adapter still returns the old H-plus-H2 sum.

- [ ] **Step 2: Define strict configuration**

Require:

~~~python
{
    "mode": "pi_rpa_sensitive_joint",
    "projected_pi_rank_tolerance": 1.0e-12,
    "projected_pi_sensitivity_alpha": 0.25,
    "joint_dpsi_weight": 0.02,
}
~~~

Reject missing/out-of-range alpha, alpha in any legacy mode, nonzero radial or
low-frequency penalties, and mixing the new mode with another loss stage.

- [ ] **Step 3: Define loader behavior**

The new mode must require exactly H and H2 physical source/response pairs,
reject ghost, require origin and dpsi, pass alpha and <code>family_power=4</code>
to the adapter, and leave old construction unchanged.

- [ ] **Step 4: Verify and commit RED**

~~~bash
python -m unittest -v \
  test_projected_pi_optimization test_loss_and_freeze test_main_sternheimer
~~~

Expected: unknown mode/config and unsupported adapter arguments.

~~~bash
git add SIAB/tests/test_projected_pi_optimization.py \
  SIAB/tests/test_loss_and_freeze.py SIAB/tests/test_main_sternheimer.py \
  SIAB/example_H_sternheimer/projected_pi_loss/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'test(siab): define RPA-sensitive joint mode'
git push origin HEAD:codex/sternheimer-siab-h
~~~

### Task 4: Implement the mode, adapter, and loader

**Files:**
- Modify: <code>SIAB/opt_orb_pytorch_dpsi/projected_pi.py</code>
- Modify: <code>SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py</code>
- Modify: <code>SIAB/opt_orb_pytorch_dpsi/optimization_loss.py</code>
- Modify: <code>SIAB/opt_orb_pytorch_dpsi/main.py</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>

- [ ] **Step 1: Add a mode-specific required alpha**

Add <code>pi_rpa_sensitive_joint</code> to <code>_LOSS_MODES</code>. Allow
rank tolerance in both projected modes. Require alpha only in the new mode:

~~~python
if mode == "pi_rpa_sensitive_joint":
    if "projected_pi_sensitivity_alpha" not in config:
        raise ValueError(
            "pi_rpa_sensitive_joint requires projected_pi_sensitivity_alpha"
        )
    alpha = normalized["projected_pi_sensitivity_alpha"]
    _validate_real("projected_pi_sensitivity_alpha", alpha, 0.0)
    if alpha > 1.0:
        raise ValueError(
            "projected_pi_sensitivity_alpha must not exceed one"
        )
~~~

Treat both projected modes as <code>projected_pi</code> primary and joint dpsi
modes in <code>compose_loss</code> and <code>selection_component</code>.

- [ ] **Step 2: Extend the adapter**

Add optional alpha and family power. Pass alpha to each family evaluator. For
the new mode:

~~~python
family_losses = torch.stack(
    tuple(family.results[name].loss for name in ("H", "H2"))
)
loss = torch.sum(family_losses.pow(family_power)).pow(
    1.0 / family_power
)
~~~

Accept only <code>family_power=4</code> in this implementation. Append optional
alpha and family-power fields to the result dataclass.

- [ ] **Step 3: Route in main**

~~~python
_PROJECTED_PI_MODES = frozenset(
    {"pi_dpsi_joint", "pi_rpa_sensitive_joint"}
)
~~~

Use this set for loading, origin/dpsi validation, adapter construction,
metadata, and output. Require one alpha across all new-mode stages and pass
<code>family_power=4</code>.

- [ ] **Step 4: Verify GREEN**

~~~bash
python -m unittest -v \
  test_projected_pi test_projected_pi_optimization \
  test_loss_and_freeze test_main_sternheimer
~~~

- [ ] **Step 5: Commit GREEN**

~~~bash
git add SIAB/opt_orb_pytorch_dpsi/projected_pi.py \
  SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py \
  SIAB/opt_orb_pytorch_dpsi/optimization_loss.py \
  SIAB/opt_orb_pytorch_dpsi/main.py \
  SIAB/example_H_sternheimer/projected_pi_loss/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): add RPA-sensitive joint loss mode'
git push origin HEAD:codex/sternheimer-siab-h
~~~

### Task 5: Enforce atomic and family improvement at checkpoint selection

**Files:**
- Modify: <code>SIAB/tests/test_loss_and_freeze.py</code>
- Modify: <code>SIAB/tests/test_main_sternheimer.py</code>
- Modify: <code>SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py</code>
- Modify: <code>SIAB/opt_orb_pytorch_dpsi/IO/func_C.py</code>
- Modify: <code>SIAB/opt_orb_pytorch_dpsi/main.py</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>

- [ ] **Step 1: Write RED checkpoint tests**

Use a differentiable fake where total family loss improves while H worsens;
prove rejection. Add a second candidate where both improve; prove selection.
Also prove the initial point is not accepted, fixed columns remain bitwise
equal, and old <code>pi_dpsi_joint</code> acceptance is unchanged.

- [ ] **Step 2: Write RED output tests**

Require new-mode metadata to contain alpha/power, initial/final H and family
losses, per-family base/sensitivity/blend, per-frequency losses and trace-log,
dielectric minima, GreenX frequencies/weights, and max overlap condition.
Keep long arrays out of <code>ORBITAL_RESULTS.txt</code>.

- [ ] **Step 3: Verify RED**

~~~bash
python -m unittest -v test_loss_and_freeze test_main_sternheimer
~~~

Expected: missing response-improvement gates and diagnostics.

- [ ] **Step 4: Implement strict baselines and acceptance**

At setup:

~~~python
loss_baseline["projected_pi_family"] = baseline_st.loss.detach().clone()
loss_baseline["projected_pi_h"] = (
    baseline_st.family_results["H"].loss.detach().clone()
)
~~~

At candidate evaluation:

~~~python
family_improved = st_result.loss < loss_baseline["projected_pi_family"]
atom_improved = (
    st_result.family_results["H"].loss
    < loss_baseline["projected_pi_h"]
)
accepted = (
    constraints_ok
    and condition_ok
    and locality_condition_ok
    and low_frequency_ok
    and (
        not rpa_sensitive_mode
        or (family_improved and atom_improved)
    )
)
~~~

Add normalized atom/family violations to <code>best_violation</code> and the
fatal error. Add compact H/H2/base/sensitivity columns to new-mode
<code>Spillage.dat</code>.

- [ ] **Step 5: Extend final schemas**

Add a separate new-mode diagnostic schema in <code>IO/func_C.py</code>.
Serialize complete arrays only in <code>PROJECTED_PI_METADATA.json</code>.
Do not alter the old schema.

- [ ] **Step 6: Verify GREEN and full suite**

~~~bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest discover -v
~~~

Record the exact count.

- [ ] **Step 7: Commit GREEN**

~~~bash
git add SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py \
  SIAB/opt_orb_pytorch_dpsi/IO/func_C.py \
  SIAB/opt_orb_pytorch_dpsi/main.py \
  SIAB/tests/test_loss_and_freeze.py SIAB/tests/test_main_sternheimer.py \
  SIAB/example_H_sternheimer/projected_pi_loss/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): gate RPA-sensitive checkpoints on atomic response'
git push origin HEAD:codex/sternheimer-siab-h
~~~

### Task 6: Build and test the five-basis historical ranking tool

**Files:**
- Create: <code>SIAB/example_H_sternheimer/projected_pi_loss/analyze_rpa_sensitive_ranking.py</code>
- Create: <code>SIAB/tests/test_rpa_sensitive_ranking.py</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>

- [ ] **Step 1: Write the failing CLI test**

The script accepts H/H2 response, source, and audit files plus exactly five
coefficient files. Freeze:

~~~python
BASIS_NU = {
    "two_d": (3, 2, 2, 0, 0),
    "first_f": (3, 2, 2, 1, 0),
    "first_g": (3, 2, 2, 1, 1),
    "second_f": (3, 2, 2, 2, 1),
    "second_g": (3, 2, 2, 1, 2),
}
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)
~~~

The synthetic pass fixture must produce multiple admissible alphas and assert
selection of the largest. A separate failure fixture exits 2 with
<code>decision=stop_galerkin_required</code>.

- [ ] **Step 2: Require complete output**

The JSON contains input hashes, zero-order audits, all alphas, selected alpha,
all basis/family/frequency values, four ordering gates, conditions,
dielectric minima, and:

~~~json
{
  "uses_sos_energy_as_numeric_input": false,
  "uses_ghost_family": false,
  "new_candidate_was_evaluated": false
}
~~~

Require Markdown and two-panel H/H2 plots comparing base and sensitivity
frequency losses. Outputs use atomic rename.

- [ ] **Step 3: Verify RED**

~~~bash
python -m unittest -v test_rpa_sensitive_ranking
~~~

Expected: missing script/import, not a fixture failure.

- [ ] **Step 4: Implement the minimal analyzer**

Reuse strict readers and the new evaluator. For every alpha:

~~~python
first_f_improves_two_d = first_f < two_d
first_g_improves_first_f = first_g < first_f
second_f_not_better = second_f >= first_g
second_g_not_better = second_g >= first_g
~~~

Choose <code>max(admissible_alpha)</code>. Do not read SOS energy files. The
historic promotion decisions are encoded by labels, not CP numbers.

- [ ] **Step 5: Verify GREEN**

~~~bash
python -m unittest -v \
  test_rpa_sensitive_ranking test_projected_pi_analysis test_projected_pi
~~~

- [ ] **Step 6: Commit and push**

~~~bash
git add \
  SIAB/example_H_sternheimer/projected_pi_loss/analyze_rpa_sensitive_ranking.py \
  SIAB/tests/test_rpa_sensitive_ranking.py \
  SIAB/example_H_sternheimer/projected_pi_loss/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): rank RPA-sensitive response metrics'
git push origin HEAD:codex/sternheimer-siab-h
~~~

### Task 7: Run the real historical gate and freeze alpha

**Files:**
- Create: <code>SIAB/example_H_sternheimer/projected_pi_loss/results/rpa_sensitive_historical_ranking.json</code>
- Create: <code>SIAB/example_H_sternheimer/projected_pi_loss/results/rpa_sensitive_historical_ranking.md</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>
- Modify: <code>sternheimer_siab_project/main.tex</code>

- [ ] **Step 1: Preflight exact archived inputs**

Use source/response/audits from:

~~~text
/work1/ghj/sternheimer_abacus_tests/siab_pi_dpsi_joint_3s2p2d1f1g_20260802/inputs
~~~

Use:

~~~text
two_d:
/work1/ghj/sternheimer_abacus_tests/siab_pi_dpsi_joint_3s2p2d_20260802/run/ORBITAL_RESULTS.txt

first_f:
/work1/ghj/sternheimer_abacus_tests/siab_pi_dpsi_joint_3s2p2d1f_prod_retry_20260802/run/ORBITAL_RESULTS.txt

first_g:
/work1/ghj/sternheimer_abacus_tests/siab_pi_dpsi_joint_3s2p2d1f1g_20260802/run/ORBITAL_RESULTS.txt

second_f:
/work1/ghj/sternheimer_abacus_tests/siab_pi_dpsi_joint_3s2p2d2f1g_20260802/run/ORBITAL_RESULTS.txt

second_g:
/work1/ghj/sternheimer_abacus_tests/siab_pi_dpsi_joint_3s2p2d1f2g_20260802/run/ORBITAL_RESULTS.txt
~~~

Require every file and record SHA256.

- [ ] **Step 2: Run on one normal node**

Submit one node, 30 CPUs, 110610 MB, and 24 hours. The immutable wrapper
records commit, source manifest, input hashes, <code>/usr/bin/time -v</code>,
runtime versions, and thread settings.

- [ ] **Step 3: Apply the hard decision**

If exit code is 2 or selected alpha is null, write
<code>stop_galerkin_required</code> to README and TeX and stop this plan. Do
not create a training input.

If it passes, freeze the largest selected alpha in JSON, Markdown, README, and
TeX. State that this is a metric gate, not a new optimized basis.

- [ ] **Step 4: Archive compact evidence**

Copy JSON, Markdown, plots, Slurm output, time report, and SHA256 manifests to
the repository results folder and matching TeX data folder. Do not copy large
response matrices.

- [ ] **Step 5: Reproduce and commit**

Re-run the analyzer from the same inputs and require numeric JSON equality.
Then:

~~~bash
git add SIAB/example_H_sternheimer/projected_pi_loss/results \
  SIAB/example_H_sternheimer/projected_pi_loss/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'test(siab): freeze RPA-sensitive historical ranking'
git push origin HEAD:codex/sternheimer-siab-h
~~~

### Task 8: Freeze and run the fixed-size optimizer

**Files:**
- Create: <code>SIAB/example_H_sternheimer/projected_pi_loss/INPUT.pi_rpa_sensitive_joint_3s2p2d1f1g</code>
- Create: <code>SIAB/example_H_sternheimer/projected_pi_loss/run_rpa_sensitive_joint.slurm</code>
- Modify: <code>SIAB/tests/test_h_sternheimer_smoke.py</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>
- Modify: <code>sternheimer_siab_project/main.tex</code>

- [ ] **Step 1: Write the failing campaign test**

Require:

~~~text
Nu = [3,2,2,1,1]
fixed 1s,2s,1p
selected first-g final ORBITAL_RESULTS as C_init_file
joint_dpsi_weight = 0.02
tau_dft/tau_dpsi = 0.05/0.10
condition_limit = 1e12
radial and low-frequency penalties zero
mode = pi_rpa_sensitive_joint
alpha exactly equals selected_alpha in ranking JSON
~~~

Require the runner to use <code>normal</code>, one node, 30 CPUs, 110610 MB,
24 hours, the frozen Python runtime, SHA256 preflight, and new-mode metadata
validation.

- [ ] **Step 2: Verify RED**

Run <code>test_h_sternheimer_smoke</code>. Expected: missing input/runner.

- [ ] **Step 3: Create the frozen input and runner**

Copy the selected first-g input and replace only the mode, exact selected
alpha, and <code>C_init_file</code> pointing at the final first-g
coefficients, not its residual seed:

~~~json
{
  "mode": "pi_rpa_sensitive_joint",
  "projected_pi_sensitivity_alpha": 0.25
}
~~~

The shown alpha is replaced by the exact Task 7 value. The smoke test reads
the ranking JSON and prevents mismatch. The runner requires:

~~~bash
grep -q '^Mode = pi_rpa_sensitive_joint$' ORBITAL_RESULTS.txt
python -m json.tool PROJECTED_PI_METADATA.json >/dev/null
~~~

- [ ] **Step 4: Verify GREEN and full suite**

Run <code>test_h_sternheimer_smoke</code>, then the full SIAB suite. Commit
the input/runner only after all pass.

- [ ] **Step 5: Stage immutable production**

Create:

~~~text
/work1/ghj/sternheimer_abacus_tests/siab_pi_rpa_sensitive_3s2p2d1f1g_20260804
~~~

Copy the selected first-g input bundle, replace only the new INPUT and source
commit, write <code>SOURCE_COMMIT</code>,
<code>SOURCE_MANIFEST.sha256</code>, and <code>INPUTS.sha256</code>, then
submit.

- [ ] **Step 6: Validate training**

Require:

~~~text
Slurm COMPLETED 0:0
finite final loss and gradients
fixed 1s,2s,1p maximum difference = 0
condition <= 1e12
DFT ratio <= 1.05
dpsi ratio <= 1.10
final H response < initial H response
final fourth-order family response < initial family response
complete PROJECTED_PI_METADATA.json
~~~

Record job, node, elapsed, MaxRSS, accepted step, alpha, all loss terms, and
final orbital SHA256. Stop before SOS if any gate fails.

- [ ] **Step 7: Document, build, inspect, commit**

Update README and TeX. Build:

~~~bash
cd /Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
~~~

Render and inspect all new pages. Commit:

~~~bash
git add SIAB/example_H_sternheimer/projected_pi_loss \
  SIAB/tests/test_h_sternheimer_smoke.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat(siab): train fixed-size RPA-sensitive H basis'
git push origin HEAD:codex/sternheimer-siab-h
~~~

### Task 9: Run the equilibrium all-band SOS/CP gate

**Files:**
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/results/</code>
- Modify: <code>SIAB/example_H_sternheimer/projected_pi_loss/README.md</code>
- Modify: <code>sternheimer_siab_project/main.tex</code>

- [ ] **Step 1: Reuse the validated three-case runner**

Use <code>run_pi_dpsi_joint_sos.slurm</code> with the new training root. Do
not change its physics. It derives all band counts from 35 AO/H:

~~~text
H2/H/H+ghost = 70/35/70 bands
~~~

Every case removes explicit <code>ABFS_ORBITAL</code> and regenerates its own
PCA-<code>1e-4</code> auxiliary basis.

- [ ] **Step 2: Preflight the physical contract**

Require:

~~~text
20-Angstrom cubic cell
0.74085-Angstrom H2 bond
100 Ry
16 identical minimax frequencies
rpa_ccp_rmesh_times=5
Massidda correction
LibRPA full Coulomb
all AO bands
H2 and H+ghost full-Coulomb SHA256 identical
~~~

- [ ] **Step 3: Submit and monitor**

Use <code>normal</code> and one full node per array task. Run postprocessing
on the login node only after all tasks are <code>COMPLETED 0:0</code> and
contain LibRPA success markers.

- [ ] **Step 4: Compute the physical decomposition**

Report:

~~~text
EcRPA(H2), EcRPA(H), EcRPA(H+ghost)
D_raw, D_CP, BSSE
D0_CP, RPAc_CP
differences from selected 3s2p2d1f1g
differences from the 108.72-kcal/mol reference
~~~

Pass only if CP increases and BSSE decreases relative to
<code>107.888474/1.070332</code> kcal/mol.

- [ ] **Step 5: Archive and document**

Archive compact JSON/Markdown, scheduler output, commits/binary hashes,
orbital/Coulomb hashes, AO/ABF dimensions, band counts, resources, and time.
Update README and TeX before interpretation.

- [ ] **Step 6: Verify and commit**

Run directed CP tests, the full SIAB suite, TeX build, unresolved-reference
scan, and visual inspection. Commit:

~~~bash
git add SIAB/example_H_sternheimer/projected_pi_loss/results \
  SIAB/example_H_sternheimer/projected_pi_loss/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'test(siab): gate RPA-sensitive H basis with SOS CP'
git push origin HEAD:codex/sternheimer-siab-h
~~~

If the gate passes, write a separate four-geometry Delta-ST/raw/CP release
plan. If it fails, retain <code>3s2p2d1f1g</code> and use the recorded
frequency/channel diagnostics to decide whether to implement a
candidate-space Galerkin Sternheimer solve.

## Final verification checklist

- [ ] Old <code>pi_dpsi_joint</code> values and output schema are unchanged.
- [ ] Sensitivity algebra passes analytic, gradient, phase, and permutation tests.
- [ ] Invalid dielectric spectra fail without clipping.
- [ ] Historical alpha freezes before any new candidate is evaluated.
- [ ] H+ghost and SOS energies never enter the optimizer.
- [ ] Fixed DZP columns remain bitwise unchanged.
- [ ] Training and physical gates are reported separately.
- [ ] Every SOS case regenerates PCA-<code>1e-4</code> auxiliary bases and uses full Coulomb.
- [ ] Every compute job uses <code>normal</code> and full-node resources.
- [ ] README and TeX distinguish proposed, RED, GREEN, queued, completed, and physical-pass states.
- [ ] Every commit has Codex author and AroundPeking committer attribution.
