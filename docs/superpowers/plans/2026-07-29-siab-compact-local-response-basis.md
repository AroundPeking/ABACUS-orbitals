# SIAB Compact Local Response Basis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, AO-budgeted H response-basis sequence with a fixed DZP core, joint H/H2 Sternheimer+dpsi optimization, and an explicit differentiable radial-tail constraint.

**Architecture:** Add a standalone radial-locality evaluator built from the same spherical-Bessel and cutoff-smoothing definition used by SIAB. Feed its scalar metric into the existing named loss composer without changing zero-weight behavior. Extend the nested selector with an AO budget and locality records, freeze all compact candidates before SOS, and keep H+ghost outside optimization.

**Tech Stack:** Python 3, PyTorch float64, SciPy spherical Bessel functions, unittest, SIAB Q/S target files, Slurm `normal`, XeLaTeX.

---

### Task 1: Add the radial-locality metric

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/radial_locality.py`
- Create: `SIAB/tests/test_radial_locality.py`

- [ ] **Step 1: Write failing tests**

Test a two-primitive synthetic radial space and require:

```python
metric = RadialSubspaceLocality(...)
result = metric.evaluate(coefficients)
assert 0.0 <= result.loss.item() <= 1.0
assert result.by_channel[("H", 0)].variable_columns == 1
```

Also require invariance under nonsingular rotations of variable columns,
exact exclusion of the fixed column, finite gradients, and explicit failure
for a linearly dependent projected variable subspace.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest SIAB.tests.test_radial_locality -v
```

Expected: import failure because `radial_locality.py` is absent.

- [ ] **Step 3: Implement the metric**

Implement:

```python
@dataclass(frozen=True)
class RadialChannelLocality:
    element: str
    l: int
    variable_columns: int
    tail_fraction: torch.Tensor
    condition: float

@dataclass(frozen=True)
class RadialLocalityResult:
    loss: torch.Tensor
    max_condition: float
    by_channel: dict

class RadialSubspaceLocality:
    def __init__(self, info_element, radial, eigenvalues, fixed_specs,
                 local_radius, condition_limit=1.0e10): ...
    def evaluate(self, coefficients): ...
```

Use `torch.trapezoid` on the configured radial mesh, project variable columns
out of the fixed radial subspace, and use Cholesky solves for both fixed and
variable Gram matrices. Do not add diagonal regularization.

- [ ] **Step 4: Verify GREEN and full unit suite**

Run the focused test and `python -m unittest discover -s SIAB/tests -v`.

- [ ] **Step 5: Commit**

Commit as `feat(siab): measure radial response locality`.

### Task 2: Add locality to the named optimization loss

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/optimization_loss.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/main.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/IO/func_C.py`
- Modify: `SIAB/tests/test_loss_and_freeze.py`
- Modify: `SIAB/tests/test_main_sternheimer.py`

- [ ] **Step 1: Write failing loss and integration tests**

Require the normalized loss keys

```text
radial_tail_weight
radial_tail_radius
radial_tail_condition_limit
```

and components

```text
radial_tail
regularization_locality
```

Require `total` to be unchanged when `radial_tail_weight=0`, and require the
gradient of `regularization_locality` to reach nonfixed coefficient columns
when the weight is positive.

- [ ] **Step 2: Verify RED**

Run the focused loss tests and confirm missing config/component failures.

- [ ] **Step 3: Implement the loss bridge**

Construct `RadialSubspaceLocality` in `main.py` only when the configured weight
is positive. Pass its result through `Opt_Orbital_Converge`, add

```python
regularization_locality = radial_tail_weight * radial_tail
```

to the total loss, logging, best-point record, and `ORBITAL_RESULTS.txt` loss
metadata. Preserve all old defaults.

- [ ] **Step 4: Verify GREEN and full unit suite**

Run focused tests and all SIAB tests.

- [ ] **Step 5: Commit**

Commit as `feat(siab): constrain response-orbital tails`.

### Task 3: Freeze an AO-budgeted compact frontier

**Files:**
- Modify: `SIAB/example_H_sternheimer/greedy_response_selection/select_response_shells.py`
- Modify: `SIAB/example_H_sternheimer/greedy_response_selection/response_selection_campaign.py`
- Modify: `SIAB/example_H_sternheimer/greedy_response_selection/selection_config.json`
- Modify: `SIAB/tests/test_select_response_shells.py`
- Modify: `SIAB/tests/test_response_selection_campaign.py`

- [ ] **Step 1: Write failing budget and manifest tests**

Require `max_ao_per_atom` to be a positive integer at least as large as the
initial basis. If the next shell would exceed the budget, return
`status="ao_budget_reached"` with all prior steps frozen. Require each step
record to contain the optimized radial-tail metric and maximum locality
condition. Reject ghost target fields and RPA energies as before.

- [ ] **Step 2: Verify RED**

Run the two focused selector test modules and confirm missing budget behavior.

- [ ] **Step 3: Implement budgeted stopping**

Add a pre-append AO-count check in the nested loop. Do not reuse the old
`global_capture=0.999` as a success requirement for this compact campaign.
Freeze the nested sequence under the budget even if the representable-space
loss has not reached the old threshold.

- [ ] **Step 4: Verify GREEN and full unit suite**

Run focused and complete SIAB tests.

- [ ] **Step 5: Commit**

Commit as `feat(siab): freeze compact response frontier`.

### Task 4: Stage and run the compact H/H2 campaign

**Files:**
- Create: `SIAB/example_H_sternheimer/compact_response_selection/README.md`
- Create: `SIAB/example_H_sternheimer/compact_response_selection/selection_config.json`
- Create: `SIAB/example_H_sternheimer/compact_response_selection/run_selection.slurm`
- Create: `SIAB/tests/test_compact_response_selection.py`

- [ ] **Step 1: Write the static production-contract test**

Require physical H and H2 targets only, full TZDP initialization, fixed DZP,
`max_ao_per_atom=48`, `normal`, one 30-thread node, `110610M`, 24 hours, and a
nonzero committed radial-tail weight/radius. Reject `debug`, ghost targets,
RPA-energy fields, and mutable source execution.

- [ ] **Step 2: Verify RED, then add the immutable runner**

The runner records source/asset hashes, uses the existing CPU PyTorch runtime,
runs the complete server SIAB suite before optimization, and writes a compact
candidate manifest containing AO counts, H/H2 response metrics, tail metrics,
condition numbers, and coefficient/orbital hashes.

- [ ] **Step 3: Commit and stage on df_dcu**

Commit as `chore(siab): stage compact H response campaign`, freeze the source
closure, verify the feature-branch SHA and target hashes, and submit only to
`normal`.

- [ ] **Step 4: Verify the completed campaign**

Require scheduler `COMPLETED 0:0`, the full test count, fixed-column bitwise
identity, no ghost target in any optimizer input, and at least one frozen
candidate below `48 AO/H`.

### Task 5: Run frozen raw-SOS gates and document results

**Files:**
- Create: `SIAB/example_H_sternheimer/compact_response_selection/run_sos_frontier.slurm`
- Modify: `SIAB/example_H_sternheimer/compact_response_selection/README.md`
- Modify: `/Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project/main.tex`

- [ ] **Step 1: Select frozen checkpoints without energy feedback**

Use the complete compact sequence or predeclared AO checkpoints. Do not change
shell order, locality weight, or coefficients after reading an SOS result.

- [ ] **Step 2: Run matched H/H2 all-band full-Coulomb SOS**

Verify ABACUS/LibRPA source and executable hashes, matching 20-Angstrom cell,
100-Ry SOS grid, 16 minimax frequencies, fixed 214-function-per-H ABFS, full
Coulomb, all bands, occupations, and final markers. Record ABACUS and LibRPA
wall time and peak memory separately.

- [ ] **Step 3: Apply the decision gate**

Run H+ghost only for compact candidates whose raw result and overlap condition
are physically promising. Accept only if raw, CP, and Delta-ST binding energies
agree within `0.1 kcal/mol`. Otherwise record whether the next change is the Pi
target or the `l<=8` primitive target.

- [ ] **Step 4: Update and verify the research note**

Move the truncated full-basis CP work into a clearly labeled side diagnostic.
Add the compact-basis formulas, candidate table, measured results, and open
gate. Compile with XeLaTeX and inspect all affected PDF pages.

- [ ] **Step 5: Commit**

Commit measured code-repository artifacts as `docs(siab): record compact response basis gate`.

