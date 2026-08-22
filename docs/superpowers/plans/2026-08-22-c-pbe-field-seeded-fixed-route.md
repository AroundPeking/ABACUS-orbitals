# C PBE Field-Seeded Fixed Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the numerically unstable zero-field fixed-occupation C cold start with a fixed-occupation field seed followed by a zero-field fixed restart, while preserving the free routes and all physical acceptance thresholds.

**Architecture:** Add one explicit `fixed_field` input mode and map the fixed branch to `fixed_field_seed -> fixed_zero_restart`.  Keep zero-field `fixed` and free-occupation modes separate so phase inputs remain exactly auditable.  Update the preparation, runtime, provenance, comparison, fake-HPC, and documentation contracts together.

**Tech Stack:** Python 3.7 standard library, `unittest`, Bash, Slurm, ABACUS LCAO PBE.

---

## File map

- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/gate_contract.py`: define and audit the `fixed_field` input mode.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_gate_contract.py`: specify fixed-field and zero-field restart behavior.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/prepare_gate.py`: stage the new first fixed phase.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/audit_gate.py`: rename the fixed phase chain and compare only final zero-field results.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_prepare_gate.py`: cover initial staging and rendered restart names.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_audit_gate.py`: cover the new eleven-phase audit map.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch_common.sh`: execute the new fixed phase chain.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`: update fake ABACUS execution, failure injection, and immutable runtime tests.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/README.md`: document the field as an orientation selector and the zero-field comparison boundary.
- Modify `docs/superpowers/plans/2026-08-22-c-atom-pbe-gate-server66-migration.md`: record the diagnostic evidence and replacement-run rule.

### Task 1: Define the fixed-field input contract

**Files:**
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/gate_contract.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_gate_contract.py`

- [ ] **Step 1: Write failing mode tests**

Add tests equivalent to:

```python
def test_fixed_field_seed_combines_integer_occupation_and_field(self):
    text = render_input(mode="fixed_field", field_dir=0, restart=False)
    self.assertIn("ocp 1\n", text)
    self.assertIn("ocp_set 3*1 19*0 1*1 21*0\n", text)
    self.assertIn("efield_flag 1\n", text)
    self.assertIn("efield_dir 0\n", text)
    self.assertIn("efield_amp 1e-4\n", text)
    self.assertNotIn("init_wfc file", text)

def test_fixed_field_requires_direction_and_rejects_restart(self):
    with self.assertRaisesRegex(ValueError, "field_dir"):
        render_input(mode="fixed_field", restart=False)
    with self.assertRaisesRegex(ValueError, "restart"):
        render_input(mode="fixed_field", field_dir=0, restart=True)
```

Extend the exact input whitelist tests so `fixed_field` accepts only the shared
keys, fixed-occupation keys, and field keys; extra or missing keys remain fatal.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest \
  SIAB.example_C_sternheimer.pbe_reference_gate.tests.test_gate_contract -v
```

Expected: `fixed_field` is rejected because it is not in `VALID_MODES`.

- [ ] **Step 3: Implement the minimal mode**

Add `fixed_field` to `VALID_MODES`.  Require a direction in `{0, 1, 2}`, reject
restart, emit the same `ocp` and `ocp_set` values as `fixed`, and emit the same
field keys and amplitude as `field`.  Keep `fixed` as the zero-field mode used
by the restart.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Task 1 command.  Expected: every gate-contract test passes.

- [ ] **Step 5: Commit Task 1**

Commit as:

```text
feat(siab): add fixed-field C reference input
```

### Task 2: Map preparation and auditing to the new fixed chain

**Files:**
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/prepare_gate.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/audit_gate.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_prepare_gate.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_audit_gate.py`

- [ ] **Step 1: Write failing phase-map tests**

Require:

```python
self.assertEqual(
    audit_gate.BRANCH_PHASES["fixed"],
    ("fixed_field_seed", "fixed_zero_restart"),
)
```

Update preparation expectations so branch `fixed` initially creates only
`fixed_field_seed`, whose INPUT is exactly
`render_input(mode="fixed_field", field_dir=0, restart=False)`.  Update the
eleven-phase audit fixture to contain:

```text
runs/fixed/fixed_field_seed
runs/fixed/fixed_zero_restart
```

and reject stale `fixed_cold` or `fixed_restart` directories.

- [ ] **Step 2: Run preparation and audit tests and verify RED**

Run:

```bash
python3 -m unittest \
  SIAB.example_C_sternheimer.pbe_reference_gate.tests.test_prepare_gate \
  SIAB.example_C_sternheimer.pbe_reference_gate.tests.test_audit_gate -v
```

Expected: old fixed phase names and input modes disagree with the new tests.

- [ ] **Step 3: Implement the phase chain**

Change `_PHASE_NAMES["fixed"]` to `fixed_field_seed`.  Change
`BRANCH_PHASES["fixed"]` and the first two `PHASE_SPECS` to:

```python
"fixed": ("fixed_field_seed", "fixed_zero_restart")
PhaseSpec("runs/fixed/fixed_field_seed", "fixed_field", False, 0)
PhaseSpec("runs/fixed/fixed_zero_restart", "fixed", True)
```

Treat `fixed_field` as field-bearing in `_field_direction` and expected-input
rendering.  In the global comparison, compute fixed drift from field seed to
zero restart but use only `fixed_zero_restart.energy_ha` as the fixed reference.
Do not change any threshold in `gate_contract.py`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 command.  Expected: both test modules pass.

- [ ] **Step 5: Commit Task 2**

Commit as:

```text
feat(siab): audit field-seeded fixed C route
```

### Task 3: Execute and simulate the new fixed route

**Files:**
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch_common.sh`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`

- [ ] **Step 1: Write failing runtime tests**

Update the fake ABACUS phase-energy perturbation set and all fixed failure
injection paths to the new names.  Require the shared runner to contain this
fixed path in order:

```bash
run_phase fixed_field_seed
prepare_restart fixed_field_seed fixed_zero_restart
run_phase fixed_zero_restart
```

Require fake four-task execution to create exactly eleven phase-complete
records and a `PBE_GATE_PASSED` result with the new fixed phase names.

- [ ] **Step 2: Run HPC contract tests and verify RED**

Run:

```bash
python3 -m unittest \
  SIAB.example_C_sternheimer.pbe_reference_gate.tests.test_hpc_contract -v
```

Expected: the runner still invokes `fixed_cold` and `fixed_restart`.

- [ ] **Step 3: Implement the runtime path**

Replace only the three fixed-branch runner calls with the sequence above.
Keep restart file copying, canonical-path load evidence, array mapping,
resource validation, duplicate protection, and free branches unchanged.

- [ ] **Step 4: Run HPC and complete local tests**

Run the Task 3 command, then:

```bash
python3 -m unittest discover \
  -s SIAB/example_C_sternheimer/pbe_reference_gate/tests -v
```

Expected: all tests pass; record the exact new test count.

- [ ] **Step 5: Commit Task 3**

Commit as:

```text
feat(siab): run field-seeded fixed C branch
```

### Task 4: Document, verify, and run the formal gate

**Files:**
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/README.md`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`
- Modify: `docs/superpowers/plans/2026-08-22-c-atom-pbe-gate-server66-migration.md`
- Create after passage: `SIAB/example_C_sternheimer/pbe_reference_gate/results/PBE_GATE_RESULT.md`

- [ ] **Step 1: Add failing documentation assertions**

Require the README to name `fixed_field_seed`, `fixed_zero_restart`,
`field amplitude 1e-4`, `3 up / 1 down`, and the rule that the field-seed
energy is excluded from the final physical comparison.

- [ ] **Step 2: Update the documents**

Record the failed beta/Kerker reproducibility tests, the two successful
field-seeded repetitions, their iteration counts and energies, and the exact
zero-field comparison boundary.  Replace the README preflight test count with
the count observed after Task 3.

- [ ] **Step 3: Run final local verification**

Run:

```bash
python3 -m unittest discover \
  -s SIAB/example_C_sternheimer/pbe_reference_gate/tests -v
bash -n SIAB/example_C_sternheimer/pbe_reference_gate/*.sh \
  SIAB/example_C_sternheimer/pbe_reference_gate/*.slurm
python3 -m py_compile SIAB/example_C_sternheimer/pbe_reference_gate/*.py
git diff --check
```

Expected: complete green test output, successful syntax/compile checks, and no
patch-format errors.

- [ ] **Step 4: Commit the completed implementation**

Commit documentation and any final contract-test updates with Codex as author
and AroundPeking as committer.

- [ ] **Step 5: Stage one new immutable server66 run**

Create `/home/ghj/abacus/260822/c-atom-pbe-equivalence-server66-<commit12>`
from an exact source archive.  Verify the source, ABACUS, Python, environment,
pseudopotential, and orbital hashes; run the full tests, shell and Python
checks, and exact `sbatch --test-only` shape; then atomically create
`PREFLIGHT_PASSED`.

- [ ] **Step 6: Submit and audit exactly once**

Confirm that no root, claim, receipt, active job, or accounting record exists
for the new stable job name.  Submit one four-task server66 array.  After every
task leaves the scheduler, run the login-node global audit and require
`PBE_GATE_PASSED`.  Report scheduler state, phase convergence, integer
occupations, final energies, threshold comparisons, wall times, and provenance
separately.  Only then write `results/PBE_GATE_RESULT.md` and proceed to a
separate C Delta-ST plan.
