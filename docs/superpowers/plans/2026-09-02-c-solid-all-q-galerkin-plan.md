# C Solid All-Q Galerkin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a solid-only diamond-C Galerkin adapter that validates the standard weighted FD8 thirteen-q dataset and produces one deterministic trust-region candidate without atomic inputs.

**Current protocol:** Tasks 1-5 below record the completed legacy six-frequency,
eight-logical-q scaffold and reduced replay.  They are retained for provenance,
not as the production contract.  Production now means 12 frequencies,
threshold-only product PCA `1e-6`, full periodic Coulomb, the thirteen FD8
representatives, and qavg head/wing only in the final LibRPA energy stage.

**Architecture:** Keep the existing q-weighted periodic loss as the single numerical implementation. Add a public single-family descent helper and a C-specific adapter that validates logical q-star labels separately from each dataset's symmetry-equivalent `selected_iq`. A reduced q1/q2/q3 mode provides regression evidence but is permanently blocked from physical release.

**Tech Stack:** Python 3, PyTorch 1.9-compatible tensor code, `unittest`, immutable JSON/manifests, Slurm contract scripts.

---

### Task 1: Public Single-Family Trust-Region Candidate

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_candidates.py`
- Modify: `SIAB/tests/test_periodic_galerkin_candidates.py`

- [ ] **Step 1: Write the failing unit tests**

Add tests that construct a `PeriodicGalerkinFamilyGradientResult` with one
`C_solid` family and assert that `build_single_family_candidate` follows the
negative normalized gradient, preserves the fixed prefix, retracts the
variable frame, is deterministic, and rejects an unknown family or invalid
trust radius.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest SIAB.tests.test_periodic_galerkin_candidates -v
```

Expected: failures because `build_single_family_candidate` and its result type
do not exist.

- [ ] **Step 3: Implement the minimal public helper**

Add:

```python
@dataclass(frozen=True)
class PeriodicGalerkinSingleFamilyCandidate:
    family: str
    trust_radius: float
    coefficients: dict
    coefficients_sha256: str


def build_single_family_candidate(result, *, fixed_nu, family, trust_radius=0.01):
    # Validate the family and radius, negate its normalized gradient, reuse
    # _retract_candidate, clone the output, and record coefficient_sha256.
```

- [ ] **Step 4: Run focused and core Galerkin tests and verify GREEN**

Run:

```bash
python3 -m unittest SIAB.tests.test_periodic_galerkin_candidates SIAB.tests.test_periodic_galerkin_fit -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the helper**

Commit with Codex as author and AroundPeking as committer using message
`Add single-family Galerkin candidate direction`.

### Task 2: Eight-Q Dataset Contract

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/build_c_solid_all_q_candidate.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/tests/test_c_solid_all_q_galerkin.py`

- [ ] **Step 1: Write failing contract tests**

Use small immutable stand-in dataset objects to exercise:

```python
validate_qstar_datasets(
    datasets,
    qstar_contract=(
        {"label": 1, "selected_iq": 1, "multiplicity": 1},
        {"label": 2, "selected_iq": 22, "multiplicity": 8},
        {"label": 3, "selected_iq": 43, "multiplicity": 4},
        {"label": 6, "selected_iq": 6, "multiplicity": 6},
        {"label": 7, "selected_iq": 27, "multiplicity": 24},
        {"label": 8, "selected_iq": 23, "multiplicity": 12},
        {"label": 11, "selected_iq": 11, "multiplicity": 3},
        {"label": 28, "selected_iq": 55, "multiplicity": 6},
    ),
    q_count=64,
    coverage="full",
)
```

Assert exact label ordering, unique `selected_iq`, multiplicity sum 64,
`q_weight=multiplicity/64`, six identical frequencies, and shared orbital,
pseudopotential, auxiliary, primitive-block, source and executable identities.
Add separate failures for one missing star, one duplicate star, one wrong
weight, one frequency mismatch and one provenance mismatch. Assert that a
declared q1/q2/q3 reduced contract passes numerical validation but returns
`physical_release_gate="hold"` and `coverage="reduced"`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m unittest SIAB.example_C_sternheimer.periodic_basis_optimization.tests.test_c_solid_all_q_galerkin -v
```

Expected: import failure because the adapter does not exist.

- [ ] **Step 3: Implement pure contract functions**

Implement `load_config`, `validate_qstar_datasets`, `build_dataset_inventory`,
`sha256`, and deterministic JSON writing. Keep logical star labels separate
from symmetry-equivalent `selected_iq`. Do not read atomic response files.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the focused module again and require all tests to pass.

- [ ] **Step 5: Commit the contract layer**

Commit with message `Validate solid-only all-q Galerkin datasets` and the
required author/committer identities.

### Task 3: Solid-Only Candidate CLI

**Files:**
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/build_c_solid_all_q_candidate.py`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/tests/test_c_solid_all_q_galerkin.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/c_diamond_solid_q123_reduced.json`

- [ ] **Step 1: Write failing CLI and manifest tests**

Assert that argument parsing accepts repeated `--qstar LABEL=PATH`, `--initial`,
`--config`, `--output-directory`, and `--source-commit`, and exposes no atomic
arguments. Use injected readers/evaluators to verify all datasets are passed as
one `C_solid` family, a single-family candidate is generated, and the output
contains `STATUS.json`, `PROVENANCE.json`, `DATASET_INVENTORY.json`,
`GRADIENT.json`, `CANDIDATE.json`, and `ORBITAL_RESULTS.txt`.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: failures because the CLI and artifact builder are not implemented.

- [ ] **Step 3: Implement the minimal CLI**

Read and validate datasets with the existing periodic reader and
`validate_dataset_contract`, call `evaluate_family_gradients` with
`dataset_families=("C_solid",) * len(datasets)`, build one bounded candidate,
evaluate its all-q loss, and write hashed artifacts. Full coverage may report
`candidate_generation_gate`; reduced coverage must report
`physical_release_gate="hold"` regardless of improvement.

- [ ] **Step 4: Run focused and existing C workflow regression tests**

Run:

```bash
python3 -m unittest \
  SIAB.example_C_sternheimer.periodic_basis_optimization.tests.test_c_solid_all_q_galerkin \
  SIAB.example_C_sternheimer.periodic_basis_optimization.tests.test_c_galerkin_binding_workflow \
  SIAB.tests.test_periodic_galerkin_candidates \
  SIAB.tests.test_periodic_galerkin_fit -v
```

Expected: all tests pass and the existing atom-solid adapter remains unchanged.

- [ ] **Step 5: Commit the CLI**

Commit with message `Add solid-only all-q Galerkin adapter` and the required
identities.

### Task 4: Reduced-Set Replay Contract And Remote Audit

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_solid_q123_reduced_df.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/tests/test_c_solid_all_q_runner_contract.sh`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/audit_c_solid_qstar_inputs.py`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/tests/test_c_solid_all_q_galerkin.py`

- [ ] **Step 1: Write failing runner and audit tests**

Require the runner to use immutable source, q1/q2/q3 datasets, the reduced
config, exact source and input hashes, duplicate-safe output creation, one MPI
rank with bounded threads, and no atomic paths. Require the audit to emit the
present logical stars, symmetry-equivalent `selected_iq`, exact weights,
missing logical stars, and whether a direct solid reference exists.

- [ ] **Step 2: Run both tests and verify RED**

Run the Python test and shell contract test. Expected: missing runner and audit
implementation failures.

- [ ] **Step 3: Implement runner and read-only audit**

The runner executes only the reduced offline replay. The audit reads manifests
and reference-result metadata but never invokes ABACUS, LibRPA, Slurm or a
network operation.

- [ ] **Step 4: Run the complete local regression set**

Run all tests from Tasks 1-4 plus `git diff --check`. Require clean output.

- [ ] **Step 5: Commit and push the completed code stage**

Commit with message `Stage reduced solid all-q Galerkin audit`, verify Codex
author and AroundPeking committer, then push the current branch.

### Task 5: Validate Reduced Replay And Define The Physical Complement

**Files:**
- No source edit before replay validation.
- Update after validation: canonical `development_notes` TeX section.

- [ ] **Step 1: Deploy the exact pushed commit immutably on df**

Verify repository commit, deployment manifest, source hashes and duplicate
status before any submission.

- [ ] **Step 2: Run one reduced q1/q2/q3 offline replay**

This job reads existing frozen datasets only. Require scheduler
`COMPLETED/0:0`, status/provenance success, finite loss and gradient,
deterministic candidate hash, `coverage="reduced"`, and
`physical_release_gate="hold"`.

- [ ] **Step 3: Produce the complement manifest**

Record the exact missing logical q stars and their producer indices, weights,
frequency grid, parent producer, executable, PP, orbital, auxiliary and grid
hashes. Record separately that no independent solid-only Delta-ST reference is
currently accepted; do not substitute the original-TZDP ordinary-SOS baseline.

- [ ] **Step 4: Estimate measured cost before physical submission**

Use completed q1/q2/q3 timing and memory to choose a duplicate-safe placement
for the missing response producers. Keep each frozen response local to the
subsequent offline optimizer. Do not submit until the complement contract and
direct reference definition are complete.

- [ ] **Step 5: Update and render the research note**

Record the code commit, replay result, missing q contract, measured cost and
remaining physical decision. Compile the canonical TeX note and inspect every
changed PDF page before reporting the stage complete.

### Task 6: Freeze The Standard FD8 Production Contract

**Files:**
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/build_c_solid_all_q_candidate.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/c_diamond_solid_fd8_q13_standard.json`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_solid_fd8_q13_standard_q1_df.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/librpa_c_solid_fd8_q13_qavg.in`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/tests/test_c_solid_all_q_galerkin.py`

- [x] **Step 1: Write and observe failing standard-protocol tests**

Require exactly thirteen representatives with multiplicities summing to 64,
12 common frequencies, product PCA `1e-6`, and separate training/final-energy
contracts.  Confirm the tests fail against the legacy-only implementation.

- [x] **Step 2: Implement the versioned standard contract**

Keep version 1 reduced/eight-q parsing unchanged.  Add version 2 with exact
FD8 labels and multiplicities, 12-frequency validation, product-PCA and full
Coulomb metadata, plus the final qavg head/wing contract.

- [x] **Step 3: Add a unique q1 timing and storage gate**

Use the frozen basis-opt ABACUS executable on 48 p1 nodes, with 12 frequency
groups and four k groups.  Produce q1 only.  Validate scheduler/program status,
all equations, manifest provenance, exact frequency grid, memory, wall time and
dataset bytes.  Do not release the other twelve representatives yet.

- [ ] **Step 4: Verify, commit, deploy and submit q1 once**

Run focused and related regression tests, shell syntax checks and
`git diff --check`.  Commit with the required attribution, deploy the exact
commit immutably, check queue and output roots for duplicates, then submit one
q1 gate.

- [ ] **Step 5: Decide whether to release the remaining twelve q points**

Require q1 `COMPLETED/0:0`, program and provenance success, exactly 12 complete
frequencies, product PCA `1e-6`, full Coulomb, a frequency-grid hash matching
the accepted standard reference, and measured runtime/storage within the
campaign limits.  Only then create the remaining-q array contract.

- [ ] **Step 6: Final qavg physical validation**

After offline Galerkin optimization and candidate promotion, run the complete
thirteen-representative reader chain and LibRPA using the frozen qavg template.
Keep the body response and final head/wing correction as separately reported
quantities; do not call a Galerkin loss or body-only energy a physical result.
