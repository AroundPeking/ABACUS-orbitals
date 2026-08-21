# C Atom PBE Equivalence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible PBE gate proving that fixed integer occupation and weak-field-seeded free occupation converge to the same zero-field triplet C atom before any C Delta-ST calculation is allowed.

**Architecture:** A small pure-Python contract module renders and audits the eleven PBE phases.  A four-task Slurm array runs one fixed branch and three field-direction branches on `df_dcu` normal nodes; a separate audit command accepts only complete zero-field results.  Calculation staging, scheduler submission, and physical acceptance remain separate so a scheduler success cannot become a false scientific pass.

**Tech Stack:** Python 3.10 standard library, `unittest`, Bash, Slurm, ABACUS LCAO PBE, SG15 C TZDP-10au.

---

## File map

- Create `SIAB/example_C_sternheimer/pbe_reference_gate/gate_contract.py`: input rendering, output parsing, and physical acceptance logic.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/prepare_gate.py`: prepare one fixed or field-direction branch from immutable assets.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/audit_gate.py`: emit `RESULT_SUMMARY.json` and `RESULT_SUMMARY.txt` only after all branches exist.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch.slurm`: run one Slurm array branch with full normal-node resources.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/submit_pbe_gate.sh`: duplicate-safe submission wrapper.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_gate_contract.py`: unit tests for input and physical-output contracts.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`: static tests for scheduler and duplicate-prevention rules.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/README.md`: physical purpose, exact run protocol, and result interpretation.
- Create after the formal run `SIAB/example_C_sternheimer/pbe_reference_gate/results/PBE_GATE_RESULT.md`: compact measured result and provenance record; do not commit large ABACUS outputs.

The implementation plan ends at the PBE decision.  Delta-ST scripts are a separate plan that is written only after `PBE_GATE_PASSED` exists.

### Task 1: Render exact fixed and free branch inputs

**Files:**
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/gate_contract.py`
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_gate_contract.py`

- [ ] **Step 1: Write failing input-contract tests**

```python
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gate_contract import render_input


class InputContractTests(unittest.TestCase):
    def test_fixed_zero_field_contract(self):
        text = render_input(mode="fixed", restart=False)
        self.assertIn("nspin 2", text)
        self.assertIn("nelec 4", text)
        self.assertIn("nupdown 2", text)
        self.assertIn("ocp 1", text)
        self.assertIn("ocp_set 3*1 19*0 1*1 21*0", text)
        self.assertIn("efield_flag 0", text)
        self.assertIn("efield_amp 0", text)
        self.assertNotIn("init_wfc file", text)

    def test_field_seed_is_not_fixed(self):
        text = render_input(mode="field", field_dir=1, restart=False)
        self.assertIn("ocp 0", text)
        self.assertNotIn("ocp_set", text)
        self.assertIn("efield_flag 1", text)
        self.assertIn("dip_cor_flag 0", text)
        self.assertIn("efield_dir 1", text)
        self.assertIn("efield_pos_max 0.8", text)
        self.assertIn("efield_pos_dec 0.1", text)
        self.assertIn("efield_amp 1e-4", text)

    def test_free_restart_removes_field_and_ocp(self):
        text = render_input(mode="free", field_dir=2, restart=True)
        self.assertIn("ocp 0", text)
        self.assertNotIn("ocp_set", text)
        self.assertIn("efield_flag 0", text)
        self.assertIn("efield_amp 0", text)
        self.assertIn("init_wfc file", text)
        self.assertIn("init_chg file", text)

    def test_rejects_invalid_field_direction(self):
        with self.assertRaisesRegex(ValueError, "field_dir"):
            render_input(mode="field", field_dir=3, restart=False)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m unittest SIAB.example_C_sternheimer.pbe_reference_gate.tests.test_gate_contract -v
```

Expected: import failure because `gate_contract.py` does not exist.

- [ ] **Step 3: Implement the minimal renderer**

Create `gate_contract.py` with constants for the frozen protocol and this public API:

```python
from __future__ import annotations

VALID_MODES = {"fixed", "field", "free"}


def render_input(*, mode: str, field_dir: int | None = None,
                 restart: bool = False) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if mode in {"field", "free"} and field_dir not in {0, 1, 2}:
        raise ValueError("field_dir must be 0, 1, or 2")
    if mode == "fixed" and field_dir is not None:
        raise ValueError("fixed mode does not accept field_dir")

    values = [
        ("INPUT_PARAMETERS", None),
        ("suffix", "C_PBE_REFERENCE_GATE"),
        ("calculation", "scf"),
        ("ntype", "1"),
        ("nelec", "4"),
        ("nspin", "2"),
        ("nupdown", "2"),
        ("nbands", "22"),
        ("basis_type", "lcao"),
        ("ecutwfc", "30"),
        ("lcao_ecut", "100"),
        ("nx", "135"),
        ("ny", "135"),
        ("nz", "135"),
        ("ks_solver", "genelpa"),
        ("dft_functional", "pbe"),
        ("symmetry", "0"),
        ("kpar", "1"),
        ("pseudo_dir", "./"),
        ("orbital_dir", "./"),
        ("scf_thr", "1e-10"),
        ("scf_nmax", "300"),
        ("mixing_type", "broyden"),
        ("mixing_beta", "0.3"),
        ("mixing_beta_mag", "0.3"),
        ("smearing_method", "fixed"),
        ("ocp", "1" if mode == "fixed" else "0"),
    ]
    if mode == "fixed":
        values.append(("ocp_set", "3*1 19*0 1*1 21*0"))
    if mode == "field":
        values.extend([
            ("efield_flag", "1"),
            ("dip_cor_flag", "0"),
            ("efield_dir", str(field_dir)),
            ("efield_pos_max", "0.8"),
            ("efield_pos_dec", "0.1"),
            ("efield_amp", "1e-4"),
        ])
    else:
        values.extend([("efield_flag", "0"), ("efield_amp", "0")])
    values.extend([
        ("out_chg", "1"),
        ("out_wfc_lcao", "1"),
        ("out_app_flag", "1"),
        ("out_mul", "1"),
    ])
    if restart:
        values.extend([("init_wfc", "file"), ("init_chg", "file")])
    return "\n".join(key if value is None else f"{key} {value}" for key, value in values) + "\n"
```

- [ ] **Step 4: Verify GREEN**

Run the Task 1 unittest command.  Expected: four tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add SIAB/example_C_sternheimer/pbe_reference_gate/gate_contract.py \
  SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_gate_contract.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m "feat(siab): define C PBE reference inputs"
```

### Task 2: Parse occupations and enforce physical acceptance

**Files:**
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/gate_contract.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_gate_contract.py`
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/audit_gate.py`

- [ ] **Step 1: Add failing parser and threshold tests**

Add tests that create minimal `running_scf.log`, `eig_occ.txt`, and `INPUT` fixtures.  The tests must assert:

```python
phase = audit_phase(case_dir, expected_mode="fixed")
self.assertEqual(phase.spin_counts, {1: 3.0, 2: 1.0})
self.assertTrue(phase.integer_occupations)
self.assertAlmostEqual(phase.energy_ev, -147.4676776027294)

with self.assertRaisesRegex(ValueError, "fractional occupation"):
    audit_phase(fractional_case_dir, expected_mode="free")

summary = compare_zero_field_results(
    fixed_energy_ha=-5.419,
    free_energies_ha={0: -5.419004, 1: -5.419003, 2: -5.419002},
    fixed_drift_kcal=0.0002,
    free_drifts_kcal={0: 0.0003, 1: 0.0002, 2: 0.0004},
)
self.assertEqual(summary["status"], "PBE_GATE_PASSED")

with self.assertRaisesRegex(ValueError, "fixed/free energy"):
    compare_zero_field_results(
        fixed_energy_ha=-5.419,
        free_energies_ha={0: -5.418, 1: -5.419, 2: -5.419},
        fixed_drift_kcal=0.0,
        free_drifts_kcal={0: 0.0, 1: 0.0, 2: 0.0},
    )
```

- [ ] **Step 2: Run and verify RED**

Expected: failures because `audit_phase` and `compare_zero_field_results` are absent.

- [ ] **Step 3: Implement parsers and comparison**

Add immutable `PhaseResult`, strict input parsing, one-energy parsing, spin-block occupation parsing, and constants:

```python
HA_TO_KCAL_MOL = 627.5094740631
INTEGER_TOL = 1e-10
DRIFT_TOL_KCAL = 0.001
ENERGY_TOL_HA = 1e-5
```

`audit_phase(path, expected_mode)` must reject missing convergence markers,
multiple or nonfinite final energies, nonzero accepted fields, an `ocp` value
that disagrees with the explicit `fixed` or `free` expectation, fractional
occupations, and spin counts other than 3/1.
`compare_zero_field_results(...)` must check every drift, every fixed/free
difference, and the maximum pairwise free-direction difference before returning
`PBE_GATE_PASSED`.

Create `audit_gate.py` as a CLI that reads these final stages:

```text
runs/fixed/fixed_cold
runs/fixed/fixed_restart
runs/dir0/free_restart1
runs/dir0/free_restart2
runs/dir1/free_restart1
runs/dir1/free_restart2
runs/dir2/free_restart1
runs/dir2/free_restart2
```

It writes JSON atomically through a temporary file and a concise text summary containing all energies, occupations, drifts, hashes, and `status=PBE_GATE_PASSED`.

- [ ] **Step 4: Run all contract tests**

Run:

```bash
python -m unittest discover -s SIAB/example_C_sternheimer/pbe_reference_gate/tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add SIAB/example_C_sternheimer/pbe_reference_gate
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m "feat(siab): audit C PBE reference equivalence"
```

### Task 3: Prepare immutable branch directories

**Files:**
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/prepare_gate.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_gate_contract.py`

- [ ] **Step 1: Write failing staging tests**

Use a temporary directory with dummy UPF/orbital files.  Assert that:

```python
prepare_branch(root, branch="fixed", pseudo=pseudo, orbital=orbital)
self.assertTrue((root / "runs/fixed/fixed_cold/INPUT").is_file())
self.assertTrue((root / "runs/fixed/fixed_cold/STRU").is_file())

prepare_branch(root, branch="dir2", pseudo=pseudo, orbital=orbital)
self.assertIn("efield_dir 2", (root / "runs/dir2/field_seed/INPUT").read_text())

with self.assertRaises(FileExistsError):
    prepare_branch(root, branch="fixed", pseudo=pseudo, orbital=orbital)
```

The generated `STRU` must contain a 20 Angstrom cubic box and centered C, and every staged asset hash must be recorded in `BRANCH_PROVENANCE.json`.

- [ ] **Step 2: Run and verify RED**

Expected: import failure for `prepare_gate`.

- [ ] **Step 3: Implement `prepare_branch` and CLI**

The branch mapping is exactly:

```python
BRANCHES = {
    "fixed": ("fixed", None),
    "dir0": ("field", 0),
    "dir1": ("field", 1),
    "dir2": ("field", 2),
}
```

Create only the cold/seed phase.  Later restart directories are created by the
Slurm runner by copying the immediately preceding completed phase, preserving
ABACUS restart filenames.  Refuse any pre-existing branch directory to prevent
result mixing.  The CLI also exposes

```bash
prepare_gate.py render --mode fixed --restart > INPUT
prepare_gate.py render --mode free --field-dir 2 --restart > INPUT
```

so the runner rewrites a copied phase through the same tested renderer rather
than editing INPUT with regular expressions.

- [ ] **Step 4: Run all tests and verify GREEN**

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit Task 3**

```bash
git add SIAB/example_C_sternheimer/pbe_reference_gate
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m "feat(siab): prepare C PBE gate branches"
```

### Task 4: Add normal-partition Slurm execution

**Files:**
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch.slurm`
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/audit_gate.py`
- Modify: `docs/superpowers/specs/2026-08-21-c-atom-reference-equivalence-design.md`
- Modify: `docs/superpowers/plans/2026-08-21-c-atom-pbe-equivalence-gate.md`

- [x] **Step 1: Write failing static HPC tests**

```python
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_pbe_branch.slurm"


class HpcContractTests(unittest.TestCase):
    def test_normal_full_node_array_contract(self):
        text = RUNNER.read_text()
        for value in (
            "#SBATCH --partition=normal",
            "#SBATCH --array=0-3",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=126500M",
            "#SBATCH --time=24:00:00",
            "export OMP_NUM_THREADS=32",
        ):
            self.assertIn(value, text)
        self.assertNotIn("debug", text.lower())

    def test_free_restart_explicitly_removes_field_and_ocp(self):
        text = RUNNER.read_text()
        self.assertIn('prepare_gate.py" render --mode free', text)
        self.assertIn('grep -q "^ocp 0$" INPUT', text)
        self.assertIn('grep -q "^efield_flag 0$" INPUT', text)
        self.assertIn('grep -q "^init_wfc file$" INPUT', text)
        self.assertIn('grep -q "^init_chg file$" INPUT', text)
```

- [x] **Step 2: Run and verify RED**

Expected: `FileNotFoundError` for the runner.

- [x] **Step 3: Implement the Slurm runner**

The implemented runner and auditor must:

1. require numeric job/array identifiers and assert the live Slurm resource
   contract from a testable `scontrol show job -o` query before preparation,
   including memory, 24-hour time limit, and exclusive allocation; preserve
   the exact raw scheduler record and its SHA256 for independent revalidation;
2. resolve `ABACUS_ARTIFACT`, verify the executable and record its SHA256;
3. map array task 0 to `fixed`, and 1--3 to `dir0`--`dir2`;
4. call `prepare_gate.py` once successfully for its branch, using a bounded
   stable array-owned cross-node guard and bounded preparation mutex so the
   four array tasks do not collide on the preparation lock or race guard
   deletion;
5. run ABACUS through `mpirun -np 1 -ppn 1` with 32 OpenMP threads;
6. require convergence, final energy, `eig_occ.txt`, both spin wavefunctions,
   and charge restart output after every phase;
7. create the next restart phase atomically without copying the old output
   directory: copy only `STRU`, `KPT`, both assets and
   `wfs1_nao.txt`, `wfs2_nao.txt`, `chgs1.cube`, `chgs2.cube`;
8. preserve the four source restart files under a non-symlink phase-local
   `restart_input_snapshot/` and publish `RESTART_PROVENANCE.json` first as
   `PLANNED`;
9. require exactly two wavefunction-load messages in `abacus.stdout` and two
   charge-load messages in `running_scf.log`, each naming the exact canonical
   phase-local restart file after relative paths are resolved against the phase
   directory; accept equivalent absolute local paths and reject any escape,
   then upgrade restart provenance to `VERIFIED`;
10. write `PHASE_COMPLETE.json` only after the phase input, 22-band 3/1
    occupations, energy, executable, resources, outputs and restart evidence
    have been rehashed;
11. write `BRANCH_COMPLETE.json` atomically only after the complete fixed or
    field/free chain finishes;
12. let the global audit publish `PBE_GATE_PASSED` only after all 11 phases and
    all four branch manifests close the evidence chain with identical staged
    pseudopotential/orbital content and frozen preparation identity.  Any
    `RUN_FAILED.json` blocks the gate.  With no Task 4 evidence, the old Task 2
    fixtures remain `DIAGNOSTIC_ONLY`; partial or inconsistent evidence fails.

Use `set -euo pipefail`, inherited ERR traps, and a branch-local atomic
`RUN_FAILED.json` that records the failed line and command before exiting.
No branch writes the global scientific pass marker.  The readable restart
contract is fixed to `out_wfc_lcao=1`, `out_app_flag=1`, and the two spin text
files; binary wavefunction restart output is not accepted.

- [x] **Step 4: Run all local tests**

Run the discovery command.  Expected: all tests pass.

- [x] **Step 5: Commit Task 4**

```bash
git add SIAB/example_C_sternheimer/pbe_reference_gate
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m "feat(siab): run C PBE gate on normal nodes"
```

### Task 5: Add duplicate-safe submission and documentation

**Files:**
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/submit_pbe_gate.sh`
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/README.md`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`

- [x] **Step 1: Write failing submission tests**

Assert that the submitter checks `SUBMITTED_JOB_ID.txt`, queries both `squeue`
and `sacct`, rejects existing `PENDING`, `RUNNING`, `COMPLETING`, or
`COMPLETED` work, and submits exactly one array job.  It must not submit an
automatic Delta-ST dependency.

- [x] **Step 2: Run and verify RED**

Expected: failure because the submitter is absent.

- [x] **Step 3: Implement submitter and README**

The submitter requires these environment variables:

```text
GATE_ROOT=/work1/ghj/c-atom-pbe-equivalence-20260821
ABACUS_ARTIFACT=/work1/ghj/delta-st-unified-abacus-20260817/artifacts/build-21661442
PSEUDO_SOURCE=/work1/ghj/open-shell-fixed-occupation-20260820/assets/C_ONCV_PBE-1.0.upf
ORBITAL_SOURCE=/work1/ghj/open-shell-fixed-occupation-20260820/assets/C_gga_10au_100Ry_3s3p2d.orb
PYTHON_EXE=/public/home/ghj/.conda/envs/ds092/bin/python
SOURCE_COMMIT=<40-lowercase-hex-commit-used-to-create-the-source-archive>
```

It resolves every path, records hashes in `SUBMISSION_PROVENANCE.json`, and
submits the four-task PBE array.  The README explains that the audit is run on
the login node after all four tasks finish and that only
`status=PBE_GATE_PASSED` permits writing the Delta-ST plan.

Task 6 copies the committed gate directory as a standalone archive without a
`.git` directory and passes `SOURCE_COMMIT` explicitly.  Submission provenance
must include SHA256 and size records for all five runtime source files.

- [x] **Step 4: Run tests and shell syntax checks**

```bash
python -m unittest discover -s SIAB/example_C_sternheimer/pbe_reference_gate/tests -v
bash -n SIAB/example_C_sternheimer/pbe_reference_gate/submit_pbe_gate.sh
bash -n SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch.slurm
```

Expected: all tests and both syntax checks pass.

- [x] **Step 5: Commit Task 5**

```bash
git add SIAB/example_C_sternheimer/pbe_reference_gate
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m "docs(siab): package C PBE reference gate"
```

#### Task 5 review hardening

- [x] Remove runtime dependence on `.git`; require explicit `SOURCE_COMMIT` and
  required `PYTHON_EXE`.
- [x] Recheck formal evidence after the durable claim and final scheduler
  query, while accepting only the current claim identity.
- [x] Create and fsync exclusive receipt files before `sbatch`, then append to
  the same inodes and fsync them again after return.
- [x] Hash and size all five standalone runtime source files.
- [x] Add standalone, race, receipt-order, and runtime-file regression tests.
- [x] Commit the review fix separately without amending Task 5.

### Task 6: Stage and run the PBE gate on df_dcu

**Files:**
- Remote create: `/work1/ghj/c-atom-pbe-equivalence-20260821/source/`
- Remote create: `/work1/ghj/c-atom-pbe-equivalence-20260821/runs/`
- Remote result: `/work1/ghj/c-atom-pbe-equivalence-20260821/RESULT_SUMMARY.json`
- Remote result: `/work1/ghj/c-atom-pbe-equivalence-20260821/RESULT_SUMMARY.txt`
- Local create after audit: `SIAB/example_C_sternheimer/pbe_reference_gate/results/PBE_GATE_RESULT.md`

- [ ] **Step 1: Verify the authenticated df_dcu channel and live resources**

Use the existing OTP ControlMaster.  Query `sinfo` and verify that normal nodes
still provide the requested 32 CPUs and 126500 MB.  If the live shape differs,
update the resource constants and their tests before submission.

- [ ] **Step 2: Stage the committed source and immutable assets**

Copy only the committed `pbe_reference_gate` directory into `source/`.  Resolve
the ABACUS artifact and verify hashes before writing the submission manifest.

- [ ] **Step 3: Submit exactly one formal array**

Run `submit_pbe_gate.sh` once.  Record its job ID and immediately verify each
array task has partition `normal`, one node, one MPI rank, 32 CPUs, 126500 MB,
and 24 hours.  Do not resubmit while any formal task exists.

- [ ] **Step 4: Audit completed branches**

After all tasks complete, inspect scheduler state, exit codes, stderr, SCF
markers, final occupations, restart artifacts, and branch-complete manifests.
Then run:

```bash
$PYTHON_EXE source/audit_gate.py --root "$GATE_ROOT"
```

Expected successful terminal line:

```text
status=PBE_GATE_PASSED
```

If any branch fails, preserve it as `DIAGNOSTIC_ONLY`, identify the exact
failed phase, and do not run Delta-ST.

- [ ] **Step 5: Record measured result and commit the run record**

Add the measured energies, occupation vectors, energy drifts, scheduler IDs,
resource use, executable/asset hashes, and pass/fail decision to the project
TeX note and the compact repository result file.  Compile and inspect
`main.pdf`.  Commit the calculation record as a separate documentation commit;
do not mix it with source changes.

### Task 7: Final verification before Delta-ST planning

**Files:**
- Verify: all files under `SIAB/example_C_sternheimer/pbe_reference_gate/`
- Verify: `docs/superpowers/specs/2026-08-21-c-atom-reference-equivalence-design.md`

- [ ] **Step 1: Run the complete local test suite**

```bash
python -m unittest discover -s SIAB/example_C_sternheimer/pbe_reference_gate/tests -v
bash -n SIAB/example_C_sternheimer/pbe_reference_gate/submit_pbe_gate.sh
bash -n SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch.slurm
git diff --check
```

- [ ] **Step 2: Verify commit attribution and clean worktree**

Every new commit must show Author `Codex <codex@openai.com>` and Committer
`AroundPeking <gonghuanjing@iphy.ac.cn>`.  The worktree must be clean except
for explicitly documented unrelated user changes.

- [ ] **Step 3: Make the physical decision**

Only `PBE_GATE_PASSED` transitions to a new Delta-ST implementation plan.
`RUNNING`, `COMPLETED` without audit, fractional occupation, nonzero accepted
field, missing restart evidence, or energy disagreement all stop the workflow.
