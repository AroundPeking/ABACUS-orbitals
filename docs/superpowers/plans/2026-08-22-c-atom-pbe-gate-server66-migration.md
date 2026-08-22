# C Atom PBE Gate Server66 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the unchanged C atom PBE reference-equivalence gate on server66 with a fully audited server66 Slurm profile while preserving the existing df_dcu profile.

**Architecture:** Put immutable scheduler values in one Python resource-profile module imported by the auditor and queried by the shell runtime.  Use two small Slurm entrypoints, one per cluster, which select a profile and source one shared branch runner.  Submission and phase provenance record both the selected entrypoint and shared runner so no executed source is outside the hash chain.

**Tech Stack:** Python 3.7 standard library, `unittest`, Bash, Slurm 19.05/22.05, Intel MPI, ABACUS LCAO PBE.

---

## File map

- Create `SIAB/example_C_sternheimer/pbe_reference_gate/resource_profiles.py`: immutable scheduler contracts and CLI serialization.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch_common.sh`: shared four-branch execution and restart logic.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch.slurm`: thin df_dcu Slurm entrypoint.
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch_server66.slurm`: thin server66 Slurm entrypoint.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/audit_gate.py`: profile-aware scheduler and runtime-source validation.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/submit_pbe_gate.sh`: explicit profile selection and complete source provenance.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`: profile, scheduler, wrapper, provenance, and end-to-end tests.
- Modify `SIAB/example_C_sternheimer/pbe_reference_gate/README.md`: exact df_dcu/server66 run contracts and migration rule.

### Task 1: Define immutable scheduler profiles

**Files:**
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/resource_profiles.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`

- [ ] **Step 1: Write failing profile tests**

Add tests that import `get_resource_profile` and assert the exact contracts:

```python
self.assertEqual(
    get_resource_profile("df_dcu"),
    {
        "name": "df_dcu", "partition": "normal", "nodes": 1,
        "ntasks": 1, "cpus_per_task": 30, "memory_mb": 110610,
        "time_limit": "1-00:00:00", "over_subscribe": "NO",
    },
)
self.assertEqual(
    get_resource_profile("server66"),
    {
        "name": "server66", "partition": "640", "nodes": 1,
        "ntasks": 1, "cpus_per_task": 48, "memory_mb": 180000,
        "time_limit": "1-00:00:00", "over_subscribe": "OK",
    },
)
with self.assertRaisesRegex(ValueError, "unknown C PBE gate profile"):
    get_resource_profile("automatic")
```

Also test that

```bash
python resource_profiles.py shell server66
```

prints exactly one pipe-separated line:

```text
server66|640|1|1|48|180000|1-00:00:00|OK
```

- [ ] **Step 2: Run the profile tests and verify RED**

Run:

```bash
python3 -m unittest \
  SIAB.example_C_sternheimer.pbe_reference_gate.tests.test_hpc_contract.HpcStaticContractTests -v
```

Expected: import failure because `resource_profiles.py` does not exist.

- [ ] **Step 3: Implement the profile module**

Create a Python-3.7-compatible module with an immutable private mapping, a
copy-returning `get_resource_profile(name)` API, and a `shell` CLI.  Reject
missing, inferred, or unknown profiles; emit only the eight fields above.

- [ ] **Step 4: Run the focused and full tests**

Run the focused command, then:

```bash
python3 -m unittest discover \
  -s SIAB/example_C_sternheimer/pbe_reference_gate/tests -v
```

Expected: the new profile tests pass and all pre-existing tests remain green.

- [ ] **Step 5: Commit Task 1**

Commit only the profile module and its tests as:

```text
feat(siab): define C PBE gate runtime profiles
```

### Task 2: Split Slurm entrypoints from shared execution

**Files:**
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch_common.sh`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch.slurm`
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/run_pbe_branch_server66.slurm`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/audit_gate.py`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`

- [ ] **Step 1: Write failing wrapper and scheduler tests**

Require the df_dcu wrapper to contain exactly its existing Slurm shape and:

```bash
export C_PBE_GATE_PROFILE=df_dcu
export C_PBE_GATE_ENTRYPOINT="${BASH_SOURCE[0]}"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/run_pbe_branch_common.sh"
```

Require the server66 wrapper to contain:

```bash
#SBATCH --partition=640
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180000M
#SBATCH --time=24:00:00
#SBATCH --no-requeue
export C_PBE_GATE_PROFILE=server66
```

Add server66 scheduler fixtures with `Partition=640`, `NumCPUs=48`,
`CPUs/Task=48`, `MinMemoryNode=180000M`, and `OverSubscribe=OK`.  Verify that
each profile accepts only its own environment and raw `scontrol` record and
rejects cross-profile CPU, memory, partition, or oversubscribe evidence.

- [ ] **Step 2: Run the new tests and verify RED**

Run the static and runtime scheduler test classes.  Expected: failures because
the server66 entrypoint and profile-aware auditor are absent.

- [ ] **Step 3: Extract the shared runner**

Move all executable content below the current df_dcu `#SBATCH` block into
`run_pbe_branch_common.sh`.  Keep the df_dcu file as a thin entrypoint and add
the thin server66 entrypoint.  The common runner must:

1. resolve `resource_profiles.py`, the entrypoint, and itself as non-symlink
   regular files;
2. obtain the selected exact contract from `resource_profiles.py shell`;
3. validate Slurm environment values against that contract;
4. source `/etc/profile.d/modules.sh` before the recorded environment script
   only for `server66`;
5. reset `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to
   the profile's `cpus_per_task` value;
6. pass the profile, entrypoint, and common-runner paths to `audit_gate.py`.

- [ ] **Step 4: Make scheduler evidence profile-aware**

In `audit_gate.py`, import `get_resource_profile`, require
`C_PBE_GATE_PROFILE`, and put `profile` in every scheduler record.  Validate
environment and raw `scontrol` fields against the selected profile rather
than module-level df_dcu constants.  Extend runtime provenance with:

```json
{
  "gate_profile": "server66",
  "entrypoint": {"path": "...", "size": 0, "sha256": "..."},
  "common_runner": {"path": "...", "size": 0, "sha256": "..."}
}
```

The independent audit must rehash both files and require one profile and one
runtime chain across all eleven phases and four branches.

- [ ] **Step 5: Verify focused and full tests**

Run the new wrapper/scheduler tests, `bash -n` on all three shell files,
`python3 -m py_compile` on every Python file, and the complete unit suite.

- [ ] **Step 6: Commit Task 2**

Commit the wrappers, common runner, auditor, and tests as:

```text
feat(siab): run C PBE gate with audited cluster profiles
```

### Task 3: Select and preserve the profile at submission

**Files:**
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/submit_pbe_gate.sh`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`

- [ ] **Step 1: Write failing submission tests**

Require `GATE_PROFILE` as an explicit environment variable.  Test rejection
of missing and unknown values.  For `df_dcu`, require the df_dcu entrypoint;
for `server66`, require the server66 entrypoint.  In both cases require the
export map to include:

```text
C_PBE_GATE_PROFILE=<selected profile>
```

Require `SUBMISSION_PROVENANCE.json` to contain the selected profile plus
file records for `resource_profiles.py`, the selected entrypoint, and
`run_pbe_branch_common.sh`.  Replacing any of those files after resolution
must stop submission.

- [ ] **Step 2: Run submission tests and verify RED**

Run:

```bash
python3 -m unittest \
  SIAB.example_C_sternheimer.pbe_reference_gate.tests.test_hpc_contract.SubmissionContractTests -v
```

Expected: failures because `GATE_PROFILE` is not consumed or recorded.

- [ ] **Step 3: Implement explicit submitter selection**

Resolve `resource_profiles.py`, the selected entrypoint, and the common runner
before the immutable claim.  Select only `df_dcu` or `server66`; never infer
from hostname.  Add their hashes and `gate_profile` to submission provenance,
and export the exact profile to the array.

- [ ] **Step 4: Verify focused and full tests**

Run the submission test class and then all tests.  Expected: one fake array
submission, complete source provenance, and no change to duplicate protection.

- [ ] **Step 5: Commit Task 3**

Commit as:

```text
feat(siab): submit C PBE gate by explicit cluster profile
```

### Task 4: Document and verify the complete implementation

**Files:**
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/README.md`
- Modify: `SIAB/example_C_sternheimer/pbe_reference_gate/tests/test_hpc_contract.py`

- [ ] **Step 1: Add a failing documentation contract test**

Require the README to name both profiles, both exact resource shapes, the
server66 executable hash, the identical C asset hashes, the effective
full-node interpretation of `OverSubscribe=OK`, and the rule that df_dcu job
`21709225` is cancelled only after server66 preflight passes.

- [ ] **Step 2: Run and verify RED**

Expected: the README test fails on missing server66 instructions.

- [ ] **Step 3: Update the README**

Document exact environment variables and commands for both profiles.  State
that scheduler completion is not physical acceptance and that only the global
audit may create `PBE_GATE_PASSED`.

- [ ] **Step 4: Run final local verification**

Run:

```bash
python3 -m unittest discover \
  -s SIAB/example_C_sternheimer/pbe_reference_gate/tests -v
bash -n SIAB/example_C_sternheimer/pbe_reference_gate/*.sh \
  SIAB/example_C_sternheimer/pbe_reference_gate/*.slurm
python3 -m py_compile SIAB/example_C_sternheimer/pbe_reference_gate/*.py
git diff --check
```

Expected: every test and static check passes with a clean worktree except for
the intended README/test changes.

- [ ] **Step 5: Commit Task 4**

Commit as:

```text
docs(siab): document C PBE gate cluster profiles
```

### Task 5: Stage, migrate, and submit one formal server66 run

**Files:**
- Create remotely: `/home/ghj/abacus/260822/c-atom-pbe-equivalence-server66-<source-hash>/`
- Preserve remotely: `/work1/ghj/c-atom-pbe-equivalence-20260822-r3/`

- [ ] **Step 1: Stage an immutable source archive on server66**

Archive the exact implementation commit and extract it into a new root.  Copy
the two local SG15 assets whose hashes match the frozen values.  Use the
server66 ABACUS executable and `/home/ghj/.bashrc` as the recorded environment
entrypoint.  Compare all local and remote SHA256 values.

- [ ] **Step 2: Run server66 preflight**

Set an executable temporary directory under the new root.  Run the complete
unit suite with server66 Python, shell syntax checks, Python compilation, and
the exact `sbatch --test-only` four-task array using profile `server66`.

- [ ] **Step 3: Recheck and cancel the exact df_dcu job**

Query `squeue`, `sacct`, elapsed time, and the r3 output tree for job
`21709225`.  Proceed only if it is still pending with zero runtime and no
physical output.  Cancel exactly `21709225`, then require accounting state
`CANCELLED` and zero elapsed time.  Preserve the root and all provenance.

- [ ] **Step 4: Submit exactly one server66 array**

Immediately recheck the canonical root, stable job name, `squeue`, and
`sacct`.  Run the profile-aware submitter once with:

```text
GATE_PROFILE=server66
ABACUS_ARTIFACT=/home/ghj/abacus/260809/sternheimer-solid-delta/artifacts/abacus-407979/abacus
ABACUS_ENV_SCRIPT=/home/ghj/.bashrc
PYTHON_EXE=/home/ghj/app/miniconda3/bin/python3
```

and the staged asset paths and exact source commit.

- [ ] **Step 5: Verify formal submission evidence**

Check the numeric job ID, four array tasks, partition `640`, one node and one
rank per task, 48 CPUs, 180000 MB, 24-hour limit, `OverSubscribe=OK`, exact
source and runtime hashes, absence of ambiguity/failure markers, and immediate
log state.  Do not submit again if any claim or receipt exists.

- [ ] **Step 6: Audit the completed physical gate**

After all four tasks reach scheduler `COMPLETED` with exit code `0:0`, run the
global audit on the login node.  Report scheduler, numerical, and physical
gates separately.  Only `status=PBE_GATE_PASSED` permits writing a separate C
Delta-ST implementation plan.
