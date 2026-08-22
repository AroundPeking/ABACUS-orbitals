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
- Create `SIAB/example_C_sternheimer/pbe_reference_gate/server66_runtime_env.sh`: minimal credential-free Intel/GCC runtime environment.
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
- Create: `SIAB/example_C_sternheimer/pbe_reference_gate/server66_runtime_env.sh`
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
4. source only the recorded environment script for both profiles;
5. reset `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to
   the profile's `cpus_per_task` value;
6. pass the profile, entrypoint, and common-runner paths to `audit_gate.py`.

Create `server66_runtime_env.sh` with `set -euo pipefail`, source
`/etc/profile.d/modules.sh`, purge inherited modules, load only `gcc10.2` and
`intel20u4`, and prepend the GCC 10.2 and Intel 2020 library directories.
It must not source `/home/ghj/.bashrc` or contain API keys, tokens, conda
initialization, aliases, or unrelated application paths.  A static test must
enforce those exclusions.  A clean-environment server66 probe must resolve
Intel MPI, show no missing ABACUS dynamic library, and report ABACUS
`v3.9.0.25`.

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

Require the exact `sbatch --export` value to be an allowlist without `ALL`.
It must contain only `GATE_ROOT`, `ABACUS_ARTIFACT`, `ABACUS_ENV_SCRIPT`,
`PSEUDO_ASSET`, `ORBITAL_ASSET`, `PYTHON_EXE`, `C_PBE_GATE_PROFILE`,
`C_PBE_GATE_ENTRYPOINT`, and `C_PBE_GATE_COMMON_RUNNER`.  Tests must seed the
submitter process with fake API-key, token, socket, conda, and unrelated path
variables, then prove none appears in the recorded command or fake batch
environment.  The spooled wrapper simulation must receive the two immutable
source paths through this allowlist.

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
and export the exact profile and immutable source paths to the array.  Replace
`--export=ALL,...` with the exact allowlist above so no ambient login
environment is forwarded.

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
server66 ABACUS executable and the staged committed
`server66_runtime_env.sh` as the recorded environment entrypoint.  Compare
all local and remote SHA256 values.

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
ABACUS_ENV_SCRIPT=<immutable-root>/source/server66_runtime_env.sh
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

## Formal execution record

On 2026-08-22, source commit `79c9c37755e85801c92ab281df1c2de35a1ff010`
passed the complete server66 preflight and was submitted exactly once as array
job `410615` under the immutable root
`/home/ghj/abacus/260822/c-atom-pbe-equivalence-server66-79c9c37755e8`.
The older df_dcu job `21709225` started and failed independently before it could
be cancelled; it produced no PBE phase results and was not resubmitted.

The server66 ABACUS calculations produced valid PBE output, including the fixed
cold and fixed-restart triplet states, but the array stopped at the restart
evidence audit.  ABACUS `v3.9.0.25` writes the path-bearing message
`Read electron density from file: <path>`, while the frozen auditor accepted
only `Read in electron density: <path>`.  This is an audit-parser compatibility
failure, not a demonstrated PBE or SCF failure.  Job `410615` and its root are
preserved as failed provenance and must never be reused.  The replacement run
requires a reviewed source commit, a new commit-derived immutable root, a fresh
preflight, and one new submission.  The physical gate remains pending until
that replacement run creates `PBE_GATE_PASSED`.

Source commit `6cae347f9e6f3767f6a7c5182bfda839c7520c93` then passed a
fresh 171-test preflight and was submitted exactly once as array job `410626`
under its own commit-derived root.  The `dir0` and `dir1` branches completed
with full restart evidence.  Two independent blockers remained:

1. server66 assigned the array parent number itself to task 3.  Querying
   `scontrol show job 410626` returned all four array records, whereas the
   canonical selector `410626_3` returned exactly the task-3 record.  The
   scheduler audit therefore must always query
   `<array_job_id>_<array_task_id>` and still verify the returned live job ID;
2. the zero-field fixed-occupation cold SCF genuinely failed to converge after
   300 iterations (`drho = 2.2274e-3`).  Its INPUT, STRU, KPT, executable,
   resources, and node matched the earlier converged run, but the trajectories
   differed from iteration 1.  The audit correctly rejected this result; no
   parser relaxation is allowed.  This degenerate open-shell convergence issue
   requires a separate one-variable physical stabilization test before another
   formal gate run.

Job `410626` and its root are failed provenance and must not be reused.  Fixing
the scheduler selector alone does not authorize a replacement formal run until
the fixed-occupation SCF protocol is reproducibly converged.

## Fixed-occupation stabilization record

The zero-field fixed cold start was not reproducible because the degenerate C
2p subspace selected different numerical trajectories from the first SCF
iteration.  Two one-variable mixing tests did not remove that instability:

| Diagnostic job | Change from frozen protocol | Repetition 0 | Repetition 1 | Decision |
| --- | --- | --- | --- | --- |
| `410636` | `mixing_beta=0.1` | converged in 154 iterations, `-147.4773363622988 eV` | failed at 300 iterations | reject |
| `410645` | `mixing_beta=0.1`, `mixing_gg0=0`, `mixing_gg0_mag=0` | failed at 300 iterations, `drho=9.02523e-4` | converged in 44 iterations, `-147.477336362294 eV` | reject |

The selected replacement uses a fixed-occupation field seed followed by a
zero-field fixed restart.  Diagnostic job `410652` ran two independent
repetitions with one node, one MPI rank, 48 OpenMP threads, and 180000 MB:

| Repetition | `fixed_field_seed` | `fixed_zero_restart` | Final zero-field energy |
| --- | --- | --- | ---: |
| 0 | converged in 45 iterations, final `drho=9.8875e-11` | converged in 29 iterations, final `drho=5.9385e-11` | `-147.4773363622931015 eV` |
| 1 | converged in 43 iterations, final `drho=9.8879e-11` | converged in 28 iterations, final `drho=4.8962e-11` | `-147.4773363622958868 eV` |

Both final zero-field states have exactly 3 up / 1 down integer occupations.
Their energies agree with each other and with the completed free routes to
about `1.6e-10 kcal/mol`, far inside the unchanged acceptance thresholds.  The
field amplitude is `1e-4` and acts only as an orientation selector.  The
field-seed energy is excluded from the final physical comparison; only the
zero-field `fixed_zero_restart` energy is compared with the three zero-field
`free_restart2` energies.  The seed-to-restart energy change remains an
independent drift check.

Commits `f852faaff4045bc05f772c9e753f0de92f3382ae`,
`886f1d31983f6c88536a6cd48326a6d8b726acfa`, and
`71f6b355c4e1f00a1bdb73fd12f48bc5f2db1a28` implement and test the input,
preparation/audit, and runtime stages respectively.  The complete local fake
HPC suite passes 176 tests and closes all 11 phase, restart, scheduler, and
runtime provenance records.  A formal server66 calculation still requires a
new documentation-complete source commit, a new commit-derived immutable root,
a full preflight, duplicate checks, and exactly one new array submission.

## Formal replacement result

The documentation-complete source commit
`7527d03bb1875cea04e9a3ec415060276d8a5ea7` passed a fresh server66 preflight
with 176 tests, shell and Python checks, exact source/artifact hashes, and an
accepted `sbatch --test-only` resource shape.  The preflight and duplicate
checks found no root, claim, active task, or accounting record for stable job
name `c_pbe_gate_507c214ec6a7`.

Exactly one formal four-task array was then submitted as job `410668` under
`/home/ghj/abacus/260822/c-atom-pbe-equivalence-server66-7527d03bb187`.
All four task records reached `COMPLETED` with exit code `0:0`; elapsed times
were 1:37, 2:14, 2:11, and 2:25 for `fixed`, `dir0`, `dir1`, and `dir2`.
Every one of the 11 SCF phases converged with exact 3 up / 1 down integer
occupations and complete restart-load evidence.

The login-node global audit returned `PBE_GATE_PASSED` and
`RESTART_CHAIN_VERIFIED`.  The fixed zero-field energy is
`-147.4773363622957 eV`.  Its difference from the three final free-route
energies is at most `1.252331571777177e-13 Ha`; the largest free/free spread is
`1.394440118929197e-13 Ha`.  The fixed seed-to-zero-restart drift is
`3.256647533451255e-5 kcal/mol`, below the frozen `0.001 kcal/mol` threshold.
The complete result is recorded in
`SIAB/example_C_sternheimer/pbe_reference_gate/results/PBE_GATE_RESULT.md`.

The C PBE reference gate is therefore complete.  A separate C Delta-ST plan
is now authorized; this result does not itself establish a Delta-ST or RPA
response.
