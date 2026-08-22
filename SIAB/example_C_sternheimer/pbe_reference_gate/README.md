# Carbon PBE Reference Gate

## Physical purpose

This gate establishes a unique neutral carbon triplet PBE reference before any
carbon Delta-ST response or SIAB basis optimization is attempted.  It checks
that two independent constructions reach the same zero-field state:

1. a cold calculation with fixed integer occupation, followed by one fixed
   zero-field restart; and
2. three weak-field seeds along the Cartesian directions, each followed by two
   zero-field free restart calculations with the field and fixed occupation
   removed.

The accepted state has four valence electrons, `nspin = 2`, `nupdown = 2`, and
integer spin populations 3 and 1.  Every route uses the same 20 Angstrom cubic
cell, centered C atom, `135 x 135 x 135` real-space grid, 30 Ry cutoff, SG15
pseudopotential, TZDP-10au orbital, PBE functional, and Gamma-only sampling.
This is a PBE reference test; it does not run Delta-ST, LibRPA, or basis
optimization.

## Scheduler contract

The explicit `GATE_PROFILE` selects one of two committed resource profiles:

| Profile | Partition | Nodes | MPI ranks | OpenMP threads | Memory per node | Time | Live Slurm token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `df_dcu` | `normal` | 1 node | 1 MPI rank | 30 OpenMP threads | 110610 MB | 24 hours | `OverSubscribe=NO` |
| `server66` | `640` | 1 node | 1 MPI rank | 48 OpenMP threads | 180000 MB | 24 hours | `OverSubscribe=OK` |

Each profile submits one four-task array whose tasks run `fixed`, `dir0`,
`dir1`, and `dir2`.  The runner sources the selected pinned environment before
resolving `mpirun`, resets all thread counts to the profile value, and checks
the live Slurm allocation before creating calculation branches.

On df_dcu's Slurm 22.05 installation, the committed `--exclusive` request is
reported as `OverSubscribe=NO`.  On server66, `OverSubscribe=OK` is the
cluster's live presentation after Slurm has granted the full 48-CPU and 180000-MB allocation;
it does not mean this gate shares either requested resource.  The runtime audit
requires the exact profile-specific token and the complete CPU and memory allocation.

The submitter uses a stable job name, checks both `squeue` and `sacct`, and
creates an immutable submission claim before calling `sbatch`.  It submits
exactly one array and never adds a Delta-ST dependency.  An existing claim,
job ID, branch, result, or failure marker blocks reuse of the gate root.  If a
scheduler query or submission receipt is ambiguous, do not retry that root;
preserve it for diagnosis and use a separately reviewed new root if a new run
is authorized.

## Submission

Set canonical absolute paths.  The ABACUS executable, `ABACUS_ENV_SCRIPT`,
pseudopotential, and orbital must be nonempty local regular files, not
symbolic links.  The submitter resolves the required Python interpreter to its
real executable.  On `df_dcu`, `ABACUS_ENV_SCRIPT` is the validated Intel
MPI/MKL setup used by the existing C calculations; the login-node OpenMPI
environment is not compatible with this ABACUS artifact.
On server66, use the committed minimal `server66_runtime_env.sh`.  It sources
`/etc/profile.d/modules.sh`, runs `module purge`, then runs
`module load gcc10.2` and `module load intel20u4`, and prepends only their GCC
and Intel runtime directories to `LD_LIBRARY_PATH`.  It does not source the
user shell startup file or import unrelated settings.

The server66 preflight and formal run must use these immutable artifacts:

| Artifact | Canonical path after staging | SHA256 |
| --- | --- | --- |
| ABACUS | `/home/ghj/abacus/260809/sternheimer-solid-delta/artifacts/abacus-407979/abacus` | `27722d5e3e5cf2c94d00ac9489152b7ea00adcf51a8b8bb3a8eed3d8d094c279` |
| C SG15 pseudopotential | `<immutable-root>/assets/C_ONCV_PBE-1.0.upf` | `e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99` |
| C TZDP-10au orbital | `<immutable-root>/assets/C_gga_10au_100Ry_3s3p2d.orb` | `7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d` |

The staging step packages this directory as a standalone source archive.  That archive
does not require `.git`; `SOURCE_COMMIT` is the exact 40-character lowercase
Git commit used to create it and is supplied explicitly by the staging step.

For df_dcu, set the profile and canonical paths explicitly:

```bash
export GATE_ROOT=/work1/ghj/c-atom-pbe-equivalence-20260821
export ABACUS_ARTIFACT=/work1/ghj/delta-st-unified-abacus-20260817/artifacts/build-21661442/abacus_3p
export ABACUS_ENV_SCRIPT=/public/home/ghj/app/src/env_60_245_intel2021.sh
export PSEUDO_SOURCE=/work1/ghj/open-shell-fixed-occupation-20260820/assets/C_ONCV_PBE-1.0.upf
export ORBITAL_SOURCE=/work1/ghj/open-shell-fixed-occupation-20260820/assets/C_gga_10au_100Ry_3s3p2d.orb
export PYTHON_EXE=/public/home/ghj/.conda/envs/ds092/bin/python
export SOURCE_COMMIT=0123456789abcdef0123456789abcdef01234567

GATE_PROFILE=df_dcu ./submit_pbe_gate.sh
```

For server66, set its own canonical root, executable, environment script,
assets, Python executable, and source commit, then submit explicitly:
The staging must replace every angle-bracket placeholder before submission;
placeholders must not be submitted literally.

```bash
export GATE_ROOT='/home/ghj/abacus/260822/c-atom-pbe-equivalence-server66-<source-hash>'
export ABACUS_ARTIFACT=/home/ghj/abacus/260809/sternheimer-solid-delta/artifacts/abacus-407979/abacus
export ABACUS_ENV_SCRIPT="$GATE_ROOT/source/server66_runtime_env.sh"
export PSEUDO_SOURCE="$GATE_ROOT/assets/C_ONCV_PBE-1.0.upf"
export ORBITAL_SOURCE="$GATE_ROOT/assets/C_gga_10au_100Ry_3s3p2d.orb"
export PYTHON_EXE=/home/ghj/app/miniconda3/bin/python3
export SOURCE_COMMIT='<exact-40-hex-source-commit>'

GATE_PROFILE=server66 ./submit_pbe_gate.sh
```

Successful submission creates immutable `SUBMITTED_JOB_ID.txt` and atomic
`SUBMISSION_PROVENANCE.json`.  The provenance records the resolved paths,
file sizes and SHA256 hashes, source commit, exact `sbatch` command, job ID,
and UTC submission time.  It also records and exports the canonical
`ABACUS_ENV_SCRIPT`.  Runtime provenance covers `gate_contract.py`,
`prepare_gate.py`, `audit_gate.py`, `resource_profiles.py`, the selected profile-specific entrypoint
(`run_pbe_branch.slurm` or
`run_pbe_branch_server66.slurm`), `run_pbe_branch_common.sh`, and
`submit_pbe_gate.sh`; each must be a nonempty non-symlink regular file.  The
branch, phase, and global evidence independently rehash the complete source
chain, environment script, ABACUS executable, and canonical `mpirun` selected
after the script is sourced.  All four branches must contain identical
records.  The
durable receipt files are created exclusively and fsynced before `sbatch`
starts, and remain under `.submission-claim/` when submission is ambiguous.

The `sbatch --export` value is an exact allowlist containing only
`GATE_ROOT`, `ABACUS_ARTIFACT`, `ABACUS_ENV_SCRIPT`, `PSEUDO_ASSET`,
`ORBITAL_ASSET`, `PYTHON_EXE`, `C_PBE_GATE_PROFILE`,
`C_PBE_GATE_ENTRYPOINT`, and `C_PBE_GATE_COMMON_RUNNER`.  Submission must not use `ALL`;
unrelated login-shell variables, credentials, and agent sockets are not part of
the batch environment contract.

For the approved migration, cancel df_dcu job 21709225 only after the complete server66 preflight has passed.
Before cancellation, recheck that the df_dcu job is still pending
with zero elapsed time and has produced no physical output.  Preserve its
immutable root and provenance, and submit the new server66 formal root exactly
once.

## Post-completion audit

After all four array tasks have left the scheduler, run the audit once on the
login node; postprocessing does not need a compute allocation:

```bash
"$PYTHON_EXE" audit_gate.py --root "$GATE_ROOT"
```

Inspect both `RESULT_SUMMARY.json` and `RESULT_SUMMARY.txt`.

A scheduler completion is not a physical pass.  Only the login-node global audit
can create `PBE_GATE_PASSED` after all branches have left the scheduler and all
physical and provenance checks have passed.

- `DIAGNOSTIC_ONLY` means the scalar numerical comparison may be readable, but
  the complete four-branch runtime and restart evidence is absent.  It is not
  a physical pass.
- `PBE_GATE_PASSED` means all 11 phases, restart-load evidence, integer triplet
  occupations, fixed/free energy agreement, assets, executable, and resource
  provenance passed the frozen contract, including identical Intel MPI/MKL
  environment-script and `mpirun` hashes in all branches.
- `PBE_GATE_FAILED` or any `RUN_FAILED.json` stops the workflow and must be
  diagnosed without weakening the acceptance thresholds.

Only `PBE_GATE_PASSED` permits writing and executing a separate later Delta-ST
equivalence plan.  Passage does not start Delta-ST automatically.
