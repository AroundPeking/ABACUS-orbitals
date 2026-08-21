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

`run_pbe_branch.slurm` is one four-task array on the `normal` partition.  Each
task receives one exclusive node, one MPI rank, 32 OpenMP threads, 126500 MB,
and 24 hours.  Array tasks run `fixed`, `dir0`, `dir1`, and `dir2`.  The runner
checks the live Slurm allocation before creating calculation branches.

The submitter uses a stable job name, checks both `squeue` and `sacct`, and
creates an immutable submission claim before calling `sbatch`.  It submits
exactly one array and never adds a Delta-ST dependency.  An existing claim,
job ID, branch, result, or failure marker blocks reuse of the gate root.  If a
scheduler query or submission receipt is ambiguous, do not retry that root;
preserve it for diagnosis and use a separately reviewed new root if a new run
is authorized.

## Submission

Set canonical absolute paths.  The ABACUS executable, pseudopotential, and
orbital must be nonempty local regular files, not symbolic links.  The
submitter resolves the required Python interpreter to its real executable.
Task 6 stages this directory as a standalone source archive.  That archive
does not require `.git`; `SOURCE_COMMIT` is the exact 40-character lowercase
Git commit used to create it and is supplied explicitly by the staging step.

```bash
export GATE_ROOT=/work1/ghj/c-atom-pbe-equivalence-20260821
export ABACUS_ARTIFACT=/work1/ghj/delta-st-unified-abacus-20260817/artifacts/build-21661442
export PSEUDO_SOURCE=/work1/ghj/open-shell-fixed-occupation-20260820/assets/C_ONCV_PBE-1.0.upf
export ORBITAL_SOURCE=/work1/ghj/open-shell-fixed-occupation-20260820/assets/C_gga_10au_100Ry_3s3p2d.orb
export PYTHON_EXE=/public/home/ghj/.conda/envs/ds092/bin/python
export SOURCE_COMMIT=0123456789abcdef0123456789abcdef01234567

./submit_pbe_gate.sh
```

Successful submission creates immutable `SUBMITTED_JOB_ID.txt` and atomic
`SUBMISSION_PROVENANCE.json`.  The provenance records the resolved paths,
file sizes and SHA256 hashes, source commit, exact `sbatch` command, job ID,
and UTC submission time.  Runtime provenance covers `gate_contract.py`,
`prepare_gate.py`, `audit_gate.py`, `run_pbe_branch.slurm`, and
`submit_pbe_gate.sh`; each must be a nonempty non-symlink regular file.  The
durable receipt files are created exclusively and fsynced before `sbatch`
starts, and remain under `.submission-claim/` when submission is ambiguous.

## Post-completion audit

After all four array tasks have left the scheduler, run the audit once on the
login node; postprocessing does not need a compute allocation:

```bash
"$PYTHON_EXE" audit_gate.py --root "$GATE_ROOT"
```

Inspect both `RESULT_SUMMARY.json` and `RESULT_SUMMARY.txt`.

- `DIAGNOSTIC_ONLY` means the scalar numerical comparison may be readable, but
  the complete four-branch runtime and restart evidence is absent.  It is not
  a physical pass.
- `PBE_GATE_PASSED` means all 11 phases, restart-load evidence, integer triplet
  occupations, fixed/free energy agreement, assets, executable, and resource
  provenance passed the frozen contract.
- `PBE_GATE_FAILED` or any `RUN_FAILED.json` stops the workflow and must be
  diagnosed without weakening the acceptance thresholds.

Only `PBE_GATE_PASSED` permits writing and executing a separate later Delta-ST
equivalence plan.  Passage does not start Delta-ST automatically.
