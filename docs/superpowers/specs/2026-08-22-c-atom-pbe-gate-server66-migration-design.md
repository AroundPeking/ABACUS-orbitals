# C atom PBE gate migration to server66

## Purpose

Move the already-defined C atom PBE reference-equivalence gate from the
queued `df_dcu` run to server66 without changing its physical question,
inputs, phase graph, or acceptance thresholds.  The server66 run must remain
a four-branch comparison between the fixed integer-occupation route and the
three weak-field-seeded, zero-field free-restart routes.

This migration changes only the runtime artifact and scheduler resource
profile.  It does not authorize Delta-Sternheimer or SIAB optimization before
the PBE gate passes.

## Frozen physics

The following definition remains byte-for-byte or numerically identical to
the existing gate:

- SG15 `C_ONCV_PBE-1.0.upf`, SHA256
  `e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99`;
- SG15 TZDP-10au `C_gga_10au_100Ry_3s3p2d.orb`, SHA256
  `7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d`;
- 20 Angstrom cubic cell, centered C atom, Gamma point, PBE, `nspin = 2`,
  `nelec = 4`, `nupdown = 2`, `nbands = 22`;
- 30 Ry wavefunction cutoff and the exact `135 x 135 x 135` grid;
- the eleven phases, restart files, occupation checks, energy thresholds,
  and required restart-load evidence defined in the existing design.

Every server66 branch uses one identical ABACUS executable.  The selected
server66-native executable is

```text
/home/ghj/abacus/260809/sternheimer-solid-delta/artifacts/abacus-407979/abacus
```

It reports ABACUS `v3.9.0.25` and has SHA256
`27722d5e3e5cf2c94d00ac9489152b7ea00adcf51a8b8bb3a8eed3d8d094c279`.
It is intentionally not copied from `df_dcu`: the two clusters use different
Intel MPI/MKL installations.  The PBE equivalence statement is internal to
the four server66 branches, so identical source-level physics and one common
server66 executable are required; binary identity across clusters is not.

The server66 environment entrypoint is a committed, minimal shell file that
sources `/etc/profile.d/modules.sh`, purges inherited modules, loads only
`gcc10.2` and `intel20u4`, and adds the required GCC and Intel library
directories.  It must not source or copy `/home/ghj/.bashrc`: that file
contains unrelated interactive settings and credentials that must not enter
the formal calculation environment.  The minimal entrypoint is hashed in the
same provenance chain as the ABACUS executable and resolved `mpirun`.

## Runtime profiles

The gate gains an explicit runtime-profile interface instead of replacing the
existing `df_dcu` constants.  Exactly two profiles are admitted:

| Profile | Partition | Nodes/task | MPI ranks | OpenMP threads | Memory | Limit |
|---|---:|---:|---:|---:|---:|---:|
| `df_dcu` | `normal` | 1 | 1 | 30 | 110610 MB | 24 h |
| `server66` | `640` | 1 | 1 | 48 | 180000 MB | 24 h |

The four array tasks remain `fixed`, `dir0`, `dir1`, and `dir2`.  On
server66, requesting all 48 configured CPUs and all 180000 MB on a node makes
the allocation effectively full-node even though Slurm 19.05 reports
`OverSubscribe=OK`.  Therefore the server66 audit verifies the complete CPU
and memory TRES rather than requiring the df_dcu-specific
`OverSubscribe=NO` token.  The committed server66 runner must request exactly
that shape and reject any live mismatch before creating a branch directory.

The profile is chosen explicitly at submission and recorded in submission,
phase, and branch provenance.  It is never inferred from the hostname.  All
four branches must record one identical profile and runtime chain.

## Source and scheduler layout

Keep the existing physical renderer, restart machinery, phase parser, and
global acceptance audit shared.  Add a server66 Slurm runner containing only
server66 resource directives and export the explicit profile name to the
shared runtime logic.  The scheduler validator reads the selected immutable
profile, then validates both Slurm environment variables and the live
`scontrol show job -o` record.

The submitter accepts only a known profile and chooses the corresponding
committed runner.  The stable job name includes the canonical gate root, so a
server66 retry cannot collide with the cancelled df_dcu history.  Submission
provenance records the profile, runner hash, executable hash, asset hashes,
exact command, job ID, and resolved paths.

The formal server66 root is a new immutable directory under
`/home/ghj/abacus/260822/`.  Existing df_dcu roots and their cancellation
records are retained and never reused.

## Migration order and duplicate prevention

1. Implement and locally test the explicit server66 profile without touching
   the frozen physical input contract.
2. Stage the exact source commit, executable, committed minimal environment
   entrypoint, and C assets under a new server66 root.
3. Run all unit tests with the server66 Python and run `sbatch --test-only`
   for the exact four-task array resource shape.
4. Recheck that df_dcu job `21709225` is still pending and has produced no
   branch output.
5. Cancel exactly `21709225` and verify `CANCELLED`, zero elapsed runtime, and
   no physical output.
6. Recheck that no server66 job or immutable submission evidence exists for
   the new root, then submit exactly once.
7. Immediately verify the server66 job ID, four array tasks, resource shape,
   source commit, hashes, and absence of failure markers.

If server66 preflight fails, leave `21709225` untouched.  If cancellation is
confirmed but server66 submission becomes ambiguous, retain the immutable
claim and receipt and do not retry the same root.

## Verification and physical gate

Tests must cover both resource profiles, rejection of cross-profile scheduler
evidence, selection of the correct committed runner, and preservation of the
profile in all provenance records.  Existing 144 tests must continue to pass,
and new server66 tests must fail before implementation and pass afterward.

After all four server66 tasks finish, scheduler `COMPLETED` and exit code
`0:0` are only execution evidence.  The independent global audit must still
verify SCF convergence, complete 22-band spin blocks, integer 3/1
occupations, restart loading, energy drift, fixed/free agreement, and
cross-direction agreement.  Only `PBE_GATE_PASSED` permits a separate C
Delta-ST plan.
