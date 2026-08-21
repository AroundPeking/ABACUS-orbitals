# C atomic reference equivalence gate for Sternheimer-SIAB

## Purpose

Before generating carbon Sternheimer training data, establish that two
independent constructions of the neutral C atomic PBE reference converge to
the same zero-field triplet state:

1. a zero-field calculation with an explicitly fixed integer occupation; and
2. a weak-field seed followed by a zero-field restart with no fixed orbital
   occupation.

Only a reference that passes this gate may be used for Delta-Sternheimer
response generation or SIAB optimization.  The final project goal remains an
optimized C numerical atomic-orbital basis whose ordinary all-band SOS-RPA
result approaches the matched Delta-ST result.

## Physical state

The SG15 C pseudopotential has four valence electrons.  The target atomic
state is the triplet valence configuration

\[
  2s^2 2p^2, \qquad N_\uparrow=3, \quad N_\downarrow=1,
\]

with `nspin = 2` and `nupdown = 2`.  The three spatial orientations of the
open-shell \(2p^2\) determinant are degenerate in the continuum atom.  A raw
matrix comparison can therefore be misleading when two calculations select
different orientations.  The gate first compares scalar zero-field energies
and occupations; the later response gate compares rotational invariants such
as the spectrum of the symmetrized response and the RPA trace-log integrand.

## Alternatives considered

### Selected: fixed occupation versus weak-field seeded free occupation

This is the least expensive independent check supported by the present
ABACUS implementation.  The fixed route supplies a deterministic reference.
The field route selects an open-shell orientation, after which both the field
and the fixed orbital occupation are removed.

### Deferred: average over three triplet determinants

An equal average over the three \(2p\)-hole orientations would restore the
rotational symmetry of the atom.  It costs roughly three response calculations
and needs an explicit response-averaging contract, so it is retained only as
a fallback if the zero-field free-occupation state is not uniquely stable.

### Rejected for the first gate: spherical fractional occupation

Fractional \(2p\) occupations are natural for a spherical atomic ensemble,
but the present Delta-ST/SOS occupation and spin factors have only been
validated for integer occupations.  Fractional occupation must not be used
silently in the current C reference.

## Frozen numerical protocol

The PBE equivalence gate uses the same representation for every route:

- pseudopotential: SG15 `C_ONCV_PBE-1.0.upf`;
- orbital: SG15 TZDP-10au `C_gga_10au_100Ry_3s3p2d.orb`;
- cubic cell: 20 Angstrom, with C at the center;
- real-space grid: exactly `135 x 135 x 135`;
- wavefunction cutoff: 30 Ry;
- Gamma point only, `symmetry = 0`;
- PBE, `nspin = 2`, `nelec = 4`, `nupdown = 2`;
- `smearing_method = fixed`;
- `out_wfc_lcao = 1` and `out_app_flag = 1`, so the LCAO restart
  reader receives the text files `wfs1_nao.txt` and `wfs2_nao.txt`;
- `out_chg = 1`, so `init_chg = file` can read `chgs1.cube` and
  `chgs2.cube`;
- identical ABACUS executable and input assets in all branches.

The text wavefunction format is mandatory here: the current ABACUS LCAO
`init_wfc = file` path reads `.txt` NAO coefficients, not the binary
`out_wfc_lcao = 2` output.  The runner must also verify the ABACUS log
messages proving that both wavefunction and charge restart files were read.

The gate is a PBE ground-state test.  No Delta-ST, auxiliary basis, Coulomb
matrix, or LibRPA calculation is allowed before it passes.

### Task 4 execution and restart evidence

The production runner is a four-task Slurm array on `normal`.  Every task uses
one exclusive node, one MPI rank, all 30 schedulable OpenMP threads, the
account maximum of 110610 MB, and a 24-hour
limit.  The four array tasks map to `fixed`, `dir0`, `dir1`, and `dir2`.
The submitted ABACUS artifact is linked against the Intel MPI/MKL environment
provided by the canonical non-symlink file
`/public/home/ghj/app/src/env_60_245_intel2021.sh`.  The submitter requires this
path as `ABACUS_ENV_SCRIPT`, records its size and SHA256, and exports it to the
runner.  The runner sources it before resolving `mpirun`, then resets
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 30.  The
login-node OpenMPI command is not an admissible fallback.  The selected
`mpirun` is resolved to a canonical executable and its path, size, and SHA256
are recorded.

Numeric Slurm job and array identifiers are mandatory.  Before any branch is
created, the runner reads the live `scontrol show job -o` record and verifies
the partition, node/rank/thread counts, requested memory, 24-hour limit, and
exclusive allocation; manifests preserve both normalized values and the raw
scheduler record together with its SHA256, and audits reparse the raw record
and verify its hash and normalized content.  On df_dcu's Slurm 22.05 build,
the job-level display for `--exclusive` is `OverSubscribe=NO`; the auditor
requires that exact token, the committed `#SBATCH --exclusive` directive, and
all 30 schedulable CPUs.  Concurrent branch preparation is
serialized by a stable, array-owned guard plus a bounded preparation mutex, so
removing one mutex cannot invalidate another task's guard observation.  A
pre-existing branch is never reused or overwritten.

Every phase records `abacus.stdout`, `abacus.stderr`, the unique converged
energy, the complete 22-band occupations for both spins, and the four
nonempty restart outputs

```text
OUT.C_PBE_REFERENCE_GATE/wfs1_nao.txt
OUT.C_PBE_REFERENCE_GATE/wfs2_nao.txt
OUT.C_PBE_REFERENCE_GATE/chgs1.cube
OUT.C_PBE_REFERENCE_GATE/chgs2.cube
```

Restart staging does not copy the preceding `OUT.*` directory.  It creates a
new hidden phase, copies only `STRU`, `KPT`, the pseudopotential, the orbital,
and the four restart files, renders a new restart `INPUT`, and atomically
publishes the phase without replacement.  The four source files are also
copied to `restart_input_snapshot/`.  `RESTART_PROVENANCE.json` first records
`PLANNED` source, destination, and snapshot hashes.  Before ABACUS starts, the
destination copies must still match the source.  After ABACUS finishes, the
snapshot must still match the preceding phase output, while the new output is
allowed to differ from its input.

A restart is upgraded to `VERIFIED` only if `restart_input_snapshot/` is a
non-symlink phase-local directory, `abacus.stdout` contains exactly the two
messages reading the canonical phase-local paths of `wfs1_nao.txt` and
`wfs2_nao.txt`, and `running_scf.log` contains exactly the two messages reading
the canonical phase-local paths of `chgs1.cube` and `chgs2.cube`.  Real relative
paths are resolved against the phase directory; equivalent absolute
phase-local paths are also accepted.  External, traversal, or symlink escapes
are rejected.  Only then is `PHASE_COMPLETE.json` published.  A branch obtains
`BRANCH_COMPLETE.json` only after its complete fixed chain
`fixed_cold -> fixed_restart` or free chain
`field_seed -> free_restart1 -> free_restart2` has been rehashed.  Branches do
not publish a scientific result.

The global audit independently reopens every control, asset, executable,
environment script, `mpirun`, output, snapshot, phase manifest, and branch
manifest.  Runtime records are copied into every phase and branch manifest,
rehashed from the recorded canonical files, and required to be identical in
all four branches.  It also independently
checks identical pseudopotential/orbital content and the frozen preparation
identity across all branches.  Eleven valid phases, four valid branch
completions, one identical ABACUS hash and observed resource contract, verified
restart-load logs, and a passed zero-field energy test are all required for
`PBE_GATE_PASSED`.  Any `RUN_FAILED.json` counts as Task 4 evidence and blocks
the gate, even if no other runtime manifest exists.  Numerically valid Task 2
fixtures without runner evidence remain `DIAGNOSTIC_ONLY`; partial or
inconsistent Task 4 evidence is rejected.

### Task 5 submission evidence

Formal execution uses one duplicate-safe submission wrapper.  It resolves the
gate root, ABACUS executable, ABACUS environment script, pseudopotential,
orbital, Python interpreter,
runner, and submitter to canonical absolute paths before submission.  The
four external runtime/physical inputs must be nonempty non-symlink regular
files, and ABACUS must be executable.  The runner receives the resolved
environment script, pseudopotential, and orbital as `ABACUS_ENV_SCRIPT`,
`PSEUDO_ASSET`, and `ORBITAL_ASSET`.

Task 6 stages the gate directory as a standalone source archive without
`.git`.  It must pass the exact staging commit through required
`SOURCE_COMMIT`, which is validated as 40 lowercase hexadecimal characters.
`PYTHON_EXE` is also required explicitly.  Submission provenance records the
size and SHA256 of `gate_contract.py`, `prepare_gate.py`, `audit_gate.py`,
`run_pbe_branch.slurm`, and `submit_pbe_gate.sh`; each runtime source must be a
nonempty non-symlink regular file.

The Slurm job name is a stable hash of the canonical gate root.  Before and
after acquiring the claim, the submitter queries both `squeue` and `sacct`;
failure or malformed output from either command makes scheduler state
unobservable and stops submission.  Any prior scheduler record, immutable job
ID, claim, formal branch, result, pass marker, or failure marker blocks reuse
of that root.  The submitter creates `.submission-claim/` atomically before
calling `sbatch`, and only its owner can submit the committed four-task array.
After durably recording a random claim identity and completing the final
scheduler query, it immediately repeats the formal-evidence check.  That check
accepts only its own claim and rejects branch, result, failure, job-ID, or
submission-provenance evidence created during the race window.  It never adds
a Delta-ST dependency.

The shell creates both receipt files with exclusive-create semantics, fsyncs
both files and the claim directory, and only then invokes `sbatch` by appending
through those existing inodes.  It fsyncs their contents and the claim
directory again after return, so a returned job ID survives interruption
before final publication.  A nonzero `sbatch` exit or malformed success
receipt produces `SUBMISSION_AMBIGUOUS.json`; the claim is retained and the
same root must never be retried.  A unique numeric receipt is published without
replacement as `SUBMITTED_JOB_ID.txt`, together with atomic
`SUBMISSION_PROVENANCE.json` containing the job ID, UTC time, exact command,
explicit source commit, resolved paths, and file sizes and SHA256 hashes.
These records authorize observation and audit only; they do not constitute
physical acceptance.

## Calculation branches

### A. Fixed zero-field reference

Run a cold PBE SCF with

```text
ocp       1
ocp_set   3*1 19*0 1*1 21*0
efield_flag 0
efield_amp  0
```

Then restart once with the same zero-field fixed-occupation input.  This
restart checks numerical stability but is not the independent physical route.

### B. Weak-field seeds

Run three independent cold PBE calculations with `ocp = 0`, one for each
`efield_dir = 0, 1, 2`.  Use

```text
efield_flag     1
dip_cor_flag    0
efield_pos_max  0.8
efield_pos_dec  0.1
efield_amp      1e-4
```

ABACUS implements this as a saw-like periodic potential.  The decreasing
segment from fractional coordinate 0.8 to 0.9 lies in the vacuum, away from
the atom at 0.5.  Dipole correction is disabled because its documented domain
is a slab calculation, not an isolated atom in a cubic box.

### C. Free zero-field restarts

For each field direction, copy the converged wavefunction and charge density
to a new directory, then run with

```text
init_wfc file
init_chg file
ocp       0
efield_flag 0
efield_amp  0
```

Keep `nupdown = 2`; the known physical spin multiplicity remains constrained,
whereas the identity of the occupied orbitals does not.  Repeat the same free
zero-field restart once more from its own output.  Only this second zero-field
result enters the equivalence comparison.

## PBE acceptance criteria

All conditions below are mandatory:

1. Every fixed and free zero-field SCF reports convergence and one finite
   final total energy.
2. Every accepted zero-field `eig_occ.txt` contains only occupations zero or
   one within `1e-10`.
3. The spin electron counts are exactly \(3\) and \(1\), and the total magnetic
   moment is consistent with the triplet constraint.
4. The energy drift between the last two zero-field stages is less than
   `0.001 kcal/mol`: cold-to-restart for the fixed route and first-to-second
   zero-field restart for each free route.
5. Each field-seeded free zero-field energy differs from the fixed zero-field
   energy by less than `1e-5 Ha`.
6. The three field-direction zero-field energies differ from one another by
   less than `1e-5 Ha`.
7. The accepted inputs have `efield_flag = 0`, `efield_amp = 0`, and `ocp = 0`
   for the free branch.  A finite-field energy is never compared with the
   fixed zero-field energy.

Scheduler completion, a nonempty output directory, or a converged finite-field
seed does not constitute passage of the gate.

## Failure handling

- If the field seed does not converge, diagnose the SCF before changing the
  field.  Only then may `3e-5` and `3e-4` a.u. be tried as bounded seed tests.
- If the field seed converges but the zero-field restart becomes fractional or
  changes spin, the free route fails; do not proceed to Delta-ST.
- If the final states remain integer but select symmetry-related determinants,
  compare invariant quantities rather than elementwise wavefunctions.
- If scalar zero-field energies disagree, first check executable and asset
  hashes, grid identity, restart loading, and residual convergence.
- If no free zero-field determinant is stable because of exact \(2p\)
  degeneracy, stop and design the explicit three-determinant ensemble.  Do not
  weaken the acceptance thresholds to force a pass.

## Delta-ST response gate after PBE passage

After the PBE gate passes, run one fixed-occupation reference and one accepted
free-occupation reference with the same zero-field numerical protocol and:

- FD8 atomic Sternheimer operator;
- `sternheimer_nfreq = 6` for the equivalence gate only;
- `exx_pca_threshold = 1e-4`;
- full Coulomb and Gamma-only atomic treatment;
- `exx_ccp_rmesh_times = rpa_ccp_rmesh_times = 1`;
- the same GreenX frequency grid in both routes;
- LibRPA trace-log postprocessing.

The response gate requires:

- all Sternheimer equations converged;
- equal auxiliary dimensions and identical Coulomb/frequency provenance;
- agreement of sorted symmetrized-response eigenvalue spectra and per-frequency
  trace-log integrands to a relative scale of `1e-3`, excluding values whose
  absolute magnitude is below the declared numerical threshold;
- an RPA correlation-energy difference below `0.1 kcal/mol`.

Raw response matrix elements are diagnostic only because symmetry-related
open-shell determinants can be represented in different Cartesian gauges.
The formal C reference uses the converged production frequency count only
after this six-frequency equivalence gate passes.

## Transition to C basis optimization

PBE and Delta-ST passage authorizes, but does not itself perform, the C basis
optimization.  The next specification will generalize the SIAB projected-Pi
optimizer from its hard-coded H/H2 families to arbitrary named physical
families.  The initial C basis policy is:

- preserve the complete DFT-quality core block of the starting C basis;
- train added radial functions against accepted C atomic and molecular
  Sternheimer responses;
- validate ordinary all-band SOS-RPA against matched Delta-ST data;
- retain diamond C as an independent transfer test before claiming a
  solid-ready C basis.

## Required artifacts

Each accepted branch records the ABACUS commit and executable hash, input and
asset hashes, full inputs, final occupations, final energies, grid fingerprint,
restart provenance, wall time, scheduler state, and the audit result.  The
comparison summary must distinguish `PBE_GATE_PASSED`,
`DELTA_ST_GATE_PASSED`, and `DIAGNOSTIC_ONLY`.
