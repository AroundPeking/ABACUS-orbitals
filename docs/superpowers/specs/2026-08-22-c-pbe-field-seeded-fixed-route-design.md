# C PBE Field-Seeded Fixed Route Design

## Goal

Establish a reproducible neutral C-atom PBE reference before C Delta-ST or
SIAB basis optimization.  The gate must show that fixed integer occupation and
free occupation relax to the same zero-field triplet state in the same SG15
TZDP-10au representation.

## Evidence for the change

The original fixed route began from a zero-field atomic density with fixed
`3 up / 1 down` occupation.  With identical INPUT, STRU, KPT, executable,
resources, and even the same server66 node, repeated calculations followed
different trajectories from the first SCF iteration.  Some converged, while
others reached 300 iterations with `drho` between about `9e-4` and `2e-3`.
Reducing both mixing factors from 0.3 to 0.1 and disabling charge and magnetic
Kerker preprocessing did not make this cold start reproducible.  This behavior
is consistent with numerical selection among the degenerate C 2p orbitals.

A diagnostic fixed-occupation field-seeded route was repeated twice with a
field amplitude of `1e-4`.  Both field seeds converged in 43--45 iterations;
after removing the field, both fixed-occupation restarts converged in 28--29
iterations.  The two final energies were

```text
-147.4773363622931 eV
-147.4773363622959 eV
```

Both had integer spin populations `3 up / 1 down`.  Their energies, the two
completed free-occupation directions, and the earlier converged fixed result
agreed within `1.5e-10 kcal/mol`.

## Selected physical protocol

The fixed branch contains two phases:

1. `fixed_field_seed`: fixed integer occupation, `nspin=2`, `nupdown=2`,
   `ocp=1`, `ocp_set=3*1 19*0 1*1 21*0`, and a Cartesian-direction-0 field
   with amplitude `1e-4`;
2. `fixed_zero_restart`: restart from the seed wave functions and charge,
   remove the field exactly, and retain the same fixed integer occupation.

The three free branches remain unchanged:

1. `field_seed` in Cartesian directions 0, 1, and 2 with free occupation;
2. `free_restart1` with the field removed and free occupation;
3. `free_restart2` as a second zero-field free restart.

The field is only a deterministic orientation selector.  Its seed energy is
not a reference energy and is never compared with the free branches.  The
physical comparison uses only `fixed_zero_restart` and the three
`free_restart2` results, all at exactly zero field.

## Acceptance contract

The existing numerical thresholds remain unchanged:

- every phase must contain exactly one ABACUS SCF convergence marker and one
  finite final energy;
- every final state must have four electrons and integer spin populations
  `3 up / 1 down`;
- every restart must prove that both spin wave functions and both spin charge
  densities were loaded from canonical phase-local paths;
- the fixed seed-to-zero restart drift, each free restart drift, the spread of
  the three free directions, and every fixed-versus-free zero-field energy
  difference must satisfy the existing kcal/mol thresholds;
- scheduler, executable, environment, pseudopotential, orbital, source, and
  restart provenance checks remain mandatory.

Only a complete four-branch result may create `PBE_GATE_PASSED`.  Scheduler
completion, an isolated converged phase, or diagnostic agreement cannot pass
the physical gate.

## Implementation boundaries

Add an explicit `fixed_field` input mode rather than weakening the meaning of
the existing zero-field `fixed` mode.  Rename the fixed branch phases to
`fixed_field_seed` and `fixed_zero_restart`; update preparation, runtime,
auditing, comparison, fake-HPC fixtures, and documentation consistently.

The free branch inputs, 20 Angstrom box, `135^3` real-space grid, 30 Ry cutoff,
SG15 pseudopotential, TZDP-10au orbital, PBE functional, Gamma sampling,
resource profiles, and all acceptance thresholds remain unchanged.

## Verification and formal execution

Tests must prove that `fixed_field` combines fixed occupation with the exact
field contract, that its zero-field restart removes every field setting, and
that stale old phase names or extra input keys are rejected.  The complete
unit suite, shell syntax checks, Python compilation, and patch checks must pass.

After review, create a new source commit and a new commit-derived immutable
server66 root.  Run the full preflight, verify that no matching job exists, and
submit exactly one four-task array.  Preserve failed jobs `410615` and `410626`
as historical provenance; neither root may be reused.

## Non-goals

This change does not run Delta-ST, optimize a C basis, change the physical PBE
representation, relax convergence thresholds, accept fractional occupations,
or use automatic retry as a substitute for a reproducible initial state.
