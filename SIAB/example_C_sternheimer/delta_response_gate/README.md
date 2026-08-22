# C atomic Delta-ST response-equivalence gate

This gate asks one physical question before C SIAB optimization: do the two
independently accepted zero-field C-triplet PBE states produce the same atomic
Delta-Sternheimer response?

The accepted PBE sources are fixed by `response_contract.py`:

- fixed occupation: `runs/fixed/fixed_zero_restart`;
- field-seeded free occupation: `runs/dir0/free_restart2`.

Both states have three spin-up and one spin-down electrons.  The PBE gate must
already report `PBE_GATE_PASSED` and `ZERO_FIELD_COMPARISON_PASSED`.

## Frozen response definition

- 20 Angstrom cubic cell, centered C atom, Gamma point;
- PBE, `nspin=2`, `nupdown=2`, `nbands=22`;
- 30 Ry and the explicit `135 x 135 x 135` FD grid;
- 10-au `3s3p2d` C orbital and the same ONCV PBE pseudopotential;
- FD8 Delta-ST, all available KS virtual candidates, solver tolerance `1e-6`;
- six GreenX minimax frequencies generated once from the union of the two
  accepted same-spin transition windows;
- `exx_pca_threshold=1e-4`, Massidda Gamma treatment and both real-space
  Coulomb ranges equal to one;
- reader-v1 Sternheimer matrices and the full Coulomb matrix;
- LibRPA `sternheimer_rpa` with `sqrt_coulomb_threshold=1e-5`.

The feature ABACUS executable used on server 66 predates the explicit
`sternheimer_delta_virtual_source` input.  Its accepted LCAO virtual candidates
implement the intended all-`ks_bands` source.  This is a recorded feature-build
exception, not a change to the physical gate.

## Compared quantities

ABACUS writes the auxiliary-potential response matrix `M`.  The physical
comparison uses the same full Coulomb metric in both branches:

```text
Pi(iw) = V_full^(-1/2) M(iw) V_full^(-1/2).
```

`audit_response_gate.py` compares the sorted eigenvalue spectra of `Pi`, the
six LibRPA trace-log integrands and the final RPA correlation energy.  It also
reports raw-`M` differences as diagnostics.  The gate requires:

- full-Coulomb matrix relative difference at most `1e-10`;
- maximum `Pi`-spectrum relative difference at most `1e-3`;
- maximum trace-log-integrand relative difference at most `1e-3`;
- absolute RPA correlation-energy difference below `0.1 kcal/mol`.

Only `DELTA_RESPONSE_GATE_PASSED` authorizes use of the C atomic response as a
SIAB training reference.

## Workflow

1. Generate `fixed_frequency_grid.dat` with `generate_frequency_grid.py` from
   the two accepted `eig_occ.txt` files and the pinned GreenX executable.
2. Create a new immutable campaign root with `prepare_response_gate.py`.
3. Run `submit_response_gate_server66.sh`.  The ABACUS array uses six nodes per
   branch, one MPI rank per frequency and 48 OpenMP cores per rank.  The LibRPA
   array starts only after both response branches pass.
4. Run `audit_response_gate.py --root CAMPAIGN_ROOT` after both LibRPA branches
   write `LIBRPA_COMPLETE.json`.

The submitter refuses a campaign with receipt files, an existing claim, or a
matching current or historical Slurm job name.  A failed campaign is preserved
and a recovery uses a new campaign root.

## Current status

The PBE zero-field gate has passed.  Frequency generation, immutable staging,
server resource contracts, reader-v1 matrix reconstruction and the physical
audit have focused unit coverage.  The production Delta-ST comparison is not
accepted until the server-66 ABACUS and LibRPA jobs finish and the audit passes.
