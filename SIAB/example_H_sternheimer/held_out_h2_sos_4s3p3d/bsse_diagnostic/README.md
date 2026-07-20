# Fixed-ABS H2 counterpoise diagnostic

This post-held-out diagnostic quantifies basis-set superposition error; it is
not an additional optimization target. It runs one spin-polarized H atom in
the full H2 basis by replacing the second H with ABACUS species `H_empty` at
the unchanged 0.74085-Angstrom bond length. ABACUS keeps the ghost AO and ABS
functions but removes its pseudopotential charge and electrons.

The old `4s3p` and smooth `4s3p3d` cases use the same explicit 214-function
per-atom ABS and all `26/56` dimer-basis bands. Their counterpoise-corrected
binding energies are evaluated as

```text
D_CP = 2 E_RPA@PBE(H + H_empty) - E_RPA@PBE(H2).
```

The isolated-atom and H2 energies come from production array `21321689`.
Do not select or re-optimize an orbital from this diagnostic; its sole purpose
is to determine how much of the `4s3p3d` overbinding is SOS virtual-space BSSE.
Stage a `SOURCE_COMMIT` file beside `run_bsse.slurm` with the full commit that
contains this diagnostic.
