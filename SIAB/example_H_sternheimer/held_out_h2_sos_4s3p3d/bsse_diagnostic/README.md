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

## Completed result

df_dcu `normal` array job `21321833` completed both tasks with exit code zero.
The old/new ghost calculations took 4:53 and 7:06. Both used 428 auxiliary
functions, all 26/56 dimer-basis bands, integer occupations, and full Coulomb.
The immutable production directory is
`/work1/ghj/sternheimer_abacus_tests/h2_joint_4s3p3d_sos_6875ce25_20260720/bsse_diagnostic`.

| basis | uncorrected binding | CP binding | total BSSE | RPAc BSSE |
| --- | ---: | ---: | ---: | ---: |
| fixed-ABS `4s3p` | 107.1831 | 105.9216 | 1.2615 | 1.2516 |
| fixed-ABS `4s3p3d` | 113.7859 | 107.0212 | 6.7647 | 6.7391 |

All values are in kcal/mol and `BSSE = D_uncorrected - D_CP`. The remaining
BSSE components `(PBE-PBE-XC, EXX, RPAc)` are `(0.0927, -0.0829, 1.2516)` for
`4s3p` and `(0.1279, -0.1023, 6.7391)` for `4s3p3d`.

After counterpoise, adding d improves the binding by `1.0996 kcal/mol`, but the
result is still about `1.70 kcal/mol` below the basis-error-free reference.
The atomic d-response optimization is therefore physically useful but not
transfer-balanced: in H2 each center can borrow the neighboring d space, while
the isolated atom cannot. A subsequent basis must be trained against a
predeclared multi-center response/borrowing gate rather than retuned to this H2
number.
