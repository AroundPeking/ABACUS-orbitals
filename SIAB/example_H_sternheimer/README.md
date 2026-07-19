# H Sternheimer SIAB experiment

This directory defines the first H-TZDP Sternheimer-supervised SIAB experiment. It contains configuration only: no ABACUS producer output or fabricated production matrices are committed here.

## Provenance status

- The H atom Sternheimer matrix is the only new supervision.
- `st_only` reads no DFT/dpsi matrices; it depends only on the H-atom Sternheimer matrix and the checked-in TZDP coefficients.
- `st_constrained` additionally reuses the historical H dimer/trimer DFT and dpsi matrices at 0.7, 0.9, and 1.3 Angstrom. The original `SIAB.py` writes these values under `Cartesian_angstrom`.
- H2 RPA is held out from optimization and is reserved for the final transfer test.
- The initial basis is the existing H TZDP basis at 8 Bohr. Its complete DZP
  core (`1s`, `2s`, and `1p`) is fixed exactly; only the TZDP-only `3s` and
  `2p` response orbitals are optimized.
- The spherical-Bessel representation is the original SIAB one: 100 Ry, 8 Bohr, with the 0.1-Bohr cutoff smoothing used to generate the checked-in H-TZDP orbital.
- The first experiment uses every producer reference row. It applies no PCA and no reference-row truncation.

The experiment is **not physics validated** until the exact ABACUS executable/source commit, pseudopotential, orbital and auxiliary-basis hashes are recorded in the producer metadata, and the planned H/H2 comparison table is populated. A completed optimizer run alone does not satisfy that validation gate.

## Run

Use the campaign runner for the formal `st_only` lane. Keep the generated
target and campaign output outside the Git working tree:

```bash
python3 run_st_only.py \
  --target /absolute/path/to/sternheimer_matrix.dat \
  --output /absolute/path/to/new-empty-campaign-directory
```

The runner materializes an `INPUT` with absolute paths, invokes
`opt_orb_pytorch_dpsi/main.py`, and writes `campaign_summary.json`. It fails if
the output directory is nonempty, the optimizer does not report pure
`st_only`, the final Sternheimer loss is worse than the initial loss, or any
fixed H-DZP coefficient changes at the float64 byte level. It also compares the
exported 801-point `1s`, `2s`, and `1p` radial functions with the checked-in
H-TZDP `.orb` after smoothing and normalization. The summary records target,
input, initial/final coefficients, reference/final orbitals, and spillage hashes
together with the loss ratio, radial error, and wall time.

For the constrained lane:

```bash
cp INPUT.st_constrained INPUT
python3 ../opt_orb_pytorch_dpsi/main.py
```

Both inputs use seed `20260718`. The optimizer must report that value for both NumPy and PyTorch. Compare the resulting named losses in `Spillage.dat` and `ORBITAL_RESULTS.txt`; do not use the held-out H2 RPA result for parameter fitting.

## Appending response shells

The checked-in TZDP coefficient file can initialize a larger response basis.
For example, changing `element.Nu.H` from `[3, 2]` to `[4, 3]` preserves the
loaded `3s2p` columns and deterministically initializes appended `4s` and `3p`
columns from the top-level seed. The optimizer prints both sets explicitly:

```text
loaded coefficient columns: [...]
appended response columns: ['H/l0/zeta4', 'H/l1/zeta3']
```

The DZP freeze list remains unchanged, so `3s`, `2p`, and all appended columns
are trainable. A new angular channel such as `Nu.H = [3, 2, 1]` is accepted
only when the Sternheimer target contains complete H `l=2`,
`m=-2,-1,0,1,2` primitive blocks. The current canonical producer target has
only `s/p` blocks and therefore cannot optimize a `d` response orbital; it must
be regenerated with `lmax >= 2` first.

The exact converged H-atom ABACUS producer input and `normal`-partition job
script are under `producer/`.  Stage the checked-in H-TZDP `.orb` and the
Dojo-NC-SR `Pseudopotential/H.upf` beside those files before submitting.  The job
uses one 30-thread MPI rank for each of the 16 minimax frequencies and refuses
to run if the immutable ABACUS executable hash changes.

## First formal `st_only` campaign

The first formal producer was df_dcu `normal` job `21311439`. It completed in
`03:45:04` on 16 nodes and converged all 656 response equations. The canonical
target has 656 reference rows, 100 primitives in four H `s/p` blocks, and SHA256
`bed58ebf61cb513da892658b848f881f724feba7f50fe64f7a0b6252bb8e0c8c`.
Its provenance records ABACUS commit `80a606f57a26`, 50 Ry, 16 frequencies,
`exx_pca_threshold=1e-6`, the Dojo H pseudopotential, and the checked-in H-TZDP
orbital hashes.

The deterministic optimization used this target and code commit `a41a9f0e`.
After 3000 Adam steps, the best Sternheimer loss was
`0.12535769112573478`, down from `0.15884642225499218` (ratio
`0.7891754145`). The best projected-overlap condition number was `73.61`.
That first campaign fixed only the 25-coefficient H level-1 `1s` column. It is
retained as an implementation diagnostic, but it is superseded by the DZP-core
campaign because the optimized upper orbitals were visibly oscillatory. The
old final coefficient SHA256 is
`278694016e5f819f2a79db4b3ddc8c5692d8dd125a908f0003295ab644eb4715`.

These numbers validate the implementation and training loop only. The optimized
upper orbitals are visibly more oscillatory. Do not promote this basis or call
it RPA-accurate until the held-out H2 LCAO-SOS/LibRPA calculation is compared
with the initial TZDP, Sternheimer, and FHI-aims references.
