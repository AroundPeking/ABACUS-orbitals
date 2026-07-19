# H Sternheimer SIAB experiment

This directory defines the first H-TZDP Sternheimer-supervised SIAB experiment. It contains configuration only: no ABACUS producer output or fabricated production matrices are committed here.

## Provenance status

- The H atom Sternheimer matrix is the only new supervision.
- `st_only` reads no DFT/dpsi matrices; it depends only on the H-atom Sternheimer matrix and the checked-in TZDP coefficients.
- `st_constrained` additionally reuses the historical H dimer/trimer DFT and dpsi matrices at 0.7, 0.9, and 1.3 Angstrom. The original `SIAB.py` writes these values under `Cartesian_angstrom`.
- `st_dpsi_joint` uses the same training data, but keeps the normalized dpsi
  loss active throughout optimization instead of using dpsi only after a hard
  threshold is crossed.
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

For the continuous dpsi lane:

```bash
cp INPUT.st_dpsi_joint INPUT
python3 ../opt_orb_pytorch_dpsi/main.py
```

Its objective is

```text
L = L_ST + lambda_dpsi * L_dpsi/L_dpsi_initial
    + lambda_DFT * max(0, L_DFT/L_DFT_initial - 1 - tau_DFT)^2
    + lambda_gate * max(0, L_dpsi/L_dpsi_initial - 1 - tau_dpsi)^2.
```

The checked-in first trial uses `lambda_dpsi=0.1`, `tau_DFT=0.05`, and
`tau_dpsi=0.10`. Accepted candidates must satisfy both hard tolerances and the
Sternheimer condition-number limit; among accepted candidates this mode keeps
the minimum total loss. The older `st_constrained` mode still keeps the minimum
Sternheimer loss. Do not tune `lambda_dpsi` against the held-out H2 RPA result.

The historical `orb_matrix.0.dat` and `orb_matrix.1.dat` producer files are not
stored in this checkout. A physical joint campaign therefore requires those
exact files to be restored or regenerated before this input can run; synthetic
tests cover the objective, gradients, routing, and selection contract only.

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

## Fixed-DZP response-space diagnostics

The superseding local campaigns fix the complete `1s,2s,1p` DZP core. Both use
the same canonical target, seed `20260718`, 3000 Adam steps, and 25 primitive
coefficients per radial orbital.

| basis shape | trainable orbitals | initial ST loss | best ST loss | ratio | condition | wall time | final response-orbital nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `3s2p` | `3s,2p` | 0.4428607140 | 0.3943529985 | 0.8904673322 | 5.1688 | 66.69 s | `3s:2, 2p:10` |
| `4s3p` | `3s,4s,2p,3p` | 0.3777740028 | 0.3174804814 | 0.8403979073 | 11.5152 | 85.42 s | `3s:14, 4s:12, 2p:10, 3p:12` |

All three fixed coefficient columns are bitwise identical to the input. Their
exported 801-point radial functions have maximum absolute error no larger than
`1.03e-14`. The different basis dimensions already have different initial
losses, so their absolute losses are not a direct ranking of transferable basis
quality.

The `3s2p` result is under the parent project directory
`../results/siab_h_dzp_core_20260719_st_only_5b47b248_conda310`; the expanded
result is under
`../results/siab_h_dzp_core_4s3p_20260719_5b47b248_conda310`. Their final
coefficient SHA256 values are `88ffd5d3ad75b785efe0a0ef56c873e4b1c82ac022d489e63c9355d9a33c449e`
and `7a0dc1c09c5e603fe6457780085e214de4f0c7637a2321d90b2fc47fef4fda65`,
respectively. Both reduce the ST
training loss, but `2p` develops ten nodes and every trainable `4s3p` response
orbital is oscillatory. Expanding an ST-only space therefore does not address
the shape problem. The next physical campaign must use `INPUT.st_dpsi_joint`
after its referenced legacy DFT/dpsi matrices are restored or regenerated.
