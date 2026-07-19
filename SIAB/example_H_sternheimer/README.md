# H Sternheimer SIAB experiment

This directory defines the first H-TZDP Sternheimer-supervised SIAB experiment. It contains configuration only: no ABACUS producer output or fabricated production matrices are committed here.

## Provenance status

- The H atom Sternheimer matrix is the only new supervision.
- `st_only` reads no DFT/dpsi matrices; it depends only on the H-atom
  Sternheimer matrix and the checked-in TZDP coefficients. It is retained only
  as an implementation regression/ablation mode. Its optimized orbitals are
  retired and must not be used as DFT or RPA production inputs.
- `st_constrained` additionally reuses the historical equilateral H3-trimer DFT and dpsi matrices at side lengths 0.7, 0.9, and 1.3 Angstrom. The original `SIAB.py` writes these values under `Cartesian_angstrom`.
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

The following runner is only for reproducing the retired `st_only`
implementation diagnostic. Keep the generated target and campaign output
outside the Git working tree:

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

The historical `orb_matrix.0.dat` and `orb_matrix.1.dat` files were regenerated
by df_dcu `normal` array job `21315279` using the producer in
`legacy_dpsi_producer/`. The three tasks completed in 31, 33, and 39 seconds on
30 MPI ranks each. The zero-order/dpsi SHA256 pairs for H3 side lengths
0.7, 0.9, and 1.3 Angstrom are, respectively:

```text
0.7  7d02cc86c20bfb34bd1efa3ca8f1a09f70aec0fbf7079d954d15e1bacc875505
     80712780be91a38e562d931028ff29053ceaa7ffbb6860ad0d0c5b96e059e015
0.9  834741f0a3fe69fb9c53ac0c2264e2f1c4d1827e26bf87df66bd14e0271c0648
     391e876b9fec0c3834a5a6cf5e479d58a53b4ca6e539b8f951bf8f76f9a875e7
1.3  23326839e68011d8877677f5021da4be032641aa263abe0ab02d40a561f9edad
     d00fbea91f2dddb346aee7fd5c406743a4f74c51bd49fb5b0009f2776f2622ab
```

All six files contain the Q, Sq, and V closing tags; the producer also verified
the final SCF marker. These matrices are the fixed DFT/dpsi input for the first
physical `st_dpsi_joint` campaign. They are generated results and remain
outside Git.

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

`INPUT.st_dpsi_joint_4s3p` is the controlled expanded campaign. Relative to
the validated `3s2p` joint input, it changes only `Nu.H` from `[3, 2]` to
`[4, 3]`; training data, seed, DZP freeze list, optimizer, and every loss
weight remain identical. This keeps H2 held out and tests whether two appended
same-angular-momentum response functions add transferable Sternheimer space.

The exact converged H-atom ABACUS producer input and `normal`-partition job
script are under `producer/`.  Stage the checked-in H-TZDP `.orb` and the
Dojo-NC-SR `Pseudopotential/H.upf` beside those files before submitting.  The job
uses one 30-thread MPI rank for each of the 16 minimax frequencies and refuses
to run if the immutable ABACUS executable hash changes.

## Retired `st_only` implementation campaign

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
retained as an implementation diagnostic, but it is superseded by the
DZP-core joint campaign because the optimized upper orbitals were visibly
oscillatory. The optimized `.orb` is retired and must not be used in any
production DFT/RPA comparison. The old final coefficient SHA256 is
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

## First physical `st_dpsi_joint` campaign

df_dcu `normal` job `21315288` used source commit `d41f975e`, one node, 30
PyTorch CPU threads, and 110610 MB. It completed in 7 minutes 14 seconds; the
optimizer itself took 6 minutes 56 seconds and reached its best accepted point
at Adam step 350. The complete DZP core remained bitwise fixed.

| component | initial | best | best/initial |
| --- | ---: | ---: | ---: |
| DFT origin | 7.6516808055e-5 | 5.0667853878e-5 | 0.66217940 |
| DFT dpsi | 9.2278755443e-4 | 5.9172304411e-4 | 0.64123431 |
| Sternheimer | 0.4428607140 | 0.4054568603 | 0.91554037 |
| `0.1 * R_dpsi` | 0.1000000000 | 0.0641234314 | 0.64123431 |
| total | 0.5428607140 | 0.4695802917 | 0.86501064 |

Both hinge penalties are zero at the selected point. The DFT and dpsi losses
therefore improve rather than merely staying below their 5% and 10% limits.
The radial node counts for `3s,2p` are `2,1` initially, `3,10` after ST-only,
and `2,1` after joint optimization. Continuous dpsi supervision removes the
high-frequency ST-only solution while still reducing the atomic ST loss.

The generated results are stored outside Git under
`../results/siab_h_joint_d41f975e_21315288`. The final coefficient, orbital,
and trajectory SHA256 values are, respectively,
`1340cd11357dea87b67ad2a58a6a8e1ae298c985bf08a66b6e9456c57dbc87df`,
`30b7e5e3d80b59778b0fee836fcd0315c0cfd827621806eb3f2c9e659b8118a7`, and
`476cc96e68da6d1fcdf9160bd8bb9015800f37e69e38958e46420163d91b45a4`.
This is a training-space result. It is not an RPA-quality claim until the
held-out H2/H LCAO-SOS and LibRPA binding energy is computed with this orbital.

## Same-producer held-out H2/H SOS-RPA result

The first held-out comparison was completed on df_dcu `normal` with the same
immutable ABACUS and LibRPA executables, inputs, pseudopotential, 20-Angstrom
cell, 0.74085-Angstrom H2 bond, 100-Ry cutoff, 16 minimax frequencies,
`exx_pca_threshold=1e-4`, `rpa_ccp_rmesh_times=5`, and full Coulomb. Array job
`21315392` used the checked-in initial TZDP orbital (SHA256
`7e398340398306a6baf1c61ea68944d81ed43667473fbcc290d6541c4a661d1c`);
job `21315382` used the joint orbital above. Both H2/H pairs completed and
LibRPA printed its success marker.

The RPA@PBE energy is evaluated as

```text
E_RPA@PBE = E_PBE - E_xc^PBE + E_x^EXX + E_c^RPA,
D_e = 2 E_RPA@PBE(H) - E_RPA@PBE(H2).
```

| basis | H2 PBE (Ry) | H PBE (Ry) | H2 EXX (Ha) | H EXX (Ha) | H2 RPAc (Ha) | H RPAc (Ha) | binding (kcal/mol) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial TZDP | -2.3319033102 | -0.9990938552 | -0.6573650319 | -0.3076354167 | -0.067918272 | -0.016168866 | 106.5360756 |
| fixed-DZP joint | -2.3320626242 | -0.9990624516 | -0.6572902713 | -0.3076186661 | -0.069145013 | -0.016673657 | 106.7371494 |
| fixed-DZP joint `4s3p` | -2.3324200613 | -0.9991830132 | -0.6565751969 | -0.3069816006 | -0.071038897 | -0.017573421 | 106.8576364 |

The binding-energy decomposition in kcal/mol is:

| basis | PBE minus PBE-XC | EXX | RPAc | total |
| --- | ---: | ---: | ---: | ---: |
| initial TZDP | 57.7944412 | 26.4145084 | 22.3271259 | 106.5360756 |
| fixed-DZP joint | 57.8851363 | 26.3886178 | 22.4633953 | 106.7371494 |
| fixed-DZP joint `4s3p` | 57.5956008 | 26.7394311 | 22.5226046 | 106.8576364 |

The joint basis improves this same-producer held-out value by only
`0.2010738 kcal/mol`; it remains about `1.98 kcal/mol` below the approximately
`108.72 kcal/mol` Thesis/FHI-aims reference and is not yet a converged RPA
basis. At the same PCA threshold the generated auxiliary dimensions are
H2/H = `68/34` for the initial basis, `66/33` for the `3s2p` joint basis, and
`126/63` for the expanded `4s3p` basis. This test therefore evaluates each
orbital together with its deterministically regenerated product auxiliary
basis. A fixed-auxiliary-space cross test is required before attributing the
change exclusively to the wave-function basis.

The old 20-Angstrom result is not a valid initial-basis control. Its same-named
orbital has SHA256
`b2d24063d577a79039e60f5084ba1bd2f3355fc45b8b0d18b7c3ba61e08394f6`,
which differs from the checked-in TZDP hash above. Phase-aligned normalized
radial shape errors for `1s,2s,3s,1p,2p` are approximately
`0.903,1.030,0.805,0.241,0.262`, respectively. Historical binding energies
from that file must not be used to assess the joint optimization.

## Expanded `4s3p` joint campaign

df_dcu `normal` job `21315401` used source commit `cb5d3e9d` and the controlled
`INPUT.st_dpsi_joint_4s3p` expansion. It completed on one full node in 2 minutes
13 seconds; the optimizer used 1 minute 27 seconds and selected evaluation 568.
The three fixed DZP columns remain bitwise equal to the input and their maximum
exported radial error is `1.02e-14`.

| component | initial | best | best/initial |
| --- | ---: | ---: | ---: |
| DFT origin | 5.6564793742e-5 | 1.6397651320e-5 | 0.28989147 |
| DFT dpsi | 8.3932898923e-4 | 2.1487703902e-4 | 0.25601051 |
| Sternheimer | 0.3777740028 | 0.3287659495 | 0.87027150 |
| `0.1 * R_dpsi` | 0.1000000000 | 0.0256010506 | 0.25601051 |
| total | 0.4777740028 | 0.3543670001 | 0.74170423 |

Both hinge penalties are zero. The final `3s,4s,2p` radial functions have
`2,4,1` nodes. The appended `3p` has three significant nodes and two additional
sign changes only in its small tail; this is far below the 10--14-node ST-only
expanded solution, but it is not by itself evidence of transferability.

The final coefficient, orbital, and trajectory SHA256 values are:

```text
bd1599244787c265c7eee140d3ad6b8938d9295954ca98c2f5119c257cd87536
b394bb7329754e38341050ca4beb3b242b78e4be50c418b8764c98226bc8f033
a212007b50647c4622a4ff95b359be5ba57b6d7ab6b21d1a4dead73a881fa3fb
```

The next held-out calculation must use all `26` H2 and `13` H bands because
`4s3p` contains 13 AO functions per atom. Reusing the `3s2p` values `18/9`
would truncate the SOS virtual space and invalidate the basis comparison.

That held-out calculation completed as df_dcu `normal` array job `21315465`.
H2/H took 2 minutes 30 seconds and 1 minute 59 seconds and both LibRPA tasks
finished successfully. The resulting binding energy is `106.8576364 kcal/mol`,
an improvement of `0.1204871 kcal/mol` over the `3s2p` joint basis and
`0.3215609 kcal/mol` over the checked-in TZDP, but still approximately
`1.86 kcal/mol` below the Thesis/FHI-aims reference. The RPA correlation
contribution changes monotonically from `22.3271259` to `22.4633953` to
`22.5226046 kcal/mol` across initial, `3s2p` joint, and `4s3p` joint bases.

Because the H2/H auxiliary dimensions also increase to `126/63`, this result
does not isolate wave-function completeness from auxiliary-basis completeness.
Do not repeatedly choose further shell counts from this H2 value: that would
turn the held-out molecule into training data. Further expansion requires a
predeclared atomic/H3 training-space criterion and a fixed-auxiliary cross test
before one final H2 evaluation.

## Angular-momentum floor in the current target

The remaining error is not evidence that frequency-dependent first-order
wavefunction fitting is ineffective. The canonical target contains
`656 = 41 * 16` response rows. The 41 auxiliary perturbations consist of 8 s,
18 p, and 15 d channels, but its primitive columns contain only four blocks:
one s block and three p magnetic blocks, each with 25 radial primitives. There
is no d primitive block.

For the occupied H 1s state, an auxiliary perturbation with angular momentum
`L` produces a first-order response with the same angular momentum. Therefore
all 15 d-channel target rows have nonzero reference norm but exactly zero
overlap with every current s/p primitive. This is a wavefunction-basis angular
cutoff, not an auxiliary-basis error: Delta-ST solves these responses on the
uniform grid, while the current SIAB candidate discards them before
optimization.

After projecting out the fixed `1s,2s,1p` DZP core, the weighted residual norm
and loss decompose as follows. The last column is the best possible loss when
all 92 numerically independent directions of the current 100-column s/p
primitive space are available.

| response channel | residual-norm share | joint `3s2p` loss in channel | joint `4s3p` loss in channel | complete s/p primitive floor |
| --- | ---: | ---: | ---: | ---: |
| s | 0.088949 | 0.212640 | 0.059163 | 0.004525 |
| p | 0.609943 | 0.140070 | 0.036717 | 0.000444 |
| d | 0.301108 | 1.000000 | 1.000000 | 1.000000 (missing) |
| total | 1.000000 | 0.405457 | 0.328766 | 0.301781 |

Thus d response alone contributes `0.301108`, or 91.59% of the final `4s3p`
ST loss. The remaining s/p loss is only `0.027658`. Relative to the maximum
response fraction capturable by the current s/p primitives, the `4s3p` joint
basis already captures 96.14%. Adding more s or p zeta functions cannot lower
the total loss below approximately `0.301781`.

The next producer target must therefore expose complete `l=2`,
`m=-2,-1,0,1,2` spherical-Bessel primitive blocks without changing the fixed
DZP orbitals. The response-basis size is then chosen from the weighted
eigenvalue spectrum of the residual covariance separately in s, p, and d,
rather than from H2 binding-energy tuning. The first candidate will keep the
joint DFT+dpsi objective and append the number of d radial functions required
by that spectrum. Only after the atomic training-space gate passes is one new
all-band H2/H SOS-RPA held-out calculation allowed.

This angular decomposition does not prove that the missing d channel equals
the full `1.86 kcal/mol` binding-energy gap: RPA energy is nonlinear and H/H2
errors can cancel. It identifies a hard, quantified training-space floor that
must be removed before auxiliary-basis thresholds, the 50-Ry atomic target
grid, or smaller residual effects can be interpreted.

## d-channel implementation status

ABACUS commit `efc128f335319114dad6f6e35bd07ceaa8bd15af` adds the output-only
input `sternheimer_siab_lmax`. Setting it to `2` leaves the loaded H `3s2p`
LCAO space and Delta-ST fixed subspace unchanged, but exports one s, three p,
and five d primitive blocks. The TDD RED job `21315686` failed because the
input and primitive parameters did not yet expose `lmax`; GREEN job `21315747`
passed all seven SIAB producer tests, and input-parser job `21315765` passed.
Release build job `21315778` staged immutable executable
`abacus-efc128f33531` with SHA256
`96ffed27de8256214f6d32ad545925e78ab7f40adae19310fb7041399642cdae`.

The exact 16-frequency, 50-Ry H producer was resubmitted as normal-partition
job `21315811`. All old physical inputs have identical hashes; its INPUT
diff contains only the suffix and `sternheimer_siab_lmax 2`. Its live progress
record has already confirmed `nprimitive=225`, i.e. nine complete 25-column
blocks. The final target hash and residual spectrum remain pending until that
job completes.

The joint objective also needs d-resolved DFT/dpsi matrices; reusing the old
`lmaxmax 1` H3 files would leave d orbitals controlled only by ST. Normal array
job `21315842` regenerated the same 0.7, 0.9, and 1.3 Angstrom H3 references
with only `lmaxmax` changed to 2. The `(origin,dpsi)` matrix SHA256 pairs are:

```text
0.7 A  82419e037258253c2bcca4ba9e74e738d717964ca905d22c96f3152d66f44988
       6c192a8aa7390d3cdd10a31baf3d833ea91c8113853f3eb4297bf1638966849e
0.9 A  e71cdc61f19d253e50d86fe2125f4a1e17e4981d57458254c96e656059f7d804
       051f3ccd888e7913393d2d6fdca9c9a33e2cc2dfbc773e814bd91a101d65ee76
1.3 A  a1bacb3f1ff59317de82c3e82191cda107e30aba040988d088bfb9a6c86bf670
       22f51f9a4d8fdd22ac1c3308a1e917a2e4d7b33e853930231763a0d0694831e4
```

The optimizer now computes a radial residual spectrum after projecting the
fixed DZP core. For each `l`, it checks that all `m` channels have the same
projected primitive overlap, sums their weighted response covariances, whitens
by the numerical-rank overlap, and diagonalizes the resulting real symmetric
matrix. The returned eigenvectors are valid real SIAB radial coefficients;
the cumulative eigenvalues determine the shell count without consulting H2.
Focused GREEN job `21315892` passed both analytic spectrum tests, and full
normal-node regression job `21315899` passed all 107 SIAB tests. The real d
shell count is intentionally not declared before job `21315811` finishes.
