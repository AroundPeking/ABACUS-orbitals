# Source-aware projected-Pi feasibility gate

This directory contains a read-only analysis. It pairs ABACUS source-v1 and
response-v1 files, reconstructs the full-Coulomb symmetrized projected
response, and ranks three fixed-size H TZDP coefficient sets. It does not
modify the optimizer and does not use SOS-RPA energy or H+ghost data.

Required physical inputs are one paired response/source and one validated
zero-order audit for H and H2, plus the initial TZDP, fixed-DZP joint, and
low-frequency-guarded `ORBITAL_RESULTS.txt` files. The coefficient interface is
exactly H `3s2p` with 25 radial primitives; `1s,2s,1p` must agree with the
initial basis within `1e-12`.

The command writes JSON, Markdown, PNG, and PDF diagnostics atomically. Exit
status `0` means both optimized bases improve the initial total loss at all
three primitive-overlap thresholds and every family is stable within 1%.
Exit status `2` means the complete diagnostics were written but the method
must stop before optimizer integration and move to a Galerkin-Sternheimer
design. Any malformed input, failed audit, or pairing error exits `1` before
creating the output directory.

The validated `df_dcu` runtime is Python 3.10, PyTorch 2.1.0+cpu, and NumPy
1.26.4. Matplotlib 3.10.3 is installed separately under
`/work1/ghj/runtime/siab-projected-pi-mpl-20260801`; add that directory to
`PYTHONPATH` instead of modifying the fixed SIAB runtime. The complete SIAB
Python regression suite passed 276 tests with this environment.

```bash
export PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801
python analyze_projected_pi.py \
  --h-response H/sternheimer_matrix.dat \
  --h-source H/STERNHEIMER_SIAB_SOURCE_V1.dat \
  --h-audit H_zero_order_identity.json \
  --h2-response H2/sternheimer_matrix.dat \
  --h2-source H2/STERNHEIMER_SIAB_SOURCE_V1.dat \
  --h2-audit H2_zero_order_identity.json \
  --initial initial_tzdp_ORBITAL_RESULTS.txt \
  --joint fixed_dzp_joint_ORBITAL_RESULTS.txt \
  --guarded low_frequency_guarded_ORBITAL_RESULTS.txt \
  --output-dir projected_pi_result
```

## Physical ranking result

Commit `ecedcd80` was staged under
`/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_feasibility_20260801`.
Job `21464513` completed on one `normal` node with 30 CPUs and 110610 MB. The
analysis took 74.30 s and used 1,905,692 KiB maximum resident memory according
to `/usr/bin/time -v`. Both strict pairs and zero-order audits passed without
warnings, fixed `1s,2s,1p` differences were exactly zero, and the decision was
`pass`.

At the nominal primitive-overlap rank threshold `1e-12`:

| basis | H loss | H2 loss | equal-family total | H/H2 rank | max condition | held-out D CP |
|---|---:|---:|---:|---:|---:|---:|
| initial TZDP | 0.1422168377 | 0.0912063047 | 0.2334231424 | 534/1032 | 3.073394e4 | 105.556881 |
| fixed-DZP joint `3s2p` | 0.1329357582 | 0.0816364120 | 0.2145721702 | 534/1032 | 5.562937e3 | 105.853882 |
| low-frequency-guarded `3s2p` | 0.1333449050 | 0.0821021186 | 0.2154470237 | 534/1032 | 5.397578e3 | 105.843252 |

The CP values are independent held-out SOS-RPA results in kcal/mol; they were
not read by the projected-Pi command. The projected-Pi order is joint,
guarded, initial from best to worst, matching the held-out CP order. The same
strict order holds at all 16 frequencies for both H and H2. Across rank
thresholds `1e-10`, `1e-11`, and `1e-12`, the maximum relative loss spread is
`9.6355e-7`, well below the 1% gate. Candidate and reference Hermitian errors
are zero at stored precision.

This result validates projected-Pi as a training-target candidate; it does not
create or validate a new orbital basis. The next implementation is specified
in `docs/superpowers/plans/2026-08-01-siab-pi-dpsi-joint.md`. Raw JSON,
Markdown, plots, scheduler records, and input/code hashes are preserved under
`results/`.

## Frozen `pi_dpsi_joint` optimizer campaign

`INPUT.pi_dpsi_joint` is the first source-aware optimization input. It starts
from the current fixed-DZP joint H `3s2p` coefficients (SHA256
`1340cd11357dea87b67ad2a58a6a8e1ae298c985bf08a66b6e9456c57dbc87df`),
uses 25 radial primitives at 8 bohr, and fixes `1s`, `2s`, and `1p` exactly.
Only `3s` and `2p` are trainable. The DFT and ordinary dpsi data are the same
three H3-distance targets used by the original joint campaign. The new primary
target consists of exactly one audited H pair and one audited H2 pair from the
16-frequency, full-Coulomb-whitened producer.

The optimized scalar is the equal-family sum `L_H + L_H2`. The ordinary dpsi
ratio has weight 1, and the DFT/dpsi acceptance hinges remain 1.05 and 1.10.
The primitive-overlap rank tolerance is `1e-12`; condition numbers above
`1e12` are rejected. No ghost target, SOS energy, radial-tail penalty, or old
lowest-frequency spillage guard enters training.

`run_pi_dpsi_joint.slurm` requires an immutable campaign directory containing
`code/`, `inputs/`, `SOURCE_COMMIT`, `SOURCE_MANIFEST.sha256`, and
`INPUTS.sha256`. It runs only on one full `normal` node with 30 CPU threads,
110610 MB, and a 24-hour limit, using the fixed Python 3.10/PyTorch 2.1 runtime.
It refuses an existing result directory and validates all code and input
hashes before optimization. The primary outputs are `ORBITAL_RESULTS.txt`,
`Spillage.dat`, and `PROJECTED_PI_METADATA.json`; the JSON stores all
frequency/family losses and source/response/audit provenance.

Before freezing this campaign, the complete SIAB Python suite passed 299 tests
on `df_dcu` in 54.18 s of unittest time and 60.09 s wall time, with 246800 KiB
maximum RSS. This is a software gate only.

## Optimized basis and independent SOS/CP result

The optimizer implementation and campaign were frozen through commits
`a1c0422b`, `d2cb3576`, `68536865`, `f3851ded`, `ebd0aa50`, and `5e4f6a72`.
Commit `3c0f40ce` made the immutable runner compatible with the Git 1.8.3.1
installed on `df_dcu`. One-step job `21471688` proved that only `3s,2p`
changed. Independent ten-step jobs `21471713` and `21471714`, run on different
nodes, produced byte-identical `ORBITAL_RESULTS.txt`, `ORBITAL_1U.dat`,
`Spillage.dat`, and `PROJECTED_PI_METADATA.json`.

Production job `21471739` ran on one `normal` node with 30 CPUs and 110610 MB.
It completed in 4:31 with exit code `0:0`; the optimizer itself took 3:56.96
and stopped after 439 accepted updates because the objective had converged.
All 440 logged rows, including the initial row, satisfied the acceptance
contract. The final fixed `1s,2s,1p` coefficient difference was exactly zero.

| training quantity | initial | final | final / initial |
|---|---:|---:|---:|
| projected Pi, H + H2 | 0.2145720661 | 0.2105926634 | 0.981454 |
| DFT loss | 5.0667854e-5 | 3.7959370e-5 | 0.749181 |
| ordinary dpsi loss | 5.9172304e-4 | 3.9033108e-4 | 0.659652 |
| lowest-frequency projected Pi | 0.0324551393 | 0.0357745610 | 1.102278 |

The final family losses are `L_H=0.1316161757` and
`L_H2=0.0789764877`; the maximum candidate condition is `2.9866e3`, below
the `1e12` limit. The lowest-frequency diagnostic increased, but it is not a
stop rule in this source-aware objective: no local-frequency guard or radial
tail penalty entered the loss. The exported orbital SHA256 is
`a839338e1c1fcaf043a65178099d77e9580c9fe87243f372e85ddde5b72d2d01`.

Commits `0a488d77` and `4a8a617b` add and correct the independent SOS/CP runner.
Array `21471869_[0-2]` ran H2, H, and H+ghost in 3:14, 2:39, and 5:49. All
three tasks completed with exit code `0:0`. They use all `18/9/18` bands, the
same 20-Angstrom cell and 0.74085-Angstrom bond, 100 Ry, 16 frequencies,
explicit 214-function-per-H ABS, `rpa_ccp_rmesh_times=5`, and full Coulomb as
the fixed-DZP joint control. For every geometry, KPT, `librpa.in`,
pseudopotential, explicit ABS, `basis_aux_out`, and the full-Coulomb matrix are
byte-identical to that control. INPUT and STRU are also identical after
normalizing only the suffix and orbital filename.

The LibRPA correlation energies are

```text
EcRPA(H2)      = -0.068995044 Ha
EcRPA(H)       = -0.016546125 Ha
EcRPA(H+ghost) = -0.017261200 Ha
```

| basis | D raw | D CP | BSSE | D0 CP | RPAc CP |
|---|---:|---:|---:|---:|---:|
| initial TZDP | 106.635342 | 105.556881 | 1.078461 | - | - |
| fixed-DZP joint `3s2p` | 106.886054 | 105.853882 | 1.032171 | 84.270376 | 21.583507 |
| low-frequency-guarded `3s2p` | 106.881909 | 105.843252 | 1.038657 | - | - |
| projected-Pi joint `3s2p` | 106.823178 | 105.932932 | 0.890247 | 84.301021 | 21.631911 |

All table energies are in kcal/mol. Relative to the previous best same-size
basis, projected-Pi optimization raises the CP binding by `0.079050` kcal/mol
and lowers BSSE by `0.141924` kcal/mol. The CP improvement contains
`+0.030645` kcal/mol from the zero-order term and `+0.048404` kcal/mol from
RPA correlation. It therefore passes both frozen promotion inequalities,
`D_CP > 105.853882` and `BSSE <= 1.082171`, without using SOS or ghost data in
training. It becomes the current best validated same-size `3s2p` basis under
this contract, but remains `2.787068` kcal/mol below the approximately 108.72
kcal/mol Delta-Sternheimer/FHI-aims target. The next improvement must address
the remaining response-space limitation rather than interpreting this
promotion gate as basis convergence.

## First response-space extension: `3s2p1d`

Frozen projected-Pi screening showed that the first `d` shell gives the
largest loss reduction per added AO. The `3s2p1d` campaign therefore keeps
the same fixed `1s,2s,1p` DZP core, starts `3s,2p` from the validated
projected-Pi basis, and starts `1d` from the previous smooth response-basis
candidate. No SOS or ghost quantity enters screening or training.

The original `joint_dpsi_weight=1` run `21473252` reduced the normalized
ordinary-dpsi term but increased projected Pi from `0.140954930` to
`0.155962458`. The cause is explicit in the objective: the initial dpsi ratio
is one while projected Pi is only about `0.141`, so the normalized dpsi term
dominates once the new `d` space is available. A frozen four-point training
sweep, with every other input and seed unchanged, gave:

| dpsi weight | job | projected Pi | DFT ratio | dpsi ratio | max condition |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 21473547 | 0.1108519974 | 1.049698 | 0.972943 | 1.3783e4 |
| 0.02 | 21473565 | 0.1093567658 | 1.049625 | 0.875856 | 1.4510e4 |
| 0.05 | 21473610 | 0.1146198830 | 1.049930 | 0.724667 | 8.2634e3 |
| 0.10 | 21473611 | 0.1344652268 | 0.671839 | 0.439707 | 6.0638e3 |
| 1.00 | 21473252 | 0.1559624580 | 0.566021 | 0.326483 | 2.7528e3 |

Weight `0.02` is selected because it has the lowest accepted projected-Pi
loss while improving ordinary dpsi and remaining inside the 5% DFT gate. Its
family losses are `L_H=0.0670012137` and `L_H2=0.0423555520`; fixed DZP
coefficients remain byte-exact. The `3s`, `2p`, and `1d` radial functions are
smooth under direct inspection. Commit `86a93bd0` freezes this weight.

Commit `25554361` also removes the old hard-coded `18/9/18` SOS band counts.
The independent runner now derives the AO count from `ORBITAL_1U.dat`; for
`3s2p1d` it obtains 14 AO per H and therefore uses all `28/14/28` bands.
Array `21473755_[0-2]` completed H2, H, and H+ghost in 3:22, 3:03, and 6:31,
respectively. It retains the 20-Angstrom cell, 100 Ry, 16 frequencies, the
same explicit 214-function-per-H ABS, `rpa_ccp_rmesh_times=5`, and full
Coulomb. The three `basis_aux_out` and full-Coulomb hashes are identical to
the validated `3s2p` control.

```text
EcRPA(H2)      = -0.073504639 Ha
EcRPA(H)       = -0.018089831 Ha
EcRPA(H+ghost) = -0.018772066 Ha
```

| basis | D raw | D CP | BSSE | D0 CP | RPAc CP |
|---|---:|---:|---:|---:|---:|
| projected-Pi joint `3s2p` | 106.823178 | 105.932932 | 0.890247 | 84.301021 | 21.631911 |
| projected-Pi joint `3s2p1d` | 107.705560 | 106.852620 | 0.852940 | 84.287061 | 22.565559 |

The first `d` shell raises CP binding by `0.919688` kcal/mol and lowers BSSE
by `0.037307` kcal/mol. It is a real response-space improvement, but it
remains `1.867380` kcal/mol below the approximately 108.72 kcal/mol
Delta-Sternheimer/FHI-aims target. The next frozen shell-screening choice is
the second `d` shell; further tuning of the one-`d` weight is not promoted.

## Second d response shell

Commit `ca409b9a` freezes the next campaign as H `3s2p2d`: the validated
`3s2p1d` coefficients are retained, the second smooth `d` candidate is
appended, and `1s`, `2s`, and `1p` remain fixed. The objective and all source,
DFT, dpsi, grid, seed, and response contracts are unchanged, including
`joint_dpsi_weight=0.02`.

The production training job `21473908` completed in 9:40 on one normal node
with 30 CPUs and 110610 MB. Projected Pi decreased from the unoptimized
second-d value `0.0966347563` to `0.0815966206`; this is 25.38% below the
selected one-d value. The final H and H2 components are `0.0535314304` and
`0.0280651901`. DFT and ordinary-dpsi losses are `0.726367` and `0.371253`
times their campaign initial values, the lowest-frequency diagnostic is
`0.0071291030`, and the maximum condition number is `9.3404e3`. The fixed
coefficient differences are exactly zero. Direct radial inspection shows
smooth `3s`, `2p`, and first-d changes; the optimized second d is tighter and
nodal but has no high-frequency oscillation. The selected orbital SHA256 is
`bba770f62dab907aa24c72a817b5ba7ee7c789e417754caf050e16fe06462e52`.

The independent all-band SOS/CP array `21474012_[0-2]` derived 19 AO per H
and used `38/19/38` H2/H/H+ghost bands. The tasks completed in 3:44, 3:12,
and 7:32. The 20-Angstrom cell, 0.74085-Angstrom bond, 100 Ry, 16 frequencies,
explicit 214-function-per-H ABS, `rpa_ccp_rmesh_times=5`, Massidda correction,
and LibRPA full Coulomb are identical to the one-d control. All three
`basis_aux_out` and full-Coulomb hashes are byte-identical to that control.

```text
EcRPA(H2)      = -0.074941838 Ha
EcRPA(H)       = -0.018521099 Ha
EcRPA(H+ghost) = -0.019241671 Ha
```

| basis | D raw | D CP | BSSE | D0 CP | RPAc CP |
|---|---:|---:|---:|---:|---:|
| projected-Pi joint `3s2p1d` | 107.705560 | 106.852620 | 0.852940 | 84.287061 | 22.565559 |
| projected-Pi joint `3s2p2d` | 108.134558 | 107.243314 | 0.891244 | 84.365262 | 22.878052 |

The second d raises CP binding by `0.390694` kcal/mol. Its zero-order and
RPA-correlation contributions improve by `0.078201` and `0.312493` kcal/mol,
while BSSE increases by only `0.038304` kcal/mol. The CP improvement is
therefore physical response-space progress rather than a smaller-BSSE
artifact. The remaining gap to 108.72 kcal/mol is `1.476686` kcal/mol, so the
next controlled extension is the third d radial shell.

## Third d response shell: rejected by CP

Commit `5d4e5f01` freezes the `3s2p3d` experiment before seeing its SOS
energy. Its initial coefficients are exactly the selected `3s2p2d` result plus
the third radial d mode from the old smooth response candidate. Server-side
contract tests passed 6/6, and smoke job `21474153` verified all eight loaded
coefficient columns. Production job `21474175` completed in 6:08.

The third d lowers projected Pi from the selected two-d value `0.0815966206`
to `0.0769860619`, a 5.65% reduction. DFT and ordinary-dpsi losses are
`0.935664` and `0.525015` times their campaign initial values; the
lowest-frequency diagnostic is `0.0063272859`, the maximum condition number
is `1.4458e4`, and the fixed DZP coefficient differences remain exactly zero.
The optimized third d is tight and nodal, with substantially larger radial
curvature than the first two d functions, but it remains smooth on the
0.01-bohr mesh.

The independent array `21474238_[0-2]` used 24 AO per H and all
`48/24/48` H2/H/H+ghost bands. It completed in 4:04, 3:22, and 8:44 under
the same 20-Angstrom, 100-Ry, 16-frequency, fixed-ABS, and full-Coulomb
contract. Auxiliary-basis and full-Coulomb hashes again match the two-d
control byte for byte.

| basis | D raw | D CP | BSSE | D0 CP | RPAc CP |
|---|---:|---:|---:|---:|---:|
| projected-Pi joint `3s2p2d` | 108.134558 | **107.243314** | 0.891244 | 84.365262 | 22.878052 |
| projected-Pi joint `3s2p3d` | 108.403727 | 107.220076 | 1.183651 | 84.363696 | 22.856381 |

Although raw binding rises by `0.269170` kcal/mol, BSSE rises by `0.292407`
kcal/mol. The CP result therefore decreases by `0.023238` kcal/mol, with
zero-order and RPA-correlation CP changes of `-0.001566` and `-0.021671`
kcal/mol. This candidate is rejected and `3s2p2d` remains the selected basis.
Projected-Pi training loss alone is no longer monotone with held-out CP once
the third d is added. Further d shells are stopped; the next response-space
screen must use the existing l=3 source blocks to construct and rank a first
f residual mode before any new SOS calculation.

## First f response shell and regenerated PCA auxiliary basis

The auxiliary-basis contract is now fixed by the project requirement: every
wave-function orbital is evaluated with the auxiliary basis generated from
that orbital at `exx_pca_threshold=1e-4`. The SOS runner removes the template
`ABFS_ORBITAL` block and no longer supplies the fixed 214-function-per-H ABS.
Different wave-function bases may therefore produce different `basis_aux_out`
dimensions. This is intentional. The earlier fixed-ABS energies remain useful
response-space diagnostics, but the selected `3s2p2d` baseline must be rerun
under this regenerated-PCA contract before comparison with the new f basis.

`build_l3_residual_seed.py` projects the selected `3s2p2d` coefficients out of
the spherical H target, diagonalizes the remaining l=3 radial covariance, and
appends only its leading eigenmode. H2 is not used to define the shared radial
metric because its Cartesian molecular environment splits the magnetic
channels; H and H2 both remain in the subsequent projected-Pi optimization.
For the current target, the leading f mode has eigenvalue `0.0765909347`,
captures `0.6994582574` of the l=3 residual covariance, and gives a spectral
gain of `0.0109415621` per added AO. The measured magnetic-channel overlap
deviation is `3.76107e-5`, below the predeclared `1e-4` uniform-grid tolerance.

The frozen candidate is `3s2p2d1f`: fixed `1s,2s,1p`, selected two-d basis as
the starting point, one residual-derived f mode, `joint_dpsi_weight=0.02`, and
no radial-tail or low-frequency penalty. Promotion still requires an
independent all-band H2/H/H+ghost LibRPA SOS/CP calculation. Both the two-d
control and the f candidate must regenerate their own PCA-1e-4 auxiliary basis.

Production optimizer job `21474551` completed on one normal node with 30 CPU
cores and 110610 MB in 7:03. It stopped after 520 accepted steps. The total
loss decreased from `0.0786841309` to `0.0733168681`; the final projected-Pi
loss is `0.0612267667`, split into `0.0420124701` for H and `0.0192142965` for
H2. The fixed `1s,2s,1p` coefficient differences are exactly zero. The final
orbital has 26 AO per H and SHA256
`4171a07bc752256aca4d64df02a8de773f367a3b9501678a4aa6c11118477249`.

The independent PCA-1e-4 SOS/CP arrays `21474591_[0-2]` and
`21474592_[0-2]` completed successfully. They used all `38/19/38` bands for
the two-d H2/H/H+ghost control and all `52/26/52` bands for the f candidate.
The regenerated auxiliary dimensions are `222/111/222` and `364/182/364`,
respectively. Thus the comparison fixes the PCA threshold, not the auxiliary
dimension.

| basis, regenerated PCA `1e-4` | D raw | D CP | BSSE | D0 CP | RPAc CP |
|---|---:|---:|---:|---:|---:|
| projected-Pi joint `3s2p2d` | 108.211885 | 107.421137 | 0.790748 | 84.351029 | 23.070107 |
| projected-Pi joint `3s2p2d1f` | 108.566874 | **107.626402** | 0.940472 | 84.345316 | 23.281086 |

The first f shell improves the CP binding by `0.205265` kcal/mol. Its
correlation contribution improves by `0.210979` kcal/mol while the zero-order
part changes by `-0.005714` kcal/mol. BSSE increases by `0.149725` kcal/mol,
so the uncorrected `108.566874` kcal/mol must not be interpreted as convergence
to the 108.72-kcal/mol reference. The CP result remains low by `1.093598`
kcal/mol, but it passes the held-out gate and replaces `3s2p2d` as the selected
response basis under the regenerated-PCA contract.

After optimizing the first f shell, the next l=3 residual mode has leading
eigenvalue `0.0308159406` and gain `0.0044022772` per added AO. The leading
l=4 mode has eigenvalue `0.0416458377` and gain `0.0046273153` per added AO.
The latter is 5.11% larger, so the next greedy candidate is `3s2p2d1f1g`, not
a second f shell. The Cartesian-grid magnetic-overlap deviation for l=4 is
`1.23551e-4`; the seed construction therefore uses the predeclared `2e-4`
uniform-grid tolerance while leaving the physical PCA threshold at `1e-4`.

`build_residual_shell_seed.py` generalizes the deterministic atomic residual
construction to any available angular channel. For the selected l=4 mode it
starts from the final `3s2p2d1f` coefficients, appends exactly one g radial
column, and produces SHA256
`9e3773070807be301bb4b5beed8865a1204f57507316bcdac8c54b0eafe1a0e0`.
The resulting `3s2p2d1f1g` candidate has 35 AO per H, keeps the same frozen
DZP and `joint_dpsi_weight=0.02` contract, and remains unselected until its
independent all-band PCA-1e-4 SOS/CP gate is complete.

Production optimizer job `21474631` completed on one normal node with 30 CPU
cores and 110610 MB in 6:00. It stopped after 352 accepted steps. Total loss
decreased from `0.0709582372` to `0.0694982341`; projected-Pi changed from
`0.0509582372` to `0.0513963414`, while the dpsi regularizer decreased from
`0.0200000000` to `0.0181018927`. The fixed `1s,2s,1p` coefficient
differences remain exactly zero. The final 35-AO-per-H orbital has SHA256
`6e5775670067ea42a36474a042b353c8bf630948b46beaa1cbb7190e5b668503`.

The independent array `21474637_[0-2]` used all `70/35/70` H2/H/H+ghost
bands. Regenerated PCA-1e-4 auxiliary dimensions are `588/294/588`.

| basis, regenerated PCA `1e-4` | D raw | D CP | BSSE | D0 CP | RPAc CP |
|---|---:|---:|---:|---:|---:|
| projected-Pi joint `3s2p2d1f` | 108.566874 | 107.626402 | 0.940472 | 84.345316 | 23.281086 |
| projected-Pi joint `3s2p2d1f1g` | 108.958806 | **107.888474** | 1.070332 | 84.348056 | 23.540417 |

The first g shell raises CP binding by `0.262072` kcal/mol: zero order and
RPA correlation contribute `0.002741` and `0.259331` kcal/mol. BSSE increases
by `0.129860` kcal/mol, so the raw `108.958806` kcal/mol value is again not a
convergence result. The CP gap to 108.72 kcal/mol is still `0.831526`
kcal/mol, but the g shell passes and is retained.

After reoptimization, the next f residual mode has eigenvalue `0.0287315351`
and gain `0.0041045050` per AO. The next g residual mode has eigenvalue
`0.0186048293` and gain `0.0020672033` per AO. The next greedy extension is
therefore a second f radial shell, giving the frozen `3s2p2d2f1g` candidate.

The second-f production job `21474683` completed in 3:03. The deterministic
residual mode itself reduced projected-Pi from `0.0513963414` to
`0.0460604943`; no subsequent Adam step improved the combined objective over
that seed. The emitted radial orbitals are identical to the seed after the
standard smoothing and orthogonalization operations. The candidate has 42 AO
per H. Independent all-band array `21474692_[0-2]` used `84/42/84` bands and
generated `746/373/746` PCA-1e-4 auxiliary functions for H2/H/H+ghost.

| basis, regenerated PCA `1e-4` | D raw | D CP | BSSE | D0 CP | RPAc CP |
|---|---:|---:|---:|---:|---:|
| selected `3s2p2d1f1g` | 108.958806 | **107.888474** | 1.070332 | 84.348056 | 23.540417 |
| candidate `3s2p2d2f1g` | 108.810647 | 107.805320 | 1.005327 | 84.351957 | 23.453362 |

Despite the lower training loss, the second f shell lowers CP binding by
`0.083154` kcal/mol. Its zero-order contribution rises by `0.003901`, but the
RPA-correlation contribution falls by `0.087055` kcal/mol. Reduced BSSE does
not rescue the physical result. The second f shell is rejected and the
selected basis remains `3s2p2d1f1g`. The next and final shell-growth control
is the lower-ranked second g residual mode; if it also fails CP, high-l shell
growth stops and the loss construction, rather than basis size, must be
revisited.

The second-g production job `21475221` completed on one normal node with 30
CPU cores and 110610 MB in 7:06. The total loss decreased from
`0.0685021900` to `0.0682289924`; projected-Pi changed from `0.0485021900`
to `0.0485189086`, while the dpsi regularizer decreased from `0.0200000000`
to `0.0197100838`. The fixed `1s,2s,1p` coefficient differences are exactly
zero. The final `3s2p2d1f2g` orbital has 44 AO per H and SHA256
`52311fe14f3b0597eb18a6b99a0b9f216fe0db49c0719c3a6e72d6fc7c51f368`.

The independent all-band array `21483740_[0-2]` used `88/44/88` bands for
H2/H/H+ghost and completed in 11:15, 10:10, and 33:43. Each case removed the
explicit `ABFS_ORBITAL` block and regenerated its auxiliary basis with
`exx_pca_threshold=1e-4`, giving dimensions `930/465/930`. H2 and H+ghost
used byte-identical full-Coulomb files.

| basis, regenerated PCA `1e-4` | D raw | D CP | BSSE | D0 CP | RPAc CP |
|---|---:|---:|---:|---:|---:|
| selected `3s2p2d1f1g` | 108.958806 | **107.888474** | 1.070332 | 84.348056 | 23.540417 |
| candidate `3s2p2d1f2g` | 108.728861 | 107.713459 | 1.015402 | 84.348998 | 23.364461 |

The raw second-g value is accidentally close to the 108.72-kcal/mol
reference, but counterpoise lowers it by `1.015402` kcal/mol. Relative to the
selected first-g basis, the second g raises the zero-order CP contribution by
only `0.000941` kcal/mol and lowers the RPA-correlation CP contribution by
`0.175956` kcal/mol, so total CP binding decreases by `0.175015` kcal/mol.
The candidate is rejected. Both the second f and second g reduce the training
objective without improving held-out full-Coulomb RPA/CP, so high-l shell
growth now stops at `3s2p2d1f1g`; the next work must revise the loss
construction and its frequency/channel weighting rather than add more AO
shells.

## RPA-sensitive fixed-size redesign: design frozen

The next cycle keeps the selected `3s2p2d1f1g` size fixed at 35 AO per H and
continues to freeze the DZP `1s,2s,1p` columns. It does not add another high-l
shell. The design is recorded in
`docs/superpowers/specs/2026-08-04-siab-rpa-sensitive-response-loss-design.md`.

For the primitive-reference symmetrized response, the RPA integrand and its
first variation are

```text
F(Pi) = Tr[log(I-Pi) + Pi]
delta F = Tr[(I-(I-Pi_ref)^-1) delta Pi].
```

The implementation will use this derivative to construct a positive
frequency/channel sensitivity norm. It retains a frozen fraction of the
current complete-matrix projected-Pi loss, because a scalar RPA-integrand
difference could cancel between channels or frequencies. Before any new
optimization, the revised metric must reproduce the independent decisions
for `3s2p2d`, first f, first g, rejected second f, and rejected second g. If no
frozen blend reproduces all four promotions/rejections, the projected target
is stopped rather than tuned to the CP energy.

The isolated H loss becomes an explicit no-regression gate, while H+ghost
remains outside the optimizer. The first new candidate is promoted only if it
improves both the current `107.888474` kcal/mol CP binding and the current
`1.070332` kcal/mol BSSE under the same PCA-`1e-4`, all-band, full-Coulomb
contract. This is a development gate, not the final workflow.

The final release criterion is that raw, CP, and Delta-ST H2 binding agree to
`0.1 kcal/mol` at `0.60`, `0.74085`, `1.00`, and `1.50` Angstrom, with Ecut
and frequency convergence below the same tolerance. Only after that gate is
the basis frozen for production; production then uses the raw SOS-RPA result
without H+ghost or CP correction. At this point only the design is complete:
no new ranking, optimized orbital, or RPA result has been produced.

### Implementation plan frozen (2026-08-04)

The executable task sequence is now recorded in
`docs/superpowers/plans/2026-08-04-siab-rpa-sensitive-response-loss.md`.
It separates RED tests, the backward-compatible implementation, historical
five-basis alpha selection, fixed-size optimization, and the equilibrium
SOS/CP gate into distinct commits. It also requires an autograd-versus-centered
finite-difference check before the new loss can be used. At design-freeze time
the state was **plan only**: no RED test, implementation, historical ranking,
new optimization, or physical result had been run. The current state is now
**Task 3 RED**. Task 2 GREEN below is historical evidence, not the current
stage. No Task 4 implementation, historical ranking, new optimization, or
physical result exists.

### Task 1 RED: per-family RPA sensitivity (2026-08-04)

The RED-only test source was added without changing production Python. The
remote GitHub checkout could not be created because `github.com` did not
resolve on `df_dcu`, so the test used the permitted disposable copy at
`/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task1-covariance`.
Its source state was a `git archive` of `SIAB` from
`36735e60f7be06a307ea620799fb637b5f422ffc` with archive SHA256
`25d3d32dda10a7a44a46d5d4698dda1e931798184baa80cffc7340aee5d33185`,
plus only the uncommitted `SIAB/tests/test_projected_pi.py` overlay with
SHA256 `fbccd4ea414eaeeb259b50a50288f4186825cee67f1d28e4b98fdc66445c3a23`.
The overlay archive SHA256 was
`98dbe377a1e6fc51b3461d4269dddd090152cdb3e8cfb67578b51adda20b5910`.
The untouched production `projected_pi.py` had SHA256
`f78d08285b296a0a41acb557bf7df622ebc8b3e74e44ba2548705ff8e02e505b`.

The corrected error fixtures independently calculate the per-frequency
minimum eigenvalue of `I-Pi` before calling the unsupported API. Their df_dcu
values are:

- reference-only invalid: reference
  `[0.9904237367701484, -0.05052223708956394]`, candidate
  `[0.9896330875745136, 0.5220694792539988]`;
- candidate-only invalid: reference
  `[0.04237367701483419, 0.5224898922320165]`, candidate
  `[-0.036691242548635206, 0.7827588542063632]`.

Thus each fixture has exactly one invalid side at `relative_tolerance=1e-12`;
the reference test no longer depends on reference-first validation order.

Common auxiliary-channel permutation coverage now includes covariance of both
`candidate_pi` and `reference_pi`, in addition to scalar-loss invariance. Both
matrices must equal the original matrix with both channel axes permuted, using
`rtol=atol=1e-13`. On df_dcu the maximum covariance residual was exactly zero
for both matrices, while
`max(abs(permuted.candidate_pi - original.candidate_pi))` was
`5.0198819048475868e-05`, above the explicit `1e-6` nontriviality threshold.
Thus an evaluator that ignores the auxiliary-channel labels cannot pass this
test.

The exact RED command was:

```bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task1-covariance/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_projected_pi
```

It ran 19 tests in 0.166 seconds: all 11 legacy tests passed and exactly these
8 new tests errored as expected:

- `test_rpa_sensitivity_matches_independent_eigendecomposition`
- `test_rpa_sensitivity_directional_gradients_match_centered_difference`
- `test_rpa_sensitivity_common_source_response_phase_is_invariant`
- `test_rpa_sensitivity_common_auxiliary_channel_permutation_is_invariant`
- `test_rpa_sensitivity_rejects_nonpositive_reference_dielectric`
- `test_rpa_sensitivity_rejects_nonpositive_candidate_dielectric`
- `test_rpa_sensitivity_rejects_numerically_zero_reference`
- `test_rpa_sensitivity_rejects_nonfinite_pi`

Every new test stopped for the same feature-specific reason:
`TypeError: ProjectedPiEvaluator.__init__() got an unexpected keyword argument
'sensitivity_alpha'`. The exit status was 1. There was no import, fixture,
or test-algebra failure. This is RED evidence only: no implementation or physical result exists.

### Historical Task 2 GREEN: per-family RPA sensitivity implementation (2026-08-04)

`SIAB/opt_orb_pytorch_dpsi/projected_pi.py` now provides the standalone
`evaluate_rpa_sensitivity(reference_pi, candidate_pi, frequency_weight,
relative_tolerance)` helper and an optional `sensitivity_alpha` argument on
`ProjectedPiEvaluator`. `ProjectedPiResult` appends these defaulted fields, so
existing positional and keyword constructors remain valid:

```text
base_loss
sensitivity_loss
frequency_base_loss
frequency_sensitivity_loss
trace_log_difference
minimum_reference_dielectric_eigenvalue
minimum_candidate_dielectric_eigenvalue
sensitivity_alpha
```

At each frequency the helper validates finite reference and candidate Pi, then
rejects a Hermitian residual larger than
`10 * relative_tolerance * max(1, ||Pi||_F)` before any eigensolve. Accepted
matrices are diagonalized as Hermitian matrices. With
`epsilon_a = 1 - lambda_a`, it requires `min(epsilon_a) >
relative_tolerance` for both spectra and defines

```text
g_a = abs(1 - 1 / epsilon_a)
W^(1/2) = U diag(sqrt(g_a / max(g))) U^H
e_p = ||W_p^(1/2) (Pi_candidate,p - Pi_reference,p) W_p^(1/2)||_F^2
r_p = ||W_p^(1/2) Pi_reference,p W_p^(1/2)||_F^2
L_sensitivity = sum_p w_p e_p / sum_p w_p r_p
L_sensitivity,p = e_p / r_p
```

The diagnostic at each frequency is

```text
Delta F_p = Tr[log(I-Pi_candidate,p) + Pi_candidate,p]
          - Tr[log(I-Pi_reference,p) + Pi_reference,p].
```

When `sensitivity_alpha` is supplied, it is normalized to a finite scalar in
`[0,1]` and the evaluator returns

```text
loss_p = alpha * base_loss_p + (1-alpha) * sensitivity_loss_p
loss   = alpha * base_loss   + (1-alpha) * sensitivity_loss.
```

When the argument is absent, the evaluator returns directly from the legacy
equations, does not run dielectric checks, and leaves the appended fields
`None`. The original direct-complex oracle remains unchanged at
`rtol=atol=1e-13`.

The GREEN test used a source archive from starting commit
`de2becbf9a73fcf58769b681295cb44be98541e9` on branch
`codex/sternheimer-siab-h`, not a GitHub checkout. The archive contains the
committed `SIAB` tree and has SHA256
`8183788af219ab61516161ef6e8bc17fa06054c93b59905c18b24014f430eb90`.
The exact overlays were:

```text
SIAB/opt_orb_pytorch_dpsi/projected_pi.py
  f7843d17f23d6dba2ce00797ad49ae0f8a548164514bb19cf3e1ad4c827cc7ec
SIAB/tests/test_projected_pi.py
  043a219c9111b69a1b42ea0836918651d17da6c1a51ad73cd5e8cd24d03325f3
```

The untouched starting `projected_pi.py` SHA256 was
`f78d08285b296a0a41acb557bf7df622ebc8b3e74e44ba2548705ff8e02e505b`.
The disposable remote tree was
`/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task2-alpha-red`.
The exact first-attempt GREEN command was:

```bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task2-alpha-red/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_projected_pi test_projected_pi_optimization
```

It ran 30 tests in 0.254 seconds and returned `OK`: 11 legacy projected-Pi
tests, the 8 approved Task 1 sensitivity tests, 6 new Task 2 tests for missing,
endpoint, out-of-range and nonfinite alpha plus reference/candidate finite and
Hermitian validation, and 5 optimization-adapter regression tests.

**GREEN code test only; no historical ranking, optimization, or physical result**

### Current Task 3 RED: RPA-sensitive joint mode (2026-08-04)

This RED-only change modifies the three requested test files and no production
Python. The `df_dcu` GitHub probe failed with
`ssh: Could not resolve hostname github.com: Name or service not known`, so the
test used the documented source-archive plus exact-overlay fallback. The
remote tree was
`/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task3-red.5Q0Dgr`.

The committed source was `SIAB` from starting commit
`21238ce7fe8bf91498da589b1bf2f0ca5f180874`. Its uncompressed `git archive`
SHA256 was:

```text
9c2f77c8315417c84e87577f1893c9c7ad4bcd03cbc39ec3d56172af327dac0b
```

The overlay archive SHA256 was
`66ac86df42018e8bb37672c94ed6ed67e0bf2d2077ae17c785a9bb1727975ee6`.
It contained exactly:

```text
SIAB/tests/test_projected_pi_optimization.py
  c6b5699fccb2adb2c0da32db64e6d3d21d640042a713425b6dec5660aafb319f
SIAB/tests/test_loss_and_freeze.py
  24a0c0bed022284fe354e8ca2c310c063c8d38ca3963621dcb37501e24b546b3
SIAB/tests/test_main_sternheimer.py
  767b68cfcc2f7294f623189ee9babbc0cabde09bd311d8533e51224a593e162a
```

The untouched Task 2 production files in that source archive had SHA256:

```text
SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py
  134bb7085e87311adeb6940a82827399cb6770959b1601c086f5c71cf271c8e3
SIAB/opt_orb_pytorch_dpsi/optimization_loss.py
  c063ecfa23270beca67737b54035a7a7fcf5505909a073b49d9671be4a9da10d
SIAB/opt_orb_pytorch_dpsi/main.py
  f83ae7540186947e58c9ff671b629719cf5c86083c78f0e6983c28d066819056
```

Local and remote SHA256 values matched before the test. The exact command was:

```bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task3-red.5Q0Dgr/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_projected_pi_optimization test_loss_and_freeze test_main_sternheimer
```

The command exited 1 after 4.330 seconds. It ran 86 test methods. All 69
pre-existing methods passed, as did 5 new backward-compatibility/rejection
methods. Therefore 74 test methods passed. Ten new methods produced 15 failure
records because the nonfinite-alpha method has 3 failing subtests and the
forbidden-penalty method has 4 failing subtests; 2 other new methods errored.
The exact unittest summary was `FAILED (failures=15, errors=2)`.

The new failing methods and feature-specific reasons were:

- `test_rpa_sensitive_family_loss_uses_fourth_order_norm`: error because the
  adapter does not accept `sensitivity_alpha` (and therefore cannot yet accept
  the frozen fourth-order contract).
- `test_rpa_sensitive_joint_accepts_exact_required_contract`: error because
  `projected_pi_sensitivity_alpha` is still an unknown config key.
- `test_rpa_sensitive_joint_requires_explicit_alpha`: the new mode is still
  unknown, so the required-alpha diagnostic is absent.
- `test_rpa_sensitive_joint_rejects_nonfinite_alpha`: 3 failure records for
  `nan`, `inf`, and `-inf`; unknown-key rejection occurs before finite-alpha
  validation.
- `test_rpa_sensitive_joint_rejects_other_loss_penalties`: 4 failure records;
  unknown-alpha rejection occurs before the four mode-specific zero-penalty
  diagnostics.
- `test_rejects_mixed_rpa_sensitive_and_other_loss_stages`: routing still
  reports only the legacy `pi_dpsi_joint` mixing rule.
- `test_rpa_sensitive_loader_builds_one_physical_pair_per_family`: the loader
  returns no projected-Pi H/H2 pairs for the new mode.
- `test_rpa_sensitive_loader_requires_exactly_h_and_h2`: the new mode accepts
  a lone H target instead of rejecting it.
- `test_rpa_sensitive_loader_requires_source_path`: the new mode does not yet
  require the source side of the physical pair.
- `test_rpa_sensitive_main_requires_origin`: routing emits the legacy
  Sternheimer origin/dpsi diagnostic.
- `test_rpa_sensitive_main_requires_linear_dpsi`: routing reaches legacy
  `cal_weight` instead of rejecting missing dpsi; the deliberate sentinel
  confirms that exact boundary.
- `test_rpa_sensitive_routes_alpha_rank_and_fourth_order_power`: the new mode
  enters the legacy spillage route, so the projected-Pi adapter is never called.

There was no import, fixture, setup, or test-algebra error. The five new tests
that already pass cover alpha rejection in every legacy mode, below/above-range
alpha rejection, ghost rejection, and exact unchanged `pi_dpsi_joint` adapter
construction. This is **Task 3 RED only**: Task 4 has not started, and there is
no historical ranking, optimization, candidate orbital, or physical result.

### Task 3 RED spec-review correction (2026-08-04)

The preceding 86-method record is the historical first Task 3 RED. This
follow-up strengthens only the three test files; production Python remains
untouched. The `df_dcu` GitHub probe again failed with
`ssh: Could not resolve hostname github.com: Name or service not known`, so the
run used the documented source-archive plus exact-overlay fallback. The source
was `SIAB` at commit `36000cdbbfebb8d5aa6f9ee8b1d40b1bc83b0e3a`, and the
remote test tree was
`/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task3-strengthened-red.ZIvJSz`.

The uncompressed source archive SHA256 was:

```text
71865df0f6471a55fbb3222243a5cfc65964b385080a38203fb555ed27da7d5e
```

The exact test-overlay archive SHA256 was
`f869fe0511fc2735e5e6e0cdcde503c1dfccbe621b9dc32b61c81efcce0071eb`.
It contained exactly:

```text
SIAB/tests/test_projected_pi_optimization.py
  12ea60556002496f2b6bd7108234060bb9f7978f0317c50e09b155650b9142e4
SIAB/tests/test_loss_and_freeze.py
  9588a4d9079240edab85a08a4eb1fd0bfc1f1e5e8f4a80af926c0214d42d5fde
SIAB/tests/test_main_sternheimer.py
  fbe860644122f691644f23465b8a59b3aca1ab430e362e53d04a241d8aa341b6
```

Local and remote hashes matched. The untouched production hashes remained:

```text
SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py
  134bb7085e87311adeb6940a82827399cb6770959b1601c086f5c71cf271c8e3
SIAB/opt_orb_pytorch_dpsi/optimization_loss.py
  c063ecfa23270beca67737b54035a7a7fcf5505909a073b49d9671be4a9da10d
SIAB/opt_orb_pytorch_dpsi/main.py
  f83ae7540186947e58c9ff671b629719cf5c86083c78f0e6983c28d066819056
```

The exact command was:

```bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task3-strengthened-red.ZIvJSz/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_projected_pi_optimization test_loss_and_freeze test_main_sternheimer
```

The command exited 1 after 2.428 seconds and ran 97 test methods. Exactly 73
methods passed. Eighteen methods produced 23 failure records because the
nonfinite-alpha method has 3 failing subtests and the forbidden-penalty method
has 4; 6 other methods errored. The exact unittest summary was
`FAILED (failures=23, errors=6)`.

The strengthened contract is feature-specific:

- The legacy default adapter's exact `torch.equal` check against the explicit
  `H.loss + H2.loss` tensor passes.
- Independent omission tests for rank tolerance, sensitivity alpha, and dpsi
  weight fail because the new mode is unknown rather than issuing each required
  field diagnostic.
- Alpha endpoints `0.0` and `1.0`, the exact required contract, projected-Pi
  primary plus weighted-dpsi composition, and `selection_component(...)=total`
  error because the alpha key or new mode is unsupported.
- The family fourth-order norm errors because the adapter does not accept
  `sensitivity_alpha`.
- Nonfinite alpha produces 3 failure records and forbidden legacy penalties
  produce 4 because unknown-alpha rejection occurs before their new-mode
  validation.
- Four separate tests for mixing with `st_only`, `st_constrained`,
  `st_dpsi_joint`, and `pi_dpsi_joint` fail because new-mode exclusivity is not
  implemented.
- Loader/routing failures cover missing H2, duplicate H, duplicate H2, a
  syntactically valid ghost target, missing source, missing origin, missing
  dpsi, absent H/H2 pair construction, and absent forwarding of rank tolerance,
  alpha, and `family_power=4`.

There was no import, fixture, mock, setup, or test-algebra error. The missing
dpsi sentinel remains an intentional routing-boundary probe. This is still
**Task 3 RED only**. There is **no Task 4 implementation, historical ranking,
optimization, candidate orbital, or physical result**.

### Task 3 RED code-quality gap closure (2026-08-04)

The preceding 97-method result remains the historical spec-review RED. This
follow-up changes only `test_projected_pi_optimization.py` and
`test_main_sternheimer.py`; production Python remains untouched. The
`df_dcu` GitHub SSH probe produced no DNS resolution before its 10-second
timeout, so the run again used a committed-source archive plus exact test
overlay. The source is `SIAB` at commit
`0d12cdfa57d40f3136c6577a3125eaac5bbefab4`.

The first overlay attempt was discarded and is not RED evidence: import stopped
with `SyntaxError: too many statically nested blocks` in the routing helper.
Replacing the combined mock context with `contextlib.ExitStack` changed only
test setup. The corrected archive provenance below is the counted run.

The uncompressed source archive SHA256 was:

```text
285f61255461d5c7e4af7c5d895e4172f060e305514aeb1d399f2fb9a3e218ef
```

The corrected two-test overlay archive SHA256 was
`bb1e741ef4762cef809f363feac76c3c171a6bc06563bdb84a97ac1dcf0731fb`.
It contained exactly:

```text
SIAB/tests/test_projected_pi_optimization.py
  0258ef94d7b3606d0e6c3afb2b0d256397703e54dcc97ce3ca7c893744ef172c
SIAB/tests/test_main_sternheimer.py
  86cb9f8092f3f4466b43c4b8d391208433ef63f95ec7e67706dc6103042dfda8
```

Local and remote hashes matched. The untouched production hashes were:

```text
SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py
  134bb7085e87311adeb6940a82827399cb6770959b1601c086f5c71cf271c8e3
SIAB/opt_orb_pytorch_dpsi/optimization_loss.py
  c063ecfa23270beca67737b54035a7a7fcf5505909a073b49d9671be4a9da10d
SIAB/opt_orb_pytorch_dpsi/main.py
  f83ae7540186947e58c9ff671b629719cf5c86083c78f0e6983c28d066819056
SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py
  9a3d1b932d46d099b46f466149460440d5731f2e07598c32e111b3272922e968
```

The corrected remote input and test trees were:

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task3-quality-red-input-v2.lxxVkE
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task3-quality-red-v2.gEHjpW
```

The exact command was:

```bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task3-quality-red-v2.gEHjpW/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_projected_pi_optimization test_loss_and_freeze test_main_sternheimer
```

The command exited 1 after 0.933 seconds and ran 101 test methods. Exactly 73
methods passed. Twenty methods produced 25 failure records because the
nonfinite-alpha method has 3 failing subtests and the forbidden-penalty method
has 4. Eight methods produced 17 error records because the invalid-family-power
method has 10 erroring subtests. The exact unittest summary was
`FAILED (failures=25, errors=17)`.

All methods that passed in the preceding 97-method RED still pass. In
particular, the legacy aggregate now uses diagnostic-rich
`torch.testing.assert_close(..., rtol=0, atol=0)` for exact H+H2 equality, and
the unchanged `pi_dpsi_joint` constructor and projected-Pi objective setter
path both pass.

The code-quality additions fail only at unsupported Task 3 contracts:

- Direct H and H2 sensitivity evaluators both return positive losses and their
  loss difference exceeds `1e-10`, proving the fixture is nontrivial before the
  aggregate adapter errors on unsupported `sensitivity_alpha`.
- The fourth-order value test also requires `sensitivity_alpha=0.25` on both
  family results. A separate gradient-enabled coefficient test requires a
  nonzero aggregate gradient to agree with a centered `1e-6` finite difference
  within `3e-7`; it currently errors at the same unsupported constructor.
- Ten subtests require rejection of family powers `1`, `2`, `3`, `5`, `4.5`,
  `nan`, `inf`, `"4"`, `None`, and `True`. They currently error before exact-4
  validation because the sensitive adapter arguments are unsupported.
- The main-routing constructor now returns a unique objective sentinel and the
  test stops only at `cal_converge`. New-mode routing fails because the
  projected-Pi setter receives no sentinel and the legacy spillage route is
  installed instead; the old `pi_dpsi_joint` control reaches the same boundary
  with its exact old constructor and objective setter calls.
- Internal projected-Pi diagnostics preserve the existing family/frequency
  records but omit `mode=pi_rpa_sensitive_joint`, `sensitivity_alpha=0.25`, and
  `family_power=4`.
- The metadata-hook test does not inspect or define a Task 5 file schema. It
  fails because the new mode routes legacy ST diagnostics and makes zero calls
  to the existing projected-Pi metadata hook.

The remaining config and loader failures are the already documented Task 3
RED: required fields, alpha validation/endpoints, loss composition/selection,
legacy-mode exclusion, exact H/H2 physical campaigns, origin/dpsi requirements,
and adapter arguments. There was no import, fixture, mock-setup, or test-algebra
error in the counted run. This is still **Task 3 RED only**: there is **no Task
4 implementation, historical ranking, optimization, candidate orbital, or
physical result**.

### Task 4 code GREEN: `pi_rpa_sensitive_joint` (2026-08-04)

Task 4 adds the training mode without changing the numerical or construction
path of `pi_dpsi_joint`. The new mode requires all three fields explicitly:

```json
{
  "mode": "pi_rpa_sensitive_joint",
  "projected_pi_rank_tolerance": 1.0e-12,
  "projected_pi_sensitivity_alpha": 0.25,
  "joint_dpsi_weight": 0.02
}
```

The alpha must be a finite real number in `[0, 1]`; legacy modes reject the
alpha key. Radial-tail and low-frequency penalties must remain zero. A new
mode campaign cannot be mixed with another loss mode. Its target list is
strictly one physical `H` and one physical `H2` entry, each with response,
source, and zero-order-audit paths; ghost, duplicate, or incomplete entries
are rejected. Origin and linear dpsi data remain mandatory.

The adapter passes the frozen alpha to both family evaluators. It retains the
legacy exact sum when alpha and family power are absent. The sensitive path
accepts only integer `family_power=4` and evaluates, without detaching,

```python
family_losses = torch.stack(
    tuple(family.results[name].loss for name in ("H", "H2"))
)
loss = torch.sum(family_losses.pow(4)).pow(1.0 / 4)
```

Both family results expose the alpha and their blended sensitivity losses are
the values entering this norm. Loss composition treats both projected modes
as projected-Pi primary plus joint dpsi. The existing projected-Pi metadata
hook records the new mode, alpha, and family power. This stage does not define
the Task 5 checkpoint-acceptance contract or a new final output schema.

The implementation also makes the minimum matching change in
`opt_orbital_converge.py`: the new mode uses the already existing
`set_projected_pi_objective` route and projected-Pi diagnostics. No atomic or
family checkpoint-improvement gate from Task 5 is present.

#### Remote GREEN provenance

`df_dcu` cannot resolve `github.com`, so no remote Git checkout was claimed or
used. The final tested tree is the absolute directory:

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task4-green-impl2.Y3YKtG/code
```

It was built from a local `git archive` of `SIAB` at
`c5ca6ff8ed8e8a482e1c0eb02a9e8f73b747659f` plus an exact five-file production
overlay. The uncompressed source archive SHA256 is:

```text
285dd50035366151a2d738dda1c20025b63e28a61669c66771c850866ff5467d
```

The final overlay archive contains no AppleDouble or extended-attribute
entries. Its SHA256 is:

```text
55df65363b7ac8044498812e969e23dde990f87be23ae390d86a030d633821ea
```

Its exact contents and local/remote matching SHA256 values are:

```text
SIAB/opt_orb_pytorch_dpsi/projected_pi.py
  edc2837f2c6d85b6ffb7cc275d58e129612ef2740ac813dbe4728feb883a0258
SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py
  aa875b9ade5394fa038e366e792d8306c5e754538e0252a153ea2185f8662c9a
SIAB/opt_orb_pytorch_dpsi/optimization_loss.py
  6ead41b5b53e79ca9ae83c8210ebe843cb6c3c7b45638fd13b3ed6c1ec503729
SIAB/opt_orb_pytorch_dpsi/main.py
  1a9d35baff0b9e98fe85d9ff814516d386c10d97bf53796120af5685eb7c3ced
SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py
  b939ae4cb3cc6e85d4b150d141988d0a9e7ad6c6ab08bbec2a81e240b688f562
```

The first implementation feedback run used
`code-task4-green-impl1.bAtGcw`. It ran 126 tests and ended with one error:
the direct legacy metadata-hook test omitted `loss_mode`, while the first
implementation indexed that key. The compatibility fix uses the actual mode
when present and preserves `pi_dpsi_joint` as the old direct-call default.
That first overlay also contained macOS AppleDouble/xattr entries, so it is not
the final provenance archive.

The exact required final command was:

```bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task4-green-impl2.Y3YKtG/code/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_projected_pi test_projected_pi_optimization \
  test_loss_and_freeze test_main_sternheimer
```

It exited zero and reported `Ran 126 tests in 1.123s` and `OK`. The count is
126 rather than 101 because the requested command includes the 25 tests in
`test_projected_pi`; the three Task 3 modules alone were also rerun and
reported `Ran 101 tests in 0.863s`, `OK`.

The risk-based full SIAB discovery initially found five missing-fixture errors
because the `SIAB`-only source archive does not contain the tracked H TZDP
coefficient file. The same commit supplied that one file through a second
`git archive`, SHA256:

```text
e9e4d9f1868a6548db1d13346555ff4a9b2fc3ea528568a300dad29a164838db
```

After extracting it into the same code tree, the exact regression command was:

```bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task4-green-impl2.Y3YKtG/code/SIAB/tests
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest discover -v
```

It exited zero and reported `Ran 355 tests in 25.231s` and `OK`. This is code
GREEN only. No historical alpha ranking, optimization, checkpoint promotion,
new orbital basis, RPA binding energy, BSSE improvement, or physical validation
has been produced.

A fresh pre-commit verification on the same five matching production-file
hashes reported `Ran 126 tests in 1.056s`, `Ran 101 tests in 0.859s`, and
`Ran 355 tests in 24.329s`; all three commands exited zero with `OK`.

### Task 4 specification-review fix: real projected-Pi writer (2026-08-04)

The Task 4 specification review found a P1 integration gap. The main-path test
mocked `IO.func_C.write_C`, so the previous GREEN did not exercise the real
writer. `main.py` supplied `mode=pi_rpa_sensitive_joint`, but `func_C.py`
accepted and routed only `pi_dpsi_joint`. A real run therefore raised
`ValueError` before `ORBITAL_RESULTS.txt` was written and before the existing
`PROJECTED_PI_METADATA.json` hook could be reached.

The regression test now calls the real writer for both projected modes with
the same components and diagnostics. It asserts that the complete sensitive
output is byte-for-byte the legacy projected-Pi output with only the `Mode`
line changed. This covers mode validation, component validation/order,
diagnostic validation/order, and protects the old `pi_dpsi_joint` schema.

The production fix defines the local projected-mode set
`{"pi_dpsi_joint", "pi_rpa_sensitive_joint"}` in `IO/func_C.py` and uses it
for every projected-only writer branch. No component or diagnostic field was
added. In particular, there are no Task 5 initial/final H arrays, family arrays,
checkpoint-acceptance fields, or output-schema changes.

#### Review-fix remote provenance

The remote tree was assembled by archive and exact overlay, not by remote Git
checkout. Its absolute code directory is:

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task4-spec-fix.OYVRA0/code
```

The source archive contains `SIAB` plus the tracked H TZDP coefficient fixture
from commit `8eb18566364424d85dba252367394f2e65b06490`:

```text
db4942142e7c0a7727ad714f22480664a917f69f90c789b9c82e3f59313a9ceb
```

The final-position RED test-only overlay and final GREEN test/production
overlay contain no macOS xattr entries. Their SHA256 values are:

```text
RED   0e0d04c14c5171a9df4b09b4f1974c9093fbea6e9de97214551dcaa2c28e963c
GREEN 56a8c31cce2633576ac3fa1efda1334c1e21d3a550328e485fac8a0bbb462260
```

The final local/remote matching file SHA256 values are:

```text
SIAB/tests/test_projected_pi.py
  84f56c3c7815375fe9233f7c1db5e4ed723515ace4e19351e246e509672d977f
SIAB/opt_orb_pytorch_dpsi/IO/func_C.py
  9df3975c4a45bdcc096b597a113bb148d0d6e163f7b1b60d93aaedb5de6fdb15
```

With the source archive plus the RED overlay, the exact targeted command was:

```bash
cd /work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task4-spec-fix.OYVRA0/code/SIAB/tests
DOJO_PATH=/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task4-spec-fix.OYVRA0/code/Dojo-NC-SR \
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_projected_pi.ProjectedPiWriterRoutingTest.test_rpa_sensitive_writer_reuses_projected_pi_schema
```

It reported `Ran 1 test in 0.016s`, `FAILED (errors=1)`, with the expected
`ValueError: invalid mode 'pi_rpa_sensitive_joint'`. After applying the GREEN
overlay, the same command reported `Ran 1 test in 0.004s`, `OK`.

The required three-module, four-module, and discovery commands used the same
explicit `DOJO_PATH`, `PYTHONPATH`, and Python executable:

```bash
# Task 3 modules
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_projected_pi_optimization test_loss_and_freeze test_main_sternheimer

# Task 4 modules
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_projected_pi test_projected_pi_optimization \
  test_loss_and_freeze test_main_sternheimer

# Full SIAB
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest discover -v
```

They exited zero and reported, respectively, `Ran 101 tests in 1.494s`,
`Ran 127 tests in 1.103s`, and `Ran 356 tests in 25.672s`, all with `OK`.
This remains a code/output-routing GREEN only; it does not provide a promoted
basis, RPA binding energy, BSSE improvement, or any physical validation.

### Task 4 code-quality review fixes (2026-08-04)

The Task 4 code-quality review identified three boundary defects:

1. The hand-written fourth-order family aggregation had a zero forward value
   but produced nonfinite coefficient gradients when both H and H2 losses were
   exactly zero.
2. The direct projected-Pi API converted `sensitivity_alpha` with `float()`
   before validating its type, so `True`, `False`, and `"0.25"` were accepted
   even though the INPUT validator rejected them.
3. Projected-Pi mode classification was duplicated in `optimization_loss.py`,
   `main.py`, `opt_orbital_converge.py`, and `IO/func_C.py`.

The new real dual-family regression uses identity overlap and a complete
three-column candidate. Both underlying family losses and the aggregate loss
are exactly zero; after `backward()`, every candidate coefficient gradient must
be finite and exactly zero. The sensitive aggregate now uses
`torch.linalg.vector_norm(family_losses, ord=4)`. Existing nonzero value and
finite-difference gradient tests remain unchanged, and the legacy
`pi_dpsi_joint` sum path is untouched.

The direct API now rejects booleans before conversion and requires a
`numbers.Real` value that is finite and lies in `[0, 1]`. Separate evaluator
and optimization-adapter tests cover `True`, `False`, and `"0.25"`; legal
numeric endpoints and `0.25` retain their previous values.

The new dependency-free `loss_modes.py` is the sole owner of `LOSS_MODES` and
`PROJECTED_PI_MODES`. The four consumers import those same objects. A routing
test locks their object identity and exact values, while the real writer test
continues to protect the existing projected-Pi component and diagnostic
schema. No Task 5 acceptance or output fields were added.

#### Code-quality review remote provenance

The remote tree was assembled from commit
`c9c5c67a0a9c953216e5f036962ad7293210ceb6` by `git archive` plus exact
overlays, not by a remote Git checkout. Its absolute code directory is:

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task4-quality.gYzayf/code
```

The source archive contains `SIAB` and the tracked H TZDP coefficient fixture.
The source, test-only RED, and test/production GREEN archive SHA256 values are:

```text
SOURCE ee8e071204d05b5085263d048c35af05a71afe60179a771854c72bf6b383d13e
RED    35a98135d962356d1cd5d16f5ca2eba57648d065b97b541a95c3551b7fb94224
GREEN  5b3d4420386f2f201a2b2911d9b87069652cf94405c9859fa512293d33f5204c
```

The exact RED and GREEN overlay tree objects are, respectively,
`3759288aad7ee463d78c62af53bdb4a08f07a4cb` and
`a831d59d7f35b0af1bd9bcc0700a9d82896bfa1e`. All archives were checked to
contain no AppleDouble entries. Local and remote SHA256 values for every
overlaid production file matched before testing.

From the absolute `SIAB/tests` directory above, the targeted command used the
explicit environment:

```bash
DOJO_PATH=.../code/Dojo-NC-SR \
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_projected_pi.ProjectedPiTest.test_sensitivity_alpha_rejects_bool_and_string_values \
  test_projected_pi_optimization.ProjectedPiOptimizationTest.test_rpa_sensitive_zero_family_losses_have_zero_finite_gradient \
  test_projected_pi_optimization.ProjectedPiOptimizationTest.test_rpa_sensitive_adapter_rejects_bool_and_string_alpha \
  test_main_sternheimer.MainRoutingTest.test_loss_modes_have_one_shared_source
```

With the RED overlay it reported `Ran 4 tests in 0.086s` and
`FAILED (failures=8)`: six rejected-type subtests did not raise, the zero-loss
gradient was nonfinite, and the shared module was absent. With the GREEN
overlay, the same command reported `Ran 4 tests in 0.047s`, `OK`.

The required regression commands used the same explicit `DOJO_PATH`,
`PYTHONPATH`, Python executable, and working directory:

```bash
# Task 3 modules
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_projected_pi_optimization test_loss_and_freeze test_main_sternheimer

# Task 4 modules
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_projected_pi test_projected_pi_optimization \
  test_loss_and_freeze test_main_sternheimer

# Full SIAB
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest discover -v
```

They exited zero and reported `Ran 104 tests in 1.279s`,
`Ran 131 tests in 1.070s`, and `Ran 360 tests in 36.116s`, all with `OK`.
This is code GREEN only: no optimization, checkpoint promotion, orbital-basis
improvement, RPA binding energy, BSSE improvement, or physical validation was
performed.

### Task 4 controller-independent final regression (2026-08-04)

After both independent reviews approved the implementation, the controller
created a fresh tree directly from final commit
`77d42e4d18a0dde4afdd9693dce30695f5a5a33b`. This run did not reuse an
implementer overlay or a remote Git checkout. The local `git archive` contained
the tracked `SIAB` and `Dojo-NC-SR` trees, had no AppleDouble entries, and had
SHA256:

```text
db0e14f9452d418812797c9ff2edee6902f8a2bfa4cc41b0576a90529431a02c
```

The uploaded archive had the same hash. Its absolute extracted tree was:

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task4-controller-final-77d42e4d
```

From that tree's `SIAB/tests` directory, the controller set
`DOJO_PATH` to the archived `Dojo-NC-SR`, used the same explicit
`PYTHONPATH` and Python executable as the implementation runs, and executed:

```bash
# Task 3 modules
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_projected_pi_optimization test_loss_and_freeze test_main_sternheimer

# Task 4 modules
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_projected_pi test_projected_pi_optimization \
  test_loss_and_freeze test_main_sternheimer

# Full SIAB
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest discover -v
```

The fresh run reported `Ran 104 tests in 0.919s`,
`Ran 131 tests in 1.037s`, and `Ran 360 tests in 24.323s`; all three exited
zero with `OK`. Their measured process wall times were 5.19, 4.71, and 28.18
seconds. This independently closes Task 4 at the code-regression gate only.
There is still no optimized basis, checkpoint promotion, RPA binding energy,
BSSE result, or physical validation.

### Task 5 RED: atomic and family checkpoint gates (2026-08-04)

Task 5 first changes only `test_loss_and_freeze.py` and
`test_main_sternheimer.py`. The checkpoint contract is the pair of strict
inequalities

```text
family_improved := L_family(C) < L_family(C0)
atom_improved   := L_H(C)      < L_H(C0)
```

in addition to the existing DFT, dpsi, condition, locality, and low-frequency
gates. The initial point has exactly the baseline losses, so both strict
comparisons are false without a step-index exception. A differentiable fake
provides one candidate whose fourth-order H/H2 family loss improves while H
worsens, followed by a candidate for which both losses improve. The tests also
require bitwise restoration of the fixed column and prove that
`pi_dpsi_joint` still accepts and selects under its old contract.

The new-mode output contract keeps compact scalar H/H2 total, base,
sensitivity, blend, gate, and condition columns in `Spillage.dat`.
`ORBITAL_RESULTS.txt` has a separate scalar-only schema. GreenX frequencies
and weights, per-frequency losses, trace-log differences, and dielectric
eigenvalue minima are long diagnostics and belong only in
`PROJECTED_PI_METADATA.json`. The old `pi_dpsi_joint` header, acceptance,
selection, and writer block are byte-locked by regression tests.

Because `df_dcu` has no GitHub DNS, the remote RED did not use or claim a Git
checkout. The local source archive was made from exact commit
`1991b1583e52726a3b6be6b34cc2584417d9045d`; its local and remote SHA256 was

```text
52fd6e85733daa76429b0d04b460c7b6e0284b580b6cb810e752a1ba137b690a
```

The final test-only ustar overlay had SHA256

```text
d28a1bad0baac1391d6bdb4a394b76410f1b66911bcaa292594d44bca85b4cdd
```

and produced matching remote test-file hashes

```text
test_loss_and_freeze.py
  97b7616f2cb16a77f1ec2cc698fd02368976ae426be2a8d501c114652331fea9
test_main_sternheimer.py
  b4b20a1ca0c6a9d6698b35fb1e535455a72b6b88dd77c1c8d239ec85faf2f974
```

The absolute extracted tree was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task5-red-1991b158-0e193bb1/tree
```

From that tree's `SIAB/tests`, with `DOJO_PATH` pointing to its archived
`Dojo-NC-SR`, the required RED command was

```bash
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_loss_and_freeze test_main_sternheimer
```

It ran 98 test methods: 93 passed, two methods produced two failure records,
and three methods produced three error records. The exact summary was
`FAILED (failures=2, errors=3)`. The failures were the absent dual gate/new
`Spillage.dat` header and the absent fatal no-accepted behavior. The errors
were the absent baseline-aware diagnostic signature, main scalar-schema
routing, and independent writer schema. No failing record came from a
subtest. The new old-mode acceptance/selection regression passed. This is
Task 5 RED only: no production implementation, optimization, historical alpha
ranking, candidate basis, SOS/CP calculation, or physical validation exists.

### Task 5 GREEN: atomic and family checkpoint gates (2026-08-04)

The new `pi_rpa_sensitive_joint` checkpoint contract is now

```text
accepted := original_gates
            and L_family(C) < L_family(C0)
            and L_H(C)      < L_H(C0)
```

where `original_gates` denotes the existing DFT, dpsi, condition, locality,
and low-frequency gates. Setup evaluates the baseline once and stores detached
clones of the aggregate H/H2 family loss and the H loss. Since the initial
point equals those baselines, both strict inequalities are false naturally;
there is no step-index exception. A candidate that improves the fourth-power
family aggregate while worsening H is rejected, whereas a later candidate
that strictly improves both is selected. Fixed columns remain bitwise equal
after candidate evaluation. The `pi_dpsi_joint` acceptance, selection,
`Spillage.dat` columns, and `ORBITAL_RESULTS.txt` block remain on their old
path.

When no candidate is accepted, the violation ordering and fatal error include
the finite normalized quantities

```text
v_family = max(0, (L_family(C) - L_family(C0))
                  / max(L_family(C0), epsilon))
v_atom   = max(0, (L_H(C) - L_H(C0)) / max(L_H(C0), epsilon))
```

so a zero or near-zero baseline does not create `NaN` or infinity. The new
`Spillage.dat` row is compact: H/H2 total, base, sensitivity, and blend losses,
the family/atom gate decisions, and the maximum overlap condition are scalar
columns only. `ORBITAL_RESULTS.txt` uses an independent new-mode scalar schema
containing alpha, family power, initial/final family and H losses, H/H2 scalar
components, maximum condition, and rank tolerance. Long arrays are serialized
only to `PROJECTED_PI_METADATA.json`: GreenX `frequency_ha` and
`frequency_weight`; per-frequency total, base, and sensitivity losses;
trace-log differences; and minimum reference/candidate dielectric eigenvalues.
JSON output is finite and uses the real metadata serialization path.

The GREEN tree was assembled without a remote checkout from the same source
archive used by RED, followed by the exact test and production overlays. The
production overlay SHA256 was

```text
4210906e15217efc0728a2a195a8ef22d4daff72d4de48afbe918c32385c6d00
```

and its three production-file SHA256 values were

```text
opt_orbital_converge.py
  4a9af6a7f4e996c5d3abb31bc4ba0f3c8966dba1f08fdcccad4a1eba6c0dba32
IO/func_C.py
  c8e9faac25b3e33f6e4beff6777a0289788e860ade0645d2fd0ee2a86575ced0
main.py
  b65ebfe58e4749b91989fee640319ca3f02baac4b26684e8f5c3ec61afc681b7
```

The Task 4 writer test still encoded the preliminary shared-schema behavior.
After that test was aligned with the final Task 5 independent scalar schema,
the one-file overlay SHA256 was

```text
5cd34e00b732a5fce02fa28b1aa22c2039bceec85b01973f4d34a013104584b3
```

with `test_projected_pi.py` SHA256
`75e79effade34786130bd7bc1fd1d110a7791744c0eaf696b53b9e1d5921c719`.
The absolute final tree was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task5-green-final-1991b158-5cd34e00/tree
```

From its `SIAB/tests` directory, using the RED environment and archived
`DOJO_PATH`, the final reproducibility rerun gave 98 test methods in 0.795 s
for the targeted two modules, 135 methods in 1.049 s for the Task 4 four
modules, and 364 methods in 26.738 s for full discovery. Every final run
reported `OK` with zero failures and zero errors; there were no failing
subtests. The three log SHA256 values were, respectively,
`5b7187ae2c86627c299ca14cdb98c821a0a299d1f081312304d084c41f0ecf14`,
`c098fd7b9457ea3ebfd28244f7840ea96ab2d88b5fcb5aa081330e4892117bc7`,
and `53ed5896d0421dd1b175341095cfafed4bfc396f1add8916c9c6e734c128911b`.
The earlier broad run exposed exactly the obsolete shared-schema expectation
and was not treated as GREEN.

This completes only the Task 5 code checkpoint gate and diagnostic schema. It
does not produce an optimized basis, rank historical alpha values, run Task 6,
or establish any RPA/SOS/CP or physical-validation result.

### Task 5 code-quality review fixes: RED (2026-08-04)

The review fixes started from exact commit
`cfb8ef4d6f61f65cda6b831007041e2d0641c6f9`. Tests were changed before
production code. The targeted cases require one `torch.no_grad()` baseline
evaluation for repeated identically configured sensitive stages, explicit
strict-gate state in the fatal diagnostic, failed-gate-aware violation
ordering, exact-zero baseline preflight, and immediate validation of every
new-mode evaluator result. They also prove that a legacy objective wired to
`pi_rpa_sensitive_joint` fails at the baseline boundary rather than later in
main or a writer.

The source archive contained tracked `SIAB` and `Dojo-NC-SR` from that exact
commit. Its local and remote SHA256 was

```text
e1a532c406f6df1af624a774e6646184540db832c6124051344ca2efaa3935e5
```

The exact test-only ustar overlay SHA256 was

```text
8a19928174a9d307846db7ba1a35e1d26ae109a38735834802d22e12328dafa9
```

with test-file SHA256 values

```text
test_loss_and_freeze.py
  7b5f6aa8f11f69d86d6c6062628e4fedf5426b67b0aaef238ec5d3afdbc98541
test_main_sternheimer.py
  9586f826f6e369d9c0572d99adeb5588d3f79c702fce8c6e16915735726dc897
```

Because `df_dcu` has no GitHub DNS, this was not a remote checkout. The
archive plus exact overlay was extracted at

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task5-quality-red-cfb8ef4d-8a199281/extract/tree
```

The run used

```text
Python:     /work1/ghj/runtime/siab-py310-cpu-20260720/bin/python
PYTHONPATH: /work1/ghj/runtime/siab-projected-pi-mpl-20260801
DOJO_PATH:  <absolute RED tree>/Dojo-NC-SR
```

The 13 targeted test methods produced feature-specific RED. The ordinary
runner reported `FAILED (failures=16, errors=12)`: four methods had direct
failures, four methods had direct errors, and 20 subtests contributed 12
failure records and eight error records. The failures exposed duplicate
gradient-enabled baselines, missing equality diagnostics and zero-baseline
preflight, late or nonspecific malformed-result failures, and the old
diagnostic helper signature. The ordinary RED log SHA256 was
`975d4cd7d305bebbc196a6b23ff605aaf5b9e10464957e4a69010013fb26b1af`;
the independent method/subtest count log SHA256 was
`501855454864ccefbab0266730517799ebd33a0d8cd0fc4d2add6b59e097b4ea`.

### Task 5 code-quality review fixes: GREEN (2026-08-04)

For a current loss `L` and positive baseline `L0`, the fatal diagnostic now
records the signed normalized delta

```text
delta = (L - L0) / max(L0, epsilon)
```

for both aggregate family and atomic H losses, together with current,
baseline, and the exact strict-comparison booleans. Acceptance remains
`L < L0`; a negative delta reports improvement but does not replace or relax
that comparison. A failed response gate has penalty
`1 + max(0, delta)`, while a passed gate has zero penalty. The violation key
first minimizes the total number of failed gates, then maximum and summed
penalties, then signed-delta tie-breaks. Consequently, “family improves and H
is equal” is preferred over the unchanged initial point, although neither is
accepted. Equality is therefore visible as a failed gate instead of a zero
violation. The projected-mode fatal condition label is
`max_projected_pi_condition`.

Since the losses are nonnegative, an exactly zero H or family baseline cannot
be strictly improved. It now fails immediately with `strict improvement
impossible`, before optimizer construction or stepping. A positive near-zero
baseline remains legal and its signed deltas remain finite through the same
`epsilon` denominator semantics.

An identically configured `(C_initial, evaluator)` sensitive baseline is
evaluated once under `torch.no_grad()`. The cache retains only detached scalar
baselines and serializable initial scalar diagnostics; the complete result and
autograd graph are discarded. Old modes keep their previous baseline path.
After baseline and candidate evaluation, the unified validator checks the
new-mode result type, configured alpha, exact fourth-order family power,
exactly `H`/`H2`, family alpha agreement, finite total/base/sensitivity/blend
data, aligned finite frequencies and positive weights, per-frequency loss and
trace-log diagnostics, dielectric eigenvalue minima, rank, and conditions.
`pi_dpsi_joint` retains its legacy result contract. Diagnostic schema choice
is now driven by the validated configured mode, never inferred from a result
attribute.

The GREEN tree reused the exact RED source and test overlays, then applied the
minimal production overlay. Its SHA256 was

```text
5634bbedb34316554541e62755813223cad10d55f50089b6eded3ce68d643f5d
```

and `opt_orbital_converge.py` had SHA256
`31c1be4d1ae585b21a1f612a3ddb97c2c694f5f1b78ec0142c7f10a1da88feaa`.
The absolute, non-checkout tree was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task5-quality-green-cfb8ef4d-5634bbed/extract/tree
```

It used the same explicit Python, `PYTHONPATH`, and tree-local `DOJO_PATH` as
RED. Results, including independently counted successful subtests, were:

```text
targeted review tests: 13 methods,  20 subtests, 0 failures, 0 errors
Task 5 two modules:    109 methods,  82 subtests, 0 failures, 0 errors
Task 4 four modules:   146 methods, 118 subtests, 0 failures, 0 errors
full unittest discover: 375 methods, 252 subtests, 0 failures, 0 errors
```

The ordinary log SHA256 values were, respectively,
`1b508982be2929ef4ddaa703ea1eceb2e4e32e0920a5c4d3945849bd9e2805cf`,
`fd70d519c8815a15cce605c25249deff34510b8e80639072172b433c68707aa4`,
`e60d70ca7ea87d25f038015e1ff908e581f15d80d861caf91cf235d18b9af7cb`,
and `56d088e900d8e16041fa0539dac2a2f55d5c632af7d336a7983866f161640a11`.
The matching count-log SHA256 values were
`5bfaf17fb7d09990b16111f9f86495f3f4a864cd5650d8469df9af56ccca7ea3`,
`c7b1af4366c7bb27089bc8a60f7951ddc5ac926fd93f8c4ba87fe8291161a0da`,
`d068409e1bc03fa173699c994e256f4cbb7a8f22dfe73874121e4e3801f55461`,
and `bc1de07af2eb06c29c04d8c78e3d5f0cb095680d7dc9c2f332a14310ab097c1c`.

This is code GREEN only. It is not an optimization, historical alpha ranking,
Task 6 implementation, candidate-basis result, or physical validation.

### Task 5 controller-independent final regression (2026-08-04)

After the Task 5 specification and code-quality reviews approved the final
implementation, the controller built a fresh archive directly from commit
`014a19f4eca823bd12fa5a6c61d97db1d63ae803`. It did not reuse either
implementation GREEN overlay and was not a remote Git checkout. The archive
contained tracked `SIAB` and `Dojo-NC-SR`, had no AppleDouble entries, and
had matching local/remote SHA256:

```text
92cc02558a781a9a7b2ccf2156aac227715f61b45bc2905d30014c4db8c48dc6
```

The absolute independent tree was:

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task5-controller-final-014a19f4
```

From its `SIAB/tests` directory, with tree-local `DOJO_PATH` and the
recorded Python/PYTHONPATH, the controller reran:

```text
Task 5 two modules:   Ran 109 tests in 1.013s, OK; process wall 5.80 s
Task 4 four modules:  Ran 146 tests in 1.204s, OK; process wall 5.06 s
Full SIAB discover:   Ran 375 tests in 25.125s, OK; process wall 29.35 s
```

All commands exited zero. This independent reconstruction closes Task 5 only
at the code-regression gate. It does not freeze an alpha, run Task 6, optimize
an orbital basis, or provide an RPA/SOS/CP physical result.

### Task 6 rejected RED: invalid synthetic cell (2026-08-05)

The first Task 6 RED attempt is retained only as rejected controller evidence.
It started from exact commit
`4e424434a36837d74548a951954675d2d50afb14`. The local source archive and
remote copy had SHA256

```text
75d716f3e1ef304923bd4a05c5d52c64ecb80a7a12a2d704a30baad165fa6b72
```

The first test overlay had SHA256
`d00c4e560a7ce8d04b6ac65683c0f549ae3bf3d0d1f3eac33e641a353c07e4a8`,
and its test file had SHA256
`323d30066f1e988981856555e40217377d0fd6853897eddc5a7793341d6646cd`.
The non-checkout staged tree was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-red-4e424434-d00c4e56/extract/tree
```

Both test methods stopped on the intentionally absent analyzer, but the
independent fixture check then failed before evaluator construction with
`ValueError: provenance cell_bohr must contain nine row-major lattice
components`. The fixture had only eight components. Therefore the two test
failures are not accepted as Task 6 RED. The rejected `red_test.log` and
`fixture_validation.log` SHA256 values are
`a4bef08ec44eb83dd9ed9eb8522652816e270a3cc8ce5cde35e7080b3bb90a5b`
and `cab0c543e7fa6ad0c152bf72672a390a54a8f136fa2a3cec21fba00bea5faecb`.

### Task 6 corrected RED: five-basis ranking CLI (2026-08-05)

The fixture was corrected to a nine-component row-major cell before any
production analyzer existed. A fresh test overlay had SHA256

```text
31bc0750b48a26406616755904defb797f9848d4d541bd33dfe7c39e2c8e45e0
```

and `test_rpa_sensitive_ranking.py` had SHA256
`6483c900be1692beefb8a5c667f2f465ec667ee6dcc06de298152ce793dbbdc1`.
The same verified source archive plus this exact overlay was staged at

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-red-corrected-4e424434-31bc0750/extract/tree
```

This is an archive reconstruction, not a remote Git checkout. The environment
was staged only after local and remote SHA256 checks matched for both the
source archive and corrected test overlay. The environment was

```text
host:       login08
Python:     /work1/ghj/runtime/siab-py310-cpu-20260720/bin/python
Python:     3.10.13
PyTorch:    2.1.0+cpu
PYTHONPATH: /work1/ghj/runtime/siab-projected-pi-mpl-20260801
DOJO_PATH:  <absolute corrected RED tree>/Dojo-NC-SR
```

Before running RED, an independent preflight used the strict response-v1,
source-v1, zero-order-audit, and coefficient readers plus
`NormalizedPhysicalFamilyProjectedPiOptimization`. It consumed both
625-primitive H/H2 pairs, both passing audits, and all five exact Nu layouts.
Every frozen alpha, `0.0, 0.1, 0.25, 0.5, 1.0`, passed all four ordering gates
in the success fixture; the separate stop fixture admitted none. The preflight
log SHA256 was
`cc3706e93a652ee8817c3ef6f59cbbaa7f0d8503676033595135e59bf58b60f8`.
The environment record SHA256 was
`c717cf8d41535c4f24bf02e52e8fa5e72e1ee4efbacd660a82fc8feb2f9762e7`.

From that tree's `SIAB/tests`, the RED command was

```bash
PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801 \
DOJO_PATH=<absolute corrected RED tree>/Dojo-NC-SR \
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python \
  -m unittest -v test_rpa_sensitive_ranking
```

It ran two methods in 0.001 s and reported exactly two failures, zero errors,
and exit code 1. Both failures were the explicit assertion that
`analyze_rpa_sensitive_ranking.py` was absent. The corrected RED log SHA256
was `bb74e8892007b7223a0c6f140d85cbd776dd9498216684aaca4d12145e67aa1c`.
This is the accepted Task 6 RED.

### Task 6 GREEN: synthetic ranking code gate (2026-08-05)

The analyzer freezes the five `BASIS_NU` tuples and five `ALPHAS`, reads only
H/H2 response, source, audit, and exactly five coefficient files, and invokes
the existing fourth-order H/H2 RPA-sensitive evaluator for every alpha. Each
alpha records every basis total, every H/H2 base, sensitivity, blended, and
frequency loss, overlap conditions, trace-log differences, dielectric
minima, and the four strict ordering gates. The selected value is
`max(admissible_alpha)`. JSON also records all input hashes, validated
zero-order audits, and the exact false flags
`uses_sos_energy_as_numeric_input`, `uses_ghost_family`, and
`new_candidate_was_evaluated`. JSON, Markdown, PNG, and PDF are all published
through same-directory temporary files and atomic `os.replace`. The plot has
separate H and H2 panels and compares base and sensitivity frequency losses
for all five labels.

The first production overlay had SHA256
`08baaacfc1a4dec811102cbe4eb77754e627691d2c45503918866d260cf63648`.
Its staged analyzer had SHA256
`8bdc99f5b698e9ee2975169a7e8386c2af0ce1720c6120caa79be1119e92b999`.
That run reached all implementations, but 31 tests reported one failure
because the Markdown header used lower-case `base loss` and `sensitivity
loss`. The other 30 tests passed. Its log SHA256 was
`f1b12da7a5d56dd28abfb6040aa4bb64c5ce8b94495ee5025becf463db092ecf`.
This run is retained as the expected TDD correction step and is not counted
as GREEN.

After changing only the visible header capitalization, the final production
overlay and analyzer SHA256 values were

```text
production overlay
  e4482b320dc20d8d908c18c3c6658490a6bb9093d9f155dfcefb2b9bf930f456
analyze_rpa_sensitive_ranking.py
  ab703e23d8d74b43908c74d971a29161bebb010a8c32e0012de153113e072797
```

Local and remote SHA256 checks matched for the source archive, corrected test
overlay, and final production overlay.

The final non-checkout reconstruction was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-green-final-4e424434-31bc0750-e4482b32/extract/tree
```

It used the corrected RED environment. The required command was

```bash
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest -v \
  test_rpa_sensitive_ranking test_projected_pi_analysis test_projected_pi
```

It ran 31 tests in 58.136 s and reported `OK`; measured process wall time was
61.43 s. The environment and required-test log SHA256 values were
`ef45404271eab0112f18d4128ad1b3f8962155b37b0debcb4dd42bc42cc2b3ca`
and `dc232975f9f236d828408dc6f70dbeb2473045388285e55235313e7662569242`.
A full `python -m unittest discover -v` then ran 377 tests in 71.810 s and
reported `OK`; measured process wall time was 79.87 s and the log SHA256 was
`f0cd28d0ed654f7283a528828565a6e04f4a597ee9bc39d0f202eac14c6981ca`.

The success fixture deliberately has multiple admissible alphas and selects
the largest, `1.0`. That value is synthetic test data only. This closes the
Task 6 code gate; it is not the real historical five-basis gate, does not
freeze a production alpha, does not optimize or evaluate a new candidate,
does not read SOS/CP energies, and does not provide physical validation.
Task 7 has not started.

### Task 6 spec-review fix: reject duplicate coefficient options (2026-08-05)

Commit `c1236f29fa819e76c0d63919811e00b1eb3dac72` is preserved as the first
Task 6 implementation. Spec review found that ordinary argparse `store`
actions silently accepted a repeated coefficient option and retained its last
value, so the CLI did not yet enforce exactly one occurrence of each frozen
coefficient label. The same review also checked the authoritative
`corrected_red_environment.txt` and `green_environment.txt` records and found
that the previously recorded patch-level Python version was wrong; the exact
version is 3.10.13.

For the follow-up RED, a local Git archive of `c1236f29` and a test-only patch
were staged with matching local/remote SHA256 values:

```text
source archive
  8c06ac5ec2377eca7b8a99ccd9352f85e07354e5e5dbdc3e589d90b7dd8c0e50
test patch
  a4bc2851ca789a071c38c144e4f16aea483a27f3741b8aa7b0c2148744bd6001
test_rpa_sensitive_ranking.py
  cc0b9f66c840d1cc17217b4831040549a508a3705c52bea6467f2a02e68dc9cd
```

The immutable archive-plus-patch tree was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-review-red-c1236f29-a4bc2851/extract/tree
```

This is not a remote checkout. On `login08`, with Python 3.10.13 at
`/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python`, PyTorch 2.1.0+cpu,
`PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801`, and tree-local
`DOJO_PATH`, the command

```bash
python -m unittest -v test_rpa_sensitive_ranking
```

ran four tests in 74.655 s. Exactly one test failed because the unchanged
analyzer returned 0 after `--two-d` was supplied twice; the missing-option,
Galerkin-stop, and largest-alpha tests passed. The environment and RED log
SHA256 values were
`870b29729dad27c78ae47100f4e2deefdf336a4166b1d60b5b33168a90b11a68`
and `de431be83d073f0d0cf0a69f0d0b53dcef1e011bd333def3407ef2586caea58f`.

The minimal fix adds a coefficient-specific `StoreCoefficientOnce` argparse
action. The first occurrence is stored; a second occurrence exits 2 with
`coefficient option --two-d may be specified only once`. All five distinct
coefficient options remain independently required, and the missing-option
test confirms omission still exits 2 through argparse's required-option
check. The production patch and fixed analyzer SHA256 values are

```text
production patch
  6a56cba99a2cd9e364d8b28a1c1d4f04afb4e3bdbc3e1de63236aac0e544d6d2
analyze_rpa_sensitive_ranking.py
  8fe60239a2e63a7f55112edadeb9400e9487410cd8a1e44036f50acc7eca35d5
```

The source archive, test patch, and production patch again matched locally
and remotely. The final non-checkout GREEN tree was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-review-green-c1236f29-a4bc2851-6a56cba9/extract/tree
```

The required command

```bash
python -m unittest -v \
  test_rpa_sensitive_ranking test_projected_pi_analysis test_projected_pi
```

ran 33 tests in 68.639 s and reported `OK`; measured process wall time was
73.92 s. The environment and required-test log SHA256 values were
`7c41580e91a00c906c63543326591f918fb6317be35372ae01996e0dcd0d44ee`
and `cb0cf98c2ef7ae7e89e3461c2c681d14231570f565beaf932fe194ad7044284e`.
Full `python -m unittest discover -v` ran 379 tests in 82.103 s and reported
`OK`; measured process wall time was 87.44 s and the log SHA256 was
`4cc15d400924a0d593346ca545a967fcc9020207da40d80f71ff80a8c94a45f5`.

This follow-up closes only the Task 6 CLI occurrence contract and corrects
its recorded environment. It does not run the real historical gate, freeze a
production alpha, optimize or evaluate a new candidate, read SOS/CP energy as
numeric input, or provide physical validation. Task 7 has not started.

### Task 6 code-quality hardening: family identity and atomic set (2026-08-05)

Commits `c1236f29fa819e76c0d63919811e00b1eb3dac72` and
`20341ac12ce75c646d9be4a57fc4b30fd5ae2f13` remain the first implementation
and specification-review fix. The code-quality review then required the H and
H2 families to be independently identified, every recorded input hash to
describe the bytes consumed, and the four required artifacts to become
visible as one directory set.

The implementation rejects response, source, or audit pairs when their
resolved path/file identity is the same or when separate files have identical
SHA256 content. Audit aliases are rejected deliberately: the CLI requires
separate H/H2 audits and the strict audit reader binds each document to its
`case`; accepting the same identity or bytes would weaken that family binding
before validation, and there is no existing identity-binding exception that
justifies it.

Each of the eleven input paths is resolved and hashed before any strict reader
or evaluator runs. Readers consume those resolved identities. Immediately
before publication, every original CLI path is resolved and hashed again;
retargeted/replaced identities and changed content fail with the affected
input label and publish nothing. Inputs are not copied.

JSON, Markdown, PNG, and PDF are written into one unique sibling directory
named with the `.<output>.staging-` prefix. The figure is closed in a `finally`
block. Any generation or verification failure recursively removes the staging
directory, leaving the final path absent and permitting a retry. After the
exact four-name set and all input snapshots pass, one directory `rename`
publishes the complete absent destination; an existing destination is still
refused. The PDF `CreationDate` and `ModDate` are fixed to 2000-01-01 UTC, so
identical path-independent runs produce byte-identical artifacts.

#### Rejected quality RED/GREEN fixture attempt

The first quality test overlay was SHA256
`64068f357193d0cba3cae2a5efa50c3e480a80af1c15074d4858d8ee61c0fc32`
and produced test file SHA256
`ce3781ced58730085722312bdd4df92d49c58ef5d32d5ac63192faea8de218b6`.
It was applied to the exact `20341ac1` archive SHA256
`300218bfc7fa985ba8d2c872ac37bdb823177dde0813259a655c81ac04b07064`
at

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-quality-red-20341ac1-64068f35/tree
```

It ran 12 tests in 174.135 s, with 11 assertion failures and no errors
(`real 177.64`, `user 1752.47`, `sys 225.77`). The log SHA256 was
`c1bfab48bd2997b42341db317fbd0b4c4f18623bf60850302809d314bd144989`.
Although those failures exposed the reviewed defects, the nominal H and H2
source fixtures were byte-identical. That violates the new family-separation
contract, so this RED is rejected rather than counted.

The corresponding attempted GREEN tree was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-quality-green-20341ac1-93e1c682/tree
```

using production overlay SHA256
`93e1c682a7080aa2c2717cd6104f940001e25221fe4317b2f1117f9788ade979`.
It correctly exposed the invalid baseline source alias and therefore reported
7 failures and 1 error among 12 tests in 50.969 s (`real 54.93`, `user
48.86`, `sys 75.79`). Its log SHA256 was
`80f907a32ea85cf0c2501f79d5b0d7ecd060185ebc387856371049ec96c42b1a`.
This attempt is also rejected and is not GREEN.

#### Accepted corrected quality RED

The source fixture was corrected by applying the family scale consistently to
both source and response overlaps. H/H2 source bytes are now distinct while
the strict pairing relation is preserved. The fresh test-only overlay and
resulting test file SHA256 values were

```text
test overlay
  103d0969b2ce53d0914cce7ddfa3c9a09d0489e465f1b59e03cc13d461011211
test_rpa_sensitive_ranking.py
  359afad9e7749a307b3d45a139682cc816d0e6fcb8d79e56cd938a79d64988fe
```

The exact `20341ac1` archive plus that overlay was reconstructed, not checked
out, at

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-quality-corrected-red-20341ac1-103d0969/tree
```

On `login08`, the environment was Python 3.10.13 at
`/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python`, PyTorch 2.1.0+cpu,
Matplotlib 3.10.3, and
`PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801`. The environment
record SHA256 was
`6d69bfb69fe386d92c9db73f49a4acd41f4172e976131e013616112ee1fefcb2`.
The staged old analyzer retained SHA256
`8fe60239a2e63a7f55112edadeb9400e9487410cd8a1e44036f50acc7eca35d5`.

`python -m unittest -v test_rpa_sensitive_ranking` ran 12 tests in 172.307 s
and reported exactly 11 targeted assertion failures, zero errors, and status 1
(`real 176.00`, `user 1762.51`, `sys 244.74`). The real-evaluator success and
stop tests both passed, independently proving that the strict readers,
pairing, evaluator, and corrected fixtures were consumable. The RED log
SHA256 was
`8ead899574df64f17f9237883bb3a87abe4eff31c40f8b69e0402428994d4506`.
This is the accepted quality RED.

#### Accepted corrected quality GREEN

The accepted production overlay and analyzer SHA256 values were

```text
production overlay
  93e1c682a7080aa2c2717cd6104f940001e25221fe4317b2f1117f9788ade979
analyze_rpa_sensitive_ranking.py
  f8137755b9e376e2484a49c4dbde01a0a1abfb188e2c81a5f98513919fecb663
```

The archive and both overlays matched locally and remotely. The immutable
archive-plus-overlays tree was

```text
/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_tdd_20260804/code-task6-quality-corrected-green-20341ac1-93e1c682/tree
```

Its environment record SHA256 was
`66137d971fc9d76e933a479bdb13c0c19d075b7908089763efbdea3cc2c3175d`.
It used the same `login08` Python/PyTorch/Matplotlib/PYTHONPATH environment as
corrected RED, with `PYTHONDONTWRITEBYTECODE=1` and
`MPLCONFIGDIR=<absolute GREEN evidence directory>/mplconfig`. From the
absolute GREEN tree's `SIAB/tests`, the exact commands were

```bash
python -m unittest -v test_rpa_sensitive_ranking
python -m unittest -v \
  test_rpa_sensitive_ranking test_projected_pi_analysis test_projected_pi
python -m unittest discover -v
```

All three exited zero:

```text
Task 6 module:
  Ran 12 tests in 142.838s, OK
  real 146.33, user 1768.24, sys 199.24
  log 723706dceb0aff0c2ad56991d84549ba4dc8bce98ea0f81fedb03790ced46bde

Required three modules:
  Ran 41 tests in 151.687s, OK
  real 155.00, user 1781.47, sys 224.81
  log 93027e7d358099cafd43202bcb4ac6d037cf6fb9f08bb8fec3b6a766235de736

Full SIAB discover:
  Ran 387 tests in 169.165s, OK
  real 175.68, user 1814.99, sys 246.85
  log 0f39b04d4336d7716c2e1c9f9ba2b08b9fa3e2f25387ccd11d0de6a72e50e04d
```

The pre/post-test tree manifest SHA256 was unchanged at
`3f08e60c169ed4335804f2977adc5b32797b2ae7cdc25b2aded9acdef8b8b9da`.
Tests cover all five repeated coefficient options and omission, exact stable
JSON keys, same-path and equal-content family aliases for all three roles,
content mutation and symlink retargeting, late PDF failure cleanup and retry,
exact four-artifact pass/stop sets, two H/H2 axes with all base/sensitivity
labels, finite values, strict gates, maximum-alpha selection, and byte
identity of all four artifacts across delayed runs.

This remains only a synthetic Task 6 code gate. It does not run the real
historical ranking, freeze a production alpha, optimize or evaluate a new
candidate, consume SOS/CP energy numerically, or establish physical
validation. Task 7 has not started.
