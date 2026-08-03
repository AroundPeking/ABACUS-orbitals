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
