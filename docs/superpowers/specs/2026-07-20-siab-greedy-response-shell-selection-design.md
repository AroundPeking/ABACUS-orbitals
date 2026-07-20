# SIAB Greedy Response-Shell Selection Design

## 1. Goal

Build a deterministic SIAB selector that starts from the fixed H DZP core,
uses Coulomb-weighted Delta-Sternheimer first-order wavefunction residuals to
append one shared radial shell at a time, and produces the smallest nested
`s/p/d/f/g` basis sequence that can reproduce the matched Delta-ST RPA result
without large molecular basis-set superposition error.

H2 RPA energies do not enter orbital construction, shell scoring, loss weights,
or stopping of the spillage sequence. They are evaluated only after the nested
sequence is frozen. The first basis in that fixed sequence that passes all RPA
and BSSE gates is the accepted candidate.

## 2. Current Limitation

The existing `radial_residual_spectrum` implementation correctly projects out
the fixed DZP core, whitens one angular block with its primitive overlap, and
returns the optimal shared radial response modes. The current production path
nevertheless has three limitations:

1. the producer and preparation script stop at `l=2`;
2. the d-shell count is chosen from a 99% atomic within-channel capture rule,
   rather than by comparing the marginal value and AO cost of all angular
   channels;
3. a one-center atomic residual cannot detect the large H2/atomic imbalance
   that produced `6.7391 kcal/mol` of RPAc BSSE for the smooth `4s3p3d` basis.

The new design preserves the validated overlap-whitened eigensolver and
replaces only the fixed-`l`, one-shot shell decision around it.

## 3. Scope

The first implementation is H-only and supports angular momenta
`l=0,1,2,3,4`. This covers s through g and matches the angular range already
present in the explicit H auxiliary basis. The fixed core remains
`1s,2s,1p`; every appended response shell is trainable.

The selector consumes three predeclared target families:

- `atom`: isolated-H Delta-ST responses;
- `multicenter`: H3 Delta-ST responses at geometries fixed before any new H2
  RPA result is inspected;
- `fragment_ghost`: one-real-H responses in the same geometric and primitive
  environment as the multicenter target, with all other centers made ghost
  centers.

The physical first-order wavefunctions remain the primary training objects.
No density-response eigenvector or H2 binding-energy target replaces them.

Periodic solids, multiple chemical elements, orbital-cutoff optimization, and
automatic auxiliary-basis generation are outside this stage.

## 4. Target Contract

Every target file must record:

- species, atom index, `l`, `m`, primitive offset, and primitive count;
- complete magnetic multiplets `m=-l,...,+l` for every included `l`;
- frequency values, minimax weights, perturbation identifiers, and effective
  response weights;
- primitive overlap, first-order-wavefunction projection matrix, and response
  norm;
- supercell, Ecut, pseudopotential, perturbing Hartree-potential kernel,
  auxiliary-basis hash, frequency-grid hash, producer commit, and binary hash.

All target families must use the same H radial primitive ordering and the same
frequency grid. The reader rejects missing magnetic channels, inconsistent
primitive counts, materially different primitive overlaps, non-finite weights,
or provenance mismatches that would make the spectra incomparable.

The producer angular cutoff is independent of the input orbital `Lmax`. For
this campaign it is fixed at `sternheimer_siab_lmax=4`.

The response perturbations use the explicit fixed
`H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs` auxiliary basis with SHA256
`d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f`.
Set `exx_pca_threshold=10` so no orbital-product PCA space replaces or augments
that fixed auxiliary basis. This is required because the historical 41-channel
PCA target contains only s, p, and d perturbations and therefore carries no
f/g first-order-wavefunction weight. The explicit auxiliary perturbations are
orthonormalized in the retained full-Coulomb eigenspace before the Delta-ST
responses are written; no additional Coulomb matrix is applied inside SIAB.

## 5. Weighted Residual

For target reference `t` and current AO coefficient set `C`, define

```text
R_t(C) = (1 - P_C) delta_psi_t^Delta-ST.
```

`P_C` is the overlap-metric projector formed from the full fixed-plus-selected
AO space. The reference weight combines the minimax frequency weight with the
declared Coulomb/perturbation weight. The selector never changes these weights
after an RPA energy is observed.

For each target family `F`, the dimensionless loss is

```text
Lbar_F(C) = L_F(C) / L_F(C_DZP),
L_F(C) = sum_t-in-F weight_t * ||R_t(C)||_S^2.
```

Normalizing each family by its fixed-DZP value prevents the largest raw target
file from dominating only because it contains more references.

## 6. Candidate Radial Modes

At every selection step, residual covariances are rebuilt after projecting out
the current basis. For one H angular channel `l`, all atoms, magnetic channels,
frequencies, perturbations, and target geometries in the selected family are
accumulated before diagonalization.

The projected primitive overlap is whitened with the existing relative-rank
cutoff `1e-4`. The Hermitian residual covariance is then diagonalized in that
whitened space. Each eigenvector is one candidate radial shell shared by all H
atoms; its eigenvalue measures the residual weight that shell can capture.

Candidates are regenerated after every accepted shell. A mode selected at
step `k` is therefore conditional on all modes selected at earlier steps.

## 7. Borrowing Metric

For the fragment target, calculate the same physical first-order response with
two projectors:

```text
Lbar_own(C)   = residual using AO functions on the real H center only,
Lbar_ghost(C) = residual using AO functions on the real and ghost centers.
```

The dimensionless borrowing gap is

```text
B(C) = max(0, Lbar_own(C) - Lbar_ghost(C)).
```

`B(C)` measures response improvement available only by borrowing neighboring
centers. It does not penalize the physical H3 response; that response remains a
separate positive training family.

## 8. Greedy Shell Score

For current basis `C` and candidate shell `c=(l,k)`, define

```text
G_atom = Lbar_atom(C) - Lbar_atom(C + c)
G_multi = Lbar_multicenter(C) - Lbar_multicenter(C + c)
G_balance = B(C) - B(C + c)
cost(c) = 2*l + 1

score(c) = (G_atom + G_multi + G_balance) / cost(c).
```

All three terms are dimensionless and have fixed unit coefficients. Positive
`G_balance` rewards a shell that makes the real center less dependent on ghost
functions; negative `G_balance` penalizes increased borrowing. Dividing by
`2*l+1` minimizes the number of actual AO functions rather than only the number
of radial shells.

A candidate is admissible only if:

- `G_atom + G_multi > 0` beyond numerical tolerance;
- its projected overlap is positive after the declared rank truncation;
- the final score is positive;
- adding it preserves complete magnetic multiplets and the fixed-DZP columns.

Ties are resolved deterministically by higher score, lower AO cost, lower `l`,
and lower residual-spectrum index, in that order.

## 9. Nested Sequence And Stopping

After accepting one shell:

1. initialize it from the selected residual eigenvector;
2. optimize all nonfixed response shells with the existing joint ST+dpsi path;
3. verify that `1s,2s,1p` remain bitwise unchanged and all DFT/dpsi gates pass;
4. recompute all target-family residuals, borrowing, and angular spectra;
5. write an immutable basis step and selection record;
6. continue with the next shell.

The spillage sequence stops independently of H2 energy when all of the
following hold:

- the combined atom-plus-multicenter residual has captured at least 99.9% of
  the fixed-DZP residual;
- no individual `l=0,...,4` channel with nonzero target weight retains more
  than 1% of its fixed-DZP residual;
- the borrowing gap has not increased from the fixed-DZP starting point.

It also stops with an explicit failure status if no admissible positive-score
candidate remains. Threshold changes require a new committed specification and
cannot be made after inspecting H2 RPA energies.

## 10. Outputs

Each step is stored under `selection_steps/step_NNN/` and contains:

- initialized and optimized coefficient files;
- generated `.orb` file;
- `selection_record.json` with all candidate scores and rejection reasons;
- per-family losses and per-`l` residual spectra;
- borrowing metric before and after the step;
- coefficient, target, executable, and source SHA256 values;
- selected `Nu.H`, total radial-shell count, and total AO-function count.

The root `selection_manifest.json` records the frozen order of the complete
nested sequence. It contains no H2 RPA energy.

## 11. RPA And BSSE Acceptance

Only after the nested sequence is complete are its candidates evaluated with
the same immutable ABACUS and LibRPA producers. Every candidate uses the same
fixed explicit auxiliary basis, full Coulomb, supercell, Ecut, minimax grid,
band completeness, and occupation checks.

For each candidate calculate H2, H, and `H + H_empty`, then report

```text
D_raw
D_CP
BSSE = D_raw - D_CP
D_Delta-ST
```

The accepted basis is the first basis in the already-frozen sequence that
satisfies

```text
abs(D_CP - D_Delta-ST) < 0.1 kcal/mol
abs(D_raw - D_CP) < 0.1 kcal/mol
```

and all DFT/dpsi gates. If no candidate passes, the campaign fails; the
selector weights or shell order are not retuned to H2. A new target family or
angular cutoff requires a new design stage and a new transfer benchmark.

The historical `107.192818 kcal/mol` raw SOS line is not an acceptance target.
It used a different orbital and no counterpoise correction. The project TeX
will relabel it as a historical, non-CP diagnostic.

## 12. Testing

Unit tests must establish RED before implementation and cover:

1. complete `l=3` and `l=4` magnetic multiplet parsing;
2. rejection of incomplete or cross-target-inconsistent primitive blocks;
3. recovery of known shared-radial spectra from multiple synthetic targets;
4. AO-cost normalization choosing the larger residual reduction per actual
   AO function rather than per radial shell;
5. a synthetic case where the borrowing term changes the selected channel;
6. deterministic tie breaking and byte-identical manifests;
7. fixed-DZP columns remaining bitwise unchanged after every append;
8. absence of any H2 energy field or input in the selector API and manifest;
9. explicit failure when no positive-score candidate remains.

Focused and full SIAB tests run on `df_dcu` `normal`, one full node with 30 CPU
threads and 110610 MB. Physics producers also run only on `normal`. Every
production result records scheduler state, exit code, source commit, binary
hash, input hashes, wall time, and peak memory.

## 13. Commit Boundaries

Implementation is split into independently reviewable commits:

1. this design only;
2. `l<=4` producer/reader contract and tests;
3. multi-target residual spectrum and AO-cost scoring tests;
4. fragment/ghost borrowing metric and tests;
5. deterministic greedy driver and manifests;
6. H target generation and remote regression evidence;
7. frozen nested sequence and radial-orbital plots;
8. fixed-ABS SOS/CP/Delta-ST acceptance results and documentation.
