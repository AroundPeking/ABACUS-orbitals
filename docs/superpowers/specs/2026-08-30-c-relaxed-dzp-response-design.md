# C Relaxed-DZP Response Optimization

## Purpose

Test whether the fixed `2s2p1d` prefix is preventing the compact carbon basis
from reproducing the Delta-Sternheimer response.  The experiment keeps the
current fixed-prefix `3s3p2d` production run as an unchanged control and adds
one partially relaxed candidate.  It does not rerun Delta-ST.

The final physical target remains the ordinary all-band SOS binding energy

\[
E_{\rm bind}=E_{\rm C\,atom}-\frac{1}{2}E_{\rm C_2\,solid},
\]

compared with the fixed converged Delta-ST reference
`6.902326 eV/C`.  The relaxed candidate is accepted only when the absolute
SOS binding-energy difference is below `0.1 eV/C`.

## Controlled candidate

Use the unoptimized `C_TZDP_10au_3s3p2d` orbital as the starting point so the
new result has the same `3s3p2d`, 22-AO-per-C size as the fixed-prefix control.
The radial-channel contract is

- total radial multiplicities: `nu=3,3,2,0,0`;
- fixed innermost profiles: `fixed_nu=1,1,0,0,0`;
- fixed channels: first s and first p radial profiles;
- relaxed channels: second and third s, second and third p, and both d radial
  profiles.

The first s and p profiles preserve the compact valence backbone.  The outer
DZP profiles are allowed to share the response fit with the added outer
profiles instead of forcing only one s, one p, and one d radial direction to
carry the entire response correction.

The occupied-capture boundary is referenced to the complete unoptimized
`3s3p2d` initial candidate, not to the reduced `1s1p` fixed prefix.  Therefore
relaxing the outer DZP profiles cannot be accepted merely because the inner
profiles still span the correct number of occupied states.

## Training objective

Retain the existing independently normalized atom-solid objective

\[
L_F(C)=
\frac{\sum_{d\in F}w_d\sum_j w_j
\lVert\Pi_d(C,i\omega_j)-\Pi_d^{\rm ref}(i\omega_j)\rVert_F^2}
{\sum_{d\in F}w_d\sum_j w_j
\lVert\Pi_d^{\rm ref}(i\omega_j)\rVert_F^2},
\qquad
L_{\rm joint}=\frac{L_{\rm C\,atom}+L_{\rm C\,solid}}{2}.
\]

Use the accepted 16-frequency C-atom target and solid q1/q2 training targets;
reserve q3 as held out.  Both physical family losses must improve separately.
Occupied capture and overlap conditioning are hard numerical gates, not the
response objective.

## PBE preservation gate

PBE correctness is defined relative to the original unoptimized TZDP results,
not a PW calculation.  Freeze these reference values and their provenance:

| Side | Reference job | Original PBE energy |
|---|---:|---:|
| spin-polarized C atom | `3156999` | `-5.329967462866793 Ha` |
| two-C diamond primitive cell | `3127074` | `-11.030536886845258 Ha` |

The corresponding original-AO PBE binding energy is
`5.042296553665174 eV/C`.  The energy tolerance `10 meV` is
`3.674932217565499e-4 Ha`.

After optimization, rerun ordinary self-consistent PBE for both sides with the
same pseudopotential, geometry, cell, k mesh, spin state, occupation contract,
ABACUS executable, and convergence thresholds as their references.  The
candidate passes only if all three inequalities hold:

\[
\left|E_{\rm C\,atom}^{\rm cand}-E_{\rm C\,atom}^{\rm original}\right|
\le 10\ {\rm meV},
\]

\[
\frac{1}{2}\left|E_{\rm C_2\,solid}^{\rm cand}
-E_{\rm C_2\,solid}^{\rm original}\right|\le 10\ {\rm meV/C},
\]

\[
\left|E_{\rm bind,PBE}^{\rm cand}
-E_{\rm bind,PBE}^{\rm original}\right|\le 10\ {\rm meV/C}.
\]

The atom must also retain the fixed integer occupations `Nup=3`, `Ndown=1`,
and both PBE calculations must be genuinely SCF converged.  Energy cancellation
cannot rescue a side that fails its individual energy gate.

## Execution and promotion gates

1. Let fixed-prefix production job `3159150` finish and validate it without
   modification.  It is the control result and must not be duplicated.
2. Run a two-step partially relaxed pilot from the original TZDP coefficients.
   Require finite atom and solid losses, no occupied-capture rejection, finite
   overlap condition, and a valid best checkpoint.
3. Submit one 500-step partially relaxed production only after the pilot
   passes.  Do not run multiple relaxed-prefix productions concurrently.
4. Require both family losses and held-out q3 response to improve over the
   unoptimized TZDP basis.  Check radial smoothness, kinetic norms, overlap
   conditioning, and the solid virtual spectrum.
5. Run the atom and solid PBE preservation gate above.  Reject the candidate
   before RPA if any PBE condition fails.
6. With `exx_pca_thr=1e-4` threshold-only product PCA, all bands, six
   frequencies, exact-grid full periodic Coulomb, and LibRPA `d4810f73`, run
   the C atom and the validated eight-q-star solid reconstruction.
7. Accept the basis only if the ordinary SOS binding energy differs from
   `6.902326 eV/C` by less than `0.1 eV/C`.  Training loss, occupied capture,
   or a PBE pass alone is not a physical acceptance.

## Failure interpretation

- Response improves but PBE fails: the outer DZP relaxation is too large; do
  not publish the candidate and do not hide the failure through atom-solid
  cancellation.
- PBE passes but SOS does not improve: projected-Pi training remains an
  insufficient proxy for the all-band SOS binding-energy difference.
- Overlap or virtual-spectrum gate fails: reject before PBE and SOS; a lower
  training loss does not justify a numerically pathological AO manifold.
- All gates pass but the error remains above `0.1 eV/C`: the compact
  `3s3p2d` angular/radial space is insufficient, motivating one controlled f
  direction with the same PBE and threshold-only auxiliary-basis gates.

