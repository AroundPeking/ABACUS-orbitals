# Fixed-DZP TZDP SOS-RPA control

This campaign compares two H `3s2p` orbital sets with one common explicit
auxiliary basis:

- `initial_tzdp`: the checked-in H TZDP 8-bohr basis;
- `fixed_dzp_joint`: the `st_dpsi_joint` result from job `21315288`, which
  keeps `1s`, `2s`, and `1p` bitwise fixed and optimizes only `3s` and `2p`.

Each lane runs H2, isolated H, and H in the complete H2 ghost basis. The six
tasks use all `18/9/18` H2/H/ghost LCAO bands, the same 214-function-per-H explicit ABS, 20
Angstrom cell, 0.74085 Angstrom bond, 100 Ry, 16 minimax frequencies,
`rpa_ccp_rmesh_times=5`, Massidda singularity correction, and full Coulomb in
LibRPA. Therefore the orbital coefficients are the only physical input that
changes between the two lanes.

For `3s2p`, the number of AO functions per H is not the number of radial
functions: it is `3 * 1 + 2 * 3 = 9` after magnetic degeneracy. The dimer and
ghost-dimer calculations therefore use 18 bands and the isolated atom uses 9.

Report both the raw and counterpoise-corrected binding energies:

```text
D_raw = 2 E(H) - E(H2)
D_CP  = 2 E(H in H2 ghost basis) - E(H2)
BSSE  = D_raw - D_CP
```

The comparison is a held-out physics gate. A lower SIAB training loss alone
does not establish that the optimized basis improves SOS-RPA.

## Low-frequency guard A/B

`INPUT.st_dpsi_joint_low_frequency_guard` keeps the existing integrated
GreenX-weighted Sternheimer objective and all fixed-DZP `3s2p` settings. Its
only two changes relative to `INPUT.st_dpsi_joint` are a guard weight of `10`
and zero allowed regression at the smallest positive imaginary frequency.
The optimizer reports the local loss at every frequency and rejects any
candidate whose lowest-frequency loss exceeds the initial TZDP value, even if
its integrated loss is lower.

Run this guarded lane with `run_joint_low_frequency_guard.slurm` from a fresh,
immutable campaign directory containing `INPUT`, `SOURCE_COMMIT`,
`SOURCE_MANIFEST.sha256`, and the same producer data used by the unguarded
lane. The pre-SOS checks are:

```text
fixed 1s/2s/1p coefficient difference <= 1e-12
final lowest-frequency loss <= initial lowest-frequency loss * (1 + 1e-12)
integrated Sternheimer loss <= 0.4095114289
DFT loss ratio <= 1.05
dpsi loss ratio <= 1.10
same 3s2p orbital count and 8-bohr cutoff
```

Passing these checks only authorizes the held-out H2/H/H+ghost calculation.
The physical decision still uses the full-Coulomb, 16-frequency raw and
counterpoise-corrected SOS-RPA binding energies defined below.

## Result

The first submitted array, `21438056_[0-5]`, used `10/5/10` bands because it
mistook the five radial functions for five AO functions. Although all six
tasks completed, those energies are invalid and are excluded. Commit
`f724919c` corrected the all-band count to `18/9/18`; replacement array
`21438483_[0-5]` completed all six tasks with exit code zero. Every case has
one ABACUS completion marker, one LibRPA success marker, integer occupations,
and valid production-output checksums. For each geometry, the two lanes have
byte-identical `basis_aux_out` and full-Coulomb matrices.

| basis | D raw | D CP | BSSE | D CP - 108.72 |
|---|---:|---:|---:|---:|
| initial TZDP | 106.635342 | 105.556881 | 1.078461 | -3.163119 |
| fixed-DZP joint `3s2p` | 106.886054 | 105.853882 | 1.032171 | -2.866118 |

All energies are in kcal/mol. Optimizing only `3s,2p` raises the raw binding
by `0.250711` kcal/mol and the CP binding by `0.297001` kcal/mol while reducing
BSSE by `0.046290` kcal/mol. It therefore passes the same-size no-regression
gate, but it does not pass the `0.1` kcal/mol Delta-ST/FHI-aims accuracy gate.

The archived text evidence and parser outputs are under
`/Users/ghj/同步空间/AITP_project/sternheimer_abacus/results/siab_h_fixed_dzp_tzdp_sos_21438483_text`.
The complete reader-v1 matrices remain on `df_dcu` under
`/work1/ghj/sternheimer_abacus_tests/siab_fixed_dzp_tzdp_sos_campaign_v2_20260730`.

## Orbital and response-space diagnostic

`analyze_orbitals.py` compares the exact exported radial orbitals and evaluates
the Sternheimer target with the same fixed `1s,2s,1p` DZP projector. Run it
from the `ABACUS-orbitals` root with a Python environment containing NumPy,
SciPy, Matplotlib, and PyTorch:

```bash
/Users/ghj/apps/anaconda3/bin/python3 \
  SIAB/example_H_sternheimer/fixed_dzp_tzdp_sos/analyze_orbitals.py
```

At the declared relative primitive-rank tolerance `1e-4`, the reproduced ST
losses are `0.4428607140` for the initial TZDP and `0.4054568603` for the
jointly optimized TZDP. The best ST-only projector containing exactly one
additional shared s radial function and one additional shared p radial
function has loss `0.3944469227`. The current joint result therefore has only
`0.0110099376` absolute ST-loss headroom at the same `3s2p` size. This is a
lower bound for the ST objective, not a guarantee that the same value can be
reached while retaining the DFT/dpsi constraints.

The metric overlap with the leading fixed-DZP-projected residual response mode
changes from `0.972241` to `0.972743` for `3s`, and from `0.952821` to
`0.992838` for `2p`. Thus the useful shape change is predominantly in `2p`;
the initial `3s` was already close to the best rank-one s direction. Even an
unlimited number of stable s/p primitive directions has a loss floor
`0.3028482444`, dominated by response angular momentum absent from this
four-block s/p target. Reoptimizing only the same two radial functions cannot
remove that floor.

The target stores `Q=<delta_psi|primitive>`, the primitive overlap `S`, and the
exact reference norm, but not the full three-dimensional grid wavefunction.
The plotted reference is therefore the rank-revealing primitive projection
`c=S^+ Q^H`, not an unprojected grid dump. For the displayed representative s
and p channels at `0.0687`, `1.305`, and `8.366 Ha`, this projection captures
between `99.656%` and `99.9996%` of the corresponding reference norm. The
original server directory also contains no full Sternheimer wavefunction file;
showing the unprojected grid reference would require a new diagnostic output.

Generated figures and the machine-readable summary are under
`results/fixed_dzp_tzdp_orbital_analysis/` in the parent project.
