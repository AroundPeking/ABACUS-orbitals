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
