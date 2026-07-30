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
