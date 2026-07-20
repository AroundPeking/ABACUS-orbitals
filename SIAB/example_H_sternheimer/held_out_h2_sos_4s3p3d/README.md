# Held-out H2/H SOS-RPA test for the smooth H 4s3p3d basis

This is the one predeclared molecular transfer test after the atomic d-response
gate. H2 was not used to choose the shell counts or optimize the orbital. Every
LCAO band is included: the new `4s3p3d` basis uses `56/28` H2/H bands, while
the old `4s3p` fixed-ABS control uses `26/13`.

The array contains three lanes:

| lane | wavefunction basis | auxiliary basis | purpose |
| --- | --- | --- | --- |
| `regenerated_4s3p3d` | smooth joint `4s3p3d` | on-site products, PCA `1e-4` | standard production result |
| `fixed_4s3p` | previous joint `4s3p` | explicit common H ABS | fixed-ABS control |
| `fixed_4s3p3d` | smooth joint `4s3p3d` | the same explicit common H ABS | isolate AO-space change |

All lanes use the same Dojo H pseudopotential, 20-Angstrom cubic cell,
0.74085-Angstrom H2 bond, 100-Ry cutoff, 16 minimax frequencies,
`rpa_ccp_rmesh_times=5`, Massidda correction, and full Coulomb in LibRPA.
The fixed lanes set `exx_pca_threshold=10`; this screens out the product-PCA
basis so the explicitly supplied `ABFS_ORBITAL` is the only auxiliary space.

Stage these immutable assets beside this README:

- `H_gga_8au_100Ry_4s3p3d.orb`: smooth rank-`1e-4` atomic-gate orbital from
  job `21317844`, SHA256
  `860f3a23b2d81486a62b48c80992a5610194899e049fe44dacb67b1ef983a7b9`.
- `H_gga_8au_100Ry_4s3p.orb`: previous joint basis, SHA256
  `b394bb7329754e38341050ca4beb3b242b78e4be50c418b8764c98226bc8f033`.
- `H_ONCV_PBE-1.0.upf`: reference pseudopotential, SHA256
  `3f5deaaf57ff09d87da2c2c186eb1654d174fe385c9edc265036b3c8869286ef`.
- `H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs`: explicit fixed ABS already used in
  the ABACUS H Sternheimer workflow, SHA256
  `d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f`.
  Its radial-channel counts correspond to 214 auxiliary functions per atom.
- `SOURCE_COMMIT`: the full ABACUS-orbitals commit staged for the campaign.

`run_sos.slurm` uses six independent `normal` array tasks. Each task requests
one full 30-core, 110610-MB node, runs one ABACUS rank followed by one LibRPA
rank, validates all immutable hashes, checks the exact band and spin counts,
requires integer occupations with the correct electron total, and refuses to
reuse existing output.

The ABACUS executable is the same immutable task-4 producer used for the
previous `4s3p` comparison: source commit
`80a606f57a2610bc2532468661b687b01f58074c`, binary SHA256
`2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d`.
The LibRPA binary SHA256 is
`defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c`.
These builds intentionally differ from current local `master_ghj`: reusing the
same producer is required here so the basis change is not mixed with a code
version change.

## Result

Not run yet. Populate this section only after all six tasks have valid ABACUS
and LibRPA completion markers and the H/H2 binding decomposition is recomputed
from the recorded outputs.
