# Held-out H2/H SOS-RPA test

This campaign tests the fixed-DZP `st_dpsi_joint` H basis without using H2 in
training. It reproduces the validated 20-Angstrom H2/H reference settings: 100
Ry, 16 GreenX minimax frequencies, PCA threshold `1e-4`, `rpa_ccp_rmesh_times=5`,
and full Coulomb with Massidda singularity correction. H2 uses a 0.74085
Angstrom bond; H is spin polarized with one spin-up electron.

Stage the following generated/provenance-controlled files beside this README:

- `H_gga_8au_100Ry_3s2p.orb`: joint `ORBITAL_1U.dat`, SHA256
  `30b7e5e3d80b59778b0fee836fcd0315c0cfd827621806eb3f2c9e659b8118a7`.
- `H_ONCV_PBE-1.0.upf`: the reference ONCV H pseudopotential, SHA256
  `3f5deaaf57ff09d87da2c2c186eb1654d174fe385c9edc265036b3c8869286ef`.

Submit `run_sos.slurm` from a clean campaign copy. Its two array tasks each use
one full `normal` node and run ABACUS followed by LibRPA. The script refuses to
reuse existing outputs and verifies all immutable executable/input hashes.
The joint orbital hash is the default. A same-producer initial-TZDP baseline
may override it explicitly with Slurm's
`EXPECTED_ORBITAL_SHA256` export; use a separate clean campaign directory.
