# Held-out H2/H SOS-RPA test for the expanded H 4s3p basis

This campaign evaluates the fixed-DZP `st_dpsi_joint` H `4s3p` basis without
using H2 in training. It differs from `../held_out_h2_sos` only in the orbital
and the number of LCAO bands: H2/H use all `26/13` bands because each H atom
has 13 AO functions. The 20-Angstrom cell, 0.74085-Angstrom H2 bond, 100-Ry
cutoff, 16 minimax frequencies, PCA threshold `1e-4`,
`rpa_ccp_rmesh_times=5`, Massidda correction, and full Coulomb are unchanged.

Stage the following files beside this README:

- `H_gga_8au_100Ry_4s3p.orb`: expanded joint `ORBITAL_1U.dat`, SHA256
  `b394bb7329754e38341050ca4beb3b242b78e4be50c418b8764c98226bc8f033`.
- `H_ONCV_PBE-1.0.upf`: reference ONCV H pseudopotential, SHA256
  `3f5deaaf57ff09d87da2c2c186eb1654d174fe385c9edc265036b3c8869286ef`.

Submit `run_sos.slurm` from a clean campaign copy. Its two array tasks each use
one full `normal` node and run ABACUS followed by LibRPA. The script verifies
all immutable hashes and refuses to reuse existing outputs.
