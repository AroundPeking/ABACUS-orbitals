# Compact H Response-Basis Campaign

This campaign replaces the old `global_capture=0.999` endpoint with a frozen
`48 AO/H` frontier. It starts from the full H TZDP `3s2p` basis, keeps the DZP
`1s,2s,1p` prefix bitwise fixed, and scores added atomic Delta-Sternheimer
residual shells with the physical H and H2 targets only.

## Predeclared locality lanes

The radial cutoff is `8 bohr` and the locality boundary is fixed at
`Rloc=4 bohr`. At the TZDP starting point, the measured normalized H and H2
response losses are `0.75274579` and `0.77624547`. The summed ST gradient norm
on variable columns is `0.14648423`; the radial-tail loss is `0.33213575` with
gradient norm `0.47755980`. Thus `w_tail=0.3067` gives equal initial gradient
norms. Before any SOS energy is read, the campaign freezes three lanes:

- `w_tail=0.0`: compact AO-budget control;
- `w_tail=0.1`: moderate locality regularization;
- `w_tail=0.3`: approximately equal ST/locality initial gradients.

All lanes retain the existing `st_dpsi_joint` DFT and dpsi constraints. Ghost
targets and RPA energies are excluded from selection and optimization.

## Acceptance sequence

Each Slurm array member uses one full `normal` node, verifies the immutable
source and existing target/asset hashes, runs the complete SIAB unit suite,
and freezes every optimized step up to the AO budget. Each frozen record must
contain the ST loss, tail fraction, weighted locality loss, and both condition
numbers. Full-Coulomb all-band SOS is a later gate and cannot change these
sequences.
