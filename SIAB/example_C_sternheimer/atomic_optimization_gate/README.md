# C atomic Sternheimer SIAB optimization gate

This is the first optimizer gate after the accepted C atomic Delta-ST producer.
It is not a transferable C basis and it is not an RPA validation result.

The gate keeps the SG15 C TZDP size at `3s3p2d`.  It freezes the DFT-safe DZP
columns `1s,2s,1p,2p,1d` and permits only `3s,3p,2d` to vary.  The target is
the 16-frequency, full-Coulomb C atomic Sternheimer matrix.  The short run uses
20 Adam steps with learning rate `0.001` and the response-only `st_only` loss.
This short loss is used only to prove that the C input, freeze map and PyTorch
gradient path are consistent.

The three source hashes are fixed in `atomic_optimization_gate.py`:

- Sternheimer matrix: `e976c164595758029cb91ebe3913af6865780ef95fdb48954c07075bd0c7e3ff`;
- SG15 C-TZDP coefficients: `b58a2183c3028e46e6f4bc55b0f21531f1253275d5c2f2c4ee4e27676c1b55f4`;
- SG15 C-TZDP orbital: `7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d`.

The audit passes only when all five DZP coefficient columns remain bitwise
unchanged, at least one of the three variable columns changes, and the best
finite Sternheimer loss is no worse than the initial loss.

After this gate, formal C training still requires C-C bonding targets.  Its
physical objective must combine the atomic target with C-C environments, and
the final raw SOS result must be checked at held-out C-C separations and
diamond volumes using `exx_pca_threshold=1e-4`.  The atomic gate alone must not
be used to select or publish a C basis.
