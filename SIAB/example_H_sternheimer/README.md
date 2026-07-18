# H Sternheimer SIAB experiment

This directory defines the first H-TZDP Sternheimer-supervised SIAB experiment. It contains configuration only: no ABACUS producer output or fabricated production matrices are committed here.

## Provenance status

- The H atom Sternheimer matrix is the only new supervision.
- `st_constrained` reuses the historical H dimer/trimer DFT and dpsi matrices at 0.7, 0.9, and 1.3 Bohr.
- H2 RPA is held out from optimization and is reserved for the final transfer test.
- The initial basis is the existing H TZDP basis at 8 Bohr. Its level-1 `1s` radial orbital is fixed exactly; the remaining `s` and `p` orbitals are optimized.
- The first experiment uses every producer reference row. It applies no PCA and no reference-row truncation.

The experiment is **not physics validated** until the exact ABACUS executable/source commit, pseudopotential, orbital and auxiliary-basis hashes are recorded in the producer metadata, and the planned H/H2 comparison table is populated. A completed optimizer run alone does not satisfy that validation gate.

## Run

Run from this directory after placing the real producer files under the exact `data/` paths in the inputs:

```bash
cp INPUT.st_only INPUT
python3 ../opt_orb_pytorch_dpsi/main.py
```

For the constrained lane:

```bash
cp INPUT.st_constrained INPUT
python3 ../opt_orb_pytorch_dpsi/main.py
```

Both inputs use seed `20260718`. The optimizer must report that value for both NumPy and PyTorch. Compare the resulting named losses in `Spillage.dat` and `ORBITAL_RESULTS.txt`; do not use the held-out H2 RPA result for parameter fitting.
