# H `3s2p2d1f` Sternheimer-SIAB basis

This directory is the compact public artifact for the current H
Sternheimer-supervised SIAB optimization. It contains the optimization input,
a portable launcher, and the resulting ABACUS numerical atomic orbital.

## Physical scope

- Fixed orbitals: `1s`, `2s`, and `1p` (the original DZP block).
- Optimized response orbitals: `3s`, `2p`, `1d`, `2d`, and `1f`.
- Response training families: isolated H and H2 at `R = 0.74085 Angstrom`.
- DFT/dpsi regularization geometries: equilateral H3 at side lengths `0.7`,
  `0.9`, and `1.3 Angstrom`.
- Radial representation: `Rcut = 8 bohr`, `Ecut = 100 Ry`, `dr = 0.01 bohr`.
- Optimizer: Adam, learning rate `0.003`, at most `3000` steps, seed
  `20260718`.
- Response loss: full-frequency H+H2 projected-Pi loss with
  `joint_dpsi_weight = 0.02` and no radial-tail penalty.

This is a single-H2-geometry optimization, not a multigeometry H2
optimization. Its ordinary all-band SOS-RPA binding at the equilibrium point
is `108.566874 kcal/mol`, only `-0.046942 kcal/mol` from the matched Delta-ST
reference. The five-point H2 curve has a maximum absolute error of
`1.3189 kcal/mol`, so this basis is an optimization artifact and current
candidate, not yet a transferable production release.

## Files

- `H_gga_8au_100Ry_sternheimer_3s2p2d1f.orb`: directly usable ABACUS orbital.
- `SIAB_INPUT.json`: exact optimization parameters.
- `run_optimization.sh`: stages `SIAB_INPUT.json` and runs the repository SIAB
  optimizer against an external training-input directory.

The orbital SHA256 is

```text
4171a07bc752256aca4d64df02a8de773f367a3b9501678a4aa6c11118477249
```

## Run

The large ABACUS response and dpsi matrices are deliberately not duplicated in
Git. Prepare a directory containing the paths referenced by
`SIAB_INPUT.json`, then run

```bash
SIAB_PYTHON=/path/to/python \
./run_optimization.sh /path/to/training_inputs /path/to/new_output_directory
```

The Python environment must provide NumPy and CPU PyTorch. The validated
campaign used Python 3.10, NumPy 1.26.4, and PyTorch 2.1.0+cpu.
