# H3 DFT/dpsi matrix producer

This producer reconstructs the original level-3 H-TZDP SIAB training set. The
checked-in historical `SIAB_INPUT` defines `STRU2` as an equilateral H3 trimer,
not an H2 dimer, at side lengths 0.7, 0.9, and 1.3 Angstrom. It uses 10 bands,
`nspin=1`, a 20-bohr cubic cell, 100-Ry PW and spherical-Bessel cutoffs, an
8-bohr spherical-Bessel radius, and the Dojo-NC-SR H pseudopotential.

The ABACUS output contract is `out_spillage=2`. Each case must produce both
`orb_matrix.0.dat` (zero-order wavefunctions) and `orb_matrix.1.dat` (the
original SIAB dpsi reference). `INPUTw` retains the historical controls while
`INPUT` also sets the current ABACUS parameters explicitly.

Stage `Dojo-NC-SR/Pseudopotential/H.upf` as `H.upf` in this directory, then
submit from a clean campaign copy:

```bash
sbatch run_abacus.slurm
```

The array has three tasks. Each task uses one full `normal` node with 30 MPI
ranks and 110610 MB. The script refuses to overwrite an existing `OUT.*`
directory and verifies the immutable ABACUS executable hash, Dojo H hash, SCF
completion marker, tagged matrix endings, and final matrix hashes.
