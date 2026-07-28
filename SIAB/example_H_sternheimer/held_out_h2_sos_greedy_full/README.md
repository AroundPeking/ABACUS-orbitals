# Full greedy-basis H2/H SOS-RPA and counterpoise gate

This campaign tests whether the converged greedy response basis can reproduce
the Delta-Sternheimer H2 binding reference before any basis compression is
attempted. The orbital is the final `13s11p10d5f4g` result from response
selection job `21406582`, with SHA256
`d518c5997f667249554ab19af1a36d12f17439ffb7f0ecaf5977c1e829be7b00`.

The three array tasks are H2, isolated H, and H plus `H_empty`. They retain the
same 20-Angstrom cell, 0.74085-Angstrom bond, 100-Ry LCAO grid, explicit
214-function-per-H auxiliary basis, `rpa_ccp_rmesh_times=5`, full Coulomb, and
16-frequency LibRPA contract as the earlier fixed-ABS controls. Every LCAO band
is included: 334 for H2 and H+ghost, and 167 per spin channel for isolated H.

The primary validation number is the counterpoise result

```text
D_CP = 2 E_RPA@PBE(H + H_empty) - E_RPA@PBE(H2).
```

The uncorrected value `2 E(H) - E(H2)` is retained to quantify RPA basis-set
superposition error. The ghost result is diagnostic only and was not used by
the response-shell selector.

For an apples-to-apples comparison, this campaign intentionally reuses the
historical binaries that produced the accepted 20-Angstrom SOS controls:

- ABACUS commit `80a606f57a2610bc2532468661b687b01f58074c`, binary SHA256
  `2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d`;
- LibRPA binary SHA256
  `defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c`.
  Its remote source directory is not a Git repository; the documented local
  matching feature branch is `codex/sternheimer-rpa-full` at `d70b5571`.

This is a deliberate historical-version exception, not a claim that these
binaries match the current `master_ghj` branches. Stage a `SOURCE_COMMIT` file
beside `run_sos_cp.slurm` before submission.

## Single-rank memory gate and ghost retry

The first production array was `21409783_[0-2]`. Isolated H completed the
full ABACUS-to-LibRPA path in 39:26, and H2 completed its ABACUS producer with
a 58,782,948-KB peak before entering LibRPA. The spin-polarized H+ghost task
exhausted the single-node memory limit after 39:08: Slurm reported
`OUT_OF_MEMORY`, exit code `0:125`, and a peak of 112,502,556 KB against the
110610-MB request.

This failure is specific to the simultaneous two-center and two-spin reader-v1
producer. The same orbital and auxiliary basis completed for two centers with
one spin and for one center with two spins. The ABACUS RPA-LRI path distributes
atom-pair tensors over MPI ranks and writes one `v1_Cs_data_<rank>.txt` file per
rank. `run_ghost_mpi_cp.slurm` therefore retries only H+ghost with two nodes,
one MPI rank and 30 OpenMP threads per node. LibRPA remains a one-rank
postprocessor and reads every rank file through the unchanged reader-v1
prefixes. No occupation, band, basis, Coulomb, frequency, or counterpoise
setting is changed by this retry.
