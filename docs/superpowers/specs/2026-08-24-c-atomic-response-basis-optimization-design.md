# C Atomic Response-Basis Optimization Design

## Goal

Build the smallest C numerical atomic-orbital candidate that materially
reduces the complete FD8 atomic Sternheimer response residual without changing
the frozen DFT-safe DZP block. This first campaign ends with a converged
`3s3p2d` checkpoint, an `l=0,...,4` residual-spectrum ranking, and one
deterministic next-shell seed. It does not promote a basis or run SOS-RPA.

## Frozen Physical Definition

- Sternheimer target SHA256:
  `e976c164595758029cb91ebe3913af6865780ef95fdb48954c07075bd0c7e3ff`.
- Target protocol: isolated C, SG15, 20 Angstrom box, `135^3` FD8 grid,
  16 imaginary frequencies, full Coulomb, SG15-TZDP product auxiliary basis
  selected with PCA threshold `1e-4`.
- Initial checkpoint SHA256:
  `3e0b83c95ce744dd75d54da9128ecbadc11fb7d3357830af697a47d0c6b6d406`.
- Candidate shell count: `Nu(C)=[3,3,2,0,0]`. The zero `f/g` entries preserve
  the complete reference residual and do not add orbitals.
- Frozen orbitals: `1s,2s,1p,2p,1d`.
- Variable orbitals in the continuation: `3s,3p,2d`.
- Loss: full-frequency normalized atomic `st_only` loss.

The updated FD8 cohesive-energy anchor is `6.904014 eV/C`. It is a later
physical validation target, not an optimizer objective in this atomic-only
campaign.

## Stage A: Converge the Existing TZDP Space

Continue from the accepted 20-step checkpoint rather than restarting from
SG15 TZDP. Use Adam at learning rate `1e-3` for at most 3000 steps. The
optimizer's existing 51-consecutive-nonimproving-step stop remains active.

The continuation passes only when:

1. all five frozen DZP coefficient columns remain bitwise identical to the
   original SG15 coefficient source;
2. every accepted loss is finite and the final best loss is below the
   20-step checkpoint loss;
3. the relative best-loss decrease over the final 100 accepted steps is below
   `1e-4`, or the built-in nonimprovement stop fires;
4. the maximum Sternheimer overlap condition remains below `1e12`.

If 3000 steps end while the 100-step relative decrease remains at least
`1e-4`, the result is recorded as `CONTINUE_REQUIRED` and is not used for
shell selection.

## Stage B: Angular-Momentum Residual Spectrum

Project every orbital in the converged candidate out of the complete atomic
response. For each `l=0,...,4`, diagonalize the shared radial residual
covariance with relative-rank tolerance `1e-12`, magnetic-overlap tolerance
`1e-4`, and condition limit `1e12`.

For each channel report:

- total residual-spectrum weight;
- leading eigenvalue;
- cumulative capture after 1, 2, and 3 radial modes where available;
- numerical rank;
- AO cost `2*l+1`;
- score `leading_eigenvalue/(2*l+1)`.

The next shell is the unique positive-score channel with the largest score.
Its initial Bessel coefficients are the leading residual eigenvector. No
random high-`l` orbital and no simultaneous multi-shell addition is allowed.
If the top two scores differ by less than 1%, the campaign stops for a physics
review instead of choosing arbitrarily.

## Stage C Boundary

The first-shell seed is an analysis artifact, not a released basis. After it
is reoptimized, formal basis selection must add C-C response environments and
the full-frequency projected-Pi objective. A candidate is promoted only after
raw SOS-RPA with `exx_pca_threshold=1e-4` agrees with the matched FD8
Delta-ST atomic/diamond result and held-out diamond volumes. Atomic
`st_only` improvement alone is never a promotion criterion.

## Resources And Provenance

Run on the df `p1` production partition. The accepted 20-step gate used about
2 GiB and averaged about three CPU cores, so request 8 CPUs and 16 GiB for the
continuation. Record Slurm state, elapsed time, TotalCPU, MaxRSS, source commit,
target/checkpoint hashes, Python/PyTorch versions, all failed parents, and all
optimizer/spectrum output hashes. Never overwrite or duplicate a completed
physical definition.

## Documentation

Archive compact result files under the development-note `data/` tree. Update
the C SIAB section with the convergence trajectory, per-`l` spectrum table,
selected shell, resources, and explicit statement that SOS validation remains
pending. Rebuild and visually inspect the affected PDF pages.
