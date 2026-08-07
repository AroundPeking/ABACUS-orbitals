# H Delta-ST Parent-Space Convergence Design

## Scope

This phase establishes whether the Bessel primitive parent space used by SIAB
can represent the uniform-grid Delta-Sternheimer response of an isolated H
atom. It is a completeness test before any compact-orbital optimization.

The calculation keeps the physical problem fixed: one spin-polarized H atom in
a 20 Angstrom cubic cell, 100 Ry uniform grid, H TZDP 8-bohr zero-order basis,
16 fixed imaginary frequencies, and an auxiliary basis generated with
`exx_pca_thr=1e-4`. The accepted reference has 33 auxiliary channels. All
response comparisons use the same auxiliary ordering and the same full Ewald
Coulomb matrix.

## Physical Reference

For occupied state `i`, perturbation channel `mu`, and imaginary frequency
`omega_h`, the uniform-grid Delta-ST calculation solves the already validated
ABACUS equation and assembles

```text
M_ref(nu,mu;h) = sum_i f_i <psi_i | P_nu | delta_omega psi_i_mu> + c.c.
```

The perturbation `P_mu` and the response matrix `M_ref` must retain the producer
channel order. The physical symmetric response is

```text
Pi_ref(h) = V_full^(-1/2) M_ref(h) V_full^(-1/2).
```

`V_full` is the positive retained eigenspace of the full Ewald Coulomb matrix.
The same eigenvalue threshold, eigenvector convention, frequency list, spin
occupation, auxiliary basis, and channel order are used on both sides of every
comparison. No cut-Coulomb matrix is allowed in this gate.

The RPA correlation energy is evaluated from

```text
E_c^RPA = (1 / 2 pi) sum_h w_h Tr[log(I - Pi(h)) + Pi(h)].
```

The numerical trace-log implementation must first reproduce an existing
ABACUS-to-LibRPA fixed-AO reference before it is used to assess the parent
space.

## Bessel Parent-Space Response

The zero-order density and occupied state remain those of the fixed H TZDP
calculation. Only the response trial space changes. For a Bessel primitive
basis `B={b_a}` with overlap `S_B`, Hamiltonian `H_B`, and perturbation matrices
`P_mu^B`, solve the finite-space Galerkin Sternheimer equation in the virtual
complement of the occupied state:

```text
[Q_B (H_B - eps_i S_B + i omega_h S_B) Q_B + P_B]
    delta c_i_mu(h) = -Q_B P_mu^B c_i.
```

The implementation may use the certified Lowdin-orthogonalized form already
provided by `evaluate_primitive_galerkin`; it must not construct or sum over an
explicit virtual eigenstate list. The resulting `M_B` is transformed with the
same numerical `V_full` as the grid reference:

```text
Pi_B(h) = V_full^(-1/2) M_B(h) V_full^(-1/2).
```

Agreement of finite-space Galerkin and finite-space SOS is only an algebra
gate. Agreement of `Pi_B` with `Pi_ref` is the physical parent-space
completeness gate.

## Independent Convergence Axes

The primitive exporter currently derives angular channels from the fixed AO
`nwl`, which limits H TZDP output to `l<=1`. Add an explicit
`sternheimer_galerkin_primitive_lmax` input so the response parent space is
independent of the zero-order AO basis. The default preserves the current
behavior; an explicit value exports all `l=0..lmax` channels.

Convergence is tested in two stages:

1. Radial scan: hold `lmax=1` and vary `bessel_nao_ecut`. This identifies the
   radial plateau without changing angular content.
2. Angular scan: hold the first radially converged setting and vary
   `lmax=1,2,3,4`. This measures the missing high-angular-momentum response.

Each point records primitive count, retained overlap rank, condition number,
maximum per-frequency relative response error, all-frequency Frobenius error,
RPA correlation energy, wall time, maximum memory, source revision, binary
hash, and input hashes.

## Interface Gate

Before the 16-frequency reference, run one exact frequency with the same
system and auxiliary basis. The run is accepted only when:

- ABACUS finishes normally and writes one complete `M` block;
- the frequency is byte-for-byte or numerically identical to the selected
  fixed-grid value;
- the auxiliary channel count is 33 and channel metadata/order hashes match;
- `V_full` dimensions and retained rank match the response;
- `M` is Hermitian after the documented `+c.c.` assembly;
- no source or parser silently substitutes cut Coulomb.

## Acceptance Criteria

The Bessel parent space is considered complete for the H gate only when one
tested point satisfies both:

```text
max_h ||Pi_B(h)-Pi_ref(h)||_F / ||Pi_ref(h)||_F < 1e-3
|E_c,B^RPA - E_c,ref^RPA| < 0.01 kcal/mol.
```

If the energy criterion is met while the matrix criterion is not, record the
cancellation and do not declare the parent space complete. If no `lmax<=4`
point passes, extend angular momentum before compacting or optimizing orbitals.

## Scientific Boundary

Passing this H-atom gate means the uncompressed Bessel parent space can carry
the relevant first-order wavefunction response. It does not yet prove that a
small atom-centered basis can do so, that H2 binding is converged, or that BSSE
is controlled. Those are subsequent compression and transferability gates.

