# Finite-AO Galerkin Sternheimer Design

## Scope

This design introduces a matrix-level evaluator for the response produced by a
finite candidate AO space. It is the first gate in the new SIAB route after the
historical projected-Pi ranking ended with `stop_galerkin_required`.

The evaluator is not a new complete-space reference and does not claim that a
basis is complete when Galerkin Sternheimer and SOS agree. Their agreement is
only an algebra and implementation gate. Basis quality is established later by
comparison with the uniform-grid Delta-ST response and by a held-out
ABACUS-to-LibRPA raw RPA binding energy.

This first implementation is deliberately matrix-only. It neither changes the
optimizer nor creates an orbital candidate. The ABACUS primitive-matrix
producer is a subsequent phase after the solver contract is certified.

## Physical Contract

Inputs are complex Hermitian matrices in one finite AO space:

- overlap `S`, shape `(n, n)`, positive definite;
- Hamiltonian `H`, shape `(n, n)`;
- perturbation operators `V`, shape `(n_aux, n, n)`;
- band occupations `f`, shape `(n,)`, ordered with the generalized
  eigenvalues;
- positive imaginary frequencies `omega`, shape `(n_freq,)`.

The implementation uses the symmetric Lowdin transformation

```text
X = S^(-1/2)
Hbar = X^H H X
Vbar_mu = X^H V_mu X
```

and uses a dense diagonalization in the first implementation to determine the
energies and occupied subspace. A complete QR of the occupied eigenvectors
provides an orthonormal complement `Uv`; the Galerkin solve does not use the
individual virtual eigenvectors returned by the diagonalization:

```text
[Uv^H (Hbar - eps_i I + i omega I) Uv] x_i_mu
    = -Uv^H Vbar_mu u_i
delta_u_i_mu = Uv x_i_mu
```

For each frequency, define

```text
A_nu_mu = sum_i f_i <u_i | Vbar_nu | delta_u_i_mu>
M = A + A^H
```

The explicit SOS oracle uses all finite-basis virtual eigenstates and the same
sign, occupations, and frequencies. It must reproduce `M` to floating-point
roundoff.

## Public Python API

Create `SIAB/opt_orb_pytorch_dpsi/galerkin_sternheimer.py` with:

```python
@dataclass(frozen=True)
class FiniteAOResponseResult:
    energy: torch.Tensor
    occupation: torch.Tensor
    frequency_ha: torch.Tensor
    response_half: torch.Tensor
    response: torch.Tensor
    overlap_condition: float


def evaluate_galerkin_response(
    overlap,
    hamiltonian,
    perturbation,
    occupation,
    frequency_ha,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
) -> FiniteAOResponseResult:
    ...


def evaluate_sos_response(
    overlap,
    hamiltonian,
    perturbation,
    occupation,
    frequency_ha,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
) -> FiniteAOResponseResult:
    ...
```

All physical tensors are CPU `torch.complex128`, except occupations and
frequencies, which are CPU `torch.float64`. The functions reject implicit dtype
or device conversion so producer mistakes fail closed.

An occupation is occupied when it is strictly positive. At least one occupied
and one virtual state are required. Frequencies must be finite and strictly
positive. `S`, `H`, and every `V_mu` must be Hermitian to a fixed relative
tolerance. The overlap eigenvalues must be positive and its condition number
must not exceed `condition_limit`.

## Invariances and Acceptance

The focused test suite must establish:

1. a diagonal two-level analytic response has the expected sign and factor;
2. Galerkin and SOS responses agree for dense complex Hermitian matrices;
3. an occupied-space phase or unitary rotation does not change the response;
4. an invertible AO coordinate transformation applied consistently to
   `S`, `H`, and `V` does not change the response;
5. output response matrices are Hermitian;
6. singular/ill-conditioned overlap, non-Hermitian inputs, illegal
   occupations, and non-positive frequencies fail explicitly;
7. gradients through the Galerkin response are finite for a nondegenerate
   matrix fixture.

For random dense fixtures, require Galerkin/SOS relative Frobenius error below
`1e-11` and maximum absolute error below `1e-12`.

## Follow-on ABACUS Contract

After this matrix solver passes independently, extend
`out_sternheimer_siab` on branch `codex/sternheimer-siab-producer` to export
primitive `S^p`, `H^p`, and `V^(p,mu)` with occupations, frequency grid,
spin convention, auxiliary whitening, full-Coulomb provenance, and source
hashes. The first real gate uses one fixed TZDP H/H2 case and compares this
evaluator against a full-bands ABACUS SOS result. Only after that gate may the
candidate coefficient matrix become trainable.

## Non-goals

- No optimization or new orbital in this phase.
- No LibRPA call in the matrix unit tests.
- No claim of AO completeness from Galerkin/SOS equality.
- No counterpoise correction as a production result.
- No tuning of the rejected projected-Pi alpha.
