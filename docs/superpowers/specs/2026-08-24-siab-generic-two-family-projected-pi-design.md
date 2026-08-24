# Generic Two-Family Projected-Pi Design

## Goal

Allow the validated SIAB projected-Pi optimizer to train on any two named
physical response families, including `C_atom` and `C2`, without changing the
existing H/H2 objective or numerical definition.

## Scope

The optimizer continues to require exactly two physical families. Their names
must be nonempty and unique, both entries must carry response-v1, source-v1,
and a matching zero-order audit, and ghost entries remain forbidden. The input
declaration order is the canonical order used in diagnostics and metadata.

Supporting three or more environments, family weights, or unequal frequency
grids would change the training objective and is outside this change.

## Objective

For named families `A` and `B`, the ordinary projected-Pi objective remains

\[
  L_{\mathrm{pair}}=L_A+L_B.
\]

The per-frequency diagnostic remains the equal-family mean

\[
  L_j^{\mathrm{diag}}=\frac{L_{A,j}+L_{B,j}}{2}.
\]

The RPA-sensitive variant retains the fourth-order family norm

\[
  L_{\mathrm{pair}}^{(4)}=(L_A^4+L_B^4)^{1/4}.
\]

Both families must therefore have byte-identical frequency values and weights.
The rank tolerance, condition limit, source-response pairing, zero-order audit,
and no-SOS/no-ghost rules are unchanged.

## Data Flow

`main._load_sternheimer_data` validates the two entries in declaration order,
loads each response/source/audit triplet, and returns named pairs without
rewriting their family names. `NormalizedPhysicalFamilyProjectedPiOptimization`
stores that ordered name tuple and uses it for scalar aggregation, frequency
diagnostics, and metadata. No element alias or orbital coefficient behavior is
changed.

## Acceptance

1. Existing H/H2 tests and numerical values remain unchanged.
2. The same fixtures renamed to `C_atom/C2` produce identical scalar and
   per-frequency results.
3. Reversed declaration order is preserved in output but does not change the
   scalar objective.
4. One family, three families, duplicate names, ghost entries, missing source
   or audit files, and unequal frequency grids remain hard failures.
5. The complete focused projected-Pi and Sternheimer-loader regression suite
   passes before the C producer campaign is prepared.

## C Campaign Boundary

This change only unlocks the interface. It does not create C source-v1 data,
run C2 Delta-ST, optimize a C basis, or authorize SOS-RPA. The first physical
campaign will use `C_atom` and a 1.544 Angstrom C2 dimer in the same 20
Angstrom isolated-cell protocol, with the accepted C 16-frequency file,
SG15/TZDP, FD8, full Coulomb, and PCA threshold `1e-4`.
