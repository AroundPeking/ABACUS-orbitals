# Selected C Basis Solid Binding-Energy Validation Design

## Objective

Validate the selected C `3s3p2d1f3g` orbital by comparing the uncorrected
all-band SOS and matched Delta-Sternheimer RPA@PBE binding energies of diamond.
The solid absolute correlation energy is a diagnostic, not the acceptance
quantity.

For the two-C primitive cell, define the positive binding energy per C atom as

\[
E_{\rm bind}^{X}
= E_{\rm atom}^{X}-\frac{1}{2}E_{\rm diamond}^{X},
\qquad X\in\{\mathrm{SOS},\Delta\mathrm{-ST}\}.
\]

The basis error is

\[
\Delta E_{\rm bind}
=E_{\rm bind}^{\rm SOS}-E_{\rm bind}^{\Delta\mathrm{-ST}}.
\]

Report the total RPA@PBE binding energy and its correlation-only contribution.
Do not apply a counterpoise correction.  The target is that the ordinary raw
SOS result obtained with the optimized orbital reproduces the Delta-ST result.

## Fixed Physical Contract

The selected orbital is
`C_gga_10au_100Ry_selected_product_pca.orb`, with radial multiplicities
`3s3p2d1f3g` and 56 AO per C atom.  Preserve the accepted selected-orbital
SHA256 and require the fixed `2s2p1d` occupied prefix to have passed the held-out
capture gate.

The isolated atom uses the independently accepted discretization:

- neutral spin-polarized C triplet, `N_up=3`, `N_down=1`;
- fixed integer occupations, with 56 bands in each spin channel;
- 20 Angstrom cubic box, centered C atom, Gamma only;
- explicit `135^3` grid and FD8;
- `exx_ccp_rmesh_times=1` and `rpa_ccp_rmesh_times=1`;
- response-aware fixed-prefix product-PCA at `1e-4`;
- full periodic Coulomb for the atom supercell;
- no analytic periodic-solid head/wing replacement.

The old `3s3p2d` atom energies and response matrices are provenance references
only.  They must never be combined with the selected-orbital diamond result.

## Frequency Matching

Each species has its own transition window.  The diamond SOS calculation and
diamond Delta-ST calculation share the exact six-frequency file already
extracted from the selected diamond SOS output.  The selected atom SOS
calculation generates its own six-point GreenX minimax grid; this exact file is
then read byte-for-byte by the atom Delta-ST calculation.  Therefore SOS and
Delta-ST are frequency matched within each term of the binding energy without
forcing the atom to use a grid generated from the solid transition spectrum.

## Execution Stages

1. **Atom producer.** Run selected-orbital fixed-triplet PBE and emit all-band
   reader-v1 data plus the response-aware exact-grid full-Coulomb matrix.  Verify
   integer occupations, 56 bands per spin, 261 auxiliary functions, finite
   `Etot_without_rpa`, and all reader files.
2. **Atom SOS.** Run normal all-band LibRPA with six minimax frequencies.  Require
   a finite body-only `EcRPA`, then extract and hash the exact atom frequency
   grid.
3. **Atom Delta-ST.** Read the atom SOS grid explicitly and solve the full atom
   Delta-ST response with `global_equation` scheduling.  Require all equations
   converged, maximum relative residual at most `1.01e-6`, byte-identical basis
   ordering relative to the atom SOS producer, and a finite LibRPA energy.
4. **Binding-energy collector.** Combine atom and diamond energies only after all
   four SOS/Delta endpoints pass.  Emit both correlation-only and total RPA@PBE
   binding energies in Ha, eV/C, and kcal/mol/C.

The atom stages may run while the existing diamond q1 timing gate waits.  No
existing costly job may be cancelled, migrated, restarted, or duplicated.

## Acceptance

The primary body gate is

\[
\left|\Delta E_{\rm bind,body}\right|
<0.1\ {\rm kcal\,mol^{-1}\,C^{-1}}.
\]

The final solid binding-energy gate uses the shared-head/wing q-averaged diamond
energies and the same Gamma-only atom energies.  Keep this
`basis_full_qavg_gate` separate from the independent
`headwing_convergence_gate`.  The C basis is not publishable until both the atom
and solid endpoint contracts pass and the head/wing convergence decision is
explicit.

## Failure Handling

Every stage writes immutable provenance before execution and a success summary
only after scheduler, SCF, occupation, response, reader-v1, Coulomb, and LibRPA
checks pass.  A missing atom term, a finite but implausible absolute energy, or a
solid-only SOS/Delta match is not a binding-energy result.  Preserve failed
outputs for diagnosis and do not retry automatically.
