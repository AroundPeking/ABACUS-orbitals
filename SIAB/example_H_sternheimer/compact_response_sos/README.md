# Compact response-basis SOS-RPA gate

This post-selection campaign evaluates the three frozen 48 AO/H response
frontiers with matched H2/H/H+ghost calculations. The task layout is
`tail_0p00`, `tail_0p10`, and `tail_0p30`, with all three physical cases for
each lane. The corresponding all-band counts are 96/48/96.

The calculation retains the accepted 20-Angstrom cell, 0.74085-Angstrom H2
bond, 100-Ry LCAO grid, 16 minimax frequencies, full Coulomb, and the explicit
214-function-per-H auxiliary basis. It intentionally uses the historical
ABACUS and LibRPA binaries from the accepted comparison so the basis is the
only changed producer input.

H and H2 enter the response-space training loss. Their SOS energies therefore
measure whether the optimized space reaches the intended result, but are not
an independent held-out selector. H+ghost remains a post-selection
counterpoise diagnostic and does not feed back into shell selection,
coefficients, or the 48-AO budget.
