# Greedy H response-shell selection

This directory freezes the H response-basis sequence before any post-selection
H2 RPA energy is used for acceptance. The fixed DFT core is `1s,2s,1p`; each accepted step adds exactly one
shared radial shell with AO cost `2*l+1`.

`selection_config.json` fixes the rank, magnetic-overlap, stopping, and random
seed thresholds. `select_response_shells.py` writes canonical coefficient and
selection records, checks the fixed columns bitwise, and invokes the existing
joint Sternheimer+dpsi optimizer as a checked subprocess. Selector inputs and
manifests reject H2/RPA energy fields.

The target runner is pinned to ABACUS commit
`f9c44f49e0dedaeb6cb7cd0aa886cd924b280c9a` and executable SHA256
`76e8ecfeba19a907507c24da5428d776bf34219f81319b574227f10772d96746`.
That build passed the fixed-ABS, global full-Coulomb whitening, reciprocal
primitive, writer, provenance, memory-gate, and OpenMP response-kernel tests.
The physical target producers remain a separate gate: atom, H2, and H2
fragment/ghost outputs must all
pass `validate_targets.py` before shell selection or any H2 acceptance result is
allowed.

`response_selection_campaign.py` is the bridge between the physical producer
files and the nested selector. It requires exactly the `atom`, `multicenter`,
and `fragment_ghost` families with their frozen roles, and round-trips SIAB's
native `ORBITAL_RESULTS.txt` coefficient block without inventing missing
columns. The three producer `STRU` files are deliberately force-tracked even
though the repository-wide ignore rules match `*STRU`; every immutable runtime
closure must include `INPUT`, `KPT`, and `STRU` for each producer.

`run_response_selection.py` executes the frozen nested loop. Atom and H2
physical targets define the radial residual spectra; the fragment/ghost target
enters only the borrowing-balance score. A selected shell is initialized from
the leading residual mode, optimized with the existing `st_dpsi_joint` path,
read back through the native coefficient bridge, and followed by a rebuilt
spectrum. A fully spanned angular channel is represented as a zero-residual
spectrum so that it is rejected deterministically while other channels remain
eligible. The optimizer template uses the independently generated H3
`lmax=2` origin/dpsi matrices, so DFT/dpsi supervision covers s, p, and d;
higher angular channels are response-driven while the DZP columns remain
bitwise fixed. The sequence manifest is frozen before any held-out energy is
evaluated.
