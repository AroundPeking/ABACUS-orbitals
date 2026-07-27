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
`c273b4ee7051138293d9988c3eb79bee36c0af10` and executable SHA256
`ff38348fbad89fde4a985c13f97b59ffc94353c22c7098e19b373c1ef7e76fee`.
That build passed the fixed-ABS, global full-Coulomb whitening, reciprocal
primitive, writer, provenance, memory-gate, and OpenMP response-kernel tests.
The atom and H2 physical targets must pass `validate_targets.py` before shell
selection. The H2 fragment/ghost producer is a separate post-selection control;
it does not gate or influence shell selection.

`response_selection_campaign.py` is the bridge between the physical producer
files and the nested selector. It requires exactly the physical `atom` and
`multicenter` families, and round-trips SIAB's native `ORBITAL_RESULTS.txt`
coefficient block without inventing missing columns. The three producer `STRU`
files are deliberately force-tracked even
though the repository-wide ignore rules match `*STRU`; every immutable runtime
closure must include `INPUT`, `KPT`, and `STRU` for each producer.

`run_response_selection.py` executes the frozen nested loop. Atom and H2
physical targets alone define the radial residual spectra, candidate score, and
stopping conditions. The score is the atom-plus-H2 normalized loss reduction
divided by the AO cost `2*l+1`. A selected shell is initialized from the leading
residual mode, optimized with the existing `st_dpsi_joint` path,
read back through the native coefficient bridge, and followed by a rebuilt
spectrum. A fully spanned angular channel is represented as a zero-residual
spectrum so that it is rejected deterministically while other channels remain
eligible. The optimizer template uses the independently generated H3
`lmax=2` origin/dpsi matrices, so DFT/dpsi supervision covers s, p, and d;
higher angular channels are response-driven while the DZP columns remain
bitwise fixed. The sequence manifest is frozen before any held-out energy is
evaluated.
