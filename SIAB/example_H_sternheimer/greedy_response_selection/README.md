# Greedy H response-shell selection

This directory freezes the H response-basis sequence before any H2 energy is
evaluated. The fixed DFT core is `1s,2s,1p`; each accepted step adds exactly one
shared radial shell with AO cost `2*l+1`.

`selection_config.json` fixes the rank, magnetic-overlap, stopping, and random
seed thresholds. `select_response_shells.py` writes canonical coefficient and
selection records, checks the fixed columns bitwise, and invokes the existing
joint Sternheimer+dpsi optimizer as a checked subprocess. Selector inputs and
manifests reject H2/RPA energy fields.

The physical target producers and the production nested-sequence command are
added only after the `l<=4` atom, H3, and fragment/ghost files pass their own
contract checks. Until then this directory defines and tests the immutable file
and optimizer boundary; it does not contain an H2 acceptance result.
