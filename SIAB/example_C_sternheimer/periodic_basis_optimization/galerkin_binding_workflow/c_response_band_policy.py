"""Explicit C band policy; preserve legacy results and original PBE protection."""

from periodic_galerkin_pbe_guard import prepare_frozen_band_guard

LEGACY = 'legacy_four_virtual'
RESPONSE = 'occupied_plus_two_virtual_v1'


def validate_band_policy(policy):
    if type(policy) is not str or policy not in (LEGACY, RESPONSE):
        raise ValueError('unknown C band guard policy')


def prepare_c_band_guard(dataset, initial, *, policy=LEGACY):
    validate_band_policy(policy)
    base = prepare_frozen_band_guard(dataset, initial, atoms_per_cell=2,
                                    extra_virtual_bands=4 if policy == LEGACY else 2)
    if policy == LEGACY:
        return base

    def guard(coefficients, *, diagnostics=False):
        result = base(coefficients, diagnostics=diagnostics,
                      include_unprotected_bands=diagnostics)
        result.update(band_guard_policy=RESPONSE, protected_virtual_bands=2,
                      first_two_virtual_accuracy='provisional_not_reference_validated',
                      higher_virtual_policy='diagnostic_only',
                      reference_policy='original_TZDP_not_reset_center',
                      actual_pbe_limit_ev_per_c=0.010)
        return result

    return guard
