"""Cheap frozen-Hamiltonian screening; never a self-consistent PBE gate."""

import math

import torch

from periodic_galerkin_basis import contract_periodic_candidate_operators
from periodic_galerkin_fit import CandidateGuardError

HARTREE_TO_EV = 27.211386245988


def prepare_frozen_band_guard(
    dataset, initial, *, atoms_per_cell=2,
    occupied_band_sum_limit_ev_per_atom=0.010,
    maximum_target_band_change_ev=0.050,
    extra_virtual_bands=4,
    allow_basis_expansion=False,
):
    """Freeze a full-k initial spectrum and protect occupied/near-edge bands.

    Band sums are not DFT total energies. These provisional screening limits
    must be followed by a same-candidate SCF and actual total-energy check.
    Diagonalization here is diagnostic only and does not replace the fixed
    occupied/source problem used by the response evaluator.
    """
    if type(atoms_per_cell) is not int or atoms_per_cell < 1:
        raise ValueError('atoms_per_cell must be a positive integer')
    if type(extra_virtual_bands) is not int or extra_virtual_bands < 0:
        raise ValueError('extra_virtual_bands must be a nonnegative integer')
    if type(allow_basis_expansion) is not bool:
        raise ValueError('allow_basis_expansion must be a boolean')
    for value in (occupied_band_sum_limit_ev_per_atom, maximum_target_band_change_ev):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('band screening limits must be finite and positive')
    weight_sum = math.fsum(r.k_weight for r in dataset.kpoints)
    # The production reader includes spin degeneracy in ABACUS k weights.
    if not dataset.kpoints or not math.isclose(weight_sum, 2., rel_tol=0., abs_tol=1e-10):
        raise ValueError('ABACUS spin-included k weights must sum to 2')

    def spectra(coefficients):
        result = []
        with torch.no_grad():
            for record in dataset.kpoints:
                op = contract_periodic_candidate_operators(
                    record, dataset.primitive_blocks, coefficients)
                identity = torch.eye(op.overlap.shape[0], dtype=torch.complex128)
                cholesky = torch.linalg.cholesky(op.overlap)
                transform = torch.linalg.solve(cholesky.T.conj(), identity)
                h = transform.T.conj().matmul(op.hamiltonian_ha).matmul(transform)
                values = torch.linalg.eigvalsh((h+h.T.conj())*.5)
                if not bool(torch.isfinite(values).all()):
                    raise CandidateGuardError('nonfinite frozen spectrum')
                result.append(values)
        return result

    reference = spectra(initial)

    def guard(coefficients, *, diagnostics=False, include_unprotected_bands=False,
              enforce_accuracy=True):
        if type(diagnostics) is not bool:
            raise ValueError('diagnostics must be a boolean')
        if type(include_unprotected_bands) is not bool or (include_unprotected_bands and not diagnostics):
            raise ValueError('unprotected-band output requires explicit diagnostics')
        if type(enforce_accuracy) is not bool:
            raise ValueError('enforce_accuracy must be a boolean')
        values = spectra(coefficients)
        changes, maximum = [], 0.
        details = []
        unprotected = []
        added = []
        vbm, cbm = -math.inf, math.inf
        for record, before, after in zip(dataset.kpoints, reference, values):
            nocc = record.occupation.numel()
            if (after.numel() <= nocc or after.numel() < before.numel()
                    or (before.shape != after.shape and not allow_basis_expansion)):
                raise CandidateGuardError('frozen spectrum band count changed')
            change = (after[:before.numel()]-before)*HARTREE_TO_EV
            changes.append(record.k_weight*float(torch.dot(change[:nocc], record.occupation)))
            maximum = max(maximum, float(change[:nocc+extra_virtual_bands].abs().max()))
            if diagnostics:
                count = min(before.numel(), nocc+extra_virtual_bands)
                details.append(dict(source_ik=record.source_ik, target_ik=record.target_ik,
                    band_indices=list(range(1, count+1)),
                    signed_changes_ev=change[:count].tolist()))
                if include_unprotected_bands:
                    unprotected.append(dict(source_ik=record.source_ik, target_ik=record.target_ik,
                        band_indices=list(range(count+1, before.numel()+1)),
                        signed_changes_ev=change[count:].tolist()))
                    if allow_basis_expansion:
                        added.append(dict(source_ik=record.source_ik, target_ik=record.target_ik,
                            band_indices=list(range(before.numel()+1,after.numel()+1)),
                            energies_ev=(after[before.numel():]*HARTREE_TO_EV).tolist()))
            vbm = max(vbm, float(after[nocc-1])*HARTREE_TO_EV)
            cbm = min(cbm, float(after[nocc])*HARTREE_TO_EV)
        band_sum = math.fsum(changes)/atoms_per_cell
        minimum_gap = cbm-vbm
        result = dict(gate=abs(band_sum) <= occupied_band_sum_limit_ev_per_atom
                      and maximum <= maximum_target_band_change_ev and minimum_gap > 0.,
                      scope='frozen_h_band_screen_not_scf_energy', scf_pbe_gate='pending',
                      occupied_band_sum_change_ev_per_atom=band_sum,
                      maximum_target_band_change_ev=maximum, minimum_gap_ev=minimum_gap,
                      occupied_band_sum_limit_ev_per_atom=occupied_band_sum_limit_ev_per_atom,
                      target_band_change_limit_ev=maximum_target_band_change_ev,
                      k_weight_sum=weight_sum,
                      k_weight_convention='ABACUS_spin_included_no_renormalization')
        if (not all(math.isfinite(x) for x in (band_sum,maximum,minimum_gap))
                or minimum_gap <= 0.):
            raise CandidateGuardError('nonfinite frozen spectrum or nonpositive indirect gap')
        # Free-energy diagnostics report accuracy drift without hiding the
        # original gate result; numerical admissibility remains mandatory.
        if enforce_accuracy and not result['gate']:
            raise CandidateGuardError('frozen occupied/target-band screen rejected candidate')
        if diagnostics:
            result.update(target_band_details=details,
                target_band_identity='sorted_eigenvalue_index_not_eigenvector_tracking')
            if include_unprotected_bands:
                result.update(unprotected_band_details=unprotected,
                    maximum_unprotected_band_change_ev=max(
                        (abs(x) for row in unprotected for x in row['signed_changes_ev']), default=0.),
                    unprotected_band_scope='diagnostic_not_rejection_threshold')
                if allow_basis_expansion:
                    result.update(added_band_details=added,
                        added_band_scope='no_original_counterpart_not_a_band_error')
        return result

    return guard
