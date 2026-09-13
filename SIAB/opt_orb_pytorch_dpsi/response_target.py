"""Fixed available-mother response covariances for shared NAO radial fitting.

Not exact Delta-ST snapshots. The two resolvents use the same frozen record,
not a guessed spatial q symmetry. Their average weights a state-fitting norm;
the physical Pi keeps production's sum of a half-response and its adjoint.
"""

import numpy as np

from periodic_available_mother import available_mother_response, _array
from response_metric import _hermitian


def spectral_response_target(hamiltonian, source_bra, eigenvalues, occupations,
                             frequencies, frequency_weights, *, k_weight, q_weight):
    h, _ = _hermitian(hamiltonian)
    source = np.asarray(source_bra, dtype=np.complex128)
    eps = np.asarray(eigenvalues, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    freq = np.asarray(frequencies, dtype=float)
    weights = np.asarray(frequency_weights, dtype=float)
    if (freq.ndim != 1 or not freq.size or weights.shape != freq.shape
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isfinite(k_weight) or k_weight <= 0
            or not np.isfinite(q_weight) or not 0 < q_weight <= 1):
        raise ValueError('invalid response target quadrature weight')
    if (source.ndim != 3 or source.shape[2] != len(h) or not source.shape[1]
            or eps.ndim != 1 or not eps.size or source.shape[0] != eps.size
            or occ.shape != eps.shape or np.any(occ < 0) or np.any(freq < 0)
            or not all(np.isfinite(v).all() for v in (source, eps, occ, freq))):
        raise ValueError('invalid frozen response target inputs')
    energies, rotations = np.linalg.eigh(h)
    gaps = energies[None, :]-eps[:, None]
    if gaps.min() <= 0:
        raise ValueError('nonpositive virtual transition gap')
    d = source@rotations
    covariance_spectral = np.zeros_like(h)
    pi = np.zeros((len(freq), source.shape[1], source.shape[1]), dtype=np.complex128)
    for n, occupation in enumerate(occ):
        # Both complete resolvents are retained. Averaging their outer products
        # produces this real factor; it is not an additional spin factor.
        reciprocal = 1./(gaps[n][None, :]+1j*freq[:, None])
        factors = (reciprocal.T@(weights[:, None]*reciprocal.conj())).real
        gram = d[n].conj().T@d[n]
        covariance_spectral += q_weight*k_weight*occupation*gram*factors
        for iw in range(len(freq)):
            half = -k_weight*occupation*(d[n]*reciprocal[iw])@d[n].conj().T
            pi[iw] += half+half.conj().T
    covariance = rotations@covariance_spectral@rotations.conj().T
    covariance, residual = _hermitian(covariance)
    if not np.isfinite(pi).all() or not np.isfinite(covariance).all():
        raise ValueError('nonfinite response target')
    return covariance, pi, dict(exact_reference_snapshot_fit=False,
        physical_release_gate='hold', branches='same_record_plus_minus_resolvent_average',
        covariance_weight='q_weight*k_weight*occupation*frequency_weight/2_per_branch',
        pi_weight='k_weight*occupation; plus_and_adjoint; no_extra_spin_q_or_frequency',
        minimum_gap=float(gaps.min()), maximum_gap=float(gaps.max()),
        target_norm2=float(np.trace(covariance).real),
        covariance_hermitian_residual=residual)


def build_available_response_targets(dataset, consume, *, relative_rank_tolerance=1e-10,
                                     progress=None):
    """Stream one complete target per q/source-k/target-k coordinate frame.

    Keeping records separate avoids any unproven gauge merge. Sharing NAO
    radial coefficients across these records is algebraically equivalent to
    summing their losses after a valid within-target-k coordinate alignment.
    No target truncation or POD rank cap is introduced.
    """
    def target_consumer(record, h, source, embedding, expected_pi, metadata):
        covariance, pi, details = spectral_response_target(h, source,
            _array(record.source_eigenvalue_ha), _array(record.occupation),
            _array(dataset.frequency_ha), _array(dataset.frequency_weights_ha),
            k_weight=record.k_weight, q_weight=dataset.q_weight)
        error = float(np.linalg.norm(pi-expected_pi)/max(np.linalg.norm(expected_pi), 1e-300))
        if error > 1e-11:
            raise ValueError('response target does not reproduce available-mother Pi')
        metadata.update(details, selected_iq=dataset.selected_iq, q_weight=dataset.q_weight,
            coordinate_merge='none_per_q_source_target_record',
            embedding_definition='V_dagger_Lowdin_dagger_S',
            pi_reconstruction_relative_error=error,
            response_equation='(H_virtual-eps_n+i*sign*omega)X=-source_virtual_dagger')
        consume(covariance, embedding, pi, metadata)

    return available_mother_response(dataset, relative_rank_tolerance=relative_rank_tolerance,
                                    progress=progress, target_consumer=target_consumer)
