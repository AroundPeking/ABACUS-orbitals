"""Single-Bloch-sector response diagnostics; no self-consistent or RPA driver."""

import numpy as np

from response_metric import _hermitian


def occupied_frames(occupied_bra):
    occupied = np.asarray(occupied_bra, dtype=np.complex128)
    if (occupied.ndim != 2 or not np.isfinite(occupied).all()
            or not 0 < occupied.shape[0] < occupied.shape[1]):
        raise ValueError('invalid occupied bra matrix')
    capture = occupied @ occupied.conj().T
    values = np.linalg.eigvalsh(capture)
    if values[0] <= 1e-12*max(values[-1], 1e-300):
        raise ValueError('occupied manifold is rank deficient')
    frame, _ = np.linalg.qr(occupied.conj().T, mode='complete')
    nocc = occupied.shape[0]
    return frame[:, :nocc], frame[:, nocc:], {
        'minimum_capture': float(values[0]),
        'maximum_capture': float(values[-1]),
        'occupied_rank': nocc,
        'virtual_rank': occupied.shape[1]-nocc,
    }


def _frame(value, dimension):
    value = np.asarray(value, dtype=np.complex128)
    if (value.ndim != 2 or value.shape[0] != dimension
            or not 0 < value.shape[1] <= dimension or not np.isfinite(value).all()):
        raise ValueError('invalid trial frame')
    if np.linalg.norm(value.conj().T@value-np.eye(value.shape[1])) > 1e-9:
        raise ValueError('trial frame is not orthonormal')
    return value


def virtual_pod(covariance, virtual_frame):
    cov, _ = _hermitian(covariance)
    virtual = _frame(virtual_frame, len(cov))
    projected, _ = _hermitian(virtual.conj().T@cov@virtual)
    values, vectors = np.linalg.eigh(projected)
    if values[0] < -1e-10*max(values[-1], 1e-300):
        raise ValueError('virtual covariance is materially indefinite')
    return virtual@vectors[:, ::-1], np.maximum(values[::-1], 0.)


def galerkin_pi(hamiltonian, source_bra, eigenvalues, occupations,
                frequencies, virtual_frame, k_weight):
    """Return the Hermitian Pi contribution with k/occupation weights once.

    H, occupied eigenvalues and frequencies must use the same energy unit.
    The supplied virtual frame must already exclude the fixed occupied span.
    Source rows have shape (occupied, perturbation, orthonormal coordinate).
    No q weight, frequency quadrature or extra spin factor is applied here.
    """
    h, _ = _hermitian(hamiltonian)
    frame = _frame(virtual_frame, len(h))
    source = np.asarray(source_bra, dtype=np.complex128)
    eps = np.asarray(eigenvalues, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    freq = np.asarray(frequencies, dtype=float)
    if (source.ndim != 3 or source.shape[2] != len(h) or source.shape[1] == 0
            or eps.ndim != 1 or eps.size == 0 or source.shape[0] != eps.size
            or occ.shape != eps.shape or freq.ndim != 1 or not freq.size
            or not all(np.isfinite(v).all() for v in (source, eps, occ, freq))
            or np.any(occ < 0.) or np.any(freq < 0.)
            or not np.isfinite(k_weight) or k_weight <= 0.):
        raise ValueError('invalid frozen-H response inputs')
    reduced, _ = _hermitian(frame.conj().T@h@frame)
    energies, rotations = np.linalg.eigh(reduced)
    gaps = energies[None, :]-eps[:, None]
    if np.min(gaps) <= 0.:
        raise ValueError('nonpositive virtual transition gap')
    spectral = frame@rotations
    d = source@spectral
    pi = np.zeros((len(freq), source.shape[1], source.shape[1]), dtype=complex)
    for iw, w in enumerate(freq):
        half = np.zeros_like(pi[iw])
        for n, occupation in enumerate(occ):
            half -= k_weight*occupation*(d[n]/(gaps[n]+1j*w))@d[n].conj().T
        pi[iw] = half+half.conj().T
    residual = np.linalg.norm(reduced@rotations-rotations*energies[None, :])
    residual /= max(np.linalg.norm(reduced), 1e-300)
    return pi, {'minimum_gap': float(np.min(gaps)),
                'maximum_gap': float(np.max(gaps)),
                'spectral_residual': float(residual),
                'virtual_rank': frame.shape[1]}


def trace_log(pi):
    """Compute Tr(log(I-Pi)+Pi) with an explicit positive-argument gate."""
    p, residual = _hermitian(pi)
    values = np.linalg.eigvalsh(p)
    argument = 1.-values
    if np.min(argument) <= 0.:
        raise ValueError('trace-log argument is not positive')
    return float(np.sum(np.log1p(-values)+values)), {
        'min_I_minus_Pi': float(np.min(argument)),
        'Pi_min': float(values[0]), 'Pi_max': float(values[-1]),
        'hermitian_residual': residual,
    }
