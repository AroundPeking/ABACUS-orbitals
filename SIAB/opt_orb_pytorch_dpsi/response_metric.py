"""Metric-correct compression of bra projections, not physical RPA energies."""

import numpy as np


def _hermitian(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not matrix.size or not np.isfinite(matrix).all()):
        raise ValueError('matrix must be finite, square and nonempty')
    scale = float(np.linalg.norm(matrix))
    residual = float(np.linalg.norm(matrix - matrix.conj().T) / max(scale, 1e-300))
    if residual > 1e-10:
        raise ValueError('matrix is not Hermitian')
    return .5*(matrix + matrix.conj().T), residual


class Metric:
    def __init__(self, overlap):
        self.overlap, self.hermitian_residual = _hermitian(overlap)
        self.values, self.vectors = np.linalg.eigh(self.overlap)
        if self.values[-1] <= 0 or self.values[0] < -1e-10*self.values[-1]:
            raise ValueError('overlap has no usable positive-semidefinite metric')

    def mask(self, rtol):
        if not np.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError('relative cutoff must be strictly between zero and one')
        return self.values > rtol*self.values[-1]

    def restore(self, bra_projection, rtol):
        q = np.asarray(bra_projection, dtype=np.complex128)
        if q.ndim != 2 or q.shape[1] != len(self.values) or not np.isfinite(q).all():
            raise ValueError('invalid bra projection')
        keep = self.mask(rtol)
        u, s = self.vectors[:, keep], self.values[keep]
        dual = u.conj().T @ q.conj().T
        x = dual / np.sqrt(s[:, None])
        c = u @ (dual/s[:, None])
        reconstructed = self.overlap @ c
        residual = np.linalg.norm(reconstructed - q.conj().T)/max(np.linalg.norm(q), 1e-300)
        return x, c, {'retained_rank': int(keep.sum()),
                      'projection_relative_residual': float(residual),
                      'retained_condition': float(s[-1]/s[0])}


def covariance(coordinates, weights):
    x = np.asarray(coordinates, dtype=np.complex128)
    weights = np.asarray(weights, dtype=float)
    if (x.ndim != 2 or weights.shape != (x.shape[1],)
            or not np.isfinite(x).all() or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError('invalid coordinates or snapshot weights')
    scaled = x * np.sqrt(weights[None, :])
    return scaled @ scaled.conj().T


def spectrum(cov):
    cov, hermitian_residual = _hermitian(cov)
    raw = np.linalg.eigvalsh(cov)[::-1]
    if raw[-1] < -1e-10*max(raw[0], 1e-300):
        raise ValueError('covariance is materially indefinite')
    eigenvalues = np.maximum(raw, 0.)
    total = float(eigenvalues.sum())
    tail = np.r_[np.cumsum(eigenvalues[::-1])[::-1], 0.]
    errors = np.sqrt(tail/max(total, 1e-300))
    tolerances = [.1, .01, .001, .0001]
    return {'eigenvalues': eigenvalues.tolist(), 'weighted_projected_norm2': total,
            'raw_min_eigenvalue': float(raw[-1]),
            'hermitian_residual': hermitian_residual,
            'relative_rms_by_rank': errors.tolist(),
            'rank_for_relative_rms': {str(t): int(np.flatnonzero(errors <= t)[0])
                                      for t in tolerances}}
