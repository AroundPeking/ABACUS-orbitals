"""Shared NAO radial fitting to fixed, metric-correct response covariances.

This fits snapshots, not RPA energy. Covariances and virtual embeddings must
already share the same orthonormal coordinates within each Bloch sector.
No frequency solve, cross-sector coordinate merge or reference recovery is
performed here. Rank-changing steps require a separate optimizer decision.
"""

import math

import numpy as np
import torch

from periodic_galerkin_basis import build_primitive_to_candidate
from response_metric import _hermitian


class ResponseFitSector:
    def __init__(self, label, primitive_blocks, virtual_embedding, covariance, *, occupied_rank):
        embedding = np.asarray(virtual_embedding, dtype=np.complex128)
        cov, _ = _hermitian(covariance)
        if (not isinstance(label, str) or not label or embedding.ndim != 2
                or not np.isfinite(embedding).all() or embedding.shape[0] != len(cov)
                or type(occupied_rank) is not int or occupied_rank < 0
                or embedding.shape[0]+occupied_rank > embedding.shape[1]):
            raise ValueError('invalid fixed response sector')
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[0] < -1e-10*max(float(eigenvalues[-1]), 1e-300):
            raise ValueError('response covariance is materially indefinite')
        self.label = label
        self.primitive_blocks = primitive_blocks
        self.embedding = torch.tensor(embedding, dtype=torch.complex128)
        self.covariance = torch.tensor(cov, dtype=torch.complex128)
        self.target_norm2 = float(np.trace(cov).real)
        self.occupied_rank = occupied_rank


def shared_radial_fit_loss(coefficients, sectors, *, relative_singular_tolerance=1e-10):
    """Return normalized snapshot residual and explicit per-sector rank reports.

    The *same* element/l radial tensors generate all atom/m columns in every
    sector through the existing SIAB mapper. Occupied directions are external
    to this virtual fit and are counted separately, never called free NAOs.
    q/k/occupation/frequency weights belong in the fixed covariance, once.
    """
    if (not math.isfinite(relative_singular_tolerance)
            or not 0 < relative_singular_tolerance < 1
            or not sectors or len({s.label for s in sectors}) != len(sectors)):
        raise ValueError('invalid shared fit sectors or singular tolerance')
    target_norm2 = math.fsum(s.target_norm2 for s in sectors)
    if not math.isfinite(target_norm2) or target_norm2 <= 0:
        raise ValueError('response fit requires nonzero total target norm')
    residuals, records = [], []
    for sector in sectors:
        available = {(block.element, block.l) for block in sector.primitive_blocks}
        requested = {(element, l) for element, channels in coefficients.items()
                     for l, channel in enumerate(channels) if channel.shape[1] > 0}
        if not requested <= available:
            raise ValueError('new angular channels have missing primitive blocks')
        basis = build_primitive_to_candidate(sector.primitive_blocks,
                                             sector.embedding.shape[1], coefficients)
        y = sector.embedding@basis.transform
        values = torch.linalg.svdvals(y).detach()
        keep = values > relative_singular_tolerance*values[0]
        rank = int(keep.sum())
        if not rank:
            raise ValueError('candidate has no represented virtual direction')
        # A pseudoinverse handles the exact null columns introduced by the
        # occupied projection without squaring Y's condition number.
        inverse = torch.linalg.pinv(y, rcond=relative_singular_tolerance)
        captured = torch.trace(inverse@sector.covariance@y).real
        residual = sector.target_norm2-captured
        if not bool(torch.isfinite(residual)) or float(residual.detach()) < -1e-8*target_norm2:
            raise ValueError('invalid projected response residual')
        residuals.append(residual)
        records.append(dict(label=sector.label, nominal_cell_ao=len(basis.columns),
            retained_virtual_rank=rank, occupied_rank=sector.occupied_rank,
            augmented_total_rank=rank+sector.occupied_rank,
            retained_singular_condition=float(values[0]/values[keep][-1]),
            discarded_virtual_columns=len(basis.columns)-rank,
            residual_norm2=float(residual.detach()), target_norm2=sector.target_norm2))
    loss = torch.stack(residuals).sum()/target_norm2
    return loss, dict(stage='shared_radial_snapshot_fit_not_RPA_acceptance',
        physical_release_gate='hold', target_norm2=target_norm2,
        relative_singular_tolerance=relative_singular_tolerance,
        radial_orbital_count=sum(c.shape[1] for channels in coefficients.values() for c in channels),
        nominal_ao_per_element={element: sum((2*l+1)*c.shape[1] for l, c in enumerate(channels))
                                for element, channels in coefficients.items()},
        virtual_rank_signature=[r['retained_virtual_rank'] for r in records],
        augmented_total_rank_by_sector=[r['augmented_total_rank'] for r in records],
        sectors=records)
