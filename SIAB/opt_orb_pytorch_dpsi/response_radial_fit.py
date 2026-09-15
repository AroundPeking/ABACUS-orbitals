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


def independent_sector_rank_lower_bound(covariance, available_virtual_rank):
    """Return the best possible residual for one unconstrained sector subspace.

    This Ky Fan bound lets every sector choose its own optimal subspace.  It is
    therefore optimistic relative to any shared, atom-centred contraction and
    diagnoses rank capacity only.
    """
    covariance = np.asarray(covariance, dtype=np.complex128)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not covariance.shape[0] or not np.isfinite(covariance).all()
            or type(available_virtual_rank) is not int
            or available_virtual_rank < 0):
        raise ValueError('invalid covariance or available virtual rank')
    covariance, _ = _hermitian(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance).real
    scale = max(float(eigenvalues[-1]), 1e-300)
    if float(eigenvalues[0]) < -1e-10*scale:
        raise ValueError('response covariance is materially indefinite')
    eigenvalues = np.maximum(eigenvalues, 0.)
    target = math.fsum(float(value) for value in eigenvalues)
    if not math.isfinite(target) or target <= 0:
        raise ValueError('response rank bound requires nonzero target norm')
    rank = min(available_virtual_rank, len(eigenvalues))
    discarded = eigenvalues[:len(eigenvalues)-rank]
    residual = math.fsum(float(value) for value in discarded)
    captured = target-residual
    return dict(scope='independent_sector_rank_only_lower_bound',
        covariance_dimension=len(eigenvalues), available_virtual_rank=rank,
        target_norm2=target, captured_norm2_upper_bound=captured,
        residual_norm2_lower_bound=residual,
        response_loss_lower_bound=residual/target)


class ResponseFitSector:
    def __init__(self, label, primitive_blocks, virtual_embedding, covariance, *,
                 occupied_rank, occupied_embedding=None):
        embedding = np.asarray(virtual_embedding, dtype=np.complex128)
        cov, _ = _hermitian(covariance)
        if (not isinstance(label, str) or not label or embedding.ndim != 2
                or not np.isfinite(embedding).all() or embedding.shape[0] != len(cov)
                or type(occupied_rank) is not int or occupied_rank < 0
                or embedding.shape[0]+occupied_rank > embedding.shape[1]):
            raise ValueError('invalid fixed response sector')
        occupied = None
        if occupied_embedding is not None:
            occupied = np.asarray(occupied_embedding, dtype=np.complex128)
            if (occupied.ndim != 2 or occupied.shape != (occupied_rank, embedding.shape[1])
                    or not np.isfinite(occupied).all()):
                raise ValueError('invalid occupied embedding')
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[0] < -1e-10*max(float(eigenvalues[-1]), 1e-300):
            raise ValueError('response covariance is materially indefinite')
        self.label = label
        self.primitive_blocks = primitive_blocks
        self.embedding = torch.tensor(embedding, dtype=torch.complex128)
        self.occupied_embedding = (None if occupied is None else
                                   torch.tensor(occupied, dtype=torch.complex128))
        self.covariance = torch.tensor(cov, dtype=torch.complex128)
        self.target_norm2 = float(np.trace(cov).real)
        self.occupied_rank = occupied_rank


def shared_radial_fit_loss(coefficients, sectors, *, relative_singular_tolerance=1e-10,
                           occupied_weight=0.):
    """Return normalized snapshot residual and explicit per-sector rank reports.

    The *same* element/l radial tensors generate all atom/m columns in every
    sector through the existing SIAB mapper.  When present, occupied and
    virtual embeddings share one retained mother metric and provide an explicit
    occupied-capture residual. q/k/occupation/frequency weights belong in the
    fixed covariance, once.
    """
    if (not math.isfinite(relative_singular_tolerance)
            or not 0 < relative_singular_tolerance < 1
            or not math.isfinite(occupied_weight) or occupied_weight < 0
            or not sectors or len({s.label for s in sectors}) != len(sectors)):
        raise ValueError('invalid shared fit sectors or singular tolerance')
    target_norm2 = math.fsum(s.target_norm2 for s in sectors)
    if not math.isfinite(target_norm2) or target_norm2 <= 0:
        raise ValueError('response fit requires nonzero total target norm')
    residuals, occupied_residuals, occupied_captures, records = [], [], [], []
    for sector in sectors:
        available = {(block.element, block.l) for block in sector.primitive_blocks}
        requested = {(element, l) for element, channels in coefficients.items()
                     for l, channel in enumerate(channels) if channel.shape[1] > 0}
        if not requested <= available:
            raise ValueError('new angular channels have missing primitive blocks')
        basis = build_primitive_to_candidate(sector.primitive_blocks,
                                             sector.embedding.shape[1], coefficients)
        y = sector.embedding@basis.transform
        # The occupied projection gives y exact null columns.  Differentiating
        # torch.linalg.pinv through those zero singular values is undefined.
        # A detached right-singular basis selects a locally full-rank set of
        # column combinations without changing span(y); the projector and its
        # derivative can then be evaluated with an ordinary Gram solve.
        _, values, right_adjoint = torch.linalg.svd(y.detach(), full_matrices=False)
        keep = values > relative_singular_tolerance*values[0]
        rank = int(keep.sum())
        if not rank:
            raise ValueError('candidate has no represented virtual direction')
        independent = right_adjoint[keep].conj().T
        reduced = y@independent
        gram_virtual = reduced.conj().T@reduced
        covariance_virtual = reduced.conj().T@sector.covariance@reduced
        captured = torch.trace(torch.linalg.solve(
            gram_virtual, covariance_virtual)).real
        residual = sector.target_norm2-captured
        if not bool(torch.isfinite(residual)) or float(residual.detach()) < -1e-8*target_norm2:
            raise ValueError('invalid projected response residual')
        residuals.append(residual)
        minimum_capture = None
        occupied_residual = None
        if sector.occupied_embedding is not None:
            occupied_y = sector.occupied_embedding@basis.transform
            full_y = torch.cat((occupied_y, y), dim=0)
            gram = full_y.conj().T@full_y
            values_full = torch.linalg.eigvalsh(gram.detach())
            maximum_full = values_full[-1].real
            keep_full = values_full.real > relative_singular_tolerance*maximum_full
            if int(keep_full.sum()) != gram.shape[0]:
                raise ValueError('candidate is rank deficient in retained mother metric')
            capture_matrix = occupied_y@torch.linalg.solve(
                gram, occupied_y.conj().T)
            capture_matrix = 0.5*(capture_matrix+capture_matrix.conj().T)
            capture_values = torch.linalg.eigvalsh(capture_matrix).real
            minimum_capture = float(capture_values[0].detach())
            occupied_residual = (sector.occupied_rank-torch.trace(capture_matrix).real)
            if (not bool(torch.isfinite(occupied_residual))
                    or float(occupied_residual.detach()) < -1e-8*max(sector.occupied_rank, 1)):
                raise ValueError('invalid occupied capture residual')
            occupied_residuals.append(torch.clamp(occupied_residual, min=0.))
            occupied_captures.append(minimum_capture)
        records.append(dict(label=sector.label, nominal_cell_ao=len(basis.columns),
            retained_virtual_rank=rank, occupied_rank=sector.occupied_rank,
            augmented_total_rank=rank+sector.occupied_rank,
            retained_singular_condition=float(values[0]/values[keep][-1]),
            discarded_virtual_columns=len(basis.columns)-rank,
            minimum_occupied_capture=minimum_capture,
            occupied_residual=(None if occupied_residual is None else
                               float(occupied_residual.detach())),
            residual_norm2=float(residual.detach()), target_norm2=sector.target_norm2))
    response_loss = torch.stack(residuals).sum()/target_norm2
    occupied_rank_total = sum(s.occupied_rank for s in sectors
                              if s.occupied_embedding is not None)
    occupied_loss = (torch.stack(occupied_residuals).sum()/occupied_rank_total
                     if occupied_residuals else response_loss.new_zeros(()))
    loss = response_loss+occupied_weight*occupied_loss
    return loss, dict(stage='shared_radial_snapshot_fit_not_RPA_acceptance',
        physical_release_gate='hold', target_norm2=target_norm2,
        response_loss=float(response_loss.detach()),
        occupied_weight=occupied_weight,
        occupied_constraint_sector_count=len(occupied_residuals),
        occupied_residual=float(occupied_loss.detach()),
        minimum_occupied_capture=(min(occupied_captures) if occupied_captures else None),
        relative_singular_tolerance=relative_singular_tolerance,
        radial_orbital_count=sum(c.shape[1] for channels in coefficients.values() for c in channels),
        nominal_ao_per_element={element: sum((2*l+1)*c.shape[1] for l, c in enumerate(channels))
                                for element, channels in coefficients.items()},
        virtual_rank_signature=[r['retained_virtual_rank'] for r in records],
        augmented_total_rank_by_sector=[r['augmented_total_rank'] for r in records],
        sectors=records)
