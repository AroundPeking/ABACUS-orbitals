"""Contracts for uncontracted jY response targets used by radial compression.

These targets are covariance/embedding snapshots in their local q/k frames.
They are not assembled ``PI.npy`` matrices and do not by themselves admit a
compressed NAO basis for production RPA or GW.
"""

import math


Q_SLOTS = (1, 2, 3, 6, 7, 8, 11, 28)
Q_MULTIPLICITIES = (1, 8, 4, 6, 24, 12, 3, 6)
Q_ARRAY_SLOTS = tuple(range(8))


def validate_target_manifest(manifest, *, lmax=3, frequency_count=12,
                             k_record_count=64, primitive_count=992,
                             q_slots=Q_ARRAY_SLOTS):
    """Validate one complete, uncontracted jY target manifest.

    The checks deliberately reject an energy-only response matrix as a fitting
    target.  Covariance and embedding files must remain separate per q/k
    coordinate frame until a later fitter proves a valid shared gauge.
    """
    if not isinstance(manifest, dict):
        raise ValueError('target manifest must be an object')
    if manifest.get('status') != 'success':
        raise ValueError('target manifest is not successful')
    if manifest.get('target_kind') != 'response_covariance_embedding':
        raise ValueError('response covariance targets required')
    if manifest.get('assembled_pi') is not False:
        raise ValueError('PI matrix cannot replace covariance targets')
    if manifest.get('lmax') != lmax:
        raise ValueError('unexpected target angular cutoff')
    if manifest.get('frequency_count') != frequency_count:
        raise ValueError('frequency contract mismatch')
    if manifest.get('k_record_count') != k_record_count:
        raise ValueError('complete k-sector target set required')
    if manifest.get('primitive_count') != primitive_count:
        raise ValueError('uncontracted mother target required')
    q_slots = tuple(q_slots)
    if not q_slots or any(slot not in Q_ARRAY_SLOTS for slot in q_slots):
        raise ValueError('unexpected q slot contract')
    sectors = manifest.get('sectors')
    if not isinstance(sectors, list) or len(sectors) != len(q_slots)*k_record_count:
        raise ValueError('complete q/k target sectors required')
    q_seen = {slot: 0 for slot in q_slots}
    for sector in sectors:
        if not isinstance(sector, dict):
            raise ValueError('invalid target sector')
        slot = sector.get('q_slot')
        if slot not in q_seen:
            raise ValueError('unexpected q slot')
        q_seen[slot] += 1
        if sector.get('target_kind') != 'response_covariance_embedding':
            raise ValueError('mixed target sector type')
        if sector.get('assembled_pi') is not False:
            raise ValueError('sector PI matrix cannot be a fit target')
        if sector.get('primitive_count') != primitive_count:
            raise ValueError('sector primitive count mismatch')
        for key in ('covariance_dimension', 'embedding_rows', 'embedding_columns'):
            if not isinstance(sector.get(key), int) or sector[key] <= 0:
                raise ValueError('invalid target sector dimensions')
        for key in ('occupied_rank', 'occupied_embedding_rows'):
            if not isinstance(sector.get(key), int) or sector[key] <= 0:
                raise ValueError('invalid occupied embedding dimensions')
        if sector['covariance_dimension'] != sector['embedding_rows']:
            raise ValueError('covariance/embedding frame mismatch')
        if sector['occupied_embedding_rows'] != sector['occupied_rank']:
            raise ValueError('occupied embedding/rank mismatch')
        if sector['embedding_rows']+sector['occupied_embedding_rows'] > sector['embedding_columns']:
            raise ValueError('occupied embedding exceeds retained mother dimension')
        if not math.isfinite(float(sector.get('target_norm2', float('nan')))):
            raise ValueError('nonfinite target norm')
    if q_seen != {slot: k_record_count for slot in q_slots}:
        raise ValueError('q/k target coverage mismatch')
    return True
