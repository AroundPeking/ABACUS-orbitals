"""Strict canonical q-star assembly helpers for the C jY workflow."""

import math


_CONTRACT = (
    (1, 1, 1), (2, 22, 8), (3, 43, 4), (6, 6, 6),
    (7, 27, 24), (8, 23, 12), (11, 11, 3), (28, 55, 6),
)


def _same_grid(left, right):
    if len(left) != 12 or len(right) != 12 or any(
            not math.isfinite(float(a)) or float(a) != float(b)
            for a, b in zip(left, right)):
        raise ValueError('frequency grid mismatch')


def canonical_q_contract(records):
    keys = [(int(row['label']), int(row['selected_iq'])) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate q record')
    if len(records) != len(_CONTRACT):
        raise ValueError('complete canonical q contract required')
    expected = [(label, iq) for label, iq, _ in _CONTRACT]
    if set(keys) != set(expected):
        raise ValueError('complete canonical q contract required')
    return True


def join_q_records(records):
    canonical_q_contract(records)
    by_key = {(int(row['label']), int(row['selected_iq'])): row for row in records}
    reference = by_key[(1, 1)]
    joined = []
    for label, iq, multiplicity in _CONTRACT:
        row = dict(by_key[(label, iq)])
        if row['multiplicity'] != multiplicity:
            raise ValueError('canonical q multiplicity mismatch')
        _same_grid(reference['frequencies'], row['frequencies'])
        _same_grid(reference['weights'], row['weights'])
        row['q_weight'] = multiplicity / 64.0
        joined.append(row)
    return joined


def collect_energies(records, reference=-.5144842779929139):
    rows = join_q_records(records)
    for original, row in zip(sorted(records,key=lambda r:r['label']),rows):
        if (original['q_weight'] != row['q_weight'] or row['status'] != 'success'
                or row['lmax'] != rows[0]['lmax']
                or row['relative_rank_tolerance'] != rows[0]['relative_rank_tolerance']
                or not all(math.isfinite(row[k]) for k in
                           ('candidate_energy_ha','reference_energy_ha'))):
            raise ValueError('mixed or failed angular energy protocol')
    candidate = math.fsum(r['candidate_energy_ha'] for r in rows)
    actual_reference = math.fsum(r['reference_energy_ha'] for r in rows)
    if not math.isfinite(reference) or abs(actual_reference-reference) > 1e-12:
        raise ValueError('full-q Delta-ST reference not recovered')
    error = (candidate-reference)*27.211386245988/2
    return dict(status='success',candidate_energy_ha=candidate,reference_energy_ha=reference,
        signed_error_ev_per_C=error,absolute_error_ev_per_C=abs(error),mother_energy_gate=abs(error)<.1,
        lmax=rows[0]['lmax'],relative_rank_tolerance=rows[0]['relative_rank_tolerance'],
        scope='uncontracted_jY_full_q_body_RPA',full_q_admitted=True,
        physical_release_gate='hold',qavg_headwing_validated=False,contracted_NAO_validated=False,
        per_q=rows)
