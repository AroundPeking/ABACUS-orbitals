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


def collect_compressed_rank_ladder(records, *, reference_energy_ha=-.5144842779929139,
                                   spdf_mother_energy_ha=None, tolerance=1e-12):
    """Combine weighted q-star energies for one shared-radial rank ladder.

    The mother-space error is measured against the frozen Delta-ST body energy.
    The compression error is measured against the accepted uncontracted spdf
    mother energy, so a good fit is not mistaken for a good mother space.
    Each profile must carry the same coefficient hash in every q record.
    """
    if spdf_mother_energy_ha is None:
        raise ValueError('spdf mother energy is required')
    if not math.isfinite(float(reference_energy_ha)) or not math.isfinite(float(spdf_mother_energy_ha)):
        raise ValueError('finite reference and mother energies required')
    rows = join_q_records(records)
    profiles = None
    result_rows = []
    for row in rows:
        current = row.get('per_profile')
        if not isinstance(current, list) or not current:
            raise ValueError('complete compressed profile records required')
        signatures = [(tuple(p.get('profile', ())), p.get('coefficients_sha256'))
                      for p in current]
        if any(not profile or not isinstance(digest, str) or len(digest) != 64
               for profile, digest in signatures):
            raise ValueError('invalid coefficient hash')
        if profiles is None:
            profiles = signatures
        elif signatures != profiles:
            raise ValueError('coefficient profile/hash mismatch across q records')
    for index, (profile, coefficients_sha256) in enumerate(profiles):
        profile_rows = [row['per_profile'][index] for row in rows]
        candidate = math.fsum(float(q['candidate_energy_ha']) for q in profile_rows)
        mother_error = (float(spdf_mother_energy_ha) - float(reference_energy_ha)) \
            * 27.211386245988 / 2
        compression_error = (candidate - float(spdf_mother_energy_ha)) \
            * 27.211386245988 / 2
        total_error = (candidate - float(reference_energy_ha)) \
            * 27.211386245988 / 2
        result_rows.append(dict(
            profile=list(profile),
            ao_per_C=profile_rows[0]['ao_per_C'],
            coefficients_sha256=coefficients_sha256,
            candidate_energy_ha=candidate,
            reference_energy_ha=float(reference_energy_ha),
            mother_energy_ha=float(spdf_mother_energy_ha),
            mother_error_ev_per_C=mother_error,
            compression_error_ev_per_C=compression_error,
            total_error_ev_per_C=total_error,
            energy_gate=abs(total_error) < .1,
            minimum_occupied_capture=min(float(q['minimum_occupied_capture'])
                                         for q in profile_rows),
            maximum_overlap_condition=max(float(q['maximum_overlap_condition'])
                                          for q in profile_rows),
            minimum_candidate_rank=min(int(q['minimum_candidate_rank'])
                                       for q in profile_rows),
            q_count=len(profile_rows)))
    return dict(status='success', scope='compressed_shared_radial_full_q_body_RPA',
        full_q_admitted=True, ordinary_sos_validated=False,
        qavg_headwing_validated=False, physical_release_gate='hold',
        mother_energy_ha=float(spdf_mother_energy_ha),
        reference_energy_ha=float(reference_energy_ha), per_profile=result_rows)
