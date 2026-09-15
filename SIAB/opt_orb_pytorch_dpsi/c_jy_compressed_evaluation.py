"""Direct full-q RPA evaluation of shared-radial jY contractions."""

from dataclasses import replace
import math


PROFILES = ((3, 3, 2, 1, 0), (4, 4, 3, 2, 0),
            (5, 5, 4, 3, 0), (6, 6, 5, 4, 0))


def _ao_per_element(profile):
    return sum((2*l+1)*count for l, count in enumerate(profile))


def validate_profile_specs(specs, *, profiles=PROFILES):
    try:
        profiles = tuple(tuple(profile) for profile in profiles)
    except TypeError as error:
        raise ValueError('compressed profiles must be a sequence') from error
    if (not profiles or len(set(profiles)) != len(profiles)
            or any(len(profile) != 5 or any(type(count) is not int or count < 0
                                           for count in profile)
                   for profile in profiles)):
        raise ValueError('compressed profiles are invalid')
    if not isinstance(specs, list) or len(specs) != len(profiles):
        raise ValueError('complete requested compressed profile set required')
    result = []
    for row, expected in zip(specs, profiles):
        if not isinstance(row, dict) or tuple(row.get('profile', ())) != expected:
            raise ValueError('compressed profile order or rank mismatch')
        path = row.get('coefficients_path')
        digest = row.get('coefficients_sha256')
        ao_count = _ao_per_element(expected)
        if not isinstance(path, str) or not path:
            raise ValueError('coefficient path is required')
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('coefficient SHA256 is required')
        if row.get('ao_per_C') != ao_count:
            raise ValueError('compressed AO count mismatch')
        result.append(dict(row))
    if len({row['coefficients_sha256'] for row in result}) != len(result):
        raise ValueError('compressed coefficient hashes must be unique')
    return result


def validate_profile_input_hashes(specs, inputs, *, profiles=PROFILES):
    specs = validate_profile_specs(specs, profiles=profiles)
    if not isinstance(inputs, dict):
        raise ValueError('input hash map is required')
    for row in specs:
        if inputs.get(row['coefficients_path']) != row['coefficients_sha256']:
            raise ValueError('compressed coefficient input hash mismatch')
    return True


def evaluate_compressed_profiles(
        dataset, specs, *, read_coefficients, evaluate_response,
        summarize_energy, relative_rank_tolerance=1e-10,
        occupied_capture_floor=.999999, prepare_block_cache=None,
        profile_callback=None, profiles=PROFILES, radial_rows=31):
    """Evaluate each compact rank in the same q dataset and RPA functional."""
    if (not math.isfinite(relative_rank_tolerance)
            or relative_rank_tolerance <= 0
            or not math.isfinite(occupied_capture_floor)
            or not 0 < occupied_capture_floor <= 1):
        raise ValueError('invalid compressed evaluation controls')
    if profile_callback is not None and not callable(profile_callback):
        raise ValueError('profile callback must be callable')
    if type(radial_rows) is not int or radial_rows <= 0:
        raise ValueError('positive radial row count required')
    specs = validate_profile_specs(specs, profiles=profiles)
    frequency_count = int(dataset.frequency_ha.numel())
    if frequency_count != 12:
        raise ValueError('fixed 12-frequency evaluation required')
    occupied_capture_tolerance = max(1e-15, 1.-occupied_capture_floor)
    loaded = []
    for spec in specs:
        profile = tuple(spec['profile'])
        loaded.append((spec, read_coefficients(
            spec['coefficients_path'], element='C', radial_rows=radial_rows,
            expected_nu=profile)))
    if prepare_block_cache is not None:
        dataset = replace(dataset, kpoints=tuple(
            prepare_block_cache(record, dataset.primitive_blocks, loaded[0][1])
            for record in dataset.kpoints))
    rows = []
    for spec, coefficients in loaded:
        profile = tuple(spec['profile'])
        response = evaluate_response(
            dataset, coefficients, contraction_backend='block',
            relative_rank_tolerance=relative_rank_tolerance,
            condition_limit=1e12,
            occupied_capture_tolerance=occupied_capture_tolerance,
            frequency_batch_size=frequency_count)
        summary = summarize_energy(
            dataset, response.response.detach().cpu().numpy())
        finite = (summary.get('candidate_energy_ha'),
                  summary.get('reference_energy_ha'),
                  response.minimum_occupied_capture,
                  response.maximum_overlap_condition)
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in finite):
            raise ValueError('nonfinite compressed RPA evaluation')
        if response.minimum_occupied_capture < occupied_capture_floor:
            raise ValueError('compressed occupied capture below floor')
        row = dict(summary, profile=list(profile), ao_per_C=spec['ao_per_C'],
                   coefficients_sha256=spec['coefficients_sha256'],
                   minimum_occupied_capture=response.minimum_occupied_capture,
                   maximum_overlap_condition=response.maximum_overlap_condition,
                   minimum_candidate_rank=response.minimum_candidate_rank,
                   occupied_capture_floor=occupied_capture_floor,
                   physical_release_gate='hold')
        rows.append(row)
        if profile_callback is not None:
            profile_callback(dict(row))
    return rows
