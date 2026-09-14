"""Prepare immutable contracts for the four-rank C jY compression ladder."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


PROFILES = ((3, 3, 2, 1, 0), (4, 4, 3, 2, 0),
            (5, 5, 4, 3, 0), (6, 6, 5, 4, 0))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ao_per_element(profile):
    return sum((2*l+1)*count for l, count in enumerate(profile))


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n',
                    encoding='ascii')


def normalize_candidates(candidates):
    if (not isinstance(candidates, (list, tuple))
            or [tuple(row.get('profile', ())) for row in candidates] != list(PROFILES)):
        raise ValueError('candidate profile ladder is incomplete or out of order')
    normalized = []
    for expected, row in zip(PROFILES, candidates):
        path = Path(row.get('coefficients_path', '')).resolve()
        if not path.is_file():
            raise ValueError('candidate coefficient file is missing')
        ao = ao_per_element(expected)
        if row.get('ao_per_C') != ao:
            raise ValueError('candidate profile AO count mismatch')
        digest = sha(path)
        if row.get('coefficients_sha256') not in (None, digest):
            raise ValueError('candidate coefficient hash mismatch')
        normalized.append(dict(profile=list(expected), ao_per_C=ao,
            coefficients_path=str(path), coefficients_sha256=digest))
    return normalized


def prepare_contracts(template_root, output_root, candidates, *, source_commit):
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(character not in '0123456789abcdef' for character in source_commit)):
        raise ValueError('full lowercase source commit required')
    template_root = Path(template_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    candidates = normalize_candidates(candidates)
    output_root.mkdir(parents=True)
    contract_hashes = {}
    for slot in range(8):
        template_path = template_root/('q%02d' % slot)/'RUNTIME_CONTRACT.json'
        template = json.loads(template_path.read_text(encoding='ascii'))
        if slot and template.get('q_slot') != slot:
            raise ValueError('template q slot mismatch')
        if not slot and template.get('q_slot', 0) != 0:
            raise ValueError('template q slot mismatch')
        contract = copy.deepcopy(template)
        contract.update(job_id='__RUNTIME_JOB_ID__', source_commit=source_commit,
            scope='compressed_shared_radial_full_q_body_RPA', q_slot=slot,
            lmax_values=[3], relative_rank_tolerance=1e-10,
            occupied_capture_floor=.999999,
            candidate_profiles=copy.deepcopy(candidates),
            full_q_admitted=False, physical_release_gate='hold')
        contract.pop('target_lmax', None)
        contract.pop('target_dir', None)
        inputs = dict(contract.get('inputs', {}))
        for candidate in candidates:
            inputs[candidate['coefficients_path']] = candidate['coefficients_sha256']
        contract['inputs'] = inputs
        stage = output_root/('q%02d' % slot)
        stage.mkdir()
        path = stage/'CONTRACT.json'
        write_new(path, contract)
        contract_hashes[str(path)] = sha(path)
    result = dict(status='success',
        scope='compressed_shared_radial_full_q_body_RPA',
        source_commit=source_commit, q_slots=list(range(8)),
        profiles=[list(row) for row in PROFILES],
        ao_per_C=[ao_per_element(row) for row in PROFILES],
        candidate_profiles=candidates, contract_sha256=contract_hashes,
        physical_release_gate='hold', ordinary_sos_validated=False)
    write_new(output_root/'PREPARATION.json', result)
    return result


def parse_candidate(value):
    try:
        name, path = value.split(':', 1)
        profile = next(row for row in PROFILES
                       if ''.join(map(str, row[:4])) == name)
    except (ValueError, StopIteration):
        raise argparse.ArgumentTypeError('candidate must be PROFILE:PATH')
    return dict(profile=list(profile), ao_per_C=ao_per_element(profile),
                coefficients_path=path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--candidate', action='append', type=parse_candidate,
                        required=True)
    args = parser.parse_args(argv)
    prepare_contracts(args.template_root, args.output_root, args.candidate,
                      source_commit=args.source_commit)


if __name__ == '__main__':
    main()
