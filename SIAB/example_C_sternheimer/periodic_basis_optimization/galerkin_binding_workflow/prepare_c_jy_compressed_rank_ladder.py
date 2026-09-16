"""Prepare immutable contracts for C jY compact-rank evaluations."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


PROFILES = ((3, 3, 2, 1, 0), (4, 4, 3, 2, 0),
            (5, 5, 4, 3, 0), (6, 6, 5, 4, 0))
EVALUATION_SCOPE = 'compressed_shared_radial_full_q_body_RPA'
GRADIENT_SCOPE = 'compressed_shared_radial_full_q_energy_gradient'


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


def normalize_profiles(profiles):
    try:
        profiles = tuple(tuple(row) for row in profiles)
    except TypeError as error:
        raise ValueError('candidate profiles must be a nonempty sequence') from error
    if (not profiles or len(set(profiles)) != len(profiles)
            or any(len(row) != 5 or not any(row)
                   or any(type(value) is not int or value < 0 for value in row)
                   for row in profiles)):
        raise ValueError('invalid candidate profiles')
    return profiles


def normalize_candidates(candidates, *, profiles=PROFILES):
    profiles = normalize_profiles(profiles)
    if (not isinstance(candidates, (list, tuple))
            or [tuple(row.get('profile', ())) for row in candidates] != list(profiles)):
        raise ValueError('candidate profile ladder is incomplete or out of order')
    normalized = []
    for expected, row in zip(profiles, candidates):
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


def accepted_artifact_files(audit_result, *, expected_run=None):
    audit_result = Path(audit_result).resolve()
    audit = json.loads(audit_result.read_text(encoding='ascii'))
    if audit.get('status') != 'success':
        raise ValueError('accepted audit result required')
    source_run = audit.get('run', expected_run)
    if source_run is None:
        raise ValueError('audit source run is missing')
    run = Path(source_run).resolve()
    if expected_run is not None and run != Path(expected_run).resolve():
        raise ValueError('audit source run mismatch')
    audit_root = audit_result.parent.parent
    paths = (audit_result, audit_result.parent/'GAUGE_MAPS.npz',
             audit_result.parent/'STATUS', audit_root/'JOB_STATUS',
             audit_root/'PROVENANCE', run/'CONTRACT.json', run/'STATUS',
             run/'PROVENANCE')
    if any(not path.is_file() for path in paths):
        raise ValueError('expanded accepted artifact is incomplete')
    return run, tuple(path.resolve() for path in paths)


def validate_baseline_audit(audit_result):
    audit_result = Path(audit_result).resolve()
    audit = json.loads(audit_result.read_text(encoding='ascii'))
    rows = audit.get('per_k', [])
    if not (audit.get('status') == 'success'
            and audit.get('finite_q_spd_source_compatibility') == 'pass'
            and audit.get('failure_reasons') == []
            and audit.get('regenerated_hamiltonian_read') is False
            and audit.get('regenerated_hamiltonian_admitted') is False
            and audit.get('full_source_primitive_count') == 1550
            and audit.get('compared_primitive_count') == 558
            and len(rows) == 64 and all(row.get('pass_gate') for row in rows)
            and sorted(row.get('source_ik') for row in rows) == list(range(1, 65))
            and sorted(row.get('target_ik') for row in rows) == list(range(1, 65))):
        raise ValueError('complete 1550-row finite-q baseline audit required')
    return audit_result


def normalize_expansion(radial_rows, primitive_expansion,
                        expanded_gamma_run, expanded_gamma_audit_result,
                        expanded_q_audit_results):
    values = (primitive_expansion, expanded_gamma_run,
              expanded_gamma_audit_result, expanded_q_audit_results)
    if all(value is None for value in values):
        if radial_rows != 31:
            raise ValueError('nondefault radial rows require an expanded mother')
        return None
    if any(value is None for value in values):
        raise ValueError('complete expanded mother inputs required')
    target_rows = primitive_expansion.get('expanded_radial_rows')
    full_count = primitive_expansion.get('expanded_primitive_count')
    if (type(target_rows) is not int or target_rows != radial_rows
            or radial_rows <= 31 or full_count != radial_rows * 25 * 2):
        raise ValueError('unexpected expanded mother dimensions')
    expected = dict(source_radial_rows=31, expanded_radial_rows=target_rows,
                    source_primitive_count=1550,
                    expanded_primitive_count=full_count,
                    expanded_spdf_primitive_count=target_rows * 16 * 2)
    if any(primitive_expansion.get(key) != value
           for key, value in expected.items()):
        raise ValueError('unexpected expanded mother contract')
    if set(expanded_q_audit_results) != set(range(1, 8)):
        raise ValueError('all seven finite-q expanded audits are required')
    gamma_run, gamma_inputs = accepted_artifact_files(
        expanded_gamma_audit_result, expected_run=expanded_gamma_run)
    q_rows = {}
    for slot, audit in expanded_q_audit_results.items():
        _, paths = accepted_artifact_files(audit)
        q_rows[slot] = dict(audit_result=Path(audit).resolve(), inputs=paths)
    return dict(primitive_expansion=copy.deepcopy(primitive_expansion),
                gamma_run=gamma_run,
                gamma_audit_result=Path(expanded_gamma_audit_result).resolve(),
                gamma_inputs=gamma_inputs, finite_q=q_rows)


def prepare_contracts(template_root, output_root, candidates, *, source_commit,
                      profiles=PROFILES, radial_rows=31,
                      primitive_expansion=None, expanded_gamma_run=None,
                      expanded_gamma_audit_result=None,
                      expanded_q_audit_results=None,
                      scope=EVALUATION_SCOPE):
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(character not in '0123456789abcdef' for character in source_commit)):
        raise ValueError('full lowercase source commit required')
    template_root = Path(template_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    profiles = normalize_profiles(profiles)
    if scope not in (EVALUATION_SCOPE, GRADIENT_SCOPE):
        raise ValueError('invalid compressed full-q scope')
    if scope == GRADIENT_SCOPE and (profiles != ((4, 4, 3, 2, 0),)
                                    or radial_rows <= 31):
        raise ValueError('energy gradients require one expanded 45-AO profile')
    if type(radial_rows) is not int or radial_rows <= 0:
        raise ValueError('radial_rows must be a positive integer')
    candidates = normalize_candidates(candidates, profiles=profiles)
    expansion = normalize_expansion(
        radial_rows, primitive_expansion, expanded_gamma_run,
        expanded_gamma_audit_result, expanded_q_audit_results)
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
            scope=scope, q_slot=slot,
            lmax_values=[3], relative_rank_tolerance=1e-10,
            occupied_capture_floor=.99999,
            radial_rows=radial_rows,
            candidate_profiles=copy.deepcopy(candidates),
            full_q_admitted=False, physical_release_gate='hold')
        contract.pop('target_lmax', None)
        contract.pop('target_dir', None)
        inputs = dict(contract.get('inputs', {}))
        if expansion is not None:
            contract.update(
                primitive_expansion=copy.deepcopy(expansion['primitive_expansion']),
                expanded_gamma_run=str(expansion['gamma_run']),
                expanded_gamma_audit_result=str(expansion['gamma_audit_result']))
            artifact_inputs = list(expansion['gamma_inputs'])
            if slot:
                baseline_result = template.get(
                    'baseline_audit_result', template.get('audit_result'))
                if baseline_result is None:
                    raise ValueError('finite-q baseline audit is missing')
                baseline = validate_baseline_audit(baseline_result)
                if not baseline.is_file():
                    raise ValueError('finite-q baseline audit file is missing')
                contract['baseline_audit_result'] = str(baseline)
                contract['audit_result'] = str(
                    expansion['finite_q'][slot]['audit_result'])
                artifact_inputs.extend(expansion['finite_q'][slot]['inputs'])
                artifact_inputs.append(baseline)
            for path in artifact_inputs:
                inputs[str(path)] = sha(path)
        for candidate in candidates:
            inputs[candidate['coefficients_path']] = candidate['coefficients_sha256']
        contract['inputs'] = inputs
        stage = output_root/('q%02d' % slot)
        stage.mkdir()
        path = stage/'CONTRACT.json'
        write_new(path, contract)
        contract_hashes[str(path)] = sha(path)
    result = dict(status='success',
        scope=scope,
        source_commit=source_commit, q_slots=list(range(8)),
        profiles=[list(row) for row in profiles],
        ao_per_C=[ao_per_element(row) for row in profiles],
        radial_rows=radial_rows,
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


def parse_slot_path(value):
    try:
        slot, path = value.split(':', 1)
        slot = int(slot)
    except ValueError as error:
        raise argparse.ArgumentTypeError('expanded q audit must be SLOT:PATH') from error
    if slot not in range(1, 8) or not path:
        raise argparse.ArgumentTypeError('expanded q audit slot must be 1..7')
    return slot, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--candidate', action='append', type=parse_candidate,
                        required=True)
    parser.add_argument('--radial-rows', type=int, default=31)
    parser.add_argument('--expanded-gamma-run', type=Path)
    parser.add_argument('--expanded-gamma-audit-result', type=Path)
    parser.add_argument('--expanded-q-audit', action='append', type=parse_slot_path)
    parser.add_argument('--energy-gradient', action='store_true')
    args = parser.parse_args(argv)
    profiles = tuple(tuple(row['profile']) for row in args.candidate)
    expansion = None
    q_audits = None
    if args.expanded_gamma_run is not None:
        gamma_contract = json.loads(
            (args.expanded_gamma_run/'CONTRACT.json').read_text(encoding='ascii'))
        expansion = gamma_contract.get('primitive_expansion')
        q_audits = dict(args.expanded_q_audit or ())
    prepare_contracts(args.template_root, args.output_root, args.candidate,
                      source_commit=args.source_commit, profiles=profiles,
                      radial_rows=args.radial_rows,
                      primitive_expansion=expansion,
                      expanded_gamma_run=args.expanded_gamma_run,
                      expanded_gamma_audit_result=args.expanded_gamma_audit_result,
                      expanded_q_audit_results=q_audits,
                      scope=(GRADIENT_SCOPE if args.energy_gradient
                             else EVALUATION_SCOPE))


if __name__ == '__main__':
    main()
