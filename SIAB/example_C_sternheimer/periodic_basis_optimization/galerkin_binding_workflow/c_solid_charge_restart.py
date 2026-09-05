"""Fail-closed charge-only recovery for the failed standard C q8 producer."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


INPUT_FILES = ('INPUT', 'STRU', 'KPT', 'C_ONCV_PBE-1.0.upf',
               'C_gga_10au_100Ry_3s3p2d.orb', 'Q1_FREQUENCY_GRID.tsv')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_input(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        words = line.split('#', 1)[0].split()
        if not words or words[0] == 'INPUT_PARAMETERS':
            continue
        if len(words) < 2 or words[0] in result:
            raise ValueError('invalid or duplicate INPUT key')
        result[words[0]] = ' '.join(words[1:])
    return result


def write_input(path, values):
    Path(path).write_text('INPUT_PARAMETERS\n' + ''.join(
        '{} {}\n'.format(key, value) for key, value in values.items()))


def unique_number(text, pattern):
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError('expected exactly one energy record')
    value = float(matches[0])
    if not math.isfinite(value):
        raise ValueError('non-finite energy')
    return value


def scf_energy(text):
    if '#SCF IS CONVERGED#' not in text or re.search(
            r'NOT\s+CONVERGED|convergence has not', text, re.I):
        raise ValueError('SCF is not accepted as converged')
    return unique_number(text, r'#TOTAL ENERGY#\s+(\S+)\s+eV')


def audit_parent(parent, job_id, state, exit_code):
    parent = Path(parent).resolve()
    if job_id != '21862974_4' or state != 'FAILED' or exit_code != '127:0':
        raise ValueError('not the terminal failed q8 parent')
    values = read_input(parent / 'INPUT')
    suffix = values['suffix']
    if (values.get('sternheimer_q_index') != '23' or values.get('nspin') != '1'
            or suffix != 'C_DIAMOND_FD8_Q8_NFREQ12_PCA1E6_BASIS_OPT_DFDCU'):
        raise ValueError('not the frozen standard q8 input')
    out = parent / ('OUT.' + suffix)
    if (out / 'STERNHEIMER_BASIS_OPT_V1/manifest.dat').exists():
        raise ValueError('response manifest exists: requires separate completion audit')
    abacus_text = (parent / 'abacus.out').read_text()
    if 'Transport retry count exceeded' not in abacus_text or 'PMPI_Barrier' not in abacus_text:
        raise ValueError('missing expected transport failure evidence')
    charge = out / (suffix + '-CHARGE-DENSITY.restart')
    if not charge.is_file() or charge.stat().st_size == 0:
        raise ValueError('missing charge restart')
    files = [parent / name for name in INPUT_FILES]
    files.extend([charge, parent / 'abacus.out', out / 'running_scf.log'])
    return {
        'status': 'audited_failed_parent', 'parent_job_id': job_id,
        'parent_state': state, 'parent_exit_code': exit_code,
        'parent_root': str(parent), 'suffix': suffix, 'charge_file': str(charge),
        'q_label': 8, 'selected_iq': 23, 'response_checkpoint_available': False,
        'pbe_energy_ev': scf_energy((out / 'running_scf.log').read_text()),
        'pbe_energy_tolerance_ev': 1.0e-7,
        'zero_order_energy_ha': unique_number(abacus_text, r'Etot_without_rpa\(Ha\):\s+(\S+)'),
        'zero_order_energy_tolerance_ha': 1.0e-8,
        'file_hashes': {str(path): sha256(path) for path in files},
    }


def verify_parent(contract):
    if contract.get('status') != 'audited_failed_parent' or contract.get('q_label') != 8:
        raise ValueError('invalid restart contract')
    for path, digest in contract['file_hashes'].items():
        if sha256(path) != digest:
            raise ValueError('parent hash mismatch: ' + path)


def write_record(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')


def prepare(contract, work):
    work = Path(work).resolve()
    verify_parent(contract)
    if any((work / name).exists() for name in ('restart-charge', 'restart-pbe', 'CHARGE_RESTART_PREPARED.json')):
        raise FileExistsError('recovery preparation already exists')
    parent = Path(contract['parent_root'])
    if work == parent or parent in work.parents:
        raise ValueError('recovery must be outside failed parent')
    if read_input(work / 'INPUT') != read_input(parent / 'INPUT'):
        raise ValueError('response physical INPUT differs from failed parent')
    for name in INPUT_FILES[1:]:
        if sha256(work / name) != contract['file_hashes'][str(parent / name)]:
            raise ValueError('response input hash differs: ' + name)
    restart = work / 'restart-charge'
    restart.mkdir()
    charge = Path(contract['charge_file'])
    shutil.copyfile(str(charge), str(restart / charge.name))
    if sha256(restart / charge.name) != contract['file_hashes'][str(charge)]:
        raise ValueError('copied charge hash mismatch')
    values = read_input(work / 'INPUT')
    values.update(init_chg='file', read_file_dir='./restart-charge/')
    write_input(work / 'INPUT', values)
    gate = work / 'restart-pbe'
    gate.mkdir()
    for name in INPUT_FILES[1:]:
        shutil.copyfile(str(work / name), str(gate / name))
    values.update(rpa='0', out_sternheimer_basis_opt='0', out_sternheimer_librpa='0',
                  read_file_dir='../restart-charge/')
    write_input(gate / 'INPUT', values)
    write_record(work / 'CHARGE_RESTART_PREPARED.json', {
        'status': 'prepared_not_physically_accepted', 'parent_job': contract['parent_job_id'],
        'charge_sha256': sha256(restart / charge.name),
        'response_input_sha256': sha256(work / 'INPUT'),
        'pbe_input_sha256': sha256(gate / 'INPUT'),
    })


def validate_scf_text(text, reference_ev, charge_name):
    energy = scf_energy(text)
    if not any('Read electron density from file:' in line and charge_name in line
               for line in text.splitlines()):
        raise ValueError('no binary charge restart load evidence')
    if abs(energy - reference_ev) > 1.0e-7:
        raise ValueError('PBE restart changed energy')
    return energy


def validate_response_energy(text, reference_ha):
    energy = unique_number(text, r'Etot_without_rpa\(Ha\):\s+(\S+)')
    if abs(energy - reference_ha) > 1.0e-8:
        raise ValueError('response zero-order energy changed')
    return energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('audit', 'prepare', 'validate-pbe', 'validate-response'))
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--scheduler-state')
    parser.add_argument('--scheduler-exit')
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    if args.operation == 'audit':
        write_record(args.contract, audit_parent(args.parent, args.job_id,
                                                args.scheduler_state, args.scheduler_exit))
        return
    contract = json.loads(args.contract.read_text())
    if args.operation == 'prepare':
        prepare(contract, args.work)
        return
    verify_parent(contract)
    work = args.work.resolve()
    stage = work / 'restart-pbe' if args.operation == 'validate-pbe' else work
    energy = validate_scf_text((stage / ('OUT.' + contract['suffix']) / 'running_scf.log').read_text(),
                               contract['pbe_energy_ev'], Path(contract['charge_file']).name)
    record = {'status': 'success', 'pbe_energy_ev': energy,
              'pbe_difference_ev': energy - contract['pbe_energy_ev'],
              'contract_sha256': sha256(args.contract)}
    if args.operation == 'validate-response':
        zero = validate_response_energy((work / 'abacus.out').read_text(), contract['zero_order_energy_ha'])
        record.update(zero_order_energy_ha=zero, zero_order_difference_ha=zero-contract['zero_order_energy_ha'])
    record['running_log_sha256'] = sha256(stage / ('OUT.' + contract['suffix']) / 'running_scf.log')
    write_record(stage / 'CHARGE_RESTART_VALIDATION.json', record)


if __name__ == '__main__':
    main()
