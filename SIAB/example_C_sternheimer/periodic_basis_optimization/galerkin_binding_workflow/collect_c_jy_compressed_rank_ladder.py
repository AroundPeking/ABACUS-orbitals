"""Collect the full-q RPA energy of occupied-safe jY contractions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/'SIAB/opt_orb_pytorch_dpsi'))
from c_jy_joined_dataset import collect_compressed_rank_ladder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n',
                    encoding='ascii')


def finalize_collection(records, *, reference_energy_ha, spdf_mother_energy_ha):
    result = collect_compressed_rank_ladder(
        records, reference_energy_ha=reference_energy_ha,
        spdf_mother_energy_ha=spdf_mother_energy_ha)
    passing = [row for row in result['per_profile'] if row['energy_gate']]
    selected = min(passing, key=lambda row: row['ao_per_C']) if passing else None
    result.update(
        candidate_selection_gate='pass' if selected else 'fail',
        selected_profile=None if selected is None else selected['profile'],
        selected_ao_per_C=None if selected is None else selected['ao_per_C'],
        selected_coefficients_sha256=(None if selected is None else
                                      selected['coefficients_sha256']),
        ordinary_sos_release_gate='pass' if selected else 'hold')
    return result


def run(contract_path, output):
    contract = json.loads(Path(contract_path).read_text(encoding='ascii'))
    if (os.environ.get('C_EXECUTION_HOST') != 'df_iopcas_ghj'
            or os.environ.get('SLURM_JOB_ID') != contract['job_id']):
        raise ValueError('registered DF collector job required')
    manifest = json.loads((REPO/'SOURCE_MANIFEST.json').read_text(encoding='ascii'))
    if manifest['commit'] != contract['source_commit']:
        raise ValueError('immutable source mismatch')
    for name, expected in manifest['files'].items():
        if sha(REPO/name) != expected:
            raise ValueError('source hash mismatch: '+name)
    for name, expected in contract['inputs'].items():
        if sha(name) != expected:
            raise ValueError('input hash mismatch: '+name)
    paths = [Path(path) for path in contract['q_result_paths']]
    if len(paths) != 8:
        raise ValueError('eight q result paths required')
    records = [json.loads(path.read_text(encoding='ascii')) for path in paths]
    if [row.get('q_slot') for row in records] != list(range(8)):
        raise ValueError('ordered complete q slots required')
    output.mkdir()
    try:
        result = finalize_collection(
            records, reference_energy_ha=contract['reference_energy_ha'],
            spdf_mother_energy_ha=contract['spdf_mother_energy_ha'])
        result.update(source_commit=contract['source_commit'],
                      input_result_sha256={str(path): sha(path) for path in paths})
        write_new(output/'RESULT.json', result)
        write_new(output/'PROVENANCE.json', dict(
            status='success', job_id=contract['job_id'],
            source_commit=contract['source_commit'],
            contract_sha256=sha(contract_path), result_sha256=sha(output/'RESULT.json'),
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        (output/'STATUS').write_text('success\n', encoding='ascii')
    except Exception as error:
        write_new(output/'FAILURE.json', dict(status='failed', error=str(error)))
        (output/'STATUS').write_text('failed\n', encoding='ascii')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.contract, arguments.output)
