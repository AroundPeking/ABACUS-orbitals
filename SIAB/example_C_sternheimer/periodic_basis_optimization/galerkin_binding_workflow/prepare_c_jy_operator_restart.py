"""Prepare one Gamma frozen-density operator check; never submit a job."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def input_values(text):
    values = {}
    for line in text.splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields or fields == ['INPUT_PARAMETERS']:
            continue
        if len(fields) < 2 or fields[0].lower() in values:
            raise ValueError('invalid or duplicate INPUT key')
        values[fields[0].lower()] = ' '.join(fields[1:])
    return values


def frequency_rows(text):
    lines = text.splitlines()
    if not lines or lines[0] != 'ABACUS_STERNHEIMER_BASIS_OPT_MANIFEST_V1':
        raise ValueError('accepted full-response manifest required')
    rows = [line.split() for line in lines if line.startswith('frequency ')]
    if len(rows) != 12 or [int(row[1]) for row in rows] != list(range(12)):
        raise ValueError('complete ordered twelve-frequency grid required')
    if any(len(row) != 4 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in row[2:])
           for row in rows):
        raise ValueError('positive finite nodes and weights required')
    return ''.join(row[2] + ' ' + row[3] + '\n' for row in rows)


def prepare(reference, build_root, output):
    acceptance = json.loads((build_root/'result/BUILD_ACCEPTANCE.json').read_text())
    binary = build_root/'build/abacus_3p'
    if (acceptance['status'] != 'success'
            or acceptance['source_commit'] != 'e1d0e64591fbd0324533c25cffbd6f94f6a1da4f'
            or acceptance['binary_sha256'] != sha(binary)):
        raise ValueError('accepted operators-only build required')
    values = input_values((reference/'INPUT').read_text())
    expected = dict(calculation='scf', nspin='1', nbands='44', basis_type='lcao',
        nx='24', ny='24', nz='24', sternheimer_fd_order='8', sternheimer_nfreq='12',
        sternheimer_q_index='1', exx_pca_threshold='1e-6', sternheimer_siab_lmax='4')
    if any(values.get(k) != v for k, v in expected.items()):
        raise ValueError('frozen Gamma INPUT mismatch')
    suffix = values['suffix']
    refout = reference/('OUT.' + suffix)
    manifest = refout/'STERNHEIMER_BASIS_OPT_V1/manifest.dat'
    manifest_text = manifest.read_text()
    frequencies = frequency_rows(manifest_text)
    metadata = {line.split()[0]: line.split()[1] for line in manifest_text.splitlines()
                if len(line.split()) == 2}
    if metadata.get('abacus_commit') != '711af860c125b9757c344a1961b63524c550cfe4':
        raise ValueError('unexpected reference producer')
    for filename, key in [('C_ONCV_PBE-1.0.upf', 'pseudopotential_sha256'),
                          ('C_gga_10au_100Ry_3s3p2d.orb', 'orbital_sha256')]:
        if sha(reference/filename) != metadata.get(key):
            raise ValueError('reference PP/orbital no longer matches accepted manifest')
    density = refout/(suffix + '-CHARGE-DENSITY.restart')
    if not density.is_file() or density.stat().st_size == 0:
        raise ValueError('accepted binary density missing')
    output.mkdir()
    (output/'frozen-density').mkdir()
    shutil.copy2(density, output/'frozen-density'/density.name)
    for filename in ('STRU', 'KPT', 'C_ONCV_PBE-1.0.upf', 'C_gga_10au_100Ry_3s3p2d.orb'):
        source = reference/filename
        if not source.is_file():
            raise ValueError('missing frozen input: ' + str(source))
        shutil.copy2(source, output/filename)
    values.update(calculation='nscf', init_chg='file', read_file_dir='./frozen-density/',
        pseudo_dir='./', orbital_dir='./', kpar='8', sternheimer_frequency_mpi='0',
        sternheimer_siab_source_only='1', sternheimer_frequency_grid_file='FREQUENCY_GRID.dat')
    (output/'INPUT').write_text('INPUT_PARAMETERS\n' + ''.join(k + ' ' + v + '\n' for k, v in values.items()))
    (output/'FREQUENCY_GRID.dat').write_text(frequencies)
    inputs = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    contract = dict(scope='Gamma_frozen_density_operator_compatibility',
        reference=str(reference), reference_manifest_sha256=sha(manifest),
        density_sha256=sha(density), source_commit=acceptance['source_commit'],
        build_acceptance_sha256=sha(build_root/'result/BUILD_ACCEPTANCE.json'),
        binary=str(binary), binary_sha256=sha(binary), inputs=inputs, suffix=suffix,
        first_order_equations_allowed=0, self_consistency_allowed=False,
        full_q_admitted=False, physical_release_gate='hold')
    (output/'CONTRACT.json').write_text(json.dumps(contract, indent=2) + '\n')
    (output/'INPUT_SHA256SUMS').write_text(''.join(digest + '  ' + path + '\n'
        for path, digest in inputs.items()))
    print(json.dumps(dict(output=str(output), contract_sha256=sha(output/'CONTRACT.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.reference, args.build_root, args.output)
