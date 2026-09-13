"""Prepare one frozen-density operator check; finite q requires accepted Gamma."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct


def gamma_coulomb_files(reference):
    """Validate the frozen reader blocks before renumbering only their filenames."""
    files = sorted(reference.glob('v1_coulomb_full_iq_1_rank*.dat'),
                   key=lambda p: int(p.stem.rsplit('rank', 1)[1]))
    if not files or len(files) > 8:
        raise ValueError('Gamma reader blocks missing or exceed export MPI ranks')
    seen, expected, metadata = set(), set(), None
    for path in files:
        data = path.read_bytes()
        if len(data) < 32:
            raise ValueError('truncated Gamma Coulomb header')
        marker, iq, naux, flag, natom, nblocks = struct.unpack_from('<6i', data)
        if (marker != -20129433 or iq != 1 or natom != 2 or flag not in (0, 1)
                or nblocks < 0 or len(data) < 32 + 12 * nblocks):
            raise ValueError('invalid Gamma Coulomb metadata')
        sizes = struct.unpack_from('<2i', data, 24)
        if min(sizes) <= 0 or sum(sizes) != naux:
            raise ValueError('invalid Gamma auxiliary sizes')
        current = (naux, flag, sizes)
        if metadata is not None and metadata != current:
            raise ValueError('inconsistent Gamma Coulomb rank metadata')
        metadata = current
        pairs = [(0, 0), (0, 1), (1, 1)]
        expected = set(range(len(pairs)))
        for block in range(nblocks):
            pair, offset = struct.unpack_from('<iq', data, 32 + 12 * block)
            if pair not in expected or pair in seen:
                raise ValueError('invalid or duplicate Gamma atom pair')
            seen.add(pair)
            a, b = pairs[pair]
            size = sizes[a] * sizes[b] * (16 if flag else 8)
            if offset < 32 + 12 * nblocks or offset + size > len(data):
                raise ValueError('truncated Gamma Coulomb payload')
            if any(not math.isfinite(value[0]) for value in
                   struct.iter_unpack('<d', data[offset:offset+size])):
                raise ValueError('nonfinite Gamma Coulomb payload')
    if seen != expected:
        raise ValueError('Gamma Coulomb atom pairs incomplete')
    return files


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


def export_q_contract(iq, gamma_acceptance):
    if iq not in (1, 22, 43, 6, 27, 23, 11, 55):
        raise ValueError('q index is not a frozen canonical representative')
    if iq == 1:
        return dict(scope='Gamma_frozen_density_operator_compatibility', selected_iq=1)
    if gamma_acceptance is None:
        raise ValueError('finite-q export requires accepted Gamma compatibility')
    result = json.loads(gamma_acceptance.read_text())
    if (result.get('status') != 'success'
            or result.get('gamma_operator_compatibility_gate') != 'pass'
            or result.get('failure_reasons') != []
            or len(result.get('per_k', [])) != 64
            or not all(row.get('pass_gate') is True for row in result['per_k'])):
        raise ValueError('Gamma compatibility did not pass')
    return dict(scope='finite_q_frozen_density_operators_pending_spd_compatibility',
        selected_iq=iq, gamma_acceptance=str(gamma_acceptance.resolve()),
        gamma_acceptance_sha256=sha(gamma_acceptance))


def prepare(reference, build_root, output, iq=1, gamma_acceptance=None):
    q_contract = export_q_contract(iq, gamma_acceptance)
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
    coulomb_files = gamma_coulomb_files(reference)
    output.mkdir()
    (output/'frozen-density').mkdir()
    shutil.copy2(density, output/'frozen-density'/density.name)
    for filename in ('STRU', 'KPT', 'C_ONCV_PBE-1.0.upf', 'C_gga_10au_100Ry_3s3p2d.orb'):
        source = reference/filename
        if not source.is_file():
            raise ValueError('missing frozen input: ' + str(source))
        shutil.copy2(source, output/filename)
    coulomb_reuse = []
    for rank, source in enumerate(coulomb_files):
        name = 'v1_coulomb_full_iq_1_rank%d.dat' % rank
        shutil.copy2(source, output/name)
        coulomb_reuse.append(dict(source=str(source), staged=name, sha256=sha(source)))
    values.update(calculation='nscf', init_chg='file', read_file_dir='./frozen-density/',
        pseudo_dir='./', orbital_dir='./', kpar='8', sternheimer_frequency_mpi='0',
        sternheimer_siab_source_only='1', sternheimer_frequency_grid_file='FREQUENCY_GRID.dat',
        sternheimer_q_index=str(iq))
    (output/'INPUT').write_text('INPUT_PARAMETERS\n' + ''.join(k + ' ' + v + '\n' for k, v in values.items()))
    (output/'FREQUENCY_GRID.dat').write_text(frequencies)
    inputs = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    contract = dict(q_contract,
        reference=str(reference), reference_manifest_sha256=sha(manifest),
        density_sha256=sha(density), source_commit=acceptance['source_commit'],
        build_acceptance_sha256=sha(build_root/'result/BUILD_ACCEPTANCE.json'),
        binary=str(binary), binary_sha256=sha(binary), inputs=inputs, suffix=suffix,
        gamma_coulomb_reuse=coulomb_reuse,
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
    parser.add_argument('--iq', type=int, default=1)
    parser.add_argument('--gamma-acceptance', type=Path)
    args = parser.parse_args()
    prepare(args.reference, args.build_root, args.output, args.iq, args.gamma_acceptance)
