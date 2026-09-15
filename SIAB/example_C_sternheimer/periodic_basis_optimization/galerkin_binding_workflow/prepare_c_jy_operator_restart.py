"""Prepare frozen-density exports with separate operator and source-only contracts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

REFERENCE_REUSE_SHA = '0bc101cdfdd6b2fb3b279a0f24404f49e182c917c0efd7de574bfb366aec0bb2'
GAMMA_SOURCE_AUDIT_SHA = '98c59af6cafcde09ffdc9807fa042bc1279b6e32b9ca0f0f69c490ec83d4e9b5'
FROZEN_AUXILIARY_COMMIT = '0613452ee9e176e5fb34364cfe296cccca706e77'
Q2_CACHE_METADATA_SHA = 'd7935a428448db3b75c52ed639487a4b5677685ef90a2ca8a14ba53944c21454'
Q2_SOURCE_AUDIT_SHA = 'b533da627f0e5d159f8b1a030f6fa0093ef8cd8f488176ee73d8d2df8935c55e'
FINITE_Q_SPECS = {
    22: ('q2', 8, Q2_CACHE_METADATA_SHA),
    43: ('q3', 4, '5760ac9dadfb49caa60c82b5fea5259f1170e46fb86223d60e0e3cc3f0280d1a'),
    6: ('q6', 6, 'f66e207b1aab6f87e0cd9845f1f8dbc94f3873bdb96b6fdf99cfcb156627398f'),
    27: ('q7', 24, '347163525f82df8c2198008d641ed6a094567d9803e0dbc8617cb6cf4c92a079'),
    23: ('q8', 12, '3dc24457f9453724aecf41c5bb0a7eb8398d9742361a6cd0f3449e75b7ad4c05'),
    11: ('q11', 3, 'e9195f42c35429aeaaf52529527f6132305db3bc249dc07ef800e28ddc9051af'),
    55: ('q28', 6, '1946a7a2b8ab02ae9bf0b5ef352d58af6b9ffaaa4ed25bfcdfa78774ef3994a0'),
}


def radial_rows_from_bessel(ecut_ry, rcut_bohr):
    """Return the exact SIAB spherical-Bessel row count."""
    values = (ecut_ry, rcut_bohr)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('positive finite Bessel cutoff and radius required')
    return int(math.sqrt(float(ecut_ry)) * float(rcut_bohr) / math.pi)


def primitive_count(radial_rows, lmax, natom):
    if (type(radial_rows) is not int or radial_rows <= 0
            or type(lmax) is not int or lmax < 0
            or type(natom) is not int or natom <= 0):
        raise ValueError('positive primitive dimensions required')
    return radial_rows * (lmax + 1) ** 2 * natom


def primitive_prefix_indices(source_rows, target_rows, lmax, natom):
    """Map a nested lower-cutoff Bessel mother into an expanded mother."""
    primitive_count(source_rows, lmax, natom)
    primitive_count(target_rows, lmax, natom)
    if source_rows > target_rows:
        raise ValueError('source rows exceed expanded mother')
    return tuple(block * target_rows + row
                 for block in range(natom * (lmax + 1) ** 2)
                 for row in range(source_rows))


def validate_bessel_contract(radial_rows, ecut_ry, rcut_bohr):
    actual = radial_rows_from_bessel(ecut_ry, rcut_bohr)
    if actual != radial_rows:
        raise ValueError('Bessel cutoff does not produce requested radial rows: '
                         '%d != %d' % (actual, radial_rows))
    return actual


def configure_expanded_mother(values, *, radial_rows, bessel_nao_ecut):
    """Change only the nested Bessel mother size and return its dimensions."""
    values = dict(values)
    try:
        source_ecut = float(values['bessel_nao_ecut'])
        rcut = float(values['bessel_nao_rcut'])
        lmax = int(values['sternheimer_siab_lmax'])
    except (KeyError, ValueError) as error:
        raise ValueError('complete Bessel mother controls required') from error
    source_rows = radial_rows_from_bessel(source_ecut, rcut)
    validate_bessel_contract(radial_rows, bessel_nao_ecut, rcut)
    if radial_rows <= source_rows:
        raise ValueError('expanded mother must add radial rows')
    values['bessel_nao_ecut'] = format(float(bessel_nao_ecut), '.15g')
    contract = dict(
        nested_spherical_bessel_expansion=True,
        source_bessel_nao_ecut_ry=source_ecut,
        expanded_bessel_nao_ecut_ry=float(bessel_nao_ecut),
        bessel_nao_rcut_bohr=rcut,
        source_radial_rows=source_rows,
        expanded_radial_rows=radial_rows,
        source_primitive_count=primitive_count(source_rows, lmax, 2),
        expanded_primitive_count=primitive_count(radial_rows, lmax, 2),
        expanded_spdf_primitive_count=primitive_count(radial_rows, 3, 2),
        source_prefix_indices_sha256=hashlib.sha256(
            struct.pack('<%dI' % primitive_count(source_rows, lmax, 2),
                        *primitive_prefix_indices(source_rows, radial_rows, lmax, 2))
        ).hexdigest(),
    )
    return values, contract


def zero_pad_radial_coefficients(coefficients, *, source_rows, target_rows):
    """Embed a compact basis exactly in a larger nested Bessel mother."""
    if not isinstance(coefficients, dict) or not coefficients or target_rows <= source_rows:
        raise ValueError('nonempty coefficients and a larger target are required')
    result = {}
    for element, channels in coefficients.items():
        padded_channels = []
        for channel in channels:
            if len(channel.shape) != 2 or channel.shape[0] != source_rows:
                raise ValueError('coefficient radial row count mismatch')
            shape = (target_rows, channel.shape[1])
            if hasattr(channel, 'new_zeros'):
                padded = channel.new_zeros(shape)
            else:
                import numpy as np
                padded = np.zeros(shape, dtype=channel.dtype)
            padded[:source_rows] = channel
            padded_channels.append(padded)
        result[element] = padded_channels
    return result


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


def freeze_auxiliary_cache(cache, output, iq, expected_metadata_sha256):
    """Serialize the pinned reference V/W, without diagonalization or truncation."""
    import numpy as np
    complete_path, metadata_path = cache / 'COMPLETE.json', cache / 'dataset.json'
    complete = json.loads(complete_path.read_text())
    if complete.get('status') != 'complete' or complete.get('format_version') != 1:
        raise ValueError('complete cache required for frozen auxiliary input')
    metadata_sha = sha(metadata_path)
    if metadata_sha != expected_metadata_sha256 or complete.get('metadata_sha256') != metadata_sha:
        raise ValueError('frozen auxiliary cache metadata hash mismatch')
    metadata = json.loads(metadata_path.read_text())
    scalar = metadata['source']['scalar']
    if scalar['selected_iq'] != [str(iq)] or scalar['kernel'] != ['full_coulomb']:
        raise ValueError('frozen auxiliary cache q or kernel mismatch')
    raw, rank = int(scalar['raw_auxiliary_dimension'][0]), int(scalar['whitened_auxiliary_rank'][0])
    if not 0 < rank <= raw:
        raise ValueError('frozen auxiliary cache dimensions invalid')
    payloads, records = {}, {}
    for key, kind, shape in (('coulomb_metric', 4, (raw, raw)), ('coulomb_whitening', 5, (raw, rank))):
        field = metadata['dataset']['fields'][key]
        if field['type'] != 'tensor' or not 0 <= field['index'] < len(metadata['arrays']):
            raise ValueError('frozen auxiliary array descriptor invalid')
        record = metadata['arrays'][field['index']]
        path = (cache / record['file']).resolve()
        if cache.resolve() not in path.parents or sha(path) != record['sha256']:
            raise ValueError('frozen auxiliary array path or hash mismatch')
        value = np.load(path, allow_pickle=False)
        if (tuple(record['shape']) != shape or value.shape != shape or value.dtype.str != record['dtype']
                or value.dtype.kind != 'c' or not np.isfinite(value).all()):
            raise ValueError('frozen auxiliary array shape, dtype or finite-value mismatch')
        payloads[key] = (struct.pack('<16sIIiiiQQ', b'ABACUS_STBOPT_V1', 1, kind, iq, 0, -1, *shape)
                         + np.asarray(value, dtype='<c16').tobytes(order='C'))
        records[key] = record
    output.mkdir()
    for key, payload in payloads.items():
        (output / (key + '.bin')).write_bytes(payload)
    return dict(origin='frozen_reference', cache=str(cache.resolve()), selected_iq=iq,
                metadata_sha256=metadata_sha, complete_sha256=sha(complete_path),
                raw_auxiliary_dimension=raw, whitened_auxiliary_rank=rank, arrays=records,
                metric_sha256=sha(output / 'coulomb_metric.bin'),
                whitening_sha256=sha(output / 'coulomb_whitening.bin'),
                reselected_or_padded=False, identity_error_is_diagnostic_only=True)


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


def validate_source_extension_evidence(reuse, gamma):
    """Admit a source export, never the regenerated Hamiltonian or its energy."""
    qrows = reuse.get('per_q', [])
    if (reuse.get('status') != 'success'
            or reuse.get('original_gamma_operator_reuse_gate') != 'pass'
            or reuse.get('failure_reasons') != []
            or reuse.get('compared_space') != 'existing_spd_subblocks_only'
            or reuse.get('regenerated_hamiltonian_admitted') is not False
            or reuse.get('missing_finite_q_sources_admitted') is not False
            or [q.get('selected_iq') for q in qrows] != [1, 22, 43, 6, 27, 23, 11, 55]):
        raise ValueError('complete original all-q reuse evidence required')
    for q in qrows:
        rows = q.get('per_k', [])
        if (q.get('pass_gate') is not True or len(rows) != 64
                or not all(r.get('pass_gate') is True for r in rows)
                or sorted(r['source_ik'] for r in rows) != list(range(1, 65))
                or sorted(r['target_ik'] for r in rows) != list(range(1, 65))):
            raise ValueError('all-q reuse failed or k routing incomplete')
    rows = gamma.get('per_k', [])
    if (gamma.get('status') != 'success' or len(rows) != 64
            or sorted(r['ik'] for r in rows) != list(range(1, 65))):
        raise ValueError('complete Gamma source audit required')
    values = [(gamma['metric']['relative'], 1e-8), (gamma['auxiliary_map_unitarity'], 5e-6)]
    for row in rows:
        values.extend([(row['overlap']['max_abs'], 1e-10),
            (row['occupied']['relative'], 1e-6), (row['occupied']['unitarity'], 1e-6),
            (row['occupied_energy_ry']['max_abs'], 1e-6),
            (row['occupied_energy_commutator_ry'], 1e-6), (row['source']['relative'], 1e-6)])
    if any(not math.isfinite(value) or value < 0 or value > limit for value, limit in values):
        raise ValueError('Gamma source/occupied/auxiliary compatibility failed')
    return dict(hamiltonian_origin='original_Gamma_at_target_k',
        overlap_origin='original_Gamma_at_target_k', occupied_origin='original_Gamma_at_target_k',
        regenerated_hamiltonian_admitted=False,
        new_source_admission='pending_finite_q_spd_check',
        gamma_full_operator_gate=gamma['gamma_operator_compatibility_gate'],
        gamma_full_operator_failure_reasons=gamma['failure_reasons'])


def validate_finite_q_pilot(result):
    rows = result.get('per_k', [])
    if (result.get('status') != 'success'
            or result.get('finite_q_spd_source_compatibility') != 'pass'
            or result.get('failure_reasons') != [] or result.get('selected_iq') != 22
            or result.get('compared_primitive_count') != 558
            or result.get('full_source_primitive_count') != 1550
            or result.get('regenerated_hamiltonian_read') is not False
            or result.get('regenerated_hamiltonian_admitted') is not False
            or len(rows) != 64 or not all(r.get('pass_gate') is True for r in rows)
            or sorted(r['source_ik'] for r in rows) != list(range(1, 65))
            or sorted(r['target_ik'] for r in rows) != list(range(1, 65))):
        raise ValueError('complete accepted q2 source pilot required')
    values = [(result['metric']['relative'], 1e-8), (result['auxiliary_map_unitarity'], 5e-6)]
    for row in rows:
        values.extend([(row['source']['relative'], 1e-6), (row['overlap']['max_abs'], 1e-10),
            (row['occupied']['relative'], 1e-6), (row['occupied']['unitarity'], 1e-6),
            (row['source_eigenvalue_ha']['max_abs'], 5e-7),
            (row['occupied_energy_commutator_ha'], 5e-7)])
    if any(not math.isfinite(v) or not 0 <= v <= limit for v, limit in values):
        raise ValueError('q2 source pilot numerical gate failed')
    return dict(full_q_admitted=False, regenerated_hamiltonian_admitted=False)


def source_extension_contract(iq, reference, reuse_path, gamma_path, pilot_path=None):
    if iq not in FINITE_Q_SPECS:
        raise ValueError('source extension requires a canonical finite q')
    pilot_contract = {}
    if iq != 22 or pilot_path is not None:
        if pilot_path is None or sha(pilot_path) != Q2_SOURCE_AUDIT_SHA:
            raise ValueError('remaining finite-q source extension requires locked q2 pilot')
        pilot = json.loads(pilot_path.read_text())
        validate_finite_q_pilot(pilot)
        pilot_contract = dict(source_extension_pilot_audit=str(pilot_path.resolve()),
                              source_extension_pilot_sha256=Q2_SOURCE_AUDIT_SHA)
    if (reuse_path is None or gamma_path is None
            or sha(reuse_path) != REFERENCE_REUSE_SHA or sha(gamma_path) != GAMMA_SOURCE_AUDIT_SHA):
        raise ValueError('source extension requires the two locked independent audits')
    reuse, gamma = json.loads(reuse_path.read_text()), json.loads(gamma_path.read_text())
    contract = validate_source_extension_evidence(reuse, gamma)
    raw = Path(reuse['reference']).resolve()
    if reference.resolve() not in raw.parents or raw.name != 'STERNHEIMER_BASIS_OPT_V1':
        raise ValueError('original operator reference path mismatch')
    for name, digest in reuse['reference_hashes'].items():
        path = (raw/name).resolve()
        if raw not in path.parents or sha(path) != digest:
            raise ValueError('original operator archive hash mismatch: '+name)
    contract.update(pilot_contract)
    contract.update(scope='finite_q_source_extension_original_operators_only', selected_iq=iq,
        reference_reuse_audit=str(reuse_path.resolve()), reference_reuse_sha256=sha(reuse_path),
        gamma_source_audit=str(gamma_path.resolve()), gamma_source_audit_sha256=sha(gamma_path),
        original_operator_directory=str(raw), original_operator_hashes=reuse['reference_hashes'],
        downstream_required_gate='finite_q_spd_source_compatibility')
    return contract


def prepare(reference, build_root, output, iq=1, gamma_acceptance=None,
            source_extension_reference_reuse=None, source_extension_gamma_audit=None,
            frozen_auxiliary_cache=None, source_extension_pilot_audit=None,
            bessel_radial_rows=None, bessel_nao_ecut=None):
    if source_extension_pilot_audit is not None and frozen_auxiliary_cache is None:
        raise ValueError('post-pilot exports require frozen auxiliary inputs')
    if frozen_auxiliary_cache is not None and (iq not in FINITE_Q_SPECS or source_extension_reference_reuse is None
                                              or source_extension_gamma_audit is None):
        raise ValueError('frozen auxiliary mode requires the admitted source-extension contract')
    if iq != 22 and source_extension_reference_reuse is not None and frozen_auxiliary_cache is None:
        raise ValueError('remaining source-extension exports require frozen auxiliary inputs')
    if source_extension_reference_reuse is not None or source_extension_gamma_audit is not None:
        if gamma_acceptance is not None:
            raise ValueError('operator and source-extension modes are mutually exclusive')
        q_contract = source_extension_contract(iq, reference, source_extension_reference_reuse,
                                             source_extension_gamma_audit, source_extension_pilot_audit)
    else:
        q_contract = export_q_contract(iq, gamma_acceptance)
    acceptance = json.loads((build_root/'result/BUILD_ACCEPTANCE.json').read_text())
    binary = build_root/'build/abacus_3p'
    required_commit = (FROZEN_AUXILIARY_COMMIT if frozen_auxiliary_cache is not None
                       else 'e1d0e64591fbd0324533c25cffbd6f94f6a1da4f')
    if (acceptance['status'] != 'success'
            or acceptance['source_commit'] != required_commit
            or acceptance['binary_sha256'] != sha(binary)):
        raise ValueError('accepted operators-only build required')
    values = input_values((reference/'INPUT').read_text())
    expected = dict(calculation='scf', nspin='1', nbands='44', basis_type='lcao',
        nx='24', ny='24', nz='24', sternheimer_fd_order='8', sternheimer_nfreq='12',
        sternheimer_q_index='1', exx_pca_threshold='1e-6', sternheimer_siab_lmax='4')
    if any(values.get(k) != v for k, v in expected.items()):
        raise ValueError('frozen Gamma INPUT mismatch')
    expansion = None
    if (bessel_radial_rows is None) != (bessel_nao_ecut is None):
        raise ValueError('expanded mother requires both radial rows and cutoff')
    if bessel_radial_rows is not None:
        values, expansion = configure_expanded_mother(
            values, radial_rows=bessel_radial_rows,
            bessel_nao_ecut=bessel_nao_ecut)
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
    if frozen_auxiliary_cache is not None:
        q_contract['frozen_auxiliary'] = freeze_auxiliary_cache(
            frozen_auxiliary_cache, output/'frozen-auxiliary', iq, FINITE_Q_SPECS[iq][2])
    inputs = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    contract = dict(q_contract,
        reference=str(reference), reference_manifest_sha256=sha(manifest),
        density_sha256=sha(density), source_commit=acceptance['source_commit'],
        build_acceptance_sha256=sha(build_root/'result/BUILD_ACCEPTANCE.json'),
        binary=str(binary), binary_sha256=sha(binary), inputs=inputs, suffix=suffix,
        gamma_coulomb_reuse=coulomb_reuse,
        first_order_equations_allowed=0, self_consistency_allowed=False,
        full_q_admitted=False, physical_release_gate='hold')
    if expansion is not None:
        contract['primitive_expansion'] = expansion
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
    parser.add_argument('--source-extension-reference-reuse', type=Path)
    parser.add_argument('--source-extension-gamma-audit', type=Path)
    parser.add_argument('--frozen-auxiliary-cache', type=Path)
    parser.add_argument('--source-extension-pilot-audit', type=Path)
    parser.add_argument('--bessel-radial-rows', type=int)
    parser.add_argument('--bessel-nao-ecut', type=float)
    args = parser.parse_args()
    prepare(args.reference, args.build_root, args.output, args.iq, args.gamma_acceptance,
            args.source_extension_reference_reuse, args.source_extension_gamma_audit,
            args.frozen_auxiliary_cache, args.source_extension_pilot_audit,
            args.bessel_radial_rows, args.bessel_nao_ecut)
