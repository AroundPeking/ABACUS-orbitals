"""Compare Gamma operators to accepted full-response data, allowing only tested gauges."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np

HEADER = struct.Struct('<16sIIiiiQQ')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def difference(reference, candidate):
    if (reference.shape != candidate.shape or not reference.size
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()):
        raise ValueError('incompatible or nonfinite comparison arrays')
    delta = candidate-reference
    return dict(relative=float(np.linalg.norm(delta)/max(np.linalg.norm(reference), 1e-300)),
                max_abs=float(np.max(np.abs(delta))))


def occupied_gauge(old, new):
    gauge = np.linalg.solve((old @ old.conj().T).T, (new @ old.conj().T).T).T
    result = difference(new, gauge @ old)
    result['unitarity'] = float(np.max(np.abs(gauge @ gauge.conj().T-np.eye(len(gauge)))))
    return gauge, result


def transform_source(source, occupied_map, auxiliary_map):
    return np.einsum('nm,hj,mjp->nhp', occupied_map, auxiliary_map.conj().T, source, optimize=True)


def _primitive_blocks(text):
    lines = text.splitlines()
    if (not lines or lines[0] != 'ABACUS_STERNHEIMER_BASIS_OPT_PRIMITIVES_V1'):
        raise ValueError('invalid primitive block header')
    rows = []
    for line in lines[1:]:
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError('invalid primitive block row')
        element = fields[0]
        try:
            values = tuple(int(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError('invalid primitive block integer') from error
        rows.append((element,) + values)
    return rows


def validate_nested_primitive_blocks(old_text, new_text, *, source_rows,
                                     target_rows, lmax, natom):
    old, new = _primitive_blocks(old_text), _primitive_blocks(new_text)
    expected_blocks = natom * (lmax + 1) ** 2
    if len(old) != expected_blocks or len(new) != expected_blocks:
        raise ValueError('expanded primitive blocks are incomplete')
    indices = []
    for block, (old_row, new_row) in enumerate(zip(old, new)):
        element, atom, l, m, old_count, old_offset = old_row
        expected = ('C', atom, l, m)
        if (new_row[:4] != expected or old_row[:4] != expected
                or old_count != source_rows or new_row[4] != target_rows
                or old_offset != block * source_rows
                or new_row[5] != block * target_rows):
            raise ValueError('expanded primitive blocks are not a nested radial prefix')
        indices.extend(range(new_row[5], new_row[5] + source_rows))
    return tuple(indices)


def nested_square_prefix(array, indices):
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError('square expanded operator required')
    return array[np.ix_(indices, indices)]


def nested_column_prefix(array, indices):
    if array.ndim < 2 or array.shape[-1] <= max(indices):
        raise ValueError('expanded operator columns are incomplete')
    return array[..., indices]


def expanded_overlap_compatible(diagnostics, reference_max_abs=0.0):
    """Bound observed quadrature roundoff by both absolute and relative error."""
    absolute_limit = max(2e-10, reference_max_abs*1e-12)
    return (np.isfinite(reference_max_abs) and reference_max_abs >= 0
            and diagnostics['max_abs'] <= absolute_limit
            and diagnostics['relative'] <= 1e-11)


def metric_signs(old, new):
    if old.shape != new.shape or old.ndim != 2 or old.shape[0] != old.shape[1]:
        raise ValueError('square matched Coulomb metrics required')
    signs = np.zeros(len(old))
    cutoff = max(np.max(np.abs(old)), np.max(np.abs(new)))*1e-8
    for root in range(len(old)):
        if signs[root]:
            continue
        signs[root] = 1
        stack = [root]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero((np.abs(old[i]) > cutoff) & (np.abs(new[i]) > cutoff)):
                if not signs[j]:
                    signs[j] = signs[i] * (1 if (old[i, j]*new[i, j].conjugate()).real >= 0 else -1)
                    stack.append(j)
    return signs


class OperatorFiles:
    def __init__(self, directory, operators_only):
        self.directory = Path(directory).resolve()
        self.hashes = {}
        self.scalar, self.frequencies, self.kpoints, self.eigenvalues, self.entries = {}, [], {}, {}, {}
        path = self.directory/'manifest.dat'
        self.hashes['manifest.dat'] = sha(path)
        lines = path.read_text().splitlines()
        expected = 'ABACUS_STERNHEIMER_BASIS_' + ('OPERATORS' if operators_only else 'OPT') + '_MANIFEST_V1'
        if not lines or lines[0] != expected:
            raise ValueError('wrong manifest kind')
        for line in lines[1:]:
            fields = line.split()
            key = fields[0]
            if key == 'frequency':
                if int(fields[1]) != len(self.frequencies) or len(fields) != 4:
                    raise ValueError('invalid frequency sequence')
                self.frequencies.append([float(v) for v in fields[2:]])
            elif key == 'kpoint':
                ik, target, count = int(fields[1]), int(fields[2]), int(fields[13])
                if ik in self.kpoints or len(fields) != 14+count:
                    raise ValueError('invalid k record')
                self.kpoints[ik] = (target, np.array([float(v) for v in fields[3:]]))
            elif key == 'eigenvalues_ry':
                ik, count = int(fields[1]), int(fields[2])
                if ik in self.eigenvalues or len(fields) != 3+count:
                    raise ValueError('invalid occupied energy record')
                self.eigenvalues[ik] = np.array([float(v) for v in fields[3:]])
            elif key == 'entry':
                row = line.split('\t')
                kind, iq, ik, freq, nr, nc = [int(v) for v in row[1:7]]
                index = kind, ik, freq
                if len(row) != 12 or index in self.entries:
                    raise ValueError('invalid or duplicate operator entry')
                self.entries[index] = dict(kind=kind, iq=iq, ik=ik, freq=freq,
                    rows=nr, columns=nc, path=row[10], sha256=row[11])
            else:
                if key in self.scalar:
                    raise ValueError('duplicate manifest scalar')
                self.scalar[key] = ' '.join(fields[1:])
        if len(self.entries) != int(self.scalar['entry_count']):
            raise ValueError('incomplete manifest entries')
        status = dict(line.split() for line in (self.directory/'status.dat').read_text().splitlines() if line.strip())
        if status.get('status') != 'success' or status.get('physics_hash') != self.scalar['physics_hash']:
            raise ValueError('failed or inconsistent producer status')
        if operators_only:
            expected_keys = {(kind, ik, -1) for kind in (1, 2, 6, 7) for ik in range(1, 65)} | {(4, 0, -1), (5, 0, -1)}
            if (set(self.entries) != expected_keys or status.get('response_solved') != 'no'
                    or status.get('solved_equations') != '0' or self.scalar.get('response_solved') != 'no'):
                raise ValueError('not a complete operators-only dataset')
        elif status.get('all_converged') != 'yes':
            raise ValueError('reference equations not accepted')

    def array(self, kind, ik=0):
        entry = self.entries[kind, ik, -1]
        path = (self.directory/entry['path']).resolve()
        if self.directory not in path.parents or sha(path) != entry['sha256']:
            raise ValueError('invalid path or operator hash')
        with path.open('rb') as stream:
            header = HEADER.unpack(stream.read(HEADER.size))
            expected = (b'ABACUS_STBOPT_V1', 1, kind, entry['iq'], ik, -1, entry['rows'], entry['columns'])
            if header != expected or path.stat().st_size != HEADER.size + entry['rows']*entry['columns']*16:
                raise ValueError('operator header or payload size mismatch')
            values = np.fromfile(stream, dtype='<c16').reshape(entry['rows'], entry['columns'])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite operator')
        self.hashes[entry['path']] = entry['sha256']
        return values


def audit(run, output):
    start = time.perf_counter()
    if (run/'STATUS').read_text().strip() != 'success':
        raise ValueError('operator job has not succeeded')
    contract = json.loads((run/'CONTRACT.json').read_text())
    old_dir = Path(contract['reference'])/('OUT.'+contract['suffix'])/'STERNHEIMER_BASIS_OPT_V1'
    if sha(old_dir/'manifest.dat') != contract['reference_manifest_sha256']:
        raise ValueError('reference manifest changed')
    old = OperatorFiles(old_dir, False)
    new = OperatorFiles(run/('OUT.'+contract['suffix'])/'STERNHEIMER_BASIS_OPERATORS_V1', True)
    expansion = contract.get('primitive_expansion')
    prefix = None
    if expansion is not None:
        target_rows = expansion.get('expanded_radial_rows')
        full_count = expansion.get('expanded_primitive_count')
        if (type(target_rows) is not int or target_rows <= 31
                or full_count != target_rows * 25 * 2):
            raise ValueError('unexpected expanded mother dimensions')
        expected = dict(source_radial_rows=31,
                        expanded_radial_rows=target_rows,
                        source_primitive_count=1550,
                        expanded_primitive_count=full_count,
                        expanded_spdf_primitive_count=target_rows * 16 * 2)
        if any(expansion.get(key) != value for key, value in expected.items()):
            raise ValueError('unexpected expanded mother contract')
        prefix = validate_nested_primitive_blocks(
            (old.directory/'primitive_blocks.dat').read_text(),
            (new.directory/'primitive_blocks.dat').read_text(),
            source_rows=31, target_rows=target_rows, lmax=4, natom=2)
        if len(prefix) != 1550 or len(set(prefix)) != 1550:
            raise ValueError('expanded primitive prefix mapping is incomplete')
        prefix_sha = hashlib.sha256(
            struct.pack('<%dI' % len(prefix), *prefix)).hexdigest()
        if prefix_sha != expansion.get('source_prefix_indices_sha256'):
            raise ValueError('expanded primitive prefix hash mismatch')
        old.hashes['primitive_blocks.dat'] = sha(old.directory/'primitive_blocks.dat')
        new.hashes['primitive_blocks.dat'] = sha(new.directory/'primitive_blocks.dat')
    if new.scalar['frozen_charge_sha256'] != contract['density_sha256']:
        raise ValueError('density provenance mismatch')
    common_metadata = ('orbital_sha256', 'pseudopotential_sha256',
        'auxiliary_basis_source', 'auxiliary_basis_sha256', 'kernel',
        'selected_iq', 'q_count', 'k_count', 'qpoint', 'q_weight',
        'raw_auxiliary_dimension', 'whitened_auxiliary_rank')
    for key in common_metadata:
        if old.scalar[key] != new.scalar[key]:
            raise ValueError('incompatible operator metadata: '+key)
    if prefix is None:
        for key in ('primitive_blocks_sha256', 'primitive_count'):
            if old.scalar[key] != new.scalar[key]:
                raise ValueError('incompatible operator metadata: '+key)
    elif (old.scalar['primitive_count'] != '1550'
          or new.scalar['primitive_count'] != str(expansion['expanded_primitive_count'])):
        raise ValueError('expanded primitive count mismatch')
    if (new.scalar['selected_iq'] != '1' or new.scalar['k_count'] != '64'
            or len(new.frequencies) != 12 or new.frequencies != old.frequencies):
        raise ValueError('Gamma 64-k twelve-frequency compatibility check required')
    v, vn, w, wn = old.array(4), new.array(4), old.array(5), new.array(5)
    signs = metric_signs(v, vn)
    metric = difference(vn, signs[:, None]*v*signs[None, :])
    t = w.conj().T @ v @ (signs[:, None]*wn)
    unitary = float(np.max(np.abs(t.conj().T @ t-np.eye(t.shape[1]))))
    failures = []
    if metric['relative'] > 1e-8 or unitary > 5e-6:
        failures.append('auxiliary_gauge')
    rows, occupied_maps = [], {}
    for ik in range(1, 65):
        if old.kpoints[ik][0] != ik or new.kpoints[ik][0] != ik:
            raise ValueError('not Gamma k routing')
        if difference(old.kpoints[ik][1], new.kpoints[ik][1])['max_abs'] > 1e-12:
            raise ValueError('kpoint/weight/occupation contract mismatch')
        new_s, new_h, new_o = new.array(1, ik), new.array(6, ik), new.array(7, ik)
        if prefix is not None:
            new_s = nested_square_prefix(new_s, prefix)
            new_h = nested_square_prefix(new_h, prefix)
            new_o = nested_column_prefix(new_o, prefix)
        reference_s = old.array(1, ik)
        s = difference(reference_s, new_s)
        reference_overlap_max_abs = float(np.max(np.abs(reference_s)))
        h = difference(old.array(6, ik), new_h)
        a, occupied = occupied_gauge(old.array(7, ik), new_o)
        occupied_maps[ik] = a
        energies = difference(old.eigenvalues[ik], new.eigenvalues[ik])
        commutator = float(np.max(np.abs(new.eigenvalues[ik][:, None]*a-a*old.eigenvalues[ik][None, :])))
        nocc, naux = len(a), t.shape[0]
        source_old = old.array(2, ik).reshape(nocc, naux, -1)
        source_new = new.array(2, ik).reshape(nocc, naux, -1)
        if prefix is not None:
            source_new = nested_column_prefix(source_new, prefix)
        d = difference(source_new, transform_source(source_old, a, t))
        h_tolerance = 5e-6 if prefix is not None else 1e-6
        overlap_pass = (expanded_overlap_compatible(s, reference_overlap_max_abs)
                        if prefix is not None
                        else s['max_abs'] <= 1e-10)
        passed = (overlap_pass and h['max_abs'] <= h_tolerance
            and energies['max_abs'] <= 1e-6 and commutator <= 1e-6
            and occupied['relative'] <= 1e-6 and occupied['unitarity'] <= 1e-6
            and d['relative'] <= 1e-6)
        row = dict(ik=ik, pass_gate=passed, overlap=s,
                   reference_overlap_max_abs=reference_overlap_max_abs,
                   hamiltonian_ry=h, occupied=occupied,
                   occupied_energy_ry=energies, occupied_energy_commutator_ry=commutator, source=d)
        rows.append(row)
        if not passed:
            failures.append('k%d' % ik)
        print(json.dumps(row), flush=True)
    result = dict(status='success', gamma_operator_compatibility_gate='pass' if not failures else 'hold',
        failure_reasons=failures, metric=metric, auxiliary_map_unitarity=unitary,
        raw_auxiliary_signs=signs.tolist(), per_k=rows, elapsed_seconds=time.perf_counter()-start,
        hashes=dict(reference=old.hashes, operators=new.hashes, contract=sha(run/'CONTRACT.json')),
        full_q_admitted=False, physical_release_gate='hold')
    if prefix is not None:
        result.update(expanded_mother_prefix_gate='pass' if not failures else 'hold',
                      primitive_expansion=expansion,
                      prefix_index_count=len(prefix),
                      anchored_old_subblock_required=True,
                      regenerated_hamiltonian_admitted=False)
    output.mkdir()
    if prefix is not None:
        map_path = output/'GAUGE_MAPS.npz'
        with map_path.open('xb') as stream:
            np.savez(stream, auxiliary_map=t, raw_auxiliary_signs=signs,
                     **{'occupied_at_k%d' % ik: value
                        for ik, value in occupied_maps.items()})
        result['gauge_maps_sha256'] = sha(map_path)
    (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'STATUS').write_text('success\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
