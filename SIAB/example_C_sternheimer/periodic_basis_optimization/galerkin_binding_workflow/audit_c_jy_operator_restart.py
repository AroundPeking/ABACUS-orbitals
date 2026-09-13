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
    if new.scalar['frozen_charge_sha256'] != contract['density_sha256']:
        raise ValueError('density provenance mismatch')
    for key in ('orbital_sha256', 'pseudopotential_sha256', 'auxiliary_basis_source', 'auxiliary_basis_sha256',
                'primitive_blocks_sha256', 'kernel', 'selected_iq', 'q_count', 'k_count', 'qpoint', 'q_weight',
                'primitive_count', 'raw_auxiliary_dimension', 'whitened_auxiliary_rank'):
        if old.scalar[key] != new.scalar[key]:
            raise ValueError('incompatible operator metadata: '+key)
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
    rows = []
    for ik in range(1, 65):
        if old.kpoints[ik][0] != ik or new.kpoints[ik][0] != ik:
            raise ValueError('not Gamma k routing')
        if difference(old.kpoints[ik][1], new.kpoints[ik][1])['max_abs'] > 1e-12:
            raise ValueError('kpoint/weight/occupation contract mismatch')
        s = difference(old.array(1, ik), new.array(1, ik))
        h = difference(old.array(6, ik), new.array(6, ik))
        a, occupied = occupied_gauge(old.array(7, ik), new.array(7, ik))
        energies = difference(old.eigenvalues[ik], new.eigenvalues[ik])
        commutator = float(np.max(np.abs(new.eigenvalues[ik][:, None]*a-a*old.eigenvalues[ik][None, :])))
        nocc, naux = len(a), t.shape[0]
        source_old = old.array(2, ik).reshape(nocc, naux, -1)
        source_new = new.array(2, ik).reshape(nocc, naux, -1)
        d = difference(source_new, transform_source(source_old, a, t))
        passed = (s['max_abs'] <= 1e-10 and h['max_abs'] <= 1e-6
            and energies['max_abs'] <= 1e-6 and commutator <= 1e-6
            and occupied['relative'] <= 1e-6 and occupied['unitarity'] <= 1e-6
            and d['relative'] <= 1e-6)
        row = dict(ik=ik, pass_gate=passed, overlap=s, hamiltonian_ry=h, occupied=occupied,
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
    output.mkdir()
    (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'STATUS').write_text('success\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
