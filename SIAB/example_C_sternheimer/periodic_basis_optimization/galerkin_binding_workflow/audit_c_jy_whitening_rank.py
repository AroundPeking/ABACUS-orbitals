"""Read-only diagnosis of changed Coulomb whitening rank; never admits sources."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_c_jy_operator_restart import OperatorFiles, difference, metric_signs, sha


def compare_whitening(v, w, vn, wn, threshold):
    if (not np.isfinite(threshold) or not 0 < threshold < 1
            or v.ndim != 2 or v.shape[0] != v.shape[1] or v.shape != vn.shape
            or w.ndim != 2 or wn.ndim != 2 or not w.size or not wn.size
            or w.shape[0] != len(v) or wn.shape[0] != len(v)
            or not all(np.isfinite(x).all() for x in (v, w, vn, wn))):
        raise ValueError('invalid metric, whitening, or cutoff')
    for metric in (v, vn):
        if np.max(np.abs(metric-metric.conj().T)) > 1e-10*max(1., np.max(np.abs(metric))):
            raise ValueError('non-Hermitian metric')
    signs = metric_signs(v, vn)
    aligned = signs[:, None]*vn*signs[None, :]
    en = [np.linalg.eigvalsh((x+x.conj().T)*.5) for x in (v, vn)]
    if any(e[-1] <= 0 or e[0] < -max(threshold, 1e-12)*e[-1] for e in en):
        raise ValueError('invalid Coulomb spectrum')
    ranks = [int(np.count_nonzero(e > threshold*e[-1])) for e in en]
    stored = [w.shape[1], wn.shape[1]]
    if any(s > r for s, r in zip(stored, ranks)):
        raise ValueError('stored rank exceeds threshold eigenspace')
    t = w.conj().T @ v @ (signs[:, None]*wn)
    return dict(status='success', metric=difference(v, aligned),
        relative_threshold=threshold, threshold_ranks=ranks, stored_ranks=stored,
        post_threshold_discarded=[r-s for r, s in zip(ranks, stored)],
        raw_eigenvalues=[e.tolist() for e in en],
        retained_identity_max_abs=[float(np.max(np.abs(x.conj().T @ m @ x-np.eye(x.shape[1]))))
                                   for m, x in ((v, w), (vn, wn))],
        reference_whitening_on_new_metric_max_abs=float(np.max(np.abs(
            w.conj().T @ aligned @ w-np.eye(w.shape[1])))),
        reference_space_projector_loss_fro=float(np.linalg.norm(t @ t.conj().T-np.eye(w.shape[1]))),
        raw_auxiliary_signs=signs.tolist(),
        rectangular_map_singular_values=np.linalg.svd(t, compute_uv=False).tolist(),
        source_reconstruction_admitted=False, physical_release_gate='hold',
        interpretation='rank diagnostics only; a missing reference channel cannot be padded or dropped')


def audit(reference, operators, output):
    complete = json.loads((reference/'COMPLETE.json').read_text())
    if (complete['status'] != 'complete'
            or sha(reference/'dataset.json') != complete['metadata_sha256']):
        raise ValueError('reference cache metadata not complete')
    metadata = json.loads((reference/'dataset.json').read_text())
    arrays, hashes = [], {}
    for name in ('coulomb_metric', 'coulomb_whitening'):
        record = metadata['arrays'][metadata['dataset']['fields'][name]['index']]
        path = (reference/record['file']).resolve()
        if reference.resolve() not in path.parents or sha(path) != record['sha256']:
            raise ValueError('reference matrix hash mismatch')
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != record['shape'] or array.dtype.str != record['dtype']:
            raise ValueError('reference matrix schema mismatch')
        arrays.append(array)
        hashes[path.name] = sha(path)
    new = OperatorFiles(operators, True)
    scalar = metadata['source']['scalar']
    for key in ('selected_iq', 'raw_auxiliary_dimension', 'coulomb_relative_threshold',
                'orbital_sha256', 'pseudopotential_sha256', 'auxiliary_basis_sha256'):
        if new.scalar[key] != ' '.join(scalar[key]):
            raise ValueError('changed physical input: '+key)
    result = compare_whitening(*arrays, new.array(4), new.array(5),
                               float(new.scalar['coulomb_relative_threshold']))
    result.update(reference_matrix_hashes=hashes, native_hashes=new.hashes,
        reference_complete_sha256=sha(reference/'COMPLETE.json'),
        reference_dataset_sha256=sha(reference/'dataset.json'),
        source_files_sha256={p.name: sha(p) for p in (Path(__file__),
            Path(__file__).with_name('audit_c_jy_operator_restart.py'))},
        numpy_version=np.__version__, reference=str(reference), operators=str(operators))
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'operators', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.reference, args.operators, args.output)
