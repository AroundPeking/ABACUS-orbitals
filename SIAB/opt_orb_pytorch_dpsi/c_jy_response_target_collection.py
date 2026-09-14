"""Validate and index a complete solid jY response-target collection."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from c_jy_response_targets import validate_target_manifest


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(2**20), b''):
            digest.update(data)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _within(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('invalid relative target path')
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError('target sector path escapes q directory')
    return path


def _validate_primitive_blocks(blocks, primitive_count, lmax):
    if not isinstance(blocks, list) or not blocks:
        raise ValueError('primitive block list required')
    offset = 0
    for block in blocks:
        if (not isinstance(block, dict) or block.get('element') != 'C'
                or type(block.get('atom_index')) is not int or block['atom_index'] < 0
                or type(block.get('l')) is not int or not 0 <= block['l'] <= lmax
                or type(block.get('m')) is not int or not -block['l'] <= block['m'] <= block['l']
                or type(block.get('n_primitive')) is not int or block['n_primitive'] <= 0
                or block.get('offset') != offset):
            raise ValueError('invalid or discontinuous primitive block list')
        offset += block['n_primitive']
    if offset != primitive_count:
        raise ValueError('primitive blocks do not cover target mother space')


def load_angular_primitive_blocks(path, expected_sha256, *, source_primitive_count, lmax):
    """Read the exported mother layout and reindex complete blocks through lmax."""
    path = Path(path)
    if (not path.is_file() or _sha(path) != expected_sha256
            or type(source_primitive_count) is not int or source_primitive_count <= 0
            or type(lmax) is not int or lmax < 0):
        raise ValueError('primitive block source contract mismatch')
    lines = [line.strip() for line in path.read_text(encoding='ascii').splitlines()
             if line.strip()]
    if not lines or lines[0] != 'ABACUS_STERNHEIMER_BASIS_OPT_PRIMITIVES_V1':
        raise ValueError('invalid primitive block file version')
    source_offset = 0
    retained_offset = 0
    retained = []
    for line in lines[1:]:
        if line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError('invalid primitive block record')
        element, atom_index, angular, magnetic, count, offset = fields
        atom_index, angular, magnetic = int(atom_index), int(angular), int(magnetic)
        count, offset = int(count), int(offset)
        if (not element or atom_index < 0 or angular < 0
                or not -angular <= magnetic <= angular or count <= 0
                or offset != source_offset):
            raise ValueError('invalid or discontinuous source primitive blocks')
        source_offset += count
        if angular <= lmax:
            retained.append(dict(element=element, atom_index=atom_index, l=angular,
                m=magnetic, n_primitive=count, offset=retained_offset))
            retained_offset += count
    if source_offset != source_primitive_count:
        raise ValueError('source primitive blocks do not cover mother space')
    _validate_primitive_blocks(retained, retained_offset, lmax)
    return retained


def collect_target_collection(root, primitive_blocks, *, q_slots=tuple(range(8)),
                              q_labels=(1, 2, 3, 6, 7, 8, 11, 28),
                              selected_iq=(1, 22, 43, 6, 27, 23, 11, 55),
                              multiplicities=(1, 8, 4, 6, 24, 12, 3, 6),
                              k_record_count=64, frequency_count=12,
                              primitive_count=992, lmax=3,
                              star_weight_denominator=64):
    """Return a hash-bound index after validating every local q/k target.

    The result retains one coordinate frame per q/k sector.  It deliberately
    does not concatenate embeddings or promote the diagnostic PI matrices to
    fitting targets.
    """
    root = Path(root).resolve()
    q_slots = tuple(q_slots)
    q_labels = tuple(q_labels)
    selected_iq = tuple(selected_iq)
    multiplicities = tuple(multiplicities)
    if (not root.is_dir() or not q_slots
            or not len(q_slots) == len(q_labels) == len(selected_iq) == len(multiplicities)
            or type(star_weight_denominator) is not int or star_weight_denominator <= 0
            or sum(multiplicities) != star_weight_denominator):
        raise ValueError('invalid complete q-star collection contract')
    _validate_primitive_blocks(primitive_blocks, primitive_count, lmax)

    per_q = []
    target_files = []
    frequencies = weights = None
    total_norm2 = 0.
    for slot, label, iq, multiplicity in zip(q_slots, q_labels, selected_iq, multiplicities):
        qroot = root/('q%02d' % slot)
        target_root = qroot/'targets'
        result_root = qroot/'result'
        manifest_path = target_root/'TARGETS.json'
        manifest = _read(manifest_path)
        validate_target_manifest(manifest, lmax=lmax, frequency_count=frequency_count,
            k_record_count=k_record_count, primitive_count=primitive_count, q_slots=(slot,))
        if (manifest.get('q_slot') != slot or manifest.get('selected_iq') != iq
                or not math.isclose(float(manifest.get('q_weight', -1)),
                                    multiplicity/star_weight_denominator,
                                    rel_tol=0., abs_tol=1e-15)):
            raise ValueError('q identity or star weight mismatch')
        current_frequencies = tuple(float(x) for x in manifest['frequency_ha'])
        current_weights = tuple(float(x) for x in manifest['frequency_weights_ha'])
        if (len(current_frequencies) != frequency_count
                or len(current_weights) != frequency_count
                or not all(math.isfinite(x) for x in current_frequencies+current_weights)
                or not all(x > 0 for x in current_weights)):
            raise ValueError('invalid frequency quadrature')
        if frequencies is None:
            frequencies, weights = current_frequencies, current_weights
        elif frequencies != current_frequencies or weights != current_weights:
            raise ValueError('q-dependent frequency quadrature')

        result = _read(result_root/'RESULT.json')
        provenance = _read(result_root/'PROVENANCE.json')
        runtime = _read(qroot/'RUNTIME_CONTRACT.json')
        if (result.get('status') != 'success' or provenance.get('status') != 'success'
                or result.get('q_slot', slot) != slot
                or result.get('selected_iq', iq) != iq
                or provenance.get('source_commit') != runtime.get('source_commit')
                or provenance.get('job_id') != runtime.get('job_id')):
            raise ValueError('q result or provenance contract mismatch')
        for relative, expected in provenance.get('files', {}).items():
            path = _within(result_root, relative)
            if not path.is_file() or _sha(path) != expected:
                raise ValueError('q result provenance hash mismatch')

        q_norm2 = 0.
        source_ik = set()
        for sector in manifest['sectors']:
            path = _within(target_root, sector['file'])
            if not path.is_file() or _sha(path) != sector.get('file_sha256'):
                raise ValueError('sector file hash mismatch')
            with np.load(path, allow_pickle=False) as arrays:
                covariance = arrays['covariance']
                embedding = arrays['embedding']
                if (covariance.shape != (sector['covariance_dimension'],)*2
                        or embedding.shape != (sector['embedding_rows'], sector['embedding_columns'])
                        or not np.isfinite(covariance).all() or not np.isfinite(embedding).all()):
                    raise ValueError('sector array shape or finite-value mismatch')
                norm2 = float(np.trace(covariance).real)
            if not math.isclose(norm2, float(sector['target_norm2']), rel_tol=1e-11, abs_tol=1e-14):
                raise ValueError('sector covariance norm mismatch')
            source_ik.add(sector.get('source_ik'))
            q_norm2 += norm2
            target_files.append(dict(q_slot=slot,
                file=str(path.relative_to(root)), sha256=sector['file_sha256']))
        if source_ik != set(range(1, k_record_count+1)):
            raise ValueError('source k coverage mismatch')

        diagnostic_path = target_root/'PI_DIAGNOSTIC.npy'
        accepted_path = result_root/('lmax%d' % lmax)/'PI.npy'
        diagnostic = np.load(diagnostic_path, allow_pickle=False)
        accepted = np.load(accepted_path, allow_pickle=False)
        if diagnostic.shape != accepted.shape or not np.isfinite(diagnostic).all():
            raise ValueError('diagnostic PI shape or finite-value mismatch')
        difference = np.abs(diagnostic-accepted)
        maximum = float(difference.max(initial=0.))
        relative = float(np.linalg.norm(diagnostic-accepted)
                         / max(float(np.linalg.norm(accepted)), 1e-300))
        if maximum != 0. or relative != 0.:
            raise ValueError('target PI does not reproduce accepted jY response')
        per_q.append(dict(q_slot=slot, q_label=label, selected_iq=iq,
            multiplicity=multiplicity, q_weight=multiplicity/star_weight_denominator,
            source_commit=runtime['source_commit'], job_id=runtime['job_id'],
            target_manifest_sha256=_sha(manifest_path),
            diagnostic_pi_sha256=_sha(diagnostic_path), accepted_pi_sha256=_sha(accepted_path),
            target_sector_count=len(manifest['sectors']), target_norm2=q_norm2,
            pi_reconstruction_max_abs=maximum, pi_reconstruction_relative_error=relative))
        total_norm2 += q_norm2

    return dict(status='success', target_kind='response_covariance_embedding',
        assembled_pi=False, lmax=lmax, primitive_count=primitive_count,
        primitive_blocks=primitive_blocks, q_array_slots=list(q_slots),
        q_labels=list(q_labels), selected_iq=list(selected_iq),
        multiplicities=list(multiplicities), star_weight_denominator=star_weight_denominator,
        frequency_count=frequency_count,
        frequency_ha=list(frequencies), frequency_weights_ha=list(weights),
        k_record_count_per_q=k_record_count,
        target_sector_count=len(q_slots)*k_record_count,
        target_norm2=total_norm2, target_files=target_files, per_q=per_q,
        coordinate_merge='none_per_q_source_target_record',
        target_completeness_gate='pass', physical_release_gate='hold')
