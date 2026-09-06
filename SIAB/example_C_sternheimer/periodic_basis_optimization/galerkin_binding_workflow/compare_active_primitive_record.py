"""Bounded single-k equivalence diagnostic, never a physical RPA result.

All chunks of an already accepted q dataset are hash checked, but only global
matrices and one k record are retained. The production reader is unchanged.
"""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

OPT = Path(__file__).resolve().parents[3] / 'opt_orb_pytorch_dpsi'
sys.path.insert(0, str(OPT))

import torch

import periodic_galerkin_data as data
from periodic_galerkin_basis import contract_periodic_candidate_operators, read_periodic_optimizer_coefficients
from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
from periodic_galerkin_reduction import reduce_periodic_active_primitives
from periodic_galerkin_rpa import periodic_rpa_objective
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


def _check_hash(path, expected):
    data._require(data._sha256(str(path)) == expected, 'SHA256 mismatch: ' + str(path))


def _load_sample(freeze_path, freeze_sha256, label, source_ik):
    started = time.monotonic()
    freeze_path = Path(freeze_path)
    _check_hash(freeze_path, freeze_sha256)
    frozen = json.loads(freeze_path.read_text())
    data._require(frozen['status'] == 'success', 'parent freeze must be accepted')
    selected = [item for item in frozen['datasets'] if item['label'] == label]
    data._require(len(selected) == 1, 'q label must be unique in accepted freeze')
    item = selected[0]
    directory = Path(item['dataset'])
    for name, expected in (('manifest.dat', item['manifest_sha256']),
                           ('status.dat', item['status_sha256'])):
        _check_hash(directory / name, expected)
    _check_hash(item['acceptance'], item['acceptance_sha256'])
    status = data._read_status(str(directory))
    scalar, frequencies, kpoints, eigenvalues, entries = data._read_manifest(str(directory))
    value = lambda key: data._one(scalar, key)
    data._require(status['physics_hash'] == value('physics_hash') == item['physics_hash'],
                  'parent physics identity mismatch')
    data._require(value('kernel') == 'full_coulomb', 'full Coulomb required')
    data._require(source_ik in kpoints, 'source_ik is absent from parent dataset')
    nk, nf = int(value('k_count')), int(value('frequency_count'))
    data._require(set(kpoints) == set(eigenvalues) == set(range(1, nk+1)), 'incomplete parent k metadata')
    data._require(set(frequencies) == set(range(nf)), 'incomplete parent frequency grid')
    expected_keys = {(4, 0, -1), (5, 0, -1)} | {(8, 0, iw) for iw in range(nf)}
    for ik in kpoints:
        expected_keys.update((kind, ik, -1) for kind in (1, 2, 6, 7))
        expected_keys.update((3, ik, iw) for iw in range(nf))
    keys = [(e.kind, e.ik, e.ifrequency) for e in entries]
    data._require(len(keys) == len(set(keys)) == int(value('entry_count'))
                  and set(keys) == expected_keys, 'incomplete or duplicate parent chunk layout')
    n, raw, white = (int(value(k)) for k in ('primitive_count', 'raw_auxiliary_dimension', 'whitened_auxiliary_rank'))
    blocks = data._read_primitive_blocks(str(directory), value('primitive_blocks_sha256'), n)
    chunks, hashed_bytes = {}, 0
    for entry in entries:
        data._validate_chunk_layout(str(directory), entry)
        key = (entry.kind, entry.ik, entry.ifrequency)
        if entry.ik == 0 or (entry.ik == source_ik and entry.kind != 3):
            chunks[key] = data._read_chunk(str(directory), entry)
        else:
            path = data._safe_chunk_path(str(directory), entry.relative_path)
            _check_hash(path, entry.sha256)
        hashed_bytes += data._HEADER.size + 16 * entry.rows * entry.columns
    metadata = kpoints[source_ik]
    occupied = len(metadata['occupation'])
    record = data.PeriodicGalerkinKPoint(
        source_ik=source_ik, target_ik=metadata['target_ik'],
        source_kpoint=metadata['source_kpoint'], target_kpoint=metadata['target_kpoint'],
        reciprocal_shift=metadata['reciprocal_shift'], k_weight=metadata['k_weight'],
        occupation=torch.tensor(metadata['occupation'], dtype=torch.float64),
        source_eigenvalue_ha=torch.tensor(eigenvalues[source_ik], dtype=torch.float64) * .5,
        overlap=chunks[(1, source_ik, -1)], hamiltonian_ha=chunks[(6, source_ik, -1)] * .5,
        occupied_projection=chunks[(7, source_ik, -1)],
        source=chunks[(2, source_ik, -1)].reshape(occupied, white, n),
        reference_projection=torch.empty(0, dtype=torch.complex128),
    )
    for matrix in (record.overlap, record.hamiltonian_ha, chunks[(4, 0, -1)]):
        data._hermitian(matrix, 'sample matrix')
    data._require(record.overlap.shape == record.hamiltonian_ha.shape == (n, n)
                  and record.occupied_projection.shape == (occupied, n), 'invalid sample dimensions')
    names = ('abacus_commit', 'executable_sha256', 'orbital_sha256', 'pseudopotential_sha256',
             'auxiliary_basis_sha256', 'primitive_blocks_sha256', 'physics_hash')
    # This partial object never leaves this diagnostic or goes to an optimizer.
    sample = data.PeriodicGalerkinDataset(
        **{name: value(name) for name in names}, selected_iq=int(value('selected_iq')),
        q_count=int(value('q_count')), qpoint=data._three(scalar, 'qpoint'), q_weight=float(value('q_weight')),
        primitive_count=n, raw_auxiliary_dimension=raw, whitened_auxiliary_rank=white,
        frequency_ha=torch.tensor([frequencies[i][0] for i in range(nf)], dtype=torch.float64),
        frequency_weights_ha=torch.tensor([frequencies[i][1] for i in range(nf)], dtype=torch.float64),
        coulomb_metric=chunks[(4, 0, -1)], coulomb_whitening=chunks[(5, 0, -1)],
        reference_response=torch.stack([chunks[(8, 0, iw)] for iw in range(nf)]),
        primitive_blocks=blocks, kpoints=(record,),
    )
    # Check parent identity again so a concurrently modified manifest cannot pass.
    _check_hash(directory / 'manifest.dat', item['manifest_sha256'])
    return sample, dict(scope='single_k_equivalence_only_not_physical_energy',
                        parent_freeze_sha256=freeze_sha256, parent=item, source_ik=source_ik,
                        original_k_count=nk, retained_k_count=1, checked_chunks=len(entries),
                        retained_chunks=len(chunks), hashed_chunk_bytes=hashed_bytes,
                        load_and_hash_seconds=time.monotonic()-started)


def _payload_bytes(dataset):
    return sum(getattr(record, name).numel() * getattr(record, name).element_size()
               for record in dataset.kpoints
               for name in ('overlap', 'hamiltonian_ha', 'occupied_projection', 'source'))


def compare_sample(sample, coefficients):
    started = time.monotonic()
    full = prepare_periodic_occupied_reference(sample)
    reduced = reduce_periodic_active_primitives(full, coefficients)
    setup_seconds = time.monotonic()-started
    snapshots = []
    fields = ('loss', 'pi_relative_squared_error', 'trace_log_relative_squared_error',
              'energy_relative_squared_error')
    for dataset in (full, reduced):
        start = time.monotonic()
        coeff = {element: [x.clone().requires_grad_(True) for x in channels]
                 for element, channels in coefficients.items()}
        operators = contract_periodic_candidate_operators(dataset.kpoints[0], dataset.primitive_blocks, coeff)
        response = evaluate_periodic_galerkin_coefficient_response(dataset, coeff, contraction_backend='block')
        objective = periodic_rpa_objective((dataset,), (response.response,))
        objective.loss.backward()
        snapshots.append(dict(
            operators=[getattr(operators, name).detach().clone()
                       for name in ('overlap', 'hamiltonian_ha', 'source', 'occupied_projection')],
            response=response.response.detach().clone(),
            loss=torch.stack([getattr(objective, name).detach() for name in fields]),
            gradients=[x.grad.detach().clone() for channels in coeff.values() for x in channels if x.numel()],
            capture=response.minimum_occupied_capture, seconds=time.monotonic()-start,
        ))
        del operators, response, objective, coeff
    a, b = snapshots
    for x, y in zip(a['operators'], b['operators']):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-13)
    for name in ('response', 'loss'):
        torch.testing.assert_close(a[name], b[name], rtol=1e-12, atol=1e-13)
    for x, y in zip(a['gradients'], b['gradients']):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-12)
    data._require(abs(a['capture']-b['capture']) <= 1e-12, 'occupied capture changed')
    return dict(equivalence_gate='pass', original_primitive_count=full.primitive_count,
                compact_primitive_count=reduced.primitive_count,
                mapping_sha256=reduced.active_primitive_reduction.mapping_sha256,
                original_operator_payload_bytes=_payload_bytes(full),
                compact_operator_payload_bytes=_payload_bytes(reduced),
                normalization_and_reduction_seconds=setup_seconds,
                full_forward_backward_seconds=a['seconds'], compact_forward_backward_seconds=b['seconds'],
                maximum_response_absolute_difference=float((a['response']-b['response']).abs().max()),
                maximum_loss_absolute_difference=float((a['loss']-b['loss']).abs().max()),
                maximum_gradient_absolute_difference=max(float((x-y).abs().max()) for x, y in zip(a['gradients'], b['gradients'])),
                occupied_capture=a['capture'], occupied_capture_absolute_difference=abs(a['capture']-b['capture']),
                scope='single_k_equivalence_only_not_physical_energy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--label', type=int, default=1)
    parser.add_argument('--source-ik', type=int, default=1)
    parser.add_argument('--coefficients', required=True)
    parser.add_argument('--coefficients-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        _check_hash(args.coefficients, args.coefficients_sha256)
        sample, evidence = _load_sample(args.freeze, args.freeze_sha256, args.label, args.source_ik)
        coefficients = read_periodic_optimizer_coefficients(args.coefficients, element='C',
                                                            radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
        result = compare_sample(sample, coefficients)
        result.update(status='success', input=evidence, coefficients_sha256=args.coefficients_sha256,
                      combined_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      formal_optimization_gate='hold', physical_release_gate='hold')
        (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        (output/'STATUS').write_text('success\n')
        print(json.dumps(result, indent=2, allow_nan=False))
    except Exception:
        (output/'STATUS').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
