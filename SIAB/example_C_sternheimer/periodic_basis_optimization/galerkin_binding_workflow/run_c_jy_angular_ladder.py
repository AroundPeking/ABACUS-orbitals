"""Frozen Gamma jY angular ladder; cached matrix algebra, never full-q admission."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/'SIAB/opt_orb_pytorch_dpsi'))
from periodic_available_mother import available_mother_response
from c_jy_compressed_evaluation import (
    evaluate_compressed_profiles, validate_profile_input_hashes,
    validate_profile_specs)
from periodic_galerkin_basis import (
    prepare_periodic_block_contraction_record,
    read_periodic_optimizer_coefficients,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset
from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
from periodic_galerkin_reduction import reduce_periodic_active_primitives
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference
from periodic_galerkin_rpa import periodic_rpa_objective

_target_spec = importlib.util.spec_from_file_location(
    '_c_jy_response_targets_contract', REPO/'SIAB/opt_orb_pytorch_dpsi/c_jy_response_targets.py')
_target_module = importlib.util.module_from_spec(_target_spec)
_target_spec.loader.exec_module(_target_module)
validate_target_manifest = _target_module.validate_target_manifest


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(2**20), b''):
            digest.update(data)
    return digest.hexdigest()


def write(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def angular_view(dataset, lmax):
    """Retain complete radial/magnetic blocks, including all cross couplings."""
    if dataset.active_primitive_reduction is not None:
        raise ValueError('unreduced jY operators required')
    if not isinstance(lmax, int) or lmax < 0:
        raise ValueError('invalid angular cutoff')
    support = {(b.element, b.l) for b in dataset.primitive_blocks}
    elements = {b.element for b in dataset.primitive_blocks}
    if any((e, l) not in support for e in elements for l in range(lmax+1)):
        raise ValueError('missing angular operator blocks')
    coefficients = {e: [torch.empty((0, 0), dtype=torch.float64)
        for _ in range(1+max(l for element, l in support if element == e))]
        for e in elements}
    for block in dataset.primitive_blocks:
        coefficients[block.element][block.l] = (
            torch.eye(block.n_primitive, dtype=torch.float64) if block.l <= lmax
            else torch.empty((block.n_primitive, 0), dtype=torch.float64))
    return reduce_periodic_active_primitives(dataset, coefficients)


def candidate_evaluation_view(dataset, lmax, *, compressed):
    """Keep compact coefficients on the unreduced mother operator.

    A reduced dataset caches the coefficient profile used for that reduction.
    Compact candidates deliberately change that profile, so their zero-column
    angular exclusions must be applied directly to the unreduced operator.
    """
    return dataset if compressed else angular_view(dataset, lmax)


def energy_summary(dataset, response):
    result = periodic_rpa_objective((dataset,), (torch.from_numpy(response),))
    row = {key: value.tolist() if isinstance(value, torch.Tensor) else value
           for key, value in asdict(result.q_records[0]).items()}
    return dict(candidate_energy_ha=float(result.candidate_energy_ha),
        reference_energy_ha=float(result.reference_energy_ha),
        pi_relative_error=float(result.pi_relative_squared_error)**.5,
        tracelog_relative_error=float(result.trace_log_relative_squared_error)**.5,
        q_weight=dataset.q_weight, per_frequency=row,
        full_q_admitted=False, physical_release_gate='hold')


def run(contract_path, output):
    contract = json.loads(contract_path.read_text())
    compressed = (contract.get('scope') ==
                  'compressed_shared_radial_full_q_body_RPA')
    if (os.environ.get('C_EXECUTION_HOST') != 'df_iopcas_ghj'
            or os.environ.get('SLURM_JOB_NUM_NODES') != '1'
            or os.environ.get('SLURM_JOB_ID') != contract['job_id']):
        raise ValueError('registered single-node DF job required')
    compressed_contract = compressed and contract['lmax_values'] == [3]
    ladder_contract = not compressed and contract['lmax_values'] == [2, 3, 4]
    if contract['relative_rank_tolerance'] != 1e-10 or not (
            compressed_contract or ladder_contract):
        raise ValueError('frozen cutoff and nested angular ladder required')
    if compressed:
        validate_profile_specs(contract.get('candidate_profiles'))
        validate_profile_input_hashes(contract['candidate_profiles'],
                                      contract.get('inputs'))
    for name, expected in contract['inputs'].items():
        if sha(name) != expected:
            raise ValueError('input hash mismatch: '+name)
    manifest = json.loads((REPO/'SOURCE_MANIFEST.json').read_text())
    if manifest['commit'] != contract['source_commit']:
        raise ValueError('immutable source mismatch')
    for name, expected in manifest['files'].items():
        if sha(REPO/name) != expected:
            raise ValueError('source hash mismatch: '+name)
    output.mkdir()
    start = time.perf_counter()
    rows = []
    try:
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        # Qref is not needed for this Galerkin solve. Its headers remain checked;
        # the independent previous audit, bound by contract, hashed all Qref.
        dataset = read_periodic_galerkin_dataset(contract['raw_directory'],
            include_reference_projection=False, verify_omitted_chunks=False)
        if (dataset.selected_iq != 1 or dataset.qpoint != (0., 0., 0.)
                or dataset.q_count != 64 or dataset.q_weight != 1/64
                or dataset.primitive_count != 1550 or len(dataset.kpoints) != 64
                or dataset.frequency_ha.numel() != 12):
            raise ValueError('complete frozen Gamma contract mismatch')
        dataset = prepare_periodic_occupied_reference(dataset)
        load_seconds = time.perf_counter()-start
        print(json.dumps(dict(stage='loaded', seconds=load_seconds)), flush=True)
        baseline = (None if compressed else
                    json.loads(Path(contract['accepted_spd_q_result']).read_text()))
        for lmax in contract['lmax_values']:
            view = candidate_evaluation_view(dataset, lmax,
                                             compressed=compressed)
            stage = output/('lmax%d' % lmax)
            stage.mkdir()
            if compressed:
                profiles = evaluate_compressed_profiles(
                    view, contract['candidate_profiles'],
                    read_coefficients=read_periodic_optimizer_coefficients,
                    evaluate_response=evaluate_periodic_galerkin_coefficient_response,
                    summarize_energy=energy_summary,
                    relative_rank_tolerance=contract['relative_rank_tolerance'],
                    occupied_capture_floor=contract['occupied_capture_floor'],
                    prepare_block_cache=prepare_periodic_block_contraction_record)
                row = dict(status='success', q_slot=0, label=1, selected_iq=1,
                    multiplicity=1, q_weight=dataset.q_weight, lmax=lmax,
                    frequencies=dataset.frequency_ha.tolist(),
                    weights=dataset.frequency_weights_ha.tolist(),
                    relative_rank_tolerance=contract['relative_rank_tolerance'],
                    primitive_count=view.primitive_count, k_record_count=64,
                    source_commit=contract['source_commit'], per_profile=profiles,
                    full_q_admitted=False, physical_release_gate='hold')
                write(stage/'RESULT.json', row)
                rows.append(row)
                del view, profiles
                continue
            with (stage/'K_PROGRESS.jsonl').open('x') as progress:
                def log(row):
                    record = dict(lmax=lmax, elapsed_seconds=time.perf_counter()-start, **row)
                    progress.write(json.dumps(record, allow_nan=False)+'\n')
                    progress.flush()
                    print(json.dumps(record, allow_nan=False), flush=True)
                if contract.get('target_lmax') == lmax:
                    from response_target import build_available_response_targets
                    target_dir = Path(contract['target_dir'])
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_rows = []

                    def consume(covariance, occupied_embedding, embedding, target_pi, metadata):
                        values = np.linalg.eigvalsh(covariance)
                        if (not np.isfinite(embedding).all() or values[-1] <= 0
                                or values[0] < -1e-10*values[-1]):
                            raise ValueError('invalid Gamma jY response target spectrum')
                        array_file = target_dir/('k%04d.npz' % metadata['source_ik'])
                        with array_file.open('xb') as stream:
                            np.savez(stream, covariance=covariance, embedding=embedding,
                                     occupied_embedding=occupied_embedding)
                        target_rows.append(dict(metadata,
                            q_slot=0, target_kind='response_covariance_embedding',
                            assembled_pi=False, lmax=lmax,
                            selected_iq=dataset.selected_iq,
                            frequency_count=int(dataset.frequency_ha.numel()),
                            primitive_count=view.primitive_count,
                            covariance_dimension=int(covariance.shape[0]),
                            embedding_rows=int(embedding.shape[0]),
                            embedding_columns=int(embedding.shape[1]),
                            occupied_embedding_rows=int(occupied_embedding.shape[0]),
                            target_norm2=float(np.trace(covariance).real),
                            covariance_minimum=float(values[0]),
                            covariance_maximum=float(values[-1]),
                            file=array_file.name, file_sha256=sha(array_file)))

                    response, detail = build_available_response_targets(
                        view, consume,
                        relative_rank_tolerance=contract['relative_rank_tolerance'],
                        progress=log)
                    if len(target_rows) != 64:
                        raise ValueError('complete Gamma target sector set required')
                    target_manifest = dict(status='success',
                        target_kind='response_covariance_embedding', assembled_pi=False,
                        lmax=lmax, q_slot=0, selected_iq=dataset.selected_iq,
                        q_weight=dataset.q_weight,
                        frequency_count=int(dataset.frequency_ha.numel()),
                        frequency_ha=dataset.frequency_ha.tolist(),
                        frequency_weights_ha=dataset.frequency_weights_ha.tolist(),
                        k_record_count=len(target_rows), primitive_count=view.primitive_count,
                        sectors=target_rows, physical_release_gate='hold')
                    validate_target_manifest(target_manifest, lmax=lmax,
                        frequency_count=int(dataset.frequency_ha.numel()),
                        k_record_count=64, primitive_count=view.primitive_count,
                        q_slots=(0,))
                    write(target_dir/'TARGETS.json', target_manifest)
                    np.save(target_dir/'PI_DIAGNOSTIC.npy', response, allow_pickle=False)
                else:
                    response, detail = available_mother_response(view,
                        relative_rank_tolerance=contract['relative_rank_tolerance'], progress=log)
            summary = energy_summary(view, response)
            if lmax == 2:
                delta = abs(summary['candidate_energy_ha']-baseline['candidate_energy_ha'])
                ref_delta = abs(summary['reference_energy_ha']-baseline['reference_energy_ha'])
                if delta > 1e-8 or ref_delta > 1e-9:
                    raise ValueError('raw spd failed accepted Gamma energy reproduction')
                summary.update(accepted_spd_energy_difference_ha=delta,
                               accepted_reference_energy_difference_ha=ref_delta)
            summary.update(lmax=lmax, status='success', primitive_count=view.primitive_count,
                           primitive_count_per_C=view.primitive_count//2,
                           relative_rank_tolerance=contract['relative_rank_tolerance'])
            np.save(stage/'PI.npy', response, allow_pickle=False)
            write(stage/'RESULT.json', dict(**summary, numerical_diagnostics=detail))
            rows.append(summary)
            del view, response, detail
        if compressed:
            final = dict(rows[0], scope='compressed_shared_radial_full_q_body_RPA',
                elapsed_seconds=time.perf_counter()-start, load_seconds=load_seconds,
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        else:
            final = dict(status='success', scope='Gamma_only_angular_diagnostic',
                full_q_admitted=False, physical_release_gate='hold', per_lmax=rows,
                elapsed_seconds=time.perf_counter()-start, load_seconds=load_seconds,
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(output/'RESULT.json', final)
        write(output/'PROVENANCE.json', dict(status='success', source_commit=manifest['commit'],
            job_id=contract['job_id'], contract_sha256=sha(contract_path),
            files={str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()},
            reference_generation=False, new_physics_production=False))
        with (output/'STATUS').open('x') as stream:
            stream.write('success\n')
    except Exception as error:
        write(output/'FAILURE.json', dict(status='failed', message=str(error),
            elapsed_seconds=time.perf_counter()-start, physical_release_gate='hold'))
        with (output/'STATUS').open('x') as stream:
            stream.write('failed\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.contract, args.output)
