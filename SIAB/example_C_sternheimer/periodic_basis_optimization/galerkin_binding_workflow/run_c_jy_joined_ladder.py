"""DF cached-matrix angular ladder with original H/S/O and audited finite-q D.

Each independent q first reproduces the accepted spd energy and response.
No ABACUS, reference solve, production SOS, or GW is invoked.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from audit_c_jy_operator_restart import OperatorFiles, sha
from audit_c_jy_reference_reuse import validate_reader_tree, validate_reference_archive
from c_jy_joined_operators import join_operator_record, validate_source_audit
from c_portable_frozen_replay import (FREEZE_SHA, INDEX_SHA, INDICES, MULT,
    hashed, require, within, validate_cache_contract)
from prepare_c_jy_operator_restart import REFERENCE_REUSE_SHA
from run_c_optimizer_comparison import BUNDLE_SHA


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def occupied_embedding_from_record(record, relative_rank_tolerance):
    """Rebuild the occupied rows missing from the frozen cache adapter.

    The finite-q reader remains frozen for cache compatibility.  This helper
    uses that reader's production Lowdin and occupied-frame routines, so the
    added rows use exactly the same retained mother coordinates as its virtual
    embedding.
    """
    import torch
    from periodic_galerkin_sternheimer import _mother_overlap_orthogonalizer
    from response_galerkin import occupied_frames

    with torch.no_grad():
        lowdin, rank, condition = _mother_overlap_orthogonalizer(
            record.overlap, relative_rank_tolerance)
    to_numpy = lambda value: value.detach().cpu().numpy()
    lowdin = to_numpy(lowdin)
    overlap = to_numpy(record.overlap)
    occupied = to_numpy(record.occupied_projection) @ lowdin
    if record.occupied_projection_normalization is not None:
        occupied = to_numpy(record.occupied_projection_normalization) @ occupied
    occupied_frame, _, captures = occupied_frames(occupied)
    embedding = occupied_frame.conj().T @ lowdin.conj().T @ overlap
    require(np.isfinite(embedding).all(), 'nonfinite occupied embedding')
    return embedding, dict(captures, retained_rank=rank,
                           overlap_condition=condition)


def consume_compatible_target(args, consume_current_target, records_by_source_ik,
                              relative_rank_tolerance):
    """Bridge frozen four-argument and current five-argument target callbacks."""
    require(len(args) in (4, 5), 'unsupported response-target callback contract')
    if len(args) == 5:
        covariance, occupied_embedding, embedding, target_pi, metadata = args
        require(np.isfinite(occupied_embedding).all()
                and occupied_embedding.ndim == 2
                and occupied_embedding.shape[0] == metadata['occupied_rank']
                and occupied_embedding.shape[1] == embedding.shape[1],
                'invalid provided occupied embedding')
        metadata.update(occupied_embedding_adapter='provided_by_response_target')
    else:
        covariance, embedding, target_pi, metadata = args
        occupied_embedding, occupied = occupied_embedding_from_record(
            records_by_source_ik[metadata['source_ik']], relative_rank_tolerance)
        require(occupied['retained_rank'] == embedding.shape[1]
                and occupied['occupied_rank'] == metadata['occupied_rank'],
                'occupied/virtual retained-frame mismatch')
        metadata.update(
            occupied_embedding_definition='O_dagger_Lowdin_dagger_S',
            occupied_embedding_adapter='rebuilt_from_frozen_reader_metric',
            occupied_embedding_minimum_capture=occupied['minimum_capture'])
    consume_current_target(covariance, occupied_embedding, embedding,
                           target_pi, metadata)


def run(contract_path, output):
    c = json.loads(contract_path.read_text())
    compressed = (c.get('scope') ==
                  'compressed_shared_radial_full_q_body_RPA')
    require(os.environ.get('C_EXECUTION_HOST') == 'df_iopcas_ghj'
        and os.environ.get('SLURM_JOB_ID') == c['job_id']
        and os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'registered DF job required')
    compressed_contract = compressed and c['lmax_values'] == [3]
    ladder_contract = not compressed and c['lmax_values'] == [2,3,4]
    require(c['relative_rank_tolerance'] == 1e-10
            and (compressed_contract or ladder_contract),
            'joined ladder must use its accepted 1e-10 control')
    repo = Path(__file__).resolve().parents[4]
    manifest = json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    require(manifest['commit'] == c['source_commit'], 'immutable source mismatch')
    for name, digest in manifest['files'].items():
        hashed(within(repo, name), digest)
    for name, digest in c['inputs'].items():
        hashed(Path(name), digest)
    bundle, reader = Path(c['bundle']), Path(c['reader_source'])
    b = json.loads(hashed(bundle/'BUNDLE.json', BUNDLE_SHA))
    for name, digest in b['files'].items():
        hashed(within(bundle,name), digest)
    validate_reader_tree(reader,b['source_files'])
    # Preserve the original cache reader identity, then import the new adapter.
    sys.path.insert(0,str(reader/'SIAB/opt_orb_pytorch_dpsi'))
    import torch
    from periodic_galerkin_basis import (
        prepare_periodic_block_contraction_record,
        read_periodic_optimizer_coefficients,
    )
    from periodic_galerkin_dataset_cache import read_periodic_galerkin_dataset_cache
    from periodic_galerkin_data import _read_primitive_blocks
    from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
    from run_c_jy_angular_ladder import candidate_evaluation_view, energy_summary
    from periodic_available_mother import available_mother_response
    compressed_spec = importlib.util.spec_from_file_location(
        '_c_jy_compressed_evaluation_contract',
        repo/'SIAB/opt_orb_pytorch_dpsi/c_jy_compressed_evaluation.py')
    compressed_module = importlib.util.module_from_spec(compressed_spec)
    compressed_spec.loader.exec_module(compressed_module)
    evaluate_compressed_profiles = compressed_module.evaluate_compressed_profiles
    validate_profile_input_hashes = compressed_module.validate_profile_input_hashes
    validate_profile_specs = compressed_module.validate_profile_specs
    if compressed:
        validate_profile_specs(c.get('candidate_profiles'))
        validate_profile_input_hashes(c['candidate_profiles'], c.get('inputs'))
    spec = importlib.util.spec_from_file_location(
        '_c_jy_response_targets_contract', repo/'SIAB/opt_orb_pytorch_dpsi/c_jy_response_targets.py')
    target_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target_module)
    validate_target_manifest = target_module.validate_target_manifest
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    freeze = json.loads(hashed(bundle/'backoff_INPUT_FREEZE.json',FREEZE_SHA))
    index = json.loads(hashed(bundle/'backoff_ACTIVE_DATA_CACHE.json',INDEX_SHA))
    records = validate_cache_contract(freeze,index)
    slot = c['q_slot']
    require(1 <= slot < 8, 'Gamma is already computed and must be reused')
    item, record = freeze['datasets'][slot], records[slot]
    original = read_periodic_optimizer_coefficients(bundle/'backoff_ORIGINAL_COEFFICIENTS.txt',
        element='C', radial_rows=31, expected_nu=(3,3,2,0,0))
    output.mkdir()
    start = time.perf_counter()
    try:
        old = read_periodic_galerkin_dataset_cache(within(bundle,b['cache_paths'][str(item['label'])]),
            cache_sha256=record['complete_sha256'],active_coefficients=original,**record['binding'])
        require(old.selected_iq == INDICES[slot] and old.q_weight == MULT[slot]/64
            and len(old.kpoints) == 64 and old.primitive_count == 558
            and old.frequency_ha.numel() == 12, 'canonical old q data required')
        audit_path = Path(c['audit_result'])
        audit = json.loads(audit_path.read_text())
        validate_source_audit(audit,old.selected_iq)
        require(audit['cache_sha256'] == record['complete_sha256'], 'different audited cache')
        maps_path = audit_path.parent/'GAUGE_MAPS.npz'
        hashed(maps_path,audit['gauge_maps_sha256'])
        reuse = json.loads(hashed(Path(c['reuse_result']),REFERENCE_REUSE_SHA))
        require(reuse['original_gamma_operator_reuse_gate'] == 'pass'
            and reuse['failure_reasons'] == [], 'original operator reuse not accepted')
        gamma = OperatorFiles(Path(reuse['reference']),False)
        validate_reference_archive(gamma.directory,
            reuse['reference_hashes']['manifest.dat'],
            '489afa553f53c236ed5fbcbd89b8732e6c2274593d6d076f1c207ffdd8876807')
        require(audit['original_operator_directory'] == str(gamma.directory)
            and audit['original_operator_hashes'] == reuse['reference_hashes'],
            'mixed original operators')
        source_run = Path(audit['run'])
        sc = json.loads(hashed(source_run/'CONTRACT.json',audit['contract_sha256']))
        source = OperatorFiles(source_run/('OUT.'+sc['suffix'])/'STERNHEIMER_BASIS_OPERATORS_V1',True)
        require(source.hashes['manifest.dat'] == audit['native_hashes']['manifest.dat'],
            'audited source manifest changed')
        blocks = _read_primitive_blocks(str(gamma.directory),old.primitive_blocks_sha256,1550)
        indices = tuple(old.active_primitive_reduction.source_indices)
        require(indices == tuple(range(279))+tuple(range(775,1054)), 'unexpected spd embedding')
        joined, checks = [], []
        with np.load(maps_path,allow_pickle=False) as maps:
            for k in old.kpoints:
                values, check = join_operator_record(dict(source_ik=k.source_ik,target_ik=k.target_ik,
                    **{n:getattr(k,n).numpy() for n in
                       ('overlap','hamiltonian_ha','occupied_projection','source')}),
                    gamma.array,source.array,maps,indices,1550)
                joined.append(replace(k,**{n:torch.from_numpy(v) for n,v in values.items()},
                    reference_projection=torch.empty(0,dtype=torch.complex128),block_contraction_cache=None))
                checks.append(dict(source_ik=k.source_ik,target_ik=k.target_ik,**check))
                print(json.dumps(dict(stage='join',k=k.source_ik,seconds=time.perf_counter()-start)),flush=True)
        require(all(reuse['reference_hashes'].get(n) == d for n,d in gamma.hashes.items()),
            'original operator payload differs from reuse audit')
        require(all(audit['native_hashes'].get(n) == d for n,d in source.hashes.items()),
            'source payload differs from audit')
        full = replace(old,primitive_count=1550,primitive_blocks=blocks,
            active_primitive_reduction=None,kpoints=tuple(joined))
        del joined, old
        write(output/'JOIN.json',dict(status='success',per_k=checks,
            original_hashes=gamma.hashes,source_hashes=source.hashes,
            gauge_maps_sha256=sha(maps_path),regenerated_hamiltonian_read=False))
        baseline = None
        if not compressed:
            baseline = json.loads(Path(c['baseline_result']).read_text())
            require(baseline['selected_iq'] == full.selected_iq
                and baseline['relative_rank_tolerance'] == 1e-10
                and baseline['cache_complete_sha256'] == record['complete_sha256'],
                'baseline mismatch')
        rows = []
        for lmax in c['lmax_values']:
            view = candidate_evaluation_view(full, lmax,
                                             compressed=compressed)
            stage = output/('lmax%d'%lmax)
            stage.mkdir()
            if compressed:
                def checkpoint_profile(profile):
                    write(stage/('profile-%03d.json' % profile['ao_per_C']), dict(
                        status='success', q_slot=slot,
                        selected_iq=full.selected_iq, q_weight=full.q_weight,
                        lmax=lmax, source_commit=c['source_commit'],
                        result=profile))

                profiles = evaluate_compressed_profiles(
                    view, c['candidate_profiles'],
                    read_coefficients=read_periodic_optimizer_coefficients,
                    evaluate_response=evaluate_periodic_galerkin_coefficient_response,
                    summarize_energy=energy_summary,
                    relative_rank_tolerance=c['relative_rank_tolerance'],
                    occupied_capture_floor=c['occupied_capture_floor'],
                    prepare_block_cache=prepare_periodic_block_contraction_record,
                    profile_callback=checkpoint_profile)
                row = dict(status='success', lmax=lmax,
                    selected_iq=full.selected_iq, label=item['label'],
                    multiplicity=MULT[slot], q_weight=full.q_weight,
                    frequencies=full.frequency_ha.tolist(),
                    weights=full.frequency_weights_ha.tolist(),
                    relative_rank_tolerance=c['relative_rank_tolerance'],
                    primitive_count=view.primitive_count, k_record_count=64,
                    source_commit=c['source_commit'], per_profile=profiles,
                    full_q_admitted=False, physical_release_gate='hold')
                write(stage/'RESULT.json', row)
                rows.append(row)
                del view, profiles
                continue
            with (stage/'K_PROGRESS.jsonl').open('x') as stream:
                def log(row):
                    row = dict(lmax=lmax,seconds=time.perf_counter()-start,**row)
                    stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
                    print(json.dumps(row,allow_nan=False),flush=True)
                target_dir = None
                if c.get('target_lmax') == lmax:
                    from response_target import (
                        build_available_response_targets as build_legacy_response_targets)
                    target_dir = Path(c['target_dir'])
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_rows = []
                    records_by_source_ik = {record.source_ik: record
                                            for record in view.kpoints}

                    def consume_current_target(covariance, occupied_embedding, embedding,
                                               target_pi, metadata):
                        values = np.linalg.eigvalsh(covariance)
                        require(np.isfinite(embedding).all() and values[-1] > 0
                            and values[0] >= -1e-10*values[-1],
                            'invalid jY response target spectrum')
                        array_file = target_dir/('k%04d.npz' % metadata['source_ik'])
                        with array_file.open('xb') as stream:
                            np.savez(stream, covariance=covariance, embedding=embedding,
                                     occupied_embedding=occupied_embedding)
                        target_rows.append(dict(metadata,
                            q_slot=slot, target_kind='response_covariance_embedding',
                            assembled_pi=False, lmax=lmax,
                            selected_iq=full.selected_iq,
                            frequency_count=int(full.frequency_ha.numel()),
                            primitive_count=view.primitive_count,
                            covariance_dimension=int(covariance.shape[0]),
                            embedding_rows=int(embedding.shape[0]),
                            embedding_columns=int(embedding.shape[1]),
                            occupied_embedding_rows=int(occupied_embedding.shape[0]),
                            target_norm2=float(np.trace(covariance).real),
                            covariance_minimum=float(values[0]),
                            covariance_maximum=float(values[-1]),
                            file=array_file.name, file_sha256=sha(array_file)))

                    def consume_legacy_target(*args):
                        consume_compatible_target(args, consume_current_target,
                                                  records_by_source_ik, 1e-10)

                    pi, detail = build_legacy_response_targets(
                        view, consume_legacy_target,
                        relative_rank_tolerance=1e-10, progress=log)
                    require(len(target_rows) == 64, 'complete jY target sector set required')
                    target_manifest = dict(status='success',
                        target_kind='response_covariance_embedding', assembled_pi=False,
                        lmax=lmax, q_slot=slot, selected_iq=full.selected_iq,
                        q_weight=full.q_weight, frequency_count=int(full.frequency_ha.numel()),
                        frequency_ha=full.frequency_ha.tolist(),
                        frequency_weights_ha=full.frequency_weights_ha.tolist(),
                        k_record_count=len(target_rows), primitive_count=view.primitive_count,
                        sectors=target_rows, physical_release_gate='hold')
                    validate_target_manifest(target_manifest, lmax=lmax,
                        frequency_count=int(full.frequency_ha.numel()),
                        k_record_count=64, primitive_count=view.primitive_count,
                        q_slots=(slot,))
                    write(target_dir/'TARGETS.json', target_manifest)
                    np.save(target_dir/'PI_DIAGNOSTIC.npy', pi, allow_pickle=False)
                else:
                    pi, detail = available_mother_response(
                        view, relative_rank_tolerance=1e-10, progress=log)
            energy = energy_summary(view,pi)
            if lmax == 2:
                previous = np.load(Path(c['baseline_result']).parent/'PI.npy',allow_pickle=False)
                delta = abs(energy['candidate_energy_ha']-baseline['candidate_energy_ha'])
                pi_delta = float(np.linalg.norm(pi-previous)/np.linalg.norm(previous))
                require(delta <= 1e-8 and pi_delta <= 1e-6
                    and abs(energy['reference_energy_ha']-baseline['reference_energy_ha']) <= 1e-12,
                    'joined spd energy/response reproduction failed: '+str((delta,pi_delta)))
                energy.update(spd_energy_difference_ha=delta,spd_pi_relative_difference=pi_delta)
            row = dict(energy,status='success',lmax=lmax,selected_iq=full.selected_iq,
                label=item['label'],multiplicity=MULT[slot],frequencies=full.frequency_ha.tolist(),
                weights=full.frequency_weights_ha.tolist(),relative_rank_tolerance=1e-10,
                primitive_count=view.primitive_count,k_record_count=64,source_commit=c['source_commit'])
            np.save(stage/'PI.npy',pi,allow_pickle=False)
            write(stage/'RESULT.json',dict(row,numerical_diagnostics=detail))
            rows.append(row)
            del view, pi, detail
        if compressed:
            final = dict(rows[0], q_slot=slot,
                scope='compressed_shared_radial_full_q_body_RPA',
                elapsed_seconds=time.perf_counter()-start,
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        else:
            final = dict(status='success',per_lmax=rows,selected_iq=full.selected_iq,
                q_slot=slot,elapsed_seconds=time.perf_counter()-start,
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                full_q_admitted=False,physical_release_gate='hold')
        write(output/'RESULT.json', final)
        write(output/'PROVENANCE.json',dict(status='success',job_id=c['job_id'],
            source_commit=c['source_commit'],contract_sha256=sha(contract_path),
            files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
            new_reference_solve=False,regenerated_hamiltonian_read=False))
        (output/'STATUS').write_text('success\n')
    except Exception as e:
        write(output/'FAILURE.json',dict(status='failed',error=str(e),physical_release_gate='hold'))
        (output/'STATUS').write_text('failed\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.contract,args.output)
