"""One accepted-Ec-gradient step, one actual PBE, then complete Galerkin forwards."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

import check_c_optimized_pbe as endpoint

ROOT = Path('/work1/ghj/c-solid-fd8-q13-standard-20260903')
GRADIENT_STAGE = ROOT/'stage-c-ec-gradient-refresh-9ee3747e'
GRADIENT_RESULT = 'b3e46a26e07a76a0931e8418c185e613bdef346fa31e52d6fb33ed8ba2bd02b5'
GRADIENT_ACCEPTANCE = '3991384900936811cfb9f5708446e2c344698bb5e455078c7a792d31e40f4c3e'
SCOPE = 'one_accepted_ec_gradient_step'


def measure_candidate(pbe_slot, prepare, collect, launcher, forward):
    preparation_sha = prepare()
    started = time.perf_counter()
    with (Path(pbe_slot)/'abacus.out').open('xb') as stream:
        subprocess.run(launcher, cwd=pbe_slot, stdout=stream, stderr=subprocess.STDOUT, check=True)
    pbe = collect(preparation_sha)
    result = dict(status='success', scope=SCOPE, actual_pbe=pbe,
        scf_seconds=time.perf_counter()-started, rpa_evaluations=0, actual_scf_count=1,
        backward_passes=0, optimizer_steps=0, physical_release_gate='hold',
        ordinary_sos_qavg='pending', gw='pending')
    if pbe.get('pbe_gate') != 'pass':
        result['candidate_gate'] = 'rejected_actual_pbe'
        return result
    started = time.perf_counter()
    center, candidate = forward()
    result.update(center=center, candidate=candidate, rpa_evaluations=2,
                  response_seconds=time.perf_counter()-started)
    for r in (center, candidate):
        for value in (r['loss'], r['rpa']['candidate_energy_ha'], r['rpa']['reference_energy_ha']):
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError('nonfinite full-q result')
        if r['loss'] < 0:
            raise ValueError('nonnegative full objective required')
    reference = center['rpa']['reference_energy_ha']
    if not math.isclose(candidate['rpa']['reference_energy_ha'], reference, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('unchanged reference required')
    old_error = abs(center['rpa']['candidate_energy_ha']-reference)
    new_error = abs(candidate['rpa']['candidate_energy_ha']-reference)
    accepted = new_error < old_error and candidate['loss'] <= center['loss']
    result['candidate_gate'] = 'improved_frozen_body' if accepted else 'rejected_no_improvement'
    result['measured_ec_delta_ev_per_c'] = (
        candidate['rpa']['candidate_energy_ha']-center['rpa']['candidate_energy_ha'])*endpoint.HARTREE_TO_EV/2
    result['body_error_ev_per_c'] = new_error*endpoint.HARTREE_TO_EV/2
    return result


def run_step(stage, source):
    import refresh_c_combined_step_gradient as refresh
    import check_c_accepted_combined_step as accepted
    from check_c_ec_gradient_step import load_accepted_ec_gradient, validate_gradient_step_candidate
    from check_c_ec_gradient_step_pbe import prepare_ec_gradient_step_pbe, collect_ec_gradient_step_pbe, PREPARATION
    from run_c_combined_step import screen_candidate
    from run_c_pbe_direction_calibration import ABACUS, ABACUS_SHA, PMI, PMI_SHA
    from backoff_c_optimized_direction import evaluate_pair
    from periodic_galerkin_ec_gradient_step import propose_ec_gradient_step
    from periodic_galerkin_basis import write_periodic_optimizer_coefficients
    from periodic_galerkin_fit import CandidateGuardError
    from export_periodic_orbitals import write_abacus_orbital

    started = time.perf_counter()
    evidence = load_accepted_ec_gradient(GRADIENT_STAGE, GRADIENT_RESULT, GRADIENT_ACCEPTANCE)
    gradient, loaded = evidence['gradient'], evidence['center']
    rt = refresh._runtime()
    rt.torch.set_num_threads(28)
    coefficients = refresh._read_coefficients(rt, loaded['coefficient_path'])
    original = refresh._read_coefficients(rt, loaded['original_coefficient_path'])
    with rt.torch.no_grad():
        trial = propose_ec_gradient_step(coefficients, gradient['energy_gradient'])
    output = stage/'result'
    output.mkdir()
    load_started = time.perf_counter()
    datasets, records, guard = rt.load_frozen_c(
        loaded['freeze_path'], loaded['result']['freeze_sha256'], original, output,
        active_cache_index=loaded['active_cache_index_path'],
        active_cache_index_sha256=loaded['result']['active_cache_index_sha256'])
    cache_seconds = time.perf_counter()-load_started
    refresh.validate_dataset_extent(datasets, loaded)
    floor = loaded['occupied_capture_floor']
    common = dict(status='success', scope=SCOPE, source_commit=source['source_commit'],
        deployment_sha256=os.environ['C_EC_STEP_DEPLOYMENT_SHA256'],
        gradient_stage=str(GRADIENT_STAGE), gradient_result_sha256=GRADIENT_RESULT,
        gradient_acceptance_sha256=GRADIENT_ACCEPTANCE,
        center_stage=gradient['accepted_stage'], center_result_sha256=gradient['accepted_result_sha256'],
        center_acceptance_sha256=gradient['accepted_acceptance_sha256'],
        active_cache_index_sha256=gradient['active_cache_index_sha256'],
        freeze_sha256=gradient['freeze_sha256'], occupied_capture_floor=floor,
        cache_loads=1, cache_load_seconds=cache_seconds, load_records=records,
        radius=trial['radius'], actual_pbe_direction_derivative='unmeasured',
        predicted_ec_delta_ev_per_c=trial['predicted_ec_delta_ha_per_cell']*endpoint.HARTREE_TO_EV/2,
        nu=[3,3,2,0,0], fixed_nu=[0]*5, ao_per_C=22, physical_release_gate='hold',
        ordinary_sos_qavg='pending', gw='pending', backward_passes=0, optimizer_steps=0)
    with rt.torch.no_grad():
        try:
            bands, capture = screen_candidate(datasets, trial['coefficients'], guard, floor)
        except CandidateGuardError as error:
            return dict(common, candidate_gate='rejected_cheap_guard', rejection_reason=str(error),
                actual_scf_count=0, rpa_evaluations=0, total_seconds=time.perf_counter()-started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    candidate_root = output/'candidate'
    candidate_root.mkdir()
    step = {k:v for k,v in trial.items() if k not in ('coefficients','direction')}
    step['direction'] = {e:[d.tolist() for d in channels] for e,channels in trial['direction'].items()}
    endpoint._write_json(candidate_root/'GRADIENT_STEP.json', step)
    write_periodic_optimizer_coefficients(candidate_root/'COEFFICIENTS.txt', trial['coefficients'])
    write_abacus_orbital(candidate_root/'C_3s3p2d_ec_step.orb', trial['coefficients'], element='C',
        ecut_ry=100., rcut_bohr=10., dr_bohr=.01, smoothing_sigma_bohr=.1)
    proposal = {k:v for k,v in step.items() if k != 'direction'}
    for k in ('center_stage','center_result_sha256','center_acceptance_sha256',
              'gradient_stage','gradient_result_sha256','gradient_acceptance_sha256',
              'nu','fixed_nu','ao_per_C','source_commit'):
        proposal[k] = common[k]
    proposal.update(status='prepared', center_coefficient_sha256=gradient['coefficient_sha256'],
        center_orbital_sha256=gradient['orbital_sha256'],
        coefficient_filename='COEFFICIENTS.txt', orbital_filename='C_3s3p2d_ec_step.orb',
        coefficient_sha256=endpoint._sha256((candidate_root/'COEFFICIENTS.txt').read_bytes()),
        orbital_sha256=endpoint._sha256((candidate_root/'C_3s3p2d_ec_step.orb').read_bytes()),
        gradient_step_sha256=endpoint._sha256((candidate_root/'GRADIENT_STEP.json').read_bytes()),
        cheap_gate=dict(gate=True, band_screen=bands, minimum_occupied_capture=capture,
            occupied_capture_floor=floor, overlap_relative_rank_tolerance=1e-12, overlap_condition_limit=1e12))
    endpoint._write_json(candidate_root/'CANDIDATE.json', proposal)
    candidate_sha = endpoint._sha256((candidate_root/'CANDIDATE.json').read_bytes())
    validate_gradient_step_candidate(candidate_root)
    slot = output/'pbe'
    def prepare():
        prepare_ec_gradient_step_pbe(common['center_stage'], common['center_result_sha256'],
            common['center_acceptance_sha256'], candidate_root, slot, abacus_binary=ABACUS,
            expected_abacus_sha256=ABACUS_SHA, mpi_library=PMI, expected_mpi_sha256=PMI_SHA)
        return endpoint._sha256((slot/PREPARATION).read_bytes())
    def collect(digest):
        return collect_ec_gradient_step_pbe(slot, digest)
    def validate_center(value):
        accepted._record(value, floor, loaded['quarter']['training_weights'])
        endpoint._match_parent_initial(value, loaded['result']['candidate'])
        accepted._same_grid_reference(value, loaded['result']['candidate'], reproduce=True)
        for name in ('pi','trace_log','energy'):
            key = name+'_relative_squared_error'
            accepted._near(value['rpa'][key], loaded['result']['candidate']['rpa'][key], key)
    def forward():
        accepted._small(candidate_root, 'CANDIDATE.json', candidate_sha)
        validate_gradient_step_candidate(candidate_root)
        restored = refresh._read_coefficients(rt, candidate_root/'COEFFICIENTS.txt')
        first, value = evaluate_pair(datasets, coefficients, restored, guard=guard,
            frequency_batch_size=loaded['quarter']['configuration']['frequency_batch_size'],
            weights=loaded['quarter']['training_weights'], occupied_capture_floor=floor,
            initial_validator=validate_center)
        accepted._record(value, floor, loaded['quarter']['training_weights'])
        accepted._same_grid_reference(value, first)
        return first, value
    measured = measure_candidate(slot, prepare, collect,
        ['srun','--mpi=pmi2','--cpu-bind=none','-n','4',str(ABACUS)], forward)
    result = dict(common, **{k:v for k,v in measured.items() if k not in common})
    result.update(candidate_sha256=candidate_sha, coefficient_sha256=proposal['coefficient_sha256'],
        orbital_sha256=proposal['orbital_sha256'], gradient_step_sha256=proposal['gradient_step_sha256'],
        total_seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['rpa_evaluations']:
        result['measured_to_predicted_ec_gain'] = result['measured_ec_delta_ev_per_c']/common['predicted_ec_delta_ev_per_c']
    load_accepted_ec_gradient(GRADIENT_STAGE, GRADIENT_RESULT, GRADIENT_ACCEPTANCE)
    validate_gradient_step_candidate(candidate_root)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    stage = Path(parser.parse_args(argv).stage).resolve()
    import refresh_c_combined_step_gradient as refresh
    from run_c_pbe_direction_calibration import ABACUS, ABACUS_SHA, PMI, PMI_SHA
    manifest = refresh.verify_source(stage, os.environ['C_EC_STEP_DEPLOYMENT_SHA256'],
                                     os.environ['C_EC_STEP_SOURCE_COMMIT'])
    wf = 'SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/'
    if wf+'run_c_ec_gradient_step.slurm' not in manifest['files']:
        raise ValueError('new batch script hash required')
    refresh._check_file(ABACUS, ABACUS_SHA)
    refresh._check_file(PMI, PMI_SHA)
    if (os.environ.get('SLURM_JOB_NUM_NODES') != '1' or os.environ.get('OMP_NUM_THREADS') != '7'
            or os.environ.get('I_MPI_PMI_LIBRARY') != str(PMI)):
        raise ValueError('frozen single-node 4x7 PMI2 layout required')
    endpoint._write_json(stage/'EXECUTION_PROVENANCE.json', dict(status='launching', server='df_dcu',
        scheduler_job=os.environ['SLURM_JOB_ID'], source_commit=manifest['source_commit'],
        abacus_sha256=ABACUS_SHA, pmi_sha256=PMI_SHA, actual_scf_budget=1, backward_budget=0,
        delta_st='not_run', librpa='not_run', ordinary_sos_qavg='pending', physical_release_gate='hold'))
    result = run_step(stage, manifest)
    refresh.verify_source(stage, os.environ['C_EC_STEP_DEPLOYMENT_SHA256'], os.environ['C_EC_STEP_SOURCE_COMMIT'])
    endpoint._write_json(stage/'result/RESULT.json', result)
    endpoint._write_json(stage/'PROVENANCE.json', dict(status='success', source_commit=manifest['source_commit'],
        result_sha256=endpoint._sha256((stage/'result/RESULT.json').read_bytes()),
        candidate_gate=result['candidate_gate'], actual_scf_count=result['actual_scf_count'],
        rpa_evaluations=result['rpa_evaluations'], backward_passes=0, physical_release_gate='hold'))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
