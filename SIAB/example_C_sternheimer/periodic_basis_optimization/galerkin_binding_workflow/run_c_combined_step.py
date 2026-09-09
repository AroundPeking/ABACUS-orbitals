"""One PBE-corrected C short step: actual SCF before full Galerkin forwards."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

from prepare_c_pbe_direction_calibration import load_frozen_c
import torch
from run_c_pbe_direction_calibration import (
    ROOT, CENTER, CENTER_SHA, PBE, PBE_PREP_SHA, PBE_COLLECTION_SHA, ABACUS, ABACUS_SHA, PMI, PMI_SHA,
)
from check_c_optimized_pbe import (
    _read_hashed, _sha256, _within, _write_json, _optimizer_artifacts,
    _match_parent_initial, _require_backoff_improvement, HARTREE_TO_EV,
)
from check_c_direction_probe_pbe import _json
from check_c_combined_step_pbe import prepare_combined_pbe, collect_combined_pbe, PREPARATION, COLLECTION
from periodic_galerkin_combined_step import combine_pbe_tangent
from periodic_galerkin_basis import read_periodic_optimizer_coefficients, write_periodic_optimizer_coefficients
from periodic_galerkin_fit import CandidateGuardError, _minimum_occupied_capture
from export_periodic_orbitals import write_abacus_orbital
from backoff_c_optimized_direction import evaluate_pair

CALIBRATION_ROOT = ROOT/'stage-c-pbe-direction-calibration-5631e8d2'
CALIBRATION_SHA = 'efa6c1448a46da7d743682ac9788e395ccdfe588ec25ba4ffa6310bdc950856a'
ACCEPTANCE_SHA = '92a69a0213c7ecf891ad13a5036fcdda11457cd1a2ce9c3861f0a4c4ba1f998e'
DIRECTIONS_SHA = 'e8265dc8b871d3a25b1ff500e3eee7999ff77ba1cf571d0be1e413ec0a22ccd9'


def screen_candidate(datasets, coefficients, guard, floor, *, enforce_band_accuracy=True):
    if type(enforce_band_accuracy) is not bool:
        raise ValueError('explicit boolean band accuracy switch required')
    bands = guard(coefficients)
    if enforce_band_accuracy and bands.get('gate') is not True:
        raise CandidateGuardError('band guard failed')
    if not enforce_band_accuracy:
        for key in ('minimum_gap_ev','maximum_target_band_change_ev','occupied_band_sum_change_ev_per_atom'):
            if not math.isfinite(bands[key]):
                raise CandidateGuardError('nonfinite band diagnostic')
        if bands['minimum_gap_ev'] <= 0:
            raise CandidateGuardError('candidate gap is not positive')
    try:
        capture = _minimum_occupied_capture(datasets, coefficients,
            relative_rank_tolerance=1e-12, condition_limit=1e12)
    except RuntimeError as error:
        expected = (
            'fixed radial prefix overlap has no positive direction',
            'fixed radial prefix overlap is rank deficient',
            'fixed radial prefix overlap condition number exceeds limit',
            'fixed radial prefix does not span the occupied manifold',
            'fixed radial prefix occupied capture is non-finite',
        )
        if str(error) not in expected:
            raise
        raise CandidateGuardError(str(error)) from error
    if not math.isfinite(capture) or capture < floor:
        raise CandidateGuardError('original occupied capture floor failed')
    return bands, capture


def measure_candidate(*, candidate_path, candidate_sha256, output, center_dir,
                      center_preparation_sha256, center_collection_sha256, launcher, forward):
    output = Path(output)
    prepare_combined_pbe(center_dir=center_dir, center_preparation_sha256=center_preparation_sha256,
        center_collection_sha256=center_collection_sha256, candidate_path=candidate_path,
        candidate_sha256=candidate_sha256, output=output)
    preparation_sha = _sha256((output/PREPARATION).read_bytes())
    started = time.perf_counter()
    with (output/'abacus.out').open('xb') as stream:
        subprocess.run(launcher, cwd=output, stdout=stream, stderr=subprocess.STDOUT, check=True)
    pbe = collect_combined_pbe(prepared_dir=output, preparation_sha256=preparation_sha)
    result = dict(status='success', scope='one_actual_pbe_tangent_step', actual_pbe=pbe,
        scf_seconds=time.perf_counter()-started, rpa_evaluations=0, optimizer_steps=0,
        physical_release_gate='hold', ordinary_sos_qavg='pending', gw='pending')
    if pbe.get('pbe_gate') != 'pass':
        result['candidate_gate'] = 'rejected_actual_pbe'
        return result
    started = time.perf_counter()
    center, candidate = forward()
    result.update(center=center, candidate=candidate, rpa_evaluations=2,
                  response_seconds=time.perf_counter()-started)
    try:
        _require_backoff_improvement(center, candidate)
    except ValueError as error:
        result.update(candidate_gate='rejected_no_improvement', rejection_reason=str(error))
    else:
        result['candidate_gate'] = 'improved_frozen_body'
    result['measured_ec_delta_ev_per_c'] = (
        candidate['rpa']['candidate_energy_ha']-center['rpa']['candidate_energy_ha'])*HARTREE_TO_EV/2
    result['body_error_ev_per_c'] = abs(
        candidate['rpa']['candidate_energy_ha']-candidate['rpa']['reference_energy_ha'])*HARTREE_TO_EV/2
    return result


def verify_source(stage):
    manifest = _json(_read_hashed(stage/'DEPLOYMENT.json', os.environ['C_COMBINED_DEPLOYMENT_SHA256']))
    source = Path(manifest['source_directory']).resolve()
    if source != Path(__file__).resolve().parents[4]:
        raise ValueError('running source differs from immutable deployment')
    for name, digest in manifest['files'].items():
        _read_hashed(_within(source, name), digest)
    _read_hashed(Path(manifest['archive_path']), manifest['archive_sha256'])
    _read_hashed(ABACUS, ABACUS_SHA)
    _read_hashed(PMI, PMI_SHA)
    return manifest


def run_step(stage, source):
    started = time.perf_counter()
    acceptance = _json(_read_hashed(CALIBRATION_ROOT/'CALIBRATION_ACCEPTANCE.json', ACCEPTANCE_SHA))
    if (acceptance.get('status') != 'success' or acceptance.get('consistency_gate') != 'pass'
            or acceptance.get('result_sha256') != CALIBRATION_SHA
            or acceptance.get('directions_sha256') != DIRECTIONS_SHA
            or acceptance.get('actual_scf_count') != 8
            or any(row[1:3] != ['COMPLETED', '0:0'] for row in acceptance['scheduler'])):
        raise ValueError('strict calibration acceptance required')
    cal_bytes = _read_hashed(CALIBRATION_ROOT/'result/PBE_DIRECTION_CALIBRATION.json', CALIBRATION_SHA)
    direction_bytes = _read_hashed(CALIBRATION_ROOT/'result/probes/DIRECTIONS.json', DIRECTIONS_SHA)
    calibration, directions = _json(cal_bytes), _json(direction_bytes)
    center, _, _ = _optimizer_artifacts(CENTER, CENTER_SHA)
    for name, digest in (('result', CENTER_SHA), ('coefficient', center['coefficient_sha256']),
                         ('orbital', center['orbital_sha256'])):
        if directions.get('center_'+name+'_sha256') != digest:
            raise ValueError('direction center identity mismatch')
    if calibration.get('center_collection_sha256') != PBE_COLLECTION_SHA:
        raise ValueError('calibration actual PBE center identity mismatch')
    root = CENTER.parent
    read = lambda p: read_periodic_optimizer_coefficients(p, element='C', radial_rows=31,
                                                        expected_nu=(3, 3, 2, 0, 0))
    coefficients, original = read(root/'INTERPOLATED_COEFFICIENTS.txt'), read(root/'ORIGINAL_COEFFICIENTS.txt')
    with torch.no_grad():
        trial = combine_pbe_tangent(coefficients, directions, calibration)
    output = stage/'result'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(28)
    datasets, records, guard = load_frozen_c(root/'INPUT_FREEZE.json', center['freeze_sha256'], original,
        output, active_cache_index=root/'ACTIVE_DATA_CACHE.json',
        active_cache_index_sha256=center['active_cache_index_sha256'])
    loaded = time.perf_counter()
    floor = max(0., center['initial']['minimum_occupied_capture']-1e-4)
    with torch.no_grad():
        try:
            bands, capture = screen_candidate(datasets, trial['coefficients'], guard, floor)
        except CandidateGuardError as error:
            return dict(status='success', candidate_gate='rejected_cheap_guard',
                rejection_reason=str(error), scope='one_actual_pbe_tangent_step',
                actual_scf_count=0, rpa_evaluations=0, physical_release_gate='hold')
    candidate_root = output/'candidate'
    candidate_root.mkdir()
    for name, content in (('DIRECTIONS.json', direction_bytes), ('CALIBRATION.json', cal_bytes)):
        with (candidate_root/name).open('xb') as stream:
            stream.write(content)
    write_periodic_optimizer_coefficients(candidate_root/'COEFFICIENTS.txt', trial['coefficients'])
    write_abacus_orbital(candidate_root/'C_3s3p2d_combined.orb', trial['coefficients'], element='C',
        ecut_ry=100., rcut_bohr=10., dr_bohr=.01, smoothing_sigma_bohr=.1)
    restored = read(candidate_root/'COEFFICIENTS.txt')
    if any(not torch.equal(a, b) for a, b in zip(restored['C'], trial['coefficients']['C'])):
        raise ValueError('exported coefficients do not reconstruct exact proposal')
    proposal = {key: value for key, value in trial.items() if key not in ('coefficients', 'direction')}
    proposal.update(status='prepared', center_result_sha256=CENTER_SHA,
        center_coefficient_sha256=center['coefficient_sha256'], center_orbital_sha256=center['orbital_sha256'],
        directions_sha256=DIRECTIONS_SHA, calibration_sha256=CALIBRATION_SHA,
        coefficient_filename='COEFFICIENTS.txt', orbital_filename='C_3s3p2d_combined.orb',
        coefficient_sha256=_sha256((candidate_root/'COEFFICIENTS.txt').read_bytes()),
        orbital_sha256=_sha256((candidate_root/'C_3s3p2d_combined.orb').read_bytes()),
        cheap_gate=dict(gate=True, band_screen=bands, minimum_occupied_capture=capture,
            occupied_capture_floor=floor, overlap_relative_rank_tolerance=1e-12, overlap_condition_limit=1e12),
        source_commit=source['source_commit'])
    _write_json(candidate_root/'CANDIDATE.json', proposal)
    candidate_sha = _sha256((candidate_root/'CANDIDATE.json').read_bytes())
    def forward():
        _read_hashed(candidate_root/'CANDIDATE.json', candidate_sha)
        _read_hashed(candidate_root/'COEFFICIENTS.txt', proposal['coefficient_sha256'])
        _read_hashed(candidate_root/'C_3s3p2d_combined.orb', proposal['orbital_sha256'])
        _optimizer_artifacts(CENTER, CENTER_SHA)
        first, evaluated = evaluate_pair(datasets, coefficients, restored, guard=guard,
            frequency_batch_size=center['configuration']['frequency_batch_size'], weights=center['training_weights'],
            occupied_capture_floor=floor,
            initial_validator=lambda value: _match_parent_initial(value, center['candidate']))
        return first, evaluated
    result = measure_candidate(candidate_path=candidate_root/'CANDIDATE.json', candidate_sha256=candidate_sha,
        output=output/'pbe_slots/candidate', center_dir=PBE, center_preparation_sha256=PBE_PREP_SHA,
        center_collection_sha256=PBE_COLLECTION_SHA,
        launcher=['srun', '--mpi=pmi2', '--cpu-bind=none', '-n', '4', str(ABACUS)], forward=forward)
    result.update(source_commit=source['source_commit'], candidate_sha256=candidate_sha,
        coefficient_sha256=proposal['coefficient_sha256'], orbital_sha256=proposal['orbital_sha256'],
        center_result_sha256=CENTER_SHA, calibration_sha256=CALIBRATION_SHA,
        calibration_acceptance_sha256=ACCEPTANCE_SHA, directions_sha256=DIRECTIONS_SHA,
        active_cache_index_sha256=center['active_cache_index_sha256'], freeze_sha256=center['freeze_sha256'],
        predicted_ec_delta_ev_per_c=trial['predicted_ec_delta_ha_per_cell']*HARTREE_TO_EV/2,
        cache_load_seconds=loaded-started, total_seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, load_records=records,
        actual_scf_count=1, radius=trial['radius'], mixing_ratio=trial['mixing_ratio'],
        finite_step_extrapolation=True, mixed_curvature='unmeasured')
    if result['rpa_evaluations']:
        result['measured_to_predicted_ec_gain'] = result['measured_ec_delta_ev_per_c']/result['predicted_ec_delta_ev_per_c']
    _read_hashed(CALIBRATION_ROOT/'result/PBE_DIRECTION_CALIBRATION.json', CALIBRATION_SHA)
    _optimizer_artifacts(CENTER, CENTER_SHA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    stage = Path(parser.parse_args().stage).resolve()
    source = verify_source(stage)
    if (os.environ.get('SLURM_JOB_NUM_NODES') != '1' or os.environ.get('OMP_NUM_THREADS') != '7'
            or os.environ.get('I_MPI_PMI_LIBRARY') != str(PMI)):
        raise ValueError('frozen single-node 4x7 PMI2 layout required')
    _write_json(stage/'EXECUTION_PROVENANCE.json', dict(status='launching', server='df_dcu',
        scheduler_job=os.environ['SLURM_JOB_ID'], source_commit=source['source_commit'],
        abacus_sha256=ABACUS_SHA, pmi_sha256=PMI_SHA, actual_scf_budget=1,
        profile='frozen-C-reference-711af860c', version_verdict='feature-branch-exception',
        exception_reason='same frozen pure PBE runtime and reference; new candidate coefficients only',
        delta_st='not_run', librpa='not_run', ordinary_sos_qavg='pending', physical_release_gate='hold'))
    result = run_step(stage, source)
    verify_source(stage)
    _write_json(stage/'result/RESULT.json', result)
    _write_json(stage/'PROVENANCE.json', dict(status='success', source_commit=source['source_commit'],
        result_sha256=_sha256((stage/'result/RESULT.json').read_bytes()),
        candidate_gate=result['candidate_gate'], actual_scf_count=result['actual_scf_count'],
        rpa_evaluations=result['rpa_evaluations'], physical_release_gate='hold'))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
