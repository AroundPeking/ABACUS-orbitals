"""Bounded same-direction step-cap diagnosis; never promotes an orbital."""

import argparse
import json
import os
from pathlib import Path
import resource
import time

import screen_c_radial_expansion as parent

cycle = parent.cycle
RADII = (.04, .08, .12)
SCOPE = 'same_eight_radial_direction_step_cap_diagnostic'


def probe_radius_boundary(measure, evaluate):
    rows = []
    for radius in RADII:
        row = dict(measure(radius), radius=radius, rpa='not_run')
        if row['gate'] == 'rejected_cheap_guard':
            rows.append(row)
            continue
        if row['gate'] != 'measured':
            raise ValueError('unknown measurement gate')
        gate = row['actual_pbe']['pbe_gate']
        if gate not in ('pass', 'fail'):
            raise ValueError('explicit actual PBE gate required')
        if gate == 'fail':
            row['gate'] = 'rejected_actual_PBE'
        else:
            measured = evaluate(radius)
            cycle.finite(measured['gain_mev'])
            row.update(gate='diagnostic_measured', rpa=measured)
        rows.append(row)
    return rows


def run(stage, source, accepted, center, gradient, launcher):
    import torch
    from periodic_galerkin_radial_diagnostics import radial_descent_direction, retract_displacement
    from periodic_galerkin_basis import write_periodic_optimizer_coefficients
    from export_periodic_orbitals import write_abacus_orbital
    from run_c_combined_step import screen_candidate
    from backoff_c_optimized_direction import evaluate_pair
    from periodic_galerkin_fit import CandidateGuardError

    start = time.perf_counter()
    out = stage/'result'
    out.mkdir()
    rt = cycle.admission._runtime()
    torch.set_num_threads(28)
    c = cycle.admission.common._read_coefficients(rt, center['coefficient_path'])
    original = cycle.admission.common._read_coefficients(rt, center['original_coefficient_path'])
    pairs = [(r['element'], r['l'], z) for r in gradient['energy_gradient']['channels']
             for z in range(len(r['radials']))]
    if pairs != [('C', l, z) for l, n in enumerate((3, 3, 2)) for z in range(n)]:
        raise ValueError('all eight existing radial coordinates required')
    basis = [radial_descent_direction(c, gradient['energy_gradient'], e, l, z) for e, l, z in pairs]
    if any(b is None for b in basis):
        raise ValueError('unresolved gradient')
    direction = accepted['tangent']['radial_weights']
    if len(direction) != 8:
        raise ValueError('eight calibrated weights required')
    def trial_at(radius):
        with torch.no_grad():
            displacement = {e: [sum((radius*w*b[e][l] for w, b in zip(direction, basis)),
                                    torch.zeros_like(block)) for l, block in enumerate(ch)]
                            for e, ch in c.items()}
            return retract_displacement(c, displacement, 1.)
    def export(slot, trial):
        slot.mkdir()
        write_periodic_optimizer_coefficients(slot/'COEFFICIENTS.txt', trial)
        write_abacus_orbital(slot/'C_3s3p2d.orb', trial, element='C', ecut_ry=100.,
                            rcut_bohr=10., dr_bohr=.01, smoothing_sigma_bohr=.1)
        return dict(coefficient_sha256=cycle.endpoint._sha256((slot/'COEFFICIENTS.txt').read_bytes()),
                    orbital_sha256=cycle.endpoint._sha256((slot/'C_3s3p2d.orb').read_bytes()))

    # Exact reconstruction is cheap and is not a replay of the known SCF/RPA point.
    hashes = export(out/'known_radius_reconstruction', trial_at(.02))
    known = accepted['trials'][0]
    if known['radius'] != .02 or any(v != known['actual_pbe'][k] for k, v in hashes.items()):
        raise ValueError('cannot reproduce the exact accepted direction at radius .02')
    load_start = time.perf_counter()
    datasets, records, guard = rt.load_frozen_c(
        center['freeze_path'], center['result']['freeze_sha256'], original, out,
        active_cache_index=center['active_cache_index_path'],
        active_cache_index_sha256=center['result']['active_cache_index_sha256'],
        band_guard_policy=cycle.RESPONSE)
    cycle.admission.common.validate_dataset_extent(datasets, center)
    load_seconds = time.perf_counter()-load_start
    inputs, orbital_name, log_name = cycle.frozen_pbe_bundle(center)
    floor = center['occupied_capture_floor']
    weights = center['quarter']['training_weights']
    trials, scfs, forwards = {}, [], []
    def measure(radius):
        slot = out/('radius_%03d' % round(radius*1000))
        trial = trial_at(radius)
        hashes = export(slot, trial)
        proposal = dict(scope=SCOPE, radius=radius, radial_weights=direction,
                        parent_result_sha256=parent.RESULT_SHA, **hashes)
        try:
            with torch.no_grad():
                band, capture = screen_candidate(datasets, trial, guard, floor)
                diagnostics = guard(trial, diagnostics=True)
        except CandidateGuardError as error:
            rejected = dict(gate='rejected_cheap_guard', reason=str(error), **proposal)
            cycle.endpoint._write_json(slot/'MEASUREMENT.json', rejected)
            print(json.dumps(rejected), flush=True)
            return rejected
        proposal.update(nu=[3,3,2,0,0], fixed_nu=[0]*5, ao_per_C=22,
                        capture=capture, occupied_capture_floor=floor,
                        cheap_gate=band, band_diagnostics=diagnostics, physical_release_gate='hold')
        cycle.endpoint._write_json(slot/'CANDIDATE.json', proposal)
        newinputs = dict(inputs)
        newinputs[orbital_name] = (slot/'C_3s3p2d.orb').read_bytes()
        sample = cycle.run_scf(slot/'pbe', newinputs, log_name, launcher)
        scfs.append(radius)
        row = dict(gate='measured', actual_pbe=sample, proposal=proposal,
                   candidate_sha256=cycle.endpoint._sha256((slot/'CANDIDATE.json').read_bytes()))
        trials[radius] = (slot, row)
        cycle.endpoint._write_json(slot/'MEASUREMENT.json', row)
        print(json.dumps(dict(radius=radius, pbe_delta_mev_per_c=1000*sample['energy_delta_ev_per_c'],
                              pbe_gate=sample['pbe_gate'])), flush=True)
        return row
    def validate_center(record):
        previous = center['result']['candidate']
        checker = cycle.admission.refresh.accepted
        checker._record(record, floor, weights)
        checker._same_grid_reference(record, previous, reproduce=True)
        cycle.endpoint._match_parent_initial(record, previous)
    def evaluate(radius):
        slot, row = trials[radius]
        for name, key in (('COEFFICIENTS.txt','coefficient_sha256'), ('C_3s3p2d.orb','orbital_sha256')):
            cycle.admission.common._check_file(slot/name, row['proposal'][key])
        cycle.admission.common._check_file(slot/'CANDIDATE.json', row['candidate_sha256'])
        trial = cycle.admission.common._read_coefficients(rt, slot/'COEFFICIENTS.txt')
        first, new = evaluate_pair(
            datasets, c, trial, guard=guard,
            frequency_batch_size=center['quarter']['configuration']['frequency_batch_size'],
            weights=weights, occupied_capture_floor=floor, initial_validator=validate_center)
        checker = cycle.admission.refresh.accepted
        checker._record(new, floor, weights)
        checker._same_grid_reference(new, first)
        checker._same_grid_reference(new, known['rpa'])
        def error(record):
            r = record['rpa']
            return abs(r['candidate_energy_ha']-r['reference_energy_ha'])*cycle.endpoint.HARTREE_TO_EV/2
        measured = dict(candidate=new, center=first, body_error_ev_per_c=error(new),
                        gain_mev=(error(known['rpa'])-error(new))*1000,
                        gain_from_center_mev=(error(first)-error(new))*1000)
        cycle.endpoint._write_json(slot/'GALERKIN.json', measured)
        forwards.append(radius)
        print(json.dumps({k:v for k,v in measured.items() if k not in ('candidate','center')}), flush=True)
        return measured
    rows = probe_radius_boundary(measure, evaluate)
    return dict(status='success', scope=SCOPE, source_commit=source['source_commit'],
                parent_result_sha256=parent.RESULT_SHA, parent_verification_sha256=parent.VERIFY_SHA,
                gradient_sha256=cycle.GRADIENT_SHA, radial_weights=direction,
                known_radius_reconstructed=hashes, known_radius_reused=.02, trials=rows,
                freeze_sha256=accepted['freeze_sha256'], active_cache_index_sha256=accepted['active_cache_index_sha256'],
                actual_scf_count=len(scfs), rpa_evaluations=2*len(forwards), backward_passes=0,
                cache_loads=1, cache_load_seconds=load_seconds, load_records=records,
                total_seconds=time.perf_counter()-start,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                nu=[3,3,2,0,0], fixed_nu=[0]*5, candidate_gate='diagnostic_only',
                physical_release_gate='hold', delta_st_runs=0, ordinary_sos='not_run', gw='not_run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage','source-commit','deployment-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    stage = Path(args.stage).resolve()
    source = cycle.admission.common.verify_source(stage, args.deployment_sha256, args.source_commit)
    for name in ('diagnose_c_radial_step_cap.py','run_c_radial_step_diagnostic.slurm'):
        if cycle.admission.refresh._WORKFLOW+name not in source['files']:
            raise ValueError('diagnostic source pin missing')
    accepted, center = parent.admit_parent()
    _, gradient = cycle.admission.admit_gradient(cycle.GRADIENT_STAGE, cycle.GRADIENT_SHA, cycle.ACCEPTANCE_SHA)
    cycle.validate_stage(stage, source, center)
    runtime = center['result']['actual_pbe']
    def check_runtime():
        for pkey, hkey in (('abacus_binary','abacus_sha256'), ('mpi_library','mpi_sha256')):
            cycle.admission.common._check_file(runtime[pkey], runtime[hkey])
    check_runtime()
    if (os.environ.get('SLURM_JOB_NUM_NODES') != '1' or os.environ.get('SLURM_NTASKS') != '4'
            or os.environ.get('OMP_NUM_THREADS') != '7'
            or Path(os.environ.get('I_MPI_PMI_LIBRARY','/missing')).resolve() != Path(runtime['mpi_library']).resolve()):
        raise ValueError('one node / four MPI x seven OMP / frozen PMI2 required')
    cycle.frozen_pbe_bundle(center)
    if args.preflight_only:
        return
    reservation = cycle.ROOT/('radial-step-boundary-'+parent.RESULT_SHA[:16])
    reservation.mkdir()
    cycle.endpoint._write_json(reservation/'RESERVATION.json', dict(
        stage=str(stage), job_id=os.environ['SLURM_JOB_ID'], scope=SCOPE, radii=RADII,
        parent_result_sha256=parent.RESULT_SHA, actual_scf_budget=3, physical_release_gate='hold'))
    result = run(stage, source, accepted, center, gradient,
                 ['srun','--mpi=pmi2','--cpu-bind=none','-n','4',runtime['abacus_binary']])
    check_runtime()
    parent.admit_parent()
    cycle.admission.common.verify_source(stage, args.deployment_sha256, args.source_commit)
    result['deployment_sha256'] = args.deployment_sha256
    cycle.endpoint._write_json(stage/'result/RESULT.json', result)
    cycle.endpoint._write_json(stage/'PROVENANCE.json', dict(
        status='success', source_commit=args.source_commit, deployment_sha256=args.deployment_sha256,
        result_sha256=cycle.endpoint._sha256((stage/'result/RESULT.json').read_bytes()),
        job_id=os.environ['SLURM_JOB_ID'], abacus_sha256=runtime['abacus_sha256'],
        pmi_sha256=runtime['mpi_sha256'], physical_release_gate='hold'))


if __name__ == '__main__':
    main()
