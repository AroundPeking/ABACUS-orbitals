"""Join admitted finite-q sources to original target-k H/S/O, in cache gauges."""
import numpy as np
from audit_c_jy_operator_restart import difference, occupied_gauge, transform_source


def validate_source_audit(audit, iq):
    rows = audit.get('per_k', [])
    if not (audit.get('status') == 'success'
            and audit.get('finite_q_spd_source_compatibility') == 'pass'
            and audit.get('failure_reasons') == [] and audit.get('selected_iq') == iq
            and audit.get('regenerated_hamiltonian_read') is False
            and audit.get('regenerated_hamiltonian_admitted') is False
            and audit.get('full_source_primitive_count') == 1550
            and audit.get('compared_primitive_count') == 558
            and len(rows) == 64 and all(r['pass_gate'] for r in rows)
            and sorted(r['source_ik'] for r in rows) == list(range(1,65))
            and sorted(r['target_ik'] for r in rows) == list(range(1,65))):
        raise ValueError('complete same-q admitted source audit required')


def join_operator_record(old, original_array, source_array, maps, indices, size):
    source, target = old['source_ik'], old['target_ik']
    s = original_array(1, target)
    h = .5*original_array(6, target)
    o = original_array(7, target)
    if s.shape != (size, size) or h.shape != s.shape or o.shape[-1] != size:
        raise ValueError('full original operator dimensions differ')
    # The cache occupied rows can have a target-dependent unitary convention.
    b, o_check = occupied_gauge(o[:, indices], old['occupied_projection'])
    o = b@o
    a = maps['occupied_at_k%d' % source]
    t = maps['auxiliary_map']
    d = source_array(2, source).reshape(old['source'].shape[0], t.shape[0], size)
    d = transform_source(d, a.conj().T, t.conj().T)
    checks = dict(overlap=difference(old['overlap'], s[np.ix_(indices,indices)]),
        hamiltonian_ha=difference(old['hamiltonian_ha'], h[np.ix_(indices,indices)]),
        occupied=o_check, source=difference(old['source'], d[:,:,indices]))
    checks['pass_gate'] = (checks['overlap']['max_abs'] <= 1e-10
        and checks['hamiltonian_ha']['max_abs'] <= 1e-10
        and o_check['relative'] <= 1e-8 and o_check['unitarity'] <= 1e-8
        and checks['source']['relative'] <= 1e-6)
    if not checks['pass_gate']:
        raise ValueError('joined spd subblock failed: '+str(checks))
    return dict(overlap=s, hamiltonian_ha=h, occupied_projection=o, source=d), checks
