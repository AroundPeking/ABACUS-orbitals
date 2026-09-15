"""Join admitted finite-q sources to original target-k H/S/O, in cache gauges."""
import numpy as np
from audit_c_jy_operator_restart import difference, occupied_gauge, transform_source


def validate_source_audit(audit, iq, *, full_source_primitive_count=1550,
                          compared_primitive_count=558):
    rows = audit.get('per_k', [])
    if not (audit.get('status') == 'success'
            and audit.get('finite_q_spd_source_compatibility') == 'pass'
            and audit.get('failure_reasons') == [] and audit.get('selected_iq') == iq
            and audit.get('regenerated_hamiltonian_read') is False
            and audit.get('regenerated_hamiltonian_admitted') is False
            and audit.get('full_source_primitive_count') == full_source_primitive_count
            and audit.get('compared_primitive_count') == compared_primitive_count
            and len(rows) == 64 and all(r['pass_gate'] for r in rows)
            and sorted(r['source_ik'] for r in rows) == list(range(1,65))
            and sorted(r['target_ik'] for r in rows) == list(range(1,65))):
        raise ValueError('complete same-q admitted source audit required')


def anchor_nested_operator_record(expanded, baseline, prefix_indices):
    """Make the complete lower-row mother an exact subspace of an expansion."""
    names = ('overlap', 'hamiltonian_ha', 'occupied_projection', 'source')
    if set(expanded) != set(names) or set(baseline) != set(names):
        raise ValueError('complete expanded and baseline operators required')
    size = expanded['overlap'].shape[0]
    old_size = baseline['overlap'].shape[0]
    prefix = tuple(prefix_indices)
    if (expanded['overlap'].shape != (size, size)
            or expanded['hamiltonian_ha'].shape != (size, size)
            or baseline['overlap'].shape != (old_size, old_size)
            or baseline['hamiltonian_ha'].shape != (old_size, old_size)
            or len(prefix) != old_size or len(set(prefix)) != old_size
            or any(type(index) is not int or not 0 <= index < size
                   for index in prefix)
            or expanded['occupied_projection'].shape[-1] != size
            or expanded['source'].shape[-1] != size
            or baseline['occupied_projection'].shape[:-1]
               != expanded['occupied_projection'].shape[:-1]
            or baseline['source'].shape[:-1] != expanded['source'].shape[:-1]
            or baseline['occupied_projection'].shape[-1] != old_size
            or baseline['source'].shape[-1] != old_size):
        raise ValueError('incompatible nested operator dimensions')
    result = {name: np.array(value, copy=True)
              for name, value in expanded.items()}
    result['overlap'][np.ix_(prefix, prefix)] = baseline['overlap']
    result['hamiltonian_ha'][np.ix_(prefix, prefix)] = baseline['hamiltonian_ha']
    result['occupied_projection'][..., prefix] = baseline['occupied_projection']
    result['source'][..., prefix] = baseline['source']
    exact = (np.array_equal(result['overlap'][np.ix_(prefix, prefix)],
                            baseline['overlap'])
             and np.array_equal(
                 result['hamiltonian_ha'][np.ix_(prefix, prefix)],
                 baseline['hamiltonian_ha'])
             and np.array_equal(result['occupied_projection'][..., prefix],
                                baseline['occupied_projection'])
             and np.array_equal(result['source'][..., prefix],
                                baseline['source']))
    if not exact:
        raise ValueError('old mother was not anchored exactly')
    return result, dict(exact_old_mother_anchor=True,
                        old_primitive_count=old_size,
                        expanded_primitive_count=size)


def join_operator_record(old, original_array, source_array, maps, indices, size,
                         *, require_hamiltonian_subblock=True):
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
    hamiltonian_pass = checks['hamiltonian_ha']['max_abs'] <= 1e-10
    checks['hamiltonian_subblock_gate'] = (
        'required_and_pass' if require_hamiltonian_subblock and hamiltonian_pass
        else 'required_and_fail' if require_hamiltonian_subblock
        else 'diagnostic_only_before_exact_anchor')
    checks['pass_gate'] = (checks['overlap']['max_abs'] <= 1e-10
        and (hamiltonian_pass or not require_hamiltonian_subblock)
        and o_check['relative'] <= 1e-8 and o_check['unitarity'] <= 1e-8
        and checks['source']['relative'] <= 1e-6)
    if not checks['pass_gate']:
        raise ValueError('joined spd subblock failed: '+str(checks))
    return dict(overlap=s, hamiltonian_ha=h, occupied_projection=o, source=d), checks


def join_nested_operator_record(
        old, baseline_original_array, baseline_source_array, baseline_maps,
        expanded_original_array, expanded_source_array, expanded_maps, *,
        baseline_indices, expanded_indices, prefix_indices, baseline_size,
        expanded_size):
    """Join both mother sizes in cache gauges and exactly anchor the old one."""
    baseline, baseline_checks = join_operator_record(
        old, baseline_original_array, baseline_source_array, baseline_maps,
        baseline_indices, baseline_size)
    expanded, expanded_checks = join_operator_record(
        old, expanded_original_array, expanded_source_array, expanded_maps,
        expanded_indices, expanded_size,
        require_hamiltonian_subblock=False)
    anchored, anchor_checks = anchor_nested_operator_record(
        expanded, baseline, prefix_indices)
    return anchored, dict(baseline=baseline_checks, expanded=expanded_checks,
                          anchor=anchor_checks)


def map_and_anchor_expanded_gamma_record(
        expanded_native, baseline, *, occupied_map, auxiliary_map,
        prefix_indices):
    """Express an expanded Gamma record in the old gauges, then anchor it."""
    mapped = dict(
        overlap=expanded_native['overlap'],
        hamiltonian_ha=expanded_native['hamiltonian_ha'],
        occupied_projection=occupied_map.conj().T
                            @ expanded_native['occupied_projection'],
        source=transform_source(expanded_native['source'],
                                occupied_map.conj().T,
                                auxiliary_map.conj().T),
    )
    return anchor_nested_operator_record(mapped, baseline, prefix_indices)
