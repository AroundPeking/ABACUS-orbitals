"""Deterministic smooth seed expansion, not a response-optimal orbital claim."""

import torch


def append_smooth_complement(coefficients, element, l, *, max_index=12):
    block=coefficients[element][l]
    if (block.ndim != 2 or block.dtype != torch.float64 or block.device.type != 'cpu'
            or not bool(torch.isfinite(block).all()) or type(max_index) is not int
            or not 0 < max_index <= block.shape[0]):
        raise ValueError('finite CPU float64 block and bounded primitive count required')
    with torch.no_grad():
        if block.shape[1]:
            u,s,_=torch.linalg.svd(block,full_matrices=False)
            if float(s.min()) <= float(s.max())*1e-12:
                raise ValueError('existing radial block is rank deficient')
        else:
            u=block
        for index in range(max_index):
            unit=torch.zeros(block.shape[0],dtype=torch.float64); unit[index]=1
            column=unit-u@(u.T@unit)
            norm=float(torch.linalg.norm(column))
            if norm < .1:
                continue
            column=column/norm
            result={e:[x.detach().clone() for x in channels] for e,channels in coefficients.items()}
            result[element][l]=torch.cat((block.detach().clone(),column[:,None]),dim=1)
            return result,dict(primitive_index=index,residual_norm=norm,
                search_primitive_count=max_index,selection='lowest_index_residual_norm_at_least_0.1',
                complement_metric='Euclidean_coefficient',old_columns='byte_identical',
                shape_scope='projected_low_index_primitive_not_strict_lowpass',
                coefficients_above_search_window_norm=float(column[max_index:].norm()),
                optimality='unoptimized_seed_not_channel_optimum')
    raise ValueError('no resolved complement in the permitted primitive window')
