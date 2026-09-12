"""Fixed Grassmann coordinates with an exact derivative and bounded SciPy L-BFGS.

No numerical-response kernel is changed here. Within-channel frame rotations
are redundant for the frozen Galerkin subspace; every horizontal degree of
freedom remains active. The local chart is not a global convergence claim.
"""

from copy import deepcopy

import numpy as np
import torch


class EvaluationBudget(RuntimeError):
    """An external or internal evaluation budget has been exhausted."""


def _flat(x, size):
    x=np.asarray(x,dtype=np.float64)
    if x.shape != (size,) or not np.isfinite(x).all():
        raise ValueError('finite flat coordinate/gradient with matching dimension required')
    return x.copy()


class GrassmannChart:
    """C_l(Z_l)=signed_QR(C0_l+Qperp_l Z_l), using a fixed orthogonal complement.

    The ambient raw gradient is pulled back through QR by automatic
    differentiation. In particular it is not treated as a Euclidean Z gradient.
    C0^T(C0+Qperp Z)=I guarantees full column rank at every finite Z.
    """

    def __init__(self, anchor):
        if not isinstance(anchor,dict) or not anchor:
            raise ValueError('nonempty element/channel mapping required')
        self.anchor={}; self.complement={}; self.dimension=0
        for element, channels in sorted(anchor.items()):
            if not isinstance(channels,(list,tuple)) or not channels:
                raise ValueError('nonempty channel sequence required')
            self.anchor[element]=[]; self.complement[element]=[]
            for c in channels:
                if (not isinstance(c,torch.Tensor) or c.ndim!=2 or c.dtype!=torch.float64
                    or c.device.type!='cpu' or not torch.isfinite(c).all()
                    or c.shape[0] < 1 or c.shape[1] > c.shape[0]):
                    raise ValueError('finite CPU float64 tall matrices required')
                n,k=c.shape
                if not torch.allclose(c.T@c,torch.eye(k,dtype=c.dtype),atol=1e-10,rtol=0):
                    raise ValueError('orthonormal anchor required')
                c=c.detach().clone()
                q=torch.linalg.qr(c,mode='complete')[0][:,k:] if k else torch.eye(n,dtype=c.dtype)
                self.anchor[element].append(c)
                self.complement[element].append(q)
                self.dimension+=(n-k)*k
        if self.dimension == 0: raise ValueError('at least one active subspace coordinate required')

    def _coefficients(self,z):
        result={}; offset=0
        for element,channels in self.anchor.items():
            result[element]=[]
            for c,qperp in zip(channels,self.complement[element]):
                n,k=c.shape; size=(n-k)*k
                if k:
                    x=c+qperp@z[offset:offset+size].reshape(n-k,k)
                    q,r=torch.linalg.qr(x,mode='reduced')
                    signs=torch.where(torch.diag(r)<0,-torch.ones(k,dtype=c.dtype),
                                      torch.ones(k,dtype=c.dtype)).detach()
                    result[element].append(q*signs)
                else: result[element].append(c.clone())
                offset+=size
        return result

    def coefficients(self,x):
        z=torch.from_numpy(_flat(x,self.dimension))
        return self._coefficients(z)

    def pullback(self,x,raw_gradient):
        if not isinstance(raw_gradient,dict) or raw_gradient.keys()!=self.anchor.keys():
            raise ValueError('gradient elements must match anchor')
        z=torch.from_numpy(_flat(x,self.dimension)).requires_grad_(True)
        c=self._coefficients(z)
        terms=[]
        for element,channels in c.items():
            gs=raw_gradient[element]
            if not isinstance(gs,(list,tuple)) or len(gs)!=len(channels):
                raise ValueError('gradient channels must match anchor')
            for b,g in zip(channels,gs):
                if (not isinstance(g,torch.Tensor) or g.shape!=b.shape or g.dtype!=b.dtype
                    or g.device.type!='cpu' or not torch.isfinite(g).all()):
                    raise ValueError('finite matching raw gradient required')
                if b.shape[1]: terms.append((b*g.detach()).sum())
        result=torch.autograd.grad(sum(terms),z)[0].detach().numpy()
        return _flat(result,self.dimension)

    def definition(self):
        return dict(chart='signed_QR(C0+Qperp*Z)',metric='Euclidean_fixed_graph_coordinates',
                    dimension=self.dimension,
                    anchor={e:[c.tolist() for c in cs] for e,cs in self.anchor.items()},
                    complement={e:[c.tolist() for c in cs] for e,cs in self.complement.items()})


def run_lbfgs(initial,value_gradient,checkpoint,*,max_steps=12,max_evaluations=25,
              coordinate_bound=.05):
    """Use analytic-gradient SciPy L-BFGS-B, retaining accepted iterates only.

    Wolfe line search needs trial gradients. Every distinct requested (f,g)
    counts, including rejected trials; callers separately count kernel work and
    cache hits. EvaluationBudget returns the last accepted point, never a trial.
    Other exceptions fail closed rather than manufacture a finite penalty.
    """
    from scipy.optimize import minimize
    if (type(max_steps) is not int or max_steps<1 or type(max_evaluations) is not int
        or max_evaluations<1 or not np.isfinite(coordinate_bound) or coordinate_bound<=0):
        raise ValueError('positive finite bounds and integer budgets required')
    x=np.asarray(initial,dtype=float)
    if x.ndim!=1 or not x.size: raise ValueError('nonempty flat initial required')
    x=_flat(x,len(x))
    if np.max(np.abs(x))>coordinate_bound: raise ValueError('initial outside chart box')
    cache={}; calls=0; accepted=0
    def fg(v):
        nonlocal calls
        v=_flat(v,len(x)); key=v.tobytes()
        if key not in cache:
            if calls >= max_evaluations: raise EvaluationBudget('L-BFGS evaluation budget')
            f,g=value_gradient(v.copy()); calls+=1
            if not np.isfinite(f) or f<0: raise ValueError('finite nonnegative objective required')
            cache[key]=(float(f),_flat(g,len(x)))
        f,g=cache[key]
        return f,g.copy()
    f,g=fg(x)
    last=dict(x=x.copy(),objective=f,gradient=g.copy(),accepted_steps=0)
    def callback(v):
        nonlocal last,accepted
        f,g=fg(v)
        if f > last['objective']+1e-13: raise ValueError('nonmonotone accepted iterate')
        step=v-last['x']; delta_g=g-last['gradient']
        accepted+=1
        last=dict(x=v.copy(),objective=f,gradient=g.copy(),accepted_steps=accepted,
                  evaluations=calls,step_norm=float(np.linalg.norm(step)),
                  s_dot_y=float(step@delta_g),actual_gain=last['objective']-f,
                  predicted_gain=-float(last['gradient']@step),
                  max_abs_coordinate=float(np.max(np.abs(v))),
                  active_bounds=int(np.count_nonzero(np.abs(v)>=coordinate_bound-1e-10)))
        checkpoint(deepcopy(last))
    info={}
    try:
        res=minimize(fg,x,jac=True,method='L-BFGS-B',callback=callback,
                     bounds=[(-coordinate_bound,coordinate_bound)]*len(x),
                     options=dict(maxiter=max_steps,maxfun=max_evaluations,maxcor=10,
                                  maxls=12,ftol=1e-14,gtol=1e-9))
        reason='solver_tolerance' if res.success else 'solver_stop'
        info=dict(scipy_status=int(res.status),scipy_message=str(res.message),
                  scipy_success=bool(res.success),scipy_nit=int(res.nit))
    except EvaluationBudget:
        reason='max_evaluations'
    return dict(last,status='diagnostic_stop',stop_reason=reason,evaluations=calls,
                physical_release_gate='hold',**info)
