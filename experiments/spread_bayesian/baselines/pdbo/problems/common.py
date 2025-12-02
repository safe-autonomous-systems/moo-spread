import numpy as np
from problems import *
from external import lhs
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.lhs import LHS  # LHS sampler

def get_problem_options():
    problems = [
        ('zdt1', ZDT1),
        ('zdt2', ZDT2),
        ('zdt3', ZDT3),
        ('dtlz2', DTLZ2),
        ('dtlz5', DTLZ5),
        ('dtlz7', DTLZ7),
        ('carside', CARSIDE),
        ('penicillin', PENICILLIN),
        ('branincurrin', BRANINCURRIN),
    ]
    return problems

# Pre-build a name→class mapping for O(1) lookups
_PROBLEM_MAP = { name: cls for name, cls in get_problem_options() }


def get_problem(name, *args, d={}, **kwargs):
    key = name.lower()
    try:
        problem_cls = _PROBLEM_MAP[key]
    except KeyError:
        available = ', '.join(sorted(_PROBLEM_MAP.keys()))
        raise ValueError(f"Unknown problem '{name}'. Available options: {available}")
    return problem_cls(*args, **kwargs)


def generate_initial_samples(problem, n_sample):
    """
    Generate feasible initial samples using CV instead of 'feasible'.
    
    Args:
        problem   : a pymoo Problem instance
        n_sample  : number of feasible points to return
    
    Returns:
        X (n_sample x n_var), Y (n_sample x n_obj)
    """
    X_feas = np.zeros((0, problem.n_var))
    Y_feas = np.zeros((0, problem.n_obj))

    sampler = LHS()

    # keep sampling until we accumulate enough feasible points
    while X_feas.shape[0] < n_sample:
        # draw n_sample candidate points in [0,1]^n_var
        Xcand = sampler.do(problem, n_sample).get("X")
        # scale to the problem bounds
        xl, xu = problem.xl, problem.xu
        Xcand = xl + (xu - xl) * Xcand

        # evaluate objectives and CV
        # return_values_of accepts "F" and "CV" for unconstrained problems (CV=0) or constrained ones
        F, CV = problem.evaluate(Xcand, return_values_of=["F", "CV"])
        # feasibility mask: CV ≤ 0 means no violation
        mask = (CV <= 0).flatten()

        # append only the feasible ones
        if np.any(mask):
            X_feas = np.vstack([X_feas, Xcand[mask]])
            Y_feas = np.vstack([Y_feas, F[mask]])

    # randomly select exactly n_sample from the pooled feasible points
    idx = np.random.permutation(X_feas.shape[0])[:n_sample]
    X_init = X_feas[idx]
    Y_init = Y_feas[idx]
    return X_init, Y_init


def build_problem(name, n_var, n_obj, n_init_sample, n_process=1):
    '''
    Build optimization problem from name, get initial samples
    Input:
        name: name of the problem (supports ZDT1-6, DTLZ1-7)
        n_var: number of design variables
        n_obj: number of objectives
        n_init_sample: number of initial samples
        n_process: number of parallel processes
    Output:
        problem: the optimization problem
        X_init, Y_init: initial samples
        pareto_front: the true pareto front of the problem (if defined, otherwise None)
    '''
    # build problem
    if name.startswith('zdt') or name == 'vlmop2':
        problem = get_problem(name, n_var=n_var)
        pareto_front = problem.pareto_front()
    elif name.startswith('dtlz'):
        problem = get_problem(name, n_var=n_var, n_obj=n_obj)
        if n_obj <= 3 and name in ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4']:
            ref_kwargs = dict(n_points=100) if n_obj == 2 else dict(n_partitions=15)
            ref_dirs = get_reference_directions('das-dennis', n_dim=n_obj, **ref_kwargs)
            pareto_front = problem.pareto_front(ref_dirs)
        elif n_obj == 3 and name in ['dtlz5', 'dtlz6']:
            pareto_front = problem.pareto_front()
        else:
            pareto_front = None
    elif name.startswith('wfg'):
        problem = get_problem(name, n_var=n_var, n_obj=n_obj)
        if n_obj == 3:
            pareto_front = problem.pareto_front()
        else:
            pareto_front = None
    else:
        try:
            problem = get_problem(name)
        except:
            raise NotImplementedError('problem not supported yet!')
        try:
            pareto_front = problem.pareto_front()
        except:
            pareto_front = None

    # get initial samples
    X_init, Y_init = generate_initial_samples(problem, n_init_sample)
    
    return problem, pareto_front, X_init, Y_init