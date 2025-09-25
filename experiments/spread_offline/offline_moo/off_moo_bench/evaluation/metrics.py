import numpy as np
from offline_moo.off_moo_bench.task_set import MORL
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD



def hv(nadir_point, y, task_name):
    
    if not isinstance(nadir_point, np.ndarray):
        nadir_point = np.array(nadir_point)

    if nadir_point.ndim > 1:
        nadir_point = nadir_point.flatten()

    nadir_point = nadir_point * 2.2  # if task_name not in MORL \
    # else nadir_point * 4
    # print("y shape:", np.array(y).shape)
    # print("nadir_point:", nadir_point)
    if task_name == "Molecule-Exact-v0":
        index_to_remove = np.all(y == [1.0, 1.0], axis=1)
        y = y[~index_to_remove]
    return Hypervolume(ref_point=nadir_point).do(y)


def igd(pareto_front, y):
    return IGD(pareto_front).do(y)
