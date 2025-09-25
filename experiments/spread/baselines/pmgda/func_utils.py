import math
import torch
import random
import numpy as np
import pickle
import copy

import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dic_ref_point = {
    "zdt1": [0.9994, 6.0576],
    "zdt2": [0.9994, 6.8960],
    "zdt3": [0.9994, 6.0571],
    "dtlz2": [2.8390, 2.9011, 2.8575],
    "dtlz4": [3.2675, 2.6443, 2.4263],
    "dtlz7": [0.9984, 0.9961, 22.8114],
    "re21": [3144.44, 0.05],
    "re33": [5.01, 9.84, 4.30],
    "re34": [1.86472022e+03, 1.18199394e+01, 2.90399938e-01],
    "re37": [1.1022, 1.20726899, 1.20318656],
    "re41": [47.04480682,  4.86997366, 14.40049127, 10.3941957 ],
}

def get_ref_point(problem='zdt1'):
    problem = problem.lower()
    try:
        ref_point = dic_ref_point[problem]
    except KeyError:
        raise ValueError(f"Unknown problem: {problem}. Available problems: {list(dic_ref_point.keys())}")
    
    return np.array(ref_point)


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def mean_std_stats(values, to_decimal=2):
    """
    Rounds each value in the list to two decimal places and returns the mean and standard deviation.

    Parameters:
        values (list of float): List of numeric values.

    Returns:
        tuple: A tuple (mean, std) of the rounded values.
    """
    # Round each value to two decimals
    rounded_values = [round(v, to_decimal) for v in values]

    # Compute mean and standard deviation using numpy
    mean_val = np.mean(rounded_values)
    std_val = np.std(rounded_values)

    return mean_val, std_val
    
def report_stats(args):
    print(f"Problem: {args.problem}")

    list_hv = []
    mean_std_hv_results = {}

    for seed in args.list_seed:
        name = (
            args.method
            + "_"
            + args.problem
            + "_"
            + str(seed)
            + "_"
            + f"T={args.iters}"
            + "_"
            + f"N={args.num_samples}"
        )

        path_to_results = args.results_store_path + name + "_hv_results.pkl"
        # val_hv = get_value_from_json(path_to_results, "hypervolume")
        with open(path_to_results, "rb") as f:
            hv_results = pickle.load(f)
            
        mean_std_hv_results[f"HV_seed_{seed}"] = hv_results["hypervolume"]
        list_hv.append(hv_results["hypervolume"])

    mean_std_hv_results["mean_std_hypervolume"] = mean_std_stats(list_hv, to_decimal=2)

    with open(
        args.results_store_path + "mean_std_hv_results" + name + ".pkl", "wb"
    ) as f:
        pickle.dump(mean_std_hv_results, f)

    print("mean_std_hypervolume", mean_std_stats(list_hv, to_decimal=2))

    return mean_std_hv_results

def eps_dominance(Obj_space, alpha=0.0):
    epsilon = alpha * np.min(Obj_space, axis=0)
    N = len(Obj_space)
    Pareto_set_idx = list(range(N))
    Dominated = []
    for i in range(N):
        candt = Obj_space[i] - epsilon
        for j in range(N):
            if np.all(candt >= Obj_space[j]) and np.any(candt > Obj_space[j]):
                Dominated.append(i)
                break
    PS_idx = list(set(Pareto_set_idx) - set(Dominated))
    return PS_idx


def get_non_dominated_points(
    points_pred, p_front, keep_shape=True, indx_only=False
):
    pf_points = copy.deepcopy(points_pred.detach())
    
    PS_idx = eps_dominance(p_front)

    if indx_only:
        return PS_idx
    
    elif keep_shape:
        PS_idx = np.sort(PS_idx)
        # Create an array of all indices
        all_indices = np.arange(p_front.shape[0])
        # Identify the indices not in PS_idx
        not_in_PS_idx = np.setdiff1d(all_indices, PS_idx)
        # For each index not in PS_idx, find the nearest index in PS_idx
        for idx in not_in_PS_idx:
            # Compute the distance to all indices in PS_idx
            distances = np.abs(PS_idx - idx)
            nearest_idx = PS_idx[np.argmin(distances)]  # Find the closest index
            pf_points[idx] = pf_points[
                nearest_idx
            ]  # Replace with the value at the closest index

        return pf_points, points_pred, PS_idx

    else:
        return pf_points[PS_idx], points_pred, PS_idx
    

def plot_pareto_front(list_fi, args, extra = None, lab = None):

    img_path = f"figs/{args.method}/{args.problem}"
    if not (os.path.exists(img_path)):
        os.makedirs(img_path)
    name = f'{img_path}/{args.problem}_{args.seed}'

    if len(list_fi) > 3:
        return None

    elif len(list_fi) == 2:
        if extra is not None:
            f1, f2 = extra
            plt.scatter(f1, f2, c="blue", s = 5)
            
        f1, f2 = list_fi
        plt.scatter(f1, f2, c="red", s = 10)
        
        plt.xlabel("$f_1$", fontsize=14)
        plt.ylabel("$f_2$", fontsize=14)
        # plt.legend()

    elif len(list_fi) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        if extra is not None:
            f1, f2, f3 = extra
            ax.scatter(f1, f2, f3, c="blue")
            
        f1, f2, f3 = list_fi
        ax.scatter(f1, f2, f3, c="red")
        
        ax.set_xlabel("$f_1$", fontsize=14)
        ax.set_ylabel("$f_2$", fontsize=14)
        ax.set_zlabel("$f_3$", fontsize=14)
        ax.view_init(elev=30, azim=45)
        # ax.legend()

    if lab is None:
        plt.savefig(
            f"{name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"{name}_{lab}.jpg",
            dpi=300,
            bbox_inches="tight",
        )