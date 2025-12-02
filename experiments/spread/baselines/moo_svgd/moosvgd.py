import math
import torch
import random
import numpy as np
import pickle
import copy

import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def solve_min_norm_2_loss(grad_1, grad_2):
    v1v1 = torch.sum(grad_1*grad_1, dim=1)
    v2v2 = torch.sum(grad_2*grad_2, dim=1)
    v1v2 = torch.sum(grad_1*grad_2, dim=1)
    gamma = torch.zeros_like(v1v1)
    gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
    gamma[v1v2>=v1v1] = 0.999
    gamma[v1v2>=v2v2] = 0.001
    gamma = gamma.view(-1, 1)
    g_w = gamma.repeat(1, grad_1.shape[1])*grad_1 + (1.-gamma.repeat(1, grad_2.shape[1]))*grad_2

    return g_w

def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

def kernel_functional_rbf(losses):
    n = losses.shape[0]
    pairwise_distance = torch.norm(losses[:, None] - losses, dim=2).pow(2)
    h = median(pairwise_distance) / math.log(n)
    kernel_matrix = torch.exp(-pairwise_distance / 5e-6*h) #5e-6 for zdt1,2,3 (no bracket)
    return kernel_matrix

def get_mgd_grad(grads):
    """
    Compute the MGDA combined descent direction given a list of gradient tensors.
    All grads are assumed to have the same shape (parameters' shape).
    Returns a tensor of the same shape as each gradient, representing the direction g.
    """
    m = len(grads)

    # Flatten gradients and stack into matrix of shape (m, p), where p is number of params
    flat_grads = [g.reshape(-1) for g in grads]
    G = torch.stack(flat_grads, dim=0)  # shape: (m, p)
    # Compute Gram matrix of size (m, m): entry (i,j) = g_i \cdot g_j
    gram_matrix = G @ G.t()  # shape: (m, m)

    # Solve quadratic problem: minimize 0.5 * alpha^T Gram * alpha s.t. sum(alpha)=1, alpha>=0
    # We use the closed-form solution via KKT for equality constraint, then adjust for alpha>=0.
    ones = torch.ones(m, device=gram_matrix.device, dtype=gram_matrix.dtype)
    # Solve Gram * alpha = mu * 1 (plus sum(alpha)=1). This is a linear system with Lagrange multiplier mu.
    # Use pseudo-inverse in case Gram is singular.
    inv_gram = torch.linalg.pinv(gram_matrix)
    alpha = inv_gram @ ones  # solution of Gram * alpha = 1 (unnormalized)
    alpha = alpha / alpha.sum()  # enforce sum(alpha) = 1

    # Clamp negative weights to 0 and renormalize if needed (active-set correction for constraints)
    if (alpha < 0).any():
        alpha = torch.clamp(alpha, min=0.0)
        if alpha.sum() == 0:
            # If all alpha became 0 (numerical issues), fall back to equal weights
            alpha = torch.ones(m, device=alpha.device) / m
        else:
            alpha = alpha / alpha.sum()

    # Compute the combined gradient direction g
    # Reshape each gradient to original shape and sum with weights
    g = torch.zeros_like(grads[0])
    for weight, grad in zip(alpha, grads):
        g += weight * grad
    return g

def get_gradient(list_grads, inputs, losses):
    n = inputs.size(0)
    #inputs = inputs.detach().requires_grad_(True)
    if len(list_grads) == 2:
        g_w = solve_min_norm_2_loss(list_grads[0], list_grads[1])
    else:
        g_w = get_mgd_grad(list_grads)
    
    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    kernel = kernel_functional_rbf(losses)
    kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs, allow_unused=True)[0]

    gradient = (kernel.mm(g_w) - kernel_grad) / n

    return gradient


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

    img_path = f"figs/{args.problem}"
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