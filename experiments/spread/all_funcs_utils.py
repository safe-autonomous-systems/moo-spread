import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn.functional as F

from pymoo.indicators.hv import HV

from lhs import LHS

import random
import cv2
import glob
import re

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    
def mgd_armijo_step(
    x_t: torch.Tensor,
    d: torch.Tensor,
    f_old,
    grads, # (N, m, d)
    problem,
    eta_init = 0.9,
    rho = 0.9,
    c1 = 1e-4,
    max_backtracks = 10,
):
    """
    Batched Armijo back-tracking line search for Multiple-Gradient-Descent (MGD).

    Returns
    -------
    eta  : torch.Tensor, shape (N,)
        Final step sizes.
    """
    
    x = x_t.clone().detach()
    
    if not torch.is_tensor(eta_init):
        eta = torch.full((x.shape[0],), float(eta_init), 
                           dtype=x.dtype, device=x.device)
    else:
        eta = eta_init.clone().to(x)
        
    grad_dot = torch.einsum('nkd,nd->nk', grads, d)

    improve = torch.ones_like(eta, dtype=torch.bool)

    for _ in range(max_backtracks):
        if not improve.any():
            break

        # Evaluate objectives at trial points
        trial_x = x[improve] + eta[improve, None] * d[improve]
        f_new = problem.evaluate(trial_x)

        # Armijo test (vectorised over objectives)
        # f_new <= f_old + c1 * eta * grad_dot  (element-wise)
        ok = (f_new <= f_old[improve] + c1 * eta[improve, None] * grad_dot[improve]).all(dim=1)

        # Update masks and step sizes
        eta[improve] = torch.where(ok, eta[improve], rho * eta[improve])
        improve_mask = improve.clone()
        improve[improve_mask] = ~ok

    return eta[:, None]

def extract_iteration(filename):
    # Matches digits following 'ParetoFront_iter' and before '.jpg'
    match = re.search(r"ParetoFront_iter(\d+)\.jpg", filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1


def create_video_with_images(image_folder, video_filename):
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    images = sorted(images, key=extract_iteration, reverse=True)
    
    # Read the first image to obtain the frame size
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the video writer: 'mp4v' codec, 10 fps, and frame size
    video = cv2.VideoWriter(
        video_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        1,  # lower number means slower video
        (width, height),
    )

    for image in images:
        frame = cv2.imread(image)
        resized_frame = cv2.resize(frame, (width, height))
        video.write(resized_frame)

    video.release()
    cv2.destroyAllWindows()


def load_model_checkpoint(
    model,
    model_dir,
    filename,
    verbose=True,
):

    filename = str(f"{model_dir}/{filename}.pth")
    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if verbose:
        print("Model loaded from: ", filename)

    return model


def convert_seconds(seconds):
    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    # Format the result
    print(f"Time: {hours} hours {minutes} minutes {remaining_seconds} seconds")


def l_simple_loss(predicted_noise, actual_noise):
    return nn.MSELoss()(predicted_noise, actual_noise)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                    produces the cumulative product of (1-beta) up to that
                    part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                    prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.from_numpy(np.array(betas)).float()


# Noise Scheduler (Cosine Schedule)
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta values over timesteps.
    """
    return betas_for_alpha_bar(
        timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )


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
    points_pred=None, problem=None, keep_shape=True, indx_only=False, p_front=None
):
    if not indx_only and points_pred is None:
        raise ValueError("points_pred cannot be None when indx_only is False.")
    if points_pred is not None:
        pf_points = copy.deepcopy(points_pred.detach())
        p_front = problem.evaluate(pf_points).cpu().numpy()
    else:
        assert p_front is not None, "p_front must be provided if points_pred is None."
    
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


def crowding_distance(points):
    """
    Compute crowding distances for points.
    points: Tensor of shape (N, D) in the objective space.
    Returns: Tensor of shape (N,) containing crowding distances.
    """
    N, D = points.shape
    distances = torch.zeros(N, device=points.device)

    for d in range(D):
        sorted_points, indices = torch.sort(points[:, d])
        distances[indices[0]] = distances[indices[-1]] = float("inf")

        min_d, max_d = sorted_points[0], sorted_points[-1]
        norm_range = max_d - min_d if max_d > min_d else 1.0

        # Compute normalized crowding distance
        distances[indices[1:-1]] += (
            sorted_points[2:] - sorted_points[:-2]
        ) / norm_range

    return distances


def select_top_n_candidates(
    points: torch.Tensor,
    n,
    problem,
    ref_point,
    style="crowding",
) -> torch.Tensor:
    """
    Selects the top `n` points from `points` based on a given style.

    Args:
        points (torch.Tensor): Candidate solutions (shape [N, D]).
        n: number of points to select
        problem: problem specification
        style: e.g. "crowding" for selecting points based on crowding distance.

    Returns:
        torch.Tensor: The best subset of points (shape [n, D]).
    """

    if style == "crowding":
        full_p_front = problem.evaluate(points)
        distances = crowding_distance(full_p_front)
        top_indices = torch.topk(distances, n).indices
        shuffled_idx = top_indices[torch.randperm(top_indices.size(0))]
        return points[shuffled_idx]
    
    elif style == "hv": # ( not recommended: too expensive !!!)
        assert ref_point is not None
        points_np = points.cpu().numpy()
        full_p_front = problem.evaluate(points).detach().cpu().numpy()
        hv = HV(ref_point=ref_point)
        full_hv = hv(full_p_front)

        hv_contributions = []
        for idx in range(n):
            subset = np.delete(points_np, idx, axis=0)
            subset_p_front = problem.evaluate(torch.from_numpy(subset).to(points.device)).detach().cpu().numpy()
            subset_hv = hv(subset_p_front)
            contribution = full_hv - subset_hv
            hv_contributions.append(contribution)

        # Select top n points by hypervolume contribution
        top_indices = np.argsort(hv_contributions)[::-1][:n]
        shuffled_idx = top_indices[torch.randperm(top_indices.size(0))]
        return points[shuffled_idx]

    else:
        raise ValueError(f"Unknown style: {style}")


def solve_min_norm_2_loss(grad_1, grad_2, return_gamma=False):
    v1v1 = torch.sum(grad_1 * grad_1, dim=1)
    v2v2 = torch.sum(grad_2 * grad_2, dim=1)
    v1v2 = torch.sum(grad_1 * grad_2, dim=1)
    gamma = torch.zeros_like(v1v1)
    gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
    gamma[v1v2 >= v1v1] = 0.999
    gamma[v1v2 >= v2v2] = 0.001
    gamma = gamma.view(-1, 1)
    g_w = (
        gamma.repeat(1, grad_1.shape[1]) * grad_1
        + (1.0 - gamma.repeat(1, grad_2.shape[1])) * grad_2
    )
    if return_gamma:
        return g_w, torch.cat(
            (gamma.reshape(1, -1), (1.0 - gamma).reshape(1, -1)),
            dim=0,
        )

    return g_w


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


def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.0


def repulsion_loss(F_, sigma=1.0, use_sigma=True):
    """
    Computes the repulsion loss over a batch of points in the objective space.
    F_: Tensors of shape (n, m), where n is the batch size.
    Only unique pairs (i < j) are considered.
    """
    n = F_.shape[0]
    # Compute pairwise differences: shape [n, n, m]
    dist_sq = torch.norm(F_[:, None] - F_, dim=2).pow(2)
    # Compute RBF values for the distances
    if use_sigma:
        repulsion = torch.exp(-dist_sq / (2 * sigma**2))
    else:
        s = median(dist_sq) / math.log(n)
        repulsion = torch.exp(-dist_sq / 5e-6 * s)

    # Normalize by the number of pairs
    loss = repulsion.sum() / n
    return loss


def adaptive_scale_delta_vect(
    g: torch.Tensor, delta_raw: torch.Tensor, grads: torch.Tensor, gamma: float = 0.9
) -> torch.Tensor:
    """
    Adaptive scaling to preserve *positivity*:

      ∇f_j(x_i)^T [ g_i + rho_i * delta_raw_i ] > 0  for all j.

    Args:
        g (torch.Tensor):         [n_points, d], the multi-objective "gradient"
                                  (which we *subtract* in the update).
        delta_raw (torch.Tensor): [n_points, d] or [1, d], the unscaled diversity/repulsion direction.
        grads (torch.Tensor):     [m, n_points, d], storing ∇f_j(x_i).
        gamma (float):            Safety factor in (0,1).

    Returns:
        delta_scaled (torch.Tensor): [n_points, d], scaled directions s.t.
        for all j:  grads[j,i]ᵀ [g[i] + delta_scaled[i]] > 0.
    """

    # 1) Compute alpha_{i,j} = ∇f_j(x_i)^T g_i
    #    shape of alpha: [n_points, m]
    alpha = torch.einsum("j i d, i d -> i j", grads, g)

    # 2) Compute beta_{i,j} = ∇f_j(x_i)^T delta_raw_i
    #    shape of beta: [n_points, m]
    beta = torch.einsum("j i d, i d -> i j", grads, delta_raw)

    # 3) We only need to restrict rho_i if alpha_{i,j} > 0 and beta_{i,j} < 0.
    #    Because for alpha + rho*beta to stay > 0, we need
    #        rho < alpha / -beta
    #    when beta<0 and alpha>0.
    mask = (alpha > 0.0) & (beta < 0.0)

    # Prepare an array of ratios = alpha / -beta, default +∞
    ratio = torch.full_like(alpha, float("inf"))

    # Where mask is True, compute ratio_{i,j}
    ratio[mask] = alpha[mask] / (-beta[mask])  # must remain below this

    # 4) For each point i, we pick rho_i = gamma * min_j ratio[i,j].
    #    If the min is +∞ => no constraints => set rho_i=1.0
    ratio_min, _ = ratio.min(dim=1)  # [n_points]
    rho = gamma * ratio_min
    # If ratio_min == +∞ => no constraint => set rho_i=1.
    inf_mask = torch.isinf(ratio_min)
    rho[inf_mask] = 1.0

    # 5) Scale delta_raw by rho_i
    delta_scaled = delta_raw * rho.unsqueeze(1)  # broadcast along dim

    return delta_scaled


def solve_for_h(
    x_t_minus,
    t,
    problem,
    g_val,
    grads,
    g_w,
    args,
    eta,
    lambda_rep,
    sigma=1.0,
    use_sigma=False,
    num_inner_steps=10,
    lr_inner=1e-2,
):
    """
    For a given batch x_t and its corresponding g(x_t), solve for h by
    minimizing -L (i.e., maximizing L) via gradient descent.

    Args:
        x_t: Tensor of shape (batch_size, input_dim), current points.
        eta: Step size for the update: x_{t-1} = x_t - eta * h.
        sigma: Parameter for the RBF kernel.
        lambda_rep: Coefficient for repulsion loss.
        num_inner_steps: Number of gradient descent steps for inner optimization.
        lr_inner: Learning rate for the inner optimization.

    Returns:
        h_opt: Optimized h (Tensor of shape (batch_size, input_dim)).
    """

    x_t_h = x_t_minus.clone().detach()
    g = g_val.clone().detach()
    if args.strict_guidance:
        g_targ = g_w.clone().detach()
    else:
        g_targ = torch.randn((1, g.shape[1]), device=g.device)
        
    if "ablation" in args.method and "repulsion" in args.ablation:
        # No repulsion term
        # h_tilde = g + (scaled noise)
        gtarg_scaled = adaptive_scale_delta_vect(
                g, g_targ, grads, gamma=args.gamma_scale_delta
            )
        return g.detach() + gtarg_scaled, [torch.tensor(0.0), torch.tensor(0.0)]
        
    if "ablation" in args.method and "diversity" in args.ablation:
        # diversity is not enforced (no noise -and- no repulsion)
        # h_tilde = g
        return g.detach(), [torch.tensor(0.0), torch.tensor(0.0)]
    
    # Initialize h
    if not args.free_initial_h:
        h = g_val.clone().detach().requires_grad_() # initialize at g
    else:
        h = torch.zeros_like(g, requires_grad=False) + 1e-6  # or as a free parameter
    h = h.requires_grad_()

    optimizer_inner = optim.Adam([h], lr=lr_inner)

    for step in range(num_inner_steps):

        if "ablation" in args.method and "noise" in args.ablation:
            gtarg_scaled = 0.0
        else:
            gtarg_scaled = adaptive_scale_delta_vect(
                h, g_targ, grads, gamma=args.gamma_scale_delta
            )

        # Alignment term: maximize <g, h>
        # To maximize L, we minimize -L:
        alignment = -torch.mean(torch.sum(g * h, dim=-1))
        # Update points:
        x_t_h = x_t_h - eta * (h + gtarg_scaled)

        if args.need_repair:
            # x_t_h.data = torch.clamp(x_t_h.data, min=args.bounds[0], max=args.bounds[1])
            x_t_h.data = repair_bounds(
                x_t_h.data, args.bounds[0], args.bounds[1], args
            )

        # Map the updated points to the objective space
        F_ = problem.evaluate(x_t_h)

        # Compute repulsion loss to encourage diversity
        if use_sigma:
            rep_loss = repulsion_loss(F_, sigma)
        else:
            rep_loss = repulsion_loss(F_, use_sigma=False)

        # Our composite objective L is:
        loss = alignment + lambda_rep * rep_loss

        optimizer_inner.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_inner.step()

    h = h + gtarg_scaled  # This is h_tilde in the paper

    return h.detach(), [alignment.detach(), rep_loss.detach()]


def get_value_from_json(file_path, key):
    """
    Reads a JSON file from file_path and returns the value for the specified key.

    Parameters:
        file_path (str): The path to the JSON file.
        key (str): The key whose value you want to retrieve.

    Returns:
        The value corresponding to the key if found, otherwise None.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        # Return the value or None if the key does not exist.
        return data.get(key)
    except Exception as e:
        print(f"Error reading the JSON file: {e}")
        return None


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


def repair_bounds(x, xl, xu, args):
    """
    Clips a tensor x of shape [N, d] such that for each column j:
        x[:, j] is clipped to be between xl[j] and xu[j].

    Parameters:
        x (torch.Tensor): Input tensor of shape [N, d].
        xl (list): A list of d lower bound values.
        xu (list): A list of d upper bound values.
        args (Namespace): Arguments containing global_clamping flag.

    Returns:
        torch.Tensor: The clipped tensor with the same shape as x.
    """

    lower = xl.detach().clone().to(x.device)
    upper = xu.detach().clone().to(x.device)

    if args.global_clamping:
        return torch.clamp(x.data.clone(), min=lower.min(), max=upper.max())
    else:
        return torch.clamp(x.data.clone(), min=lower, max=upper)


def plot_pareto_front(list_fi, t, args, extra = None, lab = None):
    
    name = (
        args.method
        + "_"
        + args.problem
        + "_"
        + f"T={args.timesteps}"
        + "_"
        + f"N={args.num_points_sample}"
    )
    if args.label is not None:
        name += f"_{args.label}"

    if len(list_fi) > 3:
        return None

    elif len(list_fi) == 2:
        if extra is not None:
            f1, f2 = extra
            plt.scatter(f1, f2, c="red", s = 5)
            
        f1, f2 = list_fi
        plt.scatter(f1, f2, c="blue", s = 10)
        
        plt.xlabel("$f_1$", fontsize=14)
        plt.ylabel("$f_2$", fontsize=14)
        plt.title(f"Reverse Time Step: {t}", fontsize=14)
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
        ax.set_title(f"Reverse Time Step: {t}", fontsize=14)
        # ax.legend()

    if not os.path.exists(f"images_ms/{args.method}/{args.problem}"):
        os.makedirs(f"images_ms/{args.method}/{args.problem}")

    if lab is None:
        plt.savefig(
            f"images_ms/{args.method}/{args.problem}/{name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"images_ms/{args.method}/{args.problem}/{name}_{lab}.jpg",
            dpi=300,
            bbox_inches="tight",
        )

    # plt.show()
    plt.close()
    
    
######################## Training data generation #########################
def generate_initial_samples(problem, n_sample):
    """
    Sample points, using LHS, based on lowest constraint violation
    """
    sampler = LHS()
    # Problem bounds
    xl, xu = problem.xl, problem.xu
    # Draw n_sample candidates in [0,1]^n_var
    Xcand = sampler.do(problem, n_sample).get("X")
    # Scale to actual bounds
    Xcand = xl + (xu - xl) * Xcand
    F = problem.evaluate(Xcand)
    return Xcand, F

 
def random_perturbation(data, perturb_scale=0.05):
    perturb = (2 * np.random.rand(*data.shape) - 1) * perturb_scale
    return data + perturb

def random_perturbation(data, perturb_scale=0.05):
    perturb = (2 * np.random.rand(*data.shape) - 1) * perturb_scale
    return data + perturb


def interpolation(data, num_samples):
    interpolated_samples = []
    n = data.shape[0]
    for _ in range(num_samples):
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        alpha = np.random.rand()
        interpolated_sample = alpha * data[idx1] + (1 - alpha) * data[idx2]
        interpolated_samples.append(interpolated_sample)
    return np.array(interpolated_samples)

def interpolation_vec(data, num_samples):
    """
    Vectorized interpolation: for each of num_samples,
    pick two random rows (without self-pairs) and mix them.
    """
    n, dim = data.shape
    rng = np.random.default_rng()

    # 1) Sample two indices per sample
    #    (this may occasionally pick the same index twice; see note)
    idx = rng.integers(0, n, size=(num_samples, 2))
    
    # 2) If you really must avoid idx1 == idx2, re-draw just those:
    mask = idx[:,0] == idx[:,1]
    while np.any(mask):
        idx[mask, 1] = rng.integers(0, n, size=mask.sum())
        mask = idx[:,0] == idx[:,1]

    # 3) Gather the two sets of points
    pts1 = data[idx[:,0]]         # shape = (num_samples, dim)
    pts2 = data[idx[:,1]]         # shape = (num_samples, dim)

    # 4) Sample all alphas in one go
    alpha = rng.random(size=(num_samples, 1))

    # 5) Interpolate
    return alpha * pts1 + (1 - alpha) * pts2

def gaussian_noise(data, noise_scale=0.05):
    noise = np.random.normal(0, noise_scale, data.shape)
    return data + noise

def data_enhancement(offspringA, augmented_needed=90):
    """
    offspringA: np.array of shape [M, d]
    augmented_needed: number of augmented samples to generate
    """

    print(f"Generating {augmented_needed} augmented samples...")
    # generate candidates
    perturbed    = random_perturbation(offspringA)
    print(f"Perturbed: {perturbed.shape}")
    interpolated = interpolation_vec(offspringA, augmented_needed // 3)
    print(f"Interpolated: {interpolated.shape}")
    noised       = gaussian_noise(offspringA)
    print(f"Noised: {noised.shape}")

    all_aug = np.vstack([perturbed, interpolated, noised])
    np.random.shuffle(all_aug)

    # slice down to the capped number
    all_aug = all_aug[:augmented_needed, :]
    print(f"All augmented: {all_aug.shape}")
    # stack originals + capped augmented
    return np.vstack([offspringA, all_aug])

