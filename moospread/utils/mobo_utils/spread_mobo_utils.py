import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import math
from pymoo.indicators.hv import HV
from moospread.utils.mobo_utils.learning.prediction import *
import os

###########################################################
### DDPM training
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

def gaussian_noise(data, noise_scale=0.05):
    noise = np.random.normal(0, noise_scale, data.shape)
    return data + noise

def data_enhancement(offspringA, augmentation_factor=2, max_generation=32):
    """
    offspringA: np.array of shape [M, d]
    augmentation_factor: desired final size = M * augmentation_factor
    max_generation: upper‐limit on number of NEW (=augmented) samples
    """
    M = offspringA.shape[0]

    # how many augmented samples we'd like
    want = M * (augmentation_factor - 1)
    # but never more than max_generation
    augmented_needed = min(want, max_generation)

    # -- now proceed exactly as before, using augmented_needed --
    # generate candidates
    perturbed    = random_perturbation(offspringA)
    interpolated = interpolation(offspringA, augmented_needed // 3)
    noised       = gaussian_noise(offspringA)

    all_aug = np.vstack([perturbed, interpolated, noised])
    np.random.shuffle(all_aug)

    # slice down to the capped number
    all_aug = all_aug[:augmented_needed, :]

    # stack originals + capped augmented
    return np.vstack([offspringA, all_aug])

def mobo_get_ddpm_dataloader(Parent, 
                             objective_functions,
                             device,
                             batch_size,
                             validation_split=0.1):
    
    rows_to_take = int(1 / 3 * Parent.shape[0])
    pop = Parent[:rows_to_take, :]
    if len(pop) % 2 == 1:
        pop = pop[:-1]
    
    augmentation_factor = 10
    augmented_pop = data_enhancement(
            pop,
            augmentation_factor=augmentation_factor
        )

    dataset = torch.tensor(augmented_pop).float().to(device)
    dataset_size = dataset.shape[0]
    
    ##### TRAINING #####
    if validation_split > 0.0:
        # Split dataset: 10% for evaluation, 90% for training
        total_size = len(dataset)
        eval_size = int(0.10 * total_size)
        train_size = total_size - eval_size

        dataset = dataset[torch.randperm(dataset.size(0))]
        train_dataset = dataset[:train_size]
        eval_dataset = dataset[train_size:]
        
        y_train = objective_functions(train_dataset)
        y_val = objective_functions(eval_dataset)
        dataset_train = torch.utils.data.TensorDataset(train_dataset, y_train)
        dataset_val = torch.utils.data.TensorDataset(eval_dataset, y_val)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
        )
        eval_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
        )
        
    else:
        # Use the entire dataset for training
        y_train = objective_functions(dataset)
        dataset_train = torch.utils.data.TensorDataset(dataset, y_train)
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True 
        )
        eval_loader = None
    return train_loader, eval_loader, dataset_size
        
###########################################################

def get_max_hv_re(problem_name: str,
                  front_dir: str,
                  ref_point: np.ndarray = None,
                  n_pareto_points: int = None) -> float:
    """
    Compute the maximum hypervolume of the approximated Pareto front
    for a given RE problem (RE21 - RE27) from the RE-problems data.

    Args:
        problem_name: one of "re21", "re22", ..., "re27".
        front_dir: path to the 'approximated_Pareto_fronts' folder cloned from
                   https://github.com/ryojitanabe/reproblems :contentReference[oaicite:2]{index=2}.
        ref_point: optional array of shape (n_obj,) dominating the front.
                   If None, it is set to 10% above the per‐objective maximum.
        n_pareto_points: ignored (front files are precomputed).

    Returns:
        max_hv: the hypervolume (float) of the true front under `ref_point`.
    """
    file_key = problem_name.upper()
    
    # 2. Build full path to data file
    fname = f"reference_points_{file_key}.dat"
    path = os.path.join(front_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pareto front file not found: {path}")

    # 3. Load the front (shape: [N, n_obj])
    front = np.loadtxt(path)  # whitespace‐delimited numeric data :contentReference[oaicite:4]{index=4}

    # 4. Determine reference point if not provided
    if ref_point is None:
        # 10% margin beyond the worst‐case nadir
        ref_point = np.max(front, axis=0) * 1.1

    # 5. Compute hypervolume via Pymoo’s HV indicator
    hv_indicator = HV(ref_point=ref_point)
    max_hv = hv_indicator.calc(front)  # yields a scalar :contentReference[oaicite:5]{index=5}

    return max_hv


def get_max_hv_pymoo(problem, n_pareto_points=100, ref_point=None):
    """
    Compute the maximum hypervolume of the true Pareto front for a Pymoo problem.

    Args:
        problem: An instance of a Pymoo Problem with a known Pareto front.
        n_pareto_points: Number of points to sample on the Pareto front.
        ref_point: Optional NumPy array of shape (n_obj,) defining the reference point.
                   If None, uses max(front) * 1.1 per objective.

    Returns:
        max_hv: Float, the hypervolume of the Pareto front w.r.t. ref_point.
    """
    # 1. Retrieve Pareto front
    if hasattr(problem, '_calc_pareto_front'):
        front = problem._calc_pareto_front(n_pareto_points)
    else:
        raise AttributeError("Problem does not implement _calc_pareto_front")

    # 2. Determine reference point
    if ref_point is None:
        # take 10% above the maximum of the front
        ref_point = np.max(front, axis=0) * 1.1

    # 3. Compute hypervolume
    hv_indicator = HV(ref_point=ref_point)
    max_hv = hv_indicator.calc(front)

    return max_hv


def objective_functions_GP(points, surrogate_model, coef_lcb, device, get_grad = False):
    x = points.detach().cpu().numpy()
    eval_result = surrogate_model.evaluate(x, std=True, calc_gradient=get_grad)
    if get_grad:
        mean = torch.from_numpy(eval_result["F"]).float().to(device)
        mean_grad = torch.from_numpy(eval_result["dF"]).float().to(device)
        std = torch.from_numpy(eval_result["S"]).float().to(device)
        std_grad = torch.from_numpy(eval_result["dS"]).float().to(device)
        
        Y_val = mean - coef_lcb * std
        Grad_val = mean_grad - coef_lcb * std_grad

        return list(torch.split(Y_val, 1, dim=1)), [Grad_val[:, i, :] for i in range(Grad_val.shape[1])]
    else:
        mean = torch.from_numpy(eval_result["F"]).float().to(device)
        std = torch.from_numpy(eval_result["S"]).float().to(device)
        Y_val = mean - coef_lcb * std
        
        return list(torch.split(Y_val, 1, dim=1))

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

def repair_bounds(x, lower, upper, args = None):
    """
    Clips a tensor x of shape [N, d] such that for each column j:
        x[:, j] is clipped to be between xl[j] and xu[j].
    """
    return torch.clamp(x.data.clone(), min=lower, max=upper)

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
    if alpha.sum() == 0:
        # If all alpha became 0 (numerical issues), fall back to equal weights
        alpha = torch.ones(m, device=alpha.device) / m
    else:
        alpha = alpha / alpha.sum()

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


def select_top_n_for_BaySpread(
    pop,                    # population to select from (tensor)
    net,                    # neural network for dominance prediction
    device,                 
    surrogate_model,        # surrogate model with .evaluate()
    coef_lcb: float,
    n: int,
    top_frac: float = 0.9
):
    """
    1) Predict pairwise dominance via NN.
    2) Identify non‑dominated indices.
       - If > n: compute crowding distance for those,
         pick the top n by distance, shuffle, and return.
       - Else: do the 'top_frac% by rank + crowding fill' to get up to n.
    """

    N = pop.shape[0]

    # 1) Predict dominance
    label_matrix, conf_matrix = nn_predict_dom_intra(pop.detach().cpu().numpy(), 
                                                     net, 
                                                     device)

    # 2) Find non‑dominated indices
    nondom_inds = [
        i for i in range(N)
        if not any(label_matrix[j, i] == 2 for j in range(N))
    ]

    # --- CASE A: too many non‑dominated → pick top-n by crowding ---
    if len(nondom_inds) > n:
        # Evaluate objectives on just the non-dominated set
        # pts_nd = torch.from_numpy(pop[nondom_inds]).float().to(device)
        pts_nd = pop[nondom_inds].to(device)
        Y_t  = torch.cat(
            objective_functions_GP(pts_nd, surrogate_model, coef_lcb, device),
            dim=1
        )

        # Compute crowding distances and select top-n
        distances = crowding_distance(Y_t)
        topk      = torch.topk(distances, n).indices.tolist()

        selected_nd = [nondom_inds[i] for i in topk]

        # Shuffle before returning
        perm = torch.randperm(n, device=pop.device)
        final_idx = torch.tensor(selected_nd, device=pop.device)[perm]
        return pop[final_idx] #.detach().cpu().numpy()

    # --- CASE B: nondom ≤ n → fill up via rank + top_frac% + crowding ---
    # 3) Compute dom counts & avg confidence for all
    dom_counts = []
    avg_conf   = []
    for i in range(N):
        dom_by = (label_matrix[:, i] == 2)
        cnt    = int(dom_by.sum())
        dom_counts.append(cnt)
        avg_conf.append(
            float(conf_matrix[dom_by, i].sum()) / cnt
            if cnt > 0 else 0.0
        )

    # 4) Sort full pop by (dom_count asc, avg_conf desc)
    idxs = list(range(N))
    idxs.sort(key=lambda i: (dom_counts[i], -avg_conf[i]))

    # 5) Keep only top top_frac% of that ranking
    k90   = int(np.floor(top_frac * N))
    top90 = idxs[:k90]

    # 6) Evaluate
    pts90 = pop[top90]
    Y_t  = torch.cat(
        objective_functions_GP(pts90, surrogate_model, coef_lcb, device),
        dim=1
    )

    # 7) Crowding distance & pick as many as needed to reach n
    distances = crowding_distance(Y_t)
    need      = n - len(nondom_inds)
    need      = max(need, 0)
    k_sel     = min(need, len(top90))
    sel90     = torch.topk(distances, k_sel).indices.tolist()
    selected_from_top_frac = [ top90[i] for i in sel90 ]

    # 8) Build final list: all nondom + selected_from_top_frac
    final_inds = nondom_inds + selected_from_top_frac

    # 9) If still short (e.g. N<n), pad with best remaining in idxs
    if len(final_inds) < n:
        remaining = [i for i in idxs if i not in final_inds]
        to_add    = n - len(final_inds)
        final_inds += remaining[:to_add]

    # 10) Shuffle final indices
    perm = torch.randperm(len(final_inds), device=pop.device)
    final_idx = torch.tensor(final_inds, device=pop.device)[perm]

    return pop[final_idx] #.detach().cpu().numpy()

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

def mgd_armijo_step(
    x_t: torch.Tensor,
    d: torch.Tensor,
    f_old,
    grads, # (N, m, d)
    objective_functions_GP,
    surrogate_model,
    args,
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
        f_new = torch.cat(objective_functions_GP(
            trial_x, surrogate_model, args.coef_lcb, args.device
        ), dim=1)

        # Armijo test (vectorised over objectives)
        # f_new <= f_old + c1 * eta * grad_dot  (element-wise)
        ok = (f_new <= f_old[improve] + c1 * eta[improve, None] * grad_dot[improve]).all(dim=1)

        # Update masks and step sizes
        eta[improve] = torch.where(ok, eta[improve], rho * eta[improve])
        improve_mask = improve.clone()
        improve[improve_mask] = ~ok

    return eta[:, None]


def adaptive_scale_delta_vect(
    g: torch.Tensor, delta_raw: torch.Tensor, grads: torch.Tensor, gamma: float = 0.9
) -> torch.Tensor:
    """
    Adaptive scaling to preserve *positivity*:

      ∇f_j(x_i)^T [ g_i + rho_i * delta_raw_i ] > 0  for all j.

    Args:
        g (torch.Tensor):         [n_points, d], the multi-objective "gradient"
                                  (which we *subtract* in the update).
        delta_raw (torch.Tensor): [n_points, d], the unscaled diversity/repulsion direction.
        grads (torch.Tensor):     [m, n_points, d], storing ∇f_j(x_i).
        gamma (float):            Safety factor in (0,1).

    Returns:
        delta_scaled (torch.Tensor): [n_points, d], scaled directions s.t.
        for all j:  grads[j,i]ᵀ [g[i] + delta_scaled[i]] > 0.
    """
    # n_points, d = g.shape
    # m = grads.shape[0]  # number of objectives

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
    f,
    surrogate_model,
    g_val,
    grads,
    args,
    eta,
    lambda_rep,
    sigma=1.0,
    use_sigma=True,
    num_inner_steps=10,
    lr_inner=1e-2,
):
    """
    For a given batch x_t and its corresponding g(x_t), solve for h by
    minimizing -L (i.e., maximizing L) via gradient descent.
    """

    x_t_h = x_t_minus.clone().detach()
    g = g_val.clone().detach()
    g_targ = torch.randn((1, g.shape[1]), device=g.device)

    # Initialize h
    h = torch.zeros_like(g, requires_grad=False) + 1e-6  # as a free parameter
    h = h.requires_grad_()

    optimizer_inner = optim.Adam([h], lr=lr_inner)

    for step in range(num_inner_steps):
        gtarg_scaled = adaptive_scale_delta_vect(
            h, g_targ, grads, gamma=args.gamma_scale_delta
        )

        # Alignment term: maximize <g, h>
        # To maximize L, we minimize -L:
        alignment = -torch.mean(torch.sum(g * h, dim=-1))
        # Update points:
        x_t_h = x_t_h - eta * (h + gtarg_scaled)

        x_t_h.data = repair_bounds(
                x_t_h.data, args.bounds[0], args.bounds[1]
            )

        # Map the updated points to the objective space
        F_vals = f(
            x_t_h, surrogate_model, args.coef_lcb, args.device
        )
        F_ = torch.cat([f_i.unsqueeze(1) for f_i in F_vals], dim=1)

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

    h = h + gtarg_scaled  # This is h*(x_t) in the paper

    return h.detach(), [alignment.detach(), rep_loss.detach()]


def select_top_n_candidates_for_bayMS(
    points: torch.Tensor,
    n,
    f,
    coef_lcb,
    style="crowding",
) -> torch.Tensor:
    """
    Selects the top `n` points from `points` based on a given style.

    Args:
        points (torch.Tensor): Candidate solutions (shape [N, D]).
        n: number of points to select
        f: objective functions (Sorogate models)
        coef_lcb: coefficient of LCB
        style: for selecting points based on a given style (E.g. 'crowding' for crowding distance).

    Returns:
        torch.Tensor: The best subset of points (shape [n, D]).
    """

    if style == "crowding":
        Y_candidate_mean = f.evaluate(points.detach().cpu().numpy())["F"]
        Y_candidata_std = f.evaluate(points.detach().cpu().numpy(), std=True)["S"]
        rows_with_nan = np.any(np.isnan(Y_candidate_mean), axis=1)
        Y_candidate_mean = Y_candidate_mean[~rows_with_nan]
        Y_candidata_std = Y_candidata_std[~rows_with_nan]
        Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
        full_p_front = torch.from_numpy(Y_candidate).float()
        # distances = crowding_distance(points)
        distances = crowding_distance(full_p_front)
        top_indices = torch.topk(distances, n).indices
        return points[top_indices]

    else:
        raise ValueError(f"Unknown style: {style}")

def sbx(sorted_pop, eta=15):
    n_pop, n_var = sorted_pop.shape
    new_pop = np.empty_like(sorted_pop)

    for i in range(0, n_pop, 2):
        parent1, parent2 = (
            sorted_pop[np.random.choice(n_pop)],
            sorted_pop[np.random.choice(n_pop)],
        )
        rand = np.random.random(n_var)
        gamma = np.empty_like(rand)
        mask = rand <= 0.5
        gamma[mask] = (2 * rand[mask]) ** (1 / (eta + 1))
        gamma[~mask] = (1 / (2 * (1 - rand[~mask]))) ** (1 / (eta + 1))

        offspring1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
        offspring2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)

        new_pop[i] = offspring1
        if i + 1 < n_pop:
            new_pop[i + 1] = offspring2

    return new_pop


def environment_selection(population, n):
    """
    environmental selection in SPEA-2
    :param population: current population
    :param n: number of selected individuals
    :return: next generation population
    """
    fitness = cal_fit(population)
    index = np.nonzero(fitness < 1)[0]
    if len(index) < n:
        rank = np.argsort(fitness)
        index = rank[:n]
    elif len(index) > n:
        del_no = trunc(population[index, :], len(index) - n)
        index = np.setdiff1d(index, index[del_no])

    population = population[index, :]
    return population, index


def trunc(pop_obj, k):
    n, m = np.shape(pop_obj)
    distance = cdist(pop_obj, pop_obj)
    distance[np.eye(n) > 0] = np.inf
    del_no = np.ones(n) < 0
    while np.sum(del_no) < k:
        remain = np.nonzero(np.logical_not(del_no))[0]
        temp = np.sort(distance[remain, :][:, remain], axis=1)
        rank = np.argsort(temp[:, 0])
        del_no[remain[rank[0]]] = True
    return del_no


def cal_fit(pop_obj):
    n, m = np.shape(pop_obj)
    dominance = np.ones((n, n)) < 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            k = int(np.any(pop_obj[i, :] < pop_obj[j, :])) - int(
                np.any(pop_obj[i, :] > pop_obj[j, :])
            )
            if k == 1:
                dominance[i, j] = True
            elif k == -1:
                dominance[j, i] = True

    s = np.sum(dominance, axis=1, keepdims=True)

    r = np.zeros(n)
    for i in range(n):
        r[i] = np.sum(s[dominance[:, i]])

    distance = cdist(pop_obj, pop_obj)
    distance[np.eye(n) > 0] = np.inf
    distance = np.sort(distance, axis=1)
    d = 1 / (distance[:, int(np.sqrt(n))] + 2)

    fitness = r + d
    return fitness


def pm_mutation(pop_dec, boundary):

    pro_m = 1
    dis_m = 20
    pop_dec = pop_dec[: (len(pop_dec) // 2) * 2, :]
    n, d = np.shape(pop_dec)

    site = np.random.random((n, d)) < pro_m / d
    mu = np.random.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.tile(boundary[0], (n, 1)), np.tile(boundary[1], (n, 1))
    pop_dec = np.minimum(np.maximum(pop_dec, lower), upper)
    norm = (pop_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
    pop_dec[temp] += (upper[temp] - lower[temp]) * (
        np.power(
            2.0 * mu[temp] + (1.0 - 2.0 * mu[temp]) * np.power(1.0 - norm, dis_m + 1.0),
            1.0 / (dis_m + 1),
        )
        - 1.0
    )
    temp = site & (mu > 0.5)
    norm = (upper[temp] - pop_dec[temp]) / (upper[temp] - lower[temp])
    pop_dec[temp] += (upper[temp] - lower[temp]) * (
        1.0
        - np.power(
            2.0 * (1.0 - mu[temp])
            + 2.0 * (mu[temp] - 0.5) * np.power(1.0 - norm, dis_m + 1.0),
            1.0 / (dis_m + 1.0),
        )
    )
    offspring_dec = np.maximum(np.minimum(pop_dec, upper), lower)
    return offspring_dec


def sort_population(pop, label_matrix, conf_matrix):
    size = len(pop)
    domination_counts = []
    avg_confidences = []
    for i in range(size):
        count = sum(label_matrix[j, i] == 2 for j in range(size))
        domination_counts.append(count)
        confidence = sum(
            conf_matrix[j, i] for j in range(size) if label_matrix[j, i] == 2
        )
        avg_confidences.append(confidence / (count if count > 0 else 1))

    sorted_pop = sorted(
        zip(pop, domination_counts, avg_confidences),
        key=lambda x: (x[1], -x[2]),
    )
    sorted_pop = [x[0] for x in sorted_pop]

    sorted_pop_array = np.array(sorted_pop)

    return sorted_pop_array


def convert_seconds(seconds):
    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    # Format the result
    print(f"Time: {hours} hours {minutes} minutes {remaining_seconds} seconds")

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