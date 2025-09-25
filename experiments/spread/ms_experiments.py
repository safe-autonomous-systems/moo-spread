import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

import math
import json
import os
import pickle
from tqdm import tqdm
import datetime
from time import time, sleep

from ms_nets import DiTMOO # TransformerDiffusionModel

from all_funcs_utils import *

from ms_args import get_problem_by_name

def train_spread(problem, args):
    #  Set the seed
    set_seed(args.seed)

    model = DiTMOO(args.input_dim,
                   args.num_obj,
                   num_blocks=args.num_blocks)
    model.to(args.device)

    # Sample initial data points from the decision space
    X, y = generate_initial_samples(problem, args.ddpm_training_size)
    assert 0 <= args.ddpm_validation_rate < 1, "ddpm_validation_rate should be in [0, 1)"

    data_size = int(X.shape[0] - int(args.ddpm_training_size * args.ddpm_validation_rate))
    X_val = X[data_size:]
    y_val = y[data_size:]
    X = X[:data_size]
    y = y[:data_size]

    tensor_x = X.float() 
    tensor_y = y.float() 
    tensor_x_val = X_val.float() 
    tensor_y_val = y_val.float() 

    dataset_train = TensorDataset(tensor_x, tensor_y)
    dataset_val = TensorDataset(tensor_x_val, tensor_y_val)
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # Initialize model, optimizer, and scheduler
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Print the number of learnable parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {total_params}")

    betas = cosine_beta_schedule(args.timesteps)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0).to(args.device)

    LOSSES = []
    print(f"DDPM Training for {args.num_epochs} epochs...")
    DDPM_FILE_BEST = str("%s/checkpoint_ddpm_best.pth" % (args.model_dir))
    DDPM_FILE_LAST = str("%s/checkpoint_ddpm_last.pth" % (args.model_dir))
    t0_epch_f = time()

    best_val_loss = np.inf
    tol_violation_epoch = args.ddpm_training_tol
    cur_violation_epoch = 0

    with tqdm(
        total=args.num_epochs,
        desc=f"Training ",
        unit="epoch",
    ) as pbar:

        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0

            # for batch in dataloader:
            for indx_batch, (batch, obj_values) in enumerate(train_dataloader):
                optimizer.zero_grad()

                # Extract batch points
                points = batch.to(args.device)
                points.requires_grad = True  # Enable gradients if needed
                obj_values = obj_values.to(args.device)

                # Sample random timesteps for each data point
                t = torch.randint(0, args.timesteps, (points.shape[0],)).to(args.device)
                alpha_bar_t = alpha_cumprod[t].unsqueeze(1)  # shape: [batch_size, 1]

                # Forward process: Add noise to points
                noise = torch.randn_like(points).to(
                    args.device
                )  # shape: [batch_size, input_dim]
                x_t = (
                    torch.sqrt(alpha_bar_t) * points
                    + torch.sqrt(1 - alpha_bar_t) * noise
                )

                # Conditioning information
                c = obj_values
                Delta_c = c[c > 0].min() if (c > 0).any() else 1e-6
                c = c + Delta_c
                # Model predicts noise
                predicted_noise = model(
                    x_t, t.float() / args.timesteps, c
                )

                # Compute loss
                ## DDPM loss
                loss_simple = l_simple_loss(predicted_noise, noise)

                # Combine losses
                loss_simple.backward()
                optimizer.step()
                epoch_loss += loss_simple.item()

            epoch_loss = epoch_loss / args.batch_size
            LOSSES.append(epoch_loss)

            # Validation
            # for batch in dataloader:
            model.eval()
            val_loss = 0.0
            for indx_batch, (val_batch, val_obj_values) in enumerate(val_dataloader):
                # Extract batch points
                val_points = val_batch.to(
                    args.device
                )  # TensorDataset wraps data in a tuple
                val_points.requires_grad = True  # Enable gradients if needed
                val_obj_values = val_obj_values.to(args.device)

                # Sample random timesteps for each data point
                t = torch.randint(0, args.timesteps, (args.batch_size,)).to(args.device)
                alpha_bar_t = alpha_cumprod[t].unsqueeze(1)  # shape: [batch_size, 1]

                # Forward process: Add noise to points
                val_noise = torch.randn_like(val_points).to(
                    args.device
                )  # shape: [batch_size, input_dim]
                val_x_t = (
                    torch.sqrt(alpha_bar_t) * val_points
                    + torch.sqrt(1 - alpha_bar_t) * val_noise
                )

                # Conditioning information
                c = val_obj_values
                Delta_c = c[c > 0].min() if (c > 0).any() else 1e-6
                c = c + Delta_c
                # Model predicts noise
                val_predicted_noise = model(
                    val_x_t, t.float() / args.timesteps, c
                )
                loss_simple = l_simple_loss(val_predicted_noise, val_noise)

                val_loss += loss_simple.item()

            val_loss = val_loss / args.batch_size

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                cur_violation_epoch = 0
                # Save the model
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "losses": LOSSES,
                    "args": args,
                }
                torch.save(checkpoint, DDPM_FILE_BEST)

            else:
                cur_violation_epoch += 1
                if cur_violation_epoch >= tol_violation_epoch:
                    print(f"Early Stopping at epoch {epoch + 1}.")
                    break

            pbar.set_postfix({"val_loss": val_loss})
            pbar.update(1)

    T_epch_f = time() - t0_epch_f
    convert_seconds(T_epch_f)
    print(datetime.datetime.now())
    print("END training !")

    checkpoint_time = {
        "train_time": T_epch_f,
        "args": args,
    }
    TRAIN_FILE = str("%s/train_infos.pth" % (args.model_dir))
    torch.save(checkpoint_time, TRAIN_FILE)
    print("Checkpoint saved at: ", TRAIN_FILE)
    if args.num_epochs > 0:
        # Save the model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "losses": LOSSES,
            "args": args,
        }
        torch.save(checkpoint, DDPM_FILE_LAST)
        print("Final model checkpoint saved at: ", DDPM_FILE_LAST)
    else:
        # load the model
        checkpoint = torch.load(DDPM_FILE_BEST, map_location=args.device)
        LOSSES = checkpoint["losses"]
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded from: ", DDPM_FILE_BEST)


def one_sampling_step_spread(
    problem, model, x_t, t, 
    ########################
    point_n0, optimizer_n0, 
    #########################
    beta_t, alpha_bar_t, args
):

    # Create a tensor of timesteps with shape (num_points_sample, 1)
    t_tensor = torch.full(
        (args.num_points_sample,),
        t,
        device=args.device,
        dtype=torch.float32,
    )
    # Compute objective values
    obj_values = problem.evaluate(x_t)
   
    # Conditioning information
    c = obj_values
    with torch.no_grad():
        predicted_noise = model(x_t, t_tensor / args.timesteps, 
                                c)
    
    g_w = None
    if args.strict_guidance: # NOT USED IN THE EXPERIMENTS
        #### Get new perturbation
        if args.num_obj == 2:
            f1_prev, f2_prev = problem.evaluate(point_n0).split(1, dim=1)
            f1_prev.sum().backward(retain_graph=True)
            grad_1 = point_n0.grad.detach().clone()
            point_n0.grad.zero_()
            f2_prev.sum().backward(retain_graph=True)
            grad_2 = point_n0.grad.detach().clone()
            point_n0.grad.zero_()
            grad_1 = torch.nn.functional.normalize(grad_1, dim=0)
            grad_2 = torch.nn.functional.normalize(grad_2, dim=0)
            optimizer_n0.zero_grad()
            g_w = solve_min_norm_2_loss(grad_1, grad_2)
            point_n0.grad = g_w
            optimizer_n0.step()
        else:
            list_fi_n0 = problem.evaluate(point_n0).split(1, dim=1)
            list_grad_i_n0 = []
            for fi_n0 in list_fi_n0:
                fi_n0.sum().backward(retain_graph=True)
                grad_i_n0 = point_n0.grad.detach().clone()
                point_n0.grad.zero_()
                grad_i_n0 = torch.nn.functional.normalize(grad_i_n0, dim=0)
                list_grad_i_n0.append(grad_i_n0)
            optimizer_n0.zero_grad()
            g_w = get_mgd_grad(list_grad_i_n0)
            point_n0.grad = g_w
            optimizer_n0.step()

    #### Reverse diffusion step
    sqrt_1_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-6))
    sqrt_1_minus_beta_t = torch.sqrt(torch.clamp(1 - beta_t, min=1e-6))
    mean = (1 / sqrt_1_minus_beta_t) * (
        x_t - (beta_t / sqrt_1_minus_alpha_t) * (predicted_noise)
    )
    if torch.isnan(mean).any():
        print("Nan values in mean !")
    std_dev = torch.sqrt(beta_t)
    z = torch.randn_like(x_t) if t > 0 else 0.0  # No noise for the final step
    x_t = mean + std_dev * z

    #### Pareto Guidance step
    if args.num_obj == 2:
        if args.need_repair:
            x_t.data = repair_bounds(x_t.data.clone(), args.bounds[0], args.bounds[1], args)
        X = x_t.clone().detach().requires_grad_()
   
        f1, f2 = problem.evaluate(X).split(1, dim=1)
        f1.sum().backward(retain_graph=True)
        grad_f1 = X.grad.detach().clone()
  
        X.grad.zero_()
        f2.sum().backward(retain_graph=True)
        grad_f2 = X.grad.detach().clone()
      
        X.grad.zero_()
        grad_f1 = torch.nn.functional.normalize(grad_f1, dim=0)
        grad_f2 = torch.nn.functional.normalize(grad_f2, dim=0)
        if torch.isnan(grad_f1).any():
            print("Nan values in grad_f1 bsolveh !")
        if torch.isnan(grad_f2).any():
            print("Nan values in grad_f2 bsolveh !")
        grads = torch.stack([grad_f1, grad_f2], dim=0) # (m, N, d)
        grads_copy = torch.stack([grad_f1, grad_f2], dim=1).detach() # (N, m, d)
        g_x_t_minus = solve_min_norm_2_loss(grad_f1, grad_f2)
    else:
        if args.need_repair:
            x_t.data = repair_bounds(x_t.data.clone(), args.bounds[0], args.bounds[1], args)
        X = x_t.clone().detach().requires_grad_()
        list_fi = problem.evaluate(X).split(1, dim=1)
        list_grad_i = []
        for fi in list_fi:
            fi.sum().backward(retain_graph=True)
            grad_i = X.grad.detach().clone()
            grad_i = torch.nn.functional.normalize(grad_i, dim=0)
            list_grad_i.append(grad_i)
            X.grad.zero_()
        grads = torch.stack(list_grad_i, dim=0) # (m, N, d)
        grads_copy = torch.stack(list_grad_i, dim=1).detach() # (N, m, d)
        g_x_t_minus = get_mgd_grad(list_grad_i)

    if torch.isnan(x_t).any():
        print("Nan values in x_t bsolveh !")
        print("x_t: ", x_t)

    if torch.isnan(g_x_t_minus).any():
        print("Nan values in g_x_t_minus bsolveh !")
        print("g_x_t_minus: ", g_x_t_minus)

    eta = mgd_armijo_step(
        x_t,
        g_x_t_minus,
        obj_values,
        grads_copy,
        problem,
        eta_init=args.eta,
    )

    h_star, adRes = solve_for_h(
        x_t,  # x_t after the standard reverse diffusion step is x_t_prime
        t,
        problem,
        g_x_t_minus,
        grads,
        g_w,
        args,
        eta=eta,
        lambda_rep=args.lambda_rep,
        sigma=args.kernel_sigma,
        use_sigma=False,
        num_inner_steps=args.num_inner_steps,
        lr_inner=args.lr_inner,
    )

    h_star = torch.nan_to_num(h_star, nan=torch.nanmean(h_star), posinf=0.0, neginf=0.0)

    if torch.isnan(h_star).any():
        print("Nan values in h_star !")
        print("h_star: ", h_star)

    x_t = x_t - eta * h_star

    return x_t, [h_star, adRes]


def sampling(problem, args):
    # Set the seed
    set_seed(args.seed)

    print(f"Problem: {args.problem}")

    name = (
        args.method
        + "_"
        + args.problem
        + "_"
        + str(args.seed)
        + "_"
        + f"T={args.timesteps}"
        + "_"
        + f"N={args.num_points_sample}"
    )
    
    if args.label is not None:
        name += f"_{args.label}"

    if not (os.path.exists(args.samples_store_path)):
        os.makedirs(args.samples_store_path)

    best_hv = -np.inf
    best_hvi_round = -np.inf

    model = DiTMOO(args.input_dim,
                   args.num_obj,
                   num_blocks=args.num_blocks)
    model.to(args.device)
    DDPM_FILE = str("%s/checkpoint_ddpm_best.pth" % (args.model_dir))
    checkpoint = torch.load(DDPM_FILE, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    betas = cosine_beta_schedule(args.timesteps)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0).to(args.device)

    # Start from random points in the decision space
    x_t = torch.rand((args.num_points_sample, args.input_dim)) # in [0, 1]
    x_t = args.bounds[0] + (args.bounds[1] - args.bounds[0]) * x_t # scale to bounds
    x_t = x_t.to(args.device)
    x_t.requires_grad = True
    if args.need_repair:
        x_t.data = repair_bounds(
            x_t.data.clone(), args.bounds[0], args.bounds[1], args
        )

    prev_pf_points = None
    num_optimal_points = 0

    point_n0 = None
    optimizer_n0 = None
    ##############################
    if args.strict_guidance: # NOT USED IN THE EXPERIMENTS
        # Initialize 1 target point for direction perturbation in SPREAD
        point_n0 = torch.rand((1, args.input_dim)) # in [0, 1]
        point_n0 = args.bounds[0] + (args.bounds[1] - args.bounds[0]) * point_n0 # scale to bounds
        point_n0 = point_n0.to(args.device)
        point_n0.requires_grad = True
        if args.need_repair:
            point_n0.data = repair_bounds(
                point_n0.data.clone(), args.bounds[0], args.bounds[1], args
            )
        optimizer_n0 = optim.Adam([point_n0], lr=args.lr)
    ##############################
    
    tol_violation_timesteps = args.sampling_tol
    cur_violation_timesteps = 0

    print(f"START sampling {args.num_points_sample} points ...")
    t0_epch_f = time()

    model.eval()

    with tqdm(
        total=args.timesteps,
        desc=f"Sampling with SPREAD .. ",
        unit="t",
    ) as pbar:

        for t in reversed(range(args.timesteps)):
            x_t.requires_grad_(True)
            if args.need_repair:
                x_t.data = repair_bounds(
                    x_t.data.clone(), args.bounds[0], args.bounds[1], args
                )
            # Compute beta_t and alpha_t
            beta_t = 1 - alphas[t]
            alpha_bar_t = alpha_cumprod[t]

            x_t, _ = one_sampling_step_spread(
                problem,
                model,
                x_t,
                t,
                ################
                point_n0,
                optimizer_n0,
                ###############
                beta_t,
                alpha_bar_t,
                args,
            )

            ########################
            if args.strict_guidance: # NOT USED IN THE EXPERIMENTS
                if args.need_repair:
                    point_n0.data = repair_bounds(
                        point_n0.data.clone(), args.bounds[0], args.bounds[1], args
                    )
            ########################

            if args.need_repair:
                pf_population = repair_bounds(
                    copy.deepcopy(x_t.detach()), args.bounds[0], args.bounds[1], args
                )
            else:
                pf_population = copy.deepcopy(x_t.detach())

            pf_points, _, _ = get_non_dominated_points(
                pf_population,
                problem,
                keep_shape=False
            )
            
            if prev_pf_points is not None:
                non_dom_points = torch.cat((prev_pf_points, pf_points), dim=0)
                pf_points, _, _ = get_non_dominated_points(
                                non_dom_points,
                                problem,
                                keep_shape=False,
                            )
                if len(pf_points) > args.num_points_sample:
                    style = "crowding"
                    pf_points = select_top_n_candidates(
                                    pf_points,
                                    args.num_points_sample,
                                    problem=problem,
                                    ref_point=args.ref_point,
                                    style=style,
                                )
                        
            prev_pf_points = pf_points
            num_optimal_points = len(pf_points)

            if tol_violation_timesteps is not None:
                temp_pareto_set_x = copy.deepcopy(pf_points)
                temp_pareto_set_y = problem.evaluate(temp_pareto_set_x).detach().cpu().numpy()

                hv = HV(ref_point=args.ref_point)
                hv_value = hv(temp_pareto_set_y)

                if (
                    hv_value > best_hv
                ):
                    best_hv = hv_value

                if round(hv_value, 3) > best_hvi_round:
                    best_hvi_round = round(hv_value, 3)
                    cur_violation_timesteps = 0
                elif tol_violation_timesteps is not None:
                    cur_violation_timesteps += 1
                    if (
                        cur_violation_timesteps >= tol_violation_timesteps
                    ): 
                        print(f"Stopping at timestep {t} during denoising.")
                        break
                    
                pbar.set_postfix({
                    "HV": hv_value,  
                    "Points": num_optimal_points,
                    })
            else:
                pbar.set_postfix({
                    "Points": num_optimal_points,
                    })
                
            if args.num_obj <= 3:
                    if (t % 100 == 0) or (t ==  args.timesteps - 1):
                        list_fi = problem.evaluate(pf_points).split(1, dim=1)
                        list_fi = [fi.detach().cpu().numpy() for fi in list_fi]
                        pareto_front = problem.pareto_front()
                    
                        
                        plot_pareto_front(list_fi, t, args, 
                                        extra= [pareto_front[:, i] for i in range(args.num_obj)],
                                        lab="t="+str(t)+"_seed="+str(args.seed))

            x_t = x_t.detach()
            pbar.update(1)

    print(
        f"END sampling !"
    )

    T_epch_f = time() - t0_epch_f

    res_y = problem.evaluate(pf_points).detach().cpu().numpy()
    res_x = pf_points.detach().cpu().numpy()

    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]
    
    hv = HV(ref_point=args.ref_point)
    hv_value = hv(res_y)
    
    hv_results = {
        "ref_point": args.ref_point,
        "hypervolume": hv_value,
        "computation_time": T_epch_f
    }

    # Store the results
    if not (os.path.exists(args.samples_store_path)):
        os.makedirs(args.samples_store_path)
    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)
        
    np.save(args.samples_store_path + name + "_x.npy", res_x)
    np.save(args.samples_store_path + name + "_y.npy", res_y)
  
    with open(args.results_store_path + name + "_hv_results.pkl", "wb") as f:
        pickle.dump(hv_results, f)

    print(f"Hypervolume: {hv_value} for seed {args.seed}")
    print("---------------------------------------")

    # Print computation time
    convert_seconds(T_epch_f)
    print(datetime.datetime.now())

    return (
        res_x, 
        res_y
    )


def evaluation(problem, args):
    # Set the seed
    set_seed(args.seed)

    print(f"Problem: {args.problem}")

    name = (
        args.method
        + "_"
        + args.problem
        + "_"
        + str(args.seed)
        + "_"
        + f"T={args.timesteps}"
        + "_"
        + f"N={args.num_points_sample}"
    )
    
    res_y = np.load(args.samples_store_path + name + "_y.npy")
    print(f"Loaded the generated samples from {args.samples_store_path}")
    hv = HV(ref_point=args.ref_point)
    hv_value = hv(res_y)

    hv_results = {
        "ref_point": args.ref_point,
        "hypervolume": hv_value,
    }


    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.pkl", "wb") as f:
        pickle.dump(hv_results, f)


def report_stats(problem, args):
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
            + f"T={args.timesteps}"
            + "_"
            + f"N={args.num_points_sample}"
        )
        
        if args.label is not None:
            name += f"_{args.label}"

        path_to_results = args.results_store_path + name + "_hv_results.pkl"
 
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
