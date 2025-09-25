import json
import os

import pickle
from tqdm import tqdm
import numpy as np
import offline_moo.off_moo_bench as ob
import torch
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.utils import get_quantile_solutions, set_seed

import math
import torch.nn as nn
import torch.optim as optim
import datetime
from time import time, sleep
from torch.utils.data import DataLoader, TensorDataset

from off_ms_args import parse_args
from off_ms_nets import DiTMOO
from off_ms_utils import (
    ALLTASKSDICT,
    MultipleModels,
    SingleModel,
    SingleModelBaseTrainer,
    get_dataloader,
    tkwargs,
)

from all_funcs_utils import *

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train_proxies(args):
    task_name = ALLTASKSDICT[args.task_name]
    print(f"Task: {task_name}")

    set_seed(args.seed)

    task = ob.make(task_name)

    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    n_obj = y.shape[1]
    data_size, n_dim = tuple(X.shape)
    model_save_dir = args.proxies_store_path
    os.makedirs(model_save_dir, exist_ok=True)

    model = MultipleModels(
        n_dim=n_dim,
        n_obj=n_obj,
        train_mode="Vallina",
        hidden_size=[2048, 2048],
        save_dir=args.proxies_store_path,
        save_prefix=f"MultipleModels-Vallina-{task_name}-{args.seed}",
    )
    model.set_kwargs(**tkwargs)

    trainer_func = SingleModelBaseTrainer

    for which_obj in range(n_obj):

        y0 = y[:, which_obj].copy().reshape(-1, 1)

        trainer = trainer_func(
            model=list(model.obj2model.values())[which_obj],
            which_obj=which_obj,
            args=args,
        )

        (train_loader, val_loader) = get_dataloader(
            X,
            y0,
            val_ratio=(
                1 - args.proxies_val_ratio
            ),  # means 0.9 for training and 0.1 for validation
            batch_size=args.proxies_batch_size,
        )

        trainer.launch(train_loader, val_loader)


def train_spread(args):

    model = DiTMOO(args.input_dim,
                   args.num_obj,
                   num_blocks=args.num_blocks)
    model.to(args.device)

    task_name = ALLTASKSDICT[args.task_name]
    print(f"Task: {task_name}")

    set_seed(args.seed)

    task = ob.make(task_name)

    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    data_size = int(X.shape[0] - args.ddpm_validation_size)
    X_val = X[data_size:]
    y_val = y[data_size:]
    X = X[:data_size]
    y = y[:data_size]

    tensor_x = torch.from_numpy(X).float()
    tensor_y = torch.from_numpy(y).float()
    tensor_x_val = torch.from_numpy(X_val).float()
    tensor_y_val = torch.from_numpy(y_val).float()

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
                # t = torch.randint(0, args.timesteps, (args.batch_size,)).to(args.device) 
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
                c = obj_values.detach()
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
                c = val_obj_values.detach()
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

    if args.num_epochs > 0:
        # Save the model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "losses": LOSSES,
            "args": args,
        }
        torch.save(checkpoint, DDPM_FILE_LAST)
    else:
        # load the model
        checkpoint = torch.load(DDPM_FILE_BEST, map_location=args.device)
        LOSSES = checkpoint["losses"]
        model.load_state_dict(checkpoint["model_state_dict"])


def one_sampling_step_spread(
    model, x_t, t, 
    beta_t, alpha_bar_t, args, classifiers, task = None
):

    # Create a tensor of timesteps with shape (num_points_sample, 1)
    t_tensor = torch.full(
        (args.num_points_sample,),
        t,
        device=args.device,
        dtype=torch.float32,
    )
    # Compute objective values
    obj_values = torch.stack(
        objective_functions(x_t, classifiers=classifiers), dim=1
    )  # shape: [num_points_sample, num_objectives]
    # Conditioning information
    c = obj_values.detach()
    with torch.no_grad():
        predicted_noise = model(x_t, t_tensor / args.timesteps, c)

    #### Reverse diffusion step (x_tprime)
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

    #### Pareto Guidance step (x_tminus1)
    if args.num_obj == 2:
        if args.need_repair:
            x_t.data = repair_bounds(x_t.data.clone(), args.bounds[0], args.bounds[1], args)
        X = x_t.clone().detach().requires_grad_()
        f1, f2 = objective_functions(X, classifiers=classifiers)
        obj_values = torch.stack([f1, f2], dim=1).detach() # Cache objective values 
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
        list_fi = objective_functions(X, classifiers=classifiers)
        obj_values = torch.stack(list_fi, dim=1).detach() # Cache objective values
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
        objective_functions,
        classifiers,
        eta_init=args.eta,
    )
    h_star, adRes = solve_for_h(
        x_t,  # x_t after the reverse diffusion step is x_t_prime
        t,
        objective_functions,
        g_x_t_minus,
        grads,
        predicted_noise,
        args,
        problem=args.problem,
        eta=eta,
        lambda_rep=args.lambda_rep, 
        sigma=args.kernel_sigma,
        use_sigma=False,
        num_inner_steps=args.num_inner_steps,
        lr_inner=args.lr_inner,
        classifiers=classifiers,
    )

    if torch.isnan(h_star).any():
        print("Nan values in h_star !")
        print("h_star: ", h_star)

    x_t = x_t - eta * h_star

    return x_t, [h_star, adRes]


def sampling(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    name = (
        args.method
        + "_"
        + task_name
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

    # Load the classifiers
    classifiers = []
    for i in range(args.num_obj):
        classifier = SingleModel(
            input_size=args.input_dim,
            which_obj=i,
            hidden_size=[2048, 2048],
            save_dir=args.proxies_store_path,
            save_prefix=f"MultipleModels-Vallina-{task_name}-{0}",
        )
        classifier.load()
        classifier = classifier.to(device)
        classifier.eval()
        classifiers.append(classifier)
    print(f"Loaded {len(classifiers)} classifiers successfully.")

    model = DiTMOO(args.input_dim,
                   args.num_obj,
                   num_blocks=args.num_blocks)
    model.to(args.device)
    DDPM_FILE = str("%s/checkpoint_ddpm_best.pth" % (args.model_dir))
    checkpoint = torch.load(DDPM_FILE, weights_only=False, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    betas = cosine_beta_schedule(args.timesteps)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0).to(args.device)

    # Start from Gaussian noise
    x_t = torch.rand((args.num_points_sample, args.input_dim)) * 2 - 1
    x_t = torch.from_numpy(task.normalize_x(x_t.cpu().numpy())).float()
    x_t = x_t.to(args.device)
    x_t.requires_grad = True
    if args.need_repair:
        x_t.data = repair_bounds(
            x_t.data.clone(), args.bounds[0], args.bounds[1], args
        )

    prev_pf_points = None
    num_optimal_points = 0

    tol_violation_timesteps = args.sampling_tol
    cur_violation_timesteps = 0

    print("START sampling ...")
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
                model,
                x_t,
                t,
                beta_t,
                alpha_bar_t,
                args,
                classifiers=classifiers,
                task = task
            )
            
            if args.need_repair:
                pf_population = repair_bounds(
                    copy.deepcopy(x_t.detach()), args.bounds[0], args.bounds[1], args
                )
            else:
                pf_population = copy.deepcopy(x_t.detach())

            pf_points, _, _ = get_non_domibated_points(
                pf_population,
                objective_functions,
                args.problem,
                classifiers=classifiers,
            )

            if prev_pf_points is not None:
                non_dom_points = torch.cat((prev_pf_points, pf_points), dim=0)
                list_fi = []
                with torch.no_grad():
                    for classifier in classifiers:
                        list_fi.append(classifier(non_dom_points).squeeze())
                pf_points, _, PS_idx = get_non_domibated_points(
                            non_dom_points,
                            objective_functions,
                            args.problem,
                            keep_shape=False,
                            classifiers=classifiers,
                            list_fi=list_fi,
                        )
                
                        
                if len(pf_points) > args.num_points_sample:
                    pf_points = select_top_n_candidates(
                                pf_points,
                                args.num_points_sample,
                                objective_functions,
                                task_name,
                                style="crowding",
                                classifiers=classifiers,
                                nadir_point = task.normalize_y(task.nadir_point, normalization_method="min-max")
                            )

            prev_pf_points = pf_points
            num_optimal_points = len(pf_points)

            if tol_violation_timesteps is not None:
                temp_pareto_set_x = copy.deepcopy(pf_points)  # .numpy()
                temp_pareto_set_y = torch.stack(
                    objective_functions(
                        temp_pareto_set_x, classifiers=classifiers
                    ),
                    dim=1,
                )  # get objective values via the classifiers
                temp_pareto_set_y = temp_pareto_set_y.detach().cpu().numpy()
                temp_pareto_set_y = task.denormalize_y(temp_pareto_set_y) # z-score denormalization
                temp_pareto_set_y = task.normalize_y(temp_pareto_set_y, normalization_method="min-max") # min-max normalization
                nadir_point = task.nadir_point
                nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")

                ## This is an approximation (w.r.t the accuracy of the classifiers )
                ## and is not the true HV ! (Let's call it "Score")
                hv_value = hv(
                    nadir_point, temp_pareto_set_y, task_name
                )

                if (
                    hv_value > best_hv
                ):  # The approximated hv_value (Score) is used for early stopping
                    best_hv = hv_value

                if round(hv_value, 3) > best_hvi_round:
                    best_hvi_round = round(hv_value, 3)
                    cur_violation_timesteps = 0
                elif tol_violation_timesteps is not None:
                    cur_violation_timesteps += 1
                    if (
                        cur_violation_timesteps >= tol_violation_timesteps
                    ):  # and not reached_conv:
                        print(f"Stopping at timestep {t} during denoising.")
                        break
                    
                pbar.set_postfix({"Score": hv_value})
            else:
                pbar.set_postfix({"Points": num_optimal_points})
                
            x_t = x_t.detach()
            pbar.update(1)
            

    print(
        f"END sampling !"
    )

    T_epch_f = time() - t0_epch_f
    
    X = task.x.copy()
    y = task.y.copy()
    
    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    fi_stack = torch.stack(
                objective_functions(
                    pf_points, classifiers=classifiers
                ),
                dim=1,
            ).detach().cpu().numpy()
    fi_stack = task.denormalize_y(fi_stack)

    # Compute the true hypervolume
    res_x = pf_points.cpu().detach().numpy()
    if args.normalization: # Denormalize the solutions
        res_x = task.denormalize_x(res_x)
    
    if task.is_discrete:
        _, dim, n_classes = tuple(res_x.shape)
        res_x = res_x.reshape(-1, dim, n_classes)
        res_x = task.to_integers(res_x)
    if task.is_sequence:
        res_x = task.to_integers(res_x)

    res_y = task.predict(res_x)  # get objective values via task.predict
    if res_y.shape[0] != res_x.shape[0]:
        res_y = res_y.T
    
    PS_indx = get_non_domibated_points(
                        torch.from_numpy(res_x).float(),
                        objective_functions,
                        args.problem,
                        keep_shape=False,
                        classifiers=classifiers,
                        list_fi = [torch.from_numpy(res_y[:,i]).float() for i in range(args.num_obj)],
                        indx_only=True
                    )
    
    res_x = res_x[PS_indx]
    res_y = res_y[PS_indx]
    
    ######## PLOT ########
    list_fi = [torch.from_numpy(res_y[:,i]).float() for i in range(args.num_obj)]
    list_fi_data = [torch.from_numpy(y[:,i]).float() for i in range(args.num_obj)]
    
    plot_pareto_front(list_fi, t, args, extra = list_fi_data, lab = "realPred")
    ######################
    
    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]

    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    nadir_point = task.nadir_point
    # For calculating hypervolume, we use the min-max normalization
    res_y = task.normalize_y(res_y, normalization_method="min-max")
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    res_y_50_percent = task.normalize_y(
        res_y_50_percent, normalization_method="min-max"
    )
    res_y_75_percent = task.normalize_y(
        res_y_75_percent, normalization_method="min-max"
    )

    # Store the results
    if not (os.path.exists(args.samples_store_path)):
        os.makedirs(args.samples_store_path)

    _, d_best = task.get_N_non_dominated_solutions(
        N=args.num_points_sample, return_x=False, return_y=True
    )
    d_best = task.normalize_y(d_best, normalization_method="min-max")

    d_best_hv = hv(nadir_point, d_best, task_name)
    hv_value = hv(nadir_point, res_y, task_name)
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

    # print(f"Pareto Set: {pareto_set}")
    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")
    # Save the results
    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value,
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
        "T_epch_f": T_epch_f,
    }

    np.save(args.samples_store_path + name + "_x.npy", res_x)
    np.save(args.samples_store_path + name + "_y.npy", res_y)

    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.json", "w") as f:
        json.dump(hv_results, f, indent=4)

    # Print computation time
    convert_seconds(T_epch_f)
    print(datetime.datetime.now())

    return res_x, res_y


def evaluation(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    name = (
        args.method
        + "_"
        + task_name
        + "_"
        + str(args.seed)
        + "_"
        + f"T={args.timesteps}"
        + "_"
        + f"N={args.num_points_sample}"
    )
    if args.label is not None:
        name += f"_{args.label}"
    
    res_x = np.load(args.samples_store_path + name + "_x.npy")
    res_y = np.load(args.samples_store_path + name + "_y.npy")
    print(f"Loaded the generated samples from {args.samples_store_path}")

    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]

    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    nadir_point = task.nadir_point
    # For calculating hypervolume, we use the min-max normalization
    res_y = task.normalize_y(res_y, normalization_method="min-max")
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    res_y_50_percent = task.normalize_y(
        res_y_50_percent, normalization_method="min-max"
    )
    res_y_75_percent = task.normalize_y(
        res_y_75_percent, normalization_method="min-max"
    )

    _, d_best = task.get_N_non_dominated_solutions(N=256, return_x=False, return_y=True)

    d_best = task.normalize_y(d_best, normalization_method="min-max")

    d_best_hv = hv(nadir_point, d_best, task_name)
    hv_value = hv(nadir_point, res_y, task_name)
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")

    # Save the results
    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value,
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
    }

    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.json", "w") as f:
        json.dump(hv_results, f, indent=4)


def report_stats(args):
    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    # task = ob.make(task_name)
    print("#######################################################")
    print(f"Task: {task_name}")
    print("#######################################################")

    list_hv_100th = []
    list_hv_75th = []
    list_hv_50th = []

    for seed in args.list_seed:
        # Get the data
        name = (
            args.method
            + "_"
            + task_name
            + "_"
            + str(seed)
            + "_"
            + f"T={args.timesteps}"
            + "_"
            + f"N={args.num_points_sample}"
        )
        if args.label is not None:
            name += f"_{args.label}"

        path_to_results = args.results_store_path + name + "_hv_results.json"

        list_hv_100th.append(get_value_from_json(path_to_results, "hypervolume/100th"))
        list_hv_75th.append(get_value_from_json(path_to_results, "hypervolume/75th"))
        list_hv_50th.append(get_value_from_json(path_to_results, "hypervolume/50th"))

    # Save the results
    mean_std_hv_results = {
        "TASK": task_name,
        "mean_std_hypervolume/100th": mean_std_stats(list_hv_100th, to_decimal=2),
        "mean_std_hypervolume/75th": mean_std_stats(list_hv_75th, to_decimal=2),
        "mean_std_hypervolume/50th": mean_std_stats(list_hv_50th, to_decimal=2),
    }

    with open(
        args.results_store_path + "mean_std_hv_results" + name + ".json", "w"
    ) as f:
        json.dump(mean_std_hv_results, f, indent=4)
        
    print("mean_std_hypervolume/100th", mean_std_stats(list_hv_100th, to_decimal=2))

    return mean_std_hv_results
