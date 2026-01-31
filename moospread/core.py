"""Main module."""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import copy
import math
import json

import cv2
import re
import glob

import os
import pickle
from tqdm import tqdm
import datetime
from time import time
import dis

from pymoo.config import Config
Config.warnings['not_compiled'] = False
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from moospread.utils import *
print
class SPREAD:
    def __init__(self, 
                 problem,
                 mode: str = "online",
                 model = None,
                 surrogate_model = None,
                 dataset = None,
                 xi_shift = None,
                 data_size: int = 10000,
                 validation_split=0.1,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 timesteps: int = 1000,
                 batch_size: int = 256,
                 train_lr: float = 1e-4,
                 train_lr_surrogate: float = 1e-4,
                 num_epochs: int = 1000,
                 num_epochs_surrogate: int = 1000,
                 train_tol: int = 100,
                 train_tol_surrogate: int = 100,
                 mobo_coef_lcb=0.1,
                 model_dir: str = "./model_dir",
                 proxies_store_path: str = "./proxies_dir",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 seed: int = 0,
                 offline_global_clamping: bool = False,
                 offline_normalization_method: str = "z_score",
                 train_func_surrogate = None,
                 plot_func = None,
                 verbose: bool = True):
        
        self.mode = mode.lower()
        if self.mode not in ["offline", "online", "bayesian"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of ['offline', 'online', 'bayesian']")
        
        assert problem is not None, "Problem must be provided"
        self.problem = problem
        if self.mode in ["online", "bayesian"]:
            assert not is_pass_function(self.problem._evaluate), "Problem must have the '_evaluate' method implemented."
        self.device = device
        
        assert 0.0 <= validation_split < 1.0, "validation_split must be in [0.0, 1.0)"
        self.validation_split = validation_split
        self.train_lr = train_lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.timesteps = timesteps
        self.train_tol = train_tol
        if self.mode in ["offline", "bayesian"]:
            self.train_lr_surrogate = train_lr_surrogate
            self.num_epochs_surrogate = num_epochs_surrogate
            self.train_tol_surrogate = train_tol_surrogate
        if self.mode == "bayesian":
            self.mobo_coef_lcb = mobo_coef_lcb
        
        self.xi_shift = xi_shift
        self.model_dir = model_dir+f"/{self.problem.__class__.__name__}_{self.mode}"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.train_func_surrogate = train_func_surrogate
        self.plot_func = plot_func

        self.seed = seed
        # Set the seed for reproducibility
        set_seed(self.seed)

        self.model = model
        if self.model is None:
            self.model = DiTMOO(
                input_dim=problem.n_var,
                num_obj=problem.n_obj,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_blocks=num_blocks
            )
        self.surrogate_model = surrogate_model
        if self.surrogate_model is not None:
            self.surrogate_given = True
        else:
            self.surrogate_given = False
        self.proxies_store_path = proxies_store_path
        
        self.verbose = verbose
        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total number of learnable parameters: {total_params}")
        
        self.dataset = dataset
        if self.mode in ["offline", "online"]:
            if self.dataset is None:
                if self.mode == "offline":
                    print("Training dataset not provided for offline mode.")
                    assert data_size > 0, "Training data size must be positive."
                    # self.problem._evaluate should exist in this case
                    assert (not is_pass_function(self.problem._evaluate)), "Problem must have the '_evaluate' method implemented when dataset is not provided in offline mode."
                    print("Generating training dataset ...")
                self.dataset = self.get_training_data(self.problem, 
                                                     num_samples=data_size)
                if self.verbose:
                    print("Training dataset generated.")
                
        if self.mode == "offline":
            assert offline_normalization_method in ["z_score", "min_max", None], "Invalid normalization method"
            if offline_normalization_method == "z_score":
                self.offline_normalization = offdata_z_score_normalize
                self.offline_denormalization = offdata_z_score_denormalize
            elif offline_normalization_method == "min_max":
                self.offline_normalization = offdata_min_max_normalize
                self.offline_denormalization = offdata_min_max_denormalize
            else:
                self.offline_normalization = lambda x: x
                self.offline_denormalization = lambda x: x
                
            if self.mode == "offline":
                X, y = self.dataset
                X = X.clone().detach()
                y = y.clone().detach()
                if self.problem.is_discrete:
                    X = offdata_to_logits(X)
                    _, n_dim, n_classes = tuple(X.shape)
                    X = X.reshape(-1, n_dim * n_classes)
                if self.problem.is_sequence:
                    X = offdata_to_logits(X)
                # For usual cases, we normalize the inputs 
                # and outputs with z-score normalization
                X, self.X_meanormin, self.X_stdormax = self.offline_normalization(X)
                y, self.y_meanormin, self.y_stdormax = self.offline_normalization(y)
                self.dataset = (X, y)

            if self.problem.has_bounds():
                xl = self.problem.xl
                xu = self.problem.xu
                ## Normalize the bounds
                # xl
                if self.problem.is_discrete:
                    xl = offdata_to_logits(xl)
                    _, n_dim, n_classes = tuple(xl.shape)
                    xl = xl.reshape(-1, n_dim * n_classes)
                if self.problem.is_sequence:
                    xl = offdata_to_logits(xl)
                xl, _, _ = self.offline_normalization(xl,
                                                      self.X_meanormin,
                                                      self.X_stdormax)
                # xu
                if self.problem.is_discrete:
                    xu = offdata_to_logits(xu)
                    _, n_dim, n_classes = tuple(xu.shape)
                    xu = xu.reshape(-1, n_dim * n_classes)
                if self.problem.is_sequence:
                    xu = offdata_to_logits(xu)
                xu, _, _ = self.offline_normalization(xu,
                                                      self.X_meanormin,
                                                      self.X_stdormax)
                ## Set the normalized bounds
                self.problem.xl = xl
                self.problem.xu = xu
                if offline_global_clamping:
                    self.problem.global_clamping = True
                    
    def objective_functions(self, 
                            points, 
                            return_as_dict: bool = False,
                            return_values_of=None,
                            get_constraint=False,
                            get_grad_mobo=False,
                            evaluate_true=False):
        if evaluate_true:
            if self.problem.need_repair:
                points = self.repair_bounds(points)
            if get_constraint:
                return self.problem.evaluate(points, return_as_dict=True, 
                                             return_values_of=["F", "G", "H"])
            return self.problem.evaluate(points, return_as_dict=return_as_dict,
                                         return_values_of=return_values_of)
        # Define the objective functions for the optimization problem
        if self.mode == "online":
            if get_constraint:
                return self.problem.evaluate(points, return_as_dict=True, 
                                             return_values_of=["F", "G", "H"])
            return self.problem.evaluate(points, return_as_dict=return_as_dict,
                                         return_values_of=return_values_of)
        elif self.mode == "offline":
            scores = []
            for proxy in self.surrogate_model:
                scores.append(proxy(points).squeeze())
            return torch.stack(scores, dim=1)
        elif self.mode == "bayesian":
            x = points.detach().cpu().numpy()
            eval_result = self.surrogate_model.evaluate(x, std=True, 
                                                        calc_gradient=get_grad_mobo)
            mean = torch.from_numpy(eval_result["F"]).float().to(self.device)
            std = torch.from_numpy(eval_result["S"]).float().to(self.device)
            Y_val = mean - self.mobo_coef_lcb * std
            if get_grad_mobo:
                out = {}
                mean_grad = torch.from_numpy(eval_result["dF"]).float().to(self.device)
                std_grad = torch.from_numpy(eval_result["dS"]).float().to(self.device)
                Grad_val = mean_grad - self.mobo_coef_lcb * std_grad
                out["dF"] = [Grad_val[:, i, :] for i in range(Grad_val.shape[1])]
                out["F"] = Y_val
            else:
                out = Y_val
            return out
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def solve(self, num_points_sample=500,
                 strict_guidance=False, 
                 rho_scale_gamma=0.9,
                 nu_t=10.0, eta_init=0.9, 
                 num_inner_steps=10, lr_inner=0.9,
                 free_initial_h=True,
                 use_sigma_rep=False, kernel_sigma_rep=0.01,
                 iterative_plot=True, plot_period=100, 
                 plot_dataset=False, plot_population=False,
                 elev=30, azim=45, legend=False,
                 max_backtracks=100, label=None, save_results=True,
                 load_models=False,
                 samples_store_path="./samples_dir/",
                 images_store_path="./images_dir/",
                 n_init_mobo=100, use_escape_local_mobo=True, 
                 n_steps_mobo=20, spread_num_samp_mobo=25,
                 batch_select_mobo=5):
        set_seed(self.seed)
        if self.mode in ["offline", "online"]:
            X, y = self.dataset
            
            if self.mode == "offline":
                #### SURROGATE MODEL TRAINING ####
                if not load_models or self.surrogate_given:
                    self.train_surrogate(X, y)
                else:
                    # Load the proxies
                    self.surrogate_model = []
                    for i in range(self.problem.n_obj):
                        classifier = SingleModel(
                                input_size=self.problem.n_var,
                                which_obj=i,
                                device=self.device,
                                hidden_size=[2048, 2048],
                                save_dir=self.proxies_store_path,
                                save_prefix=f"MultipleModels-Vallina-{self.problem.__class__.__name__}-{self.seed}",
                            )
                        classifier.load()
                        classifier = classifier.to(self.device)
                        classifier.eval()
                        self.surrogate_model.append(classifier)
   
            # Create DataLoader
            train_dataloader, val_dataloader = get_ddpm_dataloader(X,
                                                y,
                                                validation_split=self.validation_split, 
                                                batch_size=self.batch_size)
            
            #### DIFFUSION MODEL TRAINING ####
            if not load_models:
                self.train(train_dataloader, val_dataloader=val_dataloader)
            #### SPREAD SAMPLING ####
            res_x, res_y = self.sampling(num_points_sample, 
                 strict_guidance=strict_guidance, 
                 rho_scale_gamma=rho_scale_gamma,
                 nu_t=nu_t, eta_init=eta_init, 
                 num_inner_steps=num_inner_steps, lr_inner=lr_inner,
                 free_initial_h=free_initial_h,
                 use_sigma_rep=use_sigma_rep, kernel_sigma_rep=kernel_sigma_rep,
                 iterative_plot=iterative_plot, plot_period=plot_period, 
                 plot_dataset=plot_dataset, plot_population=plot_population,
                 elev=elev, azim=azim, legend=legend,
                 max_backtracks=max_backtracks, label=label,
                 save_results=save_results,
                 samples_store_path=samples_store_path,
                 images_store_path=images_store_path)
            
            return res_x, res_y
            
        elif self.mode == "bayesian":
            self.verbose = False
            hv_all_value = []
            # initialize n_init solutions 
            x_init = lhs_no_evaluation(self.problem.n_var, 
                                       n_init_mobo)
            x_init = torch.from_numpy(x_init).float().to(self.device)
            y_init = self.problem.evaluate(x_init).detach().cpu().numpy()

            # initialize dominance-classifier for non-dominance relation
            p_rel_map, s_rel_map = init_dom_rel_map(300)
            p_model = init_dom_nn_classifier(
                x_init, y_init, p_rel_map, pareto_dominance, self.problem.n_var,
            )  
            self.dominance_classifier = p_model
            
            evaluated = len(y_init)
            X = x_init.detach().cpu().numpy()
            Y = y_init
            hv = HV(ref_point=np.array(self.problem.ref_point))
            hv_value = hv(Y)
            hv_all_value.append(hv_value)
            z = torch.zeros(self.problem.n_obj).to(self.device)

            escape_flag = False
            if use_escape_local_mobo:
                # Counter for tracking iterations since last switch
                iteration_since_switch = 0
                # Parameters for switching methods
                hv_change_threshold = 0.05  # Threshold for HV value change
                hv_history_length = 3  # Number of recent iterations to consider
                hv_history = []  # Store recent HV values
                # Initialize list to store historical data
                history_Y = []
            
            # Start the main loop for Bayesian-SPREAD
            with tqdm(
                total=n_steps_mobo,
                desc=f"SPREAD (MOBO)",
                unit="k",
            ) as pbar:

                for k_iter in range(n_steps_mobo): 
                    # Solution normalization
                    transformation = StandardTransform([0, 1])
                    transformation.fit(X, Y)
                    X_norm, Y_norm = transformation.do(X, Y)

                    #### SURROGATE MODEL TRAINING ####
                    self.train_surrogate(X_norm, Y_norm)

                    if use_escape_local_mobo:
                        _, index = environment_selection(Y, len(X) // 3)
                        PopDec = X[index, :]
                    else:
                        PopDec = X

                    PopDec_dom_labels, PopDec_cfs = nn_predict_dom_intra(
                        PopDec, p_model, self.device
                    )
                    sorted_pop = sort_population(PopDec, PopDec_dom_labels, PopDec_cfs)
                        
                    if not escape_flag:
                        # **** Generate new offspring using SPREAD**** #
                        self.model.to(self.device)
                        #### DIFFUSION MODEL TRAINING ####
                        train_dataloader, val_dataloader, dataset_size = mobo_get_ddpm_dataloader(sorted_pop, 
                                                            self.objective_functions,
                                                            self.device,
                                                            self.batch_size,
                                                            self.validation_split)
                        self.train(train_dataloader, 
                                   val_dataloader=val_dataloader,
                                   disable_progress_bar=True)
                        #### SPREAD SAMPLING ####
                        new_offsprings = []
                        for i in range(spread_num_samp_mobo):
                            num_points_sample = dataset_size
                            pf_points, _ = self.sampling(num_points_sample, 
                                            strict_guidance=strict_guidance, 
                                            rho_scale_gamma=rho_scale_gamma,
                                            nu_t=nu_t, eta_init=eta_init, 
                                            num_inner_steps=num_inner_steps, lr_inner=lr_inner,
                                            free_initial_h=free_initial_h,
                                            use_sigma_rep=use_sigma_rep, kernel_sigma_rep=kernel_sigma_rep,
                                            iterative_plot=iterative_plot, plot_period=plot_period,
                                            plot_dataset=plot_dataset, plot_population=plot_population,
                                            elev=elev, azim=azim, legend=legend,
                                            max_backtracks=max_backtracks, label=label,
                                            samples_store_path=samples_store_path,
                                            images_store_path=images_store_path,
                                            disable_progress_bar=True,
                                            save_results=False, evaluate_final=False)
                            new_offsprings.append(pf_points)
                        X_psl = np.vstack(new_offsprings)
                    else:
                        #### SBX OFFSPRING GENERATION ####
                        rows_to_take = int(1 / 3 * sorted_pop.shape[0])
                        offspringA = sorted_pop[:rows_to_take, :]
                        if len(offspringA) % 2 == 1:
                            offspringA = offspringA[:-1]
                        new_pop = np.empty((0, self.problem.n_var))
                        for _ in range(1000):
                            result = sbx(offspringA, eta=15)
                            new_pop = np.vstack((new_pop, result))
                        X_psl = new_pop
                        
                    pop_size_used = X_psl.shape[0]
                        
                    # Mutate the new offspring
                    X_psl = pm_mutation(X_psl, [self.problem.xl.detach().cpu().numpy(), 
                                                    self.problem.xu.detach().cpu().numpy()])

                    Y_candidate_mean = self.surrogate_model.evaluate(X_psl)["F"]
                    Y_candidate_std = self.surrogate_model.evaluate(X_psl, std=True)["S"]

                    rows_with_nan = np.any(np.isnan(Y_candidate_mean), axis=1)
                    Y_candidate_mean = Y_candidate_mean[~rows_with_nan]
                    Y_candidate_std = Y_candidate_std[~rows_with_nan]
                    X_psl = X_psl[~rows_with_nan]

                    Y_candidate = Y_candidate_mean - self.mobo_coef_lcb * Y_candidate_std
                    Y_candidate_mean = Y_candidate
                        
                    #### BATCH SELECTION ####
                    nds = NonDominatedSorting()
                    idx_nds = nds.do(Y_norm)
                    Y_nds = Y_norm[idx_nds[0]]
                    best_subset_list = []
                    Y_p = Y_nds
                    for b in range(batch_select_mobo):
                        hv = HV(
                            ref_point=np.max(np.vstack([Y_p, Y_candidate_mean]), axis=0)
                        )
                        best_hv_value = 0
                        best_subset = None

                        for k in range(len(Y_candidate_mean)):
                            Y_subset = Y_candidate_mean[k]
                            Y_comb = np.vstack([Y_p, Y_subset])
                            hv_value_subset = hv(Y_comb)
                            if hv_value_subset > best_hv_value:
                                best_hv_value = hv_value_subset
                                best_subset = [k]

                        Y_p = np.vstack([Y_p, Y_candidate_mean[best_subset]])
                        best_subset_list.append(best_subset)
                    
                    best_subset_list = np.array(best_subset_list).T[0]

                    X_candidate = X_psl
                    X_new = X_candidate[best_subset_list]
                    Y_new = self.problem.evaluate(torch.from_numpy(X_new).float().to(self.device)).detach().cpu().numpy()
                        
                    Y_new = torch.tensor(Y_new).to(self.device)
                    X_new = torch.tensor(X_new).to(self.device)

                    X = np.vstack([X, X_new.detach().cpu().numpy()])
                    Y = np.vstack([Y, Y_new.detach().cpu().numpy()])
                    hv = HV(ref_point=np.array(self.problem.ref_point))
                    hv_value = hv(Y)
                    hv_all_value.append(hv_value)
                                    
                    rows_with_nan = np.any(np.isnan(Y), axis=1)
                    X = X[~rows_with_nan, :]
                    Y = Y[~rows_with_nan, :]

                    update_dom_nn_classifier(
                        p_model, X, Y, p_rel_map, pareto_dominance, self.problem
                    )
                        
                    hv_text = f"{hv_value:.4e}" 
                    evaluated = evaluated + batch_select_mobo

                    #### DECISION TO SWITCH OPERATOR ####
                    if use_escape_local_mobo:
                        # Current operator
                        if not escape_flag:
                            operator_text = "Diffusion"
                        else:
                            operator_text = "SBX"
                        # Update historical data and calculate reference point
                        history_Y.append(Y)
                        if len(history_Y) > k:
                            history_Y.pop(0)
                        all_Y = np.vstack(history_Y)  # Combine historical data
                        nds_hist = NonDominatedSorting()
                        idx_nds_hist = nds_hist.do(all_Y)
                        Y_nds_hist = all_Y[idx_nds_hist[0]]  # Get non-dominated individuals
                        quantile_values = np.quantile(Y_nds_hist, 0.95, axis=0)
                        ref_point_method2 = 1.1 * quantile_values
                        # Calculate approximate HV
                        hv_method2 = HV(ref_point=ref_point_method2)
                        hv_value_method2 = hv_method2(Y)
                        # Update HV value history
                        hv_history.append(hv_value_method2)
                        if len(hv_history) > hv_history_length:
                            hv_history.pop(0)

                        if len(hv_history) == hv_history_length:
                            avg_hv = sum(hv_history[:-1]) / (hv_history_length - 1)
                            if avg_hv == 0:
                                hv_change = 0
                            else:
                                hv_change = abs((hv_history[-1] - avg_hv) / avg_hv)
                            # Determine if method needs to be switched
                            if iteration_since_switch >= 2:
                                if hv_change < hv_change_threshold:
                                    escape_flag = not escape_flag
                                    iteration_since_switch = 0  # Reset counter
                            else:
                                iteration_since_switch += 1  # If already switched, increment counter

                        pbar.set_postfix({"HV": hv_text, "Operator": operator_text, "Population": pop_size_used, "Num Points": evaluated})
                    else:
                        pbar.set_postfix({"HV": hv_text, "Population": pop_size_used, "Num Points": evaluated})
                        
                    pbar.update(1)
                        
            name_t = (
                    "spread"
                    + "_"
                    + self.problem.__class__.__name__
                    + "_T"
                    + str(self.timesteps)
                    + "_K"
                    + str(n_steps_mobo)
                    + "_FE"
                    + str(n_steps_mobo*batch_select_mobo)
                    + "_"
                    + f"seed={self.seed}"
                    + "_"
                    + self.mode
                )
            
            if not (os.path.exists(samples_store_path)):
                os.makedirs(samples_store_path)

            if save_results:
                # Save the samples and HV values
                np.save(samples_store_path + name_t + "_x.npy", X)
                np.save(samples_store_path + name_t + "_y.npy", Y)
                print("\n================ Final Results ================\n")
                print(f"Total function evaluations: {evaluated}")
                print(f"Final hypervolume: {hv_value:.4e}")
                print(f"Samples and HV values are saved to {samples_store_path}\n")
                
            outfile = samples_store_path + name_t + "_hv_results.pkl"
            with open(outfile, "wb") as f:
                pickle.dump(hv_all_value, f)
                
            return X, Y, hv_all_value

        
    def train(self, 
              train_dataloader, 
              val_dataloader=None,
              disable_progress_bar=False):
        set_seed(self.seed)
        if self.verbose:
            print(datetime.datetime.now())
        
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_lr)

        betas = self.cosine_beta_schedule(self.timesteps)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0).to(self.device)

        DDPM_FILE_LAST = str("%s/checkpoint_ddpm_last.pth" % (self.model_dir))
        if val_dataloader:
            DDPM_FILE_BEST = str("%s/checkpoint_ddpm_best.pth" % (self.model_dir))
            best_val_loss = np.inf
            tol_violation_epoch = self.train_tol
            cur_violation_epoch = 0
        
        time_start = time()

        with tqdm(
            total=self.num_epochs,
            desc=f"DDPM Training",
            unit="epoch",
            disable=disable_progress_bar,
        ) as pbar:

            for epoch in range(self.num_epochs):
                self.model.train()
                train_loss = 0.0
                for indx_batch, (batch, obj_values) in enumerate(train_dataloader):
                    optimizer.zero_grad()

                    # Extract batch points
                    points = batch.to(self.device)
                    points.requires_grad = True  # Enable gradients
                    obj_values = obj_values.to(self.device)

                    # Sample random timesteps for each data point
                    t = torch.randint(0, self.timesteps, (points.shape[0],)).to(self.device)
                    alpha_bar_t = alpha_cumprod[t].unsqueeze(1)  # shape: [batch_size, 1]

                    # Forward process: Add noise to points
                    noise = torch.randn_like(points).to(
                        self.device
                    )  # shape: [batch_size, n_var]
                    x_t = (
                        torch.sqrt(alpha_bar_t) * points
                        + torch.sqrt(1 - alpha_bar_t) * noise
                    )

                    # Conditioning information
                    c = obj_values
                    if self.xi_shift is not None:
                        c = c + self.xi_shift
                    else:
                        xi_shift = c[c > 0].min() if (c > 0).any() else 1e-5
                        c = c + xi_shift
                    # Model predicts noise
                    predicted_noise = self.model(
                        x_t, t.float() / self.timesteps, c
                    )

                    # Compute loss
                    loss_simple = self.l_simple_loss(predicted_noise, 
                                                     noise.detach())
                    loss_simple.backward()
                    optimizer.step()
                    train_loss += loss_simple.item()
                    
                train_loss = train_loss / len(train_dataloader)

                if val_dataloader:
                    # Validation
                    self.model.eval()
                    val_loss = 0.0
                    for indx_batch, (val_batch, val_obj_values) in enumerate(val_dataloader):
                        # Extract batch points
                        val_points = val_batch.to(
                            self.device
                        )
                        val_points.requires_grad = True  # Enable gradients
                        val_obj_values = val_obj_values.to(self.device)

                        # Sample random timesteps for each data point
                        t = torch.randint(0, self.timesteps, (val_points.shape[0],)).to(self.device)
                        alpha_bar_t = alpha_cumprod[t].unsqueeze(1)  # shape: [batch_size, 1]

                        # Forward process: Add noise to points
                        val_noise = torch.randn_like(val_points).to(
                            self.device
                        )  # shape: [batch_size, n_var]
                        val_x_t = (
                            torch.sqrt(alpha_bar_t) * val_points
                            + torch.sqrt(1 - alpha_bar_t) * val_noise
                        )

                        # Conditioning information
                        c = val_obj_values
                        if self.xi_shift is not None:
                            c = c + self.xi_shift
                        else:
                            xi_shift = c[c > 0].min() if (c > 0).any() else 1e-6
                            c = c + xi_shift
                        # Model predicts noise
                        val_predicted_noise = self.model(
                            val_x_t, t.float() / self.timesteps, c
                        )
                        loss_simple = self.l_simple_loss(val_predicted_noise, 
                                                         val_noise.detach())
                        val_loss += loss_simple.item()

                    val_loss = val_loss / len(val_dataloader)

                    if val_loss <= best_val_loss:
                        best_val_loss = val_loss
                        cur_violation_epoch = 0
                        # Save the model
                        checkpoint = {
                            "model_state_dict": self.model.state_dict(),
                            "epoch": epoch + 1,
                        }
                        torch.save(checkpoint, DDPM_FILE_BEST)
                    else:
                        cur_violation_epoch += 1
                        if cur_violation_epoch >= tol_violation_epoch:
                            if self.verbose:
                                print(f"Early Stopping at epoch {epoch + 1}.")
                            break

                    pbar.set_postfix({"val_loss": val_loss})
                else:
                    pbar.set_postfix({"train_loss": train_loss})
                pbar.update(1)

        comp_time = time() - time_start
        if self.verbose:
            convert_seconds(comp_time)
        if self.verbose:
            print(datetime.datetime.now())
   
        if val_dataloader and self.verbose:
            print(f"Best model saved at: {DDPM_FILE_BEST}")
        # Save the model
        checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "train_time": comp_time,
                "num_epochs": self.num_epochs,
                "train_tol": self.train_tol,
                "train_lr": self.train_lr,
                "batch_size": self.batch_size,
                "timesteps": self.timesteps,
            }
        torch.save(checkpoint, DDPM_FILE_LAST)
        if self.verbose:
            print(f"Final model saved at: {DDPM_FILE_LAST}")

    def sampling(self, num_points_sample, 
                 strict_guidance=False, 
                 rho_scale_gamma=0.9,
                 nu_t=10.0, eta_init=0.9, 
                 num_inner_steps=10, lr_inner=1e-4,
                 free_initial_h=True,
                 use_sigma_rep=False, kernel_sigma_rep=0.01,
                 iterative_plot=True, plot_period=100, 
                 plot_dataset=False, plot_population=False,
                 elev=30, azim=45, legend=False,
                 max_backtracks=25, label=None,
                 samples_store_path="./samples_dir/",
                 images_store_path="./images_dir/",
                 disable_progress_bar=False,
                 save_results=True, evaluate_final=True):
        # Set the seed
        set_seed(self.seed)
        if save_results:
            # Store the results
            if not (os.path.exists(samples_store_path)):
                os.makedirs(samples_store_path)

        name = (
            "spread"
            + "_"
            + self.problem.__class__.__name__
            + "_"
            + f"T={self.timesteps}"
            + "_"
            + f"N={num_points_sample}"
            + "_"
            + f"seed={self.seed}"
            + "_"
            + self.mode
        )

        if label is not None:
            name += f"_{label}"

        self.model.to(self.device)
        DDPM_FILE = str("%s/checkpoint_ddpm_best.pth" % (self.model_dir))
        # Load the best model if exists, else load the last model
        if not os.path.exists(DDPM_FILE):
            DDPM_FILE = str("%s/checkpoint_ddpm_last.pth" % (self.model_dir))
            if not os.path.exists(DDPM_FILE):
                raise ValueError(f"No trained model found in {self.model_dir}")
        checkpoint = torch.load(DDPM_FILE, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        betas = self.cosine_beta_schedule(self.timesteps)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0).to(self.device)

        # Start from random points in the decision space
        x_t = torch.rand((num_points_sample, self.problem.n_var)) # in [0, 1]
        x_t = self.problem.bounds()[0] + (self.problem.bounds()[1] - self.problem.bounds()[0]) * x_t # scale to bounds
        if self.mode == "offline":
            x_t, _, _ = self.offline_normalization(x_t,
                                                    self.X_meanormin,
                                                    self.X_stdormax)
        x_t = x_t.to(self.device)
        x_t.requires_grad = True
        if self.problem.need_repair:
            x_t.data = self.repair_bounds(x_t.data.clone())
        if self.mode in ["online", "offline"]:
            if iterative_plot and (not is_pass_function(self.problem._evaluate)):        
                if self.problem.n_obj <= 3:   
                    pf_population = x_t.detach()
                    pf_points, _, _ = self.get_non_dominated_points(
                        pf_population,
                        keep_shape=False
                    )
                    if self.mode == "offline":
                        # Denormalize the points before plotting
                        res_x_t = pf_points.clone().detach()
                        res_x_t = self.offline_denormalization(res_x_t,
                                                                    self.X_meanormin, 
                                                                    self.X_stdormax)
                        res_pop = pf_population.clone().detach()
                        res_pop = self.offline_denormalization(res_pop,
                                                                    self.X_meanormin, 
                                                                    self.X_stdormax)
                        norm_xl, norm_xu = self.problem.bounds()
                        xl, xu = self.problem.original_bounds
                        self.problem.xl = xl
                        self.problem.xu = xu
                        if self.problem.is_discrete:
                            _, dim, n_classes = tuple(res_x_t.shape)
                            res_x_t = res_x_t.reshape(-1, dim, n_classes)
                            res_x_t = offdata_to_integers(res_x_t)
                                    
                            _, dim_pop, n_classes_pop = tuple(res_pop.shape)
                            res_pop = res_pop.reshape(-1, dim_pop, n_classes_pop)
                            res_pop = offdata_to_integers(res_pop)
                        if self.problem.is_sequence:
                            res_x_t = offdata_to_integers(res_x_t)
                            res_pop = offdata_to_integers(res_pop)
                        # we need to evaluate the true objective functions for plotting
                        list_fi =  self.objective_functions(res_x_t, 
                                                                    evaluate_true=True).split(1, dim=1)
                        list_fi_pop =  self.objective_functions(res_pop, 
                                                                        evaluate_true=True).split(1, dim=1) 
                        # restore the normalized bounds
                        self.problem.xl = norm_xl
                        self.problem.xu = norm_xu
                    else:
                        list_fi = self.objective_functions(pf_points).split(1, dim=1)
                        list_fi_pop =  self.objective_functions(pf_population.detach()).split(1, dim=1)
                    pareto_front = None
                    list_fi = [fi.detach().cpu().numpy() for fi in list_fi]
                    list_fi_pop = [fi.detach().cpu().numpy() for fi in list_fi_pop]
                    if self.problem.pareto_front() is not None:
                        pareto_front = self.problem.pareto_front()
                        pareto_front = [pareto_front[:, i] for i in range(self.problem.n_obj)]
                    if self.plot_func is not None:
                        self.plot_func(list_fi, self.timesteps,
                                       num_points_sample,
                                       extra=pareto_front,
                                       plot_dataset=plot_dataset,
                                       dataset = self.dataset,
                                       elev=elev, azim=azim, legend=legend,
                                       label=label, images_store_path=images_store_path)
                    else:
                        self.plot_pareto_front(list_fi,  self.timesteps,
                                                num_points_sample,
                                                extra=pareto_front,
                                                plot_dataset=plot_dataset,
                                                pop=list_fi_pop,
                                                elev=elev, azim=azim, legend=legend,
                                                label=label, images_store_path=images_store_path)

        prev_pf_points = None
        num_optimal_points = 0

        point_n0 = None
        optimizer_n0 = None
        if strict_guidance:
            #### Initialize 1 target point for direction perturbation.
            # (If strict_guidance, get new perturbation based on the MGD direction of
            # a single initialized point. Otherwise, a random perturbation is used (as in the paper).)
            point_n0 = torch.rand((1, self.problem.n_var)) # in [0, 1]
            point_n0 = self.problem.bounds()[0] + (self.problem.bounds()[1] - self.problem.bounds()[0]) * point_n0 # scale to bounds
            if self.mode == "offline":
                point_n0, _, _ = self.offline_normalization(point_n0,
                                                      self.X_meanormin,
                                                      self.X_stdormax)
            point_n0 = point_n0.to(self.device)
            point_n0.requires_grad = True
            if self.problem.need_repair:
                point_n0.data = self.repair_bounds(
                    point_n0.data.clone()
                )
            optimizer_n0 = optim.Adam([point_n0], lr=1e-2)

        if self.verbose:
            print(datetime.datetime.now())
            print(f"START sampling {num_points_sample} points ...")

        time_start = time()
        self.model.eval()

        with tqdm(
            total=self.timesteps,
            desc=f"SPREAD Sampling",
            unit="t",
            disable=disable_progress_bar,
        ) as pbar:

            for t in reversed(range(self.timesteps)):
                x_t.requires_grad_(True)
                if self.problem.need_repair:
                    x_t.data = self.repair_bounds(
                        x_t.data.clone()
                    )
                # Compute beta_t and alpha_t
                beta_t = 1 - alphas[t]
                alpha_bar_t = alpha_cumprod[t]
                
                x_t = self.one_spread_sampling_step(
                    x_t,
                    num_points_sample,
                    t, beta_t, alpha_bar_t, rho_scale_gamma,
                    nu_t, eta_init, num_inner_steps, lr_inner,
                    free_initial_h=free_initial_h,
                    use_sigma=use_sigma_rep, kernel_sigma=kernel_sigma_rep,
                    strict_guidance=strict_guidance, max_backtracks=max_backtracks,
                    point_n0=point_n0, optimizer_n0=optimizer_n0,
                )

                if strict_guidance and self.problem.need_repair:
                    point_n0.data = self.repair_bounds(
                            point_n0.data.clone()
                        )

                if self.problem.need_repair:
                    pf_population = self.repair_bounds(
                        copy.deepcopy(x_t.detach())
                    )
                else:
                    pf_population = copy.deepcopy(x_t.detach())

                pf_points, _, _ = self.get_non_dominated_points(
                    pf_population,
                    keep_shape=False
                )
           
                if prev_pf_points is not None:
                    pf_points = torch.cat((prev_pf_points, pf_points), dim=0)
                    if self.mode != "bayesian":
                        pf_points, _, _ = self.get_non_dominated_points(
                                        pf_points,
                                        keep_shape=False,
                                    )
                    if len(pf_points) > num_points_sample:
                        pf_points = self.select_top_n_candidates(
                                            pf_points,
                                            num_points_sample,
                                        )

                if self.mode == "bayesian" and (pf_points is None or len(pf_points) == 0):
                    pf_points = x_t.detach()
                prev_pf_points = pf_points
                num_optimal_points = len(pf_points)
                    
                if iterative_plot and (not is_pass_function(self.problem._evaluate)):
                    if self.problem.n_obj <= 3:
                        if (t % plot_period == 0) or (t ==  self.timesteps - 1):
                            if self.mode == "offline":
                                # Denormalize the points before plotting
                                res_x_t = pf_points.clone().detach()
                                res_x_t = self.offline_denormalization(res_x_t,
                                                                            self.X_meanormin, 
                                                                            self.X_stdormax)
                                res_pop = pf_population.clone().detach()
                                res_pop = self.offline_denormalization(res_pop,
                                                                            self.X_meanormin, 
                                                                            self.X_stdormax)
                                norm_xl, norm_xu = self.problem.bounds()
                                xl, xu = self.problem.original_bounds
                                self.problem.xl = xl
                                self.problem.xu = xu
                                if self.problem.is_discrete:
                                    _, dim, n_classes = tuple(res_x_t.shape)
                                    res_x_t = res_x_t.reshape(-1, dim, n_classes)
                                    res_x_t = offdata_to_integers(res_x_t)
                                    
                                    _, dim_pop, n_classes_pop = tuple(res_pop.shape)
                                    res_pop = res_pop.reshape(-1, dim_pop, n_classes_pop)
                                    res_pop = offdata_to_integers(res_pop)
                                if self.problem.is_sequence:
                                    res_x_t = offdata_to_integers(res_x_t)
                                    res_pop = offdata_to_integers(res_pop)
                                # we need to evaluate the true objective functions for plotting
                                list_fi =  self.objective_functions(res_x_t, 
                                                                    evaluate_true=True).split(1, dim=1)
                                list_fi_pop =  self.objective_functions(res_pop, 
                                                                        evaluate_true=True).split(1, dim=1)
                                list_fi_pop = [fi.detach().cpu().numpy() for fi in list_fi_pop]
                                # restore the normalized bounds
                                self.problem.xl = norm_xl
                                self.problem.xu = norm_xu
                            elif self.mode == "bayesian":
                                # we need to evaluate the true objective functions for plotting
                                list_fi = self.objective_functions(pf_points, evaluate_true=True).split(1, dim=1)
                                list_fi_pop =  self.objective_functions(pf_population.detach(), evaluate_true=True).split(1, dim=1)
                                list_fi_pop = [fi.detach().cpu().numpy() for fi in list_fi_pop]
                            else:
                                list_fi = self.objective_functions(pf_points).split(1, dim=1)
                                list_fi_pop =  self.objective_functions(pf_population.detach()).split(1, dim=1)
                                list_fi_pop = [fi.detach().cpu().numpy() for fi in list_fi_pop]
                            
                            list_fi = [fi.detach().cpu().numpy() for fi in list_fi]
                            pareto_front = None
                            if self.problem.pareto_front() is not None:
                                pareto_front = self.problem.pareto_front()
                                pareto_front = [pareto_front[:, i] for i in range(self.problem.n_obj)]

                            if self.plot_func is not None:
                                self.plot_func(list_fi, t, 
                                                   num_points_sample,
                                                   extra= pareto_front,
                                                   plot_dataset=plot_dataset,
                                                   dataset = self.dataset,
                                                   elev=elev, azim=azim, legend=legend,
                                                   label=label, images_store_path=images_store_path)
                            else:
                                self.plot_pareto_front(list_fi, t, 
                                                    num_points_sample,
                                                    extra= pareto_front,
                                                    pop=list_fi_pop if plot_population else None,
                                                    plot_dataset=plot_dataset,
                                                    elev=elev, azim=azim, legend=legend,
                                                    label=label, images_store_path=images_store_path)
                        

                x_t = x_t.detach()
                pbar.set_postfix({
                        "Points": num_optimal_points,
                        })
                pbar.update(1)
        if self.verbose:
            print(f"END sampling !")

        comp_time = time() - time_start
        
        if self.mode == "offline":
            pf_points = pf_points.detach()
            pf_points = self.offline_denormalization(pf_points,
                                                     self.X_meanormin, 
                                                     self.X_stdormax)
            if self.problem.is_discrete:
                _, dim, n_classes = tuple(pf_points.shape)
                pf_points = pf_points.reshape(-1, dim, n_classes)
                pf_points = offdata_to_integers(pf_points)
            if self.problem.is_sequence:
                pf_points = offdata_to_integers(pf_points)
            if self.problem.has_bounds():
                self.problem.xl, self.problem.xu = self.problem.original_bounds

        res_x = pf_points.detach().cpu().numpy()
        res_y = None
        if evaluate_final and (not is_pass_function(self.problem._evaluate)):
            res_y = self.problem.evaluate(pf_points).detach().cpu().numpy()
            visible_masks = np.ones(len(res_y))
            visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
            visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
            res_x = res_x[np.where(visible_masks == 1)[0]]
            res_y = res_y[np.where(visible_masks == 1)[0]]
            if save_results:
                np.save(samples_store_path + name + "_y.npy", res_y)
                hv = HV(ref_point=self.problem.ref_point)
                hv_value = hv(res_y)
                hv_results = {
                    "ref_point": self.problem.ref_point,
                    "hypervolume": hv_value,
                    "computation_time": comp_time
                }
                with open(samples_store_path + name + "_hv_results.pkl", "wb") as f:
                    pickle.dump(hv_results, f)
                if self.verbose:
                    print(f"Hypervolume: {hv_value} for seed {self.seed}")
                    print("---------------------------------------")
                    # Print computation time
                    convert_seconds(comp_time)
                    print(datetime.datetime.now())

        if save_results:
            np.save(samples_store_path + name + "_x.npy", res_x)
            
        return res_x, res_y


    def train_surrogate(self, 
                        X, y,
                        val_ratio=0.1, 
                        batch_size=32,
                        lr=1e-3, 
                        lr_decay=0.95, 
                        n_epochs=200):

        set_seed(self.seed) 
        self.surrogate_model = self.get_surrogate()
        if self.surrogate_given:
            return self.train_surrogate_user_defined(X, y)
        
         # Train the surrogate model
        if self.mode == "bayesian":
            self.surrogate_model.fit(X, y)
        elif self.mode == "offline":
            n_obj = y.shape[1]
            tkwargs = {"device": self.device, "dtype": torch.float32}
            self.surrogate_model.set_kwargs(**tkwargs)

            trainer_func = SingleModelBaseTrainer

            for which_obj in range(n_obj):

                y0 = y[:, which_obj].clone().reshape(-1, 1)

                trainer = trainer_func(
                    model=list(self.surrogate_model.obj2model.values())[which_obj],
                    which_obj=which_obj,
                    args={
                        "proxies_lr": lr,
                        "proxies_lr_decay": lr_decay,
                        "proxies_epochs": n_epochs,
                        "device": self.device,
                        "verbose": self.verbose,
                    },
                )

                (train_loader, val_loader) = offdata_get_dataloader(
                    X,
                    y0,
                    train_ratio=(
                        1 - val_ratio
                    ),
                    batch_size=batch_size,
                )

                trainer.launch(train_loader, val_loader)
            # Load the proxies
            self.surrogate_model = []
            for i in range(n_obj):
                classifier = SingleModel(
                    input_size=self.problem.n_var,
                    which_obj=i,
                    device=self.device,
                    hidden_size=[2048, 2048],
                    save_dir=self.proxies_store_path,
                    save_prefix=f"MultipleModels-Vallina-{self.problem.__class__.__name__}-{self.seed}",
                )
                classifier.load()
                classifier = classifier.to(self.device)
                classifier.eval()
                self.surrogate_model.append(classifier)
        else:
            raise ValueError(f"Mode {self.mode} does not support surrogate model!")

    def train_surrogate_user_defined(self, X, y):
        """ 
        Train the user-defined surrogate model.
        If self.mode == "offline", the train_func should return a list of trained surrogate models,
        one for each objective.
        If self.mode == "bayesian", the train_func should return a single trained surrogate model for all objectives.
        
        Parameters
        ----------
        train_func : function
            A function that takes X, y as input and returns a trained surrogate model.
        **kwargs : dict
            Additional keyword arguments to pass to the train_func.
        -----------
        """
        self.surrogate_model = self.train_func_surrogate(X, y)

    def get_surrogate(self):
        if self.surrogate_given:
            return self.surrogate_model
        else:
            if self.mode == "bayesian":
                return GaussianProcess(self.problem.n_var, 
                                        self.problem.n_obj, 
                                        nu=5)
            elif self.mode == "offline":
                os.makedirs(self.proxies_store_path, exist_ok=True)
                return MultipleModels(
                                        n_dim=self.problem.n_var,
                                        n_obj=self.problem.n_obj,
                                        train_mode="Vallina",
                                        device=self.device,
                                        hidden_size=[2048, 2048],
                                        save_dir=self.proxies_store_path,
                                        save_prefix=f"MultipleModels-Vallina-{self.problem.__class__.__name__}-{self.seed}",
                                    )
            else:
                raise ValueError(f"Mode {self.mode} does not support surrogate model!")

    def one_spread_sampling_step(
        self,
        x_t, 
        num_points_sample,
        t, beta_t, alpha_bar_t, rho_scale_gamma,
        nu_t, eta_init, num_inner_steps, lr_inner, free_initial_h,
        use_sigma=False, kernel_sigma=1.0, strict_guidance = False,
        max_backtracks=100, point_n0=None, optimizer_n0=None,
    ):

        # Create a tensor of timesteps with shape (num_points_sample, 1)
        t_tensor = torch.full(
            (num_points_sample,),
            t,
            device=self.device,
            dtype=torch.float32,
        )
        # Compute objective values
        obj_values = self.objective_functions(x_t)
    
        # Conditioning information
        c = obj_values
        with torch.no_grad():
            predicted_noise = self.model(x_t, 
                                    t_tensor / self.timesteps, 
                                    c)
        
        g_w = None
        if strict_guidance:
            #### If strict_guidance, get new perturbation based on the MGD direction of 
            # a single initialized point. Otherwise, a random perturbation is used.
            if self.mode in ["online", "offline"]:
                list_fi_n0 = self.objective_functions(point_n0).split(1, dim=1)
                list_grad_i_n0 = []
                for fi_n0 in list_fi_n0:
                    fi_n0.sum().backward(retain_graph=True)
                    grad_i_n0 = point_n0.grad.detach().clone()
                    point_n0.grad.zero_()
                    grad_i_n0 = torch.nn.functional.normalize(grad_i_n0, dim=0)
                    list_grad_i_n0.append(grad_i_n0)
            else:
                list_grad_i_n0 = self.objective_functions(point_n0, get_grad_mobo=True)["dF"]
                for i in range(len(list_grad_i_n0)):
                    # X.grad.zero_()
                    grad_i_n0 = list_grad_i_n0[i]
                    # Normalize gradients
                    grad_i_n0 = torch.nn.functional.normalize(grad_i_n0, dim=0)
                    list_grad_i_n0[i] = grad_i_n0

            optimizer_n0.zero_grad()
            mth = "mgda"
            if self.problem.n_ieq_constr + self.problem.n_eq_constr > 0:
                mth = "pmgda"
            g_w = self.get_target_dir(list_grad_i_n0, mth=mth, x=point_n0)
            point_n0.grad = g_w
            optimizer_n0.step()

        #### Reverse diffusion step
        sqrt_1_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-6))
        sqrt_1_minus_beta_t = torch.sqrt(torch.clamp(1 - beta_t, min=1e-6))
        mean = (1 / sqrt_1_minus_beta_t) * (
            x_t - (beta_t / sqrt_1_minus_alpha_t) * (predicted_noise)
        )
        std_dev = torch.sqrt(beta_t)
        z = torch.randn_like(x_t) if t > 0 else 0.0  # No noise for the final step
        x_t = mean + std_dev * z

        #### Pareto Guidance step
        if self.problem.need_repair:
            x_t.data = self.repair_bounds(x_t.data.clone())
        X = x_t.clone().detach().requires_grad_()
        if self.mode in ["online", "offline"]:
            list_fi = self.objective_functions(X).split(1, dim=1)
            list_grad_i = []
            for fi in list_fi:
                fi.sum().backward(retain_graph=True)
                grad_i = X.grad.detach().clone()
                grad_i = torch.nn.functional.normalize(grad_i, dim=0)
                list_grad_i.append(grad_i)
                X.grad.zero_()
        elif self.mode == "bayesian":
            list_grad_i = self.objective_functions(X, get_grad_mobo=True)["dF"]
            for i in range(len(list_grad_i)):
                # X.grad.zero_()
                grad_i = list_grad_i[i]
                # Normalize gradients
                grad_i = torch.nn.functional.normalize(grad_i, dim=0)
                list_grad_i[i] = grad_i
        else:
            raise ValueError(f"Mode {self.mode} not recognized!")
                
        grads = torch.stack(list_grad_i, dim=0) # (m, N, d)
        grads_copy = torch.stack(list_grad_i, dim=1).detach() # (N, m, d)
        mth = "mgda"
        if self.problem.n_ieq_constr + self.problem.n_eq_constr > 0:
            mth = "pmgda"
        g_x_t_prime = self.get_target_dir(list_grad_i, mth=mth, x=X)

        eta = self.mgd_armijo_step(
            x_t, g_x_t_prime,
            obj_values, grads_copy, # (N, m, d)
            eta_init=eta_init,
            max_backtracks=max_backtracks
        )

        h_tilde = self.solve_for_h(
            x_t,
            g_x_t_prime,
            grads,
            g_w,
            eta=eta,
            nu_t=nu_t,
            sigma=kernel_sigma,
            free_initial_h=free_initial_h,
            use_sigma=False,
            num_inner_steps=num_inner_steps,
            lr_inner=lr_inner,
            rho_scale_gamma=rho_scale_gamma
        )

        h_tilde = torch.nan_to_num(h_tilde, 
                                   nan=torch.nanmean(h_tilde), 
                                   posinf=0.0, 
                                   neginf=0.0)

        x_t = x_t - eta * h_tilde

        return x_t
            
    def solve_for_h(
        self,
        x_t_prime,
        g_x_t_prime,
        grads,
        g_w,
        eta,
        nu_t,
        sigma=1.0,
        use_sigma=False,
        num_inner_steps=10,
        lr_inner=1e-2,
        strict_guidance=False,
        free_initial_h=False,
        rho_scale_gamma=0.9
    ):
        """        
        Returns:
            h_tilde: Optimized h (Tensor of shape (batch_size, n_var)).
        """

        x_t_h = x_t_prime.clone().detach()
        g = g_x_t_prime.clone().detach()

        if strict_guidance:
            g_targ = g_w.clone().detach()
        else:
            g_targ = torch.randn((1, g.shape[1]), device=g.device)

        # Initialize h
        if not free_initial_h:
            h = g_x_t_prime.clone().detach().requires_grad_() # initialize at g
        else:
            h = torch.zeros_like(g, requires_grad=False) + 1e-6  # or as a free parameter
        h = h.requires_grad_()

        optimizer_inner = optim.Adam([h], lr=lr_inner)

        for step in range(num_inner_steps):

            gtarg_scaled = self.adaptive_scale_delta_vect(
                    h, g_targ, grads, gamma=rho_scale_gamma
                )

            # Alignment term: maximize <g, h>
            # To maximize L, we minimize -L:
            alignment = -torch.mean(torch.sum(g * h, dim=-1))
            
            # Update points:
            x_t_h = x_t_h - eta * (h + gtarg_scaled)
            if self.problem.need_repair:
                x_t_h.data = self.repair_bounds(
                    x_t_h.data
                )
            # Map the updated points to the objective space
            F_ = self.objective_functions(x_t_h)
            # Compute repulsion loss to encourage diversity
            if use_sigma:
                rep_loss = self.repulsion_loss(F_, sigma)
            else:
                rep_loss = self.repulsion_loss(F_, use_sigma=False)

            # Our composite objective L is:
            loss = alignment + nu_t * rep_loss

            optimizer_inner.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_inner.step()

        h_tilde = h + gtarg_scaled  # This is h_tilde in the paper

        return h_tilde.detach()

    def get_training_data(self, problem, num_samples=10000):
        """
        Sample points, using LHS, based on lowest constraint violation
        """
        sampler = LHS()
        # Problem bounds
        xl, xu = problem.xl, problem.xu
        # Draw n_sample candidates in [0,1]^n_var
        Xcand = sampler.do(problem, num_samples).get("X")
        # Scale to actual bounds
        Xcand = xl + (xu - xl) * Xcand
        F = problem.evaluate(Xcand)
        return Xcand, F
    
    def betas_for_alpha_bar(self, T, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param T: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.from_numpy(np.array(betas)).float()

    def cosine_beta_schedule(self, s=0.008):
        """
        Cosine schedule for beta values over timesteps.
        """
        return self.betas_for_alpha_bar(
            self.timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        
    def l_simple_loss(self, predicted_noise, actual_noise):
        return nn.MSELoss()(predicted_noise, actual_noise)

    def get_target_dir(self, grads, mth="mgda", x=None):
        m = len(grads)
        if self.problem.n_ieq_constr + self.problem.n_eq_constr > 0:
            assert mth != "mgda", "MGDA not supported with constraints. Use mth ='pmgda'."
        
        if mth == "mgda":
            """
            Compute the MGDA combined descent direction given a list of gradient tensors.
            All grads are assumed to have the same shape (parameters' shape).
            Returns a tensor of the same shape as each gradient, representing the direction g.
            """
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
        elif mth == "pmgda":
            pre_h_vals=None
            constraint_mtd='pbi'
            SOLVER = PMGDASolver(self.problem, prefs=None, 
                             n_prob=grads[0].shape[0], n_obj=self.problem.n_obj, 
                            verbose=False)
            y = self.objective_functions(x, get_constraint=True)
            if "H" in y:
                pre_h_vals = y["H"].sum(dim=1)
                constraint_mtd='eq'
            elif "G" in y:
                pre_h_vals = y["G"].sum(dim=1)
                print("pre_h_vals.shape:", pre_h_vals.shape)
                constraint_mtd='ineq'
            y = y["F"]
            alphas = SOLVER.compute_weights(x, y, pre_h_vals=pre_h_vals, 
                                            constraint_mtd=constraint_mtd)
            alphas = torch.nan_to_num(alphas, nan=torch.nanmean(alphas),
                                      posinf=0.0, neginf=0.0).split(1, dim=1)
            g = torch.zeros_like(grads[0])
            for weight, grad in zip(alphas, grads):
                g += weight * grad
        else:
            raise ValueError(f"Method {mth} not recognized.")
        return g
            
    def mgd_armijo_step(
        self,
        x_t: torch.Tensor,
        d: torch.Tensor,
        f_old,
        grads, # (N, m, d)
        eta_init=0.9,
        rho=0.9,
        c1=1e-4,
        max_backtracks=100,
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
            f_new = self.objective_functions(trial_x)

            # Armijo test (vectorised over objectives)
            # f_new <= f_old + c1 * eta * grad_dot  (element-wise)
            ok = (f_new <= f_old[improve] + c1 * eta[improve, None] * grad_dot[improve]).all(dim=1)

            # Update masks and step sizes
            eta[improve] = torch.where(ok, eta[improve], rho * eta[improve])
            improve_mask = improve.clone()
            improve[improve_mask] = ~ok

        return eta[:, None]
    
    def adaptive_scale_delta_vect(
        self,
        g: torch.Tensor, 
        delta_raw: torch.Tensor, 
        grads: torch.Tensor, 
        gamma: float = 0.9
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
   
        # Compute alpha_{i,j} = ∇f_j(x_i)^T g_i
        #    shape of alpha: [n_points, m]
        alpha = torch.einsum("j i d, i d -> i j", grads, g)

        # Compute beta_{i,j} = ∇f_j(x_i)^T delta_raw_i
        #    shape of beta: [n_points, m]
        beta = torch.einsum("j i d, i d -> i j", grads, delta_raw)

        # We only need to restrict rho_i if alpha_{i,j} > 0 and beta_{i,j} < 0.
        #    Because for alpha + rho*beta to stay > 0, we need
        #        rho < alpha / -beta
        #    when beta<0 and alpha>0.
        mask = (alpha > 0.0) & (beta < 0.0)

        # Prepare an array of ratios = alpha / -beta, default +∞
        ratio = torch.full_like(alpha, float("inf"))

        # Where mask is True, compute ratio_{i,j}
        ratio[mask] = alpha[mask] / (-beta[mask])  # must remain below this

        # For each point i, we pick rho_i = gamma * min_j ratio[i,j].
        #    If the min is +∞ => no constraints => set rho_i=1.0
        ratio_min, _ = ratio.min(dim=1)  # [n_points]
        rho = gamma * ratio_min
        # If ratio_min == +∞ => no constraint => set rho_i=1.
        inf_mask = torch.isinf(ratio_min)
        rho[inf_mask] = 1.0

        # Scale delta_raw by rho_i
        delta_scaled = delta_raw * rho.unsqueeze(1)  # broadcast along dim

        return delta_scaled
    
    def repair_bounds(self, x):
        """
        Clips a tensor x of shape [N, d] such that for each column j:
            x[:, j] is clipped to be between xl[j] and xu[j].

        Parameters:
            x (torch.Tensor): Input tensor of shape [N, d].

        Returns:
            torch.Tensor: The clipped tensor with the same shape as x.
        """

        xl, xu = self.problem.bounds()[0], self.problem.bounds()[1]
        lower = xl.detach().clone().to(x.device)
        upper = xu.detach().clone().to(x.device)

        if self.problem.global_clamping:
            return torch.clamp(x.data.clone(), min=lower.min(), max=upper.max())
        else:
            return torch.clamp(x.data.clone(), min=lower, max=upper)
        
    def repulsion_loss(self, 
                       F_, 
                       sigma=1.0, 
                       use_sigma=False):
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
            tensor = dist_sq.detach().flatten()
            tensor_max = tensor.max()[None]
            median_dist = (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.0
            s = median_dist / math.log(n)
            repulsion = torch.exp(-dist_sq / 5e-6 * s)

        # Normalize by the number of pairs
        loss = (2/(n*(n-1))) * repulsion.sum()
        return loss

    def eps_dominance(self, Obj_space, alpha=0.0):
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
        self,
        points_pred=None, 
        keep_shape=True, 
        indx_only=False, 
        p_front=None
    ):
        if not indx_only and points_pred is None:
            raise ValueError("points_pred cannot be None when indx_only is False.")
        if points_pred is not None:
            pf_points = copy.deepcopy(points_pred.detach())
            p_front = self.objective_functions(pf_points).detach().cpu().numpy()
        else:
            assert p_front is not None, "p_front must be provided if points_pred is None."
        
        if self.mode in ["online", "offline"]:
            PS_idx = self.eps_dominance(p_front)
        elif self.mode == "bayesian":
            N = points_pred.shape[0]
            # 1) Predict dominance
            label_matrix, _ = nn_predict_dom_intra(points_pred.detach().cpu().numpy(), 
                                                            self.dominance_classifier, 
                                                            self.device)
            # 2) Find non‑dominated indices
            PS_idx = [
                i for i in range(N)
                if not any(label_matrix[j, i] == 2 for j in range(N))
            ]
        else:
            raise ValueError(f"Mode {self.mode} not recognized!")
            
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


    def crowding_distance(self, points):
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
        self,
        points: torch.Tensor,
        n: int,
        top_frac: float = 0.9
    ) -> torch.Tensor:
        """
        Selects the top `n` points from `points` based on crowding distance.

        Returns:
            torch.Tensor: The best subset of points (shape [n, D]).
        """
        
        if self.mode in ["online", "offline"]:
            if len(points) <= n:
                final_idx = torch.randperm(points.size(0))
            else:
                full_p_front = self.objective_functions(points)
                distances = self.crowding_distance(full_p_front)
                top_indices = torch.topk(distances, n).indices
                final_idx = top_indices[torch.randperm(top_indices.size(0))]
        else:
            N = points.shape[0]
            # 1) Predict dominance
            label_matrix, conf_matrix = nn_predict_dom_intra(points.detach().cpu().numpy(), 
                                                            self.dominance_classifier, 
                                                            self.device)
            # 2) Find non‑dominated indices
            nondom_inds = [
                i for i in range(N)
                if not any(label_matrix[j, i] == 2 for j in range(N))
            ]

            # --- CASE A: too many non‑dominated → pick top-n by crowding ---
            if len(nondom_inds) > n:
                # Evaluate objectives on just the non-dominated set
                pts_nd = points[nondom_inds].to(self.device)
                Y_t  = self.objective_functions(pts_nd)

                # Compute crowding distances and select top-n
                distances = self.crowding_distance(Y_t)
                topk      = torch.topk(distances, n).indices.tolist()

                selected_nd = [nondom_inds[i] for i in topk]

                # Shuffle before returning
                perm = torch.randperm(n, device=points.device)
                final_idx = torch.tensor(selected_nd, device=points.device)[perm]
                return points[final_idx] #.detach().cpu().numpy()

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

            # 4) Sort full points by (dom_count asc, avg_conf desc)
            idxs = list(range(N))
            idxs.sort(key=lambda i: (dom_counts[i], -avg_conf[i]))

            # 5) Keep only top top_frac% of that ranking
            k90   = int(np.floor(top_frac * N))
            top90 = idxs[:k90]

            # 6) Evaluate
            pts90 = points[top90]
            Y_t  = self.objective_functions(pts90)

            # 7) Crowding distance & pick as many as needed to reach n
            distances = self.crowding_distance(Y_t)
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
            perm = torch.randperm(len(final_inds), device=points.device)
            final_idx = torch.tensor(final_inds, device=points.device)[perm]

        return points[final_idx]
    
    def plot_pareto_front(self,
                          list_fi, 
                          t,
                          num_points_sample,
                          extra=None,
                          label=None,
                          plot_dataset=False,
                          pop=None,
                          elev=30, azim=45, legend=False,
                          images_store_path="./images_dir/"):
        name = (
            "spread"
            + "_"
            + self.problem.__class__.__name__
            + "_"
            + f"T={self.timesteps}"
            + "_"
            + f"N={num_points_sample}"
            + "_" 
            + f"t={t}"
            + "_"
            + f"seed={self.seed}"
            + "_"
            + self.mode
        )
        if label is not None:
            name += f"_{label}"

        if len(list_fi) > 3:
            return None

        elif len(list_fi) == 2:
            fig, ax = plt.subplots()
            if plot_dataset and (self.dataset) is not None:
                _, Y = self.dataset
                # Denormalize the data
                Y = self.offline_denormalization(Y,
                                                self.y_meanormin,
                                                self.y_stdormax)
                ax.scatter(Y[:, 0], Y[:, 1],
                            c="violet", s=5, alpha=1.0,
                            label="Training Data")
                            
            if extra is not None:
                f1, f2 = extra
                ax.scatter(f1, f2, c="yellow", s = 5, alpha=1.0,
                            label="Pareto Optimal")
                            
            if pop is not None:
                f_pop1, f_pop2 = pop
                ax.scatter(f_pop1, f_pop2, c="blue", s=10, alpha=1.0,
                            label="Gen Population")
                
            f1, f2 = list_fi
            ax.scatter(f1, f2, c="red", s=10, alpha=1.0,
                        label="Gen Optimal")
            
            ax.set_xlabel("$f_1$", fontsize=14)
            ax.set_ylabel("$f_2$", fontsize=14)
            ax.set_title(f"Reverse Time Step: {t}", fontsize=14)
            ax.text(
                -0.17, 0.5,
                self.problem.__class__.__name__.upper() + f"({self.mode})",
                transform=ax.transAxes,      
                va='center',
                ha='center',
                rotation='vertical',
                fontsize=20,
                fontweight='bold'
            )

        elif len(list_fi) == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            
            if plot_dataset and (self.dataset is not None):
                _, Y = self.dataset
                # Denormalize the data
                Y = self.offline_denormalization(Y,
                                                self.y_meanormin,
                                                self.y_stdormax)
                ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2],
                           c="violet", s=5, alpha=1.0,
                           label="Training Data")
            
            if extra is not None:
                f1, f2, f3 = extra
                ax.scatter(f1, f2, f3, c="yellow", s = 5, alpha=0.05,
                           label="Pareto Optimal")
                        
            if pop is not None:
                f_pop1, f_pop2, f_pop3 = pop
                ax.scatter(f_pop1, f_pop2, f_pop3, c="blue", s=10, alpha=1.0,
                           label="Gen Population")
                
            f1, f2, f3 = list_fi
            ax.scatter(f1, f2, f3, c="red", s = 10, alpha=1.0,
                       label="Gen Optimal")
            
            ax.set_xlabel("$f_1$", fontsize=14)
            ax.set_ylabel("$f_2$", fontsize=14)
            ax.set_zlabel("$f_3$", fontsize=14)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"Reverse Time Step: {t}", fontsize=14)
            ax.text2D(
                -0.17, 0.5,
                self.problem.__class__.__name__.upper() + f"({self.mode})",
                transform=ax.transAxes,
                va='center',
                ha='center',
                rotation='vertical',
                fontsize=20,
                fontweight='bold'
            )

        img_dir = f"{images_store_path}/{self.problem.__class__.__name__}_{self.mode}"
        if label is not None:
            img_dir += f"_{label}"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if legend:
            plt.legend(fontsize=12)
        
        plt.savefig(
                f"{img_dir}/{name}.jpg",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close()

    def create_video(self, image_folder, output_video,
                            total_duration_s=20.0,
                            first_transition_s=2.0,
                            fps=30,
                            extensions=("*.jpg", "*.png", "*.jpeg", "*.bmp")):
        """Create a video from images in `image_folder`, sorted by t=... in filename.
        The first transition (first->second image) lasts `first_transition_s` seconds.
        The remaining transitions share the remaining time equally.
        The output video has total duration `total_duration_s` seconds at `fps` frames per second.
        """

        # Collect and sort by t=... (descending)
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(image_folder, ext)))
        if not paths:
            raise RuntimeError(f"No images found in {image_folder}")

        t_pat = re.compile(r"t=(\d+)")
        def t_val(p):
            m = t_pat.search(p)
            return int(m.group(1)) if m else -1

        paths.sort(key=lambda p: t_val(p), reverse=True)
        N = len(paths)
        if N < 2:
            raise RuntimeError("Need at least two images for a transition.")

        # Read first to get size
        first_img = cv2.imread(paths[0])
        if first_img is None:
            raise RuntimeError(f"Cannot read: {paths[0]}")
        h, w = first_img.shape[:2]
        size = (w, h)

        # Prepare writer
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, float(fps), size)

        # Frame budget
        F_total = int(round(total_duration_s * fps))
        F_first = int(round(first_transition_s * fps))
        F_first = max(0, min(F_first, F_total))  # clamp just in case
        self.frames_written = 0

        def write_transition(img1, img2, d):
            """Blend img1->img2 over d frames; d==0 means hard cut; d==1 means single frame of img2."""
            if d <= 0:
                return
            img1 = cv2.resize(img1, size)
            img2 = cv2.resize(img2, size)
            if d == 1:
                writer.write(img2); self.frames_written += 1
                return
            for j in range(d):
                alpha = j / (d - 1)  # includes endpoints: j=0 -> img1, j=d-1 -> img2
                frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
                writer.write(frame); self.frames_written += 1

        # Load all images (resized) once to avoid repeated disk I/O
        imgs = []
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                im = first_img.copy()
            imgs.append(cv2.resize(im, size))

        # 1) First transition: fixed 2 seconds (or less if total is tiny)
        write_transition(imgs[0], imgs[1], F_first)

        # 2) Remaining transitions share remaining frames
        remaining_frames = F_total - self.frames_written
        remaining_transitions = max(0, N - 2)

        if remaining_transitions > 0:
            # Distribute remaining frames across the remaining transitions.
            # Some transitions may get 0 or 1 frame (hard/near-hard cut) if time is tight.
            base = 0 if remaining_transitions == 0 else remaining_frames // remaining_transitions
            extra = 0 if remaining_transitions == 0 else remaining_frames % remaining_transitions
            # Ensure we don't exceed the total budget:
            for i in range(remaining_transitions):
                d = base + (1 if i < extra else 0)
                write_transition(imgs[i + 1], imgs[i + 2], d)

        # 3) If we still have spare frames (due to rounding), hold last frame
        last = imgs[-1]
        while self.frames_written < F_total:
            writer.write(last); self.frames_written += 1

        writer.release()
        print(f"✅ Saved: {output_video}  | duration={total_duration_s}s, fps={fps}, frames={F_total}")
