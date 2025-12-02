"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""

import os
import argparse
import numpy as np
import torch
import pickle
import sys
from problem import get

import random

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.problems import get_problem
from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from pymoo.config import Config
Config.warnings['not_compiled'] = False

from utils import set_seed, convert_seconds, pm_mutation

from tqdm import tqdm
import datetime
from time import time, sleep

from model import ParetoSetModel

# Set up command line argument parsing
parser = argparse.ArgumentParser(
    description="Run experiments with specified problem set."
)
parser.add_argument(
    "--prob",
    type=str,
    default="dtlz2",
    choices=[
        "zdt1",
        "zdt2",
        "zdt3",
        "dtlz2",
        "dtlz5",
        "dtlz7",
        "carside",
        "penicillin",
        "branincurrin"
    ],
    help="Specify the problem set to run.",
)

# Parse command line arguments
args = parser.parse_args()
# Set problem set
ins_list = [args.prob]

# number of independent runs
n_run = 5
ALL_SEED = [1000, 2000, 3000, 4000, 5000]
# number of initialized solutions
n_init = 100 
# number of iterations, and batch size per iteration
n_iter = 20 
n_sample = 5 
max_fes = 100
n_algo_steps = int(max_fes / n_sample)

# PSL 
# number of learning steps
n_steps = 250 # 1000 
# number of sampled preferences per step
n_pref_update = 10
# coefficient of LCB
coef_lcb = 0.1
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000 
# number of optional local search
n_local = 1
# device
device = 'cuda'
# -----------------------------------------------------------------------------

hv_list = {}

# Ref point
dic = {
    "zdt1": [0.9994, 6.0576],
    "zdt2": [0.9994, 6.8960],
    "zdt3": [0.9994, 6.0571],
    "dtlz2": [2.8390, 2.9011, 2.8575],
    "dtlz5": [2.6672, 2.8009, 2.8575],
    "dtlz7": [0.9984, 0.9961, 22.8114],
}

for test_ins in ins_list:
    t0_task = time()
    
    name_t = (
            "psl_mobo"
            + "_"
            + test_ins
            + "_K"
            + str(n_algo_steps)
            + "_FE"
            + str(max_fes)
        )
    
    log_results = "logs/saved_hvs/"
    
    if not (os.path.exists(log_results)):
        os.makedirs(log_results)
    
    # get problem info
    bounds = None
    ref_point = None
    hv_all_value = np.zeros([n_run, n_algo_steps + 1])
    if test_ins.startswith("zdt"):
        problem = get_problem(test_ins, n_var=20)
    elif test_ins.startswith("dtlz"):
        problem = get_problem(test_ins, n_var=20, n_obj=3)
    else:
        problem, bounds, ref_point = get(test_ins)
    n_dim = problem.n_var
    n_obj = problem.n_obj

    if ref_point is None:
        ref_point = dic[test_ins]
    if bounds is None:
        bounds = [torch.from_numpy(problem.xl).float(), 
                      torch.from_numpy(problem.xu).float()]
        
    print("************************************************************")
    print("Task: ", test_ins)
    print("Device: ", device)
    print("Num Objectives: ", n_obj)
    print("Num Dimensions: ", n_dim)
    print("Reference point: ", ref_point)
    print("Bounds: ", bounds)
    print(datetime.datetime.now())
    print("************************************************************")

    outfile = log_results + name_t + ".pkl"
    
    # repeatedly run the algorithm n_run times
    for run_iter in range(args.start_run, n_run):
        
        print("Independent run No. ", run_iter + 1)
        print(datetime.datetime.now())
        print("--------------------------------------------------")
        set_seed(ALL_SEED[run_iter])
        
        # initialize n_init solutions 
        x_init = lhs(n_dim, n_init)
        y_init = problem.evaluate(x_init)
        evaluated = len(y_init)
        
        X = x_init
        Y = y_init
        
        hv = HV(ref_point=np.array(ref_point))
        hv_value = hv(Y)
        hv_all_value[run_iter, 0] = hv_value
    
        z = torch.zeros(n_obj).to(device)
        
        i_step = 0
        with tqdm(
            total=n_algo_steps,
            desc=f"PSL_MOBO {test_ins}",
            unit="k",
        ) as pbar:
 
            while i_step < n_algo_steps:
                # intitialize the model and optimizer 
                psmodel = ParetoSetModel(n_dim, n_obj)
                psmodel.to(device)
                    
                # optimizer
                optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
            
                # solution normalization
                transformation = StandardTransform([0,1])
                transformation.fit(X, Y)
                X_norm, Y_norm = transformation.do(X, Y) 
                
                # train GP surrogate model 
                surrogate_model = GaussianProcess(n_dim, n_obj, nu = 5)
                surrogate_model.fit(X_norm,Y_norm)
                
                z =  torch.min(torch.cat((z.reshape(1,n_obj),torch.from_numpy(Y_norm).to(device) - 0.1)), axis = 0).values.data
                
                # nondominated X, Y 
                nds = NonDominatedSorting()
                idx_nds = nds.do(Y_norm)
                
                X_nds = X_norm[idx_nds[0]]
                Y_nds = Y_norm[idx_nds[0]]
                
                # t_step Pareto Set Learning with Gaussian Process
                for t_step in range(n_steps):
                    psmodel.train()
                    
                    # sample n_pref_update preferences
                    alpha = np.ones(n_obj)
                    pref = np.random.dirichlet(alpha,n_pref_update)
                    pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
                    
                    # get the current coressponding solutions
                    x = psmodel(pref_vec)
                    x_np = x.detach().cpu().numpy()
                    
                    # obtain the value/grad of mean/std for each obj
                    mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
                    mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)
                    
                    std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
                    std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).to(device)
                    
                    # calculate the value/grad of tch decomposition with LCB
                    value = mean - coef_lcb * std
                    value_grad = mean_grad - coef_lcb * std_grad
                
                    tch_idx = torch.argmax((1 / pref_vec) * (value - z), axis = 1)
                    tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
                    tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update,1) *  value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis = 1) 

                    tch_grad = tch_grad / torch.norm(tch_grad, dim = 1)[:, None]
                    
                    # gradient-based pareto set model update 
                    optimizer.zero_grad()
                    psmodel(pref_vec).backward(tch_grad)
                    optimizer.step()  
                    
                # solutions selection on the learned Pareto set
                psmodel.eval()
                
                # sample n_candidate preferences
                alpha = np.ones(n_obj)
                pref = np.random.dirichlet(alpha,n_candidate)
                pref  = torch.tensor(pref).to(device).float() + 0.0001
        
                # generate correponding solutions, get the predicted mean/std
                X_candidate = psmodel(pref).to(torch.float64)
                X_candidate_np = X_candidate.detach().cpu().numpy()
                X_psl = X_candidate_np
                 
                pop_size_used = X_psl.shape[0]
                
                Y_candidate_mean = surrogate_model.evaluate(X_psl)["F"]
                Y_candidata_std = surrogate_model.evaluate(X_psl, std=True)["S"]

                rows_with_nan = np.any(np.isnan(Y_candidate_mean), axis=1)
                Y_candidate_mean = Y_candidate_mean[~rows_with_nan]
                Y_candidata_std = Y_candidata_std[~rows_with_nan]
                X_psl = X_psl[~rows_with_nan]

                Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
                Y_candidate_mean = Y_candidate
                
                X_candidate_np = X_candidate.detach().cpu().numpy()
                Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
                
                Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
                Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std

                # optional TCH-based local Exploitation 
                if n_local > 0:
                    X_candidate_tch = X_candidate_np
                    z_candidate = z.cpu().numpy()
                    pref_np = pref.cpu().numpy()
                    for j in range(n_local):
                        candidate_mean =  surrogate_model.evaluate(X_candidate_tch)['F']
                        candidate_mean_grad =  surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']
                        
                        candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
                        candidate_std_grad = surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']
                        
                        candidate_value = candidate_mean - coef_lcb * candidate_std
                        candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad
                        
                        candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis = 1)
                        candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)]
                        
                        candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)].reshape(n_candidate,1) * candidate_grad[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)] 
                        candidate_tch_grad +=  0.01 * np.sum(candidate_grad, axis = 1) 
                        
                        X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
                        X_candidate_tch[X_candidate_tch <= 0]  = 0
                        X_candidate_tch[X_candidate_tch >= 1]  = 1  
                        
                    X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])
                    
                    Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
                    Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
                    
                    Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
                
                # greedy batch selection 
                best_subset_list = []
                Y_p = Y_nds
                for b in range(n_sample):
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

                X_candidate = X_candidate_np # X_psl
                X_new = X_candidate[best_subset_list]
                Y_new = problem.evaluate(X_new)
                    
                Y_new = torch.tensor(Y_new).to(device)
                X_new = torch.tensor(X_new).to(device)
                
                X = np.vstack([X, X_new.detach().cpu().numpy()])
                Y = np.vstack([Y, Y_new.detach().cpu().numpy()])
                hv = HV(ref_point=np.array(ref_point))
                hv_value = hv(Y)
                hv_all_value[run_iter, 1 + i_step] = hv_value
     
                rows_with_nan = np.any(np.isnan(Y), axis=1)
                X = X[~rows_with_nan, :]
                Y = Y[~rows_with_nan, :]
                
                hv_text = f"{hv_value:.4e}" 
                evaluated = evaluated + n_sample
                
                i_step += 1
                
                pbar.set_postfix({"HV": hv_text, "Population": pop_size_used, "Num Points": evaluated})
                pbar.update(1)
                
        # Load+update the HV list if existed or saved it.
        if run_iter == 0:
            with open(outfile, "wb") as f:
                pickle.dump(hv_all_value, f)
        else:
            # Load the existing HV list
            with open(outfile, "rb") as f:
                old_hv_all_value = pickle.load(f)
            # Update the HV list with new values
            old_hv_all_value[run_iter, :] = hv_all_value[run_iter, :]
            # Save the updated HV list
            with open(outfile, "wb") as f:
                pickle.dump(old_hv_all_value, f)
                
        # Save X and Y for the current run
        res_file_dir = "logs/pf_results/" + name_t
        if not (os.path.exists(res_file_dir)):
            os.makedirs(res_file_dir)
        np.save(res_file_dir + "/" + "pf_res" + str(run_iter) + "_x.npy", X)
        np.save(res_file_dir + "/" + "pf_res" + str(run_iter) + "_y.npy", Y)

        print("************************************************************")

    T_task = time() - t0_task
    convert_seconds(T_task)

