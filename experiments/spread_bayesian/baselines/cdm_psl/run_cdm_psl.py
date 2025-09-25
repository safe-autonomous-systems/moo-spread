import argparse
import numpy as np
import torch
import pickle
import sys
from problem import get

import json
import os
import pickle

import random

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.problems import get_problem
from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from pymoo.config import Config
Config.warnings['not_compiled'] = False

from evolution.utils import *
from learning.model_init import *
from learning.model_update import *
from learning.prediction import *
from diffusion import gen_offspring
from utils import sbx, environment_selection, pm_mutation, sort_population, convert_seconds, set_seed

from tqdm import tqdm
import datetime
from time import time, sleep

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
        "branincurrin",
    ],
    help="Specify the problem set to run.",
)
parser.add_argument(
    "--switch_operator",
    type=int,
    choices=[0, 1],
    default=1,
    help="0 = off, 1 = on"
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

# PSL parameters
# Maximum number of function evaluations
max_fes = 100 
# number of learning steps
n_steps = int(max_fes / n_sample)
coef_lcb = 0.1
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000
device = "cuda" 
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

# --------------------------------------------------------------------------------------------------

# Update historical data and calculate reference point
def get_ref_point_method2(Y, history_Y, k, scale_factor=1.1, quantile=0.95):
    history_Y.append(Y)
    if len(history_Y) > k:
        history_Y.pop(0)
    all_Y = np.vstack(history_Y)  # Combine historical data
    nds = NonDominatedSorting()
    idx_nds = nds.do(all_Y)
    Y_nds = all_Y[idx_nds[0]]  # Get non-dominated individuals
    quantile_values = np.quantile(Y_nds, quantile, axis=0)
    return scale_factor * quantile_values


for test_ins in ins_list:
    if args.switch_operator:
        print("Switching operator: ON")
    else:
        print("Switching operator: OFF")
    t0_task = time()
    
    if args.switch_operator:
        sfix = "switch"
    else:
        sfix = "pure"
        
    name_t = (
            "cdm_psl"
            + "_"
            + test_ins
            + "_K"
            + str(n_steps)
            + "_FE"
            + str(max_fes)
            + "_" + sfix
        )
    
    log_results = "logs/saved_hvs/"
    
    if not (os.path.exists(log_results)):
        os.makedirs(log_results)
    
    # Get problem info
    bounds = None
    ref_point = None
    hv_all_value = np.zeros([n_run, n_steps + 1])
    if test_ins.startswith("zdt"):
        problem = get_problem(test_ins, n_var=20)
        lbound = torch.from_numpy(problem.xl).float()
        ubound = torch.from_numpy(problem.xu).float()
    elif test_ins.startswith("dtlz"):
        problem = get_problem(test_ins, n_var=20, n_obj=3)
        lbound = torch.from_numpy(problem.xl).float()
        ubound = torch.from_numpy(problem.xu).float()
    else:
        problem, bounds, ref_point = get(test_ins)
        lbound, ubound = bounds
            
    n_dim = problem.n_var
    n_obj = problem.n_obj

    if ref_point is None:
        ref_point = dic[test_ins]
    
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
    # Repeatedly run the algorithm n_run times
    for run_iter in range(args.start_run, n_run):

        print("Independent run No. ", run_iter + 1)
        print(datetime.datetime.now())
        print("--------------------------------------------------")
        set_seed(ALL_SEED[run_iter])
        
        x_init = lhs(n_dim, n_init)
        y_init = problem.evaluate(x_init)
        p_rel_map, s_rel_map = init_dom_rel_map(300)

        p_model = init_dom_nn_classifier(
            x_init, y_init, p_rel_map, pareto_dominance, problem
        )  # init Pareto-Net
        evaluated = len(y_init)

        X = x_init
        Y = y_init

        hv = HV(ref_point=np.array(ref_point))
        hv_value = hv(Y)
        hv_all_value[run_iter, 0] = hv_value

        z = torch.zeros(n_obj )

        # Counter for tracking iterations since last switch
        iteration_since_switch = 0

        # Parameters for switching methods
        hv_change_threshold = 0.05  # Threshold for HV value change
        hv_history_length = 3  # Number of recent iterations to consider
        hv_history = []  # Store recent HV values
        use_diffusion = True

        # Initialize list to store historical data
        history_Y = []
        i_step = 0
        
        with tqdm(
            total=n_steps,
            desc=f"CDM_PSL {test_ins}",
            unit="k",
        ) as pbar:

            # while evaluated < n_init + max_fes:
            while i_step < n_steps:
                    transformation = StandardTransform([0, 1])
                    transformation.fit(X, Y)
                    X_norm, Y_norm = transformation.do(X, Y)
                    
                    ## Train surrogate model
                    surrogate_model = GaussianProcess(n_dim, n_obj, nu=5)
                    surrogate_model.fit(X_norm, Y_norm)
                    
                    ## Data Extraction (Algorithm 2)
                    _, index = environment_selection(Y, len(X) // 3) ###### (!)
                    real = X[index, :]
                    label = np.zeros((len(Y), 1))
                    label[index, :] = 1
                    

                    nds = NonDominatedSorting()
                    idx_nds = nds.do(Y_norm)

                    Y_nds = Y_norm[idx_nds[0]]
                    PopDec = real
                    PopDec_dom_labels, PopDec_cfs = nn_predict_dom_intra(
                        PopDec, p_model, device
                    )
                    sorted_pop = sort_population(PopDec, PopDec_dom_labels, PopDec_cfs)
                    number_of_dv = sorted_pop.shape[1]
                    
                    if use_diffusion:
                        # Generate offspring using Diffusion
                        X_psl = gen_offspring(
                            sorted_pop, number_of_dv, surrogate_model, [lbound, ubound]
                        )
                    else:
                        # Generate offspring using SBX
                        rows_to_take = int(1 / 3 * sorted_pop.shape[0])
                        offspringA = sorted_pop[:rows_to_take, :]

                        if len(offspringA) % 2 == 1:
                            offspringA = offspringA[:-1]
                        new_pop = np.empty((0, n_dim))
                        for _ in range(n_candidate):
                            result = sbx(offspringA, eta=15)
                            new_pop = np.vstack((new_pop, result))

                        X_psl = new_pop
                        
                    pop_size_used = X_psl.shape[0]
                        
                    
                    # Mutate the new offspring
                    X_psl = pm_mutation(X_psl, [lbound, ubound])

                    Y_candidate_mean = surrogate_model.evaluate(X_psl)["F"]
                    # print("Y_candidate_mean: ", Y_candidate_mean.shape)
                    Y_candidata_std = surrogate_model.evaluate(X_psl, std=True)["S"]

                    rows_with_nan = np.any(np.isnan(Y_candidate_mean), axis=1)
                    Y_candidate_mean = Y_candidate_mean[~rows_with_nan]
                    Y_candidata_std = Y_candidata_std[~rows_with_nan]
                    X_psl = X_psl[~rows_with_nan]

                    Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
                    Y_candidate_mean = Y_candidate

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

                    X_candidate = X_psl
                    X_new = X_candidate[best_subset_list]
                    Y_new = problem.evaluate(X_new)
                    
                    Y_new = torch.tensor(Y_new).to(device)
                    X_new = torch.tensor(X_new).to(device)

                    X = np.vstack([X, X_new.detach().cpu().numpy()])
                    Y = np.vstack([Y, Y_new.detach().cpu().numpy()])
                    hv = HV(ref_point=np.array(ref_point))
                    hv_value = hv(Y)
                    hv_all_value[run_iter, 1 + i_step] = hv_value
                    
                    i_step += 1
                    
                    rows_with_nan = np.any(np.isnan(Y), axis=1)
                    X = X[~rows_with_nan, :]
                    Y = Y[~rows_with_nan, :]

                    update_dom_nn_classifier(
                        p_model, X, Y, p_rel_map, pareto_dominance, problem
                    )
                    
                    hv_text = f"{hv_value:.4e}" 
                    evaluated = evaluated + n_sample


                    if args.switch_operator:
                        # Print current operator
                        if use_diffusion:
                            operator_text = "Diffusion"
                        else:
                            operator_text = "SBX"
                        

                        # Calculate approximate HV
                        ref_point_method2 = get_ref_point_method2(Y, history_Y, k=3)
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
                                    use_diffusion = not use_diffusion
                                    iteration_since_switch = 0  # Reset counter
                            else:
                                iteration_since_switch += 1  # If already switched, increment counter

                        pbar.set_postfix({"HV": hv_text, "Operator": operator_text, "Population": pop_size_used, "Num Points": evaluated})
                    else:
                        pbar.set_postfix({"HV": hv_text, "Population": pop_size_used, "Num Points": evaluated})
                    pbar.update(1)
                    
        # Save X and Y for the current run
        res_file_dir = "logs/pf_results/" + name_t
        if not (os.path.exists(res_file_dir)):
            os.makedirs(res_file_dir)
        np.save(res_file_dir + "/" + "pf_res" + str(run_iter) + "_x.npy", X)
        np.save(res_file_dir + "/" + "pf_res" + str(run_iter) + "_y.npy", Y)
                    
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
            
    T_task = time() - t0_task
    convert_seconds(T_task)