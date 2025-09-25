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
from bayesian_spread import gen_offspring_via_spread
from bay_ms_utils import * # select_top_n_for_BaySpread, convert_seconds, set_seed
from model_spread import DiTMOO # TransformerDiffusionModel

from tqdm import tqdm
import datetime
from time import time, sleep

# Set up command line argument parsing
parser = argparse.ArgumentParser(
    description="Bayesian-SPREAD"
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
parser.add_argument(
        "--num_blocks",
        default=3,
        type=int,
        help="Number of DiT blocks in the model",
    )
parser.add_argument("--timesteps", type=int, default=25) # 25
parser.add_argument("--batch-size", type=int, default=500)
parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate for DDPM training"
    )
parser.add_argument("--num-epochs", type=int, default=250)
parser.add_argument("--patience", default= 100)
parser.add_argument("--kernel-sigma", default=0.01)
parser.add_argument("--lambda-rep", default=10)
parser.add_argument("--num-inner-steps", default=10)
parser.add_argument("--num_samp_iters", default= 5)

parser.add_argument("--n_init", default= 100, help="Number of initial solutions")
parser.add_argument("--max_fes", default= 100, help="Maximum number of function evaluations")
parser.add_argument("--coef_lcb", default= 0.1, help="Coefficient of LCB")

parser.add_argument("--n_sample", default= 5, help="Batch size per MOBO step")
parser.add_argument(
    "--switch_operator",
    type=int,
    choices=[0, 1],
    default=1,
    help="Like in CDM-PSL: wether to switch the operator or not.")

parser.add_argument("--n_run", default= 5, # 3
                    help="Number of independent runs")
parser.add_argument("--ALL_SEED", default= [1000, 2000, 3000, 4000, 5000], help="List of seeds for independent runs")

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set problem set
ins_list = [args.prob]

assert args.n_run <= len(args.ALL_SEED), "The number of independent runs should be at most the number of seeds."

def set_opt_args(args):

    if args.num_obj == 2:
        args.gamma_scale_delta = 0.9
        args.eta = 0.9
        args.lr_inner = 0.9
 
    else:
        args.gamma_scale_delta = 0.01
        args.eta = 0.9
        args.lr_inner = 5e-4
        
        if args.prob in ["dtlz7",]:
            args.gamma_scale_delta = 0.01
            args.eta = 0.9
            args.lr_inner = 0.9
            
        if args.prob in ["carside"]:
            args.gamma_scale_delta = 0.01
            args.eta = 0.9
            args.lr_inner = 0.9
        
# -----------------------------------------------------------------------------

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

hv_list = {}

dic = {
    "zdt1": [0.9994, 6.0576],
    "zdt2": [0.9994, 6.8960],
    "zdt3": [0.9994, 6.0571],
    "dtlz2": [2.8390, 2.9011, 2.8575],
    "dtlz5": [2.6672, 2.8009, 2.8575],
    "dtlz7": [0.9984, 0.9961, 22.8114],
}

# --------------------------------------------------------------------------------------------------

# number of learning steps
n_steps = int(args.max_fes / args.n_sample) # 20
for test_ins in ins_list:
    t0_task = time()
    
    if args.switch_operator:
        sfix = "switch"
    else:
        sfix = "pure"
        
    name_t = (
            "bay_spread"
            + "_"
            + test_ins
            + "_K"
            + str(n_steps)
            + "_FE"
            + str(args.max_fes)
            + "_" + sfix
        )

    log_results = "logs/saved_hvs/"
    
    if not (os.path.exists(log_results)):
        os.makedirs(log_results)
    
    # Get problem info
    ref_point = None
    hv_all_value = np.zeros([args.n_run, n_steps + 1])
    if test_ins.startswith("zdt"):
        problem = get_problem(test_ins, n_var=20)
        args.xl = torch.from_numpy(problem.xl).float()
        args.xu = torch.from_numpy(problem.xu).float()
    elif test_ins.startswith("dtlz"):
        problem = get_problem(test_ins, n_var=20, n_obj=3)
        args.xl = torch.from_numpy(problem.xl).float()
        args.xu = torch.from_numpy(problem.xu).float()
    else:
        problem, bounds, ref_point = get(test_ins)
        args.xl, args.xu = bounds
    
    if ref_point is None:
        ref_point = dic[test_ins]
    
    args.input_dim = problem.n_var
    args.num_obj = problem.n_obj
    args.xl, args.xu = args.xl.float().to(args.device), args.xu.float().to(args.device)
    args.bounds = [args.xl, args.xu]
    # set additional arguments
    set_opt_args(args)
    
    print("************************************************************")
    print("Task: ", test_ins)
    print("Device: ", args.device)
    print("Num Objectives: ", args.num_obj)
    print("Num Dimensions: ", args.input_dim)
    print("Reference point: ", ref_point)
    print("Bounds: ", args.bounds)
    print(datetime.datetime.now())
    print("************************************************************")

    outfile = log_results + name_t + ".pkl"
    
    # Repeatedly run the algorithm n_run times
    for run_iter in range(args.start_run, args.n_run):
        
        print("Independent run No. ", run_iter + 1)
        print(datetime.datetime.now())
        print("--------------------------------------------------")
        set_seed(args.ALL_SEED[run_iter])
            
        # initialize n_init solutions 
        x_init = lhs(args.input_dim, args.n_init)
        y_init = problem.evaluate(x_init)

        # initialize dominance-classifier for non-dominance relation
        p_rel_map, s_rel_map = init_dom_rel_map(300)
        p_model = init_dom_nn_classifier(
            x_init, y_init, p_rel_map, pareto_dominance, problem
        )  
        
        evaluated = len(y_init)
        X = x_init
        Y = y_init
        hv = HV(ref_point=np.array(ref_point))
        hv_value = hv(Y)
        hv_all_value[run_iter, 0] = hv_value
        
        z = torch.zeros(args.num_obj).to(args.device)

        use_diffusion = True
        if args.switch_operator:
             # Counter for tracking iterations since last switch
            iteration_since_switch = 0
            # Parameters for switching methods
            hv_change_threshold = 0.05  # Threshold for HV value change
            hv_history_length = 3  # Number of recent iterations to consider
            hv_history = []  # Store recent HV values

            # Initialize list to store historical data
            history_Y = []
        
        i_step = 0
        # Start the main loop for Bayesian-SPREAD
        with tqdm(
            total=n_steps,
            desc=f"SPREAD {test_ins}",
            unit="k",
        ) as pbar:

            while i_step < n_steps: 
                    # Solution normalization
                    transformation = StandardTransform([0, 1])
                    transformation.fit(X, Y)
                    X_norm, Y_norm = transformation.do(X, Y)
                    
                    ## Train GP surrogate model 
                    surrogate_model = GaussianProcess(args.input_dim, args.num_obj, nu=5)
                    surrogate_model.fit(X_norm, Y_norm)
                    
                    if args.switch_operator:
                        ## Data Extraction (like in CDM-PSL)
                        _, index = environment_selection(Y, len(X) // 3)
                        PopDec = X[index, :]
                    else:
                        PopDec = X

                    PopDec_dom_labels, PopDec_cfs = nn_predict_dom_intra(
                        PopDec, p_model, args.device
                    )
                    sorted_pop = sort_population(PopDec, PopDec_dom_labels, PopDec_cfs)
            
                    
                    if use_diffusion:
                        # Generate offspring using SPREAD
                        model = DiTMOO(args.input_dim,
                                        args.num_obj,
                                        num_blocks=args.num_blocks)
                        model.to(args.device)
                        X_psl = gen_offspring_via_spread(
                                model, 
                                sorted_pop, 
                                surrogate_model, 
                                p_model, 
                                args
                            )
                    else:
                        # Generate offspring using SBX
                        rows_to_take = int(1 / 3 * sorted_pop.shape[0])
                        offspringA = sorted_pop[:rows_to_take, :]
                        if len(offspringA) % 2 == 1:
                            offspringA = offspringA[:-1]
                        new_pop = np.empty((0, args.input_dim))
                        for _ in range(1000):
                            result = sbx(offspringA, eta=15)
                            new_pop = np.vstack((new_pop, result))
                        X_psl = new_pop
                    
                    pop_size_used = X_psl.shape[0]
                    
                    # Mutate the new offspring
                    X_psl = pm_mutation(X_psl, [args.xl.detach().cpu().numpy(), args.xu.detach().cpu().numpy()])

                    Y_candidate_mean = surrogate_model.evaluate(X_psl)["F"]
                    Y_candidata_std = surrogate_model.evaluate(X_psl, std=True)["S"]

                    rows_with_nan = np.any(np.isnan(Y_candidate_mean), axis=1)
                    Y_candidate_mean = Y_candidate_mean[~rows_with_nan]
                    Y_candidata_std = Y_candidata_std[~rows_with_nan]
                    X_psl = X_psl[~rows_with_nan]

                    Y_candidate = Y_candidate_mean - args.coef_lcb * Y_candidata_std
                    Y_candidate_mean = Y_candidate
                    
                    # Batch selection (like in PSL-MOBO)   
                    nds = NonDominatedSorting()
                    idx_nds = nds.do(Y_norm)
                    Y_nds = Y_norm[idx_nds[0]]
                    best_subset_list = []
                    Y_p = Y_nds
                    for b in range(args.n_sample):
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
                    
                    Y_new = torch.tensor(Y_new).to(args.device)
                    X_new = torch.tensor(X_new).to(args.device)

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
                    evaluated = evaluated + args.n_sample
                    
                    if args.switch_operator:
                        # Current operator
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
    
    T_task = time() - t0_task
    convert_seconds(T_task)
    
    
