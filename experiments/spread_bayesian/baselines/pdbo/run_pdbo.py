import os
os.environ['OMP_NUM_THREADS'] = '1' # speed up
import numpy as np
from problems.common import build_problem, get_problem
from mobo.algorithms import get_algorithm
from baselines.vis_data_export import DataExport
from arguments import get_args
from utils import save_args, setup_logger
import argparse
import numpy as np
import torch
import pickle
import sys
from utils import set_seed, convert_seconds

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message=".*default value of `n_init`.*",  # regex to match the KMeans warning
    category=FutureWarning,
    module="sklearn.cluster._kmeans"
)   
warnings.filterwarnings(
    "ignore",
    message="delta_grad == 0.0. Check if the approximated function is linear."
)

import json

import random

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from pymoo.config import Config
Config.warnings['not_compiled'] = False

from tqdm import tqdm
import datetime
from time import time, sleep

# load arguments
args, framework_args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert args.n_run <= len(args.ALL_SEED), "The number of independent runs should be at most the number of seeds."

dic = {
    "zdt1": [0.9994, 6.0576],
    "zdt2": [0.9994, 6.8960],
    "zdt3": [0.9994, 6.0571],
    "dtlz2": [2.8390, 2.9011, 2.8575],
    "dtlz5": [2.6672, 2.8009, 2.8575],
    "dtlz7": [0.9984, 0.9961, 22.8114],
}


# --------------------------------------------------------------------------------------------------

# print(args)
# number of learning steps
n_steps = int(args.max_fes / args.batch_size)
args.n_iter = n_steps

test_ins = args.problem

t0_task = time()
        
name_t = (
            args.algo.lower()
            + "_"
            + test_ins
            + "_K"
            + str(n_steps)
            + "_FE"
            + str(args.max_fes)
        )

log_results = "logs/saved_hvs/"
    
if not (os.path.exists(log_results)):
    os.makedirs(log_results)

hv_all_value = np.zeros([args.n_run, n_steps + 1])
# Get problem info
args.n_var = 20
if test_ins.startswith("zdt"):
    args.n_obj = 2
    problem = get_problem(test_ins, n_var=args.n_var)
elif test_ins.startswith("dtlz"):
    args.n_obj = 3
    problem = get_problem(test_ins, n_var=args.n_var, n_obj=args.n_obj)
else:
    args.n_var = None
    args.n_obj = None
    problem = get_problem(test_ins)


if test_ins in dic:
    args.ref_point = torch.tensor(dic[test_ins], dtype=torch.float32)
else:
    args.ref_point = problem.ref_point
    
args.n_obj = problem.n_obj
args.n_var = problem.n_var
bounds = (problem.xl, problem.xu)

print("************************************************************")
print("Task: ", test_ins)
print("Device: ", args.device)
print("Num Objectives: ", args.n_obj)
print("Num Dimensions: ", args.n_var)
print("Reference point: ", args.ref_point)
print("Bounds: ", bounds)
print(datetime.datetime.now())
print("************************************************************")

outfile = log_results + name_t + ".pkl"
# Repeatedly run the algorithm n_run times
for run_iter in range(args.start_run, args.n_run):
    print("Independent run No. ", run_iter + 1)
    print(datetime.datetime.now())
    print("--------------------------------------------------")
    args.seed = args.ALL_SEED[run_iter]
    set_seed(args.seed)
    
    # build problem, get initial samples
    problem, true_pfront, X_init, Y_init = build_problem(args.problem, args.n_var, args.n_obj, args.n_init_sample, args.n_process)
    args.n_var, args.n_obj = problem.n_var, problem.n_obj

    # initialize optimizer
    optimizer = get_algorithm(args.algo)(problem, args.n_iter, args.ref_point, framework_args)

    # save arguments & setup logger
    save_args(args, framework_args)
    logger = setup_logger(args)
    print(problem, optimizer, sep='\n')
        
    # initialize data exporter
    exporter = DataExport(optimizer, X_init, Y_init, args)

    evaluated = len(Y_init)
    X = X_init
    Y = Y_init
    hv = HV(ref_point=np.array(args.ref_point))
    hv_value = hv(Y)
    hv_all_value[run_iter, 0] = hv_value

    # optimization
    solution = optimizer.solve(X_init, Y_init)
        
    i_step = 0
    # Start the main loop for Bayesian-SPREAD
    with tqdm(
        total=n_steps,
        desc=f"{args.algo.upper()} {test_ins}",
        unit="k",
    ) as pbar:

        while i_step < args.n_iter:
            # get new design samples and corresponding performance
            X_next, Y_next = next(solution)

            X = np.vstack([X, X_next]) 
            Y = np.vstack([Y, Y_next]) 
            hv = HV(ref_point=np.array(args.ref_point))
            hv_value = hv(Y)
            hv_all_value[run_iter, 1 + i_step] = hv_value
                    
            i_step += 1
            hv_text = f"{hv_value:.4e}" 
            evaluated = evaluated + len(Y_next)
            
            exporter.update(X_next, Y_next)
            exporter.write_csvs()
            
            pbar.set_postfix({"HV": hv_text, "Num Points": evaluated}) 
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
            
    if logger is not None:
        logger.close()
            
hv_array = np.array(hv_all_value)
hv_means = np.mean(hv_array, axis=0)

T_task = time() - t0_task
convert_seconds(T_task)

