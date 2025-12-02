import os

import numpy as np
import torch
import timeit

from problems import get_problem_torch
from model import ParetoSetModel

from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

from func_utils import *

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam

from pymoo.util.plotting import plot

import time
from tqdm import tqdm

from pymoo.indicators.hv import HV

import pickle
import os

from argparse import ArgumentParser

ALL_SEED = [1000, 2000, 3000, 4000, 5000]

def parse_args():
    """
    Parse the command line arguments for the STCH
    """
    parser = ArgumentParser(
        description="STCH"
    )
    parser.add_argument("--seed", default=0, type=int, choices=ALL_SEED + [0])
    parser.add_argument("--list_seed", default=ALL_SEED)
    parser.add_argument(
        "--method", default="stch", 
        type=str,
        choices=["stch"]
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="solve",
        choices=["solve", "report"],
        help="The mode to run the script in."
    )

    parser.add_argument(
        "--problem",
        required=True,
        type=str,
        choices=["zdt1", "zdt2", "zdt3", 
                   "dtlz2", "dtlz4",  "dtlz7",
                  "re21", "re33", "re34", "re37", "re41",
        ],
        help="The name of the problem to solve.",
    )
    parser.add_argument(
        "--num_samples",
        default=200,
        type=int,
        help="The number of solutions to generate.",
    )
    parser.add_argument(
        "--nsample",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--nvariable",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--input_dim",
        default=30,
        type=int,
        help="The dimensionality of the input space.",
    )
    parser.add_argument(
        "--iters",
        default=5000,
        type=int,
        help="The number of iterations to run.",
    )
    
    parser.add_argument(
        "--samples_store_path",
        default="generated_samples_ms/",
        type=str,
        help="The folder to store the generated samples",
    )
    
    parser.add_argument(
        "--results_store_path",
        default="saved_metrics_ms/",
        type=str,
        help="The folder to store the computed metrics in hypervolume",
    )
    parser.add_argument(
        "--n_pref_update",
        default=10,
        type=int,
        help="The number of preference updates to perform.",
    )
    parser.add_argument(
        "--objs",
        default=None,
        type=int,
        help="The number of objectives in many-objective problems.",
    )
    return parser.parse_args()
    
args = parse_args()
if args.objs is not None:
    args.method = args.method + "_M" + str(args.objs)
if args.nvariable is not None:
    args.input_dim = args.nvariable
    args.method = args.method + "_D" + str(args.nvariable)

args.samples_store_path = args.samples_store_path + args.method + "/"
args.results_store_path = args.results_store_path + args.method + "/"

if args.nsample is not None:
    args.num_samples = args.nsample

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)
            
def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

if __name__ == '__main__':
    
    if args.mode == "solve":
        
        set_seed(args.seed)
        
        name = (
            args.method
            + "_"
            + args.problem
            + "_"
            + str(args.seed)
            + "_"
            + f"T={args.iters}"
            + "_"
            + f"N={args.num_samples}"
        )
    
        test_ins = args.problem
        
        if args.problem.startswith("zdt"):
            problem = get_problem_torch(args.problem, n_var=args.input_dim)
        elif args.problem.startswith("dtlz"):
            if args.objs is not None:
                objs=args.objs
            else:
                objs=3
            problem = get_problem_torch(args.problem, n_var=args.input_dim, n_obj=objs)
        else:
            try:
                problem = get_problem_torch(args.problem)
                args.input_dim = problem.n_var
            except ValueError:
                raise ValueError(f"Unknown problem: {args.problem}")
            
        n_dim = problem.n_var
        args.input_dim = n_dim
        n_obj = problem.n_obj
        
        print("************************************************************")
        print("Task: ", args.problem)
        print("Device: ", args.device)
        print("Num Samples: ", args.num_samples)
        print("Num Dimensions: ", args.input_dim)
        print("Num Iterations: ", args.iters)
        print("Method: ", args.method)
        print("************************************************************")
            
        if args.objs is not None:
            num_obj = args.objs
        else:
            num_obj = None
        ref_point = get_ref_point(test_ins.lower(), num_obj=num_obj) 
        args.ref_point = ref_point
        
        nadir_point = np.array(ref_point)* 1.1
        ideal_point = np.zeros(len(ref_point))
            
        start = timeit.default_timer()
        store_hv_step = 0
            
        z = torch.zeros(n_obj).to(args.device)
        psmodel = ParetoSetModel(n_dim, n_obj)
        psmodel.to(args.device)

        # optimizer
        optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
        
        start_time = time.time()
        if n_obj <=3:
            ParetoFront = problem.pareto_front()
        
        # t_step Pareto Set Learning with gradient descent
        with tqdm(
            total=args.iters,
            desc=f"{args.method.upper()} {test_ins}",
            unit="k",
        ) as pbar:
            for t_step in range(args.iters):
                psmodel.train()
            
                # sample n_pref_update preferences
                alpha = np.ones(n_obj)
                pref = np.random.dirichlet(alpha,args.n_pref_update)
                pref_vec  = torch.tensor(pref).to(args.device).float() 
                
                # get the current coressponding solutions
                x = psmodel(pref_vec)
                value = problem.evaluate(x)  
                value = (value - torch.tensor(ideal_point).to(args.device)) / torch.tensor(nadir_point - ideal_point).to(args.device) 
                
            
                if args.method == 'ls':
                    ls_value =  torch.sum(pref_vec * (value - z), axis = 1)
                    loss =  torch.sum(ls_value)
                    
                mu =  0.01 
                stch_value = mu* torch.logsumexp(pref_vec * (value - z) / mu, axis = 1)   
                loss =  torch.sum(stch_value)

                # gradient-based pareto set model update 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
                
                pbar.update(1)
            pbar.close()
            
            
        # calculate and report the hypervolume value
        with torch.no_grad():
                
            psmodel.eval()
                
            generated_pf = []
            generated_ps = []
                
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha,args.num_samples)
            pref_vec  = torch.tensor(pref).to(args.device).float() 
            
            sol = psmodel(pref_vec)

            args.bounds = [problem.xl.to(sol.device), problem.xu.to(sol.device)]
            res_x = torch.clamp(sol, min=args.bounds[0], max=args.bounds[1])  # Ensure solutions are within bounds
            res_y = problem.evaluate(res_x).detach().cpu().numpy()
            res_x = res_x.detach().cpu().numpy()
            print("resy.shape:", res_y.shape)
            
            comp_time = time.time()-start_time
            print('Time:', comp_time)

            visible_masks = np.ones(len(res_y))
            visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
            visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0

            res_x = res_x[np.where(visible_masks == 1)[0]]
            res_y = res_y[np.where(visible_masks == 1)[0]]
            
            # Store the results
            if not (os.path.exists(args.samples_store_path)):
                os.makedirs(args.samples_store_path)
            if not (os.path.exists(args.results_store_path)):
                os.makedirs(args.results_store_path)
                
            np.save(args.samples_store_path + name + "_x.npy", res_x)
            np.save(args.samples_store_path + name + "_y.npy", res_y)
            
            if n_obj <=3:
                fs_list = [res_y[:, i] for i in range(res_y.shape[1])]
                plot_pareto_front(fs_list, 
                                    args, 
                                    extra=[ParetoFront[:, i] for i in range(ParetoFront.shape[1])],
                                    lab=str(args.seed))
            
            hv = HV(ref_point=args.ref_point)
            if res_y.shape[1] < 10:
                hv_value = hv(res_y)
            else:
                hv_value = None
            
            hv_results = {
                "ref_point": args.ref_point,
                "hypervolume": hv_value,
                "computation_time": comp_time
            }

            print(f"Hypervolume for seed {args.seed}:", hv_value)
            print("--------------------------------------------")

            with open(args.results_store_path + name + "_hv_results.pkl", "wb") as f:
                pickle.dump(hv_results, f)

    elif args.mode == "report":
        # Report the statistics
        print("Reporting statistics...")
        report_stats(args)
        print("########################################")
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Available modes: solve, report.")   
        

