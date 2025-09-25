import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam

from func_utils import *
from problems import get_problem_torch

from solver import PMGDASolver, get_uniform_pref

from pymoo.util.plotting import plot

from tqdm import tqdm
import time

from pymoo.indicators.hv import HV

import pickle
import os

from argparse import ArgumentParser

ALL_SEED = [1000, 2000, 3000, 4000, 5000]

def parse_args():
    """
    Parse the command line arguments for the MOO-SVGD
    """
    parser = ArgumentParser(
        description="PMGDA"
    )
    parser.add_argument("--seed", default=0, type=int, choices=ALL_SEED + [0])
    parser.add_argument("--list_seed", default=ALL_SEED)
    parser.add_argument(
        "--method", default="pmgda", 
        type=str, 
        choices=["pmgda", "pmgda_ablation"]
    )
    parser.add_argument(
        "--ablation", default="time",
        type=str,
        choices=["time"]
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

    return parser.parse_args()
    
args = parse_args()
if "ablation" in args.method:
    args.method = args.method + "_" + args.ablation
if args.nvariable is not None:
    args.method = args.method + "_D" + str(args.nvariable)
args.samples_store_path = args.samples_store_path + args.method + "/"
args.results_store_path = args.results_store_path + args.method + "/"

if args.nsample is not None:
    args.num_samples = args.nsample

if __name__ == '__main__':
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.nvariable is not None:
        args.input_dim = args.nvariable
    
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
        
        if args.problem.startswith("zdt"):
            problem = get_problem_torch(args.problem, n_var=args.input_dim)
        elif args.problem.startswith("dtlz"):
            problem = get_problem_torch(args.problem, n_var=args.input_dim, n_obj=3)
        else:
            try:
                problem = get_problem_torch(args.problem)
                args.input_dim = problem.n_var
            except ValueError:
                raise ValueError(f"Unknown problem: {args.problem}")
        
        print("************************************************************")
        print("Task: ", args.problem)
        print("Device: ", args.device)
        print("Num Samples: ", args.num_samples)
        print("Num Dimensions: ", args.input_dim)
        print("Num Iterations: ", args.iters)
        print("Method: ", args.method)
        print("************************************************************")
        
        x = torch.rand((args.num_samples, args.input_dim)).to(args.device)
        x.requires_grad = True
        optimizer = Adam([x], lr=5e-4)
        
        args.ref_point = get_ref_point(args.problem)
        hv = HV(ref_point=args.ref_point)
        args.bounds = [problem.xl.to(args.device), problem.xu.to(args.device)]


        prefs = get_uniform_pref(n_prob=args.num_samples, 
                                 n_obj=problem.n_obj, clip_eps=1e-6)
        SOLVER = PMGDASolver(problem, prefs, 
                             n_prob=args.num_samples, n_obj=problem.n_obj, 
                            verbose=True)
        
        start_time = time.time()
        hv_results = []

        with tqdm(
            total=args.iters,
            desc=f"PNGDA: ",
            unit="i",
        ) as pbar:
            for i in range(args.iters):
                optimizer.zero_grad()
                y = problem.evaluate(x)
                alphas = SOLVER.compute_weights(x, y)
                alphas = torch.nan_to_num(alphas, nan=torch.nanmean(alphas), posinf=0.0, neginf=0.0)
                loss = torch.sum(alphas.to(args.device) * y)
                loss.backward()
                optimizer.step()

                x.data = torch.clamp(x.data.clone(), 
                                    min=args.bounds[0],
                                    max=args.bounds[1])

                
                pbar.set_postfix({
                    "loss": loss.item(),
                    })
                pbar.update(1)
                    
        comp_time = time.time() - start_time
        print('Time:', comp_time)

        res_x = torch.clamp(x.data.clone(), 
                            min=args.bounds[0], 
                            max=args.bounds[1])  # Ensure solutions are within bounds
        res_y = problem.evaluate(res_x).detach().cpu().numpy()
        res_x = res_x.detach().cpu().numpy()

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
            "computation_time": comp_time
        }
        
        if problem.n_obj <=3:
            ParetoFront = problem.pareto_front()
            plot_pareto_front([res_y[:, i] for i in range(res_y.shape[1])], 
                                args, 
                                extra=[ParetoFront[:, i] for i in range(ParetoFront.shape[1])],
                                lab=str(args.seed))

        print("Hypervolume for seed {}: {}".format(args.seed, hv_value))
        print("########################################")

        # Store the results
        if not (os.path.exists(args.samples_store_path)):
            os.makedirs(args.samples_store_path)
        if not (os.path.exists(args.results_store_path)):
            os.makedirs(args.results_store_path)
            
        np.save(args.samples_store_path + name + "_x.npy", res_x)
        np.save(args.samples_store_path + name + "_y.npy", res_y)
        with open(args.results_store_path + name + "_hv_results.pkl", "wb") as f:
            pickle.dump(hv_results, f)

    elif args.mode == "report":
        # Report the statistics
        print("Reporting statistics...")
        report_stats(args)
        print("########################################")
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Available modes: solve, report.")