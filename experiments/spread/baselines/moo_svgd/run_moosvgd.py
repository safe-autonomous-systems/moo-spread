import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from moosvgd import *
from pymoo.problems import get_problem
from pymoo.util.plotting import plot
from problems.loss_functions import *

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
        description="MOO-SVGD"
    )
    parser.add_argument("--seed", default=0, type=int, choices=ALL_SEED + [0])
    parser.add_argument("--list_seed", default=ALL_SEED)
    parser.add_argument(
        "--mode",
        type=str,
        default="solve",
        choices=["solve", "report"],
        help="The mode to run the script in."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="moo_svgd",
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

if args.nsample is not None:
    args.num_samples = args.nsample
    
args.samples_store_path = args.samples_store_path + args.method + "/"
args.results_store_path = args.results_store_path + args.method + "/"

if __name__ == '__main__':
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        
        cur_problem = args.problem.lower()
        if cur_problem == "re21":
            args.input_dim = 4
        elif cur_problem == "re33":
            args.input_dim = 4
        elif cur_problem == "re34":
            args.input_dim = 5
        elif cur_problem == "re37":
            args.input_dim = 4
        elif cur_problem == "re41":
            args.input_dim = 7

        args.bounds = get_bounds(cur_problem, n_var=args.input_dim, device=args.device)
        x = torch.rand((args.num_samples, args.input_dim)).to(args.device)
        x.requires_grad = True
        optimizer = Adam([x], lr=5e-4)
        
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
        args.ref_point = get_ref_point(cur_problem, num_obj=num_obj)
        hv = HV(ref_point=args.ref_point)
        start_time = time.time()
        hv_results = []
        with tqdm(
            total=args.iters,
            desc=f"MOO-SVGD: ",
            unit="i",
        ) as pbar:
            for i in range(args.iters):
                list_fs = loss_function(x, problem=cur_problem, n_obj=num_obj)
                pfront = torch.stack(list_fs, dim=1) 
                pfront = pfront.detach().cpu().numpy()
                pfront_idx = get_non_dominated_points(
                    x, pfront,
                    indx_only=True
                )
                pfront = pfront[pfront_idx]

                if pfront.shape[1] <= 3:
                    if i == args.iters - 1:
                        if cur_problem.startswith("re"):
                            file_key = cur_problem.upper()
                            path = f"problems/pf_re_tasks/reference_points_{file_key}.dat"
                            if not os.path.isfile(path):
                                raise FileNotFoundError(f"Pareto front file not found: {path}")
                            pareto_front = np.loadtxt(path)
                        else:
                            try:
                                problem = get_problem(cur_problem)
                                pareto_front = problem.pareto_front()
                            except ValueError:
                                raise ValueError(f"Can't get Pareto front for problem: {cur_problem}")
                        plot_pareto_front([fi.detach().cpu().numpy() for fi in list_fs], 
                                        args, 
                                        extra=[pareto_front[:, i] for i in range(pareto_front.shape[1])],
                                        lab=str(args.seed))
                        
                list_grad_i = []
                for fi in list_fs:
                    fi.sum().backward(retain_graph=True)
                    grad_i = x.grad.detach().clone()
                    grad_i = torch.nn.functional.normalize(grad_i, dim=0)
                    list_grad_i.append(grad_i)
                    x.grad.zero_()
                optimizer.zero_grad()
                x.grad = get_gradient(list_grad_i, x, torch.stack(list_fs, dim=-1))
                optimizer.step()

                x.data = torch.clamp(x.data.clone(), min=args.bounds[0], max=args.bounds[1])

                pbar.update(1)

        comp_time = time.time() - start_time
        print('Time:', comp_time)

        list_fs = loss_function(x, problem=cur_problem, n_obj=num_obj)
        pfront = torch.stack(list_fs, dim=1)
        res_y = pfront.detach().cpu().numpy()
        res_x = x.detach().cpu().numpy()

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