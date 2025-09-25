import json
import os
import datetime
from time import time, sleep
import numpy as np
import torch
from ms_args import parse_args, set_opt_args
from ms_experiments import (
    evaluation,
    sampling,
    train_spread,
    report_stats,
)

from problems import get_problem_torch
from pymoo.problems.multi import MODAct
from ms_args import get_problem_by_name

import random
from pymoo.config import Config
Config.warnings['not_compiled'] = False

dic_ref_point = {
    "zdt1": [0.9994, 6.0576],
    "zdt2": [0.9994, 6.8960],
    "zdt3": [0.9994, 6.0571],
    "dtlz2": [2.8390, 2.9011, 2.8575],
    "dtlz4": [3.2675, 2.6443, 2.4263],
    "dtlz7": [0.9984, 0.9961, 22.8114],
    "re21": [3144.44, 0.05],
    "re33": [5.01, 9.84, 4.30],
    "re34": [1.86472022e+03, 1.18199394e+01, 2.90399938e-01],
    "re37": [1.1022, 1.20726899, 1.20318656],
    "re41": [47.04480682,  4.86997366, 14.40049127, 10.3941957 ],
}

def main(args):
    print(datetime.datetime.now())
    problem = get_problem_by_name(args.problem, args)
    # get bounds
    args.xl = problem.xl.float()
    args.xu = problem.xu.float()
    args.bounds = [args.xl, args.xu]
    args.need_repair = True # TODO: make this dynamic based on the problem
    # Get problem information
    args.num_obj = problem.n_obj
    args.input_dim = problem.n_var
    args.ref_point = dic_ref_point[args.problem]

    # Set SPREAD hyperparameters
    set_opt_args(args)
    
    # Save log config
    if not os.path.exists(args.config_store_path):
        os.makedirs(args.config_store_path)

    if "cpu" in args.method:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************************************************************")
    print("Task: ", args.problem)
    print("Device: ", args.device)
    print("Num Objectives: ", args.num_obj)
    print("Num Dimensions: ", args.input_dim)
    print("Reference point: ", args.ref_point)
    print("Bounds: ", args.bounds)
    print("Gamma Scale Delta: ", args.gamma_scale_delta)
    print("Eta: ", args.eta)
    print("Learning Rate Inner: ", args.lr_inner)
    print("************************************************************")
    
    if "ablation" in args.method:
        print("-------------------------------------------------")
        print("Running ablation study with SPREAD.")
        print("Ablation Type: ", args.ablation)
        print("-------------------------------------------------")
    
    start_time = time()
    if args.mode == "train_spread":
        train_spread(problem, args)
    elif args.mode == "sampling":
        if "ablation" in args.method and "rho" in args.ablation:
            print("-------------------------------------------------")
            print("Rho List: ", args.rho_list)
            print("-------------------------------------------------")
            for rho in args.rho_list:
                args.gamma_scale_delta = rho
                args.label = f"rho_{rho}"
                print("-------------------------------------------------")
                print(f"Running sampling with rho: {rho}")
                sampling(problem, args)
        elif "ablation" in args.method and "lambda_rep" in args.ablation:
            print("-------------------------------------------------")
            print("Lambda List: ", args.lambda_list)
            print("-------------------------------------------------")
            for lambda_rep in args.lambda_list:
                args.lambda_rep = lambda_rep
                args.label = f"lambda_{lambda_rep}"
                print("-------------------------------------------------")
                print(f"Running sampling with lambda_rep: {lambda_rep}")
                sampling(problem, args)
        else:
            sampling(problem, args)
    elif args.mode == "evaluation":
        evaluation(problem, args)
    elif args.mode == "report":
        report_stats(problem, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    print(f"Total time: {time() - start_time} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(args)
