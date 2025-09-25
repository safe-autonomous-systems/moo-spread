"""This module contains the function to parse command line arguments."""

from argparse import ArgumentParser
import torch
from all_funcs_utils import *
from pymoo.problems.many.wfg import WFG1
from problems import get_problem_torch

ALL_SEED = [1000, 2000, 3000, 4000, 5000]

def parse_args():
    """
    Parse the command line arguments for SPREAD
    """
    parser = ArgumentParser(
        description="SPREAD: Sampling-based Pareto front Refinement via Efficient Adaptive Diffusion"
    )
    parser.add_argument("--seed", default=0, type=int, choices=ALL_SEED + [0])
    parser.add_argument("--list_seed", default=ALL_SEED)
    parser.add_argument(
        "--num_blocks",
        default=3,
        type=int,
        help="Number of DiT blocks in the model",
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
        "--method", default="spread_ablation",
        type=str,
        choices=["spread", "spread_ablation"]
    )
    parser.add_argument(
        "--ablation", default="time",
        type=str,
        choices=["rho", "time", "noise", "repulsion", 
                 "diversity", "lambda_rep", "ditblocks"]
    )
    parser.add_argument(
        "--rho_list", 
        default=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    )
    parser.add_argument(
        "--lambda_list", default=[0.0, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    )
    parser.add_argument(
        "--label",
        default=None,
        help="The label for the experiment."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train_spread", "sampling", "evaluation", "report"],
        help="True denotes we need to train the flow matching model",
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
        "--config_store_path",
        default="log_configs_ms/",
        type=str,
        help="The folder to store the log configurations",
    )
    
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate for DDPM training"
    )
    parser.add_argument(
        "--ddpm_training_size",
        default=10000,
        type=int,
        help="Size of the training set for training the DDPM",
    )
    parser.add_argument(
        "--ddpm_validation_rate",
        default=0.1,
        type=float,
        help="Size of the validation set for training the DDPM",
    )
    parser.add_argument(
        "--ddpm_training_tol",
        default=100,
        type=int,
        help="Early Stopping for DDPM training",
    )

    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--sampling_tol", type=int, default=None)
    parser.add_argument("--strict_guidance", default=False)
    parser.add_argument("--num-points-sample", type=int, default=200)
    parser.add_argument("--nsample", type=int, default=None)
    parser.add_argument("--nvariable", type=int, default=None)
    parser.add_argument("--free_initial_h", default=True)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--kernel-sigma", default=0.01)
    parser.add_argument("--lambda-rep", default=10)
    parser.add_argument("--num-inner-steps", default=10)
    parser.add_argument("--gamma_scale_delta", default= 0.01)
    parser.add_argument(
        "--eta",
        default=0.1,
        help="Guidance learning rate for 'spread'",
    )
    parser.add_argument(
        "--lr_inner",
        type=float,
        default= 5e-4,
        help="learning rate for the inner optimization",
    )

    parser.add_argument(
        "--global_clamping",
        default=False,
        help="If True, use global clamping: [min=xl.min(), max=xu.max()]",
    )

    args = parser.parse_args()
    
    if "ablation" in args.method:
        args.method = args.method + "_" + args.ablation
        
    if args.nvariable is not None:
        args.method = args.method + "_D" + str(args.nvariable)
    
    if "ablation" in args.method and "ditblocks" in args.ablation:
        args.method = args.method + f"_Block{args.num_blocks}"
        print("-------------------------------------------------")
        print(f"Number of DiT Blocks: {args.num_blocks}")
        print("-------------------------------------------------")

    args.model_dir = f"log_ms/{args.method}/saved_{args.problem}_T{args.timesteps}_B{args.batch_size}"  
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    args.samples_store_path = args.samples_store_path + args.method + "/"
    args.results_store_path = args.results_store_path + args.method + "/"
    
    if args.nsample is not None:
        args.num_points_sample = args.nsample

    return args

def get_problem_by_name(name, args=None):
    if args is not None and args.nvariable is not None:
        n_var = args.nvariable
    else:
        n_var = 30
    # Get the problem instance based on the provided problem name
    name = name.lower()
    if name.startswith("zdt"):
        problem = get_problem_torch(name, n_var=n_var)
    elif name.startswith("dtlz"):
        problem = get_problem_torch(name, n_var=n_var, n_obj=3)
    else:
        try:
            problem = get_problem_torch(name)
        except ValueError:
            raise ValueError(f"Unknown problem: {name}")
    return problem


def set_opt_args(args):

    if args.num_obj == 2:
        args.gamma_scale_delta = 0.9
        args.eta = 0.9 
        args.lr_inner = 0.9
            
    else:
        args.gamma_scale_delta = 0.001
        args.eta = 0.9 # 0.1
        args.lr_inner = 0.9
        
        if args.problem in ["dtlz4"]:
            args.gamma_scale_delta = 0.001
            args.eta = 0.1
            args.lr_inner = 0.9

    return args