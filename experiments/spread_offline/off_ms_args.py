"""This module contains the function to parse command line arguments."""

from argparse import ArgumentParser, ArgumentTypeError

from off_ms_utils import all_task_names

import torch
from all_funcs_utils import *
from types import SimpleNamespace

from offline_moo.off_moo_bench.problem.dtlz import DTLZ
from offline_moo.off_moo_bench.problem.synthetic_func import SyntheticProblem

def str2bool(v):
    """
    Convert a string to a boolean value. This function is used as a type for the ArgumentParser
    to avoid potential errors when parsing boolean values.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ArgumentTypeError("Boolean value expected.")


ALL_SEED = [1000, 2000, 3000, 4000, 5000]


def parse_args():  # Parse command line arguments
    """
    Parse the command line arguments for the MOO-SPREAD
    """
    parser = ArgumentParser(
        description="SPREAD: Sampling-based Pareto front Refinement via Efficient Adaptive Diffusion"
    )
    parser.add_argument("--seed", default=0, type=int, choices=ALL_SEED + [0])
    parser.add_argument("--list_seed", default=ALL_SEED)
    
    parser.add_argument(
        "--method", default="spread",
        type=str,
        choices=["spread"]
    )
    parser.add_argument(
        "--label",
        default=None,
        help="The label for the experiment."
    )

    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        choices=all_task_names,
        help="Which task we will use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train_proxies", "train_spread", "sampling", "evaluation", "report"],
        help="True denotes we need to train the flow matching model",
    )
    parser.add_argument(
        "--normalization",
        type=str2bool,
        nargs="?",
        default=True,
        help="True denotes normalizing the inputs and outputs",
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
        help="The folder to store the computed metrics in hyper-volume",
    )
    parser.add_argument(
        "--config_store_path",
        default="log_configs_ms/",
        type=str,
        help="The folder to store the log configurations",
    )
    parser.add_argument(
        "--proxies_store_path",
        default="proxies_model/",
        type=str,
        help="The folder to store the trained proxies model",
    )
    parser.add_argument(
        "--proxies_epochs",
        default=200,
        type=int,
        help="Number of epochs to train the proxies model",
    )
    parser.add_argument(
        "--proxies_lr",
        default=1e-3,
        type=float,
        help="Learning rate for the optimizer for training proxies model",
    )
    parser.add_argument(
        "--proxies_lr_decay",
        default=0.98,
        type=float,
        help="Learning rate decay for the optimizer for training proxies model",
    )
    parser.add_argument(
        "--proxies_batch_size",
        default=128,
        type=int,
        help="Batch size for training the proxies model",
    )
    parser.add_argument(
        "--proxies_val_ratio",
        default=0.1,
        type=float,
        help="The ratio of the validation set",
    )

    parser.add_argument(
        "--num_blocks",
        default=3,
        type=int,
        help="Number of DiT blocks in the model",
    )
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_tol", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate for DDPM training"
    )
    parser.add_argument(
        "--lr_inner",
        type=float,
        default= 5e-4,
        help="learning rate for the inner optimization",
    )
    parser.add_argument("--num-points-sample", type=int, default=256)
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
        "--b_rate",
        default=0.28,
        help="To decide: Per-dimension clamping --- or --- Global clamping",
    )
    parser.add_argument(
        "--ddpm_validation_size",
        default=1000,
        type=int,
        help="Size of the validation set for training the DDPM",
    )
    parser.add_argument(
        "--ddpm_training_tol",
        default=100,
        type=int,
        help="Early Stopping for DDPM training",
    )

    args = parser.parse_args()
    args.num_obj, args.input_dim = get_data_dims(args.task_name)
    args.problem = args.task_name
    args.model_dir = f"log_ms/spread/saved_{args.task_name}_T{args.timesteps}_B{args.batch_size}"  # _v{ALL_SEED.index(args.seed)}
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    set_opt_args(args)
    
    args.samples_store_path = args.samples_store_path + args.method + "/"
    args.results_store_path = args.results_store_path + args.method + "/"
    args.config_store_path = args.config_store_path + args.method + "/"
    args.proxies_store_path = args.proxies_store_path + args.method + "/"

    return args


def set_opt_args(args):
    
    print("Objectives: ", args.num_obj)

    if args.num_obj == 2:
        args.gamma_scale_delta = 0.9
        args.eta = 0.9
        args.lr_inner = 0.9 

        if args.problem in ["zdt4"]:
            args.gamma_scale_delta = 0.0001 
            args.eta = 0.02 
            args.lr_inner = 0.002 
        
        if args.problem in ["zdt6"]:
            args.gamma_scale_delta = 0.1 
            args.eta = 0.5 
            args.lr_inner = 0.9 
            
    else:
        args.gamma_scale_delta = 0.001
        args.eta = 0.1
        args.lr_inner = 0.9
        
        if args.problem in ["dtlz4", "dtlz6"]:
            args.gamma_scale_delta = 0.0001  
            args.eta = 0.01  
            args.lr_inner = 0.9 

    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    # Precompute lower bound and upper bound for the repair method
    if (
        task.xl is not None
        and task.xu is not None
        and (isinstance(task.problem, DTLZ) or isinstance(task.problem, SyntheticProblem))
    ):
        xl = task.xl 
        xu = task.xu 
        
        print("xl = ", xl.tolist())
        print("xu = ", xu.tolist())

        args.bounds_orig = [xl.tolist(), xu.tolist()]

        if task.is_discrete:
            xl = task.to_logits(np.int64(xl).reshape(1, -1))
            _, dim, n_classes = tuple(xl.shape)
            xl = xl.reshape(-1, dim * n_classes)
        if task.is_sequence:
            xl = task.to_logits(np.int64(xl).reshape(1, -1))
        xl = task.normalize_x(xl)
        if task.is_discrete:
            xu = task.to_logits(np.int64(xu).reshape(1, -1))
            _, dim, n_classes = tuple(xu.shape)
            xu = xu.reshape(-1, dim * n_classes)
        if task.is_sequence:
            xu = task.to_logits(np.int64(xu).reshape(1, -1))
        xu = task.normalize_x(xu)
        
        print("xl normalized = ", xl.tolist())
        print("xu normalized = ", xu.tolist())

        args.need_repair = True
        args.bounds = [xl.tolist()[0], xu.tolist()[0]]

        global_range = max(args.bounds[1]) - min(args.bounds[0])
        dist_upper = max(args.bounds[1]) - min(args.bounds[1])
        dist_lower = max(args.bounds[0]) - min(args.bounds[0])
        print(f"Upper bounds rate: {dist_upper/global_range} *** Lower bounds rate: {dist_lower/global_range}.")
        
        if args.problem in ["zdt4", "zdt6", "dtlz6"]:
            print("--- Per-dimension clamping is activated.")
            args.global_clamping = False
        else:
            print("--- Global clamping is activated.")
            args.global_clamping = True
            
    else:
        xl = None
        xu = None
        args.need_repair = False
        print("NO BOUNDS !")
