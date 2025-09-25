"""This module contains the function to parse command line arguments."""

from argparse import ArgumentParser, ArgumentTypeError

from baselines.eas.EAs_utils import all_task_names

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
        description="EAs"
    )
    parser.add_argument("--seed", default=0, type=int, choices=ALL_SEED + [0])
    parser.add_argument("--list_seed", default=ALL_SEED)

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
        choices=["train_proxies", "sampling", "evaluation", "report"],
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
        default="generated_samples_ms/eas/",
        type=str,
        help="The folder to store the generated samples",
    )
    parser.add_argument(
        "--results_store_path",
        default="saved_metrics_ms/eas/",
        type=str,
        help="The folder to store the computed metrics in hyper-volume",
    )
    parser.add_argument(
        "--config_store_path",
        default="log_configs/eas/",
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

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num-points-sample", type=int, default=256)

    parser.add_argument("--method", default="nsga3")

    args = parser.parse_args()
    args.num_obj, args.input_dim = get_data_dims(args.task_name)
    args.problem = args.task_name
    args.model_dir = f"log_ms/eas/saved_{args.method}_{args.task_name}"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    set_opt_args(args)

    return args


def set_opt_args(args):

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
            
    else:
        xl = None
        xu = None
        args.need_repair = False
        print("NO BOUNDS !")
