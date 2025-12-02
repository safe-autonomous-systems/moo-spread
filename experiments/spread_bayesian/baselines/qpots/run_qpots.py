import torch
import warnings
from botorch.utils.transforms import unnormalize
import os
import numpy as np
import pickle

warnings.filterwarnings('ignore')
device = torch.device("cpu")

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.function import Function
from qpots.utils.utils import expected_hypervolume

from qpots.utils.utils import convert_seconds, set_seed, arg_parser

import random

from tqdm import tqdm
import datetime
from time import time, sleep

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.problems import get_problem

args = arg_parser()
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

def repairing_bounds(X, bounds):
    """
    Repair the input tensor `x` to ensure it is within the specified bounds.
    
    Parameters
    ----------
    x : Tensor
        Input tensor to be repaired.
    bounds : tuple
        A tuple containing lower and upper bounds for each dimension.
        
    Returns
    -------
    Tensor
        The repaired tensor with values within the specified bounds.
    """
    xl, xu = bounds
    X = torch.clamp(X, min=xl, max=xu)
    mn, mx = X.min(), X.max()
    # If all values are the same, add noise to avoid singularity
    if mn == mx:
        noise = torch.empty_like(X).uniform_(-1e6, 1e6)
        X = X + noise.to(X.device)
    return X

# print(args)
# number of learning steps
n_steps = int(args.max_fes / args.q)
args.n_iter = n_steps

test_ins = args.problem

t0_task = time()
        
name_t = (
            "qpots"
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

if "zdt" in args.problem:
    args.dim = 20
    args.nobj = 2
elif "dtlz" in args.problem:
    args.dim = 20
    args.nobj = 3
else:
    args.dim = None
    args.nobj = None
    
tf = Function(args.problem, dim=args.dim, nobj=args.nobj)
args.dim = tf.dim
args.nobj = tf.nobj
if args.problem in dic:
    args.ref_point = torch.tensor(dic[args.problem], dtype=torch.float64)
else:
    args.ref_point = -1 * tf.get_ref_point() # -1 for minimization

bounds = tf.get_bounds()

print("************************************************************")
print("Task: ", test_ins)
print("Device: ", args.device)
print("Num Objectives: ", args.nobj)
print("Num Dimensions: ", args.dim)
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
    
    x_init = lhs(args.dim, args.ntrain)
    train_x = (t := torch.tensor(x_init, dtype=torch.float64)).sub(t.min()).div(t.max().sub(t.min()))

    x_init = unnormalize(train_x, bounds)
    x_init = repairing_bounds(x_init, bounds)
    y_init = -1 * tf.evaluate(x_init)
    evaluated = len(y_init)
    hv = HV(ref_point=args.ref_point.numpy())
    hv_value = hv(y_init.detach().cpu().numpy())
    hv_all_value[run_iter, 0] = hv_value

    X = x_init.detach().cpu().numpy()
    Y = y_init.detach().cpu().numpy()
    
    train_y = tf.evaluate(x_init) 
    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=bounds, nobj=args.nobj, ncons=0, device=device)
    gps.fit_gp()

    acq = Acquisition(tf, gps, device=device, q=args.q)
    
    i_step = 0
    with tqdm(
        total=n_steps,
        desc=f"qPOTS {test_ins}",
        unit="k",
    ) as pbar:

        while i_step < args.iters:
            add_kwargs = {
                "nystrom": args.nystrom,
                "iters": args.iters,
                "nychoice": args.nychoice,
                "dim": args.dim,
                "ngen": args.ngen,
                "q": args.q,
            }
            newx = acq.qpots(bounds, i_step, **add_kwargs)
            newy = tf.evaluate(unnormalize(newx.reshape(-1, args.dim), bounds))

            x_new = unnormalize(newx.reshape(-1, args.dim), bounds)
            x_new = repairing_bounds(x_new, bounds)
            y_new = -1 * tf.evaluate(x_new)
            X = np.vstack([X, x_new.detach().cpu().numpy()])
            Y = np.vstack([Y, y_new.detach().cpu().numpy()])
            hv = HV(ref_point=args.ref_point.numpy())
            hv_value = hv(Y)
            hv_all_value[run_iter, 1 + i_step] = hv_value
                    
            i_step += 1
            hv_text = f"{hv_value:.4e}"
            evaluated = evaluated + len(y_new)

            train_x = torch.row_stack([train_x, newx.view(-1, args.dim)])
            train_y = torch.row_stack([train_y, newy])
            gps = ModelObject(train_x, train_y, bounds, args.nobj, args.ncons, device=device)
            gps.fit_gp()
            
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
            

T_task = time() - t0_task
convert_seconds(T_task)