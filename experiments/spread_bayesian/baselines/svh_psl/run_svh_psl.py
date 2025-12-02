import numpy as np
import torch
import pickle

from problem import get
from pymoo.problems import get_problem

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.gp_model.gaussian_process import GaussianProcess
from mobo.transformation import StandardTransform

from model import ParetoSetModel
from SVGD.moosvgd import get_gradient
from torch.autograd import Variable
import random
from mobo.utils import set_seed
import os

from partitioning import sampling_vector_randomly, sampling_vector_evenly
import time
import torch.nn as nn 

# -----------------------------------------------------------------------------
import argparse
import os

import random

from pymoo.config import Config
Config.warnings['not_compiled'] = False

from utils import set_seed, convert_seconds, pm_mutation

from tqdm import tqdm
import datetime
from time import time, sleep

# Set up command line argument parsing
parser = argparse.ArgumentParser(
    description="Run experiments with specified problem set."
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
# Parse command line arguments
args = parser.parse_args()
# Set problem set
ins_list = [args.prob]
# ------------------------------------------------------------------


dic = {
    "zdt1": [0.9994, 6.0576],
    "zdt2": [0.9994, 6.8960],
    "zdt3": [0.9994, 6.0571],
    "dtlz2": [2.8390, 2.9011, 2.8575],
    "dtlz5": [2.6672, 2.8009, 2.8575],
    "dtlz7": [0.9984, 0.9961, 22.8114],
}

get_kernel = False
# number of independent runs
n_run = 5
ALL_SEED = [1000, 2000, 3000, 4000, 5000]
# number of initialized solutions
n_init = 100
# number of iterations, and batch size per iteration
n_iter = 20
batch_size = 5
max_fes = 100
# PSL 
# number of learning steps
n_steps = 250
# coefficient of LCB
coef_lcb = 0.1

n_candidate = 1000
local_kernel= True
binary = True

# device
def get_device(no_cuda=False, gpus=None):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")
# device=get_device(gpus ='2')
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

c_ = 1
alpha_ = 0.1
lr_ = 1e-3

for test_ins in ins_list:
    t0_task = time()
    
    name_t = (
            "svh_psl"
            + "_"
            + test_ins
            + "_K"
            + str(n_iter)
            + "_FE"
            + str(max_fes)
        )
    
    log_results = "logs/saved_hvs/"
    
    if not (os.path.exists(log_results)):
        os.makedirs(log_results)
        
    # get problem info
    bounds = None
    ref_point = None
    hv_all_value = np.zeros([n_run, n_iter + 1])
    if test_ins.startswith("zdt"):
        problem = get_problem(test_ins, n_var=20)
    elif test_ins.startswith("dtlz"):
        problem = get_problem(test_ins, n_var=20, n_obj=3)
    else:
        problem, bounds, ref_point = get(test_ins)
    n_dim = problem.n_var
    n_obj = problem.n_obj
    if ref_point is None:
        ref_point = dic[test_ins]
    if bounds is None:
        bounds = [torch.from_numpy(problem.xl).float(), 
                      torch.from_numpy(problem.xu).float()]
    
    print("************************************************************")
    print("Task: ", test_ins)
    print("Device: ", device)
    print("Num Objectives: ", n_obj)
    print("Num Dimensions: ", n_dim)
    print("Reference point: ", ref_point)
    print("Bounds: ", bounds)
    print(datetime.datetime.now())
    print("************************************************************")

    outfile = log_results + name_t + ".pkl"
    
    n_pref_update = 10

    n_test = 100*n_obj
    pref_vec_test = sampling_vector_evenly(n_obj, n_test)
    
    for run_iter in range(n_run):
        
        print("Independent run No. ", run_iter + 1)
        print(datetime.datetime.now())
        print("--------------------------------------------------")
        set_seed(ALL_SEED[run_iter])
        
        front_list,gp_list, x_list, y_list = {}, {}, {},{}
        front_list = np.zeros((n_iter, n_test, n_obj))
        x_list = np.zeros((n_iter, n_test, n_dim))
        y_list = np.zeros((n_iter, n_test, n_obj))
        gp_list = np.zeros((n_iter, n_test, n_obj))
        x_init = lhs(n_dim, n_init)
        if test_ins.startswith('zdt') or test_ins.startswith('dtlz'):
            y_init = problem.evaluate(x_init)
            X = x_init
            Y = y_init
        else:
            y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
            X = x_init
            Y = y_init.to(torch.float64).cpu().numpy()
            
        
        hv = HV(ref_point=np.array(ref_point))
        hv_value = hv(Y)
        hv_all_value[run_iter, 0] = hv_value
        
        evaluated = len(y_init)
         
        z = torch.zeros(n_obj).to(device)
        
        i_iter = 0
        with tqdm(
            total=n_iter,
            desc=f"SVH_PSL {test_ins}",
            unit="k",
        ) as pbar:

            while i_iter < n_iter:
    
                # create pareto set model
                psmodel = ParetoSetModel(n_dim, n_obj)
                psmodel.to(device)

                # get parameters of the Pareto set model for learning SVH
                params = list(psmodel.parameters())

                # optimizer
                optimizer = torch.optim.Adam(psmodel.parameters(), lr=lr_)
                if test_ins in ['zdt1','zdt2','zdt3']:
                    X_norm = X
                    Y_norm = Y
                else:
                    # solution normalization
                    transformation = StandardTransform([0,1])
                    transformation.fit(X, Y)
                    X_norm,Y_norm = transformation.do(X,Y)

                nds = NonDominatedSorting()
                idx_nds = nds.do(Y_norm)
                X_norm = torch.tensor(X_norm).to(device)
                Y_norm = torch.tensor(Y_norm).to(device)
                
                # train GP surrogate model 
                surrogate_model = GaussianProcess(X_norm, Y_norm,300, n_dim, n_obj, nu = 2.5, device = device)
                surrogate_model.fit(X_norm,Y_norm)
                
                z =  torch.min(torch.cat((z.reshape(1,n_obj),Y_norm - 0.1)), axis = 0).values.data
                
                X_nds = X_norm[idx_nds[0]]
                Y_nds = Y_norm[idx_nds[0]]
                
                hv_max = 0.0

                for t_step in range(n_steps):
                    psmodel.train()
                    
                    # sample n_pref_update preferences
                    alpha = np.ones(n_obj)
                    pref = np.random.dirichlet(alpha,n_pref_update)
                    pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
                    # get the current coressponding solutions
                    x = psmodel(pref_vec)

                    x_np = x.detach().cpu().numpy()
                    mean, std = surrogate_model.predict(x)
                    mean = torch.stack(mean)
                    std = torch.stack(std)

                    value = mean - coef_lcb * std

                    value = value.T
                    idx_che = None
                    loss,idx_che = torch.max((value - z)* pref_vec, dim=1)
                    

                    grad = []
                    for idx,loss_ in enumerate(loss):
                        grad_ = []
                        loss_.backward(retain_graph=True)
                        for i,param in enumerate(params):
                            grad_.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
                            param.grad.zero_()
                        grad.append(torch.cat(grad_, dim=0))

                    grad = torch.stack(grad, dim=0)
                    grad_1 = torch.nn.functional.normalize(grad, dim=0)
                    grad_2 = None

                    # gradient-based pareto set model update 
                    optimizer.zero_grad()
                    grad = get_gradient(grad_1, params, value, alpha_, c_ ,grad_2,pref_vec,idx_che, local_kernel=local_kernel)
                    grad = torch.sum(grad, dim=0)

                    idx = 0
                    for i,param in enumerate(params):
                        param.grad.data = grad[idx:idx+param.numel()].reshape(param.shape)
                        idx +=param.numel()
                    optimizer.step()

                psmodel.eval()

                # solutions selection on the learned Pareto set
                alpha = np.ones(n_obj)
                pref = np.random.dirichlet(alpha,n_candidate)
                pref  = torch.tensor(pref).to(device).float() + 0.0001

                # generate correponding solutions, get the predicted mean/std
                X_candidate = psmodel(pref).to(torch.float32)
                # print(X_candidate)
                X_candidate_np = X_candidate.detach().cpu().numpy()
                Y_candidate_mean, Y_candidata_std = surrogate_model.predict(X_candidate)

                Y_candidate_mean = torch.stack(Y_candidate_mean)
                Y_candidata_std = torch.stack(Y_candidata_std)
                Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std

                Y_candidate = Y_candidate.T.detach().cpu().numpy()
                
                # greedy batch selection 
                best_idx = []
                best_subset_list = []
                Y_p = Y_nds.detach().cpu().numpy()
                # print(Y_candidate)
                for b in range(batch_size):
                    hv = HV(ref_point=np.max(np.vstack([Y_p,Y_candidate]), axis = 0))
                    best_hv_value = 0
                    best_subset = None
                    
                    for k in range(len(Y_candidate)):
                        # if k not in best_idx:
                            Y_subset = Y_candidate[k]
                            Y_comb = np.vstack([Y_p,Y_subset])
                            hv_value_subset = hv(Y_comb)
                            if hv_value_subset > best_hv_value:
                                best_hv_value = hv_value_subset
                                best_subset = [k]  

                    Y_p = np.vstack([Y_p,Y_candidate[best_subset]])
                    best_subset_list.append(best_subset)  
                    
                best_subset_list = np.array(best_subset_list).T[0]
                
                # evaluate the selected batch_size solutions
                X_candidate = torch.tensor(X_candidate_np).to(device)
                X_new = X_candidate[best_subset_list]
                if test_ins.startswith('zdt') or test_ins.startswith('dtlz'):
                    Y_new = problem.evaluate(X_new.detach().cpu().numpy())
                else:
                    Y_new = problem.evaluate(X_new)
                
                # update the set of evaluated solutions (X,Y)
                X = np.vstack([X,X_new.detach().cpu().numpy()])

                if test_ins.startswith('zdt') or test_ins.startswith('dtlz'):
                    Y = np.vstack([Y,Y_new])
                else:
                    Y = np.vstack([Y,Y_new.detach().cpu().numpy()])
                
                hv = HV(ref_point=np.array(ref_point))
                hv_value = hv(Y)
                hv_all_value[run_iter, 1 + i_iter] = hv_value

                pref_vec  = torch.Tensor(pref_vec_test).to(device).float()

                x = psmodel(pref_vec)
        
                Y_candidate_mean__, Y_candidata_std__ = surrogate_model.predict(x)
                Y_candidate_mean__ = torch.stack(Y_candidate_mean__)
                Y_candidata_std__ = torch.stack(Y_candidata_std__)
                Y_candidate__ = Y_candidate_mean__ - coef_lcb*Y_candidata_std__
                

                front_list[i_iter] = Y_candidate_mean__.T.detach().cpu().numpy()
                gp_list[i_iter] = Y_candidate__.T.detach().cpu().numpy()
                x_list[i_iter] = x.detach().cpu().numpy()
        
                if test_ins.startswith('zdt') or test_ins.startswith('dtlz'):
                    y_list[i_iter] = problem.evaluate(x.detach().cpu().numpy())
                else:
                    y_list[i_iter] = problem.evaluate(x.to(device)).detach().cpu().numpy()

            
                hv_text = f"{hv_value:.4e}" 
                evaluated = evaluated + batch_size
                
                i_iter += 1
                
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
        

        print("************************************************************")


    T_task = time() - t0_task
    convert_seconds(T_task)