import json
import os

import pickle
from tqdm import tqdm
import numpy as np
import offline_moo.off_moo_bench as ob
import torch
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.utils import get_quantile_solutions, set_seed

import math
import torch.nn as nn
import torch.optim as optim
import datetime
from time import time, sleep
from torch.utils.data import DataLoader, TensorDataset

from baselines.eas.EAs_args import parse_args
from baselines.eas.EAs_nets import VectorFieldNet
from baselines.eas.EAs_utils import (
    ALLTASKSDICT,
    MultipleModels,
    SingleModel,
    SingleModelBaseTrainer,
    get_dataloader,
    tkwargs,
)

import random
import cv2
import glob
import re

import json

from pymoo.optimize import minimize

from baselines.eas.all_utils import *

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train_proxies(args):
    task_name = ALLTASKSDICT[args.task_name]
    print(f"Task: {task_name}")

    set_seed(args.seed)

    task = ob.make(task_name)

    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    n_obj = y.shape[1]
    data_size, n_dim = tuple(X.shape)
    model_save_dir = args.proxies_store_path
    os.makedirs(model_save_dir, exist_ok=True)

    model = MultipleModels(
        n_dim=n_dim,
        n_obj=n_obj,
        train_mode="Vallina",
        hidden_size=[2048, 2048],
        save_dir=args.proxies_store_path,
        save_prefix=f"MultipleModels-Vallina-{task_name}-{args.seed}",
    )
    model.set_kwargs(**tkwargs)

    trainer_func = SingleModelBaseTrainer

    for which_obj in range(n_obj):

        y0 = y[:, which_obj].copy().reshape(-1, 1)

        trainer = trainer_func(
            model=list(model.obj2model.values())[which_obj],
            which_obj=which_obj,
            args=args,
        )

        (train_loader, val_loader) = get_dataloader(
            X,
            y0,
            val_ratio=(
                1 - args.proxies_val_ratio
            ),  # means 0.9 for training and 0.1 for validation
            batch_size=args.proxies_batch_size,
        )

        trainer.launch(train_loader, val_loader)

def sampling(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    name = (
        args.method
        + "_"
        + task_name
        + "_"
        + str(args.seed)
        + "_"
        + f"T={args.timesteps}"
        + "_"
        + f"N={args.num_points_sample}"
    )

    if not (os.path.exists(args.samples_store_path)):
        os.makedirs(args.samples_store_path)

    # Load the classifiers
    classifiers = []
    for i in range(args.num_obj):
        classifier = SingleModel(
            input_size=args.input_dim,
            which_obj=i,
            hidden_size=[2048, 2048],
            save_dir=args.proxies_store_path,
            save_prefix=f"MultipleModels-Vallina-{task_name}-{0}",
        )
        classifier.load()
        classifier = classifier.to(device)
        classifier.eval()
        classifiers.append(classifier)
    print(f"Loaded {len(classifiers)} classifiers successfully.")

    algorithm = get_pymoo_algo(args)
    problem_is = pymoo_objective_functions(args.device, 
                                           n_var = args.input_dim, 
                                           n_obj = args.num_obj, 
                                           xl = np.array(args.bounds[0]), 
                                           xu = np.array(args.bounds[1]), 
                                           classifiers=classifiers)
    
    print("START sampling ...")
    t0_epch_f = time()

    res = minimize(
            problem_is,
            algorithm,
            seed=args.seed,
            termination=("n_gen", args.timesteps),
            verbose=True,
        )

    print(
        f"END sampling !"
    )

    T_epch_f = time() - t0_epch_f
    
    # Compute the true hypervolume
    res_x = res.X
    if args.normalization: # Denormalize the solutions
        res_x = task.denormalize_x(res_x)
    
    if task.is_discrete:
        _, dim, n_classes = tuple(res_x.shape)
        res_x = res_x.reshape(-1, dim, n_classes)
        res_x = task.to_integers(res_x)
    if task.is_sequence:
        res_x = task.to_integers(res_x)

    res_y = task.predict(res_x)  # get objective values via task.predict
    if res_y.shape[0] != res_x.shape[0]:
        res_y = res_y.T
    
    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    nadir_point = task.nadir_point
    # For calculating hypervolume, we use the min-max normalization
    res_y = task.normalize_y(res_y, normalization_method="min-max")
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    res_y_50_percent = task.normalize_y(
        res_y_50_percent, normalization_method="min-max"
    )
    res_y_75_percent = task.normalize_y(
        res_y_75_percent, normalization_method="min-max"
    )

    # Store the results
    if not (os.path.exists(args.samples_store_path)):
        os.makedirs(args.samples_store_path)

    _, d_best = task.get_N_non_dominated_solutions(
        N=args.num_points_sample, return_x=False, return_y=True
    )
    d_best = task.normalize_y(d_best, normalization_method="min-max")

    d_best_hv = hv(nadir_point, d_best, task_name)
    hv_value = hv(nadir_point, res_y, task_name)
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

    # print(f"Pareto Set: {pareto_set}")
    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")
    # Save the results
    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value,
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
    }

    np.save(args.samples_store_path + name + "_x.npy", res_x)
    np.save(args.samples_store_path + name + "_y.npy", res_y)

    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.json", "w") as f:
        json.dump(hv_results, f, indent=4)

    # Print computation time
    convert_seconds(T_epch_f)
    print(datetime.datetime.now())

    return res_x, res_y


def evaluation(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    # Get the data
    name = (
        args.method
        + "_"
        + task_name
        + "_"
        + str(args.seed)
        + "_"
        + f"T={args.timesteps}"
        + "_"
        + f"N={args.num_points_sample}"
    )
    res_x = np.load(args.samples_store_path + name + "_x.npy")
    res_y = np.load(args.samples_store_path + name + "_y.npy")
    print(f"Loaded the generated samples from {args.samples_store_path}")

    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    nadir_point = task.nadir_point
    # For calculating hypervolume, we use the min-max normalization
    res_y = task.normalize_y(res_y, normalization_method="min-max")
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    res_y_50_percent = task.normalize_y(
        res_y_50_percent, normalization_method="min-max"
    )
    res_y_75_percent = task.normalize_y(
        res_y_75_percent, normalization_method="min-max"
    )

    _, d_best = task.get_N_non_dominated_solutions(N=256, return_x=False, return_y=True)

    d_best = task.normalize_y(d_best, normalization_method="min-max")

    d_best_hv = hv(nadir_point, d_best, task_name)
    hv_value = hv(nadir_point, res_y, task_name)
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")

    # Save the results
    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value,
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
    }

    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.json", "w") as f:
        json.dump(hv_results, f, indent=4)


def report_stats(args):
    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    list_hv_100th = []
    list_hv_75th = []
    list_hv_50th = []

    for seed in args.list_seed:
        # Get the data
        name = (
            args.method
            + "_"
            + task_name
            + "_"
            + str(seed)
            + "_"
            + f"T={args.timesteps}"
            + "_"
            + f"N={args.num_points_sample}"
        )

        path_to_results = args.results_store_path + name + "_hv_results.json"

        list_hv_100th.append(get_value_from_json(path_to_results, "hypervolume/100th"))
        list_hv_75th.append(get_value_from_json(path_to_results, "hypervolume/75th"))
        list_hv_50th.append(get_value_from_json(path_to_results, "hypervolume/50th"))

    # Save the results
    mean_std_hv_results = {
        "TASK": task_name,
        "mean_std_hypervolume/100th": mean_std_stats(list_hv_100th, to_decimal=2),
        "mean_std_hypervolume/75th": mean_std_stats(list_hv_75th, to_decimal=2),
        "mean_std_hypervolume/50th": mean_std_stats(list_hv_50th, to_decimal=2),
    }

    with open(
        args.results_store_path + "mean_std_hv_results" + name + ".json", "w"
    ) as f:
        json.dump(mean_std_hv_results, f, indent=4)
        
    print("mean_std_hypervolume/100th", mean_std_stats(list_hv_100th, to_decimal=2))

    return mean_std_hv_results
