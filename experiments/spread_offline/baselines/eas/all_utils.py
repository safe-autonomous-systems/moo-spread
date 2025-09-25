import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

import random
import cv2
import glob
import re

import json

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from baselines.eas.EAs_utils import get_reference_directions
from pymoo.core.problem import Problem

def convert_seconds(seconds):
    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    # Format the result
    print(f"Time: {hours} hours {minutes} minutes {remaining_seconds} seconds")


def get_value_from_json(file_path, key):
    """
    Reads a JSON file from file_path and returns the value for the specified key.

    Parameters:
        file_path (str): The path to the JSON file.
        key (str): The key whose value you want to retrieve.

    Returns:
        The value corresponding to the key if found, otherwise None.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        # Return the value or None if the key does not exist.
        return data.get(key)
    except Exception as e:
        print(f"Error reading the JSON file: {e}")
        return None


def mean_std_stats(values, to_decimal=2):
    """
    Rounds each value in the list to two decimal places and returns the mean and standard deviation.

    Parameters:
        values (list of float): List of numeric values.

    Returns:
        tuple: A tuple (mean, std) of the rounded values.
    """
    # Round each value to two decimals
    rounded_values = [round(v, to_decimal) for v in values]

    # Compute mean and standard deviation using numpy
    mean_val = np.mean(rounded_values)
    std_val = np.std(rounded_values)

    return mean_val, std_val

# MOPs
def objective_functions(x, classifiers):
    assert classifiers is not None
    scores = []
    for classifier in classifiers:
        scores.append(classifier(x).squeeze())
    return scores

def get_pymoo_algo(args, n_partitions=12):

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions(
        "das-dennis", args.num_obj, n_partitions=n_partitions
    )

    # create the algorithm object
    if "nsga3" in args.method:
        algorithm = NSGA3(
            pop_size=args.num_points_sample, 
            ref_dirs=ref_dirs
        )
    elif "moead" in args.method:
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
        )
    else:
        raise ValueError("Invalid algorithm.")

    return algorithm

class pymoo_objective_functions(Problem):

    def __init__(self, device, n_var, n_obj, xl, xu, classifiers=None, **kwargs):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            vtype=float,
            **kwargs,
        )

        self.classifiers = classifiers
        self.device = device

    def _evaluate(self, x, out, *args, **kwargs):
        assert self.classifiers is not None
        scores = []
        with torch.no_grad():  # Disable gradient tracking
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            for classifier in self.classifiers:
                scores.append(classifier(x).squeeze().detach().cpu().numpy())
        out["F"] = np.column_stack(scores)

    def __deepcopy__(self, memo):
        # Create a new instance without deep copying the classifiers.
        # Instead, share the same classifier reference.
        new_instance = type(self)(
            device=self.device,
            n_var=self.n_var,
            classifiers=self.classifiers,  # Preserve the classifiers reference
            **{},  # add any additional parameters if needed
        )
        memo[id(self)] = new_instance
        return new_instance
    
    
