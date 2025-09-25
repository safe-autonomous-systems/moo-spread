import torch
import numpy as np

from re_eng import Function

def get(name, *args, **kwargs):
    name = name.lower()

    if "re" not in name:
        try:
            # Set up problem
            tf = Function(name, n_var = None, n_obj=None)
            bounds = tf.get_bounds()
            ref_point = -1*tf.get_ref_point()  # negative reference point for minimization
        except Exception as e:
            raise Exception(f"Error initializing problem {name}: {e}")
        
        return tf, bounds, ref_point