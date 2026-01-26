import numpy as np
import random
import torch


def set_seed(seed: int = 0):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True   

def convert_seconds(seconds):
    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    # Format the result
    print(f"Time: {hours} hours {minutes} minutes {remaining_seconds} seconds")