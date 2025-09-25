import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import os
import sys

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a simple neural network
class VectorFieldNet(nn.Module):
    def __init__(self, D, M=512):
        super(VectorFieldNet, self).__init__()

        self.D = D
        self.M = M
        self.net = nn.Sequential(
            nn.Linear(D, M),
            nn.SELU(),
            nn.Linear(M, M),
            nn.SELU(),
            nn.Linear(M, M),
            nn.SELU(),
            nn.Linear(M, D),
        )

    def forward(self, x):
        return self.net(x)
