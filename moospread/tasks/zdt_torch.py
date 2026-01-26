import torch
import torch.nn as nn
from pymoo.problems import get_problem
from moospread.problem import PymooProblemTorch
import numpy as np

######## ZDT Problems ########
# Note: For the sake of differentiability, we will use the strict bounds:
# - Lower bound: 0 + 1e-6 (instead of 0)
# - Upper bound: 1 - 1e-6 (instead of 1)
# ref_point: The default reference points are suitable for the "online" mode.

class ZDT(PymooProblemTorch):
    """
    Base class to ensure PyTorch differentiability.
    Provides a default `evaluate` that preserves gradients and guards against NaNs.
    """

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, 
                         xl=torch.zeros(n_var, dtype=torch.float) + 1e-6,
                         xu=torch.ones(n_var, dtype=torch.float) - 1e-6,
                         vtype=float, **kwargs)
        
class ZDT1(ZDT):
    """
    ZDT1 test problem in PyTorch, fully differentiable.
    """
    def __init__(self, n_var=30, ref_point=None, **kwargs):
        super().__init__(n_var, **kwargs)
        if ref_point is None:
            self.ref_point = [0.9994, 6.0576]
        else:
            self.ref_point = ref_point
    
    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        x = torch.linspace(0.0, 1.0, n_pareto_points, device=self.device)
        return torch.stack([x, 1.0 - torch.sqrt(x)], dim=1)

    def _evaluate(self, x: torch.Tensor, out: dict, *args, **kwargs) -> None:
        # x: (batch_size, n_var)
        # Objective f1
        f1 = x[:, 0]
        # Auxiliary g
        g = 1.0 + 9.0 / (self.n_var - 1) * torch.sum(x[:, 1:], dim=1)
        # Avoid negative division
        term = torch.clamp(f1 / g, min=0.0)
        # Objective f2
        f2 = g * (1.0 - torch.sqrt(term))
        out["F"] = torch.stack([f1, f2], dim=1)

class ZDT2(ZDT):
    """
    ZDT2 test problem in PyTorch, fully differentiable.
    """
    def __init__(self, n_var=30, ref_point=None, **kwargs):
        super().__init__(n_var, **kwargs)
        if ref_point is None:
            self.ref_point = [0.9994, 6.8960]
        else:
            self.ref_point = ref_point
            
    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        x = torch.linspace(0.0, 1.0, n_pareto_points, device=self.device)
        return torch.stack([x, 1.0 - x.pow(2)], dim=1)

    def _evaluate(self, x: torch.Tensor, out: dict, *args, **kwargs) -> None:
        f1 = x[:, 0]
        c = torch.sum(x[:, 1:], dim=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        term = torch.clamp(f1 / g, min=0.0)
        f2 = g * (1.0 - term.pow(2))
        out["F"] = torch.stack([f1, f2], dim=1)

class ZDT3(ZDT):
    """
    ZDT3 test problem in PyTorch, fully differentiable.
    """
    def __init__(self, n_var=30, ref_point=None, **kwargs):
        super().__init__(n_var, **kwargs)
        if ref_point is None:
            self.ref_point = [0.9994, 6.0571]
        else:
            self.ref_point = ref_point
     
    def _calc_pareto_front(
        self,
        n_points: int = 100,
        flatten: bool = True
    ) -> torch.Tensor:
        regions = [
            [0.0, 0.0830015349],
            [0.182228780, 0.2577623634],
            [0.4093136748, 0.4538821041],
            [0.6183967944, 0.6525117038],
            [0.8233317983, 0.8518328654]
        ]
        pf_list = []
        points_per_region = int(n_points / len(regions))
        for r in regions:
            x1 = torch.linspace(r[0], r[1], points_per_region, device=self.device)
            x2 = 1.0 - torch.sqrt(x1) - x1 * torch.sin(10.0 * torch.pi * x1)
            pf_list.append(torch.stack([x1, x2], dim=1))
        return torch.cat(pf_list, dim=0) if flatten else torch.stack(pf_list, dim=0)

    def _evaluate(self, x: torch.Tensor, out: dict, *args, **kwargs) -> None:
        f1 = x[:, 0]
        c = torch.sum(x[:, 1:], dim=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        term = torch.clamp(f1 / g, min=0.0)
        f2 = g * (1.0 - torch.sqrt(term) - term * torch.sin(10.0 * torch.pi * f1))
        out["F"] = torch.stack([f1, f2], dim=1)
