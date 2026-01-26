import torch
import torch.nn as nn
from pymoo.problems import get_problem
from pymoo.util.remote import Remote
from moospread.problem import PymooProblemTorch
import numpy as np

######## MW Problems ########
# Note: For the sake of differentiability, we will use the strict bounds:
# - Lower bound: 0 + 1e-6 (instead of 0)
# - Upper bound: 1 - 1e-6 (instead of 1)
# ref_point: The default reference points are suitable for the "online" mode.

import torch
from torch import Tensor
from typing import Optional

class MW(PymooProblemTorch):
    def __init__(self, n_var, n_obj,
                n_ieq_constr: int = 0,
                n_eq_constr: int = 0,
                **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, 
                         xl=torch.zeros(n_var, dtype=torch.float) + 1e-6,
                         xu=torch.ones(n_var, dtype=torch.float) - 1e-6,
                         n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr,
                         vtype=float, **kwargs)
        
    # ---- smooth building blocks (torch versions of LA1/2/3) ----
    @staticmethod
    def LA1(A, B, C, D, theta: Tensor) -> Tensor:
        # A * (sin(B * pi * theta^C))^D
        t = theta.pow(C)
        return torch.as_tensor(A, dtype=theta.dtype, device=theta.device) * \
               torch.sin(torch.as_tensor(B, dtype=theta.dtype, device=theta.device) * torch.pi * t).pow(D)

    @staticmethod
    def LA2(A, B, C, D, theta: Tensor) -> Tensor:
        # A * (sin(B * theta^C))^D
        t = theta.pow(C)
        return torch.as_tensor(A, dtype=theta.dtype, device=theta.device) * \
               torch.sin(torch.as_tensor(B, dtype=theta.dtype, device=theta.device) * t).pow(D)

    @staticmethod
    def LA3(A, B, C, D, theta: Tensor) -> Tensor:
        # A * (cos(B * theta^C))^D
        t = theta.pow(C)
        return torch.as_tensor(A, dtype=theta.dtype, device=theta.device) * \
               torch.cos(torch.as_tensor(B, dtype=theta.dtype, device=theta.device) * t).pow(D)

    # ---- distance/landscape functions ----
    def g1(self, X: Tensor) -> Tensor:
        """
        X: (batch, n_var) tensor
        returns: (batch,) tensor
        """
        if not isinstance(X, Tensor):
            X = torch.as_tensor(X, dtype=torch.get_default_dtype())
        d = self.n_var
        n = d - self.n_obj

        # z shape: (batch, d - (n_obj-1))
        z = X[:, self.n_obj - 1:].pow(n)

        # i shape: (d - (n_obj-1),)
        i = torch.arange(self.n_obj - 1, d, device=X.device, dtype=X.dtype)
        offset = 0.5 + i / (2.0 * d)  # broadcast over batch
        delta = z - offset
        exp_term = 1.0 - torch.exp(-10.0 * (delta * delta))
        distance = 1.0 + exp_term.sum(dim=1)
        return distance

    def g2(self, X: Tensor) -> Tensor:
        """
        X: (batch, n_var) tensor
        returns: (batch,) tensor
        """
        if not isinstance(X, Tensor):
            X = torch.as_tensor(X, dtype=torch.get_default_dtype())
        d = self.n_var
        n = float(d)

        i = torch.arange(self.n_obj - 1, d, device=X.device, dtype=X.dtype)  # (len,)
        z = 1.0 - torch.exp(-10.0 * (X[:, self.n_obj - 1:] - i / n).pow(2))
        contrib = (0.1 / n) * (z * z) + 1.5 - 1.5 * torch.cos(2.0 * torch.pi * z)
        distance = 1.0 + contrib.sum(dim=1)
        return distance

    def g3(self, X: Tensor) -> Tensor:
        """
        X: (batch, n_var) tensor
        returns: (batch,) tensor
        """
        if not isinstance(X, Tensor):
            X = torch.as_tensor(X, dtype=torch.get_default_dtype())

        a = X[:, self.n_obj - 1:]                         # last block
        b = X[:, self.n_obj - 2:-1] - 0.5                 # previous block (shifted)
        inner = a + (b * b) - 1.0
        contrib = 2.0 * inner.pow(2)
        distance = 1.0 + contrib.sum(dim=1)
        return distance


class MW7(MW):
    def __init__(self, n_var: int = 15, ref_point=None, **kwargs):
        super().__init__(n_var=n_var, 
                         n_obj=2, 
                         n_ieq_constr=2, **kwargs)
        if ref_point is None:
            self.ref_point = [2.0, 2.0]
        else:
            self.ref_point = ref_point
        
    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        return Remote.get_instance().load("pymoo", "pf", "MW", "MW7.pf")

    @torch.no_grad()
    def _check_shapes(self, X: Tensor):
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"X must have shape (batch, {self.n_var})")

    def _evaluate(self, X: torch.Tensor, out: dict, *args, **kwargs) -> None:
        """
        X: (batch, n_var) tensor
        returns: dict with
            F: (batch, 2)
            G: (batch, 2)
        """
        if not isinstance(X, Tensor):
            X = torch.as_tensor(X, dtype=torch.get_default_dtype())
        self._check_shapes(X)

        # g from MW.g3 (>= 1, so safe to divide by)
        g = self.g3(X)                       # (batch,)

        f0 = g * X[:, 0]                     # (batch,)
        # 1 - (f0/g)^2, clamped for numerical safety
        one_minus_sq = 1.0 - (f0 / g).pow(2)
        f1 = g * torch.sqrt(torch.clamp(one_minus_sq, min=0.0))

        # atan2 handles all quadrants and f0=0 safely (better than arctan(f1/f0))
        atan = torch.atan2(f1, f0)          # (batch,)

        # Constraints via LA2 (uses sin(B * theta^C)^D)
        term0 = 1.2 + torch.abs(self.LA2(0.4, 4.0, 1.0, 16.0, atan))
        g0 = f0.pow(2) + f1.pow(2) - term0.pow(2)

        term1 = 1.15 - self.LA2(0.2, 4.0, 1.0, 8.0, atan)
        g1 = term1.pow(2) - f0.pow(2) - f1.pow(2)

        F = torch.stack([f0, f1], dim=1)    # (batch, 2)
        G = torch.stack([g0, g1], dim=1)    # (batch, 2)
        out["F"] = F
        out["G"] = G



















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
