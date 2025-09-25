import torch
from pymoo.problems import get_problem
from problems.core import PymooProblemTorch
import math
from torch import Tensor

######## DTLZ Problems ########

class DTLZ(PymooProblemTorch):
    r"""Base class for DTLZ problems.

    See [Deb2005dtlz]_ for more details on DTLZ.
    """

    def __init__(self, n_var=30, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj,
                         xl=torch.zeros(n_var, dtype=torch.float) + 1e-6,
                         xu=torch.ones(n_var, dtype=torch.float) - 1e-6,
                         vtype=float, **kwargs)

        if n_var <= n_obj:
            raise ValueError(
                f"n_var must be > n_obj, but got {n_var} and {n_obj}."
            )
        self.continuous_inds = list(range(n_var))
        self.k = n_var - n_obj + 1
       
        
class DTLZ2(DTLZ):
    r"""DLTZ2 test problem.

    d-dimensional problem evaluated on `[0, 1]^d`:

        f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
        g(x) = \sum_{i=m}^{d-1} (x_i - 0.5)^2

    The pareto front is given by the unit hypersphere \sum{i} f_i^2 = 1.
    Note: the pareto front is completely concave. The goal is to minimize
    both objectives.
    """
    
    def pareto_front(self):
        return get_problem("dtlz2", n_var=self.n_var, n_obj=self.n_obj).pareto_front()

    def _evaluate(self, X: torch.Tensor, 
                  out: dict, *args, **kwargs) -> None:
        X_m = X[..., -self.k :]
        g_X = (X_m - 0.5).pow(2).sum(dim=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.n_obj):
            idx = self.n_obj - 1 - i
            f_i = g_X_plus1.clone()
            f_i *= torch.cos(X[..., :idx] * pi_over_2).prod(dim=-1)
            if i > 0:
                f_i *= torch.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        out["F"] = torch.stack(fs, dim=-1)

        
class DTLZ4(DTLZ):
    def pareto_front(self):
        return get_problem("dtlz4", n_var=self.n_var, n_obj=self.n_obj).pareto_front()

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        alpha  = 100
        g_X = (X_M - 0.5).pow(2).sum(dim=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.n_obj):
            idx = self.n_obj - 1 - i
            f_i = g_X_plus1.clone()
            f_i *= torch.cos((X_[..., :idx]**alpha) * pi_over_2).prod(dim=-1)
            if i > 0:
                f_i *= torch.sin((X_[..., idx]**alpha) * pi_over_2)
            fs.append(f_i)
        out["F"] = torch.stack(fs, dim=-1)
        
 
class DTLZ7(DTLZ):
    r"""DTLZ7 test problem.

    d-dimensional problem evaluated on `[0, 1]^d`:
        f_0(x) = x_0
        f_1(x) = x_1
        ...
        f_{M-1}(x) = (1 + g(X_m)) * h(f_0, f_1, ..., f_{M-2}, g, x)
        h(f_0, f_1, ..., f_{M-2}, g, x) =
        M - sum_{i=0}^{M-2} f_i(x)/(1+g(x)) * (1 + sin(3 * pi * f_i(x)))

    This test problem has 2M-1 disconnected Pareto-optimal regions in the search space.

    The pareto frontier corresponds to X_m = 0.
    """
    
    def pareto_front(self):
        return get_problem("dtlz7", n_var=self.n_var, n_obj=self.n_obj).pareto_front()

    def _evaluate(self, X: torch.Tensor, 
                  out: dict, *args, **kwargs) -> None:
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(X[..., i])
        f = torch.stack(f, dim=-1)

        g_X = 1 + 9 / self.k * torch.sum(X[..., -self.k :], dim=-1)
        h = self.n_obj - torch.sum(
            f / (1 + g_X.unsqueeze(-1)) * (1 + torch.sin(3 * math.pi * f)), dim=-1
        )
        out["F"] = torch.cat([f, ((1 + g_X) * h).unsqueeze(-1)], dim=-1)