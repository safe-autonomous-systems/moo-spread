"""
This module defines an abstract base class for 
multi-objective optimization problems using PyTorch.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from pymoo.util.cache import Cache


def default_shape(problem, n):
    n_var = problem.n_var
    return {
        'F': (n, problem.n_obj),
        'G': (n, problem.n_ieq_constr),
        'H': (n, problem.n_eq_constr),
        'dF': (n, problem.n_obj, n_var),
        'dG': (n, problem.n_ieq_constr, n_var),
        'dH': (n, problem.n_eq_constr, n_var),
        'CV': (n, 1),
    }

class PymooProblemTorch(ABC):
    def __init__(
        self,
        n_var: int = -1,
        n_obj: int = 1,
        n_ieq_constr: int = 0,
        n_eq_constr: int = 0,
        xl=None,
        xu=None,
        device: torch.device = None,
        elementwise: bool = False,
        replace_nan_by: float = None,
        strict: bool = True,
        **kwargs
    ):
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr = n_eq_constr
        self.device = device or torch.device('cpu')
        self.elementwise = elementwise
        self.replace_nan_by = replace_nan_by
        self.strict = strict
        self.ref_point = None

        # bounds
        if n_var > 0 and xl is not None:
            self.xl = (
                xl
                if isinstance(xl, torch.Tensor)
                else torch.full((n_var,), xl, dtype=torch.float, device=self.device)
            )
        else:
            self.xl = None
        if n_var > 0 and xu is not None:
            self.xu = (
                xu
                if isinstance(xu, torch.Tensor)
                else torch.full((n_var,), xu, dtype=torch.float, device=self.device)
            )
        else:
            self.xu = None

        self.is_discrete = False
        self.is_sequence = False
        self.original_bounds = None
        self.need_repair = False
        if self.has_bounds():
            self.original_bounds = [self.xl.clone(), self.xu.clone()]
            self.global_clamping = False
            self.need_repair = True

    def has_bounds(self) -> bool:
        return self.xl is not None and self.xu is not None

    def has_constraints(self) -> bool:
        return (self.n_ieq_constr + self.n_eq_constr) > 0

    def name(self) -> str:
        return self.__class__.__name__
    
    def bounds(self):
        return [self.xl, self.xu]
    
    def evaluate(
        self,
        x: torch.Tensor,
        return_as_dict: bool = False,
        return_values_of=None
    ):
        """
        Evaluate objectives on input tensor x.

        Args:
            x: Tensor of shape (batch_size, n_var) with requires_grad=True.
            return_as_dict: If True, return the full dict of computed values.
            return_values_of: Optional list of keys from the internal out dict to return alongside F.

        Returns:
            If return_as_dict=True: the out dict (including all computed values).
            Else if return_values_of provided: a tuple (F, dict_of_requested_values).
            Else: Tensor F of shape (batch_size, n_obj).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        x.data = torch.nan_to_num(x.data, nan=0.0, posinf=1.0, neginf=0.0)

        # Ensure is is within bounds
        if self.has_bounds():
            if self.xl is not None:
                x.data = torch.max(x.data, self.xl.to(x.device))
            if self.xu is not None:
                x.data = torch.min(x.data, self.xu.to(x.device))

        out: dict = {}
        self._evaluate(x, out)

        # Prepare base return
        F = out.get("F")
        # If specific extra values requested
        extra = None
        if return_values_of is not None:
            if isinstance(return_values_of, (list, tuple)):
                extra = {k: out[k] for k in return_values_of if k in out}
            else:
                key = return_values_of
                extra = {key: out[key]} if key in out else {}

        if return_as_dict:
            if extra is not None:
                out.update(extra)
            return out

        if extra is not None:
            if len(extra) == 1:
                return list(extra.values())[0]
            return tuple(extra[k] for k in extra)

        return F

    @abstractmethod
    def _evaluate(self, x, out: dict):
        """
        User-implemented evaluation. 
        If elementwise: x is 1d Tensor of shape (n_var,).
        Otherwise x is 2d Tensor of shape (n_samples, n_var).
        out should be populated with keys 'F', 'G', 'H' as torch.Tensors.
        """
        pass
    
    @Cache
    def pareto_front(self, *args, **kwargs):
        pf = self._calc_pareto_front(*args, **kwargs)
        if pf is None:
            return None
        if not isinstance(pf, torch.Tensor):
            pf = torch.as_tensor(pf, dtype=torch.float32)
        if pf.dim() == 1:
            pf = pf.unsqueeze(0)
        if pf.size(1) == 2:
            pf = pf[pf[:, 0].argsort()]
        return pf

    def __str__(self):
        return (
            f"Problem(name={self.name()}, n_var={self.n_var}, n_obj={self.n_obj}, "
            f"n_ieq_constr={self.n_ieq_constr}, n_eq_constr={self.n_eq_constr})"
        )


class BaseProblem(PymooProblemTorch):
    """
    Base Problem class to ensure PyTorch differentiability.
    """

    def __init__(self, n_var=30, n_obj=2, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, 
                         xl=torch.zeros(n_var, dtype=torch.float) + 1e-6,
                         xu=torch.ones(n_var, dtype=torch.float) - 1e-6,
                         vtype=float, **kwargs)
        
    def _calc_pareto_front(self, 
                           n_pareto_points: int = 100) -> torch.Tensor:
        pass

    def _evaluate(self, x: torch.Tensor, 
                  out: dict, 
                  *args, 
                  **kwargs) -> None:
        pass