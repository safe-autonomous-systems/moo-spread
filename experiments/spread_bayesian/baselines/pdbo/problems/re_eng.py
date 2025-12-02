from botorch.test_functions.multi_objective import (
    BraninCurrin, Penicillin,
     CarSideImpact,
)
from torch import Tensor
from typing import Callable, Optional
import torch

import autograd.numpy as anp

from .problem import Problem
    
class CARSIDE(Problem):
    def __init__(self, n_var=7, n_obj=4):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=anp.double)
        
        self.Obj = Function(name = "carside", n_var = None, n_obj = None)
        
        self.n_var = self.Obj.n_var
        self.n_obj = self.Obj.n_obj
        self.xl, self.xu = self.Obj.get_bounds()
        self.xl = self.xl.numpy()
        self.xu = self.xu.numpy()
        self.ref_point = -1*self.Obj.get_ref_point()
        
    def _evaluate_F(self, x):
        return self.Obj.evaluate(x)
    
    def name(self):
        return "CarSideImpact"
    
    
class PENICILLIN(Problem):
    def __init__(self, n_var=7, n_obj=3):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=anp.double)

        self.Obj = Function(name = "penicillin", n_var = None, n_obj = None)
        
        self.n_var = self.Obj.n_var
        self.n_obj = self.Obj.n_obj
        self.xl, self.xu = self.Obj.get_bounds()
        self.xl = self.xl.numpy()
        self.xu = self.xu.numpy()
        self.ref_point = -1*self.Obj.get_ref_point()
        
    def _evaluate_F(self, x):
        return self.Obj.evaluate(x)
    
    def name(self):
        return "Penicillin"

class BRANINCURRIN(Problem):
    def __init__(self, n_var=2, n_obj=2):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=anp.double)

        self.Obj = Function(name = "branincurrin", n_var = None, n_obj = None)

        self.n_var = self.Obj.n_var
        self.n_obj = self.Obj.n_obj
        self.xl, self.xu = self.Obj.get_bounds()
        self.xl = self.xl.numpy()
        self.xu = self.xu.numpy()
        self.ref_point = -1*self.Obj.get_ref_point()
        
    def _evaluate_F(self, x):
        return self.Obj.evaluate(x)
    
    def name(self):
        return "BraninCurrin"

class Function:
    """
    Interface for multi-objective test functions.

    This class provides an abstraction over BoTorch test functions and allows for 
    user-defined objective functions. It supports retrieving function bounds and 
    constraints when available.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        n_var: int = 2,
        n_obj: int = 2,
        custom_func: Optional[Callable[[Tensor], Tensor]] = None,
        bounds: Optional[Tensor] = None,
        cons: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """
        Initialize a test function for multi-objective optimization.

        Parameters
        ----------
        name : str, optional
            Name of the predefined test function (case-insensitive). 
            If None, a custom function must be provided.
        dim : int
            Dimensionality of the input space. Defaults to 2.
        nobj : int
            Number of objectives for the test function. Defaults to 2.
        custom_func : Callable, optional
            A user-defined function that takes a tensor `X` as input and 
            returns an output tensor. If provided, `name` is ignored.
        bounds : Tensor, optional
            A tensor specifying the lower and upper bounds for the function.
            Required if using a custom function.
        cons : Callable, optional
            A constraint function that maps inputs to constraint values.

        Raises
        ------
        ValueError
            If a custom function is provided but `bounds` is not specified.
            If an unknown function name is provided.
        """
        self.name = name.lower() if name else None
        self.n_var = n_var
        self.n_obj = n_obj
        self.custom_func = custom_func
        self.bounds = bounds
        self.cons = cons

        if self.custom_func:
            if self.bounds is None:
                raise ValueError("Custom functions must specify bounds.")
        else:
            self._initialize_function()

    def _initialize_function(self):
        """
        Initialize a predefined BoTorch test function.

        This method sets up the corresponding function, bounds, and constraints 
        based on the selected function name.

        Raises
        ------
        ValueError
            If the specified function name is not recognized.
        """
        func_map = {
            "branincurrin": lambda: BraninCurrin(negate=True),
            "penicillin": lambda: Penicillin(negate=True),
            "carside": lambda: CarSideImpact(negate=True),
        }

        if self.name not in func_map:
            raise ValueError(f"Unknown test function '{self.name}'. Check the available functions.")

        # Initialize function, bounds, and constraints
        self.func = func_map[self.name]()
        self.bounds = self.func.bounds.double()
        if hasattr(self.func, "evaluate_slack"):
            self.cons = self.func.evaluate_slack
            
        if self.n_var is None:
            self.n_var = self.func.dim
        if self.n_obj is None:
            self.n_obj = self.func.num_objectives
            
    def evaluate(self, X: Tensor) -> Tensor:
        """
        Evaluate the test function or custom function on input `X`.

        Parameters
        ----------
        X : Tensor
            A tensor of shape `(n, dim)`, where `n` is the number of points and `dim` is the input dimension.

        Returns
        -------
        Tensor
            A tensor of shape `(n, nobj)` containing the function outputs.
        """
        
        X = torch.from_numpy(X)
        xl, xu = self.bounds
        X = torch.clamp(X, min=xl, max=xu)
        mn, mx = X.min(), X.max()
        if mn == mx:
            noise = torch.empty_like(X).uniform_(-1e6, 1e6)
            X = X + noise.to(X.device)
        if self.custom_func:
            return self.custom_func(X)
        return -1 * self.func(X).cpu().numpy()

    def get_bounds(self) -> Tensor:
        """
        Retrieve the bounds for the function.

        Returns
        -------
        Tensor
            A tensor containing the lower and upper bounds for each input dimension.
        """
        return self.bounds

    def get_cons(self) -> Optional[Callable]:
        """
        Retrieve the constraint function for the test function.

        Returns
        -------
        Callable or None
            The constraint function if available; otherwise, None.
        """
        return self.cons
    
    def get_ref_point(self) -> Tensor:
        """
        Retrieve the reference point for the function.

        Returns
        -------
        Tensor
            A tensor containing the reference point for the function.
        """
        return self.func.ref_point