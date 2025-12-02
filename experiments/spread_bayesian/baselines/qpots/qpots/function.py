from botorch.test_functions.multi_objective import (
    BraninCurrin, DTLZ2, DTLZ5, DTLZ7, Penicillin,
    CarSideImpact,
    ZDT3, ZDT1, ZDT2,
)
from torch import Tensor
from typing import Callable, Optional
import torch


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
        dim: int = 2,
        nobj: int = 2,
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
        self.dim = dim
        self.nobj = nobj
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
            "dtlz2": lambda: DTLZ2(self.dim, num_objectives=self.nobj, negate=True),
            "dtlz5": lambda: DTLZ5(self.dim, num_objectives=self.nobj, negate=True),
            "dtlz7": lambda: DTLZ7(self.dim, num_objectives=self.nobj, negate=True),
            "penicillin": lambda: Penicillin(negate=True),
            "carside": lambda: CarSideImpact(negate=True),
            "zdt1": lambda: ZDT1(dim=self.dim, num_objectives=self.nobj, negate=True),
            "zdt2": lambda: ZDT2(dim=self.dim, num_objectives=self.nobj, negate=True),
            "zdt3": lambda: ZDT3(dim=self.dim, num_objectives=self.nobj, negate=True),
        }

        if self.name in func_map:
            # Initialize function, bounds, and constraints
            self.func = func_map[self.name]()
        elif self.custom_func is not None:
            self.func = self.custom_func
        else:
            raise ValueError(f"Unknown test function '{self.name}'. Check the available functions.")
        if self.custom_func is None:
            self.bounds = self.func.bounds.double()
        if hasattr(self.func, "evaluate_slack"):
            self.cons = self.func.evaluate_slack
            
        if self.dim is None:
            self.dim = self.func.dim
        if self.nobj is None:
            self.nobj = self.func.num_objectives
            
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
        # X = torch.from_numpy(X)
        lbound, ubound = self.bounds
        X = torch.clamp(X, min=lbound, max=ubound)
        # Ensure x is not a constant batch (rare case)
        mn, mx = X.min(), X.max()
        if mn == mx:
            noise = torch.empty_like(X).uniform_(-1e6, 1e6)
            X = X + noise.to(X.device)
            
        if self.custom_func:
            return torch.from_numpy(self.custom_func.evaluate(X.detach().cpu().numpy())).to(X.device, dtype=X.dtype)
        return self.func(X) #.cpu().numpy()

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
        if self.custom_func:
            return None
        return self.func.ref_point
