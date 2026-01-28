import torch
import torch.nn as nn
from torch import Tensor
from pymoo.problems import get_problem
from moospread.problem import PymooProblemTorch
from importlib import resources
import os
import numpy as np
import math


######## RE Problems ########
# Note: For the sake of differentiability, we will use the strict bounds:
# - Lower bound: xl + 1e-6 (instead of xl)
# - Upper bound: xu - 1e-6 (instead of xu)
# ref_point: The default reference points are suitable for the "bayesian" mode.

"Adapted from: https://botorch.readthedocs.io/en/stable/_modules/botorch/test_functions/multi_objective.html"

class BraninCurrin(PymooProblemTorch):
    r"""Two objective problem composed of the Branin and Currin functions.

    Branin (rescaled):

        f(x) = (
        15*x_1 - 5.1 * (15 * x_0 - 5) ** 2 / (4 * pi ** 2) + 5 * (15 * x_0 - 5)
        / pi - 5
        ) ** 2 + (10 - 10 / (8 * pi)) * cos(15 * x_0 - 5))

    Currin:

        f(x) = (1 - exp(-1 / (2 * x_1))) * (
        2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
        ) / 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20

    """
    
    def __init__(self, path = None, ref_point=None, negate=True, **kwargs):
        super().__init__(n_var=2, n_obj=2, 
                         xl=torch.zeros(2, dtype=torch.float) + 1e-6,
                         xu=torch.ones(2, dtype=torch.float) - 1e-6,
                         vtype=float, **kwargs)
        if ref_point is None:
            self.ref_point = [18.0, 6.0]
        else:
            self.ref_point = ref_point
            
        self.path = path
        self.max_hv = 59.36011874867746  # this is approximated using NSGA-II
        
    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        if self.path is not None:
            front = np.loadtxt(self.path)
            return torch.from_numpy(front).to(self.device)
        else:
            return None
        
    def _branin(self, X: Tensor) -> Tensor:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * math.pi**2) * X[..., 0].pow(2)
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1.pow(2) + t2 + 10
    
    def _rescaled_branin(self, X: Tensor) -> Tensor:
        # return to Branin bounds
        x_0 = 15 * X[..., 0] - 5
        x_1 = 15 * X[..., 1]
        return self._branin(torch.stack([x_0, x_1], dim=-1))

    @staticmethod
    def _currin(X: Tensor) -> Tensor:
        x_0 = X[..., 0]
        x_1 = X[..., 1]
        factor1 = 1 - torch.exp(-1 / (2 * x_1))
        numer = 2300 * x_0.pow(3) + 1900 * x_0.pow(2) + 2092 * x_0 + 60
        denom = 100 * x_0.pow(3) + 500 * x_0.pow(2) + 4 * x_0 + 20
        return factor1 * numer / denom
    
    def _evaluate(self, X: torch.Tensor, out: dict, *args, **kwargs) -> None:
        # branin rescaled with inputsto [0,1]^2
        branin = self._rescaled_branin(X=X)
        currin = self._currin(X=X)
        out["F"] = torch.stack([branin, currin], dim=-1)
    
class Penicillin(PymooProblemTorch):
    r"""A penicillin production simulator from [Liang2021]_.

    This implementation is adapted from
    https://github.com/HarryQL/TuRBO-Penicillin.

    The goal is to maximize the penicillin yield while minimizing
    time to ferment and the CO2 byproduct.

    The function is defined for minimization of all objectives.

    The reference point was set using the `infer_reference_point` heuristic
    on the Pareto frontier obtained via NSGA-II.
    """
    
    Y_xs = 0.45
    Y_ps = 0.90
    K_1 = 10 ** (-10)
    K_2 = 7 * 10 ** (-5)
    m_X = 0.014
    alpha_1 = 0.143
    alpha_2 = 4 * 10 ** (-7)
    alpha_3 = 10 ** (-4)
    mu_X = 0.092
    K_X = 0.15
    mu_p = 0.005
    K_p = 0.0002
    K_I = 0.10
    K = 0.04
    k_g = 7.0 * 10**3
    E_g = 5100.0
    k_d = 10.0**33
    E_d = 50000.0
    lambd = 2.5 * 10 ** (-4)
    T_v = 273.0  # Kelvin
    T_o = 373.0
    R = 1.9872  # CAL/(MOL K)
    V_max = 180.0
    
    def __init__(self, path = None, ref_point=None, negate=True, **kwargs):
        super().__init__(n_var=7, n_obj=3, 
                         xl=torch.tensor([60.0, 0.05, 293.0, 0.05, 0.01, 500.0, 5.0], dtype=torch.float) + 1e-6,
                         xu=torch.tensor([120.0, 18.0, 303.0, 18.0, 0.5, 700.0, 6.5], dtype=torch.float) - 1e-6,
                         vtype=float, **kwargs)
        if ref_point is None:
            self.ref_point = [25.935, 57.612, 935.5]
        else:
            self.ref_point = ref_point
            
        self.path = path
        self.max_hv = 2183455.909507436
        
    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        if self.path is not None:
            front = np.loadtxt(self.path)
            return torch.from_numpy(front).to(self.device)
        else:
            return None

    @classmethod
    def penicillin_vectorized(cls, X_input: Tensor) -> Tensor:
        r"""Penicillin simulator, simplified and vectorized.

        The 7 input parameters are (in order): culture volume, biomass
        concentration, temperature, glucose concentration, substrate feed
        rate, substrate feed concentration, and H+ concentration.

        Args:
            X_input: A `n x 7`-dim tensor of inputs.

        Returns:
            An `n x 3`-dim tensor of (negative) penicillin yield, CO2 and time.
        """
        V, X, T, S, F, s_f, H_ = torch.split(X_input, 1, -1)
        P, CO2 = torch.zeros_like(V), torch.zeros_like(V)
        H = torch.full_like(H_, 10.0).pow(-H_)

        active = torch.ones_like(V).bool()
        t_tensor = torch.full_like(V, 2500)

        for t in range(1, 2501):
            if active.sum() == 0:
                break
            F_loss = (
                V[active]
                * cls.lambd
                * torch.special.expm1(5 * ((T[active] - cls.T_o) / (cls.T_v - cls.T_o)))
            )
            dV_dt = F[active] - F_loss
            mu = (
                (cls.mu_X / (1 + cls.K_1 / H[active] + H[active] / cls.K_2))
                * (S[active] / (cls.K_X * X[active] + S[active]))
                * (
                    (cls.k_g * torch.exp(-cls.E_g / (cls.R * T[active])))
                    - (cls.k_d * torch.exp(-cls.E_d / (cls.R * T[active])))
                )
            )
            dX_dt = mu * X[active] - (X[active] / V[active]) * dV_dt
            mu_pp = cls.mu_p * (
                S[active] / (cls.K_p + S[active] + S[active].pow(2) / cls.K_I)
            )
            dS_dt = (
                -(mu / cls.Y_xs) * X[active]
                - (mu_pp / cls.Y_ps) * X[active]
                - cls.m_X * X[active]
                + F[active] * s_f[active] / V[active]
                - (S[active] / V[active]) * dV_dt
            )
            dP_dt = (
                (mu_pp * X[active])
                - cls.K * P[active]
                - (P[active] / V[active]) * dV_dt
            )
            dCO2_dt = cls.alpha_1 * dX_dt + cls.alpha_2 * X[active] + cls.alpha_3

            # UPDATE
            P[active] = P[active] + dP_dt  # Penicillin concentration
            V[active] = V[active] + dV_dt  # Culture medium volume
            X[active] = X[active] + dX_dt  # Biomass concentration
            S[active] = S[active] + dS_dt  # Glucose concentration
            CO2[active] = CO2[active] + dCO2_dt  # CO2 concentration

            # Update active indices
            full_dpdt = torch.ones_like(P)
            full_dpdt[active] = dP_dt
            inactive = (V > cls.V_max) + (S < 0) + (full_dpdt < 10e-12)
            t_tensor[inactive] = torch.minimum(
                t_tensor[inactive], torch.full_like(t_tensor[inactive], t)
            )
            active[inactive] = 0

        return torch.stack([-P, CO2, t_tensor], dim=-1)

    def _evaluate(self, X: torch.Tensor, out: dict, *args, **kwargs) -> None:
        # This uses in-place operations. Hence, the clone is to avoid modifying
        # the original X in-place.
        out["F"] = self.penicillin_vectorized(X.view(-1, self.dim).clone()).view(
            *X.shape[:-1], self.num_objectives
        )
        

class VehicleSafety(PymooProblemTorch):
    r"""Optimize Vehicle crash-worthiness.

    See [Tanabe2020]_ for details.

    The reference point is 1.1 * the nadir point from
    approximate front provided by [Tanabe2020]_.

    The maximum hypervolume is computed using the approximate
    pareto front from [Tanabe2020]_.
    """
    
    def __init__(self, path = None, ref_point=None, negate=True, **kwargs):
        super().__init__(n_var=5, n_obj=3, 
                         xl=torch.ones(5, dtype=torch.float) + 1e-6,
                         xu=3*torch.ones(5, dtype=torch.float) - 1e-6,
                         vtype=float, **kwargs)
        if ref_point is None:
            self.ref_point = [1864.72022, 11.81993945, 0.2903999384]
        else:
            self.ref_point = ref_point
            
        self.path = path
        self.max_hv = 246.81607081187002
        
    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        if self.path is not None:
            front = np.loadtxt(self.path)
            return torch.from_numpy(front).to(self.device)
        else:
            return None


    def _evaluate(self, X: torch.Tensor, out: dict, *args, **kwargs) -> None:
        X1, X2, X3, X4, X5 = torch.split(X, 1, -1)
        f1 = (
            1640.2823
            + 2.3573285 * X1
            + 2.3220035 * X2
            + 4.5688768 * X3
            + 7.7213633 * X4
            + 4.4559504 * X5
        )
        f2 = (
            6.5856
            + 1.15 * X1
            - 1.0427 * X2
            + 0.9738 * X3
            + 0.8364 * X4
            - 0.3695 * X1 * X4
            + 0.0861 * X1 * X5
            + 0.3628 * X2 * X4
            - 0.1106 * X1.pow(2)
            - 0.3437 * X3.pow(2)
            + 0.1764 * X4.pow(2)
        )
        f3 = (
            -0.0551
            + 0.0181 * X1
            + 0.1024 * X2
            + 0.0421 * X3
            - 0.0073 * X1 * X2
            + 0.024 * X2 * X3
            - 0.0118 * X2 * X4
            - 0.0204 * X3 * X4
            - 0.008 * X3 * X5
            - 0.0241 * X2.pow(2)
            + 0.0109 * X4.pow(2)
        )
        f_X = torch.cat([f1, f2, f3], dim=-1)
        out["F"] = f_X
