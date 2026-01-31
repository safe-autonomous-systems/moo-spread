import torch
import torch.nn as nn
from pymoo.problems import get_problem
from moospread.problem import PymooProblemTorch
from importlib import resources
import os
import numpy as np

######## RE Problems ########
# Note: For the sake of differentiability, we will use the strict bounds:
# - Lower bound: xl + 1e-6 (instead of xl)
# - Upper bound: xu - 1e-6 (instead of xu)
# ref_point: The default reference points are suitable for the "offline" mode.

class RE21(PymooProblemTorch):
    def __init__(self, path = None, ref_point=None, **kwargs):
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma  # = 1.
        super().__init__(n_var=4, n_obj=2, 
                         xl=torch.tensor([tmp_val, 
                                          (2.0**0.5)*tmp_val, 
                                          (2.0**0.5)*tmp_val, 
                                          tmp_val]) + 1e-6,
                         xu=torch.full((4,), 
                                       3.0*tmp_val) - 1e-6,
                         vtype=float, **kwargs)
        if ref_point is None:
            self.ref_point = [3144.44, 0.05]
            # self.ref_point = [3144.44, 0.05] # suitable for "online" mode
        else:
            self.ref_point = ref_point
            
        self.path = path
    
    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        try:
            front = np.loadtxt(self.path)
        except:
            front = np.loadtxt(resources.files("moospread.tasks.pf_re_tasks").joinpath(f"reference_points_{self.__class__.__name__}.dat"))
        else:
            assert self.path is not None, "Path to Pareto front file not specified."
        return torch.from_numpy(front).to(self.device)

    def _evaluate(self, X: torch.Tensor, out: dict, *args, **kwargs) -> None:
        """
        X: (N, 4) with columns [x1, x2, x3, x4]
        Returns F: (N, 2) with [f0, f1]
        """
        # ensure batch dimension
        if X.dim() == 1:
            X = X.unsqueeze(0)

        if X.size(-1) != 4:
            raise ValueError("X must be a (N, 4) tensor.")

        x1, x2, x3, x4 = X.unbind(dim=1)

        # constants
        F = 10.0
        sigma = 10.0
        E = 2.0e5
        L = 200.0
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=X.dtype, device=X.device))

        # objectives (vectorized)
        f0 = L * (2.0 * x1 + sqrt2 * x2 + torch.sqrt(x3) + x4)

        # small clamp for numerical safety (bounds already keep > 0)
        eps = 1e-12
        x1c = torch.clamp(x1, min=eps)
        x2c = torch.clamp(x2, min=eps)
        x3c = torch.clamp(x3, min=eps)
        x4c = torch.clamp(x4, min=eps)

        f1 = ((F * L) / E) * (
            (2.0 / x1c) + (2.0 * sqrt2 / x2c) - (2.0 * sqrt2 / x3c) + (2.0 / x4c)
        )

        out["F"] = torch.stack([f0, f1], dim=1)  # (N, 2)
        

class RE33(PymooProblemTorch):
    def __init__(self, path=None, ref_point=None, **kwargs):
        super().__init__(n_var=4, n_obj=3, 
                         xl=torch.tensor([55.0, 75.0, 1000.0, 11.0])+ 1e-6,
                         xu=torch.tensor([80.0, 110.0, 3000.0, 20.0])- 1e-6,
                         vtype=float, **kwargs)
        if ref_point is None:
            self.ref_point = [8.01,    8.84, 2343.30]
            # self.ref_point = [5.01, 9.84, 4.30] # suitable for "online" mode
        else:
            self.ref_point = ref_point
            
        self.path = path
    
    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        try:
            front = np.loadtxt(self.path)
        except:
            front = np.loadtxt(resources.files("moospread.tasks.pf_re_tasks").joinpath(f"reference_points_{self.__class__.__name__}.dat"))
        else:
            assert self.path is not None, "Path to Pareto front file not specified."
        return torch.from_numpy(front).to(self.device)


    def _evaluate(self, X: torch.Tensor, out: dict, *args, **kwargs) -> None:
        """
        X: (N, 4) with columns [x1, x2, x3, x4]
        Returns: (N, 3) with [f0, f1, f2], where f2 is sum of constraint violations.
        """
        if X.dim() != 2 or X.size(-1) != 4:
            raise ValueError("X must be a (N, 4) tensor.")

        x1, x2, x3, x4 = X.unbind(dim=1)

        # ----- Objectives -----
        # f0 = 4.9e-5 * (x2^2 - x1^2) * (x4 - 1)
        d2 = x2**2 - x1**2
        f0 = 4.9e-5 * d2 * (x4 - 1.0)

        # f1 = (9.82e6) * (x2^2 - x1^2) / (x3 * x4 * (x2^3 - x1^3))
        d3 = x2**3 - x1**3
        eps = torch.tensor(1e-12, dtype=X.dtype, device=X.device)

        def safe_signed(x):
            s = torch.where(x >= 0, 1.0, -1.0)
            return s * torch.clamp(x.abs(), min=eps)

        denom_f1 = (torch.clamp(x3, min=eps) *
                    torch.clamp(x4, min=eps) *
                    safe_signed(d3))
        f1 = (9.82e6) * d2 / denom_f1

        # ----- Constraints (g >= 0 means satisfied) -----
        # g0 = (x2 - x1) - 20
        g0 = (x2 - x1) - 20.0

        # g1 = 0.4 - x3 / (pi * (x2^2 - x1^2))
        denom_g1 = torch.pi * safe_signed(d2)
        g1 = 0.4 - (x3 / denom_g1)

        # g2 = 1 - (2.22e-3 * x3 * (x2^3 - x1^3)) / (x2^2 - x1^2)^2
        denom_g2 = safe_signed(d2)**2
        g2 = 1.0 - (2.22e-3 * x3 * d3) / denom_g2

        # g3 = (2.66e-2 * x3 * x4 * (x2^3 - x1^3)) / (x2^2 - x1^2) - 900
        denom_g3 = safe_signed(d2)
        g3 = (2.66e-2 * x3 * x4 * d3) / denom_g3 - 900.0

        Gs = torch.stack([g0, g1, g2, g3], dim=1)  # (N, 4)
        # violation = -g if g < 0 else 0
        violations = torch.where(Gs < 0, -Gs, torch.zeros_like(Gs))
        f2 = violations.sum(dim=1)

        out["F"] = torch.stack([f0, f1, f2], dim=1)  # (N, 3)



class RE34(PymooProblemTorch):
    def __init__(self, path = None, ref_point=None, **kwargs):
        super().__init__(n_var=5, n_obj=3, 
                         xl=torch.full((5,), 1.0) + 1e-6,
                         xu=torch.full((5,), 3.0) - 1e-6,
                         vtype=float, **kwargs)
        if ref_point is None:
            self.ref_point = [1702.52, 11.68, 0.26]
            # self.ref_point = [1.86472022e+03, 1.18199394e+01, 2.90399938e-01] # suitable for "online" mode
        else:
            self.ref_point = ref_point
            
        self.path = path

    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        try:
            front = np.loadtxt(self.path)
        except:
            front = np.loadtxt(resources.files("moospread.tasks.pf_re_tasks").joinpath(f"reference_points_{self.__class__.__name__}.dat"))
        else:
            assert self.path is not None, "Path to Pareto front file not specified."
        return torch.from_numpy(front).to(self.device)

    def _evaluate(self, x: torch.Tensor, out: dict, *args, **kwargs) -> None:
        # ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)                      # [5] → [1,5]

        # unpack variables
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]

        # prepare output
        f = torch.zeros((x.shape[0], self.n_obj),
                        dtype=x.dtype, device=x.device)

        # objective 1
        f[:, 0] = (
              1640.2823
            + 2.3573285 * x1
            + 2.3220035 * x2
            + 4.5688768 * x3
            + 7.7213633 * x4
            + 4.4559504 * x5
        )

        # objective 2
        f[:, 1] = (
              6.5856
            + 1.15    * x1
            - 1.0427  * x2
            + 0.9738  * x3
            + 0.8364  * x4
            - 0.3695  * x1 * x4
            + 0.0861  * x1 * x5
            + 0.3628  * x2 * x4
            - 0.1106  * x1 * x1
            - 0.3437  * x3 * x3
            + 0.1764  * x4 * x4
        )

        # objective 3
        f[:, 2] = (
             -0.0551
            + 0.0181 * x1
            + 0.1024 * x2
            + 0.0421 * x3
            - 0.0073 * x1 * x2
            + 0.0240 * x2 * x3
            - 0.0118 * x2 * x4
            - 0.0204 * x3 * x4
            - 0.0080 * x3 * x5
            - 0.0241 * x2 * x2
            + 0.0109 * x4 * x4
        )

        # if input was a single vector, return shape (3,)
        out["F"] = f

       
class RE37(PymooProblemTorch):
    "Rocket Injector Design (RE37)."

    def __init__(self, path=None, ref_point=None, **kwargs):
        super().__init__(n_var=4, n_obj=3, 
                         xl=torch.zeros(4) + 1e-6,
                         xu=torch.ones(4) - 1e-6,
                         vtype=float, **kwargs)
        if ref_point is None:
            self.ref_point = [0.99, 0.96, 0.99]
            # self.ref_point = [1.1022, 1.20726899, 1.20318656] # suitable for "online" mode
        else:
            self.ref_point = ref_point
            
        self.path = path

    def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
        try:
            front = np.loadtxt(self.path)
        except:
            front = np.loadtxt(resources.files("moospread.tasks.pf_re_tasks").joinpath(f"reference_points_{self.__class__.__name__}.dat"))
        else:
            assert self.path is not None, "Path to Pareto front file not specified."
        return torch.from_numpy(front).to(self.device)

    def _evaluate(self, x: torch.Tensor, out: dict, *args, **kwargs) -> None:
        # Unpack features: each is (B,)
        xAlpha, xHA, xOA, xOPTT = x.unbind(dim=1)

        # Compute objectives vectorized over batch
        f1 = (
            0.692
            + 0.477 * xAlpha
            - 0.687 * xHA
            - 0.080 * xOA
            - 0.0650 * xOPTT
            - 0.167  * xAlpha**2
            - 0.0129 * xHA * xAlpha
            + 0.0796 * xHA**2
            - 0.0634 * xOA * xAlpha
            - 0.0257 * xOA * xHA
            + 0.0877 * xOA**2
            - 0.0521 * xOPTT * xAlpha
            + 0.00156 * xOPTT * xHA
            + 0.00198 * xOPTT * xOA
            + 0.0184  * xOPTT**2
        )

        f2 = (
            0.153
            - 0.322  * xAlpha
            + 0.396  * xHA
            + 0.424  * xOA
            + 0.0226 * xOPTT
            + 0.175  * xAlpha**2
            + 0.0185 * xHA * xAlpha
            - 0.0701 * xHA**2
            - 0.251  * xOA * xAlpha
            + 0.179  * xOA * xHA
            + 0.0150 * xOA**2
            + 0.0134 * xOPTT * xAlpha
            + 0.0296 * xOPTT * xHA
            + 0.0752 * xOPTT * xOA
            + 0.0192 * xOPTT**2
        )

        f3 = (
            0.370
            - 0.205  * xAlpha
            + 0.0307 * xHA
            + 0.108  * xOA
            + 1.019  * xOPTT
            - 0.135  * xAlpha**2
            + 0.0141 * xHA * xAlpha
            + 0.0998 * xHA**2
            + 0.208  * xOA * xAlpha
            - 0.0301 * xOA * xHA
            - 0.226  * xOA**2
            + 0.353  * xOPTT * xAlpha
            - 0.0497 * xOPTT * xOA
            - 0.423  * xOPTT**2
            + 0.202  * xHA * xAlpha**2
            - 0.281  * xOA * xAlpha**2
            - 0.342  * xHA**2 * xAlpha
            - 0.245  * xHA**2 * xOA
            + 0.281  * xOA**2 * xHA
            - 0.184  * xOPTT**2 * xAlpha
            - 0.281  * xHA * xAlpha * xOA
        )

        # Stack into (B, 3)
        out["F"] = torch.stack([f1, f2, f3], dim=1)


class RE41(PymooProblemTorch):
    r"""Car side impact problem (RE41).

    See [Tanabe2020]_ for details.
    
    """
    def __init__(self, path=None, ref_point=None, **kwargs):
        super().__init__(n_var=7, n_obj=4, 
                         xl=torch.tensor([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], dtype=torch.float) + 1e-6,
                         xu=torch.tensor([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2], dtype=torch.float) - 1e-6,
                         vtype=float, **kwargs)
        if ref_point is None:
            self.ref_point = [42.65,  4.43, 13.08, 13.45]
            # self.ref_point = [47.04480682,  4.86997366, 14.40049127, 10.3941957 ] # suitable for "online" mode
        else:
            self.ref_point = ref_point
            
        self.path = path

        self.continuous_inds = list(range(7))

    def _evaluate(self, X: torch.Tensor, out: dict, *args, **kwargs) -> None:
        X1, X2, X3, X4, X5, X6, X7 = torch.split(X, 1, -1)
        f1 = (
            1.98
            + 4.9 * X1
            + 6.67 * X2
            + 6.98 * X3
            + 4.01 * X4
            + 1.78 * X5
            + 10**-5 * X6
            + 2.73 * X7
        )
        f2 = 4.72 - 0.5 * X4 - 0.19 * X2 * X3
        V_MBP = 10.58 - 0.674 * X1 * X2 - 0.67275 * X2
        V_FD = 16.45 - 0.489 * X3 * X7 - 0.843 * X5 * X6
        f3 = 0.5 * (V_MBP + V_FD)
        g1 = 1 - 1.16 + 0.3717 * X2 * X4 + 0.0092928 * X3
        g2 = (
            0.32
            - 0.261
            + 0.0159 * X1 * X2
            + 0.06486 * X1
            + 0.019 * X2 * X7
            - 0.0144 * X3 * X5
            - 0.0154464 * X6
        )
        g3 = (
            0.32
            - 0.214
            - 0.00817 * X5
            + 0.045195 * X1
            + 0.0135168 * X1
            - 0.03099 * X2 * X6
            + 0.018 * X2 * X7
            - 0.007176 * X3
            - 0.023232 * X3
            + 0.00364 * X5 * X6
            + 0.018 * X2.pow(2)
        )
        g4 = 0.32 - 0.74 + 0.61 * X2 + 0.031296 * X3 + 0.031872 * X7 - 0.227 * X2.pow(2)
        g5 = 32 - 28.98 - 3.818 * X3 + 4.2 * X1 * X2 - 1.27296 * X6 + 2.68065 * X7
        g6 = (
            32
            - 33.86
            - 2.95 * X3
            + 5.057 * X1 * X2
            + 3.795 * X2
            + 3.4431 * X7
            - 1.45728
        )
        g7 = 32 - 46.36 + 9.9 * X2 + 4.4505 * X1
        g8 = 4 - f2
        g9 = 9.9 - V_MBP
        g10 = 15.7 - V_FD
        g = torch.cat([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10], dim=-1)
        zero = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        g = torch.where(g < 0, -g, zero)
        f4 = g.sum(dim=-1, keepdim=True)
        out["F"] = torch.cat([f1, f2, f3, f4], dim=-1)
