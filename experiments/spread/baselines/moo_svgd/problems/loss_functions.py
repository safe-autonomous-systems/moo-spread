import torch
import numpy as np
import math

def loss_function(X, problem='zdt1', n_obj=None):
    
    if problem.startswith("zdt"):
        f = X[:, 0]
        g = X[:, 1:]

        if problem == 'zdt1':
            g = g.sum(dim=1, keepdim=False) * (9./29.) + 1.
            h = 1. - torch.sqrt(f/g)
        
        if problem == 'zdt2':
            g = g.sum(dim=1, keepdim=False) * (9./29.) + 1.
            h = 1. - (f/g)**2

        if problem == 'zdt3':
            g = g.sum(dim=1, keepdim=False) * (9./29.) + 1.
            h = 1. - torch.sqrt(f/g) - (f/g)*torch.sin(10.*np.pi*f)

        return f, g*h
    
    elif problem.startswith("dtlz"):
        if n_obj is None:
            n_obj = 3
        k = X.shape[-1] - n_obj + 1
        
        if problem == 'dtlz2':
            X_m = X[..., -k :]
            g_X = (X_m - 0.5).pow(2).sum(dim=-1)
            g_X_plus1 = 1 + g_X
            fs = []
            pi_over_2 = math.pi / 2
            for i in range(n_obj):
                idx = n_obj - 1 - i
                f_i = g_X_plus1.clone()
                f_i *= torch.cos(X[..., :idx] * pi_over_2).prod(dim=-1)
                if i > 0:
                    f_i *= torch.sin(X[..., idx] * pi_over_2)
                fs.append(f_i)
                
        elif problem == "dtlz4":
            X_, X_M = X[:, :n_obj - 1], X[:, n_obj - 1:]
            alpha  = 100
            g_X = (X_M - 0.5).pow(2).sum(dim=-1)
            g_X_plus1 = 1 + g_X
            fs = []
            pi_over_2 = math.pi / 2
            for i in range(n_obj):
                idx = n_obj - 1 - i
                f_i = g_X_plus1.clone()
                f_i *= torch.cos((X_[..., :idx]**alpha) * pi_over_2).prod(dim=-1)
                if i > 0:
                    f_i *= torch.sin((X_[..., idx]**alpha) * pi_over_2)
                fs.append(f_i)
                
        elif problem == 'dtlz7':
            fs = []
            for i in range(0, n_obj - 1):
                fs.append(X[..., i])
            f_temp = torch.stack(fs, dim=-1)

            g_X = 1 + 9 / k * torch.sum(X[..., -k :], dim=-1)
            h = n_obj - torch.sum(
                f_temp / (1 + g_X.unsqueeze(-1)) * (1 + torch.sin(3 * math.pi * f_temp)), dim=-1
            )
            fs.append((1 + g_X) * h)
            
        return fs
    
    elif problem.startswith("re"):
        fs = []
        
        if problem == "re21":
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

            fs = [f0, f1]
        
        if problem == "re33":
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

            fs = [f0, f1, f2]

                
        if problem == 're34':
            # ensure batch dimension
            if X.dim() == 1:
                X = X.unsqueeze(0)                      # [5] → [1,5]

            # unpack variables
            x1 = X[:, 0]
            x2 = X[:, 1]
            x3 = X[:, 2]
            x4 = X[:, 3]
            x5 = X[:, 4]

            # objective 1
            fs.append(
                1640.2823
                + 2.3573285 * x1
                + 2.3220035 * x2
                + 4.5688768 * x3
                + 7.7213633 * x4
                + 4.4559504 * x5
            )

            # objective 2
            fs.append(
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
            fs.append(
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
        
        if problem == 're37':
            # Unpack features: each is (B,)
            xAlpha, xHA, xOA, xOPTT = X.unbind(dim=1)

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

            fs = [f1,
                  f2, 
                  f3]

        if problem == 're41':
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
            fs = [f1.squeeze(1), 
                  f2.squeeze(1), 
                  f3.squeeze(1), 
                  f4.squeeze(1)]

        return fs
    
def get_bounds(problem, n_var=30, device='cpu'):
    problem = problem.lower()
    if problem.startswith("zdt") or problem.startswith("dtlz"):
        return (torch.tensor([0.0 + 1e-6] * n_var, device=device), 
                torch.tensor([1.0 - 1e-6] * n_var, device=device))
        
    elif problem.startswith("re21"):
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma  # = 1.
        xl = torch.tensor([tmp_val, 
                              (2.0**0.5)*tmp_val,  
                              (2.0**0.5)*tmp_val, 
                              tmp_val]) + 1e-6
        xu = torch.full((4,), 3.0*tmp_val) - 1e-6
        return (xl.to(device), xu.to(device))
    elif problem.startswith("re33"):
        return (torch.tensor([55.0, 75.0, 1000.0, 11.0], device=device)+ 1e-6,
                torch.tensor([80.0, 110.0, 3000.0, 20.0], device=device)- 1e-6)
    elif problem.startswith("re34"):
        return (torch.full((5,), 1.0, device=device) + 1e-6,
                torch.full((5,), 3.0, device=device) - 1e-6)
    elif problem.startswith("re37"):
        return (torch.tensor([0.0 + 1e-6] * 4, device=device), 
                torch.tensor([1.0 - 1e-6] * 4, device=device))
    elif problem.startswith("re41"):
        return (torch.tensor([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], device=device) + 1e-6,
                torch.tensor([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2], device=device) - 1e-6)
    else:
        raise ValueError(f"Unknown problem: {problem}")

dic_ref_point = {
    "zdt1": [0.9994, 6.0576],
    "zdt2": [0.9994, 6.8960],
    "zdt3": [0.9994, 6.0571],
    "dtlz2": [2.8390, 2.9011, 2.8575],
    "dtlz4": [3.2675, 2.6443, 2.4263],
    "dtlz7": [0.9984, 0.9961, 22.8114],
    "re21": [3144.44, 0.05],
    "re33": [5.01, 9.84, 4.30],
    "re34": [1.86472022e+03, 1.18199394e+01, 2.90399938e-01],
    "re37": [1.1022, 1.20726899, 1.20318656],
    "re41": [47.04480682,  4.86997366, 14.40049127, 10.3941957 ],
}

dic_ref_point_M10 = {
    "dtlz2": [2.8390, 2.9011, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575],
    "dtlz4": [3.2675, 2.6443, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263],
    "dtlz7": [0.9984, 0.9961, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114],
}

dic_ref_point_M14 = {
    "dtlz2": [2.8390, 2.9011, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575, 2.8575],
    "dtlz4": [3.2675, 2.6443, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263, 2.4263],
    "dtlz7": [0.9984, 0.9961, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114, 22.8114],
}

def get_ref_point(problem='zdt1', num_obj=None):
    problem = problem.lower()
    try:
        if num_obj ==10:
            ref_point = dic_ref_point_M10[problem]
        elif num_obj ==14:
            ref_point = dic_ref_point_M14[problem]
        else:
            ref_point = dic_ref_point[problem]
    except KeyError:
        raise ValueError(f"Unknown problem: {problem}. Available problems: {list(dic_ref_point.keys())}")
    
    return np.array(ref_point)

