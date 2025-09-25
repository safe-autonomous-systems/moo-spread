import torch
import numpy as np

from re_eng import Function

def get(name, *args, **kwargs):
    name = name.lower()

    if "re" not in name:
        try:
            # Set up problem
            tf = Function(name, n_var = None, n_obj=None)
            bounds = tf.get_bounds()
            ref_point = -1*tf.get_ref_point()  # negative reference point for minimization
        except Exception as e:
            raise Exception(f"Error initializing problem {name}: {e}")
        
        return tf, bounds, ref_point

    PROBLEM = {
        "re41": RE41,
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)

class RE41:
    def __init__(self):
        self.problem_name = 'RE41'
        self.n_obj = 4
        self.n_var = 7
        self.n_constraints = 0
        self.n_original_constraints = 10

        self.lbound = torch.tensor([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]).float()
        self.ubound = torch.tensor([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]).float()

    def evaluate(self, x):
        # ensure x is at least 2D: (7,) -> (1,7)
        x = np.atleast_2d(x)
        N = x.shape[0]
        
        x = torch.from_numpy(x).to("cuda")
        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        # x = x * (self.ubound - self.lbound) + self.lbound
        x = torch.clamp(x, min=self.lbound, max=self.ubound)
        # Ensure x is not a constant batch (rare case)
        mn, mx = x.min(), x.max()
        if mn == mx:
            noise = torch.empty_like(x).uniform_(-1e6, 1e6)
            x = x + noise.to(x.device)

        # allocate outputs
        f = torch.zeros((N, self.n_obj), dtype=x.dtype, device=x.device)
        g = torch.zeros((N, self.n_original_constraints), dtype=x.dtype, device=x.device)

        # unpack columns
        x1, x2, x3, x4, x5, x6, x7 = x.unbind(dim=1)

        # objectives
        f[:, 0] = (
            1.98
            + 4.9  * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.78 * x5
            + 1e-5 * x6
            + 2.73 * x7
        )
        f[:, 1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3

        Vmbp = 10.58 - 0.674   * x1 * x2 - 0.67275 * x2
        Vfd  = 16.45 - 0.489   * x3 * x7 - 0.843   * x5 * x6
        f[:, 2] = 0.5 * (Vmbp + Vfd)

        # constraints (violation >= 0)
        g[:, 0] =  1    - (1.16 - 0.3717  * x2 * x4 - 0.0092928 * x3)
        g[:, 1] =  0.32 - (0.261 - 0.0159  * x1 * x2 - 0.06486  * x1
                        - 0.019  * x2 * x7 + 0.0144   * x3 * x5 + 0.0154464 * x6)
        g[:, 2] =  0.32 - (0.214 + 0.00817 * x5
                        - 0.045195 * x1 - 0.0135168 * x1
                        + 0.03099  * x2 * x6 - 0.018    * x2 * x7
                        + 0.007176 * x3 + 0.023232   * x3
                        - 0.00364  * x5 * x6 - 0.018    * x2**2)
        g[:, 3] =  0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3
                        - 0.031872 * x7 + 0.227    * x2**2)
        g[:, 4] = 32    - (28.98 + 3.818  * x3 - 4.2     * x1 * x2
                        + 1.27296 * x6 - 2.68065  * x7)
        g[:, 5] = 32    - (33.86 + 2.95   * x3 - 5.057   * x1 * x2
                        - 3.795  * x2 - 3.4431   * x7 + 1.45728)
        g[:, 6] = 32    - (46.36 - 9.9    * x2 - 4.4505  * x1)
        g[:, 7] = 4     - f[:, 1]
        g[:, 8] = 9.9   - Vmbp
        g[:, 9] = 15.7  - Vfd

        # only positive violations
        g = torch.where(g < 0, -g, torch.zeros_like(g))

        # aggregate violation as 4th objective
        f[:, 3] = g.sum(dim=1)

        # if single sample, return 1D
        if N == 1:
            return f.view(-1).detach().cpu().numpy()
        return f.detach().cpu().numpy()
