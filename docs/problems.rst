.. _problems:

Problems
========

The ``moospread`` package provides a PyTorch class ``PymooProblemTorch`` inspired by
`pymoo <https://pymoo.org/>`_, which can be used to define custom problems for the SPREAD solver.
As an example, consider the ``ZDT2`` problem, a bi-objective benchmark (typically defined with
:math:`n=30` decision variables) with a non-convex Pareto-optimal front.

**Decision variables**

.. math::

   x = (x_1, \ldots, x_n) \in [0,1]^n.

**Objectives (to be minimized)**

.. math::

   \begin{aligned}
   f_1(x) &= x_1,\\
   g(x)   &= 1 + \frac{9}{n-1}\sum_{i=2}^{n} x_i,\\
   f_2(x) &= g(x)\left(1 - \left(\frac{f_1(x)}{g(x)}\right)^2\right).
   \end{aligned}

Equivalently, using the helper function :math:`h(f_1,g) = 1 - (f_1/g)^2`, one can write
:math:`f_2(x) = g(x)\,h(f_1(x), g(x))`.

**Pareto-optimal set and Pareto front**

The Pareto-optimal set is given by:

.. math::

   0 \le x_1^\star \le 1
   \quad\text{and}\quad
   x_i^\star = 0 \;\; \text{for } i=2,\ldots,n.

The corresponding Pareto front (in objective space) is:

.. math::

   f_2 = 1 - f_1^2,
   \qquad 0 \le f_1 \le 1.

**Implementation**

This problem can be implemented as follows:

.. code-block:: python

   import torch

   class ZDT2(PymooProblemTorch):
       def __init__(
           self,
           n_var: int = 30,          # number of decision variables
           ref_point=None,           # reference point for hypervolume calculation
           **kwargs
       ):
           super().__init__(
               n_var=n_var,
               n_obj=2,  # number of objectives
               xl=torch.zeros(n_var, dtype=torch.float) + 1e-6,  # lower bounds
               xu=torch.ones(n_var, dtype=torch.float) - 1e-6,   # upper bounds
               vtype=float,
               **kwargs
           )
           self.ref_point = ref_point

       # Provide the true Pareto front (if known; otherwise return None).
       # If this returns None, hypervolume will not be computed.
       def _calc_pareto_front(self, n_pareto_points: int = 100) -> torch.Tensor:
           x = torch.linspace(0.0, 1.0, n_pareto_points, device=self.device)
           return torch.stack([x, 1.0 - x.pow(2)], dim=1)

       # Implement the objective function evaluation.
       # Use PyTorch operations to preserve differentiability.
       def _evaluate(self, x: torch.Tensor, out: dict, *args, **kwargs) -> None:
           f1 = x[:, 0]
           c = torch.sum(x[:, 1:], dim=1)
           g = 1.0 + 9.0 * c / (self.n_var - 1)
           term = torch.clamp(f1 / g, min=0.0)
           f2 = g * (1.0 - term.pow(2))
           out["F"] = torch.stack([f1, f2], dim=1)


In the :ref:`offline-setting`, the ``_evaluate`` method is not required, since the optimization process relies exclusively on a provided dataset and surrogate proxies rather than on evaluations of the true objective functions.
In that case, features that depend on true evaluations (e.g., hypervolume computation or certain plotting utilities) may be unavailable.
``moospread`` also provides several :ref:`test-problems` that can be accessed via ``moospread.tasks``.
