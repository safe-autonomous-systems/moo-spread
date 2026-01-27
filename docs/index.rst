moospread
=========

The ``moospread`` package implements in PyTorch the SPREAD method proposed in our paper
`SPREAD: Sampling-based Pareto Front Refinement via Efficient Adaptive Diffusion <https://openreview.net/forum?id=4731mIqv89>`_.

SPREAD is a sampling-based approach for multi-objective optimization that leverages diffusion models to refine and generate well-spread Pareto front approximations efficiently.
It combines the expressive power of diffusion models with multi-objective optimization principles to achieve both strong convergence to the Pareto front and high diversity across the objective space.
The method demonstrates competitive performance compared to state-of-the-art approaches while providing a flexible framework for different optimization settings.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   online-setting
   offline-setting
   mobo-setting
   problems
   test-problems
