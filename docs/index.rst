moospread
=========

The ``moospread`` package implements in PyTorch the SPREAD method proposed in our paper
`SPREAD: Sampling-based Pareto Front Refinement via Efficient Adaptive Diffusion <https://openreview.net/forum?id=4731mIqv89>`_.

SPREAD is a sampling-based approach for multi-objective optimization that leverages diffusion models to refine and generate well-spread Pareto front approximations efficiently. 
It combines the expressive power of diffusion models with multi-objective optimization principles to achieve both strong convergence to the Pareto front and high diversity across the objective space. 

.. list-table::
   :widths: 33 33 33
   :align: center

   * - .. figure:: _static/ZDT2_n200_online.gif
          :width: 300px

     - .. figure:: _static/RE21_n200_offline.gif
          :width: 300px

     - .. figure:: _static/BraninCurrin_n5_bayesian_hv.gif
          :width: 300px

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   problems
   test-problems
   online-setting
   offline-setting
   mobo-setting
   visualization
    
   
