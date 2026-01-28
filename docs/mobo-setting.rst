.. _mobo-setting:

MOBO Setting
============

In the multi-objective Bayesian optimization (MOBO) setting, the true objective functions are available but expensive to evaluate.
Because the evaluation budget is typically small, collecting a large diffusion training dataset directly is infeasible.
To address this limitation, ``moospread`` adopts the data-augmentation strategy of
`CDM-PSL <https://ojs.aaai.org/index.php/AAAI/article/view/34913>`_.
This approach produces a sufficiently large training dataset for the diffusion model while respecting the evaluation-budget constraints.

In this example, we use the Branin–Currin problem, which is a bi-objective benchmark task.

.. code-block:: python

   import numpy as np
   import torch

   # Import the SPREAD solver
   from moospread import SPREAD

   # Import a test problem
   from moospread.tasks import BraninCurrin

   # Define the problem
   problem = BraninCurrin()


When initializing the SPREAD solver, the parameter ``mode`` must be set to ``"bayesian"``.
Setting ``timesteps`` to a relatively small value (e.g., ``25``) is important to reduce computational cost and accelerate optimization.

You may use a custom surrogate model architecture by specifying ``surrogate_model`` and its corresponding training function via ``train_func_surrogate``.
By default, ``moospread`` uses Gaussian processes as surrogate models.

.. code-block:: python

   # Initialize the SPREAD solver
   solver = SPREAD(
       problem,
       num_blocks=2,
       timesteps=25,
       num_epochs=1000,
       train_tol=100,
       mode="bayesian",
       model_dir="./model_dir",
       seed=2026,
       verbose=True
   )


To solve the problem, specify:

- the number of initial samples using ``n_init_mobo``,
- the number of MOBO iterations using ``n_steps_mobo``,
- the batch size selected at each step using ``batch_select_mobo``, and
- the number of generations using ``spread_num_samp_mobo``.

As shown in the experiments reported in the paper, using an auxiliary operator to escape local minima is beneficial.
This can be enabled by setting ``use_escape_local_mobo=True``.

In the MOBO setting, we recommend the following parameter choices:

- For bi-objective problems:

  - ``rho_scale_gamma = 0.9``
  - ``eta_init = 0.9``
  - ``lr_inner = 0.9``

- For problems with more than two objectives:

  - ``rho_scale_gamma = 0.01 / 0.0001``
  - ``eta_init = 0.9``
  - ``lr_inner = 0.9 / 5e-4``

If intermediate Pareto fronts are not required, set ``iterative_plot=False``.

.. code-block:: python

   # Solve the problem
   res_x, res_y = solver.solve(
       rho_scale_gamma=0.9,
       eta_init=0.9,
       lr_inner=0.9,
       iterative_plot=True,
       plot_period=10,
       max_backtracks=25,
       save_results=True,
       samples_store_path="./samples_dir/",
       images_store_path="./images_dir/",
       n_init_mobo=100,
       use_escape_local_mobo=True,
       n_steps_mobo=20,
       spread_num_samp_mobo=25,
       batch_select_mobo=5
   )
