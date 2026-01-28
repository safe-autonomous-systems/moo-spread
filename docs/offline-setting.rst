.. _offline-setting:

Offline Setting
===============

SPREAD is particularly well suited for offline optimization scenarios.
In this setting, the objective functions are not available during optimization.
Instead, each task provides a fixed dataset of the form ``(X, F(X))``, and the diffusion model is trained solely on this static dataset.

We provide several benchmark tasks in :ref:`test-problems`, whose datasets can be downloaded from
`Off-MOO-Bench <https://github.com/lamda-bbo/offline-moo?tab=readme-ov-file>`_.

In this example, we use the rocket injector design task (RE37), which is a three-objective problem.

.. code-block:: python

   import numpy as np
   import torch

   # Import the SPREAD solver
   from moospread import SPREAD

   # Import a test problem
   from moospread.tasks import RE37

   # Define the problem
   problem = RE37()


When initializing the SPREAD solver, the parameter ``mode`` must be set to ``"offline"``, and the
training dataset ``(X, y)`` must be provided via the ``dataset`` argument.
Setting ``timesteps`` to values around ``1000`` is typically sufficient to obtain good performance.

You may also use a custom surrogate model architecture by specifying ``surrogate_model`` and its corresponding training function via ``train_func_surrogate``.
By default, ``moospread`` uses the ``MultipleModels`` surrogate from
`Offline Multi-Objective Optimization <https://proceedings.mlr.press/v235/xue24b.html>`_,
trained on the provided static dataset as a proxy for the objective functions.

.. code-block:: python

   # Load data
   X = np.load("/path_to_data/re37-x-0.npy")
   y = np.load("/path_to_data/re37-y-0.npy")

   X = torch.from_numpy(X).float()
   y = torch.from_numpy(y).float()

   # Initialize the SPREAD solver
   solver = SPREAD(
       problem,
       dataset=(X, y),
       num_blocks=2,
       timesteps=1000,
       num_epochs=1000,
       train_tol=100,
       train_tol_surrogate=100,
       mode="offline",
       model_dir="./model_dir",
       proxies_store_path="./proxies_dir",
       seed=2026,
       verbose=True
   )


If no dataset is provided, the SPREAD solver must have access to the true objective functions in order to generate a training dataset using Latin Hypercube Sampling (LHS).
In this case, the ``data_size`` argument must be specified, allowing an online problem to be treated as an offline task.

To solve the problem, specify the number of solutions to generate using the ``num_points_sample`` argument.
If the solver is reused with the same initialization, retraining of both the diffusion model and the surrogate models can be avoided by setting ``load_models=True``.

In the offline setting, we recommend the following parameter choices:

- For bi-objective problems:
  
  - ``rho_scale_gamma = 0.9 / 0.1 / 0.0001``
  - ``eta_init = 0.9 / 0.5 / 0.02``
  - ``lr_inner = 0.9 / 0.002``

- For problems with more than two objectives:
  
  - ``rho_scale_gamma = 0.001 / 0.0001``

If intermediate Pareto fronts are not required, set ``iterative_plot=False``.

.. code-block:: python

   # Solve the problem
   results = solver.solve(
       num_points_sample=200,
       rho_scale_gamma=0.001,
       eta_init=0.1,
       lr_inner=0.9,
       iterative_plot=True,
       plot_period=10,
       max_backtracks=25,
       save_results=True,
       samples_store_path="./samples_dir/",
       images_store_path="./images_dir/"
   )
