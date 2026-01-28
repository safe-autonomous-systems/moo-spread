.. _online-setting:

Online Setting
==============

In the online setting, we assume unrestricted access to the true objective functions.
This allows us to generate as many training samples as needed before the diffusion training phase.
Here, we consider the ZDT2 benchmark problem as an example.

.. code-block:: python

   import numpy as np
   import torch

   # Import the SPREAD solver
   from moospread import SPREAD

   # Import a test problem
   from moospread.tasks import ZDT2

   # Define the problem
   problem = ZDT2(n_var=30)


When initializing the SPREAD solver, the parameter ``mode`` must be set to ``"online"`` and the size of the training dataset is specified using the ``data_size`` argument.
If no value is provided, a default of ``10000`` is used.
Setting ``timesteps`` to values greater than or equal to ``1000`` typically leads to better performance.

.. code-block:: python

   # Initialize the SPREAD solver
   solver = SPREAD(
       problem,
       data_size=10000,
       num_blocks=2,
       timesteps=5000,
       num_epochs=1000,
       train_tol=100,
       mode="online",
       seed=2026,
       verbose=True
   )


To solve the problem, specify the number of solutions to generate using the ``num_points_sample`` argument.
If the solver is reused with the same initialization, retraining can be avoided by setting ``load_models=True``.

In the online setting, we recommend:

- setting ``rho_scale_gamma = 0.9`` for bi-objective problems, and
- setting ``rho_scale_gamma = 0.001`` for problems with more than two objectives.

If intermediate Pareto fronts are not needed, simply set ``iterative_plot=False``.

.. code-block:: python

   # Solve the problem
   results = solver.solve(
       num_points_sample=200,
       rho_scale_gamma=0.9,
       iterative_plot=True,
       plot_period=10,
       max_backtracks=25,
       save_results=True,
       samples_store_path="./samples_dir/",
       images_store_path="./images_dir/"
   )
