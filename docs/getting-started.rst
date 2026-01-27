.. _getting-started:

Getting Started
===============

Welcome to the **moospread** documentation.

This page provides a minimal example showing how to install the package and solve a standard multi-objective optimization problem using the SPREAD solver.

Installation
------------

Create and activate a new conda environment, then install the package from PyPI:

.. code-block:: bash

   conda create -n moospread python=3.11
   conda activate moospread
   pip install moospread


Alternatively, to install the latest development version from GitHub:

.. code-block:: bash

   conda create -n moospread python=3.11
   conda activate moospread
   git clone https://github.com/safe-autonomous-systems/moo-spread.git
   cd moo-spread
   pip install -e .


Basic Usage
-----------

This example shows how to solve an online multi-objective optimization benchmark problem (ZDT2) using the **SPREAD** solver.

.. code-block:: python

   import numpy as np
   import torch

   # Import the SPREAD solver
   from moospread import SPREAD

   # Import a test problem
   from moospread.tasks import ZDT2

   # Define the problem
   n_var = 30
   problem = ZDT2(n_var=n_var)

   # Initialize the SPREAD solver
   solver = SPREAD(
       problem,
       data_size=10000,
       timesteps=1000,
       num_epochs=1000,
       train_tol=100,
       mode="online",
       seed=2026,
       verbose=True
   )

   # Solve the problem
   results = solver.solve(
       num_points_sample=200,
       iterative_plot=True,
       plot_period=10,
       max_backtracks=25,
       save_results=True,
       samples_store_path="./samples_dir/",
       images_store_path="./images_dir/"
   )


This will train a diffusion-based multi-objective solver, approximate the Pareto front of the ZDT2 problem in the online setting, and store the generated samples and plots in the specified directories.

Next Steps
----------

To explore more advanced configurations, see:

- :ref:`online-setting`
- :ref:`offline-setting`
- :ref:`mobo-setting`

You can also define your own optimization problem following the guidelines in :ref:`problems`, if it is not listed in :ref:`test-problems`: .
