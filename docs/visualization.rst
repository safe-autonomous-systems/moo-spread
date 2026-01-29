.. _visualization:

Visualization
=============

Iterative Plots
---------------

When ``iterative_plot`` is enabled during problem solving, ``moospread`` stores images of successive Pareto fronts in the directory specified by ``images_store_path``.
Several arguments of ``solver.solve(...)`` can be used to control the visualization process:

- ``plot_period``: frequency of plotting,
- ``plot_dataset``: whether to display the training dataset alongside the generated solutions,
- ``plot_population``: whether to plot the full population of samples.

You may also provide a custom plotting function when initializing the SPREAD solver via the ``plot_func`` argument.
The custom plotting function must accept the same arguments as ``solver.plot_pareto_front``.


Video
-----

A main motivation for enabling ``iterative_plot`` is to visualize the optimization process as a video after completion.
This can be achieved using:

.. code-block:: python

   solver.create_video(
       image_folder,
       output_video,
       total_duration_s,
       first_transition_s,
       fps
   )

This function creates a video from the images stored in ``image_folder``, which are sorted according to the time index encoded in their filenames (``t=...``).

The first transition (from the first to the second image) lasts ``first_transition_s`` seconds, while the remaining transitions share the remaining time equally.
The resulting video has a total duration of ``total_duration_s`` seconds and is rendered at ``fps`` frames per second.
