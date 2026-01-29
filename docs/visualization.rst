.. _visualization:

Visualization
=============

Iterative plots
---------------

If ``iterative_plot`` plot is enabled when solving a problem, ``moospread`` stores in the directory provided via ``images_store_path``, 
the images of the successive Pareto fronts. Some arguments of ``solver.solve(...)`` to control the plots are ``plot_period``, 
``plot_dataset`` (whether to plot the training dataset along), and ``plot_population`` (whether to plot the full population of samples).  
You can also provide your own plotting function at the SPREAD solver initialization, via the argument ``plot_func``. 
Your  plotting function should accept the same arguments as ``moospread.plot_pareto_front``.

Video
-----

One reason to enable ``iterative_plot`` is to visualize the optimization process at the end via a video. This can be done using:

.. code-block:: python

   SPREAD.create_video(image_folder, 
                      output_video,
                      total_duration_s,
                      first_transition_s,
                      fps)

This will create a video from images in ``image_folder``, sorted by t=... in filename. 
The first transition (first->second image) lasts ``first_transition_s`` seconds, while the remaining transitions share the remaining time equally.
The output video has a total duration of ``total_duration_s`` seconds at ``fps`` frames per second.
