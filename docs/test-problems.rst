.. _test-problems:

Test Problems
=============

The test problems provided by ``moospread`` can be accessed via ``moospread.tasks``.
We include several commonly used multi-objective optimization benchmark problems.

The provided reference points (used for hypervolume computation) are suitable for specific settings.
For other settings, a reference point can be specified when defining the problem
(e.g., ``RE21(ref_point=[..., ...])``).
Please refer to our paper for recommended reference points for each optimization setting.

.. list-table::
   :header-rows: 1

   * - Problem
     - Variables
     - Objectives
     - Pareto Front Shape
     - Recommended Setting
   * - ZDT1
     - :math:`d \ge 2`
     - 2
     - Convex
     - :ref:`online-setting`
   * - ZDT2
     - :math:`d \ge 2`
     - 2
     - Concave
     - :ref:`online-setting`
   * - ZDT3
     - :math:`d \ge 2`
     - 2
     - Disconnected
     - :ref:`online-setting`
   * - DTLZ2
     - :math:`d \ge 2`
     - :math:`m \ge 3`
     - Concave
     - :ref:`online-setting`
   * - DTLZ4
     - :math:`d \ge 2`
     - :math:`m \ge 3`
     - Concave
     - :ref:`online-setting`
   * - DTLZ7
     - :math:`d \ge 2`
     - :math:`m \ge 3`
     - Disconnected
     - :ref:`online-setting`
   * - RE21 (Four-bar truss design)
     - 4
     - 2
     - Convex
     - :ref:`offline-setting`
   * - RE33 (Disc brake design)
     - 4
     - 3
     - –
     - :ref:`offline-setting`
   * - RE34 (Vehicle crashworthiness design)
     - 5
     - 3
     - –
     - :ref:`offline-setting`
   * - RE37 (Rocket injector design)
     - 4
     - 3
     - –
     - :ref:`offline-setting`
   * - RE41 (Car side impact design)
     - 7
     - 4
     - –
     - :ref:`offline-setting`
   * - BraninCurrin
     - 2
     - 2
     - –
     - :ref:`mobo-setting`
   * - Penicillin
     - 7
     - 3
     - –
     - :ref:`mobo-setting`
   * - VehicleSafety
     - 5
     - 3
     - –
     - :ref:`mobo-setting`