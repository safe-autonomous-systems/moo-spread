.. _test-problems:

Test Problems
==============

Accessible via ``moospread.tasks``, we provide some commonly used multi-objective optimization test problems.
The provided reference points (for hypervolume calculation) are suitable for a specific setting. 
For the other settings, it can be provided at the problem definition (e.g. ``RE21(ref_point = [..., ...])``).
Please take a look at our paper for reference points suggestions per setting.

.. list-table::
   :header-rows: 1

   * - Problem
     - Variables
     - Objectives
     - Pareto Front Shape 
     - Reference point
   * - ZDT1
     - :math:`d \ge 2`
     - 2
     - Convexe
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
     - Convexe
     - :ref:`offline-setting`
   * - RE33 (Disc brake design)
     - 4
     - 3
     - :math:`-`
     - :ref:`offline-setting`
   * - RE34 (Vehicle crashworthiness design)
     - 5
     - 3
     - :math:`-`
     - :ref:`offline-setting`
   * - RE37 (Rocket injector design)
     - 4
     - 3
     - :math:`-`
     - :ref:`online-setting`
   * - RE41 (Car side impact design)
     - 7
     - 4
     - :math:`-`
     - :ref:`offline-setting`
   * - BraninCurrin
     - 2
     - 2
     - :math:`-`
     - :ref:`mobo-setting`
   * - Penicillin
     - 7
     - 3
     - :math:`-`
     - :ref:`mobo-setting`
   * - VehicleSafety
     - 5
     - 3
     - :math:`-`
     - :ref:`mobo-setting`



