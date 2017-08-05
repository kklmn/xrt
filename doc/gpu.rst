.. _calculations_on_GPU:

Calculations on GPU
===================

GPU can be used for several types of calculations, e.g. of an undulator source.
The gain is enormous. You should try it. Even on an "ordinary" Intel processor
the execution with OpenCL becomes significantly faster.

Here are some benchmarks on a system with Intel Core i7-3930K 3.20 GHz CPU,
ASUS Radeon R9 290 GPU, Python 2.7.6 64-bit, Windows 7 64-bit.
Script examples\\withRaycing\\01_SynchrotronSources\\synchrotronSources.py,
1 million rays, execution times in seconds:

+------------------+-----------------------------------+
| CPU 1 process    | 5172, 1 CPU process loaded        | 
+------------------+-----------------------------------+
| CPU 10 processes | 1245, with heavily loaded system  |
+------------------+-----------------------------------+
| openCL with CPU  | 163, with highly loaded system    |
+------------------+-----------------------------------+
| openCL with GPU  | 132, with almost idle CPU         |
+------------------+-----------------------------------+

You will need AMD/NVIDIA drivers (if you have a GPU, however this is not a must),
a CPU only OpenCL runtime, pytools and pyopencl.

.. note::

    When using OpenCL, no further parallelization is possible by means of
    multithreading or multiprocessing. You should turn them off by using the
    default values processes=1 and threads=1 in the run properties.

Please run the script ``tests\raycing\info_opencl.py`` for getting information
about your OpenCL platforms and devices. You will pass then the proper indices
in the lists of the platforms and devices as parameters to pyopencl methods or,
alternatively, pass 'auto' to ``targetOpenCL``.

.. important::

    Consider the :ref:`warnings and tips <usage_GPU_warnings>` on using xrt
    with GPUs.

.. hint::

    Consider also :ref:`Speed tests <tests>` for a few selected cases.
