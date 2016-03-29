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
| openCL with GPU  | 132, with amost idle CPU          |
+------------------+-----------------------------------+

You will need AMD/NVIDIA drivers (if you have a GPU, however this is not a must),
a CPU only OpenCL runtime, pytools and pyopencl.

.. note::

    So far, multiprocessing (the python module, not the parallel calculations!)
    in xrt does not work together with OpenCL. You have to turn it off
    (processes=1). Even with a single process, calculations on a good GPU can be
    much faster than with multiple processes on a multicore CPU. Multithreading
    (e.g., threads=4) can still be used with OpenCL, although with little gain.

Please run the script ``tests\raycing\info_opencl.py`` for getting information
about your OpenCL platforms and devices. You will pass then the proper indices
in the lists of the platforms and devices as parameters to pyopencl methods or,
alternatively, pass 'auto' to ``targetOpenCL``.

.. note::

    Consider the :ref:`warnings and tips <usage_GPU_warnings>` on using xrt
    with GPUs.
