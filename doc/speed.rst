.. _tests:

Speed tests
===========

The scripts used for these tests can be found in ``\tests\speed\``

The following computers have been used:

.. [1] A workstation with Intel Core i7-7700K @4.20 GHz×4(8), 16 GB
       and with AMD FirePro W9100 GPU.
       Windows 10 with Python 2.7.10 64 bit and 3.6.1 64 bit,
       Ubuntu 16.04 LTS with Python 2.7.12 64 bit and 3.5.2 64 bit.

.. [2] An ASUS UX430UQ laptop with Intel Core i7-7500U CPU @ 2.70 GHz×2(4),
       16 GB and with NVIDIA GeForce 940MX GPU.
       Windows 10 with Python 2.7.10 64 bit and 3.6.1 64 bit,
       Ubuntu 16.10 with Python 2.7.12 64 bit and 3.5.2 64 bit.

.. [3] A DELL GPU node with 2 Nvidia Tesla K80 GPUs (double-chip each),
   CentOS 7 -- run by Zdeněk Matěj (MAX IV)

.. [4] A DELL GPU node with 4 Nvidia Tesla P100 GPUs, CentOS 7 -- run by Zdeněk Matěj (MAX IV)

.. [5] A CPU node with Intel Xeon E5-2650 v3 @ 2.30GHz×20(40) -- run by Zdeněk Matěj (MAX IV)

.. [6] A CPU node with Intel Xeon E5-2650 v4 @ 2.20GHz×24(48) -- run by Zdeněk Matěj (MAX IV)

.. [7] A CPU node with Intel Xeon Gold 6130 @ 2.10GHz×32(64) -- run by Zdeněk Matěj (MAX IV)

.. note::

    The tests here were reduced in the number of rays/samples as compared to
    real calculations to let them run reasonably quickly. Longer calculations
    would demonstrate yet bigger difference between the slowest and the fastest
    cases, as the overheads (job distribution, collecting of histograms and
    plotting) would become relatively less important.

The tables below show execution times in seconds. Some cells have two values:
for Python 2 and for Python 3.

.. automodule:: tests.speed.1_SourceZCrystalThetaAlpha_speed

.. automodule:: tests.speed.2_synchrotronSources_speed

.. automodule:: tests.speed.3_Softi_CXIw2D_speed

Summary
-------

- Except for the case of computing with OpenCL on GPU, calculations in Linux
  are usually significantly faster than in Windows. *Especially when using
  multithreading or multiprocessing, the execution in Linux is dramatically
  faster.*

- There is no significant difference in speed between Python 2 and Python 3,
  except for multiprocessing in Windows, where Python 2 performs better.

- For geometric ray tracing a decent laptop can be a reasonable choice.