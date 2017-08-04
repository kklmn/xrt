.. _tests:

Speed tests
===========

The scripts used for these tests can be found in ``\tests\speed\``

The following computers have been used:

.. [1] An office desktop with Intel Core i7-3930K @3.20 GHz×6(12), 16 GB
       and with AMD FirePro W9100 GPU.
       Windows 7 with Python 2.7.10 64 bit and 3.6.1 64 bit,
       Ubuntu 16.04 with Python 2.7.12 64 bit and 3.5.2 64 bit.

.. [2] An ASUS UX430UQ laptop with Intel Core i7-7500U CPU @ 2.70 GHz×2(4),
       16 GB and with NVIDIA GeForce 940MX GPU.
       Windows 10 with Python 2.7.10 64 bit and 3.6.1 64 bit,
       Ubuntu 16.10 with Python 2.7.12 64 bit and 3.5.2 64 bit.

.. note::

    The tests here were reduced in the number of rays/samples as compared to
    real calculations to let them run reasonably quickly. Longer calculations
    would demonstrate yet bigger difference between the slowest and the fastest
    cases, as the overheads (job distribution, collecting of histograms and
    plotting) would become relatively less important.

Each cell in the tables below has two execution times in seconds: in Python 2
and in Python 3.

.. automodule:: tests.speed.1_SourceZCrystalThetaAlpha_speed

.. automodule:: tests.speed.2_synchrotronSources_speed

.. automodule:: tests.speed.3_Softi_CXIw2D_speed

Summary
-------

- Except for the case of computing with OpenCL on GPU, calculations in Linux
  are usually significantly faster than in Windows. Especially when using
  multithreading or multiprocessing, the execution in Linux is dramatically
  faster.

- There is no significant difference in speed between Python 2 and Python 3,
  except for multiprocessing in Windows, where Python 2 performs better.

- For geometric ray tracing a decent laptop can be a reasonable choice.