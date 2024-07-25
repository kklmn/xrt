Using xrt 
=========

Scripting in python
-------------------

You need to prepare a script that gives instructions on how to get the wanted 
ray properties and prepare the graphs. The scripting is different for different 
backends (backend is a module or an external program that supplies ray distributions).
Currently, xrt supports two backends: :ref:`raycing <scriptingRaycing>` -- an 
internal backend -- and :ref:`shadow  <scriptingShadow>`.

Please consider the supplied examples.

Running a python script
-----------------------

You can run it from your shell as `python your-script.py` or from within
an IDE, e.g. `Spyder <https://github.com/spyder-ide/spyder>`_.

Interacting with the plots
--------------------------

:mod:`matplotlib` provides an interactive navigation toolbar that allows zoom,
pan and save functions.

Saving the results
------------------

You can save a plot using its navigation toolbar or by specifying the parameter
`saveName` of the plot in your Python script.

You can also save any of the plot attributes through a script. For this, use
the parameter ``afterScript`` of ``xrt.runner.run_ray_tracing`` to point to a
function that will be executed after all the repeats have been completed. A
typical usage of such a function is to pickle 1D and 2D histograms, fwhm values
etc. For example:

- ``plot.xaxis.total1D`` is the 1D histogram of ``xaxis`` of ``plot``. Use
  an appropriate axis, there are three available: ``xaxis``, ``yaxis`` and
  ``caxis``.
- ``plot.xaxis.binCenters`` and ``axis.binEdges`` are the edges and centers of
  ``xaxis`` bins
- ``plot.total2D`` is the 2D histogram
- ``plot.total4D`` is the 4D mutual intensity, see *fluxKind* for the
  appropriate options
- ``plot.dx``, ``plot.dy`` and ``plot.dE`` store fwhm's of the three 1D
  histograms
- flux or power are accessed as ``plot.flux`` and ``plot.power``

If you want to save plot results in a scan, you can do this in a generator, as
explained :ref:`here <scriptingRaycing>`.
