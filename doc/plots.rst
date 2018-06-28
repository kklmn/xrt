.. _plots:

Customizing your plots
======================

.. automodule:: xrt.plotter

The plots can be customized at the time of their creation using many optional
parameters of :meth:`xrt.plotter.XYCPlot.__init__` constructor.

.. autoclass:: xrt.plotter.XYCPlot
   :members: __init__

Each of the 3 axes: xaxis, yaxis and caxis can be customized using many optional
parameters of :meth:`xrt.plotter.XYCAxis.__init__` constructor.

.. autoclass:: xrt.plotter.XYCAxis
   :members: __init__

Customizing the look of the graphs
----------------------------------

The elements of the graphs (fonts, line thickness etc.) can be modified `by
editing <http://matplotlib.sourceforge.net/users/customizing.html>`_
the ``matplotlibrc`` file.

Alternatively, you can do this "dynamically", e.g.::

    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.linewidth'] = 0.5

The sizes and positions of the graphs are controlled by several constants
specified in :mod:`xrt`. You can either modify them directly in the
module or in your script, like it was done for creating the :ref:`logo 
image<logo>` (see also ``logo_xrt.py`` script in the examples)::

    import xrt.plotter as xrtp
    xrtp.xOrigin2d = 4
    xrtp.yOrigin2d = 4
    xrtp.height1d = 64
    xrtp.heightE1d = 64
    xrtp.xspace1dtoE1d = 4

Saving the histogram arrays
---------------------------

The histograms can be saved to and later restored from a :ref:`persistent file
<persistentName>`. Doing so is always recommended as this can save you time
afterwards if you decide to change a label or a font size or to make the graphs 
negative: you can do this without re-doing ray tracing. Remember, however, that 
a new run of the same script will initialize the histograms not from zero but 
from the last saved state. You will notice this by the displayed number of rays. 
If you want to initialize from zero, just delete the persistent file. Note that 
you can run :func:`~xrt.runner.run_ray_tracing` with *repeats=0* in order to
re-display the saved plots without starting ray tracing.