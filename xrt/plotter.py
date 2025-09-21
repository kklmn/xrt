# -*- coding: utf-8 -*-
u"""
Module :mod:`plotter` provides classes describing axes and plots, as well as
containers for the accumulated arrays (histograms) for subsequent
pickling/unpickling or for global flux normalization. The module defines
several constants for default plot positions and sizes. The user may want to
modify them in the module or externally as in the xrt_logo.py example.

.. note::

    Each plot has a 2D positional histogram, two 1D positional histograms and,
    typically, a 1D color histogram (e.g. energy).

    .. warning::
        The two 1D positional histograms are not calculated from the 2D one!

    In other words, the 1D histograms only respect their corresponding limits
    and not the other dimension’s limits. There can be situations when the 2D
    image is black because the screen is misplaced but one 1D histogram may
    still show a beam distribution if in that direction the screen is
    positioned correctly. This was the reason why the 1D histograms were
    designed not to be directly dependent on the 2D one – this feature
    facilitates the troubleshooting of misalignments. On the other hand, this
    behavior may lead to confusion if a part of the 2D distribution is outside
    of the visible 2D area. In such cases one or two 1D histograms may show a
    wider distribution than the one visible on the 2D image. For correcting
    this behavior, one can mask the beam by apertures or by selecting the
    physical or optical limits of an optical element.

.. tip::

    If you do not want to create plot windows (e.g. when they are too many or
    when you run xrt on a remote machine) but only want to save plots, you can
    use a non-interactive matplotlib backend such as Agg (for PNGs), PDF, SVG
    or PS::

        matplotlib.use('agg')

    Importantly, this must be done at the very top of your script, right after
    import matplotlib and before importing anything else.

"""
from __future__ import unicode_literals
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "27 Jun 2022"

import os
import copy
import pickle
import numpy as np
import scipy as sp
# from scipy.interpolate import UnivariateSpline
from scipy.interpolate import make_interp_spline, PPoly

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from . import runner
# from runner import runCardVals, runCardProcs
from .backends import raycing
try:
    from .gui.commons import qt
    hasQt = True
except ImportError:
    hasQt = False

from matplotlib.figure import Figure

try:  # for Python 3 compatibility:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    unicode = unicode
    basestring = basestring

# otherwise it does not work correctly on my Ubuntu9.10 and mpl 0.99.1.1:
mpl.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['axes.linewidth'] = 0.75
#mpl.rcParams['backend'] = 'Qt5agg'
#mpl.rcParams['backend'] = 'Agg'
#mpl.rcParams['xtick.major.pad'] = '5'
#mpl.rcParams['ytick.major.pad'] = '5'
import matplotlib.pyplot as plt

epsHist = 1e-100  # prevents problem with normalization of histograms
# [Sizes and positions of plots]
dpi = 100
xOrigin2d = 80  # all sizes are in pixels
yOrigin2d = 48
space2dto1d = 4
height1d = 84
xspace1dtoE1d = 112
yspace1dtoE1d = 76
heightE1dbar = 10
heightE1d = 84
xSpaceExtraWhenNoEHistogram = 42
xSpaceExtra = 22
ySpaceExtra = 28
# [Sizes and positions of texts]
xlabelpad = 4  # x-axis label to axis
ylabelpad = 4  # y-axis label to axis

xTextPos = 1.02  # 0 to 1 relative to the figure size
yTextPosNrays = 1.0
yTextPosNraysR = 1.32
yTextPosGoodrays = 0.8
yTextPosGoodraysR = 1.1
yTextPosI = 0.58
xTextPosDx = 0.5
yTextPosDx = 1.02
xTextPosDy = 1.05
yTextPosDy = 0.5
xTextPosStatus = 0.999
yTextPosStatus = 0.001
yTextPosNrays1 = 0.88
yTextPosNrays2 = 0.66
yTextPosNrays3 = 0.44
yTextPosNrays4 = 0.22
# [Bins]
defaultBins = 128
defaultPixelPerBin = 2
extraMargin = 4  # bins. Extra margins to histograms when limits are not given.
# [Axis labels]
axisLabelFontSize = 10
defaultXTitle = '$x$'
defaultXUnit = 'mm'
defaultYTitle = '$z$'
defaultYUnit = 'mm'
defaultCTitle = 'energy'
defaultCUnit = 'eV'
defaultFwhmFormatStrForXYAxes = '%.1f'
defaultFwhmFormatStrForCAxis = '%.2f'
# [Development]
colorFactor = 0.85  # 2./3 for red-to-blue
colorSaturation = 0.85
# # end of rc-file ##


def versiontuple(v):
    a = v.split(".")
    return tuple(map(int, [''.join(c for c in s if c.isdigit()) for s in a]))


if hasQt:
    class MyQtFigCanvas(qt.FigCanvas):
        windowClosed = qt.Signal(int)

        def __init__(self, figure, xrtplot):
            super(MyQtFigCanvas, self).__init__(figure)
            self.xrtplot = xrtplot


def serialize_plots(data):
    plotsDict = raycing.OrderedDict()
    for iplot, plot in enumerate(data):
        plotname = 'plot{:02d}'.format(iplot+1)
        kwargs = raycing.get_init_kwargs(plot)
        kwargs['_object'] = "xrt.plotter.XYCPlot"

        for ax in ['xaxis', 'yaxis', 'caxis']:
            axkwargs = raycing.get_init_kwargs(getattr(plot, ax))
            axkwargs['_object'] = "xrt.plotter.XYCAxis"
            kwargs[ax] = axkwargs
        plotsDict[plotname] = kwargs
    return plotsDict


def deserialize_plots(data):
    plotsList = []
    plotDefArgs = dict(raycing.get_params("xrt.plotter.XYCPlot"))
    axDefArgs = dict(raycing.get_params("xrt.plotter.XYCAxis"))

    if not isinstance(data['Project']['plots'], dict):
        print("Plots are not defined")
        return []

    for plotName, plotProps in data['Project']['plots'].items():
        plotKwargs = {}

        for pname, pval in plotProps.items():
            if pname in ['_object']:
                continue
            if pname in ['xaxis', 'yaxis', 'caxis']:
                axKwargs = {}
                for axname, axval in pval.items():
                    if axname == '_object':
                        continue
                    if pname == 'caxis' and axname == 'unit':
                        axDefArgs[axname] = 'eV'
                    if axname in axDefArgs and axval != str(axDefArgs[axname]):
                        axKwargs[axname] = raycing.parametrize(axval)
                plotKwargs[pname] = XYCAxis(**axKwargs)

            else:
                if pname in plotDefArgs and pval != str(plotDefArgs[pname]):
                    plotKwargs[pname] = raycing.parametrize(pval)
        try:
            newPlot = XYCPlot(**plotKwargs)
            plotsList.append(newPlot)
        except:
            print("Plot init failed")
    return plotsList


class XYCAxis(object):
    u"""
    Contains a generic record structure describing each of the 3 axes:
    X, Y and Color (typ. Energy)."""

    def __init__(
        self, label='', unit='mm', factor=None, data='auto', limits=None,
        offset=0, bins=defaultBins, ppb=defaultPixelPerBin,
        density='histogram', invertAxis=False, outline=0.5,
            fwhmFormatStr=defaultFwhmFormatStrForXYAxes):
        u"""
        *label*: str
            The label of the axis without unit. This label will appear in the
            axis caption and in the FWHM label.

        *unit*: str
            The unit of the axis which will follow the label in parentheses
            and appear in the FWHM value

        *factor*: float
            Useful in order to match your axis units with the units of the
            ray tracing backend. For instance, the shadow length unit is cm.
            If you want to display the positions as mm: *factor=10*;
            if you want to display energy as keV: *factor=1e-3*.
            Another usage of *factor* is to bring the coordinates of the ray
            tracing backend to the world coordinates. For instance, z-axis in
            shadow is directed off the OE surface. If the OE is faced upside
            down, z is directed downwards. In order to display it upside, set
            minus to *factor*.
            if not specified, *factor* will default to a value that depends
            on *unit*. See :meth:`def auto_assign_factor`.

        *data*: int for shadow, otherwise array-like or function object
            shadow:
                zero-based index of columns in the shadow binary files:

                ======  ====================================================
                 0      x
                 1      y
                 2      z
                 3      x'
                 4      y'
                 5      z'
                 6      Ex s polariz
                 7      Ey s polariz
                 8      Ez s polariz
                 9      lost ray flag
                10      photon energy
                11      ray index
                12      optical path
                13      phase (s polarization)
                14      phase (p polarization)
                15      x component of the electromagnetic vector (p polar)
                16      y component of the electromagnetic vector (p polar)
                17      z component of the electromagnetic vector (p polar)
                18      empty
                ======  ====================================================
            raycing:
                use the following functions (in the table below) or pass your
                own one. See :mod:`raycing` for more functions, e.g. for the
                polarization properties. Alternatively, you may pass an array
                of the length of the beam arrays.

                =======  ===================================================
                 x       raycing.get_x
                 y       raycing.get_y
                 z       raycing.get_z
                 x'      raycing.get_xprime
                 z'      raycing.get_zprime
                 energy  raycing.get_energy
                =======  ===================================================

            If *data* = 'auto' then *label* is searched for "x", "y", "z",
            "x'", "z'", "energy" and if one of them is found, *data* is
            assigned to the listed above index or function. In raycing backend
            the automatic assignment is additionally implemented for *label*
            containing 'degree (for degree of polarization)', 'circular' (for
            circular polarization rate), 'path', 'incid' or 'theta' (for
            incident angle), 'order' (for grating diffraction order), 's',
            'phi', 'r' or 's' (for parametric representation of OE).

        *limits*: 2-list of floats [min, max]
            Axis limits. If None, the *limits* are taken as ``np.min`` and
            ``np.max`` for the corresponding array acquired after the 1st ray
            tracing run. If *limits* == 'symmetric', the limits are forced to
            be symmetric about the origin. Can also be set outside of the
            constructor as, e.g.::

                plot1.xaxis.limits = [-15, 15]

        *offset*: float
            An offset value subtracted from the axis tick labels to be
            displayed separately. It is useful for the energy axis, where the
            band width is most frequently much smaller than the central value.
            Ignored for x and y axes.

            +-----------------+--------------------+
            | no offset       |  non-zero offset   |
            +=================+====================+
            | |image_offset0| | |image_offset5000| |
            +-----------------+--------------------+

            .. |image_offset0| imagezoom:: _images/offset0.png
               :scale: 50 %
            .. |image_offset5000| imagezoom:: _images/offset5000.png
               :scale: 50 %

        *bins*: int
            Number of bins in the corresponding 1D and 2D histograms.
            See also *ppb* parameter.

        *ppb*: int
            Screen-pixel-per-bin value. The graph arrangement was optimized
            for *bins* * *ppb* = 256. If your *bins* and *ppb* give a very
            different product, the graphs may look ugly (disproportional)
            with overlapping tick labels.

        *density*: 'histogram' or 'kde'
            The way the sample density is calculated: by histogram or by kde
            [KDE]_.

        *invertAxis*: bool
            Inverts the axis direction. Useful for energy axis in energy-
            dispersive images in order to match the colors of the energy
            histogram with the colors of the 2D histogram.

        *outline*: float within [0, 1]
            Specifies the minimum brightness of the outline drawn over the
            1D histogram. The maximum brightness equals 1 at the maximum of
            the 1D histogram.

            +--------------------+--------------------+--------------------+
            |         =0         |         =0.5       |         =1         |
            +====================+====================+====================+
            | |image_outline0.0| | |image_outline0.5| | |image_outline1.0| |
            +--------------------+--------------------+--------------------+

            .. |image_outline0.0| imagezoom:: _images/outline00.png
               :scale: 50 %
            .. |image_outline0.5| imagezoom:: _images/outline05.png
               :scale: 50 %
            .. |image_outline1.0| imagezoom:: _images/outline10.png
               :scale: 50 %

        *fwhmFormatStr*: str
            Python format string for the FWHM value, e.g. '%.2f'. if None, the
            FWHM value is not displayed.


        """
        self.label = label
        self.unit = unit
        if self.label:
            self.displayLabel = self.label
        else:
            self.displayLabel = ''
        if self.unit:
            self.displayLabel += ' (' + self.unit + ')'
        self.factor = factor
        self.data = data
        self.limits = limits
        self.offset = offset
        self.offsetDisplayUnit = self.unit
        self.offsetDisplayFactor = 1
        self.bins = bins
        self.ppb = ppb
        self.pixels = bins * ppb
        self.density = density
        self.extraMargin = extraMargin
        self.invertAxis = invertAxis
        if outline < 0:
            outline = 0
        if outline > 1:
            outline = 1
        self.outline = outline
        self.fwhmFormatStr = fwhmFormatStr
        self.max1D = 0
        self.max1D_RGB = 0
        self.globalMax1D = 0
        self.globalMax1D_RGB = 0
        self.useCategory = False

    def auto_assign_data(self, backend):
        """
        Automatically assign data arrays given the axis label."""
        if "energy" in self.label:
            if backend == 'shadow':
                self.data = 10
            elif backend == 'raycing':
                self.data = raycing.get_energy
        elif "x'" in self.label:
            if backend == 'shadow':
                self.data = 3
            elif backend == 'raycing':
                self.data = raycing.get_xprime
        elif "z'" in self.label:
            if backend == 'shadow':
                self.data = 5
            elif backend == 'raycing':
                self.data = raycing.get_zprime
        elif "x" in self.label:
            if backend == 'shadow':
                self.data = 0
            elif backend == 'raycing':
                self.data = raycing.get_x
        elif "y" in self.label:
            if backend == 'shadow':
                self.data = 1
            elif backend == 'raycing':
                self.data = raycing.get_y
        elif "z" in self.label:
            if backend == 'shadow':
                self.data = 2
            elif backend == 'raycing':
                self.data = raycing.get_z
        elif "degree" in self.label:
            self.data = raycing.get_polarization_degree
        elif "circular" in self.label:
            self.data = raycing.get_circular_polarization_rate
        elif "incid" in self.label or "theta" in self.label:
            self.data = raycing.get_incidence_angle
        elif "phi" in self.label:
            self.data = raycing.get_phi
        elif "order" in self.label:
            self.data = raycing.get_order
        elif "s" in self.label:
            self.data = raycing.get_s
        elif "path" in self.label:
            self.data = raycing.get_path
        elif "r" in self.label:
            self.data = raycing.get_r
        elif "a" in self.label:
            self.data = raycing.get_a
        elif "b" in self.label:
            self.data = raycing.get_b
        else:
            raise ValueError(
                'cannot auto-assign data for axis "{0}"!'.format(self.label))

    def auto_assign_factor(self, backend):
        """
        Automatically assign factor given the axis label."""
        factor = 1.
        if self.unit in ['keV', ]:
            factor = 1e-3
        elif self.unit in ['mrad', 'meV']:
            factor = 1.0e3
        elif self.unit in [r'$\mu$rad', u'µrad', u'urad']:
            factor = 1.0e6
        else:
            if backend == 'shadow':
                if self.unit in ['m', ]:
                    factor = 1e-2
                elif self.unit in ['mm', ]:
                    factor = 10.
                elif self.unit in [r'$\mu$m', u'µm', 'um']:
                    factor = 1.0e4
                elif self.unit in ['nm', ]:
                    factor = 1.0e7
            elif backend == 'raycing':
                if self.unit in ['m', ]:
                    factor = 1e-3
                elif self.unit in ['mm', ]:
                    factor = 1.
                elif self.unit in [r'$\mu$m', u'µm', 'um']:
                    factor = 1.0e3
                elif self.unit in ['nm', ]:
                    factor = 1.0e6
                elif self.unit in ['pm', 'nrad']:
                    factor = 1.0e9
                elif self.unit in ['fm', ]:
                    factor = 1.0e12
                elif self.unit.startswith('deg'):
                    factor = np.degrees(1)
                elif self.unit.startswith('mdeg'):
                    factor = np.degrees(1)*1e3
        self.factor = factor


class XYCPlot(object):
    u"""
    Container for the accumulated histograms. Besides giving the beam
    images, this class provides with useful fields like *dx*, *dy*, *dE*
    (FWHM), *cx*, *cy*, *cE* (centers) and *intensity* which can be used in
    scripts for producing scan-like results."""

    def __init__(
        self, beam=None, rayFlag=(1,), xaxis=None, yaxis=None, caxis=None,
        aspect='equal', xPos=1, yPos=1, ePos=1, title='',
        invertColorMap=False, negative=False,
        fluxKind='total', fluxUnit='auto',
        fluxFormatStr='auto', contourLevels=None, contourColors=None,
        contourFmt='%.1f', contourFactor=1., saveName=None,
        persistentName=None, oe=None, raycingParam=0,
            beamState=None, beamC=None, useQtWidget=False):
        u"""
        *beam*: str
            The beam to be visualized.

            In raycing backend:
                The key in the dictionary returned by
                :func:`~xrt.backends.raycing.run.run_process()`. The values of
                that dictionary are beams (instances of
                :class:`~xrt.backends.raycing.sources.Beam`).

            In shadow backend:
                The Shadow output file (``star.NN``, `mirr.NN`` or
                ``screen.NNMM``). It will also appear in the window caption
                unless *title* parameter overrides it.

            This parameter is used for the automatic determination of the
            backend in use with the corresponding meaning of the next two
            parameters. If *beam* contains a dot, shadow backend is assumed.
            Otherwise raycing backend is assumed.

        *rayFlag*: int or tuple of ints
            shadow: 0=lost rays, 1=good rays, 2=all rays.
            raycing: a tuple of integer ray states: 1=good, 2=out, 3=over,
            4=alive (good + out), -NN = dead at oe number NN (numbering starts
            with 1).

        *xaxis*, *yaxis*, *caxis*: instance of :class:`XYCAxis` or None.
            If None, a default axis is created. If caxis='category' and the
            backend is raycing, then the coloring is given by ray category, the
            color axis histogram is not displayed and *ePos* is ignored.

            .. warning::
                The axes contain arrays for the accumulation of histograms. If
                you create the axes outside of the plot constructor then make
                sure that these are not used for another plot. Otherwise the
                histograms will be overwritten!

        *aspect*: str or float
            Aspect ratio of the 2D histogram, = 'equal', 'auto' or numeric
            value (=x/y). *aspect* =1 is the same as *aspect* ='equal'.

        *xPos*, *yPos*: int
            If non-zero, the corresponding 1D histograms are visible.

        *ePos*: int
            Flag for specifying the positioning of the color axis histogram:

            +-------------------------+---------------------------------------+
            | *ePos* =1: at the right |             |image_ePos1|             |
            | (default, as usually    |                                       |
            | the diffraction plane   |                                       |
            | is vertical)            |                                       |
            +-------------------------+---------------------------------------+
            | *ePos* =2: at the top   |             |image_ePos2|             |
            | (for horizontal         |                                       |
            | diffraction plane)      |                                       |
            +-------------------------+---------------------------------------+
            | *ePos* =0: no           |             |image_ePos0|             |
            | color axis histogram    |                                       |
            +-------------------------+---------------------------------------+

            .. |image_ePos1| imagezoom:: _images/ePos=1.png
               :scale: 50 %
               :loc: upper-right-corner
            .. |image_ePos2| imagezoom:: _images/ePos=2.png
               :scale: 50 %
               :loc: upper-right-corner
            .. |image_ePos0| imagezoom:: _images/ePos=0.png
               :scale: 50 %
               :loc: upper-right-corner


        *title*: str
            If non-empty, this string will appear in the window caption,
            otherwise the *beam* will be used for this.

        *invertColorMap*: bool
            Inverts colors in the HSV color map; seen differently, this is a
            0.5 circular shift in the color map space. This inversion is
            useful in combination with *negative* in order to keep the same
            energy coloring both for black and for white images.

        *negative*: bool
            Useful for printing in order to save black inks.
            See also *invertColorMap*.

            * =False: black bknd for on-screen presentation
            * =True: white bknd for paper printing

            The following table demonstrates the combinations of
            *invertColorMap* and *negative*:

            +-------------+-------------------------+-------------------------+
            |             |    *invertColorMap*     |     *invertColorMap*    |
            |             |    =False               |     =True               |
            +=============+=========================+=========================+
            | *negative*  |        |image00|        |        |image10|        |
            | =False      |                         |                         |
            +-------------+-------------------------+-------------------------+
            | *negative*  |        |image01|        |        |image11|        |
            | =True       |                         |                         |
            +-------------+-------------------------+-------------------------+

            .. |image00| imagezoom:: _images/invertColorMap=0_negative=0.png
               :scale: 50 %
            .. |image01| imagezoom:: _images/invertColorMap=0_negative=1.png
               :scale: 50 %
            .. |image10| imagezoom:: _images/invertColorMap=1_negative=0.png
               :scale: 50 %
               :loc: upper-right-corner
            .. |image11| imagezoom:: _images/invertColorMap=1_negative=1.png
               :scale: 50 %
               :loc: upper-right-corner

            Note that *negative* inverts only the colors of the graphs, not
            the white global background. Use a common graphical editor to
            invert the whole picture after doing *negative=True*:

            .. imagezoom:: _images/negative=1+fullNegative.png
               :scale: 50 %
               :align: center

            (such a picture would nicely look on a black journal cover, e.g.
            on that of Journal of Synchrotron Radiation ;) )

        .. _fluxKind:

        *fluxKind*: str
            Can begin with 's', 'p', '+-45', 'left-right', 'total', 'power',
            'Es', 'Ep' and 'E'. Specifies what kind of flux to use for the
            brightness of 2D and for the height of 1D histograms. If it ends
            with 'log', the flux scale is logarithmic.

            If starts with 'E' then the *field amplitude* or mutual intensity
            is considered, not the usual intensity, and accumulated in the 2D
            histogram or in a 3D stack:

            - If ends with 'xx' or 'zz', the corresponding 2D cuts of mutual
              intensity are accumulated in the main 2D array (the one visible
              as a 2D histogram). The plot must have equal axes.

            - If ends with '4D', the complete mutual intensity is calculated
              and stored in *plot.total4D* with the shape
              (xaxis.bins*yaxis.bins, xaxis.bins*yaxis.bins).

            .. warning::

                Be cautious with the size of the mutual intensity object, it is
                four-dimensional!

            - If ends with 'PCA', the field images are stored in *plot.field3D*
              with the shape (repeats, xaxis.bins, yaxis.bins) for further
              Principal Component Analysis.

            - If without these endings, the field aplitudes are simply summed
              in the 2D histogram.

        *fluxUnit*: 'auto' or None
            If a synchrotron source is used and *fluxUnit* is 'auto', the
            flux will be displayed as 'ph/s' or 'W' (if *fluxKind* == 'power').
            Otherwise the flux is a unitless number of rays times
            transmittivity | reflectivity.

        *fluxFormatStr*: str
            Format string for representing the flux or power. You can use a
            representation with powers of ten by utilizing 'p' as format
            specifier, e.g. '%.2p'.

        *contourLevels*: sequence
            A sequence of levels on the 2D image for drawing the contours, in
            [0, 1] range. If None, the contours are not drawn.

        *contourColors*: sequence or color
            A sequence of colors corresponding to *contourLevels*. A single
            color value is applied to all the contours. If None, the colors are
            automatic.

        *contourFmt*: str
            Python format string for contour values.

        *contourFactor*: float
            Is applied to the levels and is useful in combination with
            *contourFmt*, e.g. *contourFmt* = r'%.1f mW/mm$^2$',
            *contourFactor* = 1e3.

        *saveName*: str or list of str or None
            Save file name(s). The file type(s) are given by extensions:
            png, ps, svg, pdf. Typically, *saveName* is set outside of the
            constructor. For example::

                filename = 'filt%04imum' %thick #without extension
                plot1.saveName = [filename + '.pdf', filename + '.png']

        .. _persistentName:

        *persistentName*: str or None
            File name for reading and storing the accumulated histograms and
            other ancillary data. Ray tracing will resume the histogramming
            from the state when the persistent file was written. If the file
            does not exist yet, the histograms are initialized to zeros. The
            persistent file is rewritten when ray tracing is completed and
            the number of repeats > 0.

            .. warning::
                Be careful when you use it: if you intend to start from zeros,
                make sure that this option is switched off or the pickle files
                do not exist! Otherwise you do resume, not really start anew.

            if *persistentName* ends with '.mat', a Matlab file is generated.

        *oe*: instance of an optical element or None
            If supplied, the rectangular or circular areas of the optical
            surfaces or physical surfaces, if the optical surfaces are not
            specified, will be overdrawn. Useful with raycing backend for
            footprint images.

        *raycingParam*: int
            Used together with the *oe* parameter above for drawing footprint
            envelopes. If =2, the limits of the second crystal of DCM are taken
            for drawing the envelope; if =1000, all facets of a diced crystal
            are displayed.

        *beamState*: str
            Used in raycing backend. If not None, gives another beam that
            determines the state (good, lost etc.) instead of the state given
            by *beam*. This may be used to visualize the *incoming* beam but
            use the states of the *outgoing* beam, so that you see how the beam
            upstream of the optical element will be masked by it. See the
            examples for capillaries.

        *beamC*: str
            The same as *beamState* but refers to colors (when not of
            'category' type).


        """
        if not hasQt:
            useQtWidget = False
        if not useQtWidget:
            plt.ion()
        self.colorSaturation = colorSaturation

        self.beam = beam  # binary shadow image: star, mirr or screen
        if beam is None:
            self.backend = 'raycing'
        elif 'star.' in beam or 'mirr.' in beam or 'screen.' in beam:
            self.backend = 'shadow'
        elif ('dummy' in beam) or (beam == ''):
            self.backend = 'dummy'
        elif isinstance(rayFlag, (tuple, list)):
            self.backend = 'raycing'
        else:
            self.backend = 'dummy'
        self.beamState = beamState
        self.beamC = beamC
        self.rayFlag = rayFlag
        self.fluxKind = fluxKind
        self.fluxUnit = fluxUnit
        if xaxis is None:
            self.xaxis = XYCAxis(defaultXTitle, defaultXUnit)
        else:
            self.xaxis = xaxis
        if yaxis is None:
            self.yaxis = XYCAxis(defaultYTitle, defaultYUnit)
        else:
            self.yaxis = yaxis
        if (caxis is None) or isinstance(caxis, basestring):
            self.caxis = XYCAxis(defaultCTitle, defaultCUnit, factor=1.,)
            self.caxis.fwhmFormatStr = defaultFwhmFormatStrForCAxis
            if isinstance(caxis, basestring):
                self.caxis.useCategory = True
                ePos = 0
        else:
            self.caxis = caxis

        if self.backend != 'dummy':
            for axis in self.xaxis, self.yaxis, self.caxis:
                if axis.data == 'auto':
                    axis.auto_assign_data(self.backend)
                if axis.factor is None:
                    axis.auto_assign_factor(self.backend)

        self.reset_bins2D()

        if isinstance(aspect, (int, float)):
            if aspect <= 0:
                aspect = 1.
        self.aspect = aspect
        self.dpi = dpi

        self.ePos = ePos  # Position of E histogram, 1=right, 2=top, 0=none

        self.negative = negative
        if self.negative:
            facecolor = 'w'  # white
        else:
            facecolor = 'k'  # black
        # MatplotlibDeprecationWarning: The axisbg attribute was deprecated in
        # version 2.0. Use facecolor instead.
        kwmpl = {}
        if versiontuple(mpl.__version__) >= versiontuple("2.0.0"):
            kwmpl['facecolor'] = facecolor
        else:
            kwmpl['axisbg'] = facecolor

        self.invertColorMap = invertColorMap
        self.utilityInvertColorMap = False
        self.fluxFormatStr = fluxFormatStr
        self.saveName = saveName
        self.persistentName = persistentName
        self.cx, self.dx = 0, 0
        self.cy, self.dy = 0, 0
        self.cE, self.dE = 0, 0

        xFigSize = float(xOrigin2d + self.xaxis.pixels + space2dto1d +
                         height1d + xSpaceExtra)
        yFigSize = float(yOrigin2d + self.yaxis.pixels + space2dto1d +
                         height1d + ySpaceExtra)
        if self.ePos == 1:
            xFigSize += xspace1dtoE1d + heightE1d + heightE1dbar
        elif self.ePos == 2:
            yFigSize += yspace1dtoE1d + heightE1d + heightE1dbar
        if self.ePos != 1:
            xFigSize += xSpaceExtraWhenNoEHistogram

        if useQtWidget:
            self.fig = Figure(figsize=(xFigSize/dpi, yFigSize/dpi), dpi=dpi)
        else:
            self.fig = plt.figure(figsize=(xFigSize/dpi, yFigSize/dpi),
                                  dpi=dpi)
        self.local_size_inches = self.fig.get_size_inches()

        self.fig.delaxes(self.fig.gca())
        if title != '':
            self.title = title
        elif isinstance(beam, basestring):
            self.title = beam
        else:
            self.title = ' '
        if useQtWidget:
            self.canvas = MyQtFigCanvas(figure=self.fig, xrtplot=self)

        try:
            self.fig.canvas.manager.set_window_title(self.title)
        except AttributeError:
            pass

        if plt.get_backend().lower() in (
                x.lower() for x in mpl.rcsetup.non_interactive_bk):
            xExtra = 0  # mpl backend-dependent (don't know why) pixel sizes
            yExtra = 0  # mpl backend-dependent (don't know why) pixel sizes
        else:  # interactive backends:
            if True:  # runner.runCardVals.repeats > 1:
                xExtra = 0
                yExtra = 2
            else:
                xExtra = 0
                yExtra = 0

        frameon = True
        rect2d = [xOrigin2d / xFigSize, yOrigin2d / yFigSize,
                  (self.xaxis.pixels-1+xExtra) / xFigSize,
                  (self.yaxis.pixels-1+yExtra) / yFigSize]
        self.ax2dHist = self.fig.add_axes(
            rect2d, aspect=aspect, xlabel=self.xaxis.displayLabel,
            ylabel=self.yaxis.displayLabel, autoscale_on=False,
            frameon=frameon, **kwmpl)
        self.ax2dHist.xaxis.labelpad = xlabelpad
        self.ax2dHist.yaxis.labelpad = ylabelpad

        rect1dX = copy.deepcopy(rect2d)
        rect1dX[1] = rect2d[1] + rect2d[3] + space2dto1d/yFigSize
        rect1dX[3] = height1d / yFigSize
        self.ax1dHistX = self.fig.add_axes(
            rect1dX, sharex=self.ax2dHist, autoscale_on=False, frameon=frameon,
            visible=(xPos != 0), **kwmpl)

        rect1dY = copy.deepcopy(rect2d)
        rect1dY[0] = rect2d[0] + rect2d[2] + space2dto1d/xFigSize
        rect1dY[2] = height1d / xFigSize
        self.ax1dHistY = self.fig.add_axes(
            rect1dY, sharey=self.ax2dHist, autoscale_on=False, frameon=frameon,
            visible=(yPos != 0), **kwmpl)

        # make some labels invisible
        pset = plt.setp
        pset(
            self.ax1dHistX.get_xticklabels() +
            self.ax1dHistX.get_yticklabels() +
            self.ax1dHistY.get_xticklabels() +
            self.ax1dHistY.get_yticklabels(),
            visible=False)

        self.ax1dHistX.set_yticks([])
        self.ax1dHistY.set_xticks([])

        self.ax1dHistX.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(
            useOffset=False))
        self.ax1dHistY.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(
            useOffset=False))
#        for tick in (self.ax2dHist.xaxis.get_major_ticks() + \
#          self.ax2dHist.yaxis.get_major_ticks()):
#            tick.label1.set_fontsize(axisLabelFontSize)

        self.ax1dHistXOffset = self.fig.text(
            rect1dY[0]+rect1dY[2], 0.01, '', ha='right', va='bottom',
            color='gray')  # , fontweight='bold')
        self.ax1dHistYOffset = self.fig.text(
            0.01, rect1dX[1]+rect1dX[3], '', rotation=90, ha='left', va='top',
            color='gray')  # , fontweight='bold')

        if self.ePos == 1:  # right
            rect1dE = copy.deepcopy(rect1dY)
            rect1dE[0] = rect1dY[0] + rect1dY[2] + xspace1dtoE1d/xFigSize
            rect1dE[2] = heightE1dbar / xFigSize
            rect1dE[3] *= float(self.caxis.pixels) / self.yaxis.pixels
            self.ax1dHistEbar = self.fig.add_axes(
                rect1dE, ylabel=self.caxis.displayLabel, autoscale_on=False,
                frameon=frameon, **kwmpl)
            self.ax1dHistEbar.yaxis.labelpad = xlabelpad
            self.ax1dHistEOffset = self.fig.text(
                rect1dE[0], rect1dE[1]+rect1dE[3], '', ha='left', va='bottom',
                color='g')  # , fontweight='bold')
            rect1dE[0] += rect1dE[2]
            rect1dE[2] = heightE1d / xFigSize
            self.ax1dHistE = self.fig.add_axes(
                rect1dE, sharey=self.ax1dHistEbar, autoscale_on=False,
                frameon=frameon, **kwmpl)
            pset(
                self.ax1dHistEbar.get_xticklabels() +
                self.ax1dHistE.get_xticklabels() +
                self.ax1dHistE.get_yticklabels(), visible=False)
            pset(self.ax1dHistEbar, xticks=())
            self.ax1dHistE.yaxis.set_major_formatter(
                mpl.ticker.ScalarFormatter(useOffset=False))
            if self.caxis.limits is not None:
                self.ax1dHistE.set_ylim(self.caxis.limits)
            self.ax1dHistE.set_xticks([])
        elif self.ePos == 2:  # top
            rect1dE = copy.deepcopy(rect1dX)
            rect1dE[1] = rect1dX[1] + rect1dX[3] + yspace1dtoE1d/yFigSize
            rect1dE[3] = heightE1dbar / yFigSize
            rect1dE[2] *= float(self.caxis.pixels) / self.xaxis.pixels
            self.ax1dHistEbar = self.fig.add_axes(
                rect1dE, xlabel=self.caxis.displayLabel, autoscale_on=False,
                frameon=frameon, **kwmpl)
            self.ax1dHistEbar.xaxis.labelpad = xlabelpad
            self.ax1dHistEOffset = self.fig.text(
                rect1dE[0]+rect1dE[2]+0.01, rect1dE[1]-0.01, '',
                ha='left', va='top', color='g')  # , fontweight='bold')
            rect1dE[1] += rect1dE[3]
            rect1dE[3] = heightE1d / yFigSize
            self.ax1dHistE = self.fig.add_axes(
                rect1dE, sharex=self.ax1dHistEbar, autoscale_on=False,
                frameon=frameon, **kwmpl)
            pset(
                self.ax1dHistEbar.get_yticklabels() +
                self.ax1dHistE.get_yticklabels() +
                self.ax1dHistE.get_xticklabels(), visible=False)
            pset(self.ax1dHistEbar, yticks=())
            self.ax1dHistE.xaxis.set_major_formatter(
                mpl.ticker.ScalarFormatter(useOffset=False))
            if self.caxis.limits is not None:
                self.ax1dHistE.set_xlim(self.caxis.limits)
            self.ax1dHistE.set_yticks([])

        allAxes = [self.ax1dHistX, self.ax1dHistY, self.ax2dHist]
        if self.ePos != 0:
            allAxes.append(self.ax1dHistE)
            allAxes.append(self.ax1dHistEbar)
        for ax in allAxes:
            for axXY in (ax.xaxis, ax.yaxis):
                for line in axXY.get_ticklines():
                    line.set_color('grey')

        mplTxt = self.ax1dHistX.text if useQtWidget else plt.text

        if self.ePos == 1:
            self.textDE = mplTxt(
                xTextPosDy, yTextPosDy, ' ', rotation='vertical',
                transform=self.ax1dHistE.transAxes, ha='left', va='center')
        elif self.ePos == 2:
            self.textDE = mplTxt(
                xTextPosDx, yTextPosDx, ' ',
                transform=self.ax1dHistE.transAxes, ha='center', va='bottom')

        self.nRaysAll = np.int64(0)
        self.nRaysAllRestored = np.int64(-1)
        self.intensity = 0.
        transform = self.ax1dHistX.transAxes
        self.textGoodrays = None
        self.textI = None
        self.power = 0.
        self.flux = 0.
        self.contourLevels = contourLevels
        self.contourColors = contourColors
        self.contourFmt = contourFmt
        self.contourFactor = contourFactor
        self.displayAsAbsorbedPower = False

        self.textNrays = None
        if self.backend == 'shadow' or self.backend == 'dummy':
            self.textNrays = mplTxt(
                xTextPos, yTextPosNrays, ' ', transform=transform, ha='left',
                va='top')
            self.nRaysNeeded = np.int64(0)
            if self.rayFlag != 2:
                self.textGoodrays = mplTxt(
                    xTextPos, yTextPosGoodrays, ' ', transform=transform,
                    ha='left', va='top')
            self.textI = mplTxt(
                xTextPos, yTextPosI, ' ', transform=transform, ha='left',
                va='top')
        elif self.backend == 'raycing':
            # =0: ignored, =1: good,
            # =2: reflected outside of working area, =3: transmitted without
            #     intersection
            # =-NN: lost (absorbed) at OE#NN-OE numbering starts from 1 !!!
            #       If NN>1000 then
            # the slit with ordinal number NN-1000 is meant.
            self.nRaysAlive = np.int64(0)
            self.nRaysGood = np.int64(0)
            self.nRaysOut = np.int64(0)
            self.nRaysOver = np.int64(0)
            self.nRaysDead = np.int64(0)
            self.nRaysAccepted = np.int64(0)
            self.nRaysAcceptedE = 0.
            self.nRaysSeeded = np.int64(0)
            self.nRaysSeededI = 0.
            self.textNrays = mplTxt(
                xTextPos, yTextPosNraysR, ' ', transform=transform, ha='left',
                va='top')
            self.textGood = None
            self.textOut = None
            self.textOver = None
            self.textAlive = None
            self.textDead = None
            if 1 in self.rayFlag:
                self.textGood = mplTxt(
                    xTextPos, yTextPosNrays1, ' ', transform=transform,
                    ha='left', va='top')
            if 2 in self.rayFlag:
                self.textOut = mplTxt(
                    xTextPos, yTextPosNrays2, ' ', transform=transform,
                    ha='left', va='top')
            if 3 in self.rayFlag:
                self.textOver = mplTxt(
                    xTextPos, yTextPosNrays3, ' ', transform=transform,
                    ha='left', va='top')
            if 4 in self.rayFlag:
                self.textAlive = mplTxt(
                    xTextPos, yTextPosGoodraysR, ' ', transform=transform,
                    ha='left', va='top')
            if not self.caxis.useCategory:
                self.textI = mplTxt(
                    xTextPos, yTextPosNrays4, ' ', transform=transform,
                    ha='left', va='top')
            else:
                if (np.array(self.rayFlag) < 0).sum() > 0:
                    self.textDead = mplTxt(
                        xTextPos, yTextPosNrays4, ' ', transform=transform,
                        ha='left', va='top')

        self.textDx = mplTxt(
            xTextPosDx, yTextPosDx, ' ', transform=self.ax1dHistX.transAxes,
            ha='center', va='bottom')
        self.textDy = mplTxt(
            xTextPosDy, yTextPosDy, ' ', rotation='vertical',
            transform=self.ax1dHistY.transAxes, ha='left', va='center')
        self.textStatus = mplTxt(
            xTextPosStatus, yTextPosStatus, '', transform=self.fig.transFigure,
            ha='right', va='bottom', fontsize=9)
        self.textStatus.set_color('r')

        self.ax1dHistX.imshow(
            np.zeros((2, 2, 3)), aspect='auto', interpolation='nearest',
            origin='lower', figure=self.fig)
        self.ax1dHistY.imshow(
            np.zeros((2, 2, 3)), aspect='auto', interpolation='nearest',
            origin='lower', figure=self.fig)
        if self.ePos != 0:
            self.ax1dHistE.imshow(
                np.zeros((2, 2, 3)), aspect='auto', interpolation='nearest',
                origin='lower', figure=self.fig)
            self.ax1dHistEbar.imshow(
                np.zeros((2, 2, 3)), aspect='auto', interpolation='nearest',
                origin='lower', figure=self.fig)
        self.ax2dHist.imshow(
            np.zeros((2, 2, 3)), aspect=self.aspect, interpolation='nearest',
            origin='lower', figure=self.fig)
        self.contours2D = None
        self.contours2DLabels = None

        self.oe = oe
        self.oeSurfaceLabels = []
        self.raycingParam = raycingParam
        self.draw_footprint_area()
        if self.xaxis.limits is not None:
            if not isinstance(self.xaxis.limits, str):
                self.ax2dHist.set_xlim(self.xaxis.limits)
                self.ax1dHistX.set_xlim(self.xaxis.limits)
        if self.yaxis.limits is not None:
            if not isinstance(self.yaxis.limits, str):
                self.ax2dHist.set_ylim(self.yaxis.limits)
                self.ax1dHistY.set_ylim(self.yaxis.limits)

        self.cidp = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        if not useQtWidget:
            plt.ioff()
        self.fig.canvas.draw()

    def reset_bins2D(self):
        if self.fluxKind.startswith('E'):
            dtype = np.complex128
        else:
            dtype = np.float64
        self.total2D = np.zeros((self.yaxis.bins, self.xaxis.bins),
                                dtype=dtype)
        self.total2D_RGB = np.zeros((self.yaxis.bins, self.xaxis.bins, 3))
        self.max2D_RGB = 0
        self.globalMax2D_RGB = 0
        self.size2D = self.yaxis.bins * self.xaxis.bins
        self.is4D = self.fluxKind.lower().endswith('4d')
        if self.is4D:
            self.total4D = np.zeros((self.size2D, self.size2D), dtype=dtype)
        self.isPCA = self.fluxKind.lower().endswith('pca')
        if self.isPCA:
            self.total4D = []

        for ax in [self.xaxis, self.yaxis, self.caxis]:
            if isinstance(ax, XYCAxis):
                ax.binEdges = np.zeros(ax.bins + 1)
                ax.total1D = np.zeros(ax.bins)
                ax.total1D_RGB = np.zeros((ax.bins, 3))

    def update_user_elements(self):
        return  # 'user message'

    def clean_user_elements(self):
        pass

    def on_press(self, event):
        """
        Defines the right button click event for stopping the loop.
        """
        if event.button == 3:
            runner.runCardVals.stop_event.set()
            self.textStatus.set_text("stopping ...")

    def timer_callback(self, evt=None):
        """
        This code will be executed on every timer tick. We have to start
        :meth:`runner.dispatch_jobs` here as otherwise we cannot force the
        redrawing.
        """
        if self.areProcessAlreadyRunning:
            return
        self.areProcessAlreadyRunning = True
        runner.dispatch_jobs()

    def set_axes_limits(self, xmin, xmax, ymin, ymax, emin, emax):
        """
        Used in multiprocessing for automatic limits of the 3 axes: x, y and
        energy (caxis). It is meant only for the 1st ray tracing run.
        """
#        if (self.xaxis.limits is None) or isinstance(self.xaxis.limits, str):
# the check is not needed: even if the limits have been already set, they may
# change due to *aspect*; this is checked in :mod:`multipro`.
        self.xaxis.limits = [xmin, xmax]
        self.yaxis.limits = [ymin, ymax]
        self.caxis.limits = [emin, emax]

    def draw_footprint_area(self):
        """
        Useful with raycing backend for footprint images.
        """
        if self.oe is None:
            return
        if self.oe.surface is None:
            return
        if isinstance(self.oe.surface, basestring):
            surface = self.oe.surface,
        else:
            surface = self.oe.surface

        if len(self.oeSurfaceLabels) > 0:
            for isurf, surf in enumerate(surface):
                self.oeSurfaceLabels[isurf].set_text(surf)
                return
        r = [0, 0, 0, 0]
        if self.raycingParam == 2:  # the second crystal of DCM
            limsPhys = self.oe.limPhysX2, self.oe.limPhysY2
            limsOpt = self.oe.limOptX2, self.oe.limOptY2
        elif (self.raycingParam >= 1000) and hasattr(self.oe, "xStep"):
            # all facets of a diced crystal
            if self.oe.limPhysX[1] == np.inf:
                return
            if self.oe.limPhysY[1] == np.inf:
                return
            if self.xaxis.limits is None:
                return
            if self.yaxis.limits is None:
                return
            ixMin = int(round(max(self.oe.limPhysX[0], self.xaxis.limits[0]) /
                        self.oe.xStep))
            ixMax = int(round(min(self.oe.limPhysX[1], self.xaxis.limits[1]) /
                        self.oe.xStep))
            iyMin = int(round(max(self.oe.limPhysY[0], self.yaxis.limits[0]) /
                        self.oe.yStep))
            iyMax = int(round(min(self.oe.limPhysY[1], self.yaxis.limits[1]) /
                        self.oe.yStep))
            surface = []
            limFacetXMin, limFacetXMax = [], []
            limFacetYMin, limFacetYMax = [], []
            for ix in range(ixMin, ixMax+1):
                for iy in range(iyMin, iyMax+1):
                    surface.append('')
                    cx = ix * self.oe.xStep
                    cy = iy * self.oe.yStep
                    dxHalf = self.oe.dxFacet / 2
                    dyHalf = self.oe.dyFacet / 2
                    limFacetXMin.append(max(cx-dxHalf, self.oe.limPhysX[0]))
                    limFacetXMax.append(min(cx+dxHalf, self.oe.limPhysX[1]))
                    limFacetYMin.append(max(cy-dyHalf, self.oe.limPhysY[0]))
                    limFacetYMax.append(min(cy+dyHalf, self.oe.limPhysY[1]))
            limsPhys = \
                (limFacetXMin, limFacetXMax), (limFacetYMin, limFacetYMax)
            limsOpt = None, None
        else:
            limsPhys = self.oe.limPhysX, self.oe.limPhysY
            limsOpt = self.oe.limOptX, self.oe.limOptY
        for isurf, surf in enumerate(surface):
            for ilim1, ilim2, limPhys, limOpt in zip(
                    (0, 2), (1, 3), limsPhys, limsOpt):
                if limOpt is not None:
                    if raycing.is_sequence(limOpt[0]):
                        r[ilim1], r[ilim2] = limOpt[0][isurf], limOpt[1][isurf]
                    else:
                        r[ilim1], r[ilim2] = limOpt[0], limOpt[1]
                else:
                    if raycing.is_sequence(limPhys[0]):
                        r[ilim1], r[ilim2] = \
                            limPhys[0][isurf], limPhys[1][isurf]
                    else:
                        r[ilim1], r[ilim2] = limPhys[0], limPhys[1]
            r[0] *= self.xaxis.factor
            r[1] *= self.xaxis.factor
            r[2] *= self.yaxis.factor
            r[3] *= self.yaxis.factor
            if isinstance(self.oe.shape, (str, unicode)):
                if self.oe.shape.startswith('ro') and\
                        (self.raycingParam < 1000):
                    envelope = mpl.patches.Circle(
                        ((r[1]+r[0])*0.5, (r[3]+r[2])*0.5), (r[1]-r[0])*0.5,
                        fc="#aaaaaa", lw=0, alpha=0.25)
                elif self.oe.shape.startswith('rect') or\
                        (self.raycingParam >= 1000):
                    envelope = mpl.patches.Rectangle(
                        (r[0], r[2]), r[1] - r[0], r[3] - r[2],
                        fc="#aaaaaa", lw=0, alpha=0.25)
            elif isinstance(self.oe.shape, list):
                envelope = mpl.patches.Polygon(self.oe.shape, closed=True,
                                               fc="#aaaaaa", lw=0, alpha=0.25)
            self.ax2dHist.add_patch(envelope)
            if self.raycingParam < 1000:
                if self.yaxis.limits is not None:
                    yTextPos = max(r[2], self.yaxis.limits[0])
                else:
                    yTextPos = r[2]
                # osl = self.ax2dHist.text(
                #     (r[0]+r[1]) * 0.5, yTextPos, surf, ha='center',
                #     va='top', color='w')
                osl = self.ax2dHist.text(
                    r[1], yTextPos, surf, ha='right', va='bottom', color='w',
                    # fontweight='bold',
                    alpha=0.5)
                self.oeSurfaceLabels.append(osl)

    def plot_hist1d(self, what_axis_char):
        """Plots the specified 1D histogram as imshow and calculates FWHM with
        showing the ends of the FWHM bar.
        Parameters:
            *what_axis_char*: str [ 'x' | 'y' | 'c' ]
                defines the axis
        Returns:
            *center*, *fwhm*: floats
                the center and fwhm values for later displaying.
        """
        if what_axis_char == 'x':
            axis = self.xaxis
            graph = self.ax1dHistX
            orientation = 'horizontal'
            histoPixelHeight = height1d
            offsetText = self.ax1dHistXOffset
        elif what_axis_char == 'y':
            axis = self.yaxis
            graph = self.ax1dHistY
            orientation = 'vertical'
            histoPixelHeight = height1d
            offsetText = self.ax1dHistYOffset
        elif what_axis_char == 'c':
            axis = self.caxis
            graph = self.ax1dHistE
            if self.ePos == 1:
                orientation = 'vertical'
            elif self.ePos == 2:
                orientation = 'horizontal'
            offsetText = self.ax1dHistEOffset
            histoPixelHeight = heightE1d

        t1D = axis.total1D
        axis.max1D = float(np.max(t1D))
        if axis.max1D > epsHist:
            if runner.runCardVals.passNo > 0:
                mult = 1.0 / axis.globalMax1D
            else:
                mult = 1.0 / axis.max1D
            xx = t1D * mult
        else:
            xx = t1D
        if runner.runCardVals.passNo > 0:
            xxMaxHalf = float(np.max(xx)) * 0.5  # for calculating FWHM
        else:
            xxMaxHalf = 0.5

        t1D_RGB = axis.total1D_RGB
        axis.max1D_RGB = float(np.max(t1D_RGB))
        if axis.max1D_RGB > epsHist:
            if runner.runCardVals.passNo > 1:
                mult = 1.0 / axis.globalMax1D_RGB
            else:
                mult = 1.0 / axis.max1D_RGB
            xxRGB = t1D_RGB * mult
        else:
            xxRGB = t1D_RGB

        if orientation[0] == 'h':
            map2d = np.zeros((histoPixelHeight, len(xx), 3))
            for ix, cx in enumerate(xx):
                maxPixel = int(round((histoPixelHeight-1) * cx))
                if 0 <= maxPixel <= (histoPixelHeight-1):
                    map2d[0:maxPixel, ix, :] = xxRGB[ix, :]
                    if axis.outline:
                        maxRGB = np.max(xxRGB[ix, :])
                        if maxRGB > 1e-20:
                            scaleFactor = \
                                1 - axis.outline + axis.outline/maxRGB
                            map2d[maxPixel-1, ix, :] *= scaleFactor
            extent = None
            if (axis.limits is not None) and\
                    (not isinstance(axis.limits, str)):
                ll = [lim-axis.offset for lim in axis.limits]
                extent = [ll[0], ll[1], 0, 1]
        elif orientation[0] == 'v':
            map2d = np.zeros((len(xx), histoPixelHeight, 3))
            for ix, cx in enumerate(xx):
                maxPixel = int(round((histoPixelHeight-1) * cx))
                if 0 <= maxPixel <= (histoPixelHeight-1):
                    map2d[ix, 0:maxPixel, :] = xxRGB[ix, :]
                    if axis.outline:
                        maxRGB = np.max(xxRGB[ix, :])
                        if maxRGB > 1e-20:
                            scaleFactor = \
                                1 - axis.outline + axis.outline/maxRGB
                            map2d[ix, maxPixel-1, :] *= scaleFactor
            extent = None
            if (axis.limits is not None) and \
                    not (isinstance(axis.limits, str)):
                ll = [lim-axis.offset for lim in axis.limits]
                extent = [0, 1, ll[0], ll[1]]

        if self.negative:
            map2d = 1 - map2d
        if self.utilityInvertColorMap:
            map2d = mpl.colors.rgb_to_hsv(map2d)
            map2d[:, :, 0] -= 0.5
            map2d[map2d < 0] += 1
            map2d = mpl.colors.hsv_to_rgb(map2d)
        graph.images[0].set_data(map2d)
        if extent is not None:
            graph.images[0].set_extent(extent)

        # del graph.lines[:]  # otherwise it accumulates the FWHM lines
        for line in graph.lines:
            line.remove()

        axis.binCenters = (axis.binEdges[:-1]+axis.binEdges[1:]) * 0.5
        if axis.max1D > 0:
            wantDiscrete = (xx[0] > xxMaxHalf) or (xx[-1] > xxMaxHalf)
            if not wantDiscrete:
                try:
                    # spl = UnivariateSpline(axis.binCenters, xx-xxMaxHalf, s=0)
                    # roots = spl.roots()
                    spl = make_interp_spline(axis.binCenters, xx-xxMaxHalf)
                    roots = PPoly.from_spline(spl, False).roots()
                    histFWHMlow = min(roots) - axis.offset
                    histFWHMhigh = max(roots) - axis.offset
                except ValueError:
                    wantDiscrete = True
            if wantDiscrete:
                args = np.argwhere(xx >= xxMaxHalf)
                iHistFWHMlow = np.min(args)
                iHistFWHMhigh = np.max(args) + 1
                histFWHMlow = axis.binEdges[iHistFWHMlow] - axis.offset
                histFWHMhigh = axis.binEdges[iHistFWHMhigh] - axis.offset

            if axis.fwhmFormatStr is not None:
                xFWHM = [histFWHMlow, histFWHMhigh]
                yFWHM = [xxMaxHalf, xxMaxHalf]
                if orientation[0] == 'h':
                    graph.plot(xFWHM, yFWHM, '+', color='grey')
                elif orientation[0] == 'v':
                    graph.plot(yFWHM, xFWHM, '+', color='grey')
        else:
            histFWHMlow = 0
            histFWHMhigh = 0

        if axis.offset:
            ll = [lim-axis.offset for lim in axis.limits]
            offsetText.set_text('{0}{1:g} {2}'.format(
                '+' if axis.offset > 0 else '',
                axis.offset*axis.offsetDisplayFactor, axis.offsetDisplayUnit))
            offsetText.set_visible(True)
        else:
            ll = axis.limits
            offsetText.set_visible(False)

        if orientation[0] == 'h':
            if not isinstance(axis.limits, str):
                graph.set_xlim(ll)
            graph.set_ylim([0, 1])
        elif orientation[0] == 'v':
            graph.set_xlim([0, 1])
            if not isinstance(axis.limits, str):
                graph.set_ylim(ll)

        weighted1D = axis.total1D * axis.binCenters
        xxAve = axis.total1D.sum()
        if xxAve != 0:
            xxAve = weighted1D.sum() / xxAve
        return xxAve, histFWHMhigh - histFWHMlow

    def plot_colorbar(self):
        """
        Plots a color bar adjacent to the caxis 1D histogram.
        """
        a = np.linspace(0, colorFactor, self.caxis.pixels, endpoint=True)
        a = np.asarray(a).reshape(1, -1)
        if self.invertColorMap:
            a -= 0.5
            a[a < 0] += 1
        if self.caxis.limits is None:
            return
        eMin, eMax = [lim-self.caxis.offset for lim in self.caxis.limits]
        a = np.vstack((a, a))
        if self.ePos == 1:
            a = a.T
            extent = [0, 1, eMin, eMax]
        else:
            extent = [eMin, eMax, 0, 1]

        a = np.dstack(
            (a, np.ones_like(a) * self.colorSaturation, np.ones_like(a)))
        a = mpl.colors.hsv_to_rgb(a)
        if self.negative:
            a = 1 - a
        self.ax1dHistEbar.images[0].set_data(a)
        self.ax1dHistEbar.images[0].set_extent(extent)

        if self.caxis.invertAxis:
            if self.ePos == 2:
                self.ax1dHistEbar.set_xlim(self.ax1dHistEbar.get_xlim()[::-1])
            elif self.ePos == 1:
                self.ax1dHistEbar.set_ylim(self.ax1dHistEbar.get_ylim()[::-1])

    def plot_hist2d(self):
        """
        Plots the 2D histogram as imshow.
        """
        tRGB = self.total2D_RGB
        self.max2D_RGB = float(np.max(tRGB))
        if self.max2D_RGB > 0:
            if runner.runCardVals.passNo > 1:
                mult = 1.0 / self.globalMax2D_RGB
            else:
                mult = 1.0 / self.max2D_RGB
            xyRGB = tRGB * mult
        else:
            xyRGB = tRGB
        if self.negative:
            xyRGB = 1 - xyRGB
        if self.utilityInvertColorMap:
            xyRGB = mpl.colors.rgb_to_hsv(xyRGB)
            xyRGB[:, :, 0] -= 0.5
            xyRGB[xyRGB < 0] += 1
            xyRGB = mpl.colors.hsv_to_rgb(xyRGB)
        xyRGB[xyRGB < 0] = 0
        xyRGB[xyRGB > 1] = 1
# #test:
#        xyRGB[:,:,:]=0
#        xyRGB[1::2,1::2,0]=1
        extent = None
        if (self.xaxis.limits is not None) and (self.yaxis.limits is not None):
            if (not isinstance(self.xaxis.limits, str)) and\
               (not isinstance(self.yaxis.limits, str)):
                extent = [self.xaxis.limits[0]-self.xaxis.offset,
                          self.xaxis.limits[1]-self.xaxis.offset,
                          self.yaxis.limits[0]-self.yaxis.offset,
                          self.yaxis.limits[1]-self.yaxis.offset]
        self.ax2dHist.images[0].set_data(xyRGB)
        if extent is not None:
            self.ax2dHist.images[0].set_extent(extent)

        if self.xaxis.invertAxis:
            self.ax2dHist.set_xlim(self.ax2dHist.get_xlim()[::-1])
        if self.yaxis.invertAxis:
            self.ax2dHist.set_ylim(self.ax2dHist.get_ylim()[::-1])

        if self.contourLevels is not None:
            if self.contours2D is not None:
                for contour in self.contours2D.collections:
                    try:
                        contour.remove()
                    except ValueError:
                        pass
                for label in self.contours2DLabels:
                    try:
                        label.remove()
                    except ValueError:
                        pass
            dx = float(self.xaxis.limits[1]-self.xaxis.limits[0]) /\
                self.xaxis.bins
            dy = float(self.yaxis.limits[1]-self.yaxis.limits[0]) /\
                self.yaxis.bins
            if dx == 0:
                dx = 1.
            if dy == 0:
                dy = 1.
            x = np.linspace(
                self.xaxis.limits[0] + dx/2, self.xaxis.limits[1] - dx/2,
                self.xaxis.bins)
            y = np.linspace(
                self.yaxis.limits[0] + dy/2, self.yaxis.limits[1] - dy/2,
                self.yaxis.bins)
            X, Y = np.meshgrid(x, y)
            norm = self.nRaysAll * dx * dy
            if norm > 0:
                Z = copy.copy(self.total2D) / norm
                Z = sp.ndimage.filters.gaussian_filter(Z, 3, mode='nearest')\
                    * self.contourFactor
                self.contourMax = np.max(Z)
                if True:  # self.contourMax > 1e-4:
                    contourLevels =\
                        [lev*self.contourMax for lev in self.contourLevels]
                    self.contours2D = self.ax2dHist.contour(
                        X, Y, Z, levels=contourLevels,
                        colors=self.contourColors)
                    self.contours2DLabels = self.ax2dHist.clabel(
                        self.contours2D, fmt=self.contourFmt, inline=True,
                        fontsize=10)

    def textFWHM(self, axis, textD, average, hwhm):
        """Updates the text field that has average of the *axis* plus-minus the
        HWHM value."""
        deltaStr = axis.label + '$ = $' + axis.fwhmFormatStr +\
            r'$\pm$' + axis.fwhmFormatStr + ' %s'
        textD.set_text(deltaStr % (average, hwhm, axis.unit))

    def _pow10(self, x, digits=1):
        """
        Returns a string representation of the scientific notation of the given
        number formatted for use with LaTeX or Mathtext, with specified number
        of significant decimal digits.
        """
        x = float(x)
        if (x <= 0) or np.isnan(x).any():
            return '0'
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = np.round(x / float(10**exponent), digits)
        return r"{0:.{2}f}$\cdot$10$^{{{1:d}}}$".format(
            coeff, exponent, digits)

#    def _round_to_n(self, x, n):
#        """Round x to n significant figures"""
#        return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)
#
#    def _str_fmt10(self, x, n=2):
#        " Format x into nice Latex rounding to n"
#        if x <= 0: return "0"
#        try:
#            power = int(np.log10(self._round_to_n(x, 0)))
#            f_SF = self._round_to_n(x, n) * pow(10, -power)
#        except OverflowError:
#            return "0"
#        return r"{0}$\cdot$10$^{{{1}}}$".format(f_SF, power)

    def _get_flux(self):
        self.flux = float(self.intensity) / self.nRaysAll *\
            self.nRaysSeededI / self.nRaysSeeded

    def _get_power(self):
        self.power = self.intensity / self.nRaysAll

    def plot_plots(self):
        """
        Does all graphics update.
        """
        self.cx, self.dx = self.plot_hist1d('x')
        self.cy, self.dy = self.plot_hist1d('y')

        if self.ePos != 0:
            self.cE, self.dE = self.plot_hist1d('c')
            self.plot_colorbar()
            if self.caxis.fwhmFormatStr is not None:
                self.textFWHM(self.caxis, self.textDE, self.cE, self.dE/2)
        self.plot_hist2d()

        if self.textNrays:
            self.textNrays.set_text(r'$N_{\rm all} = $%s' % self.nRaysAll)
        if self.textGoodrays:
            if (runner.runCardVals.backend == 'shadow'):
                strDict = {0: r'lost', 1: r'good'}
                self.textGoodrays.set_text(
                    ''.join([r'$N_{\rm ', strDict[self.rayFlag[0]],
                             r'} = $%s']) % self.nRaysNeeded)
        if self.textI:
            if self.fluxFormatStr == 'auto':
                cond = (self.fluxUnit is None) or \
                    self.fluxKind.startswith('power')
                if (runner.runCardVals.backend == 'raycing'):
                    cond = cond or (self.nRaysSeeded == 0)
                if cond:
                    fluxFormatStr = '%g'
                else:
                    fluxFormatStr = '%.2p'
            else:
                fluxFormatStr = self.fluxFormatStr
            isPowerOfTen = False
            if fluxFormatStr.endswith('p'):
                pos = fluxFormatStr.find('.')
                if 0 < pos+1 < len(fluxFormatStr):
                    isPowerOfTen = True
                    powerOfTenDecN = int(fluxFormatStr[pos+1])
        if (runner.runCardVals.backend == 'raycing'):
            for iTextPanel, iEnergy, iN, substr in zip(
                [self.textGood, self.textOut, self.textOver, self.textAlive,
                 self.textDead],
                [raycing.hueGood, raycing.hueOut, raycing.hueOver, 0,
                 raycing.hueDead],
                [self.nRaysGood, self.nRaysOut, self.nRaysOver,
                 self.nRaysAlive, self.nRaysDead],
                    ['good', 'out', 'over', 'alive', 'dead']):
                if iTextPanel is not None:
                    iTextPanel.set_text(''.join(
                        [r'$N_{\rm ', substr, r'} = $%s']) % iN)
                    if self.caxis.useCategory:
                        eMin, eMax = self.caxis.limits
                        if iEnergy == 0:
                            color = 'black'
                        else:
                            hue = (iEnergy-eMin) / (eMax-eMin) * colorFactor
#                            hue = iEnergy / 10.0 * colorFactor
                            color = np.dstack((hue, 1, 1))
                            color = \
                                mpl.colors.hsv_to_rgb(color)[0, :].reshape(3, )
                        iTextPanel.set_color(color)
            if self.textI:
                if (self.fluxUnit is None) or (self.nRaysSeeded == 0):
                    intensityStr = r'$\Phi = $'
                    if isPowerOfTen:
                        intensityStr += self._pow10(
                            self.intensity, powerOfTenDecN)
                    else:
                        intensityStr += fluxFormatStr % self.intensity
                    self.textI.set_text(intensityStr)
                else:
                    if self.fluxKind.startswith('power'):
                        if self.nRaysAll > 0:
                            self._get_power()
                            if self.displayAsAbsorbedPower:
                                powerStr2 = r'P$_{\rm abs} = $'
                            else:
                                powerStr2 = r'P$_{\rm tot} = $'
                            powerStr = powerStr2 + fluxFormatStr + ' W'
                            self.textI.set_text(powerStr % self.power)
                    else:
                        if (self.nRaysAll > 0) and (self.nRaysSeeded > 0):
                            self._get_flux()
                            if isPowerOfTen:
                                intensityStr = self._pow10(
                                    self.flux, powerOfTenDecN)
                            else:
                                intensityStr = fluxFormatStr % self.flux
                            intensityStr = \
                                r'$\Phi = ${0} ph/s'.format(intensityStr)
                            self.textI.set_text(intensityStr)
            self.update_user_elements()
        if (runner.runCardVals.backend == 'shadow'):
            if self.textI:
                intensityStr = r'$I = $'
                if isPowerOfTen:
                    intensityStr += self._pow10(
                        self.intensity, powerOfTenDecN)
                else:
                    intensityStr += fluxFormatStr % self.intensity
                self.textI.set_text(intensityStr)

        if self.xaxis.fwhmFormatStr is not None:
            self.textFWHM(self.xaxis, self.textDx, self.cx, self.dx/2)
        if self.yaxis.fwhmFormatStr is not None:
            self.textFWHM(self.yaxis, self.textDy, self.cy, self.dy/2)

        self.fig.canvas.draw()

    def save(self, suffix=''):
        """
        Saves matplotlib figures with the *suffix* appended to the file name(s)
        in front of the extension.
        """
        if self.saveName is None:
            return
        if isinstance(self.saveName, basestring):
            fileList = [self.saveName, ]
        else:  # fileList is a sequence
            fileList = self.saveName
        for aName in fileList:
            (fileBaseName, fileExtension) = os.path.splitext(aName)
            saveName = ''.join([fileBaseName, suffix, fileExtension])
            self.fig.savefig(saveName, dpi=self.dpi)
            # otherwise mpl qt backend wants to change it (only in Windows):
            self.fig.set_size_inches(self.local_size_inches)
            self.fig.canvas.draw()

    def clean_plots(self):
        """
        Cleans the graph in order to prepare it for the next ray tracing.
        """
        runner.runCardVals.iteration = 0
        runner.runCardVals.stop_event.clear()
        runner.runCardVals.finished_event.clear()
        for axis in [self.xaxis, self.yaxis, self.caxis]:
            axis.total1D[:] = np.zeros(axis.bins)
            axis.total1D_RGB[:] = np.zeros((axis.bins, 3))
        self.total2D[:] = np.zeros((self.yaxis.bins, self.xaxis.bins))
        self.total2D_RGB[:] = np.zeros((self.yaxis.bins, self.xaxis.bins, 3))
        if self.is4D:
            if self.fluxKind.startswith('E'):
                dtype = np.complex128
            else:
                dtype = np.float64
            self.total4D[:] = np.zeros((self.size2D, self.size2D), dtype=dtype)
        elif self.isPCA:
            self.total4D = []

        try:
            self.fig.canvas.window().setWindowTitle(self.title)

        except AttributeError:
            pass
        self.nRaysAll = np.int64(0)
        self.nRaysAllRestored = np.int64(-1)
        self.nRaysAccepted = np.int64(0)
        self.nRaysAcceptedE = 0.
        self.nRaysSeeded = np.int64(0)
        self.nRaysSeededI = 0.
        self.intensity = 0.
        self.cidp = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.fig.canvas.draw()
        if self.ePos != 0:
            if self.caxis.fwhmFormatStr is not None:
                self.textDE.set_text('')
        self.textNrays.set_text('')
        if self.backend == 'shadow':
            self.nRaysNeeded = np.int64(0)
            if self.textGoodrays is not None:
                self.textGoodrays.set_text('')
        if self.backend == 'raycing':
            self.nRaysAlive = np.int64(0)
            self.nRaysGood = np.int64(0)
            self.nRaysOut = np.int64(0)
            self.nRaysOver = np.int64(0)
            self.nRaysDead = np.int64(0)
            if self.textGood is not None:
                self.textGood.set_text('')
            if self.textOut is not None:
                self.textOut.set_text('')
            if self.textOver is not None:
                self.textOver.set_text('')
            if self.textAlive is not None:
                self.textAlive.set_text('')
            if self.textDead is not None:
                self.textDead.set_text('')
        if self.textI:
            self.textI.set_text('')
        if self.xaxis.fwhmFormatStr is not None:
            self.textDx.set_text('')
        if self.yaxis.fwhmFormatStr is not None:
            self.textDy.set_text('')
        self.clean_user_elements()
        if self.contours2D is not None:
            for contour in self.contours2D.collections:
                contour.remove()
            for label in self.contours2DLabels:
                label.remove()
            for artist in self.ax2dHist.collections:
                artist.remove()

        self.plot_plots()

    def set_negative(self):
        """
        Utility function. Makes all plots in the graph negative (in color).
        """
        self.negative = not self.negative
        if self.negative:
            facecolor = 'w'  # previously - axisbg (depreceted)
        else:
            facecolor = 'k'
        axesList = [self.ax2dHist, self.ax1dHistX, self.ax1dHistY]
        if self.ePos != 0:
            axesList.append(self.ax1dHistE)
            axesList.append(self.ax1dHistEbar)
        for axes in axesList:
            axes.set_axis_bgcolor(facecolor)
        self.plot_plots()

    def set_invert_colors(self):
        """
        Utility function. Inverts the color map.
        """
        self.invertColorMap = not self.invertColorMap  # this variable is used
        # at the time of handling the ray-tracing arrays, as it is cheaper
        # there but needs an additional inversion at the time of plotting if
        # requested by user.
        self.utilityInvertColorMap = not self.utilityInvertColorMap  # this
        # variable is used at the time of plotting
        self.plot_plots()

    def card_copy(self):
        """
        Returns a minimum set of properties (a "card") describing the plot.
        Used for passing it to a new process or thread.
        """
        return PlotCard2Pickle(self)

    def store_plots(self):
        """
        Pickles the accumulated arrays (histograms) and values (like flux) into
        the binary file *persistentName*.
        """
        saved = SaveResults(self)
        if runner.runCardVals.globalNorm:
            runner.runCardVals.savedResults.append(saved)
        if self.persistentName and (self.nRaysAll > self.nRaysAllRestored):
            if raycing.is_sequence(self.persistentName):
                pn = self.persistentName[0]
            else:
                pn = self.persistentName
            if pn.endswith('mat'):
                import scipy.io as io
                # if os.path.isfile(self.persistentName):
                #     os.remove(self.persistentName)
                io.savemat(pn, vars(saved))
            else:
                f = open(pn, 'wb')
                pickle.dump(saved, f, protocol=2)
                f.close()

    def restore_plots(self):
        """
        Restores itself from a file, if possible.
        """
        try:
            if self.persistentName:
                if raycing.is_sequence(self.persistentName):
                    pns = self.persistentName
                else:
                    pns = self.persistentName,
                for pn in pns:
                    if pn.endswith('mat'):
                        import scipy.io as io
                        saved_dic = {}
                        io.loadmat(pn, saved_dic)
                        saved = SaveResults(self)
                        saved.__dict__.update(saved_dic)
                    else:
                        pickleFile = open(pn, 'rb')
                        saved = pickle.load(pickleFile)
                        pickleFile.close()
                    saved.restore(self)
            if True:  # _DEBUG:
                print('persistentName=', self.persistentName)
                print('saved nRaysAll=', self.nRaysAll)
        except (IOError, TypeError):
            pass


class XYCPlotWithNumerOfReflections(XYCPlot):
    def update_user_elements(self):
        if not hasattr(self, 'ax1dHistE'):
            return
        if hasattr(self, 'textUser'):
            self.clean_user_elements()
        else:
            self.textUser = []
        bins = self.caxis.total1D.nonzero()[0]
        self.ax1dHistE.yaxis.set_major_locator(MaxNLocator(integer=True))
        yPrev = -1e3
        fontSize = 8
        for i, b in enumerate(bins):
            binVal = int(round(abs(
                self.caxis.binEdges[b]+self.caxis.binEdges[b+1]) / 2))
            textOut = ' n({0:.0f})={1:.1%}'.format(
                binVal, self.caxis.total1D[b] / self.intensity)
            y = self.caxis.binEdges[b+1] if i < (len(bins)-1) else\
                self.caxis.binEdges[b]
            tr = self.ax1dHistE.transData.transform
            if abs(tr((0, y))[1] - tr((0, yPrev))[1]) < fontSize:
                continue
            yPrev = y
            color = self.caxis.total1D_RGB[b] / max(self.caxis.total1D_RGB[b])
#            va = 'bottom' if binVal < self.caxis.limits[1] else 'top'
            va = 'bottom' if i < (len(bins) - 1) else 'top'
            myText = self.ax1dHistE.text(
                0, y, textOut, ha='left', va=va, size=fontSize, color=color)
            self.textUser.append(myText)

    def clean_user_elements(self):
        if hasattr(self, 'textUser'):
            for text in reversed(self.ax1dHistE.texts):
                if text in self.textUser:
                    # self.ax1dHistE.texts.remove(text)
                    try:
                        text.remove()
                    except ValueError:
                        pass
                del text
            del self.textUser[:]


class PlotCard2Pickle(object):
    """
    Container for a minimum set of properties (a "card") describing the plot.
    Used for passing it to a new process or thread. Must be pickleable.
    """

    def __init__(self, plot):
        self.xaxis = plot.xaxis
        self.yaxis = plot.yaxis
        self.caxis = plot.caxis
        self.aspect = plot.aspect
        self.beam = plot.beam
        self.beamState = plot.beamState
        self.beamC = plot.beamC
        self.rayFlag = plot.rayFlag
        self.invertColorMap = plot.invertColorMap
        self.ePos = plot.ePos
        self.colorFactor = colorFactor
        self.colorSaturation = colorSaturation
        self.fluxKind = plot.fluxKind
        self.title = plot.title


class SaveResults(object):
    """
    Container for the accumulated arrays (histograms) and values (like flux)
    for subsequent pickling/unpickling or for global flux normalization.
    """

    def __init__(self, plot):
        """
        Stores the arrays and values and finds the global histogram maxima.
        """
        self.xtotal1D = copy.copy(plot.xaxis.total1D)
        self.xtotal1D_RGB = copy.copy(plot.xaxis.total1D_RGB)
        self.ytotal1D = copy.copy(plot.yaxis.total1D)
        self.ytotal1D_RGB = copy.copy(plot.yaxis.total1D_RGB)
        self.etotal1D = copy.copy(plot.caxis.total1D)
        self.etotal1D_RGB = copy.copy(plot.caxis.total1D_RGB)
        self.total2D = copy.copy(plot.total2D)
        self.total2D_RGB = copy.copy(plot.total2D_RGB)

        axes = [plot.xaxis, plot.yaxis]
        if plot.ePos:
            axes.append(plot.caxis)
            self.cE, self.dE = copy.copy(plot.cE), copy.copy(plot.dE)
        self.cx, self.dx = copy.copy(plot.cx), copy.copy(plot.dx)
        self.cy, self.dy = copy.copy(plot.cy), copy.copy(plot.dy)

        for axis in axes:
            if axis.globalMax1D < axis.max1D:
                axis.globalMax1D = axis.max1D
            if axis.globalMax1D_RGB < axis.max1D_RGB:
                axis.globalMax1D_RGB = axis.max1D_RGB
        if plot.globalMax2D_RGB < plot.max2D_RGB:
            plot.globalMax2D_RGB = plot.max2D_RGB
        self.nRaysAll = copy.copy(plot.nRaysAll)
        self.intensity = copy.copy(plot.intensity)
        if plot.backend == 'shadow':
            self.nRaysNeeded = copy.copy(plot.nRaysNeeded)
        elif plot.backend == 'raycing':
            self.nRaysAlive = copy.copy(plot.nRaysAlive)
            self.nRaysGood = copy.copy(plot.nRaysGood)
            self.nRaysOut = copy.copy(plot.nRaysOut)
            self.nRaysOver = copy.copy(plot.nRaysOver)
            self.nRaysDead = copy.copy(plot.nRaysDead)
            if (plot.nRaysSeeded > 0):
                self.nRaysAccepted = copy.copy(plot.nRaysAccepted)
                self.nRaysAcceptedE = copy.copy(plot.nRaysAcceptedE)
                self.nRaysSeeded = copy.copy(plot.nRaysSeeded)
                self.nRaysSeededI = copy.copy(plot.nRaysSeededI)
                self.flux = copy.copy(plot.flux)
            self.power = copy.copy(plot.power)

        self.xlimits = copy.copy(plot.xaxis.limits)
        self.ylimits = copy.copy(plot.yaxis.limits)
        self.elimits = copy.copy(plot.caxis.limits)
        self.xbinEdges = copy.copy(plot.xaxis.binEdges)
        self.ybinEdges = copy.copy(plot.yaxis.binEdges)
        self.ebinEdges = copy.copy(plot.caxis.binEdges)
        self.fluxKind = copy.copy(plot.fluxKind)

    def restore(self, plot):
        """
        Restores the arrays and values after unpickling or after running the
        ray-tracing series and finding the global histogram maxima.
        """
# squeeze is needed even for floats,
# otherwise for matlab it is returned as [[value]]
        plot.xaxis.total1D += np.squeeze(self.xtotal1D)
        plot.xaxis.total1D_RGB += np.squeeze(self.xtotal1D_RGB)
        plot.yaxis.total1D += np.squeeze(self.ytotal1D)
        plot.yaxis.total1D_RGB += np.squeeze(self.ytotal1D_RGB)
        plot.caxis.total1D += np.squeeze(self.etotal1D)
        plot.caxis.total1D_RGB += np.squeeze(self.etotal1D_RGB)
        plot.total2D += np.squeeze(self.total2D)
        plot.total2D_RGB += np.squeeze(self.total2D_RGB)

        plot.nRaysAll += np.squeeze(self.nRaysAll)
        plot.nRaysAllRestored += np.squeeze(self.nRaysAll)
        plot.intensity += np.squeeze(self.intensity)
        if plot.backend == 'shadow':
            plot.nRaysNeeded += np.squeeze(self.nRaysNeeded)
        elif plot.backend == 'raycing':
            plot.nRaysAlive += np.squeeze(self.nRaysAlive)
            plot.nRaysGood += np.squeeze(self.nRaysGood)
            plot.nRaysOut += np.squeeze(self.nRaysOut)
            plot.nRaysOver += np.squeeze(self.nRaysOver)
            plot.nRaysDead += np.squeeze(self.nRaysDead)
            if hasattr(self, 'nRaysSeeded'):
                if self.nRaysSeeded > 0:
                    plot.nRaysAccepted += np.squeeze(self.nRaysAccepted)
                    plot.nRaysAcceptedE += np.squeeze(self.nRaysAcceptedE)
                    plot.nRaysSeeded += np.squeeze(self.nRaysSeeded)
                    plot.nRaysSeededI += np.squeeze(self.nRaysSeededI)

        plot.xaxis.limits = np.copy(np.squeeze(self.xlimits))
        plot.yaxis.limits = np.copy(np.squeeze(self.ylimits))
        plot.caxis.limits = np.copy(np.squeeze(self.elimits))
        plot.xaxis.binEdges = np.copy(np.squeeze(self.xbinEdges))
        plot.yaxis.binEdges = np.copy(np.squeeze(self.ybinEdges))
        plot.caxis.binEdges = np.copy(np.squeeze(self.ebinEdges))
        plot.fluxKind = np.array_str(np.copy(np.squeeze(self.fluxKind)))


#    def __getstate__(self):
#        odict = self.__dict__.copy() # copy the dict since we change it
#        del odict['plot']  # remove plot reference, it cannot be pickled
#        return odict
