# -*- coding: utf-8 -*-
u"""Package xrt (XRayTracer) is a python software library for ray tracing and
wave propagation in x-ray regime. It is primarily meant for modeling
synchrotron sources, beamlines and beamline elements. Includes a GUI for
creating a beamline and interactively viewing it in 3D.

+----------------+---------------+-----------+
|     |Itot|     |   |vcmSi-P|   | |VortexB| |
+----------------+---------------+-----------+

.. |Itot| animation:: _images/Itot
   :alt: &ensp;Intensity of undulator radiation on a transverse flat screen.
       The long axis is energy. One can observe two harmonics crossing the
       central line: the former is odd and the latter is even.

.. |vcmSi-P| animation:: _images/vcmSi-FootprintP
   :alt: &ensp;Absorbed power and power density on a mirror at varying pitch
       angle.
   :loc: upper-right-corner

.. |VortexB| animation:: _images/Laguerre-Gauss-ani
   :alt: &ensp;Vortex beam with <i>l</i>=1 and <i>p</i>=1. Colored by phase.
   :loc: upper-right-corner

.. automodule:: xrt.gui.xrtQook

.. automodule:: xrt.gui.xrtGlow

Features of xrt
---------------

* *Rays and waves*. Classical ray tracing and :ref:`wave propagation <waves>`
  via Kirchhoff integrals, also freely intermixed. No further approximations,
  such as thin lens or paraxial. The optical surfaces may have :ref:`figure
  errors, analytical or measured<warping>`. In wave propagation, partially
  coherent radiation is treated by incoherent addition of coherently diffracted
  fields generated per electron.

* *Publication quality graphics*. 1D and 2D position histograms are
  *simultaneously* coded by hue and brightness. Typically, colors represent
  energy and brightness represents beam intensity. The user may select other
  quantities to be encoded by colors: angular and positional distributions,
  various polarization properties, beam categories, number of reflections,
  incidence angle etc. Brightness can also encode partial flux for a selected
  polarization and incident or absorbed power. Publication quality plots are
  provided by `matplotlib <http://matplotlib.org>`_ with image formats PNG,
  PostScript, PDF and SVG.

* *Unlimited number of rays*. The colored histograms are *cumulative*. The
  accumulation can be stopped and resumed.

* *Parallel execution*. xrt can be run :ref:`in parallel <tests>` in several
  threads or processes (can be opted), which accelerates the execution on
  multi-core computers. Alternatively, xrt can use the power of GPUs via OpenCL
  for running special tasks such as the calculation of an undulator source or
  performing wave propagation.

* *Scripting in Python*. xrt can be run within Python scripts to generate a
  series of images under changing geometrical or physical parameters. The image
  brightness and 1D histograms can be normalized to the global maximum
  throughout the series.

* :ref:`Synchrotron sources <synchrotron-sources>`. Bending magnet, wiggler,
  undulator and elliptic undulator are calculated internally within xrt. There
  is also a legacy approach to sampling synchrotron sources using the codes
  `ws` and `urgent` which are parts of XOP package. Please look the section
  :ref:`comparison-synchrotron-sources` for the comparison between the
  implementations. If the photon source is one of the synchrotron sources, the
  total flux in the beam is reported not just in number of rays but in physical
  units of ph/s. The total power or absorbed power can be opted instead of flux
  and is reported in W. The power density can be visualized by isolines. The
  magnetic gap of undulators can be :ref:`tapered <tapering_comparison>`.
  Undulators can be calculated in :ref:`near field <near_field_comparison>`.
  :ref:`Custom magnetic field <undulator_custom>` is also possible. Undulators
  can be :ref:`calculated on GPU <calculations_on_GPU>`, with a high gain in
  computation speed, which is important for tapering and near field
  calculations.

* *Shapes*. There are several predefined shapes of optical elements implemented
  as python classes. The inheritance mechanism simplifies creation of other
  shapes. The user specifies methods for the surface height and the surface
  normal. For asymmetric crystals, the normal to the atomic planes can be
  additionally given. The surface and the normals are defined either in local
  (x, y) coordinates or in user-defined parametric coordinates. Parametric
  representation enables closed shapes such as capillaries or wave guides. It
  also enables exact solutions for complex shapes (e.g. a logarithmic spiral)
  without any expansion. The methods of finding the intersections of rays with
  the surface are very robust and can cope with pathological cases as sharp
  surface kinks. Notice that the search for intersection points does not
  involve any approximation and has only numerical inaccuracy which is set by
  default as 1 fm. Any surface can be combined with a (differently and variably
  oriented) crystal structure and/or (variable) grating vector. Surfaces can be
  faceted.

* *Energy dispersive elements*. Implemented are :meth:`crystals in dynamical
  diffraction <xrt.backends.raycing.materials.Crystal.get_amplitude>`,
  gratings (also with efficiency calculations), Fresnel zone plates,
  Bragg-Fresnel optics and :meth:`multilayers in dynamical diffraction
  <xrt.backends.raycing.materials.Multilayer.get_amplitude>`. Crystals can work
  in Bragg or Laue cases, in reflection or in transmission. The
  two-field polarization phenomena are fully preserved, also within the Darwin
  diffraction plateau, thus enabling the ray tracing of crystal-based phase
  retarders.

* *Materials*. The material properties are incorporated using :class:`three
  different tabulations <xrt.backends.raycing.materials.Material>` of the
  scattering factors, with differently wide and differently dense energy
  meshes. Refractive index and absorption coefficient are calculated from the
  scattering factors. Two-surface bodies, such as plates or refractive lenses,
  are treated with both refraction and absorption.

* *Multiple reflections*. xrt can trace multiple reflections in a single
  optical element. This is useful, for example in 'whispering gallery' optics
  or in Montel or Wolter mirrors.

* *Non-sequential optics*. xrt can trace non-sequential optics where different
  parts of the incoming beam meet different surfaces. Examples of such optics
  are :ref:`poly-capillaries<polycapillary>` and Wolter mirrors.

* *Singular optics*. xrt correctly propagates :ref:`vortex beams
  <test-Laguerre-Gaussian>`, which can be used for studying the creation of
  vortex beams by transmissive or reflective optics.

* *Global coordinate system*. The optical elements are positioned in a global
  coordinate system. This is convenient for modeling a real synchrotron
  beamline. The coordinates in this system can be directly taken from a CAD
  library. The optical surfaces are defined in their local systems for the
  user's convenience.

* *Beam categories*. xrt discriminates rays by several categories: `good`,
  `out`, `over` and `dead`. This distinction simplifies the adjustment of
  entrance and exit slits. An alarm is triggered if the fraction of dead rays
  exceeds a specified level.

* *Portability*. xrt runs on Windows and Unix-like platforms, wherever you can
  run python.

* *Examples*. xrt comes with many examples; see the galleries, the links are at
  the top bar.

Python 2 and 3
--------------

The code can run in both Python branches without any modification.

Dependencies
------------

numpy, scipy and matplotlib are required. If you use OpenCL for calculations on
GPU or CPU, you need AMD/NVIDIA drivers, ``Intel CPU only OpenCL runtime``
(these are search key words), pytools and pyopencl. PyQt4 or PyQt5 are needed
for xrtQook. Spyder (as library of Spyder IDE) is highly recommended for nicer
view of xrtQook. OpenGL is required for xrtGlow.

See :ref:`detailed instructions for installing dependencies <instructions>`.

Get xrt
-------

xrt is available as source distribution from `pypi.python.org
<https://pypi.python.org/pypi/xrt>`_ and from `GitHub
<https://github.com/kklmn/xrt>`_. The distribution archive also includes tests,
and examples. The complete documentation is available online at
`Read the Docs <http://xrt.readthedocs.io>`_ and offline as zip file at
`GitHub <https://github.com/kklmn/xrt-docs>`_.

Installation
------------

Unzip the .zip file into a suitable directory and use
`sys.path.append(path-to-xrt)` in your script. You can also install xrt to the
standard location by running ``python setup.py install`` from the directory
where you have unzipped the archive, which is less convenient if you try
different versions of xrt and/or different versions of python. Note that
python-64-bit is by ~20% faster than the 32-bit version (tested with
WinPython). Also consider :ref:`Speed tests <tests>` for making choice between
Windows and Linux and between Python 2 and Python 3.

Citing xrt
----------

Please cite xrt as:
`K. Klementiev and R. Chernikov, "Powerful scriptable ray tracing package xrt",
Proc. SPIE 9209, Advances in Computational Methods for X-Ray Optics III,
92090A; doi:10.1117/12.2061400 <http://dx.doi.org/10.1117/12.2061400>`_.

Acknowledgments
---------------

Josep Nicolás and Jordi Juanhuix (synchrotron Alba) are acknowledged for
discussion and for their Matlab codes used as examples at early stages of the
project. Summer students of DESY Andrew Geondzhian and Victoria Kabanova are
acknowledged for their help in coding the classes of synchrotron sources. Rami
Sankari and Alexei Preobrajenski (MAX IV Laboratory) are thanked for
discussion, testing and comparing with external codes. Hasan Yavaş, Jozef
Bednarčik, Dmitri Novikov and Alexey Zozulya (DESY Photon Science) are
acknowledged for supplied cases. Hamed Tarawneh (MAX IV Laboratory) has
initiated the custom field undulator project and supplied tables of magnetic
field. Bernard Kozioziemski (LLNL) has pointed to the importance of considering
the *total* absorption cross-section (not just photoelectric absorption) in the
cases of light materials at high energy. Emilio Heredia (CLS) is thanked for
valuable suggestions on GUI.

"""

# ========Convention: note the difference from PEP8 for variables!=============
# Naming:
#   * classes MixedUpperCase
#   * varables lowerUpper _or_ lower
#   * functions and methods underscore_separated _or_ lower
# =============================================================================

from .version import __versioninfo__, __version__, __date__
__module__ = "xrt"
__author__ = \
    "Konstantin Klementiev (MAX IV Laboratory), " +\
    "Roman Chernikov (Canadian Light Source)"
__email__ = "konstantin DOT klementiev AT gmail DOT com, " +\
            "rchernikov AT gmail DOT com"
__license__ = "MIT license"

# __all__ = ['plotter', 'runner', 'multipro']
