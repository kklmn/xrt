# -*- coding: utf-8 -*-
"""
Package :mod:`~xrt.backends.raycing` provides the internal backend of xrt. It
defines beam sources in the module :mod:`~xrt.backends.raycing.sources`,
rectangular and round apertures in :mod:`~xrt.backends.raycing.apertures`,
optical elements in :mod:`~xrt.backends.raycing.oes`, material properties
(essentially reflectivity, transmittivity and absorption coefficient) for
interfaces and crystals in :mod:`~xrt.backends.raycing.materials` and screens
in :mod:`~xrt.backends.raycing.screens`.

Coordinate systems
------------------

.. imagezoom:: _images/axes.png
   :align: right

The following coordinate systems are considered (always right-handed):

1) *The global coordinate system*. It is arbitrary (user-defined) with one
   requirement driven by code simplification: Z-axis is vertical. For example,
   the system origin of Alba synchrotron is in the center of the ring at the
   ground level with Y-axis northward, Z upright and the units in mm.

   .. note::
       The positions of all optical elements, sources, screens etc. are given
       in the global coordinate system. This feature simplifies the beamline
       alignment when 3D CAD models are available.

2) *The local systems*.

   a) *of the beamline*. The local Y direction (the direction of the source)
      is determined by *azimuth* parameter of
      :class:`~xrt.backends.raycing.BeamLine` -- the angle measured cw from the
      global Y axis. The local beamline Z is also vertical and upward. The
      local beamline X is to the right. At *azimuth* = 0 the global system and
      the local beamline system are parallel to each other. In most of the
      supplied examples the global system and the local beamline system
      coincide.

   b) *of an optical element*. The origin is on the optical surface. Z is
      out-of-surface. At pitch, roll and yaw all zeros the local oe system
      and the local beamline system are parallel to each other.

      .. note::
          Pitch, roll and yaw rotations (correspondingly: Rx, Ry and Rz) are
          defined relative to the local axes of the optical element. The local
          axes rotate together with the optical element!

      .. note::
          The rotations are done in the following default sequence: yaw, roll,
          pitch. It can be changed by the user for any particular optical
          element. Sometimes it is necessary to define misalignment angles in
          addition to the positional angles. Because rotations do not commute,
          an extra set of angles may become unavoidable, which are applied
          after the positional rotations.
          See :class:`~xrt.backends.raycing.oes.OE`.

      The user-supplied functions for the surface height (z) and the normal as
      functions of (x, y) are defined in the local oe system.

   c) *of other beamline elements: sources, apertures, screens*. Z is upward
      and Y is along the beam line. The origin is given by the user. Usually it
      is on the original beam line.

xrt sequentially transforms beams (instances of
:class:`~xrt.backends.raycing.sources.Beam`) -- containers of arrays which hold
beam properties for each ray. Geometrical beam properties such as *x, y, z*
(ray origins) and *a, b, c* (directional cosines) as well as polarization
characteristics depend on the above coordinate systems. Therefore, beams are
usually represented by two different objects: one in the global and one in a
local system.

Units
-----

For the internal calculations, lengths are assumed to be in mm, although for
reflection geometries and simple Bragg cases (thick crystals) this convention
is not used. Angles are unitless (radians). Energy is in eV.

For plotting, the user may select units and conversion factors. The latter are
usually automatically deduced from the units.

Beam categories
---------------

xrt discriminates rays by several categories:

a) ``good``: reflected within the working optical surface;
b) ``out``: reflected outside of the working optical surface, i.e. outside of
   a metal stripe on a mirror;
c) ``over``: propagated over the surface without intersection;
d) ``dead``: arrived below the optical surface and thus absorbed by the OE.

This distinction simplifies the adjustment of entrance and exit slits. The
user supplies `physical` and `optical` limits, where the latter is used to
define the ``out`` category (for rays between `physical` and `optical` limits).
An alarm is triggered if the fraction of dead rays exceeds a specified level.

.. _scriptingRaycing:

Scripting in python
-------------------

The user of :mod:`~xrt.backends.raycing` must do the following:

1) Instantiate class :class:`~xrt.backends.raycing.BeamLine` and fill it with
   sources, optical elements, screens etc.
2) Create a module-level function that returns a dictionary of beams -- the
   instances of :class:`~xrt.backends.raycing.sources.Beam`. Assign this
   function to the module variable `xrt.backends.raycing.run.run_process`.
   The beams should be obtained by the methods shine() of a source, expose() of
   a screen, reflect() or multiple_reflect() of an optical element, propagate()
   of an aperture.
3) Use the keys in this dictionary for creating the plots (instances of
   :class:`~xrt.plotter.XYCPlot`). Note that at the time of instantiation the
   plots are just empty placeholders for the future 2D and 1D histograms.
4) Run :func:`~xrt.runner.run_ray_tracing()` function for the created plots.

Additionally, the user may define a generator that will run a loop of ray
tracing jobs for changing geometry settings (mimics a real scan) or for
different material properties etc. The generator should modify the beamline
elements and output file names of the plots before *yield*. After the *yield*
the plots are ready and the generator may use their fields, e.g. *intensity* or
*dE* or *dy* or others to prepare a scan plot. Typically, this sequence is
contained within a loop; after the loop the user may prepare the final scan
plot using matplotlib functionality. The generator is passed to
:func:`~xrt.runner.run_ray_tracing()` as a parameter.

Consider an example of a generator::

    def energy_scan(beamLine, plots, energies):
        flux = np.zeros_like(energies)
        for ie, e in enumerate(energies):
            print(f'energy {e:.1f} eV, {ie+1} of {len(energies)}')
            beamLine.fixedEnergy = e
            beamLine.source.eMin = e - 0.5  # defines 1 eV energy band
            beamLine.source.eMax = e + 0.5
            for plot in plots:
                plot.saveName = [plot.baseName + f'-{ie}-{e:.1f}eV.png']

            yield
            # now all plots for this scan point are ready
            flux[ie] = plot.flux

        # now the whole scan is complete
        integratedFlux = np.trapz(flux, energies)
        print(f'total flux = {integratedFlux:.3g} ph/s')

        with open("ray_tracing_c.pickle", 'wb') as f:
            pickle.dump([energies, flux, integratedFlux], f)

        plt.plot(energies, flux)
        plt.show()

... and an example of passing this generator to
:func:`~xrt.runner.run_ray_tracing()`::

    def ray_study(nrays, repeats):
        beamLine = build_beamline(nrays)
        plots = define_plots(beamLine)
        energies = np.linspace(11800, 12600, 401)
        xrtr.run_ray_tracing(
            plots, repeats=repeats, beamLine=beamLine,
            generator=energy_scan, generatorArgs=[beamLine, plots, energies])

Find more generators in the supplied examples.
"""

from __future__ import print_function
import types
import sys
# import os
import numpy as np
# from itertools import compress
from itertools import islice
from collections import OrderedDict
from functools import wraps, partial
import re
import copy
import inspect
import uuid
import importlib
import json
import xml.etree.ElementTree as ET
import time
import queue

from matplotlib.colors import hsv_to_rgb
import colorama
if sys.version_info < (3, 1):
    from inspect import getargspec
else:
    from inspect import getfullargspec as getargspec


colorama.init(autoreset=True)

__module__ = "raycing"
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "26 Mar 2016"

_DEBUG_ = True  # If False, exceptions inside the module are ignored
_VERBOSITY_ = 10   # [0-100] Regulates the level of diagnostics printout

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

from .physconsts import SIE0, CH  # analysis:ignore


safe_globals = {
    'np': np,
    '__builtins__': {}  # Disable built-in functions
}

stateGood, stateOut, stateOver = 1, 2, 3

zEps = 1e-12  # mm: target accuracy in z while searching for intersection
misalignmentTolerated = 0.1  # for automatic checking of oe center position
accuracyInPosition = 0.1  # accuracy for positioning of oe
dt = 1e-5  # mm: margin around OE within which the intersection is searched
ds = 0.  # mm: margin used in multiple reflections
nrays = 100000
maxIteration = 100  # max number of iterations while searching for intersection
maxHalfSizeOfOE = 1000.
maxDepthOfOE = 100.
# maxZDeviationAtOE = 100.

# colors of the rays in a 0-10 range (red-violet)
hueGood = 3.
hueOut = 8.
hueOver = 1.6
hueDead = 0.2
hueMin = 0.
hueMax = 10.

targetOpenCL = 'auto'
precisionOpenCL = 'auto'
# targetOpenCL = (0, 0)
# precisionOpenCL = 'float32'

allBeamFields = ('energy', 'x', 'xprime', 'y', 'z', 'zprime', 'xzprime',
                 'a', 'b', 'path', 'phase_shift', 'reflection_number', 'order',
                 'circular_polarization_rate', 'polarization_degree',
                 'polarization_psi',  'ratio_ellipse_axes', 's', 'r',
                 'theta', 'phi', 'incidence_angle',
                 'elevation_d', 'elevation_x', 'elevation_y', 'elevation_z',
                 'Ep_amp', 'Ep_phase', 'Es_amp', 'Es_phase')

orientationArgSet = {'center', 'pitch', 'roll', 'yaw', 'bragg',
                     'braggOffset', 'rotationSequence', 'positionRoll'}

shapeArgSet = {'limPhysX', 'limPhysY', 'limPhysX2', 'limPhysY2', 'opening',
               'R', 'r', 'Rm', 'Rs', 'p', 'q', 'f1', 'f2', 'pAxis',
               'parabolaAxis', 'shape'}

colors = 'BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN',\
    'WHITE', 'RESET'

msg_start = {
        "command": "start"}
msg_stop = {
        "command": "stop"}
msg_exit = {
        "command": "exit"}

allUnitsAng = {'rad': 1.,
               'mrad': 1e-3,
               'urad': 1e-6,
               'deg': np.pi/180.,
               'mdeg': 1e-3*np.pi/180.,
               'arcsec': np.pi/180./3600.}

allUnitsAngStr = {'rad': u'rad',
                  'mrad': u'mrad',
                  'urad': u'µrad',
                  'deg': u'°',
                  'mdeg': u'm°',
                  'arcsec': r'arcsec'}

allUnitsLen = {'angstroem': 1e-7,
               'nm': 1e-6,
               'um': 1e-3,
               'mm': 1.,
               'm': 1e3,
               'km': 1e6}

allUnitsLenStr = {'angstroem': u'Å',
                  'nm': r'nm',
                  'um': u'µm',
                  'mm': r'mm',
                  'm': r'm',
                  'km': r'km'}

allUnitsEnergy = {'meV': 1e-3,
                  'eV': 1,
                  'keV': 1e3,
                  'MeV': 1e6,
                  'GeV': 1e9}

allUnitsEnergyStr = {'meV': 'meV',
                     'eV': 'eV',
                     'keV': 'keV',
                     'MeV': 'MeV',
                     'GeV': 'GeV'}

allUnitsEmittance = {'pmrad': 1e-3,
                     'nmrad': 1}

allUnitsEmittanceStr = {'pmrad': 'pm⋅rad',
                        'nmrad': 'nm⋅rad'}

allUnitsCurrent = {'mA': 1e-3,
                   'A': 1}

allUnitsCurrentStr = {'mA': 'mA',
                      'A': 'A'}

lengthUnitParams = {'center': 'mm',
                    'R': 'mm',
                    'r': 'mm',
                    'Rm': 'mm',
                    'Rs': 'mm',
                    'dx': 'mm',
                    'dy': 'mm',
                    'dz': 'mm',
                    'beta': 'm'}  # WIP


class NamedArrayBase(np.ndarray):
    _names = []

    def __new__(cls, values=None, dtype=float, **kwargs):

        num_elements = len(cls._names)

        if values is None and kwargs:
            values = [kwargs.get(name, 0.0) for name in cls._names]
        elif values is not None:
            if len(values) != num_elements:
                raise ValueError(
                    f'Expected {num_elements} elements, got {len(values)}.')
        else:
            values = np.zeros(num_elements, dtype=dtype)

        try:
            obj = np.asarray(values, dtype=dtype).view(cls)
        except ValueError:
            obj = values
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __getattr__(self, attr):
        if attr in self._names:
            idx = self._names.index(attr)
            return self[idx]
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        if attr in self._names:
            idx = self._names.index(attr)
            self[idx] = value
        else:
            super().__setattr__(attr, value)

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        return not np.array_equal(self, other)

    def __repr__(self):
        components = ', '.join(f'{name}={getattr(self, name)}'
                               for name in self._names)
        return f'{type(self).__name__}({components})'

    def __str__(self):
        return '[' + ', '.join(str(val) for val in self) + ']'


def NamedArrayFactory(names, default_dtype=float):
    class_name = 'NamedArray_' + '_'.join(names)
    cls = type(class_name, (NamedArrayBase,), {
        '_names': names,
        '__module__': __name__  # Make sure it points to the real module
    })

    # Register the class in the module’s namespace so pickle can find it
    sys.modules[__name__].__dict__[class_name] = cls
    return cls


Center = NamedArrayFactory(['x', 'y', 'z'])
Limits = NamedArrayFactory(['lmin', 'lmax'])
Opening = NamedArrayFactory(['left', 'right', 'bottom', 'top'])
Image2D = NamedArrayFactory(['width', 'height'], default_dtype=int)


def center_property():
    def getter(self):
        return self._center if self._centerVal is None else self._centerVal

    def setter(self, center):
        centerInit = copy.deepcopy(center)
        if isinstance(center, str):
            center = [x.strip().lower() for x in center.strip('[]').split(",")]
            tmp = []
            for value in center:
                try:
                    value = float(value)
                except ValueError:
                    pass
                tmp.append(value)

#        if any(['auto' in str(x) for x in center]):
        if any([isinstance(x, str) for x in center]):
            self._centerInit = centerInit
            self._centerVal = None
            self._center = copy.deepcopy(center)
        else:
            self._centerVal = Center(center)

    return property(getter, setter)


def colorPrint(s, fcolor=None, bcolor=None):
    style = getattr(colorama.Fore, fcolor) if fcolor in colors else \
        colorama.Fore.RESET
    style += getattr(colorama.Back, bcolor) if bcolor in colors else \
        colorama.Back.RESET
    print('{0}{1}'.format(style, s))


def is_sequence(arg):
    """Checks whether *arg* is a sequence."""
    result = (not hasattr(arg, "strip") and hasattr(arg, "__getitem__") or
              hasattr(arg, "__iter__"))
    if result:
        try:
            arg[0]
        except IndexError:
            result = False
        if result:
            result = not isinstance(arg, (basestring, unicode))
    return result


def distance_xy(p1, p2):
    """Calculates 2D distance between p1 and p2. p1 and p2 are vectors of
    length >= 2."""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def distance_xyz(p1, p2):
    """Calculates 2D distance between p1 and p2. p1 and p2 are vectors of
    length >= 3."""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5


def rotate_x(y, z, cosangle, sinangle):
    """3D rotaion around *x* (pitch). *y* and *z* are values or arrays.
    Positive rotation is for positive *sinangle*. Returns *yNew, zNew*."""
    return cosangle*y - sinangle*z, sinangle*y + cosangle*z


def rotate_y(x, z, cosangle, sinangle):
    """3D rotaion around *y* (roll). *x* and *z* are values or arrays.
    Positive rotation is for positive *sinangle*. Returns *xNew, zNew*."""
    return cosangle*x + sinangle*z, -sinangle*x + cosangle*z


def rotate_z(x, y, cosangle, sinangle):
    """3D rotaion around *z*. *x* and *y* are values or arrays.
    Positive rotation is for positive *sinangle*. Returns *xNew, yNew*."""
    return cosangle*x - sinangle*y, sinangle*x + cosangle*y


def rotate_beam(beam, indarr=None, rotationSequence='RzRyRx',
                pitch=0, roll=0, yaw=0, skip_xyz=False, skip_abc=False,
                is2ndXtal=False):
    """Rotates the *beam* indexed by *indarr* by the angles *yaw, roll, pitch*
    in the sequence given by *rotationSequence*. A leading '-' symbol of
    *rotationSequence* reverses the sequences.
    """
    angles = {'z': yaw, 'y': roll, 'x': pitch}
    rotates = {'z': rotate_z, 'y': rotate_y, 'x': rotate_x}
    if not skip_xyz:
        coords1 = {'z': beam.x, 'y': beam.x, 'x': beam.y}
        coords2 = {'z': beam.y, 'y': beam.z, 'x': beam.z}
    if not skip_abc:
        vcomps1 = {'z': beam.a, 'y': beam.a, 'x': beam.b}
        vcomps2 = {'z': beam.b, 'y': beam.c, 'x': beam.c}

    if rotationSequence[0] == '-':
        seq = rotationSequence[6] + rotationSequence[4] + rotationSequence[2]
    else:
        seq = rotationSequence[1] + rotationSequence[3] + rotationSequence[5]
    for s in seq:
        angle, rotate = angles[s], rotates[s]
        if not skip_xyz:
            c1, c2 = coords1[s], coords2[s]
        if not skip_abc:
            v1, v2 = vcomps1[s], vcomps2[s]
        if angle != 0:
            cA = np.cos(angle)
            sA = np.sin(angle)
            if indarr is None:
                indarr = slice(None)
            if not skip_xyz:
                c1[indarr], c2[indarr] = rotate(c1[indarr], c2[indarr], cA, sA)
            if not skip_abc:
                v1[indarr], v2[indarr] = rotate(v1[indarr], v2[indarr], cA, sA)


def rotate_xyz(x, y, z, indarr=None, rotationSequence='RzRyRx',
               pitch=0, roll=0, yaw=0):
    """Rotates the arrays *x*, *y* and *z* indexed by *indarr* by the angles
    *yaw, roll, pitch* in the sequence given by *rotationSequence*. A leading
    '-' symbol of *rotationSequence* reverses the sequences.
    """
    angles = {'z': yaw, 'y': roll, 'x': pitch}
    rotates = {'z': rotate_z, 'y': rotate_y, 'x': rotate_x}
    coords1 = {'z': x, 'y': x, 'x': y}
    coords2 = {'z': y, 'y': z, 'x': z}

    if rotationSequence[0] == '-':
        seq = rotationSequence[6] + rotationSequence[4] + rotationSequence[2]
    else:
        seq = rotationSequence[1] + rotationSequence[3] + rotationSequence[5]
    for s in seq:
        angle, rotate = angles[s], rotates[s]
        c1, c2 = coords1[s], coords2[s]
        if angle != 0:
            cA = np.cos(angle)
            sA = np.sin(angle)
            if indarr is None:
                indarr = slice(None)
            c1[indarr], c2[indarr] = rotate(c1[indarr], c2[indarr], cA, sA)
    return x, y, z


def rotate_point(point, rotationSequence='RzRyRx', pitch=0, roll=0, yaw=0):
    """Rotates the *point* (3-sequence) by the angles *yaw, roll, pitch*
    in the sequence given by *rotationSequence*. A leading '-' symbol of
    *rotationSequence* reverses the sequences.
    """
    angles = {'z': yaw, 'y': roll, 'x': pitch}
    rotates = {'z': rotate_z, 'y': rotate_y, 'x': rotate_x}
    ind1 = {'z': 0, 'y': 0, 'x': 1}
    ind2 = {'z': 1, 'y': 2, 'x': 2}
    newp = [coord for coord in point]
    if rotationSequence[0] == '-':
        seq = rotationSequence[6] + rotationSequence[4] + rotationSequence[2]
    else:
        seq = rotationSequence[1] + rotationSequence[3] + rotationSequence[5]
    for s in seq:
        angle, rotate = angles[s], rotates[s]
        if angle != 0:
            cA = np.cos(angle)
            sA = np.sin(angle)
            newp[ind1[s]], newp[ind2[s]] = rotate(
                newp[ind1[s]], newp[ind2[s]], cA, sA)
    return newp


def global_to_virgin_local(bl, beam, lo, center=None, part=None):
    """Transforms *beam* from the global to the virgin (i.e. with pitch, roll
    and yaw all zeros) local system. The resulting local beam is *lo*. If
    *center* is provided, the rotation Rz is about it, otherwise is about the
    origin of *beam*. The beam arrays can be sliced by *part* indexing array.
    *bl* is an instance of :class:`BeamLine`"""
    if part is None:
        part = np.ones(beam.x.shape, dtype=bool)
    if center is None:
        center = [0, 0, 0]
    lo.x[part] = beam.x[part] - center[0]
    lo.y[part] = beam.y[part] - center[1]
    lo.z[part] = beam.z[part] - center[2]
    if isinstance(bl, BeamLine):
        a0, b0 = bl.sinAzimuth, bl.cosAzimuth
        if a0 == 0:
            lo.a[part] = beam.a[part]
            lo.b[part] = beam.b[part]
        else:
            lo.x[part], lo.y[part] = rotate_z(lo.x[part], lo.y[part], b0, a0)
            lo.a[part], lo.b[part] = \
                rotate_z(beam.a[part], beam.b[part], b0, a0)
        lo.c[part] = beam.c[part]  # unchanged
    elif isinstance(bl, (list, tuple)):
        lx, ly, lz = bl
        xyz = lo.x[part], lo.y[part], lo.z[part]
        lo.x[part], lo.y[part], lo.z[part] = (
            sum(c*b for c, b in zip(lx, xyz)),
            sum(c*b for c, b in zip(ly, xyz)),
            sum(c*b for c, b in zip(lz, xyz)))
        abc = beam.a[part], beam.b[part], beam.c[part]
        lo.a[part], lo.b[part], lo.c[part] = (
            sum(c*b for c, b in zip(lx, abc)),
            sum(c*b for c, b in zip(ly, abc)),
            sum(c*b for c, b in zip(lz, abc)))


def virgin_local_to_global(bl, vlb, center=None, part=None,
                           skip_xyz=False, skip_abc=False, is2ndXtal=False):
    """Transforms *vlb* from the virgin (i.e. with pitch, roll and yaw all
    zeros) local to the global system and overwrites the result to *vlb*. If
    *center* is provided, the rotation Rz is about it, otherwise is about the
    origin of *beam*. The beam arrays can be sliced by *part* indexing array.
    *bl* is an instance of :class:`BeamLine`"""
    if part is None:
        part = np.ones(vlb.x.shape, dtype=bool)
    a0, b0 = bl.sinAzimuth, bl.cosAzimuth
    if a0 != 0:
        if not skip_abc:
            vlb.a[part], vlb.b[part] = rotate_z(
                vlb.a[part], vlb.b[part], b0, -a0)
        if not skip_xyz:
            vlb.x[part], vlb.y[part] = rotate_z(
                vlb.x[part], vlb.y[part], b0, -a0)
    if (center is not None) and (not skip_xyz):
        vlb.x[part] += center[0]
        vlb.y[part] += center[1]
        vlb.z[part] += center[2]


def xyz_from_xz(obj, x=None, z=None):
    if isinstance(x, basestring) and isinstance(z, basestring):
        return 'auto'
    bl = obj.bl
    if isinstance(x, (list, tuple, np.ndarray)):
        norm = sum([xc**2 for xc in x])**0.5
        retx = [xc/norm for xc in x]
    else:
        retx = bl.cosAzimuth, -bl.sinAzimuth, 0.

    if isinstance(z, (list, tuple, np.ndarray)):
        norm = sum([zc**2 for zc in z])**0.5
        retz = [zc/norm for zc in z]
    else:
        retz = 0., 0., 1.

    xdotz = np.dot(retx, retz)
    if abs(xdotz) > 1e-8:
        try:
            objName = obj.name
        except AttributeError:
            objName = obj.__class__.__name__
        colorPrint('x and z must be orthogonal, got xz={0:.4e} for {1}'
                   .format(xdotz, objName), 'RED')
    rety = np.cross(retz, retx)
    return [retx, rety, retz]


def check_alarm(self, incoming, beam):
    """Appends an alarm string to the list of beamline alarms if the alarm
    condition is fulfilled."""
    incomingSum = incoming.sum()
    try:
        objName = self.name
    except AttributeError:
        objName = self.__class__.__name__
    if incomingSum > 0:
        badState = beam.state == self.lostNum
        badSum = badState.sum()
        badFlux = (beam.Jss[badState] + beam.Jpp[badState]).sum()
        allFlux = (beam.Jss + beam.Jpp).sum()
        ratio = float(badSum)/incomingSum
        ratioFlux = badFlux / allFlux
        if ratio > self.alarmLevel:
            alarmStr = ('{0}{1} absorbes {2:.2%} of rays or {3:.2%} of flux ' +
                        'at {4:.0%} alarm level!').format(
                'Alarm! ', objName, ratio, ratioFlux, self.alarmLevel)
            self.bl.alarms.append(alarmStr)
    else:
        self.bl.alarms.append('no incident rays to {0}!'.format(objName))


def get_energy(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.E


def get_x(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.x


def get_y(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.y


def get_z(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.z


def get_s(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.s


def get_phi(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.phi


def get_r(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.r


def get_a(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.a


def get_b(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.b


def get_xprime(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.a / beam.b


def get_zprime(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.c / beam.b


def get_xzprime(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return (beam.a**2 + beam.c**2)**0.5 / beam.b


def get_path(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.path


def get_order(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.order if hasattr(beam, 'order') else beam.state


def get_reflection_number(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.nRefl if hasattr(beam, 'nRefl') else beam.state


def get_elevation_d(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.elevationD
# if hasattr(beam, 'elevationD') else np.zeros_like(beam.x)


def get_elevation_x(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.elevationX  # if hasattr(beam, 'elevationX') else beam.x


def get_elevation_y(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.elevationY  # if hasattr(beam, 'elevationY') else beam.y


def get_elevation_z(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.elevationZ  # if hasattr(beam, 'elevationZ') else beam.z


def get_Es_amp(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.abs(beam.Es) if hasattr(beam, 'Es') else np.zeros_like(beam.x)


def get_Ep_amp(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.abs(beam.Ep) if hasattr(beam, 'Ep') else np.zeros_like(beam.x)


def get_Es_phase(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.angle(beam.Es) if hasattr(beam, 'Es') else np.zeros_like(beam.x)
#    return np.arctan2(beam.Es.imag, beam.Es.real)


def get_Ep_phase(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.angle(beam.Ep) if hasattr(beam, 'Ep') else np.zeros_like(beam.x)
#    return np.arctan2(beam.Ep.imag, beam.Ep.real)


def get_polarization_degree(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    II = (beam.Jss + beam.Jpp)
    II[II <= 0] = 1.
    pd = np.sqrt((beam.Jss-beam.Jpp)**2 + 4.*abs(beam.Jsp)**2) / II
    pd[II <= 0] = 0.
    return pd


def get_ratio_ellipse_axes(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    dI2 = (beam.Jss - beam.Jpp)**2
    return 2. * beam.Jsp.imag /\
        (np.sqrt(dI2 + 4*abs(beam.Jsp)**2) + np.sqrt(dI2 + 4*beam.Jsp.real**2))


def get_circular_polarization_rate(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    II = (beam.Jss + beam.Jpp)
    II[II <= 0] = 1.
    cpr = 2. * beam.Jsp.imag / II
    cpr[II <= 0] = 0.
    return cpr


def get_polarization_psi(beam):
    """Angle between the semimajor axis of the polarization ellipse relative to
    the s polarization. Used for retrieving data for x-, y- or c-axis of a
    plot."""
#    return 0.5 * np.arctan2(2.*beam.Jsp.real, beam.Jss-beam.Jpp) * 180 / np.pi
    return 0.5 * np.arctan2(2.*beam.Jsp.real, beam.Jss-beam.Jpp)


def get_phase_shift(beam):  # in units of pi!
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.angle(beam.Jsp) / np.pi


def get_incidence_angle(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.theta if hasattr(beam, 'theta') else np.zeros_like(beam.x)


get_theta = get_incidence_angle


def get_output(plot, beamsReturnedBy_run_process):
    """Used by :mod:`multipro` for creating images of *plot* - instance of
    :class:`XYCPlot`. *beamsReturnedBy_run_process* is a dictionary of
    :class:`Beam` instances returned by user-defined :func:`run_process`.

    :func:`get_output` creates an indexing array corresponding to the requested
    properties of rays in *plot*. It also calculates the number of rays with
    various properties defined in `raycing` backend.
     """
    beam = beamsReturnedBy_run_process[plot.beam]
    if plot.beamState is None:
        beamState = beam.state
    else:
        beamState = beamsReturnedBy_run_process[plot.beamState].state
    nrays = len(beam.x)

    locAlive = (beamState > 0).sum()
    part = np.zeros(nrays, dtype=bool)
    locGood = 0
    locOut = 0
    locOver = 0
    locDead = 0
    for rayFlag in plot.rayFlag:
        locPart = beamState == rayFlag
        if rayFlag == 1:
            locGood = locPart.sum()
        if rayFlag == 2:
            locOut = locPart.sum()
        if rayFlag == 3:
            locOver = locPart.sum()
        if rayFlag < 0:
            locDead += locPart.sum()
        part = part | locPart
    if hasattr(beam, 'accepted'):
        locAccepted = beam.accepted
        locAcceptedE = beam.acceptedE
        locSeeded = beam.seeded
        locSeededI = beam.seededI
    else:
        locAccepted = 0
        locAcceptedE = 0
        locSeeded = 0
        locSeededI = 0

    if hasattr(beam, 'displayAsAbsorbedPower'):
        plot.displayAsAbsorbedPower = True

    if isinstance(plot.xaxis.data, types.FunctionType):
        x = plot.xaxis.data(beam) * plot.xaxis.factor
    elif isinstance(plot.xaxis.data, np.ndarray):
        x = plot.xaxis.data * plot.xaxis.factor
    else:
        raise ValueError('cannot find x data for plot {0}'.format(plot.beam))

    if isinstance(plot.yaxis.data, types.FunctionType):
        y = plot.yaxis.data(beam) * plot.yaxis.factor
    elif isinstance(plot.yaxis.data, np.ndarray):
        y = plot.yaxis.data * plot.yaxis.factor
    else:
        raise ValueError('cannot find y data for plot {0}'.format(plot.beam))

    if plot.caxis.useCategory:
        cData = np.zeros_like(beamState)
        cData[beamState == stateGood] = hueGood
        cData[beamState == stateOut] = hueOut
        cData[beamState == stateOver] = hueOver
        cData[beamState < 0] = hueDead
        intensity = np.ones_like(x)
        flux = intensity
    else:
        if plot.beamC is None:
            beamC = beam
        else:
            beamC = beamsReturnedBy_run_process[plot.beamC]
        if isinstance(plot.caxis.data, types.FunctionType):
            cData = plot.caxis.data(beamC) * plot.caxis.factor
        elif isinstance(plot.caxis.data, np.ndarray):
            cData = plot.caxis.data * plot.caxis.factor
        else:
            raise ValueError('cannot find data for cData!')

        if plot.fluxKind.startswith('power'):
            intensity = ((beam.Jss + beam.Jpp) *
                         beam.E * beam.accepted / beam.seeded * SIE0)
        elif plot.fluxKind.startswith('s'):
            intensity = beam.Jss
        elif plot.fluxKind.startswith('p'):
            intensity = beam.Jpp
        elif plot.fluxKind.startswith('+-45'):
            intensity = 2*beam.Jsp.real
        elif plot.fluxKind.startswith('left-right'):
            intensity = 2*beam.Jsp.imag
        elif plot.fluxKind.startswith('E'):
            if plot.fluxKind.startswith('Es'):
                intensity = beam.Es
                flux = beam.Jss
            elif plot.fluxKind.startswith('Ep'):
                intensity = beam.Ep
                flux = beam.Jpp
            else:
                intensity = beam.Es + beam.Ep
                flux = beam.Jss + beam.Jpp
        else:
            intensity = beam.Jss + beam.Jpp

        if not plot.fluxKind.startswith('E'):
            flux = intensity

    return x[part], y[part], intensity[part], flux[part], cData[part], nrays,\
        locAlive, locGood, locOut, locOver, locDead,\
        locAccepted, locAcceptedE, locSeeded, locSeededI


def auto_units_angle(angle, defaultFactor=1.):
    if isinstance(angle, basestring):
        if len(re.findall("auto", angle)) > 0:
            return angle
        elif len(re.findall("mrad", angle)) > 0:
            return float(angle.split("m")[0].strip())*1e-3
        elif len(re.findall("urad", angle)) > 0:
            return float(angle.split("u")[0].strip())*1e-6
        elif len(re.findall("nrad", angle)) > 0:
            return float(angle.split("n")[0].strip())*1e-9
        elif len(re.findall("rad", angle)) > 0:
            return float(angle.split("r")[0].strip())
        elif len(re.findall("deg", angle)) > 0:
            return np.radians(float(angle.split("d")[0].strip()))
        else:
            print("Could not identify the units")
            return angle
    elif angle is None or isinstance(angle, (list, tuple)):
        return angle
    else:
        return angle * defaultFactor


def append_to_flow(meth, bOut, frame):
    oe = meth.__self__
    if oe.bl is None:
        return
    if oe.bl.flowSource != 'legacy':
        return
    argValues = inspect.getargvalues(frame)
    fdoc = re.findall(r"Returned values:.*", meth.__doc__)
    if fdoc:
        fdoc = fdoc[0].replace("Returned values: ", '').split(',')
        if 'needNewGlobal' in argValues.args[1:]:
            if argValues.locals['needNewGlobal']:
                fdoc.insert(0, 'beamGlobal')

    kwArgsIn = OrderedDict()
    kwArgsOut = OrderedDict()
    for arg in argValues.args[1:]:
        if str(arg) == 'beam':
            kwArgsIn[arg] = id(argValues.locals[arg])
        else:
            kwArgsIn[arg] = argValues.locals[arg]

    for outstr, outbm in zip(list(fdoc), bOut):
        kwArgsOut[outstr.strip()] = id(outbm)

    oe.bl.flow.append([oe.uuid, meth.__func__, kwArgsIn, kwArgsOut])


def append_to_flow_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargsIn):
        methStr = func.__name__

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargsIn)
        bound_args.apply_defaults()

        kwargs = {k: v for k, v in bound_args.arguments.items() if k != 'self'}

        beamIn = None
        if 'beam' in kwargs:
            beamIn = 'beam'
        elif 'accuBeam' in kwargs:
            beamIn = 'accuBeam'

        toGlobal = kwargs.get('toGlobal', True)

        if beamIn and kwargs[beamIn] is not None:
            if hasattr(self, 'bl') and self.bl is not None and\
                    not self.bl.flowSource.endswith('refract'):
                if is_valid_uuid(kwargs[beamIn]):
                    beamId = kwargs[beamIn]
                    kwargs[beamIn] = self.bl.beamsDictU[beamId][
                            'beamGlobal' if toGlobal else 'beamLocal']
                else:
                    beamId = kwargs[beamIn].parentId

                if methStr != 'shine':
                    self.bl.auto_align(self, kwargs[beamIn])
        if hasattr(self, 'get_orientation'):
            self.get_orientation()

        result = func(self, **kwargs)

        if hasattr(self, 'bl') and self.bl is not None:
            if isinstance(result, tuple):
                for a in result:
                    a.parentId = self.uuid
                if len(result) > 2:
                    result[0].parentId = self.uuid
                    if self.bl.flowSource.endswith('refract'):
                        ret_dict = {}
                    else:
                        ret_dict = {'beamGlobal': result[0],
                                    'beamLocal1': result[1],
                                    'beamLocal2': result[2]}
                else:
                    ret_dict = {'beamGlobal': result[0],
                                'beamLocal': result[1]}
            else:
                result.parentId = self.uuid
                if methStr in ['propagate', 'expose']:
                    ret_dict = {'beamLocal': result}
                else:
                    ret_dict = {'beamGlobal' if toGlobal else
                                'beamLocal': result}
            if ret_dict:
                if 'beam' in kwargs:
                    kwargs['beam'] = beamId
                self.bl.flowU[self.uuid] = {methStr: kwargs}
                self.bl.beamsDictU[self.uuid] = ret_dict

        return result
    return wrapper

def is_auto_align_required(oe):
    needAutoAlign = False
    for autoParam in ["_center", "_pitch", "_bragg"]:
        naParam = autoParam.strip("_")
        if hasattr(oe, autoParam) and hasattr(oe, naParam):
            if str(getattr(oe, autoParam)) == str(getattr(oe, naParam)):
                if _VERBOSITY_ > 20:
                    print(autoParam, str(getattr(oe, autoParam)),
                          naParam, str(getattr(oe, naParam)))
                needAutoAlign = True
                if _VERBOSITY_ > 10:
                    try:
                        objName = oe.name
                    except AttributeError:
                        objName = oe.__class__.__name__
                    print("{0}.{1} requires auto-calculation".format(
                        objName, naParam))
    return needAutoAlign


def set_name(elementClass, name):
    if name not in [None, 'None', '']:
        elementClass.name = name
    elif not hasattr(elementClass, 'name'):
        elementClass.name = '{0}{1}'.format(
            elementClass.__class__.__name__,
            elementClass.ordinalNum if hasattr(elementClass, 'ordinalNum')
            else '')


def vec_to_quat(vec, alpha):
    """ Quaternion from vector and angle"""

    return np.insert(vec*np.sin(alpha*0.5), 0, np.cos(alpha*0.5))


def multiply_quats(qf, qt):
    """Multiplication of quaternions"""

    return [qf[0]*qt[0]-qf[1]*qt[1]-qf[2]*qt[2]-qf[3]*qt[3],
            qf[0]*qt[1]+qf[1]*qt[0]+qf[2]*qt[3]-qf[3]*qt[2],
            qf[0]*qt[2]-qf[1]*qt[3]+qf[2]*qt[0]+qf[3]*qt[1],
            qf[0]*qt[3]+qf[1]*qt[2]-qf[2]*qt[1]+qf[3]*qt[0]]


def quat_vec_rotate(vec, q):
    """Rotate vector by a quaternion"""

    qn = np.copy(q)
    qn[1:] *= -1
    return multiply_quats(multiply_quats(
        q, vec_to_quat(vec, np.pi*0.25)), qn)[1:]


def get_init_val(value):
    if str(value) == 'round':
        return str(value)

    if "," in str(value):  # mixed list.
        s = str(value).replace(" ", "").replace("(", "[").replace(")", "]")
        if s.startswith('[['):  # nested list
            try:
                v = eval(s, safe_globals)
            except (NameError, SyntaxError):
                v = s
            return v

        paravalue = str(value).strip('[]() ')
        listvalue = []
        for c in paravalue.split(','):
            c_strip = str(c).strip()
            try:
                if not c_strip:
                    continue
                v = eval(c_strip, safe_globals)
            except (NameError, SyntaxError):
                v = c_strip
            listvalue.append(v)
        return listvalue

    try:
        return eval(str(value), safe_globals)
    except (NameError, SyntaxError):  # Intentionally string
        return str(value)


def get_params(objStr):  # Returns a collection of default parameters
    uArgs = OrderedDict()
    args = []
    argVals = []
#    objStr = "{0}.{1}".format(oeObj.__module__, type(oeObj).__name__)
    components = objStr.split('.')
    module_path = '.'.join(components[:-1])
    class_name = components[-1]
    moduleObj = importlib.import_module(module_path)
#    print("get_params for", class_name)
    try:
        objRef = getattr(moduleObj, class_name)
    except:  # TODO: remove if works correctly
        raise

    isMethod = False
    if hasattr(objRef, 'hiddenParams'):
        hpList = objRef.hiddenParams
    else:
        hpList = []

    if inspect.isclass(objRef):
        for parent in (inspect.getmro(objRef))[:-1]:
            for namef, objf in inspect.getmembers(parent):
                if inspect.ismethod(objf) or inspect.isfunction(objf):
                    argSpec = getargspec(objf)
                    if namef == "__init__":
                        argnames = []
                        argdefaults = []
                        if argSpec[3] is not None:
                            argnames = argSpec[0][1:]
                            argdefaults = argSpec[3]
                        elif hasattr(argSpec, 'kwonlydefaults') and\
                                argSpec.kwonlydefaults:
                            argnames = argSpec.kwonlydefaults.keys()
                            argdefaults = argSpec.kwonlydefaults.values()
                        for arg, argVal in zip(argnames, argdefaults):
                            if arg == 'bl':
                                argVal = None
                            if arg not in args and arg not in hpList:
                                uArgs[arg] = argVal
                    if namef == "__init__" or namef.endswith("pop_kwargs"):
                        kwa = re.findall(r"(?<=kwargs\.pop).*?\)",
                                         inspect.getsource(objf),
                                         re.S)
                        if len(kwa) > 0:
                            kwa = [re.split(
                                ",", kwline.strip("\n ()"),
                                maxsplit=1) for kwline in kwa]
                            for kwline in kwa:
                                arg = kwline[0].strip("\' ")
                                if len(kwline) > 1:
                                    argVal = kwline[1].strip("\' ")
                                else:
                                    argVal = "None"
                                if arg not in args and arg not in hpList:
                                    uArgs[arg] = get_init_val(argVal)
                    attr = 'varkw' if hasattr(argSpec, 'varkw') else \
                        'keywords'
                    if namef == "__init__" and\
                            str(argSpec.varargs) == 'None' and\
                            str(getattr(argSpec, attr)) == 'None':
                        break  # To prevent the parent class __init__
            else:
                continue
            break
    elif inspect.ismethod(objRef) or inspect.isfunction(objRef):
        argList = getargspec(objRef)
        if argList[3] is not None:
            if objRef.__name__ == 'run_ray_tracing':
                uArgs = OrderedDict(zip(argList[0], argList[3]))
            else:
                isMethod = True
                uArgs = OrderedDict(zip(argList[0][1:], argList[3]))

    if hasattr(moduleObj, 'allArguments') and not isMethod:
        for argName in moduleObj.allArguments:
            if str(argName) in uArgs.keys():
                args.append(argName)
                argVals.append(uArgs[argName])
    else:
        args = list(uArgs.keys())
        argVals = list(uArgs.values())

    return zip(args, argVals)


def parametrize(value):
    value = get_init_val(value)
    if isinstance(value, tuple):
        value = list(value)
    return value


def create_paramdict_oe(paramDictStr, defArgs, beamLine=None):
    kwargs = OrderedDict()

    for paraname, paravalue in paramDictStr.items():
        if not isinstance(paravalue, str):
            paravalue = str(paravalue)  # TODO: temporary workaround
        if (paraname in defArgs and paravalue != str(defArgs[paraname])) or\
                paraname in ['bl', 'uuid']:

            if paraname == 'center':
                paravalue = paravalue.strip('[]() ')
                paravalue =\
                    [get_init_val(c.strip())
                     for c in str.split(
                     paravalue, ',')]
            elif paraname.startswith('limPhys'):
                paravalue = paravalue.strip('[]() ')
                paravalue =\
                    [get_init_val(c.strip())
                     for c in str.split(
                     paravalue, ',')]
            elif paraname.startswith('material'):
                if str(paravalue) in beamLine.matnamesToUUIDs:
                    paravalue = beamLine.matnamesToUUIDs[paravalue]
#                if is_valid_uuid(paravalue):
#                    paravalue = beamLine.materialsDict[paravalue]
#                elif paravalue in beamLine.matnamesToUUIDs:
#                    paravalue = beamLine.materialsDict.get(
#                            beamLine.matnamesToUUIDs[paravalue])
            elif paraname == 'bl':
                paravalue = beamLine
            else:
                paravalue = parametrize(paravalue)
            kwargs[paraname] = paravalue

    return kwargs


def create_paramdict_mat(paramDictStr, defArgs, bl=None):
    kwargs = OrderedDict()

    for paraname, paravalue in paramDictStr.items():
        if (paraname in defArgs and paravalue != str(defArgs[paraname])) or\
                paravalue == 'bl':
            if paraname.lower() in ['tlayer', 'blayer', 'coating',
                                    'substrate']:
                if str(paravalue) in bl.matnamesToUUIDs:
                    paravalue = bl.matnamesToUUIDs[paravalue]
#                if is_valid_uuid(paravalue):
#                    paravalue = bl.materialsDict[paravalue]
#                elif paravalue in bl.matnamesToUUIDs:
#                    paravalue = bl.materialsDict.get(
#                            bl.matnamesToUUIDs[paravalue])
            else:
                paravalue = parametrize(paravalue)
            kwargs[paraname] = paravalue
    return kwargs


def get_obj_str(obj):
    return "{0}.{1}".format(obj.__module__, type(obj).__name__)


def get_init_kwargs(oeObj, compact=True, needRevG=False, blname=None,
                    resolveAuto=True):

    if needRevG:
        globRev = {str(v): k for k, v in globals().items()}

    defArgs = dict(get_params(get_obj_str(oeObj)))

    initArgs = {}
    for arg, val in defArgs.items():
        try:
            if hasattr(oeObj, arg):
                if arg == 'data':
                    continue
                if hasattr(oeObj, f'_{arg}Init') and not resolveAuto:
                    realval = getattr(oeObj, f'_{arg}Init')
                    print(oeObj.name, f'_{arg}Init', realval)
                else:
                    realval = getattr(oeObj, arg)

                if arg == 'bl':
                    realval = blname
                if arg == 'elements':
                    realval = [x.name for x in realval]

                if str(arg).lower().startswith(
                        ('material', 'coating', 'substrate', 'tlay', 'blay')):
                    if hasattr(realval, 'uuid'):
                        realval = realval.uuid
                    elif needRevG:
                        realval = globRev[str(realval)]
                    else:
                        print("Do something with your material")
                        raise
                if realval != val:
                    defArgs[arg] = str(realval)
                    if compact:
                        initArgs[arg] = str(realval)
                else:
                    defArgs[arg] = str(val)
            else:
                defArgs[arg] = str(val)

        except RuntimeError:  # Unclear error on plot init
            pass

    return initArgs if compact else defArgs


def is_valid_uuid(uuid_string):
    try:
        _ = uuid.UUID(str(uuid_string))
        return True
    except ValueError:
        return False


def run_process_from_file(beamLine):
    outDict = {}

    for oeid, meth in beamLine.flowU.items():
        oe = beamLine.oesDict[oeid][0]
        for func, fkwargs in meth.items():
            getattr(oe, func)(**fkwargs)
    for beamName, beamTag in beamLine.beamNamesDict.items():
        outDict[beamName] = beamLine.beamsDictU[beamTag[0]][beamTag[1]]

    return outDict


def build_hist(beam, limits=None, isScreen=False, shape=[256, 256],
               cDataFunc=None, cLimits=None):
    """This is a simplified standalone implementation of
    multipro.do_hist2d()
    cData is one of get_NNN methods or None. In the latter case the function
    returns only intensity histogram
    """

    good = (beam.state == 1) | (beam.state == 2)
    if isScreen:
        x, y, z = beam.x[good], beam.z[good], beam.y[good]
    else:
        x, y, z = beam.x[good], beam.y[good], beam.z[good]
    goodlen = len(beam.x[good])
    hist2dRGB = None
    hist2d = np.zeros((shape[1], shape[0]), dtype=np.float64)

    if limits is None and goodlen > 0:
        limits = np.array([[np.min(x), np.max(x)],
                           [np.min(y), np.max(y)],
                           [np.min(z), np.max(z)]])

    if goodlen > 0:
        beamLimits = [limits[1], limits[0]] or None
        flux = beam.Jss[good]+beam.Jpp[good]
        hist2d, yedges, xedges = np.histogram2d(
            y, x, bins=[shape[1], shape[0]], range=beamLimits, weights=flux)

    if cDataFunc is not None:
        hist2dRGB = np.zeros((shape[1], shape[0], 3), dtype=np.float64)
        cData = cDataFunc(beam)[good]
        if cLimits is None:
            colorMin, colorMax = np.min(cData), np.max(cData)
        else:
            colorMin, colorMax = cLimits[0], cLimits[-1]
        cData01 = ((cData - colorMin) * 0.85 /
                   (colorMax - colorMin)).reshape(-1, 1)

        cDataHSV = np.dstack(
            (cData01, np.ones_like(cData01) * 0.85,
             flux.reshape(-1, 1)))
        cDataRGB = (hsv_to_rgb(cDataHSV)).reshape(-1, 3)

        hist2dRGB = np.zeros((shape[0], shape[1], 3), dtype=np.float64)
        hist2d = None
        if len(beam.x[good]) > 0:
            for i in range(3):  # over RGB components
                hist2dRGB[:, :, i], yedges, xedges = np.histogram2d(
                    y, x, bins=shape, range=beamLimits,
                    weights=cDataRGB[:, i])

        hist2dRGB /= np.max(hist2dRGB)
        hist2dRGB = np.uint8(hist2dRGB*255)

    return hist2d, hist2dRGB, limits


def propagationProcess(q_in, q_out):

    handler = MessageHandler()
    repeats = 0
    while True:
        try:
            message = q_in.get_nowait()
#            print("MH", message)
            handler.process_message(message)
        except queue.Empty:
            pass
#            time.sleep(0.1)
        if handler.exit:
            break

        if handler.stop:
            time.sleep(0.1)
#            continue
        elif handler.needUpdate:
            # TODO: run propagation downstream of the updated element
            started = True if handler.startEl is None else False

            for oeid, meth in handler.bl.flowU.items():
                if not started:  # Skip until the modified element
                    if handler.startEl == oeid:
                        started = True
                    else:
                        continue
                oe = handler.bl.oesDict[oeid][0]

                for func, fkwargs in meth.items():
                    try:
                        getattr(oe, func)(**fkwargs)
                    except Exception as e:
#                        raise
                        print(e)
                        continue

                    for autoAttr in ['pitch', 'bragg', 'center']:
                        if (hasattr(oe, f'_{autoAttr}') and hasattr(
                                oe, f'_{autoAttr}Val')):
                            if getattr(oe, f'_{autoAttr}') != getattr(
                                    oe, f'_{autoAttr}Val'):
                                msg_autopos_update = {
                                        'pos_attr': autoAttr,
                                        'pos_value': getattr(oe, autoAttr),
                                        'sender_name': oe.name,
                                        'sender_id': oe.uuid,
                                        'status': 0}
                                q_out.put(msg_autopos_update)

                    for autoAttr in ['footprint']:
                        if (hasattr(oe, autoAttr) and len(
                                getattr(oe, autoAttr)) > 0):
                            msg_autopos_update = {
                                    'pos_attr': autoAttr,
                                    'pos_value': getattr(oe, autoAttr),
                                    'sender_name': oe.name,
                                    'sender_id': oe.uuid,
                                    'status': 0}
                            q_out.put(msg_autopos_update)

                    msg_beam = {'beam': handler.bl.beamsDictU[oe.uuid],
                                'sender_name': oe.name,
                                'sender_id': oe.uuid,
                                'status': 0}
                    q_out.put(msg_beam)
                    # TODO: histDict
                    if hasattr(oe, 'expose') and hasattr(oe, 'image'):
                        msg_hist = {'histogram': oe.image,
                                    'sender_name': oe.name,
                                    'sender_id': oeid,
                                    'status': 0}
                        q_out.put(msg_hist)
            handler.bl.forceAlign = False
            q_out.put({"status": 0, "repeat": repeats})
            handler.needUpdate = False
            handler.startEl = None
            time.sleep(0.1)

#            handler.stop = True
            repeats += 1
        else:
            time.sleep(0.1)


def to_valid_var_name(name, default='unnamed'):
    # Replace invalid characters with underscores
    var_name = re.sub(r'\W|^(?=\d)', '_', name.strip())

    # Ensure the name is not empty or a Python keyword
    if not var_name or not re.match(r'[A-Za-z_]', var_name[0]):
        var_name = f"{default}_{var_name}"

    # Avoid Python reserved keywords
    import keyword
    if keyword.iskeyword(var_name):
        var_name += '_var'

    return var_name


class MessageHandler:
    def __init__(self, bl=None):
        self.bl = bl
        self.stop = True
        self.needUpdate = False
        self.autoUpdate = True
        self.startEl = None
        self.exit = False

    def handle_create(self, message):

        objuuid = message.get("uuid")

        object_type = message.get("object_type")
        kwargs = message.get("kwargs", {})

        if object_type == 'beamline':
            self.bl = BeamLine()
            self.bl.deserialize(kwargs)
            self.bl.flowSource = 'Qook'
        elif object_type == 'oe':
            self.bl.init_oe_from_json(kwargs)
        elif object_type == 'mat':
            self.bl.init_material_from_json(kwargs)

        if self.autoUpdate:
            if object_type != 'mat':
                self.needUpdate = True
                self.startEl = objuuid

    def handle_modify(self, message):
        objuuid = message.get("uuid")
        object_type = message.get("object_type")
        kwargs = message.get("kwargs", {})

        if object_type == 'oe':
            element = self.bl.oesDict.get(objuuid)

            if element is not None:
                for key, value in kwargs.items():
                    args = key.split('.')
                    arg = args[0]
                    if len(args) > 1:
                        field = args[-1]
                        if field == 'energy':
                            if arg == 'bragg':
                                value = [float(value)]
                            else:
                                value = element[0].material.get_Bragg_angle(
                                        float(value))
                        else:
                            arrayValue = getattr(element[0], arg)
                            setattr(arrayValue, field, value)
                            value = arrayValue
                    setattr(element[0], arg, value)
                    if arg.lower().startswith('center'):
                        self.bl.sort_flow()

                if self.autoUpdate:
                    self.needUpdate = True
                    if len(kwargs) == 1 and 'name' in kwargs:
                        self.needUpdate = False
                if hasattr(element[0], 'propagate'):
                    kwargs = list(self.bl.flowU[objuuid].values())[0]
                    modifiedEl = kwargs['beam']
                else:
                    modifiedEl = objuuid
                keys = list(self.bl.flowU.keys())
                if self.startEl is None:
                    self.startEl = modifiedEl
                elif keys.index(modifiedEl) < keys.index(self.startEl):
                    self.startEl = modifiedEl
        elif object_type == 'mat':
            # element = self.bl.materialsDict.get(objuuid)
            # We reinstantiate the material object instead of updating. Single-
            # property update not supported yet for materials.
            # object will use the same uuid
            if objuuid in self.bl.materialsDict:
                del self.bl.materialsDict[objuuid]
            self.bl.init_material_from_json(objuuid, kwargs)

            self.startEl = None
            for oeid, oeLine in self.bl.oesDict.items():
                oeObj = oeLine[0]
                for prop in ["_material", "_material2"]:
                    try:
                        matProp = getattr(oeObj, prop)
                        if matProp == objuuid:
                            self.startEl = oeid
                            break
                    except AttributeError:
                        pass

            if self.autoUpdate and self.startEl is not None:
                self.needUpdate = True

    def handle_flow(self, message):
        oeuuid = message.get('uuid')
        kwargs = message.get('kwargs')
#        print("handle_flow", message)
        self.bl.update_flow_from_json(oeuuid, kwargs)
        self.bl.sort_flow()
        if self.autoUpdate:
            self.needUpdate = True
            self.startEl = oeuuid

    def handle_delete(self, message):
        objuuid = message.get("uuid")
        object_type = message.get("object_type")
        if object_type == "oe":
            self.bl.delete_oe_by_id(objuuid)
            if self.autoUpdate:
                self.needUpdate = True
                self.startEl = objuuid
        elif object_type == "mat":
            self.startEl = None
            for oeid, oeLine in self.bl.oesDict.items():
                oeObj = oeLine[0]
                for prop in ["_material", "_material2"]:
                    try:
                        matProp = getattr(oeObj, prop)
                        if matProp == objuuid:
                            self.startEl = oeid
                            break
                    except AttributeError:
                        pass
            self.bl.delete_mat_by_id(objuuid)

            if self.autoUpdate and self.startEl is not None:
                self.needUpdate = True

    def handle_start(self, message):
        print("Starting processing loop.")
        self.stop = False
        self.needUpdate = True

    def handle_exit(self, message):
        print("Exiting.")
        self.exit = True

    def handle_run_once(self, message):
        print("Starting processing loop.")
        self.needUpdate = True
        self.startEl = None

    def handle_auto_update(self, message):
#         print("Starting processing loop.")
        print("modifying auto-update.", message)
        kwargs = message.get('kwargs')
        if kwargs is not None:
            auto_update = kwargs.get('value')

        if bool(auto_update):
            self.needUpdate = True
            self.startEl = None

        self.autoUpdate = bool(auto_update)

    def handle_stop(self, message):
        print("Stopping processing loop.")
        self.stop = True

    def process_message(self, message):
        # Build a dispatch dictionary mapping commands to methods.
        command_handlers = {
            "create": self.handle_create,
            "modify": self.handle_modify,
            "delete": self.handle_delete,
            "start": self.handle_start,
            "stop": self.handle_stop,
            "exit": self.handle_exit,
            "flow": self.handle_flow,
            "run_once": self.handle_run_once,
            "auto_update": self.handle_auto_update,
        }

        command = message.get("command")
        handler = command_handlers.get(command)
        if handler:
            # print(message)
            handler(message)
        else:
            print(f"Unknown command: {command}")


#class DynamicBeamline:
#    """Placeholder for a headless dynamic beamline"""
#    def __init__(self, bl, epicsPrefix=None):
#
#        self.epicsPrefix = epicsPrefix
#
#        self.beamline.deserialize(beamLayout)
#        self.input_queue = Queue()
#        self.output_queue = Queue()
#
#        self.calc_process = Process(
#                target=propagationProcess,
#                args=(self.input_queue, self.output_queue))
#        self.calc_process.start()
#        msg_init_bl = {
#                "command": "create",
#                "object_type": "beamline",
#                "kwargs": beamLayout
#                }
#        self.input_queue.put(msg_init_bl)
#        self.loopRunning = True
#
#        self.timer = qt.QTimer()
#        self.timer.timeout.connect(
#            partial(self.check_progress, self.output_queue))
#        self.timer.start(10)  # Adjust the interval as needed
#        if self.epicsPrefix is not None and self.renderingMode == 'dynamic':
#            try:
#                os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
#                os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
#                self.epicsInterface = EpicsBeamline(
#                        bl=self.beamline,
#                        prefix=epicsPrefix,
#                        callback=self.update_beamline_async)
##                self.build_epics_device(epicsPrefix, softioc, builder,
##                                        asyncio_dispatcher)
#            except ImportError:
#                print("pythonSoftIOC not installed")
#                self.epicsPrefix = None
#
#    async def update_beamline_async(self, oeid, argName, argValue):
#        self.update_beamline(oeid, {argName: argValue})
#
#    def update_beamline(self, oeid, kwargs):
#        for argName, argValue in kwargs.items():
#            if oeid is None:
#                if self.epicsPrefix is not None:
#                    if argName == 'Acquire':
#                        self.epicsInterface.pv_records['AcquireStatus'].set(1)
#                        if str(argValue) == '1':
#                            if hasattr(self, 'input_queue'):
#                                self.input_queue.put({
#                                            "command": "run_once",
#                                            "object_type": "beamline"
#                                            })
#                    elif argName == 'AutoUpdate':
#                        if hasattr(self, 'input_queue'):
#                            self.input_queue.put({
#                                        "command": "auto_update",
#                                        "object_type": "beamline",
#                                        "kwargs": {"value": int(argValue)}
#                                        })
#                return
#
#            oe = self.beamline.oesDict[oeid][0]
#
#            args = argName.split('.')
#            arg = args[0]
#            if len(args) > 1:
#                field = args[-1]
#                if field == 'energy':
#                    if arg == 'bragg':
#                        argValue = [float(argValue)]
#                    else:
#                        argValue = oe.material.get_Bragg_angle(float(argValue))
#                else:
#                    arrayValue = getattr(oe, arg)
#                    setattr(arrayValue, field, argValue)
#                    argValue = arrayValue
#
#            # updating local beamline tree
#            setattr(oe, arg, argValue)
#            if arg in orientationArgSet:
#                self.meshDict[oeid].update_transformation_matrix()
#            elif arg in shapeArgSet:
#                self.needMeshUpdate = oeid
#
#            # updating the beamline model in the runner
#        if self.epicsPrefix is not None:
#            self.epicsInterface.pv_records['AcquireStatus'].set(1)
#        message = {"command": "modify",
#                   "object_type": "beamline",
#                   "uuid": oeid,
#                   "kwargs": kwargs.copy()
##                        "kwargs": {arg: argValue.tolist() if isinstance(
##                                argValue, np.ndarray) else argValue}
#                        }
#        if hasattr(self, 'input_queue'):
#            self.input_queue.put(message)
#
#    def check_progress(self, progress_queue):
##        progress = None
#        while not progress_queue.empty():
#            msg = progress_queue.get()
#            if 'beam' in msg:
##                print(msg['sender_name'], msg['sender_id'], msg['beam'])
#                for beamKey, beam in msg['beam'].items():
#                    self.update_beam_footprint(beam, (msg['sender_id'],
#                                                      beamKey))
#                    self.beamline.beamsDictU[msg['sender_id']][beamKey] = beam
#            elif 'histogram' in msg and self.epicsPrefix is not None:
#                histPvName = f'{to_valid_var_name(msg["sender_name"])}:image'
#                if histPvName in self.epicsInterface.pv_records:
#                    imgHist = np.flipud(msg['histogram'])  # Appears flipped
#                    self.epicsInterface.pv_records[histPvName].set(
#                            imgHist.flatten())
#            elif 'repeat' in msg:
#                print("Total repeats:", msg['repeat'])
#                if self.epicsPrefix is not None:
#                    self.epicsInterface.pv_records['AcquireStatus'].set(0)
#                self.glDraw()
#
#    def close_calc_process(self):
#        if hasattr(self, 'calc_process') and\
#                self.calc_process is not None:
#            self.input_queue.put(msg_exit)
#            self.calc_process.join(timeout=1)
#            if self.calc_process.is_alive():
#                self.calc_process.terminate()
#                self.calc_process.join()


class EpicsDevice:
    def __init__(self, bl, prefix, callback):

        self.bl = bl
        self.epicsPrefix = prefix
        self.pv_map = {}
        self.dbl = set()

        try:
            from softioc import softioc, builder, asyncio_dispatcher
        except ImportError:
            print("Missing softioc dependencies")
            return
        # Create an asyncio dispatcher, the event loop is now running
        self.dispatcher = asyncio_dispatcher.AsyncioDispatcher()

        # Set the record prefix
        builder.SetDeviceName(prefix)
        pv_records = {}
        pvFields = {'name'} | orientationArgSet | shapeArgSet

        pv_records['Acquire'] = builder.boolOut(
            'Acquire', ZNAM=0, ONAM=1,
            initial_value=0, always_update=True,
            on_update=partial(callback, None, 'Acquire'))

        pv_records['AcquireStatus'] = builder.boolIn(
            'AcquireStatus', ZNAM=0, ONAM=1,
            initial_value=0)

        pv_records['AutoUpdate'] = builder.boolOut(
            'AutoUpdate', ZNAM=0, ONAM=1,
            initial_value=1, always_update=True,
            on_update=partial(callback, None, 'AutoUpdate'))

        for oeid, oeline in bl.oesDict.items():
            oeObj = oeline[0]
            oename = to_valid_var_name(oeObj.name)
            self.pv_map[oeid] = {}

            if hasattr(oeObj, 'material') and oeObj.material is not None:
                if hasattr(oeObj.material, 'get_Bragg_angle'):
                    if hasattr(oeObj, 'bragg'):
                        e_field = 'bragg.energy'
                        initial_e = np.abs(CH / (
                                2*oeObj.material.d*np.sin(
                                        oeObj.bragg-oeObj.braggOffset)))
                    else:
                        e_field = 'pitch.energy'
                        initial_e = np.abs(CH / (
                                2*oeObj.material.d*np.sin(oeObj.pitch)))
                    pvname = f'{oename}:ENERGY'
                    pv_records[pvname] = builder.aOut(
                            pvname,
                            initial_value=initial_e,
                            always_update=True,
                            on_update=partial(callback, oeid, e_field))
                    self.pv_map[oeid]['bragg'] = pv_records[pvname]

            if hasattr(oeObj, 'expose') and oeObj.limPhysX is not None:
                pvname = f'{oename}:image'
                histShape = getattr(oeObj, 'histShape')
                imageLength = histShape[0]*histShape[1]
                pv_records[pvname] = builder.WaveformIn(
                    pvname,
                    length=imageLength
                    )

                for fIndex, field in enumerate(['width', 'height']):
                    pvname = f'{oename}:histShape:{field}'
                    dimObj = getattr(oeObj, 'histShape')
                    if dimObj is not None:
                        pv_records[pvname] = builder.aOut(
                            pvname,
                            initial_value=dimObj[fIndex],
                            always_update=True,
                            on_update=partial(callback,
                                              oeid, f'histShape.{field}'))
                        self.pv_map[oeid][f'histShape.{field}'] =\
                            pv_records[pvname]

            for argName in pvFields:
                if hasattr(oeObj, argName):
                    if argName in ['name', 'rotationSequence']:
                        pvname = f'{oename}:{argName}'
                        pv_records[pvname] = builder.stringOut(
                            pvname,
                            initial_value=str(getattr(oeObj, argName)),
                            always_update=True,
                            on_update=partial(callback, oeid, argName))
                        self.pv_map[oeid][argName] = pv_records[pvname]
                    elif argName in ['center']:
                        for field in ['x', 'y', 'z']:
                            pvname = f'{oename}:{argName}:{field}'
                            pv_records[pvname] = builder.aOut(
                                pvname,
                                initial_value=getattr(oeObj.center, field),
                                always_update=True,
                                on_update=partial(callback, oeid,
                                                  f'{argName}.{field}'))
                            self.pv_map[oeid][f'{argName}.{field}'] =\
                                pv_records[pvname]
                    elif argName in ['limPhysX', 'limPhysY', 'limPhysX2',
                                     'limPhysY2']:  # TODO: startswith?
                        for fIndex, field in enumerate(['lmin', 'lmax']):
                            pvname = f'{oename}:{argName}:{field}'
                            limObj = getattr(oeObj, argName)
                            if limObj is not None:
                                pv_records[pvname] = builder.aOut(
                                    pvname,
                                    initial_value=limObj[fIndex],
                                    always_update=True,
                                    on_update=partial(callback, oeid,
                                                      f'{argName}.{field}'))
                                self.pv_map[oeid][f'{argName}.{field}'] =\
                                    pv_records[pvname]
                    elif argName in ['opening']:
                        for field in oeObj.kind:
                            pvname = f'{oename}:{argName}:{field}'
                            limObj = getattr(oeObj, argName)
                            if limObj is not None:
                                pv_records[pvname] = builder.aOut(
                                    pvname,
                                    initial_value=getattr(limObj, field),
                                    always_update=True,
                                    on_update=partial(callback, oeid,
                                                      f'{argName}.{field}'))
                                self.pv_map[oeid][f'{argName}.{field}'] =\
                                    pv_records[pvname]
                    else:
                        pvname = f'{oename}:{argName}'
                        pv_records[pvname] = builder.aOut(
                            pvname,
                            initial_value=getattr(oeObj, argName),
                            always_update=True,
                            on_update=partial(callback, oeid, argName))
                        self.pv_map[oeid][argName] = pv_records[pvname]

        [print(f'{self.epicsPrefix}:{recName}') for recName in pv_records]
        builder.LoadDatabase()
        softioc.iocInit(self.dispatcher)
        self.pv_records = pv_records
        for key in self.pv_records.keys():
            self.dbl.add(f'{prefix}:{key}')


class AlignmentBeam(object):
    def __init__(self):
        for prop in ['a', 'b', 'c', 'x', 'y', 'z', 'E']:
            setattr(self, prop, np.zeros(2))


class BeamLine(object):
    u"""
    Container class for beamline components. It also defines the beam line
    direction and height."""

    def __init__(self, azimuth=0., height=0., alignE='auto', fileName=None,
                 name='beamLine', description=''):
        u"""
        *azimuth*: float
            Is counted in cw direction from the global Y axis. At
            *azimuth* = 0 the local Y coincides with the global Y.

        *height*: float
            Beamline height in the global system.

        *alignE*: float or 'auto'
            Energy for automatic alignment in [eV]. If 'auto', alignment energy
            is defined as the middle of the Source energy range.
            Plays a role if the *pitch* or *bragg* parameters of the energy
            dispersive optical elements were set to 'auto'.


        """

        self.azimuth = azimuth
#        self.sinAzimuth = np.sin(azimuth)  # a0
#        self.cosAzimuth = np.cos(azimuth)  # b0
        self.height = height
        self.alignE = alignE
        self.sources = []
        self.oes = []
        self.slits = []
        self.screens = []
        self.alarms = []
        self.name = name
        self.description = description
        self.oesDict = OrderedDict()
        self.oenamesToUUIDs = {}  # Reverse lookup for oe names
        self.matnamesToUUIDs = {}  # Reverse lookup for mat names
        self.flow = []
        self.materialsDict = OrderedDict()
        self.beamsDict = OrderedDict()
        self.flowSource = 'legacy'
        self.forceAlign = False
        self.beamsDictU = OrderedDict()
        self.flowU = OrderedDict()
        self.beamNamesDict = {}  # Used in run_process_from_file
        self.beamsRevDict = OrderedDict()
        self.beamsRevDictUsed = {}
        self.blViewer = None
        self.blExplorer = None
        self.statusSignal = None
        self.layoutStr = None
        if fileName:
            if str(fileName).lower().endswith("xml"):
                self.load_from_xml(fileName)
            elif str(fileName).lower().endswith("json"):
                self.load_from_json(fileName)

    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value
        self.sinAzimuth = np.sin(value)
        self.cosAzimuth = np.cos(value)

    def orient_along_global_Y(self, center='auto'):
        if center == 'auto':
            center0 = self.sources[0].center
        a0, b0 = self.sinAzimuth, self.cosAzimuth
        for oe in self.sources + self.oes + self.slits + self.screens:
            newC = [c-c0 for c, c0 in zip(oe.center, center0)]
            newC[0], newC[1] = rotate_z(newC[0], newC[1], b0, a0)
            oe.center = newC
            if hasattr(oe, 'jack1'):
                oe.jack1 = [c-c0 for c, c0 in zip(oe.jack1, center0)]
                oe.jack1[0], oe.jack1[1] = \
                    rotate_z(oe.jack1[0], oe.jack1[1], b0, a0)
            if hasattr(oe, 'jack2'):
                oe.jack2 = [c-c0 for c, c0 in zip(oe.jack2, center0)]
                oe.jack2[0], oe.jack2[1] = \
                    rotate_z(oe.jack2[0], oe.jack2[1], b0, a0)
            if hasattr(oe, 'jack3'):
                oe.jack3 = [c-c0 for c, c0 in zip(oe.jack3, center0)]
                oe.jack3[0], oe.jack3[1] = \
                    rotate_z(oe.jack3[0], oe.jack3[1], b0, a0)

        self.azimuth = 0

    def prepare_flow(self):
        def _warning(v1=None, v2=None):
            if v1 is None or v2 is None:
                addw = ""
            else:
                addw = "\nThis beam has been used for {0} and is attempted"\
                    " for {1}.".format(v1, v2)
            print("Warning: the flow seems corrupt. Make sure each propagation"
                  " method assigns returned beams to local variables." + addw)
        if self.flowSource != 'legacy':
            return
        frame = inspect.currentframe()
        localsDict = frame.f_back.f_locals
        globalsDict = frame.f_back.f_globals
        for objectName, memObject in globalsDict.items():
            if len(re.findall('raycing.materials', str(type(memObject)))) > 0:
                self.materialsDict[objectName] = memObject

        for objectName, memObject in localsDict.items():
            if len(re.findall('sources_beams.Beam', str(type(memObject)))) > 0:
                self.beamsDict[objectName] = memObject
                self.beamsRevDict[id(memObject)] = objectName
            if objectName == 'outDict':
                for odObjectName, odMemObject in memObject.items():
                    self.beamsDict[odObjectName] = odMemObject
                    self.beamsRevDict[id(odMemObject)] = odObjectName
        if self.flow is not None and len(self.beamsRevDict) > 0:
            for segment in self.flow:
                for iseg in [2, 3]:
                    for argName, argVal in segment[iseg].items():
                        if len(re.findall('beam', str(argName))) > 0:
                            if iseg == 3:
                                if argVal in self.beamsRevDictUsed:
                                    _warning(self.beamsRevDictUsed[argVal],
                                             segment[0])
                                self.beamsRevDictUsed[argVal] = segment[0]
                            try:
                                segment[iseg][argName] =\
                                    self.beamsRevDict[argVal]
                            except KeyError:
                                segment[iseg][argName] = 'beamTmp'
                                _warning()
        self.flowSource = 'prepared_to_run'

    def auto_align(self, oe, beam):
        if self.flowSource == 'Qook':
            self.forceAlign = True
        if not (self.forceAlign or is_auto_align_required(oe)):
            return

        autoCenter = [False] * 3
        autoPitch = autoBragg = False
        alignE = self._alignE if hasattr(self, '_alignE') else self.alignE

        if hasattr(oe, '_center'):
            autoCenter = [isinstance(x, str) for x in oe._center]
#            autoCenter = ['auto' in str(x) for x in oe._center]

        if hasattr(oe, '_pitch'):
            try:
                if isinstance(oe._pitch, (list, tuple)):
                    alignE = float(oe._pitch[-1])
                autoPitch = oe._pitch is not None
            except Exception:
                print("Automatic Bragg angle calculation failed.")
                raise

        if hasattr(oe, '_bragg'):
            try:
                if isinstance(oe._bragg, (list, tuple)):
                    alignE = float(oe._bragg[-1])
                autoBragg = True
            except Exception:
                print("Automatic Bragg angle calculation failed.")
                raise

        if any(autoCenter) or autoPitch or autoBragg:
            good = (beam.state == 1) | (beam.state == 2)
            if self.flowSource == 'Qook':
                beam.state[0] = 1
#                beam.E[0] = alignE
            intensity = beam.Jss[good] + beam.Jpp[good]
            totalI = np.sum(intensity)
            inBeam = AlignmentBeam()
            for fieldName in ['x', 'y', 'z', 'a', 'b', 'c']:
                field = getattr(beam, fieldName)
                if totalI == 0:
                    fNorm = 1.
                else:
                    fNorm = np.sum(field[good] * intensity) / totalI
                try:
                    setattr(inBeam, fieldName,
                            np.ones(2) * fNorm)
                    if self.flowSource == 'Qook':
                        field[0] = fNorm
                        setattr(inBeam, fieldName, field)
                except Exception:
                    print("Cannot find direction for automatic alignment.")
                    raise

            dirNorm = np.sqrt(inBeam.a[0]**2 + inBeam.b[0]**2 + inBeam.c[0]**2)
            inBeam.a[0] /= dirNorm
            inBeam.b[0] /= dirNorm
            inBeam.c[0] /= dirNorm

            if self.flowSource == 'Qook':
                beam.a[0] /= dirNorm
                beam.b[0] /= dirNorm
                beam.c[0] /= dirNorm

        if any(autoCenter):
            centerList = copy.deepcopy(oe.center)
            bStartC = np.array([inBeam.x[0], inBeam.y[0], inBeam.z[0]])
            bStartDir = np.array([inBeam.a[0], inBeam.b[0], inBeam.c[0]])
            fixedCoord = np.where(np.invert(np.array(autoCenter)))[0]
            autoCoord = np.where(autoCenter)[0]
            for dim in fixedCoord:
                if np.abs(bStartDir[dim]) > 1e-3:
                    plNorm = np.squeeze(np.identity(3)[dim, :])
                    newCenter = bStartC - (np.dot(
                        bStartC, plNorm) - oe.center[dim]) /\
                        np.dot(bStartDir, plNorm) * bStartDir
                    if np.linalg.norm(newCenter - bStartC) > 0:
                        break
            for dim in autoCoord:
                centerList[dim] = newCenter[dim]
            oe.center = centerList
            if _VERBOSITY_ > 0:
                print(oe.name, "center:", oe.center)

        if autoBragg or autoPitch:
            if self.flowSource == 'Qook':
                inBeam.E[0] = alignE
            try:
                if is_sequence(oe.material):
                    mat = oe.material[oe.curSurface]
                else:
                    mat = oe.material
                if not hasattr(mat, 'get_Bragg_angle'):
                    if autoPitch:
                        oe.pitch = 0
                    elif autoBragg:
                        oe.bragg = 0
                    return
                braggT = mat.get_Bragg_angle(alignE)
                alphaT = 0.
                lauePitch = 0.
                if mat.kind == 'multilayer':
                    braggT += -mat.get_dtheta(alignE)
                else:
                    alphaT = 0 if oe.alpha is None else oe.alpha
                    if mat.geom.startswith('Laue'):
                        lauePitch = 0.5 * np.pi
                    else:
                        braggT += -mat.get_dtheta(alignE, alphaT)

                loBeam = copy.deepcopy(inBeam)  # Beam(copyFrom=inBeam)
                global_to_virgin_local(self, inBeam, loBeam, center=oe.center)
                rotate_beam(loBeam, roll=-(oe.positionRoll + oe.roll),
                            yaw=-oe.yaw, pitch=0)
                theta0 = np.arctan2(-loBeam.c[0], loBeam.b[0])
                th2pitch = np.sqrt(1. - loBeam.a[0]**2)
                targetPitch = np.arcsin(np.sin(braggT) / th2pitch) - theta0
                targetPitch += alphaT + lauePitch
                if autoBragg:
                    if autoPitch:
                        oe.pitch = 0
                    oe.bragg = targetPitch - oe.pitch
                    if _VERBOSITY_ > 0:
                        print("{0}: Bragg={1} at E={2}".format(
                                oe.name, oe.bragg, alignE))
                else:  # autoPitch
                    oe.pitch = targetPitch
                    if _VERBOSITY_ > 0:
                        print(oe.name, "pitch:", oe.pitch)
            except Exception as e:
                if _DEBUG_:
                    raise e
                else:
                    pass

    def propagate_flow(self, startFrom=0, signal=None):
        if self.oesDict is None or self.flow is None:
            return
        totalStages = len(self.flow[startFrom:])
        for iseg, segment in enumerate(self.flow[startFrom:]):
            segOE = self.oesDict[segment[0]][0]
            fArgs = OrderedDict()
            for inArg in segment[2].items():
                if inArg[0].startswith('beam'):
                    if inArg[1] is None:
                        inBeam = None
                        break
                    fArgs[inArg[0]] = self.beamsDict[inArg[1]]
                    inBeam = fArgs['beam']
                else:
                    fArgs[inArg[0]] = inArg[1]
            try:
                if inBeam is None:
                    continue
            except NameError:
                pass

            try:  # protection againt incorrect propagation parameters
                if signal is not None:
                    signalStr = "Propagation: {0} {1}(), %p% done.".format(
                        str(segment[0]),
                        str(segment[1]).split(".")[-1].strip(">").split(
                                " ")[0])
                    signal.emit((float(iseg+1)/float(totalStages), signalStr))
                    self.statusSignal =\
                        [signal, iseg+1, totalStages, signalStr]
            except Exception:
                pass

            try:
                outBeams = segment[1](segOE, **fArgs)
            except Exception:
                if _DEBUG_:
                    raise
                else:
                    continue

            if isinstance(outBeams, tuple):
                for outBeam, beamName in zip(list(outBeams),
                                             list(segment[3].values())):
                    self.beamsDict[beamName] = outBeam
            else:
                self.beamsDict[str(list(segment[3].values())[0])] = outBeams

    def sort_flow(self):
        visited = set()
        result = []
        newFlow = OrderedDict()

        def get_beam_id(oeid):
            methDict = self.flowU.get(oeid)
            if methDict is None:
                return None
            for kwargs in methDict.values():
                sourceid = kwargs.get('beam')
                return sourceid

        def dfs(oeid):
            visited.add(oeid)
            for rcv in receivers.get(oeid, []):
                if rcv not in visited:
                    dfs(rcv)
            result.append(oeid)

        def distance(id1, id2):
            line1 = self.oesDict.get(id1)
            line2 = self.oesDict.get(id2)

            if line1 and line2:
                obj1 = line1[0]
                obj2 = line2[0]
            else:
                return 0

            eid1c = np.array(getattr(obj1, 'center', [0, 0, 0]))
            eid2c = np.array(getattr(obj2, 'center', [0, 0, 0]))

            try:
                dist = np.linalg.norm(eid1c - eid2c)
            except:
                dist = 0
                
            return dist

        receivers = {}
        for oeid in self.flowU:
            sourceid = get_beam_id(oeid)
            if sourceid is not None:
                receivers.setdefault(sourceid, []).append(oeid)

        for sourceid, rcv in receivers.items():
            if sourceid in self.flowU:
                rcv.sort(key=lambda oid: distance(oid, sourceid), reverse=True)

        for oe in self.flowU:
            if oe not in visited:
                dfs(oe)

        for oeuuid in reversed(result):
            tmpRec = self.flowU.get(oeuuid)
            if tmpRec is not None:
                newFlow[oeuuid] = tmpRec
                
        print("NEW FLOW:", newFlow)

        self.flowU = newFlow

    def sort_materials(self, matDict=None, fromJSON=False):
        visited = set()
        visiting = set()
        sortedMatList = []
        matDeps = ['tlayer', 'blayer', 'coating', 'substrate']

        def get_dep_obj(matObj):
            deps = []
            for attr in matDeps:
                if hasattr(matObj, attr):
                    v = getattr(matObj, attr)
                    if v is not None and hasattr(v, 'uuid'):
                        deps.append(getattr(v, 'uuid'))
            return deps

        def get_dep_json(pDict):
            deps = []
            props = pDict.get('properties')
            if props is not None:
                for attr in matDeps:
                    v = props.get(attr)
                    if v is not None:
                        deps.append(v)
            return deps
    
        def dfs(mId, mProps):
            if mId in visited:
                return
            if mId in visiting:
                raise ValueError(f"Circular dependency detected involving {mId}")
            visiting.add(mId)
            for dep in get_dependencies(mProps):
                dfs(dep, mProps)
            visiting.remove(mId)
            visited.add(mId)
            sortedMatList.append(mId)

        if matDict is None:
            matDict = self.materialsDict

        get_dependencies = get_dep_json if fromJSON else get_dep_obj
    
        for mId, mProps in matDict.items():
            dfs(mId, mProps)

        return sortedMatList        

    def index_materials(self):
        materialsDict = OrderedDict()
        for ename, eLine in self.oesDict.items():
            oe = eLine[0]
            for attr in ['material', 'material2']:
                if hasattr(oe, attr):
                    attrMat = getattr(oe, attr)
                    if not is_sequence(attrMat):
                        seqMat = (attrMat,)
                    for newMat in seqMat:
                        if newMat is not None and newMat.uuid not in\
                                materialsDict:
                            materialsDict[newMat.uuid] = newMat
                        for subAttr in ['tlayer', 'blayer',
                                        'coating', 'substrate']:
                            if hasattr(newMat, subAttr):
                                subMat = getattr(newMat, subAttr)
                                if subMat.uuid not in materialsDict:
                                    materialsDict[subMat.uuid] = subMat
                                    materialsDict.move_to_end(
                                            subMat.uuid,
                                            last=False)
        self.materialsDict.update(materialsDict)

    def glow(self, scale=[], centerAt='', startFrom=0, colorAxis=None,
             colorAxisLimits=None, generator=None, generatorArgs=[], v2=False,
             **kwargs):
        if generator is not None:
            gen = generator(*generatorArgs)
            try:
                if sys.version_info < (3, 1):
                    gen.next()
                else:
                    next(gen)
            except StopIteration:
                return

        try:
            from ...gui import xrtGlow as xrtglow
        except ImportError as e:
            print("Cannot import xrtGlow. "
                  "If you run your script from an IDE, don't.")
            print(e)
            return

        from .run import run_process
        run_process(self)

        if self.blViewer is None:
            app = xrtglow.qt.QApplication.instance()
            if app is None:
                app = xrtglow.qt.QApplication(sys.argv)
            if v2:
                self.index_materials()
#                materialsDict = OrderedDict()
#                for ename, eLine in self.oesDict.items():
#                    oe = eLine[0]
#                    for attr in ['material', 'material2']:
#                        if hasattr(oe, attr):
#                            attrMat = getattr(oe, attr)
#                            if not is_sequence(attrMat):
#                                seqMat = (attrMat,)
#                            for newMat in seqMat:
#                                if newMat is not None and newMat.uuid not in\
#                                        materialsDict:
#                                    materialsDict[newMat.uuid] = newMat
#                                for subAttr in ['tlayer', 'blayer',
#                                                'coating', 'substrate']:
#                                    if hasattr(newMat, subAttr):
#                                        subMat = getattr(newMat, subAttr)
#                                        if subMat.uuid not in materialsDict:
#                                            materialsDict[subMat.uuid] = subMat
#                                            materialsDict.move_to_end(
#                                                    subMat.uuid,
#                                                    last=False)
#                self.materialsDict.update(materialsDict)
                _ = self.export_to_json()  # layoutStr is populated inside

                self.blViewer = xrtglow.xrtGlow(layout=self.layoutStr,
                                                **kwargs)
            else:
                rayPath = self.export_to_glow()
                self.blViewer = xrtglow.xrtGlow(rayPath)
            self.blViewer.generator = generator
            self.blViewer.generatorArgs = generatorArgs
            self.blViewer.customGlWidget.generator = generator
            self.blViewer.setWindowTitle("xrtGlow")
            self.blViewer.startFrom = startFrom
            self.blViewer.bl = self
            if scale:
                try:
                    self.blViewer.updateScaleFromGL(scale)
                except Exception:
                    pass
            if centerAt:
                try:
                    self.blViewer.centerEl(centerAt)
                except Exception:
                    pass
            if colorAxis:
                try:
                    colorCB = self.blViewer.colorControls[0]
                    colorCB.setCurrentIndex(colorCB.findText(colorAxis))
                except Exception:
                    pass
            if colorAxisLimits:
                try:
                    self.blViewer.customGlWidget.colorMin,\
                        self.blViewer.customGlWidget.colorMax = colorAxisLimits
                    self.blViewer.changeColorAxis(None, newLimits=True)
                except Exception:
                    pass

            self.blViewer.show()
            sys.exit(app.exec_())
        else:
            self.blViewer.show()

    def explore(self, plots=None):
        try:
            from ...gui import xrtQook as xrtqook
        except ImportError as e:
            print("Cannot import xrtGlow. "
                  "If you run your script from an IDE, don't.")
            print(e)
            return

        from .run import run_process
        run_process(self)

        if self.blExplorer is None:
            app = xrtqook.qt.QApplication.instance()
            if app is None:
                app = xrtqook.qt.QApplication(sys.argv)
            self.index_materials()
            layout = self.export_to_json()
            if plots is not None and not layout['plots']:
                layout['plots'].update(plots)
            self.blExplorer = xrtqook.XrtQook(loadLayout=layout)
            self.blExplorer.setWindowTitle("xrtQook")
            self.blExplorer.show()

    def export_to_glow(self, signal=None):
        def calc_weighted_center(beam):
            good = (beam.state == 1) | (beam.state == 2)
            intensity = beam.Jss[good] + beam.Jpp[good]
            totalI = np.sum(intensity)
            if totalI == 0:
                beam.wCenter = np.array([0., 0., 0.])
            else:
                beam.wCenter = np.array(
                    [np.sum(beam.x[good] * intensity),
                     np.sum(beam.y[good] * intensity),
                     np.sum(beam.z[good] * intensity)]) /\
                    totalI

        if self.flow is not None:
            beamDict = OrderedDict()
            rayPath = []
            outputBeamMatch = OrderedDict()
            oesDict = OrderedDict()
            totalStages = len(self.flow)
            for iseg, segment in enumerate(self.flow):
                try:
                    if signal is not None:
                        signalStr = "Processing {0} beams, %p% done.".format(
                            str(segment[0]))
                        signal.emit((float(iseg+1) / float(totalStages),
                                     signalStr))
                except Exception:
                    if _DEBUG_:
                        raise
                    else:
                        pass

                try:
                    methStr = str(segment[1])

                    oeStr = segment[0]
                    segOE = self.oesDict[oeStr][0]
                    if segOE is None:  # Protection from non-initialized OEs
                        continue
                    oesDict[oeStr] = self.oesDict[oeStr]
                    if 'beam' in segment[2].keys():
                        if str(segment[2]['beam']) == 'None':
                            continue
                        tmpBeamName = segment[2]['beam']
                        beamDict[tmpBeamName] = copy.deepcopy(
                            self.beamsDict[tmpBeamName])

                    if 'beamGlobal' in segment[3].keys():
                        outputBeamMatch[segment[3]['beamGlobal']] = oeStr

                    if len(re.findall('raycing.sou',
                                      str(type(segOE)).lower())):
                        gBeamName = segment[3]['beamGlobal']
                        beamDict[gBeamName] = self.beamsDict[gBeamName]
                        rayPath.append([oeStr, gBeamName, None, None])
                    elif len(re.findall(('expose'), methStr)) > 0 and\
                            len(re.findall(('expose_global'), methStr)) == 0:
                        gBeam = self.oesDict[oeStr][0].expose_global(
                            self.beamsDict[tmpBeamName])
                        gBeamName = '{}toGlobal'.format(
                            segment[3]['beamLocal'])
                        beamDict[gBeamName] = gBeam
                        if tmpBeamName in outputBeamMatch:
                            # if no good rays, the condition is False
                            rayPath.append([outputBeamMatch[tmpBeamName],
                                            tmpBeamName, oeStr, gBeamName])
                    elif len(re.findall(('double'), methStr)) +\
                            len(re.findall(('multiple'), methStr)) > 0:
                        lBeam1Name = segment[3]['beamLocal1']
                        gBeam = copy.deepcopy(self.beamsDict[lBeam1Name])
                        segOE.local_to_global(gBeam)
                        g1BeamName = '{}toGlobal'.format(lBeam1Name)
                        beamDict[g1BeamName] = gBeam
                        rayPath.append([outputBeamMatch[tmpBeamName],
                                        tmpBeamName, oeStr, g1BeamName])
                        gBeamName = segment[3]['beamGlobal']
                        beamDict[gBeamName] = self.beamsDict[gBeamName]
                        rayPath.append([oeStr, g1BeamName,
                                       oeStr, gBeamName])
                    elif len(re.findall(('propagate'), methStr)) > 0:
                        if 'beamGlobal' in segment[3].keys():
                            lBeam1Name = segment[3]['beamGlobal']
                            gBeamName = lBeam1Name
                        else:
                            lBeam1Name = segment[3]['beamLocal']
                            gBeamName = '{}toGlobal'.format(lBeam1Name)
                        gBeam = copy.deepcopy(self.beamsDict[lBeam1Name])
                        segOE.local_to_global(gBeam)
                        beamDict[gBeamName] = gBeam
                        rayPath.append([outputBeamMatch[tmpBeamName],
                                        tmpBeamName, oeStr, gBeamName])
                    else:
                        gBeamName = segment[3]['beamGlobal']
                        beamDict[gBeamName] = self.beamsDict[gBeamName]
                        rayPath.append([outputBeamMatch[tmpBeamName],
                                        tmpBeamName, oeStr, gBeamName])
                except Exception as e:
                    if _DEBUG_:
                        raise e
                    else:
                        continue

        totalBeams = len(beamDict)
        for itBeam, tBeam in enumerate(beamDict.values()):
            if signal is not None:
                try:
                    signalStr = "Calculating trajectory, %p% done."
                    signal.emit((float(itBeam+1)/float(totalBeams), signalStr))
                except Exception:
                    if _DEBUG_:
                        raise
                    else:
                        pass
            if tBeam is not None:
                calc_weighted_center(tBeam)
        return [rayPath, beamDict, oesDict]

    def init_oe_from_json(self, elProps, isString=True):
        oeParams = elProps.get('properties')

        if elProps.get('_object') is None:
            return
        else:
            oeModule, oeClass = elProps['_object'].rsplit('.', 1)
            oeModule = importlib.import_module(oeModule)
            defArgs = dict(get_params(elProps['_object']))

            if isString:
                initKWArgs = create_paramdict_oe(oeParams, defArgs, self)
            else:
                initKWArgs = oeParams

            try:
                _ = getattr(oeModule, oeClass)(**initKWArgs)
                initStatus = 0
            except Exception as e:  # TODO: Needs testing
                print(oeClass, "Init problem:", e)
                initStatus = 1
                # raise

        self.oenamesToUUIDs[oeParams['name']] = oeParams['uuid']
        self.update_flow_from_json(oeParams['uuid'], elProps)

        return initStatus

    def delete_oe_by_id(self, elid):
        for oename, oeid in self.oenamesToUUIDs.items():
            if oeid == elid:
                del self.oenamesToUUIDs[oename]
                break

        if elid in self.flowU:
            del self.flowU[elid]
            for eluuid, props in list(self.flowU.items()):
                for methName, methArgs in list(props.items()):
                    if methArgs.get('beam') == elid:
                        methArgs['beam'] = None

        if elid in self.beamsDictU:
            del self.beamsDictU[elid]

        if elid in self.oesDict:
            del self.oesDict[elid]

    def delete_mat_by_id(self, matid):
        for matname, tmpid in self.matnamesToUUIDs.items():
            if tmpid == matid:
                del self.matnamesToUUIDs[matname]
                break

        for elid, elLine in self.oesDict.items():
            elObj = elLine[0]
            for prop in ['material', 'material2']:
                if hasattr(elObj, prop) and getattr(elObj,
                        prop) == self.materialsDict[matid]:
                    setattr(elObj, prop, None)

        for tmpid, matobj in self.materialsDict.items():
            elObj = elLine[0]
            for prop in ['tLayer', 'bLayer', 'coating', 'substrate']:
                if hasattr(matobj, prop) and getattr(matobj,
                        prop) == self.materialsDict[matid]:
                    setattr(matobj, prop, None)

        if matid in self.materialsDict:
            del self.materialsDict[matid]

    def update_flow_from_json(self, oeid, methDict):
        for methStr, methArgs in methDict.items():
            if methStr in ['properties', '_object']:
                continue
            else:
                fArgs = {}
                isEmpty = False
                for argName, argVal in methArgs['parameters'].items():
                    if argName == "beam":
                        if is_valid_uuid(argVal):
                            fArgs[argName] = argVal
                        elif argVal == 'None' or argVal is None:
                            isEmpty = True
                        else:
                            beamTag = self.beamNamesDict.get(str(argVal))
                            if beamTag is not None:
                                fArgs[argName] = beamTag[0]
                            else:
                                print(argVal, "missing in beamNamesDict")
                                return
                    else:
                        fArgs[argName] = parametrize(argVal)

                if isEmpty:
                    self.flowU.pop(oeid, None)
                    continue

                self.flowU[oeid] = {methStr: fArgs}
                if 'output' in methArgs:
                    self.beamsDictU[oeid] = {}
                    for beamType, beamName in methArgs['output'].items():
                        self.beamNamesDict[str(beamName)] = (oeid, beamType)
                        self.beamsDictU[oeid][beamType] = None
                    break  # Normally just one method per element.

    def populate_oes_dict_from_json(self, dictIn):
        if not isinstance(dictIn, dict):
            return
        for elName, elProps in dictIn.items():
            if elName in ['properties', '_object']:
                continue
            if is_valid_uuid(elName):
                elKey = elName
                if 'name' not in elProps['properties']:
                    tmpName = "{}_{}".format(
                            str(dictIn['_object']).split('.')[-1],
                            np.random.randint(1000, 9999))
                    elProps['properties']['name'] = tmpName
                # TODO: check if the name is unique
#                self.oenamesToUUIDs[elProps['properties']['name']] = elKey
            else:
                elKey = str(uuid.uuid4())
                elProps['properties']['name'] = elName
            elProps['properties']['uuid'] = elKey
            _ = self.init_oe_from_json(elProps)  # oesDict populated in oe init

        for elName, eluuid in self.oenamesToUUIDs.items():
            if elName in dictIn:
                dictIn[eluuid] = dictIn.pop(elName)

    def init_material_from_json(self, matName, dictIn):
        matModule, matClass = dictIn['_object'].rsplit('.', 1)
        matModule = importlib.import_module(matModule)
        matParams = dictIn['properties']

        defArgs = dict(get_params(dictIn['_object']))
        initKWArgs = create_paramdict_mat(matParams, defArgs, self)

        if is_valid_uuid(matName):
            initKWArgs['uuid'] = matName
        else:
            initKWArgs['name'] = matName

        matObject = None
        initKWArgs['bl'] = self
        max_retries = 5
        delay = 0.001
        for retry in range(max_retries):
            try:
                matObject = getattr(matModule, matClass)(**initKWArgs)
                initStatus = 0
                break
    #            print("Initalized", matObject, initKWArgs)
            except FileNotFoundError:
                delay *= 5
                print("File read retry", retry, "delay", delay, "s")
                time.sleep(delay)
            except Exception as e:
                matObject = getattr(matModule, "EmptyMaterial")()
                matObject.uuid = initKWArgs.get('uuid')
                matObject.name = initKWArgs.get('name')
                initStatus = 1
                print(matClass, "Init problem. Falling back to EmptyMaterial")
                print(e)
                break
    #            raise

        self.matnamesToUUIDs[matObject.name] = matObject.uuid
        self.materialsDict[matObject.uuid] = matObject
        return initStatus

    def populate_materials_dict_from_json(self, dictIn):
        if not isinstance(dictIn, dict):
            return

        matSorted = self.sort_materials(matDict=dictIn, fromJSON=True)

#        for matName, matProps in dictIn.items():
        for matName in matSorted:
            matProps = dictIn.get(matName)
            if matProps is not None:
                _ = self.init_material_from_json(matName, matProps)

    def load_from_xml(self, openFileName):
        def xml_to_dict(element):
            # Recursively convert XML elements into a dictionary
            if len(element) == 0:  # Base case: if element has no children
                return element.text

            result = OrderedDict()
            for child in element:
                result[child.tag] = xml_to_dict(child)

            return result

        with open(openFileName, "r", encoding="utf-8") as f:
            treeImport = ET.parse(f)

#        treeImport = ET.parse(openFileName)
        root = treeImport.getroot()
        xml_dict = OrderedDict()
        xml_dict[root.tag] = xml_to_dict(root)
        self.deserialize(xml_dict)

    def load_from_json(self, openFileName):
        with open(openFileName, 'r', encoding="utf-8") as file:
            data = json.load(file)
        self.deserialize(data)

    def deserialize(self, data):
        self.layoutStr = data
        beamlineName = next(islice(data['Project'].keys(), 2, 3))
        self.name = beamlineName
        beamlineInitKWargs = data['Project'][beamlineName]['properties']
        for key, value in beamlineInitKWargs.items():
            setattr(self, key, get_init_val(value))

        self.populate_materials_dict_from_json(data['Project']['Materials'])

        self.populate_oes_dict_from_json(data['Project'][beamlineName])
        if 'flow' in data['Project'].keys():
            self.flowU = data['Project']['flow']

    def export_to_json(self):

        matDict = OrderedDict()
        beamlineDict = OrderedDict()
        plotsDict = OrderedDict()
        runDict = None
        descriptionStr = None

        if self.layoutStr is not None:
            plotsDict = self.layoutStr['Project'].get('plots')
            runDict = self.layoutStr['Project'].get('run_ray_tracing')
            descriptionStr = self.layoutStr['Project'].get('description')

            if not isinstance(plotsDict, dict):
                plotsDict = {}

            if runDict is not None:
                runDict['beamLine'] = self.name

        for objName, objInstance in self.materialsDict.items():
            matRecord = OrderedDict()
            matRecord['properties'] = get_init_kwargs(objInstance,
                                                      compact=False)
            matRecord['_object'] = get_obj_str(objInstance)

            if not matRecord['properties']['name']:
                matRecord['properties']['name'] = objName
            field = objInstance.uuid if hasattr(
                    objInstance, 'uuid') else objName
            matDict[field] = matRecord

        blArgs = get_init_kwargs(self)

        beamlineDict['properties'] = blArgs
        beamlineDict['_object'] = get_obj_str(self)

        for oeid, oeline in self.oesDict.items():
            oeObj = oeline[0]
            oeRecord = OrderedDict()
            oeRecord['properties'] = get_init_kwargs(oeObj, compact=True,
                                                     blname=self.name,
                                                     resolveAuto=False)
            oeRecord['_object'] = get_obj_str(oeObj)
            if 'name' not in oeRecord['properties']:
                tmpName = "{}_{}".format(
                        str(oeRecord['_object']).split('.')[-1],
                        np.random.randint(1000, 9999))
                oeRecord['properties']['name'] = tmpName
            beamlineDict[oeid] = oeRecord                
#            print("Exporting to json", oeRecord['properties']['name'], oeRecord['properties'])

#        beamsDict = {}
#
#        # TODO: replace with flowU?
#        for segment in self.flow:
#            method = segment[1]
#            module = method.__module__
#            class_name = method.__class__.__name__
#            method_name = method.__name__
#            methDict = OrderedDict()
#            methDict['_object'] = "{0}.{1}.{2}".format(module, class_name,
#                                                       method_name)
#            methDict['parameters'] = {k: str(v) for k, v
#                                      in segment[2].items()}
#            methDict['output'] = segment[3]
#            beamlineDict[segment[0]][method_name] = methDict
#            for bname in segment[3].values():
#                beamsDict[bname] = None

        projectDict = OrderedDict()
        projectDict['Beams'] = self.beamNamesDict
        projectDict['Materials'] = matDict
        projectDict[self.name] = beamlineDict
        projectDict['flow'] = self.flowU

        if plotsDict is not None:
            for plot, props in plotsDict.items():
                for argName, argVal in props.items():
                    if argName == 'beam' and argVal in self.beamNamesDict:
                        props[argName] = self.beamNamesDict.get(argVal)
                        break
        projectDict['plots'] = plotsDict
        projectDict['run_ray_tracing'] = runDict
        projectDict['description'] = descriptionStr

        self.layoutStr = {'Project': projectDict}
#        print("EXPORT:", self.layoutStr)
        return projectDict
