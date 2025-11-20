# -*- coding: utf-8 -*-
"""
Optical elements
----------------

Module :mod:`~xrt.backends.raycing.oes` defines a generic optical element in
class :class:`OE`. Its methods serve mainly for propagating the beam downstream
the beamline. This is done in the following sequence: for each ray transform
the beam from global to local coordinate system, find the intersection point
with the OE surface, define the corresponding state of the ray, calculate the
new direction, rotate the coherency matrix to the local s-p basis, calculate
reflectivity or transmittivity and apply it to the coherency matrix, rotate the
coherency matrix back and transform the beam from local to the global
coordinate system.

Module :mod:`~xrt.backends.raycing.oes` defines also several other optical
elements with various geometries.

.. autoclass:: OE()
   :members: __init__, local_z, local_n, local_n_distorted, local_g, reflect,
             multiple_reflect, prepare_wave
.. autoclass:: DicedOE(OE)
   :members: __init__, facet_center_z, facet_center_n, facet_delta_z,
             facet_delta_n
.. autoclass:: JohannCylinder(OE)
   :members: __init__
.. autoclass:: JohanssonCylinder(JohannCylinder)
.. autoclass:: JohannToroid(OE)
   :members: __init__
.. autoclass:: JohanssonToroid(JohannToroid)
.. autoclass:: GeneralBraggToroid(JohannToroid)

.. autoclass:: DicedJohannToroid(DicedOE, JohannToroid)
.. autoclass:: DicedJohanssonToroid(DicedJohannToroid, JohanssonToroid)
.. autoclass:: LauePlate(OE)
.. autoclass:: BentLaueCylinder(OE)
   :members: __init__
.. autoclass:: BentLaue2D(OE)
   :members: __init__
.. autoclass:: GroundBentLaueCylinder(BentLaueCylinder)
.. autoclass:: BentLaueSphere(BentLaueCylinder)
.. autoclass:: BentFlatMirror(OE)
.. autoclass:: ToroidMirror(OE)
.. .. autoclass:: MirrorOnTripodWithTwoXStages(OE, stages.Tripod, stages.TwoXStages)  # analysis:ignore
..    :members: __init__
.. .. autoclass:: SimpleVCM(OE)
.. .. autoclass:: VCM(MirrorOnTripodWithTwoXStages)
.. .. autoclass:: SimpleVFM(OE)
.. .. autoclass:: VFM(MirrorOnTripodWithTwoXStages, SimpleVFM)
.. .. autoclass:: DualVFM(MirrorOnTripodWithTwoXStages)
..    :members: __init__
.. .. autoclass:: EllipticalMirror(OE)
.. .. autoclass:: ParabolicMirror(OE)
.. autoclass:: EllipticalMirrorParam(OE)
.. autoclass:: ParabolicalMirrorParam(EllipticalMirrorParam)
.. autoclass:: HyperbolicMirrorParam(OE)
.. autoclass:: ConicalMirror(OE)
   :members: __init__
.. autoclass:: DCM(OE)
   :members: __init__, double_reflect
.. autoclass:: DCMwithSagittalFocusing(DCM)
   :members: __init__
.. .. autoclass:: DCMOnTripodWithOneXStage(DCM, stages.Tripod, stages.OneXStage)
..    :members: __init__
.. autoclass:: Plate(DCM)
   :members: __init__, double_refract
.. autoclass:: ParaboloidFlatLens(Plate)
   :members: __init__
.. autoclass:: ParabolicCylinderFlatLens(ParaboloidFlatLens)
.. autoclass:: DoubleParaboloidLens(ParaboloidFlatLens)
.. autoclass:: DoubleParabolicCylinderLens(ParabolicCylinderFlatLens)
.. autoclass:: SurfaceOfRevolution(OE)
.. autoclass:: ParaboloidCapillaryMirror(SurfaceOfRevolution)
   :members: __init__
.. autoclass:: EllipsoidCapillaryMirror(SurfaceOfRevolution)
   :members: __init__
.. autoclass:: HyperboloidCapillaryMirror(SurfaceOfRevolution)
   :members: __init__
.. autoclass:: NormalFZP(OE)
   :members: __init__, rays_good
.. autoclass:: GeneralFZPin0YZ(OE)
   :members: __init__
.. autoclass:: BlazedGrating(OE)
   :members: __init__
.. autoclass:: LaminarGrating(OE)
.. autoclass:: VLSLaminarGrating(OE)
   :members: __init__

.. _distorted:

Distorted surfaces
~~~~~~~~~~~~~~~~~~

For introducing an error to an ideal surface you must define two methods in
your descendant of the :class:`OE`: ``local_z_distorted`` (or
``local_r_distorted`` for a parametric surface) and ``local_n_distorted``. The
latter method returns two angles d_pitch and d_roll or a 3D vector that will be
added to the local normal. See the docstrings of :meth:`OE.local_n_distorted`
and the example ':ref:`warping`'.
"""
from __future__ import print_function
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "15 Jul 2025"
__all__ = ('OE', 'DicedOE', 'JohannCylinder', 'JohanssonCylinder',
           'JohannToroid', 'JohanssonToroid', 'GeneralBraggToroid',
           'DicedJohannToroid', 'DicedJohanssonToroid', 'LauePlate',
           'BentLaueCylinder', 'BentLaue2D', 'GroundBentLaueCylinder',
           'BentLaueSphere', 'BentFlatMirror', 'ToroidMirror',
           'EllipticalMirrorParam', 'ParabolicalMirrorParam',
           'HyperbolicMirrorParam', 'ConicalMirror',
           'ParaboloidCapillaryMirror', 'EllipsoidCapillaryMirror',
           'HyperboloidCapillaryMirror',
           'DCM', 'DCMwithSagittalFocusing', 'Plate',
           'ParaboloidFlatLens', 'ParabolicCylinderFlatLens',
           'DoubleParaboloidLens', 'DoubleParabolicCylinderLens',
           'SurfaceOfRevolution', 'NormalFZP',
           'GeneralFZPin0YZ', 'BlazedGrating', 'LaminarGrating',
           'VLSLaminarGrating')
import collections
__allSectioned__ = collections.OrderedDict([
    ('Generic',
        ('OE', 'DicedOE', 'DCM', 'Plate', 'SurfaceOfRevolution')),
    ('Curved mirrors',
        ('BentFlatMirror', 'ToroidMirror', 'EllipticalMirrorParam',
         'ParabolicalMirrorParam', 'HyperbolicMirrorParam',
         'ConicalMirror')),
    ('Crystal optics',
        ('JohannCylinder', 'JohanssonCylinder', 'JohannToroid',
         'JohanssonToroid', 'GeneralBraggToroid', 'DicedJohannToroid',
         'DicedJohanssonToroid', 'LauePlate', 'BentLaueCylinder', 'BentLaue2D',
         'GroundBentLaueCylinder', 'BentLaueSphere',
         'DCMwithSagittalFocusing')),
    ('Refractive optics',
        ('ParaboloidFlatLens', 'ParabolicCylinderFlatLens',
         'DoubleParaboloidLens', 'DoubleParabolicCylinderLens')),
    ('Capillary Mirrors',
        ('ParaboloidCapillaryMirror', 'EllipsoidCapillaryMirror',
         'HyperboloidCapillaryMirror')),
    ('Gratings and zone plates',
        ('NormalFZP', 'GeneralFZPin0YZ', 'BlazedGrating', 'LaminarGrating',
         'VLSLaminarGrating'))
    ])

import os
# import gc
import numpy as np

from .. import raycing
from . import myopencl as mcl
from . import stages as rst
from .physconsts import CH  # keep it  # analysis:ignore

from .oes_base import OE, DCM, allArguments
from .oes_bragg import *
from .oes_laue import *
from .oes_parametric import *
from .oes_refractive import *
from .oes_gratings import *
from .oes_3d import *

# try:
#     import pyopencl as cl  # analysis:ignore
#     isOpenCL = True
# except ImportError:
#     isOpenCL = False

if mcl.isOpenCL or mcl.isZMQ:
    isOpenCL = True
else:
    isOpenCL = False

__fdir__ = os.path.dirname(__file__)

allParamsSorted = []


def flatten(x):
    if x is None:
        x = [0, 0]
    if isinstance(x, (list, tuple)):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


class MirrorOnTripodWithTwoXStages(OE, rst.Tripod, rst.TwoXStages):
    """Combines a simple mirror with a tripod support + two X-stages."""

    def __init__(self, *args, **kwargs):
        r"""
        *jack1*, *jack2*, *jack3*: 3-lists
            3d points in the global coordinate system at the horizontal state
            of OE.

        *tx1*, *tx2*: 2-lists
            [x, y] points in local system. dx is the nominal x shift of the
            center in local system.


        """
        kwargs, argsT = rst.Tripod.pop_kwargs(self, **kwargs)
        kwargs, argsX = rst.TwoXStages.pop_kwargs(self, **kwargs)
        OE.__init__(self, *args, **kwargs)
        rst.Tripod.__init__(self, *argsT)
        rst.TwoXStages.__init__(self, *argsX)

    def get_orientation(self):
        """Finds orientation (x, z, and 3 rotations) given two x stages and
        three jacks."""
        rst.TwoXStages.get_orientation(self)
        rst.Tripod.get_orientation(self)


class BentFlatMirror(OE):
    """Implements cylindrical parabolic mirror. Exemplifies inclusion of a new
    parameter (here, *R*) without the need of explicit repetition of all the
    parameters of the parent class."""

    cl_plist = ("R", "limPhysY")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        return 0.5*(y*y - cl_plist.s1 * cl_plist.s1) / cl_plist.s0;
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res = (float3)(0.,-y/cl_plist.s0,1.);
        return normalize(res);
    }"""

    def __init__(self, *args, **kwargs):
        """
        *R*: float or 2-tuple.
            Meridional radius. Can be given as (*p*, *q*) for automatic
            calculation based the "Coddington" equations.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    @property
    def R(self):
        return self._R if self._RVal is None else self._RVal

    @R.setter
    def R(self, R):
        if isinstance(R, (list, tuple)):
            self._R = R
            self._RVal = self.get_Rmer_from_Coddington(*R)
        elif R is None:
            self._R = None
            self._RVal = 1e100
        else:
            self._R = None
            self._RVal = R

    def __pop_kwargs(self, **kwargs):
        self.R = kwargs.pop('R', 5.0e6)
        return kwargs

    def local_z(self, x, y):
        """Determines the surface of OE at (x, y) position. Here: a
        meridionally bent parabolic cylinder with fixed ends and a sag at the
        center."""
        return (y**2 - self.limPhysY[0]**2) / 2.0 / self.R

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        a = 0.  # -dz/dx
        b = -y / self.R  # -dz/dy
        c = 1.
#        norm = (a**2 + b**2 + c**2)**0.5
#        return a/norm, b/norm, c/norm
        norm = (b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]


SimpleVCM = BentFlatMirror


class VCM(SimpleVCM, MirrorOnTripodWithTwoXStages):
    """Implements Vertically Collimating Mirror on support."""

    def __init__(self, *args, **kwargs):
        kwargs, argsT = rst.Tripod.pop_kwargs(self, **kwargs)
        kwargs, argsX = rst.TwoXStages.pop_kwargs(self, **kwargs)
        SimpleVCM.__init__(self, *args, **kwargs)
        rst.Tripod.__init__(self, *argsT)
        rst.TwoXStages.__init__(self, *argsX)


class ToroidMirror(OE):
    """Implements toroidal mirror. Exemplifies inclusion of new
    parameters (here, *R* and *r*) without the need of explicit repetition
    of all the parameters of the parent class."""

    cl_plist = ("R", "r")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        return 0.5 * y * y / cl_plist.s0  + cl_plist.s1 -
            sqrt(cl_plist.s1 * cl_plist.s1 - x * x);
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res;
        res.s0 = -x / sqrt(pown(cl_plist.s1, 2) - pown(x, 2));
        res.s1 = -y / cl_plist.s0;
        res.s2 = 1.;
        return normalize(res);
    }"""

    def __init__(self, *args, **kwargs):
        """
        *R*: float, 2- or 3-tuple.
            Meridional radius. Can be given as (*p*, *q*) or (*p*, *q*, *pitch*)
            for automatic calculation based the "Coddington" equations. If
            pitch is not given explicitly, it will be taken from the *pitch*
            attribute.

        *r*: float or 2- or 3-tuple.
            Sagittal radius. Can be given as (*p*, *q*) or (*p*, *q*, *pitch*)
            for automatic calculation based the "Coddington" equations. If
            pitch is not given explicitly, it will be taken from the *pitch*
            attribute.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    @property
    def R(self):
        return self._R if self._RVal is None else self._RVal

    @R.setter
    def R(self, R):
        if isinstance(R, (list, tuple)):
            self._R = R
            self._RVal = self.get_Rmer_from_Coddington(*R)
        elif R in [0, None]:
            self._R = None
            self._RVal = 1e100
        else:
            self._R = None
            self._RVal = R

    @property
    def r(self):
        return self._r if self._rVal is None else self._rVal

    @r.setter
    def r(self, r):
        if isinstance(r, (list, tuple)):
            self._r = r
            self._rVal = self.get_rsag_from_Coddington(*r)
        elif r in [0, None]:
            self._r = None
            self._rVal = 1e100
        else:
            self._r = None
            self._rVal = r

    def __pop_kwargs(self, **kwargs):
        self.R = kwargs.pop('R', 5.0e6)
        self.r = kwargs.pop('r', 50.)
        return kwargs

    def local_z(self, x, y):
        rx = 1 - (np.asarray(x)/self.r)**2
        rx[rx < 0] = 0.  # becomes flat at the equator
        return y**2/2.0/self.R + self.r*(1 - rx**0.5)

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        rx = 1 - (np.asarray(x)/self.r)**2
        ax = np.where(rx < 0, 0, rx**(-0.5))  # becomes flat at the equator
        a = -x / self.r * ax  # -dz/dx
        b = -y / self.R  # -dz/dy
        c = 1.
        norm = (a**2 + b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]


SimpleVFM = ToroidMirror


class VFM(SimpleVFM, MirrorOnTripodWithTwoXStages):
    """Implements Vertically Focusing Mirror with the fixed ends."""

    cl_plist = ("R", "r", "limPhysY", "limOptX")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float z = cl_plist.s1 - sqrt(pown(cl_plist.s1,2) - x**2);
        if cl_plist.s4 < cl_plist.s5
        {
            float zMax = cl_plist.s1 -
                sqrt(pown(cl_plist.s1,2) - pown(cl_plist.s5,2));
            if (z > zMax) z = zMax;
        }
        z += (y*y - pown(cl_plist.s2,2)) / 2.0 / cl_plist.s0;
        return z;
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res;
        res.s0 = -x / sqrt(pown(cl_plist.s1, 2) - pown(x, 2));
        if cl_plist.s4 < cl_plist.s5
        {
            if ((x < cl_plist.s4) | (x > cl_plist.s5)) res.s0 = 0;
        }
        res.s1 = -y / cl_plist.s0;
        res.s2 = 1.;
        return normalize(res);
    }"""

    def __init__(self, *args, **kwargs):
        limPhysY = kwargs.get('limPhysY', None)
        if limPhysY is None:
            raise AttributeError('limPhysY must be given')
        kwargs, argsT = rst.Tripod.pop_kwargs(self, **kwargs)
        kwargs, argsX = rst.TwoXStages.pop_kwargs(self, **kwargs)
        SimpleVFM.__init__(self, *args, **kwargs)
        rst.Tripod.__init__(self, *argsT)
        rst.TwoXStages.__init__(self, *argsX)

    def local_z(self, x, y):
        """Determines the surface of OE at (x, y) position. Here: a circular
        sagittal cylinder, meridionally parabolically bent with fixed ends and
        a sag at the center."""
        z = self.r - (self.r**2 - x**2)**0.5
        if self.limOptX is not None:
            zMax = self.r - (self.r**2 - self.limOptX[1]**2)**0.5
            z[z > zMax] = zMax
        z += (y**2 - self.limPhysY[0]**2) / 2.0 / self.R
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        a = -x * (self.r**2 - x**2)**(-0.5)  # -dz/dx
        if self.limOptX is not None:
            a[(x < self.limOptX[0]) | (x > self.limOptX[1])] = 0.0  # -dz/dx
        b = -y / self.R  # -dz/dy
        c = 1.
        norm = (a**2 + b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]


class DualVFM(MirrorOnTripodWithTwoXStages):
    """Implements Vertically Focusing Mirror with two toroids."""

    cl_plist = ("r1", "r2", "xCylinder1", "hCylinder1",
                "xCylinder2", "hCylinder2", "limPhysY", "R")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
      float z=0;
      if (x<0)
      {
          z = cl_plist.s1 - cl_plist.s5 - sqrt(pown(cl_plist.s1,2) -
              pown((x - cl_plist.s4), 2));
      }
      else
      {
          z = cl_plist.s0 - cl_plist.s3 - sqrt(pown(cl_plist.s0,2) -
              pown((x - cl_plist.s2), 2));
      }
      if (z>0) z = 0;
      z += (y*y - pown(cl_plist.s6,2)) / 2.0 / cl_plist.s8;
      return z;
    }"""

    def __init__(self, *args, **kwargs):
        """
        *r1*, *r2*: float
            Sagittal radii of the cylinders.

        *xCylinder1*, *xCylinder2*: float
            Local x coordinates of the cylinder axes.

        *hCylinder1*, *hCylinder2*: float
            z depth of the cylinder at x = xCylinder1 and x = xCylinder2.
            Positive depth is under the nominal flat surface (local z = 0).


        """
        kwargs = self.__pop_kwargs(**kwargs)
        MirrorOnTripodWithTwoXStages.__init__(self, *args, **kwargs)
        self.hCylinder = 0

    def __pop_kwargs(self, **kwargs):
        self.R = kwargs.pop('R', 5.0e6)
        self.r1 = kwargs.pop('r1', 70.0)
        self.xCylinder1 = kwargs.pop('xCylinder1', 23.5)
        self.hCylinder1 = kwargs.pop('hCylinder1', 3.7035)
        self.r2 = kwargs.pop('r2', 35.98)
        self.xCylinder2 = kwargs.pop('xCylinder2', -25.0)
        self.hCylinder2 = kwargs.pop('hCylinder2', 6.9504)
        return kwargs

    def local_z(self, x, y):
        """Determines the surface of OE at (x, y) position. Here: two circular
        sagittal cylinders, meridionally parabolically bent with fixed ends and
        a sag at the center."""
        z = np.zeros_like(x)
        ind = x < 0

        with np.errstate(invalid='ignore'):
            tmp2 = self.r2**2 - (x[ind] - self.xCylinder2)**2
            z[ind] = self.r2 - self.hCylinder2 - tmp2**0.5
            z[ind][tmp2 <= 0] = 0
            tmp1 = self.r1**2 - (x[~ind] - self.xCylinder1)**2
            z[~ind] = self.r1 - self.hCylinder1 - tmp1**0.5
            z[~ind][tmp1 <= 0] = 0

        z[np.isnan(z)] = 0.
        z[z > 0] = 0.
        z += (y**2 - self.limPhysY[0]**2) / 2.0 / self.R
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        a = np.zeros_like(x)
        ind = x < 0

        with np.errstate(invalid='ignore'):
            tmp2 = self.r2**2 - (x[ind] - self.xCylinder2)**2
            a[ind] = -(x[ind] - self.xCylinder2) * tmp2**(-0.5)  # -dz/dx
            a[ind][tmp2 <= 0] = 0
            tmp1 = self.r1**2 - (x[~ind] - self.xCylinder1)**2
            a[~ind] = -(x[~ind] - self.xCylinder1) * tmp1**(-0.5)  # -dz/dx
            a[~ind][tmp1 <= 0] = 0

        z = self.local_z(x, y)
        a[np.isnan(a)] = 0.
        a[z > 0] = 0.  # -dz/dx
        b = -y / self.R  # -dz/dy
        c = 1.
        norm = (a**2 + b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]

    def select_surface(self, surfaceName):
        """Updates self.curSurface index and finds dx offset corresponding to
        the requested cylinder(toroid)."""
        self.curSurface = self.surface.index(surfaceName)
        if self.curSurface == 0:
            self.dx = -self.xCylinder1
            self.hCylinder = self.hCylinder1
            self.r = self.r1
        elif self.curSurface == 1:
            self.dx = -self.xCylinder2
            self.hCylinder = self.hCylinder2
            self.r = self.r2
        self.get_surface_limits()
        self.set_x_stages()


class ConicalMirror(OE):
    """Conical mirror with its base parallel to the side of the cone."""

    def __init__(self, *args, **kwargs):
        r"""
        *L0*: float
            Distance from the center of the mirror to the vertex of the cone.
            This distance is measured along the surface, NOT along the axis.

        *theta*: float
            Opening angle of the cone (axis to surface) in radians.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = raycing.auto_units_angle(theta)
        self.tt = np.tan(self._theta)
        self.t2t = np.tan(2*self._theta)
        self.redfocus = np.cos(self._theta)**2 /\
            (1./self.tt-1./self.t2t)

    def __pop_kwargs(self, **kwargs):
        self.L0 = kwargs.pop('L0', 1000.)  # distance to the cone vertex
        self.theta = kwargs.pop('theta', np.pi/6.)
        return kwargs

    def local_z(self, x, y):
        sqroot = np.sqrt(0.25*self.t2t**2*(y - self.L0)**2 -
                         self.redfocus*self.t2t*x**2)
        z = -0.5*self.t2t*(y-self.L0)-np.sign(self.t2t)*sqroot
        return z

    def local_n(self, x, y):
        sqroot = np.sign(self.t2t)*np.sqrt(0.25*self.t2t**2*(y - self.L0)**2 -
                                           self.redfocus*x*x*self.t2t)
        a = -x*self.redfocus*self.t2t/sqroot  # -dz/dx
        b = .5*self.t2t + 0.25*self.t2t**2*(y-self.L0)/sqroot  # -dz/dy
        c = 1.
        norm = (a**2 + b**2 + 1.)**0.5
        return [a/norm, b/norm, c/norm]


class DCMwithSagittalFocusing(DCM):  # composed by Roelof van Silfhout
    """Creates a DCM with a horizontally focusing 2nd crystal."""

    def __init__(self, *args, **kwargs):
        r"""
        Assume Bragg planes and physical surface planes are parallel
        (no miscut angle).

        *Rs*: float
            Sagittal radius of second crystal.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        DCM.__init__(self, *args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        self.Rs = kwargs.pop('Rs', 1e12)
        return kwargs

    def local_z2(self, x, y):
        return self.Rs - np.sqrt(self.Rs**2 - x**2)

    def local_n2(self, x, y):
        a = -x / self.Rs  # -dz/dx
        c = (self.Rs**2-x**2)**0.5 / self.Rs
        b = np.zeros_like(y)  # -dz/dy
        return [a, b, c]


class DCMOnTripodWithOneXStage(DCM, rst.Tripod, rst.OneXStage):
    """Combines a DCM with a tripod support + one X-stage."""

    def __init__(self, *args, **kwargs):
        r"""
        *jack1*, *jack2*, *jack3*: 3-lists
            3d points in the *general* coordinate system at the horizontal
            state of OE.

        *dx*: float
            The nominal *x* shift of the center in local system.


        """
        kwargs, argsT = rst.Tripod.pop_kwargs(self, **kwargs)
        kwargs, argsX = rst.OneXStage.pop_kwargs(self, **kwargs)
        DCM.__init__(self, *args, **kwargs)
        rst.Tripod.__init__(self, *argsT)
        rst.OneXStage.__init__(self, *argsX)

        for alim in (self.limOptX2, self.limOptY2):
            if alim is not None:
                if not (raycing.is_sequence(alim[0]) and
                        raycing.is_sequence(alim[1])):
                    raise ValueError('"limOptX" must be a tuple of sequences!')
                if not (len(alim[0]) == len(alim[1]) == len(self.surface)):
                    raise ValueError(
                        'len(self.limOptX[0,1]) != len(surface) !!!')

        for alim in (self.limPhysX2[0], self.limPhysX2[1],
                     self.limPhysY2[0], self.limPhysY2[1]):
            if raycing.is_sequence(alim):
                if len((self.surface)) != len(alim):
                    raise ValueError(
                        'length of "surface" and "limPhys..." must be equal!')

    def get_orientation(self):
        """Finds orientation (x, z and 3 rotations) given one x stage and
        three jacks."""
        rst.Tripod.get_orientation(self)
