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
             multiple_reflect
.. autoclass:: DicedOE(OE)
   :members: __init__, facet_center_z, facet_center_n, facet_delta_z,
             facet_delta_n
.. autoclass:: JohannCylinder(OE)
   :members: __init__
.. autoclass:: JohanssonCylinder(JohannCylinder)
.. autoclass:: JohannToroid(OE)
   :members: __init__
.. autoclass:: JohanssonToroid(JohannToroid)
.. autoclass:: DicedJohannToroid(DicedOE, JohannToroid)
.. autoclass:: DicedJohanssonToroid(DicedJohannToroid, JohanssonToroid)
.. autoclass:: LauePlate(OE)
.. autoclass:: BentLaueCylinder(OE)
   :members: __init__
.. autoclass:: GroundBentLaueCylinder(BentLaueCylinder)
.. autoclass:: BentLaueSphere(BentLaueCylinder)
.. .. autoclass:: MirrorOnTripodWithTwoXStages(OE, stages.Tripod, stages.TwoXStages)  # analysis:ignore
..    :members: __init__
.. .. autoclass:: SimpleVCM(OE)
.. .. autoclass:: VCM(MirrorOnTripodWithTwoXStages)
.. .. autoclass:: SimpleVFM(OE)
.. .. autoclass:: VFM(MirrorOnTripodWithTwoXStages, SimpleVFM)
.. .. autoclass:: DualVFM(MirrorOnTripodWithTwoXStages)
..    :members: __init__
.. autoclass:: EllipticalMirror(OE)
.. autoclass:: ParabolicMirror(OE)
.. autoclass:: EllipticalMirrorParam(OE)
.. autoclass:: DCM(OE)
   :members: __init__, double_reflect
.. autoclass:: DCMOnTripodWithOneXStage(DCM, stages.Tripod, stages.OneXStage)
   :members: __init__
.. autoclass:: Plate(DCM)
   :members: __init__, double_refract
.. autoclass:: ParaboloidFlatLens(Plate)
   :members: __init__
.. autoclass:: ParabolicCylinderFlatLens(ParaboloidFlatLens)
.. autoclass:: DoubleParaboloidLens(ParaboloidFlatLens)
.. autoclass:: SurfaceOfRevolution(OE)
.. autoclass:: NormalFZP(OE)
   :members: __init__, rays_good
.. autoclass:: GeneralFZPin0YZ(OE)
   :members: __init__
.. autoclass:: BlazedGrating(OE)
   :members: __init__

.. _distorted:

Distorted surfaces
~~~~~~~~~~~~~~~~~~

For introducing an error to an ideal surface you must define two methods in
your descendant of the :class:`OE`: ``local_z_distorted`` (or
``local_r_distorted`` for a parametric surface) and ``local_n_distorted``. The
latter method returns two angles d_pitch and d_roll or a 3D vector that will be
added to the local normal. See the docstrings of
:meth:`OE.local_n_distorted`` and the example ':ref:`warping`'.
"""
from __future__ import print_function
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "16 Mar 2017"
__all__ = ('OE', 'DicedOE', 'JohannCylinder', 'JohanssonCylinder',
           'JohannToroid', 'JohanssonToroid', 'GeneralBraggToroid',
           'DicedJohannToroid', 'DicedJohanssonToroid', 'LauePlate',
           'BentLaueCylinder', 'GroundBentLaueCylinder', 'BentLaueSphere',
           'BentFlatMirror', 'ToroidMirror',
           'EllipticalMirror', 'EllipticalMirrorParam',
           'ParabolicMirror', 'ParabolicalMirrorParam',
           'DCM', 'DCMwithSagittalFocusing', 'Plate',
           'ParaboloidFlatLens', 'ParabolicCylinderFlatLens',
           'DoubleParaboloidLens', 'SurfaceOfRevolution', 'NormalFZP',
           'GeneralFZPin0YZ', 'BlazedGrating')
import os
# import copy
# import gc
import numpy as np
from scipy import interpolate

from .. import raycing
from . import stages as rst
from .physconsts import CH
from .oes_base import OE, DCM
try:
    import pyopencl as cl  # analysis:ignore
    isOpenCL = True
except ImportError:
    isOpenCL = False

__dir__ = os.path.dirname(__file__)
_DEBUG = False


def flatten(x):
    if x is None:
        x = [0, 0]
    if isinstance(x, (list, tuple)):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


class DicedOE(OE):
    """Base class for a diced optical element. It implements a flat diced
    mirror."""
    def __init__(self, *args, **kwargs):
        """
        *dxFacet*, *dyFacet*: float
            Size of the facets.

        *dxGap*, *dyGat*: float
            Width of the gap between facets.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        self.dxFacet = kwargs.pop('dxFacet', 2.1)
        self.dyFacet = kwargs.pop('dyFacet', 1.4)
        self.dxGap = kwargs.pop('dxGap', 0.05)
        self.dyGap = kwargs.pop('dyGap', 0.05)
        self.xStep = self.dxFacet + self.dxGap
        self.yStep = self.dyFacet + self.dyGap
        return kwargs

    def facet_center_z(self, x, y):
        """Z of the facet centers at (*x*, *y*)."""
        return 0  # just flat

    def facet_center_n(self, x, y):
        """Surface normal or (Bragg normal and surface normal)."""
        return [0, 0, 1]  # just flat

    def facet_delta_z(self, u, v):
        """Local Z in the facet coordinates."""
        return 0

    def facet_delta_n(self, u, v):
        """Local surface normal (always without Bragg normal!) in the facet
        coordinates. In the asymmetry case the lattice normal is taken as
        constant over the facet and is given by :meth:`facet_center_n`.
        """
        return [0, 0, 1]

    def local_z(self, x, y, skipReturnZ=False):
        """Determines the surface of OE at (*x*, *y*) position."""
        cx = (x / self.xStep).round() * self.xStep  # center of the facet
        cy = (y / self.yStep).round() * self.yStep  # center of the facet
        cz = self.facet_center_z(cx, cy)
        cn = self.facet_center_n(cx, cy)
        fx = x - cx  # local coordinate in the facet
        fy = y - cy
        if skipReturnZ:
            return fx, fy, cn
        z = cz + (self.facet_delta_z(fx, fy) - cn[-3]*fx - cn[-2]*fy) / cn[-1]
#        inGaps = (abs(fx)>self.dxFacet/2) | (abs(fy)>self.dyFacet/2)
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (*x*, *y*) position. For an
        asymmetric crystal, *local_n* returns 2 normals: the 1st one of the
        atomic planes and the 2nd one of the surface. Note the order!"""
        fx, fy, cn = self.local_z(x, y, skipReturnZ=True)
        deltaNormals = self.facet_delta_n(fx, fy)
        if isinstance(deltaNormals[2], np.ndarray):
            useDeltaNormals = True
        else:
            useDeltaNormals = (deltaNormals[2] != 1)
        if useDeltaNormals:
            cn[-1] += deltaNormals[-1]
            cn[-2] += deltaNormals[-2]
            norm = (cn[-1]**2 + cn[-2]**2 + cn[-3]**2)**0.5
            cn[-1] /= norm
            cn[-2] /= norm
            cn[-3] /= norm
        if self.alpha:
            bAlpha, cAlpha = raycing.rotate_x(cn[1], cn[2],
                                              self.cosalpha, -self.sinalpha)
            return [cn[0], bAlpha, cAlpha, cn[-3], cn[-2], cn[-1]]
        else:
            return cn

    def rays_good(self, x, y, is2ndXtal=False):
        """Returns *state* value as inherited from :class:`OE`. The rays that
        fall inside the gaps are additionally considered as lost."""
# =1: good (intersected)
# =2: reflected outside of working area, =3: transmitted without intersection,
# =-NN: lost (absorbed) at OE#NN - OE numbering starts from 1 !!!
        locState = OE.rays_good(self, x, y, is2ndXtal)
        fx, fy, cn = self.local_z(x, y, skipReturnZ=True)
        inGaps = (abs(fx) > self.dxFacet/2) | (abs(fy) > self.dyFacet/2)
        locState[inGaps] = self.lostNum
        return locState


class JohannCylinder(OE):
    """Simply bent reflective crystal."""

    cl_plist = ("crossSectionInt", "Rm", "alpha")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res=0;
        if (cl_plist.s0 == 0)
        {
            res=cl_plist.s1 - sqrt(cl_plist.s1*cl_plist.s1-y*y);
        }
        else
        {
            res=0.5 * y * y / cl_plist.s1;
        }
        return res;
    }"""

    def __init__(self, *args, **kwargs):
        """
        *Rm*: float
            Meridional radius.

        *crossSection*: str
            Determines the bending shape: either 'circular' or 'parabolic'.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        self.crossSectionInt = 0 if self.crossSection.startswith('circ') else 1
        OE.__init__(self, *args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        self.Rm = kwargs.pop('Rm', 1000.)  # R meridional
        self.crossSection = kwargs.pop('crossSection', 'circular')
        if not (self.crossSection.startswith('circ') or
                self.crossSection.startswith('parab')):
            raise ValueError('unknown crossSection!')
        return kwargs

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position."""
        if self.crossSection.startswith('circ'):  # 'circular'
            sq = self.Rm**2 - y**2
#            sq[sq<0] = 0
            return self.Rm - np.sqrt(sq)
        else:  # 'parabolic'
            return y**2 / 2.0 / self.Rm

    def local_n_cylinder(self, x, y, R, alpha):
        """The main part of :meth:`local_n`. It introduces two new arguments
        to simplify the implementation of :meth:`local_n` in the derived class
        :class:`JohanssonCylinder`."""
        a = np.zeros_like(x)  # -dz/dx
        b = -y / R  # -dz/dy
        if self.crossSection.startswith('circ'):  # 'circular'
            c = (R**2 - y**2)**0.5 / R
        else:  # 'parabolic'
            norm = (b**2 + 1)**0.5
            b /= norm
            c = 1. / norm
        if alpha:
            bAlpha, cAlpha = \
                raycing.rotate_x(b, c, self.cosalpha, -self.sinalpha)
            return [a, bAlpha, cAlpha, a, b, c]
        else:
            return [a, b, c]

    def local_n(self, x, y):
        """Determines the normal vector of OE at (*x*, *y*) position. For an
        asymmetric crystal, *local_n* returns 2 normals."""
        return self.local_n_cylinder(x, y, self.Rm, self.alpha)


class JohanssonCylinder(JohannCylinder):
    """Ground-bent (Johansson) reflective crystal."""
    def local_n(self, x, y):
        """Determines the normal vectors of OE at (*x*, *y*) position: of the
        atomic planes and of the surface."""
        nSurf = self.local_n_cylinder(x, y, self.Rm, None)
# approximate expressions:
#        nBragg = self.local_n_cylinder(x, y, self.Rm*2, self.alpha)
#        return [nBragg[0], nBragg[1], nBragg[2],
#           nSurf[-3], nSurf[-2], nSurf[-1]]
# exact expressions:
        a = np.zeros_like(x)
        b = -y
        c = (self.Rm**2 - y**2)**0.5 + self.Rm
        if self.alpha:
            b, c = raycing.rotate_x(b, c, self.cosalpha, -self.sinalpha)
        norm = np.sqrt(b**2 + c**2)
        return [a/norm, b/norm, c/norm, nSurf[-3], nSurf[-2], nSurf[-1]]


class JohannToroid(OE):
    """2D bent reflective crystal."""

    cl_plist = ("Rm", "Rs")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float z = cl_plist.s0 - cl_plist.s1 -
            sqrt(cl_plist.s0*cl_plist.s0 - y*y);
        float cosangle = sqrt(z*z - x*x) / fabs(z);
        float sinangle = -x / fabs(z);
        float2 tmpz = rotate_y(0, z, cosangle, sinangle);
        return tmpz.s1 + cl_plist.s1
    }"""

    def __init__(self, *args, **kwargs):
        """
        *Rm* and *Rs*: float
            Meridional and sagittal radii.


        """
        kwargs = self.pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    def pop_kwargs(self, **kwargs):
        self.Rm = kwargs.pop('Rm', 1000.)  # R meridional
        self.Rs = kwargs.pop('Rs', None)  # R sagittal
        if self.Rs is None:
            self.Rs = self.Rm
        return kwargs

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position."""
        z = self.Rm - self.Rs - (self.Rm**2 - y**2)**0.5
        cosangle, sinangle = (z**2 - x**2)**0.5 / abs(z), -x/abs(z)
        bla, z = raycing.rotate_y(0, z, cosangle, sinangle)
        return z + self.Rs

    def local_n(self, x, y):
        """Determines the normal vectors of OE at (*x*, *y*) position: of the
        atomic planes and of the surface."""
        return self.local_n_toroid(x, y, self.Rm, self.Rs, self.alpha)

    def local_n_toroid(self, x, y, Rm, Rs, alpha):
        """The main part of :meth:`local_n`. It introduces three new arguments
        to simplify the implementation of :meth:`local_n` in the derived class
        :class:`JohanssonToroid`."""
        a = np.zeros_like(x)
        b = -y / Rm
        c = (Rm**2 - y**2)**0.5 / Rm
        if alpha:
            aAlpha = np.zeros_like(x)
            bAlpha, cAlpha = \
                raycing.rotate_x(b, c, self.cosalpha, -self.sinalpha)
        r = Rs - (Rm - (Rm**2 - y**2)**0.5)
        cosangle, sinangle = (r**2 - x**2)**0.5 / r, -x/r
        a, c = raycing.rotate_y(a, c, cosangle, sinangle)
        if alpha:
            aAlpha, cAlpha = \
                raycing.rotate_y(aAlpha, cAlpha, cosangle, sinangle)
            return [aAlpha, bAlpha, cAlpha, a, b, c]
        else:
            return [a, b, c]


class JohanssonToroid(JohannToroid):
    """Ground-2D-bent (Johansson) reflective optical element."""
    def local_n(self, x, y):
        """Determines the normal vectors of OE at (*x*, *y*) position: of the
        atomic planes and of the surface."""
        nSurf = self.local_n_toroid(x, y, self.Rm, self.Rs, None)
# approximate expressions:
#        nBragg = self.local_n_cylinder(x, y, self.Rm*2, self.alpha)
#        return [nBragg[0], nBragg[1], nBragg[2],
#            nSurf[-3], nSurf[-2], nSurf[-1]]
# exact expressions:
        a = np.zeros_like(x)
        b = -y
        c = (self.Rm**2 - y**2)**0.5 + self.Rm
        norm = np.sqrt(b**2 + c**2)
        b, c = b/norm, c/norm
        if self.alpha:
            b, c = raycing.rotate_x(b, c, self.cosalpha, -self.sinalpha)
        r = self.Rs - (self.Rm - (self.Rm**2 - y**2)**0.5)
        cosangle, sinangle = (r**2 - x**2)**0.5 / r, -x/r
        a, c = raycing.rotate_y(a, c, cosangle, sinangle)
        if self.alpha:
            a, c = raycing.rotate_y(a, c, cosangle, sinangle)
        return [a, b, c, nSurf[-3], nSurf[-2], nSurf[-1]]


class GeneralBraggToroid(JohannToroid):
    """Ground-2D-bent reflective optical element with 4 independent radii:
    meridional and sagittal for the surface (*Rm* and *Rs*) and the atomic
    planes (*RmBragg* and *RsBragg*)."""
    def pop_kwargs(self, **kwargs):
        kw = JohannToroid.pop_kwargs(self, **kwargs)
        self.RmBragg = kw.pop('RmBragg', self.Rm)  # R Bragg meridional
        self.RsBragg = kw.pop('RsBragg', self.Rs)  # R Bragg sagittal
        return kw

    def local_n(self, x, y):
        """Determines the normal vectors of OE at (*x*, *y*) position: of the
        atomic planes and of the surface."""
        nSurf = self.local_n_toroid(x, y, self.Rm, self.Rs, None)
        nSurfBr = self.local_n_toroid(x, y, self.RmBragg, self.RsBragg, None)
        return [nSurfBr[0], nSurfBr[1], nSurfBr[2],
                nSurf[-3], nSurf[-2], nSurf[-1]]


class DicedJohannToroid(DicedOE, JohannToroid):
    """A diced version of :class:`JohannToroid`."""
    def __init__(self, *args, **kwargs):
        kwargs = self.pop_kwargs(**kwargs)
        DicedOE.__init__(self, *args, **kwargs)

    def pop_kwargs(self, **kwargs):
        return JohannToroid.pop_kwargs(self, **kwargs)

    def facet_center_z(self, x, y):
        return JohannToroid.local_z(self, x, y)

    def facet_center_n(self, x, y):
        return JohannToroid.local_n(self, x, y)


class DicedJohanssonToroid(DicedJohannToroid, JohanssonToroid):
    """A diced version of :class:`JohanssonToroid`."""
    def facet_center_n(self, x, y):
        return JohanssonToroid.local_n(self, x, y)

    def facet_delta_z(self, u, v):
        return v**2 / 2.0 / self.Rm

    def facet_delta_n(self, u, v):
        a = 0.  # -dz/dx
        b = -v / self.Rm  # -dz/dy
        c = 1.
        norm = (b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]


class LauePlate(OE):
    """A flat Laue plate. The thickness is defined in its *material* part."""
    def local_n(self, x, y):
        a, b, c = 0, 0, 1
        if self.alpha:
            bB, cB = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
        else:
            bB, cB = c, -b
        return [a, bB, cB, a, b, c]


class BentLaueCylinder(OE):
    """Simply bent reflective optical element in Laue geometry (duMond)."""

    cl_plist = ("crossSectionInt", "R")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
      if (cl_plist.s0 == 0)
        {
          return cl_plist.s1 - sqrt(cl_plist.s1*cl_plist.s1 - y*y);
        }
      else
        {
          return 0.5 * y * y / cl_plist.s1;
        }
    }"""

    def __init__(self, *args, **kwargs):
        """
        *R*: float or 2-tuple.
            Meridional radius. Can be given as (*p*, *q*) for automatic
            calculation based the "Coddington" equations.

        *crossSection*: str
            Determines the bending shape: either 'circular' or 'parabolic'.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        self.crossSectionInt = 0 if self.crossSection.startswith('circ') else 1
        OE.__init__(self, *args, **kwargs)
        if isinstance(self.R, (tuple, list)):
            self.R = self.get_Rmer_from_Coddington(self.R[0], self.R[1])

    def __pop_kwargs(self, **kwargs):
        self.R = kwargs.pop('R', 1.0e4)
        self.crossSection = kwargs.pop('crossSection', 'parabolic')
        if not (self.crossSection.startswith('circ') or
                self.crossSection.startswith('parab')):
            raise ValueError('unknown crossSection!')
        return kwargs

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position."""
        if self.crossSection.startswith('circ'):  # 'circular'
            return self.R - np.sqrt(self.R**2 - y**2)
        else:  # 'parabolic'
            return y**2 / 2.0 / self.R

    def local_n_cylinder(self, x, y, R, alpha):
        """The main part of :meth:`local_n`. It introduces two new arguments
        to simplify the implementation of :meth:`local_n` in the derived class
        :class:`GroundBentLaueCylinder`."""
        a = np.zeros_like(x)  # -dz/dx
        b = -y / R  # -dz/dy
        if self.crossSection.startswith('circ'):  # 'circular'
            c = (R**2 - y**2)**0.5 / R
        elif self.crossSection.startswith('parab'):  # 'parabolic'
            norm = (b**2 + 1)**0.5
            b /= norm
            c = 1. / norm
        else:
            raise ValueError('unknown crossSection!')
        if alpha:
            bB, cB = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
        else:
            bB, cB = c, -b
        return [a, bB, cB, a, b, c]

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        return self.local_n_cylinder(x, y, self.R, self.alpha)


class GroundBentLaueCylinder(BentLaueCylinder):
    """Ground-bent reflective optical element in Laue geometry."""
    def local_n(self, x, y):
        """Determines the normal vectors of OE at (*x*, *y*) position: of the
        atomic planes and of the surface."""
        nSurf = self.local_n_cylinder(x, y, self.R, None)
# approximate expressions:
#        nBragg = self.local_n_cylinder(x, y, self.R*2, self.alpha)
#        return [nBragg[0], nBragg[1], nBragg[2],
#            nSurf[-3], nSurf[-2], nSurf[-1]]
# exact expressions:
        a = np.zeros_like(x)
        b = -y
        c = (self.R**2 - y**2)**0.5 + self.R
        if self.alpha:
            b, c = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
        else:
            b, c = c, -b
        norm = np.sqrt(b**2 + c**2)
        return [a/norm, b/norm, c/norm, nSurf[-3], nSurf[-2], nSurf[-1]]


class BentLaueSphere(BentLaueCylinder):
    """Spherically bent reflective optical element in Laue geometry."""

    cl_plist = ("crossSectionInt", "R")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        if (cl_plist.s0 == 0)
        {
            return cl_plist.s1 - sqrt(cl_plist.s1*cl_plist.s1 - x*x -y*y);
        }
        else
        {
          return 0.5 * (y * y + x * x) / cl_plist.s1;
        }
    }"""

    def local_z(self, x, y):
        if self.crossSection.startswith('circ'):  # 'circular'
            return self.R - np.sqrt(self.R**2 - x**2 - y**2)
        else:  # 'parabolic'
            return (x**2+y**2) / 2.0 / self.R

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        if self.crossSection.startswith('circ'):  # 'circular'
            a = -x * (self.R**2 - x**2 - y**2)**(-0.5)  # -dx/dy
            b = -y * (self.R**2 - x**2 - y**2)**(-0.5)  # -dz/dy
        else:  # 'parabolic'
            a = -x / self.R  # -dz/dx
            b = -y / self.R  # -dz/dy
        c = 1.
        norm = (a**2 + b**2 + 1)**0.5
        aB = 0.
        bB = c
        cB = -b
        normB = (b**2 + c**2)**0.5
        return [aB/normB, bB/normB, cB/normB, a/norm, b/norm, c/norm]


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
        if isinstance(self.R, (tuple, list)):
            self.R = self.get_Rmer_from_Coddington(self.R[0], self.R[1])

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
        *R*: float or 2-tuple.
            Meridional radius. Can be given as (*p*, *q*) for automatic
            calculation based the "Coddington" equations.

        *r*: float or 2-tuple.
            Sagittal radius. Can be given as (*p*, *q*) for automatic
            calculation based the "Coddington" equations.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        if isinstance(self.R, (tuple, list)):
            self.R = self.get_Rmer_from_Coddington(self.R[0], self.R[1])
        if isinstance(self.r, (tuple, list)):
            self.r = self.get_rsag_from_Coddington(self.r[0], self.r[1])

    def __pop_kwargs(self, **kwargs):
        self.R = kwargs.pop('R', 5.0e6)
        self.r = kwargs.pop('r', 50.)
        return kwargs

    def local_z(self, x, y):
        rx = self.r**2 - x**2
        try:
            rx[rx < 0] = 0.
        except TypeError:
            pass
        return y**2/2.0/self.R + self.r - rx**0.5

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        a = -x * (self.r**2-x**2)**(-0.5)  # -dz/dx
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
        z = np.empty_like(x)
        ind = x < 0
        z[ind] = self.r2 - self.hCylinder2 -\
            (self.r2**2 - (x[ind] - self.xCylinder2)**2)**0.5
        z[~ind] = self.r1 - self.hCylinder1 -\
            (self.r1**2 - (x[~ind] - self.xCylinder1)**2)**0.5
        z[z > 0] = 0.
        z += (y**2 - self.limPhysY[0]**2) / 2.0 / self.R
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        ind = x < 0
        a = np.empty_like(x)
        a[ind] = -(x[ind] - self.xCylinder2) * \
            (self.r2**2 - (x[ind] - self.xCylinder2)**2)**(-0.5)  # -dz/dx
        a[~ind] = -(x[~ind] - self.xCylinder1) * \
            (self.r1**2 - (x[~ind] - self.xCylinder1)**2)**(-0.5)  # -dz/dx
        z = self.local_z(x, y)
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


class EllipticalMirror(OE):
    """Implements cylindrical elliptical mirror."""

    cl_plist = ("p", "alpha", "ae", "be", "ce")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
      float delta_y = cl_plist.s0 * cos(cl_plist.s1) - cl_plist.s4;
      float delta_z = -cl_plist.s0 * sin(cl_plist.s1);
      return -cl_plist.s3 *
          sqrt(1 - (pown(((y+delta_y)/cl_plist.s2),2))) - delta_z;
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
      float3 res;
      float delta_y = cl_plist.s0 * cos(cl_plist.s1) - cl_plist.s4;
      res.s0 = 0;
      res.s1 = -cl_plist.s3 * (y+delta_y) /
          sqrt(1 - (pown((y+delta_y)/cl_plist.s2,2)) / pown(cl_plist.s2,2));
      res.s2 = 1.;
      return normalize(res);
    }"""

    def __init__(self, *args, **kwargs):
        """
        *p* and *q*: float
            *p* and *q* arms of the mirror, both are positive.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.get_orientation()

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p')
        self.q = kwargs.pop('q')
        self.isCylindrical = kwargs.pop('isCylindrical', True)  # always!
        self.pcorrected = 0
        return kwargs

    def get_orientation(self):
        if self.pcorrected and self.pitch0 != self.pitch:
            self.pcorrected = 0
        if not self.pcorrected:
            self.gamma = np.pi - 2*self.pitch
            self.ce = 0.5 * np.sqrt(
                self.p**2 + self.q**2 - 2*self.p*self.q * np.cos(self.gamma))
            self.ae = 0.5 * (self.p+self.q)
            self.be = np.sqrt(self.ae*self.ae - self.ce*self.ce)
            self.alpha = np.arccos((4 * self.ce**2 - self.q**2 + self.p**2) /
                                   (4*self.ce*self.p))
            self.delta = 0.5*np.pi - self.alpha - 0.5*self.gamma
            self.pitch = self.pitch - self.delta
            self.pitch0 = self.pitch
            self.pcorrected = 1

    def local_z(self, x, y):
        delta_y = self.p * np.cos(self.alpha) - self.ce
        delta_z = -self.p * np.sin(self.alpha)
        return -self.be * np.sqrt(1 - ((y+delta_y)/self.ae)**2) - delta_z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        delta_y = self.p * np.cos(self.alpha) - self.ce
#        delta_z = -self.p*np.sin(self.alpha)
        a = 0  # -dz/dx
        b = -self.be * (y+delta_y) /\
            (np.sqrt(1 - ((y+delta_y)/self.ae)**2) * self.ae**2)  # -dz/dy
        c = 1.
        norm = (a**2 + b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]


class ParabolicMirror(OE):
    """Implements parabolic mirror. The user supplies the focal distance *p*.
    if *p*>0, the mirror is collimating, otherwise focusing. The figure is a
    parabolic cylinder."""

    cl_plist = ("p", "pp", "delta_y", "delta_z")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
      return -sqrt(2*cl_plist.s1*(y+cl_plist.s2))-cl_plist.s3;
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
      float3 res;
      res.s0 = 0;
      res.s1 = sign(cl_plist.s0) * sqrt(0.5 * cl_plist.s1 / (y+cl_plist.s2));
      res.s2 = 1.;
      return normalize(res);
    }"""

    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.get_orientation()

    def __pop_kwargs(self, **kwargs):
        """ Here 'p' means the distance between the focus and the mirror
        surface, not the parabola parameter"""
        self.p = kwargs.pop('p')
        self.isCylindrical = kwargs.pop('isCylindrical', True)  # always!
        self.pcorrected = 0
        return kwargs

    def get_orientation(self):
        if self.pcorrected and self.pitch0 != self.pitch:
            self.pcorrected = 0
        if not self.pcorrected:
            self.alpha = np.abs(2 * self.pitch)
            self.pp = self.p * (1-np.cos(self.alpha))
            print("Parabola paremeter: " + str(self.pp))
            self.delta_y = 0.5 * self.p * (1+np.cos(self.alpha))
            self.delta_z = -np.abs(self.p) * np.sin(self.alpha)
            self.pitch = 2 * self.pitch
            self.pitch0 = self.pitch
            self.pcorrected = 1

    def local_z(self, x, y):
        return -np.sqrt(2 * self.pp * (y+self.delta_y)) - self.delta_z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # delta_y = 0.5*self.p*(1+np.cos(self.alpha))
        a = 0
        b = np.sign(self.p) * np.sqrt(0.5 * self.pp / (y+self.delta_y))
        c = 1.
        norm = np.sqrt(b**2 + 1)
        return [a/norm, b/norm, c/norm]


class EllipticalMirrorParam(OE):
    """The elliptical mirror is implemented as a parametric surface. The
    parameterization is the following: *s* - is local coordinate along the
    major axis with origin at the ellipse center. *phi* and *r* are local polar
    coordinates in planes normal to the major axis at every point *s*. The
    polar axis is upwards.

    The user supplies the two focal distances *p* and *q* (both are positive)
    and the *pitch* angle. If *isCylindrical* is True, the figure is an
    elliptical cylinder, otherwise it is an ellipsoid of revolution around the
    major axis.

    .. warning::

        If you want to change any of *p*, *q* or *pitch* from outside of the
        constructor, you must invoke the method :meth:`reset_pqpitch` to
        recalculate the ellipsoid parameters."""

    cl_plist = ("ellipseA", "ellipseB", "y0", "z0",
                "cosGamma", "sinGamma", "isCylindrical")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float s, float phi)
    {
        float r = cl_plist.s1 * sqrt(1 - s*s / pown(cl_plist.s0,2));
        if (cl_plist.s6) r /= fabs(cos(phi));
        if (fabs(phi) <= 0.5*PI) r = 1.e20;
        return r;
    }"""
    cl_xyz_param = """
    float3 xyz_to_param(float8 cl_plist, float x, float y, float z)
    {
        float2 xy;
        xy = rotate_x(y - cl_plist.s2, z - cl_plist.s3,
                      cl_plist.s4, cl_plist.s5);
        return (float3)(xy.s0, atan2(x, xy.s1), sqrt(x*x + xy.s1*xy.s1));
    }"""

    def __init__(self, *args, **kwargs):
        """
        *p* and *q*: float
            *p* and *q* arms of the mirror, both are positive.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.isParametric = True
        self.reset_pqpitch(self.p, self.q, self.pitch)

    def reset_pqpitch(self, p=None, q=None, pitch=None):
        """This method allows re-assignment of *p*, *q* and *pitch* from
        outside of the constructor.
        """
        if p is not None:
            self.p = p
        if q is not None:
            self.q = q
        if pitch is not None:
            self.pitch = pitch
        absPitch = abs(self.pitch)
        gamma = np.arctan2((self.p - self.q) * np.sin(absPitch),
                           (self.p + self.q) * np.cos(absPitch))
        self.cosGamma = np.cos(gamma)
        self.sinGamma = np.sin(gamma)
        self.y0 = (self.q - self.p)/2. * np.cos(absPitch)
        self.z0 = (self.q + self.p)/2. * np.sin(absPitch)
        self.ellipseA = (self.q + self.p)/2.
        self.ellipseB = np.sqrt(self.q * self.p) * np.sin(absPitch)

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p', 1000)  # source-to-mirror
        self.q = kwargs.pop('q', 1000)  # mirror-to-focus
        self.isCylindrical = kwargs.pop('isCylindrical', False)
        return kwargs

    def xyz_to_param(self, x, y, z):
        yNew, zNew = raycing.rotate_x(y - self.y0, z - self.z0, self.cosGamma,
                                      self.sinGamma)
        return yNew, np.arctan2(x, zNew), np.sqrt(x**2 + zNew**2)  # s, phi, r

    def param_to_xyz(self, s, phi, r):
        x = r * np.sin(phi)
        y = s
        z = r * np.cos(phi)
        yNew, zNew = raycing.rotate_x(y, z, self.cosGamma, -self.sinGamma)
        return x, yNew + self.y0, zNew + self.z0

    def local_r(self, s, phi):
        r = self.ellipseB * np.sqrt(1 - s**2 / self.ellipseA**2)
        if self.isCylindrical:
            r /= abs(np.cos(phi))
        return np.where(abs(phi) > np.pi/2, r, np.ones_like(phi)*1e20)

    def local_n(self, s, phi):
        A2s2 = self.ellipseA**2 - s**2
        A2s2[A2s2 <= 0] = 1e22
        nr = -self.ellipseB / self.ellipseA * s / np.sqrt(A2s2)
        norm = np.sqrt(nr**2 + 1)
        b = nr / norm
        if self.isCylindrical:
            a = np.zeros_like(phi)
            c = 1. / norm
        else:
            a = -np.sin(phi) / norm
            c = -np.cos(phi) / norm
        bNew, cNew = raycing.rotate_x(b, c, self.cosGamma, -self.sinGamma)
        return [a, bNew, cNew]


class ParabolicalMirrorParam(EllipticalMirrorParam):
    """The parabolical mirror is implemented as a parametric surface. The
    parameterization is the following: *s* - is local coordinate along the
    major axis with origin at the ellipse center. *phi* and *r* are local polar
    coordinates in planes normal to the major axis at every point *s*. The
    polar axis is upwards.

    The user supplies the focal distance *p* or *q* (both are positive)
    and the *pitch* angle. If *isCylindrical* is True, the figure is an
    elliptical cylinder, otherwise it is an ellipsoid of revolution around the
    major axis.

    .. warning::

        If you want to change any of *p*, *q* or *pitch* from outside of the
        constructor, you must invoke the method :meth:`reset_pqpitch` to
        recalculate the ellipsoid parameters."""
    def __init__(self, *args, **kwargs):
        """
        *p* or *q*: float
            *p* and *q* arms of the mirror, both are positive. One and only one
            of them must be given.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.isParametric = True
        self.reset_pqpitch(self.p, self.q, self.pitch)

    def reset_pqpitch(self, p=None, q=None, pitch=None):
        """This method allows re-assignment of *p*, *q* and *pitch* from
        outside of the constructor.
        """
        if p is not None:
            self.p = p
        if q is not None:
            self.q = q
        if ((self.p is not None) and (self.q is not None)) or\
                ((self.p is None) and (self.q is None)):
            print('p={0}, q={1}'.format(self.p, self.q))
            raise ValueError('One and only one of p or q must be None!')
        if pitch is not None:
            self.pitch = pitch
        absPitch = abs(self.pitch)
        if self.p is None:
            self.y0 = self.q * np.cos(absPitch)
            self.z0 = self.q * np.sin(absPitch)
            self.parabParam = -self.q * np.sin(absPitch)**2
            gamma = absPitch
        else:
            self.y0 = -self.p * np.cos(absPitch)
            self.z0 = self.p * np.sin(absPitch)
            self.parabParam = self.p * np.sin(absPitch)**2
            gamma = -absPitch
        self.cosGamma = np.cos(gamma)
        self.sinGamma = np.sin(gamma)

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p', None)  # source-to-mirror
        self.q = kwargs.pop('q', None)  # mirror-to-focus
        self.isCylindrical = kwargs.pop('isCylindrical', False)
        return kwargs

    def local_r(self, s, phi):
        r = 2 * (self.parabParam*s + self.parabParam**2)**0.5
        if self.isCylindrical:
            r /= abs(np.cos(phi))
        return np.where(abs(phi) > np.pi/2, r, np.ones_like(phi)*1e20)

    def local_n(self, s, phi):
        nr = self.parabParam / (self.parabParam*s + self.parabParam**2)**0.5
        norm = np.sqrt(nr**2 + 1)
        b = nr / norm
        if self.isCylindrical:
            a = np.zeros_like(phi)
            c = 1. / norm
        else:
            a = -np.sin(phi) / norm
            c = -np.cos(phi) / norm
        bNew, cNew = raycing.rotate_x(b, c, self.cosGamma, -self.sinGamma)
        return [a, bNew, cNew]


class DCMwithSagittalFocusing(DCM):  # composed by Roelof van Silfhout
    """Creates a DCM with a horizontal focusing 2nd crystal."""

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


class Plate(DCM):
    """Implements a body with two surfaces. Is derived from :class:`DCM`
    because it also has two interfaces but the parameters referring to the 2nd
    crystal should be ignored."""
    def __init__(self, *args, **kwargs):
        """
        *t*: float
            Tthickness in mm.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        DCM.__init__(self, *args, **kwargs)
        self.cryst2perpTransl = -self.t
        if not self.material.kind.startswith("plate"):
            print('Warning: the material of {0} is not of kind "plate"!'.
                  format(self.name))

    def __pop_kwargs(self, **kwargs):
        self.t = kwargs.pop('t', 0)  # difference of z zeros in mm
        return kwargs

    def double_refract(self, beam=None, needLocal=True):
        """
        Returns the refracted beam in global and two local (if *needLocal*
        is true) systems.

        .. Returned values: beamGlobal, beamLocal1, beamLocal2
        """
        self.material2 = self.material
        self.cryst2perpTransl = -self.t
        return self.double_reflect(beam, needLocal, fromVacuum1=True,
                                   fromVacuum2=False)


class ParaboloidFlatLens(Plate):
    """Implements a refractive lens or a stack of lenses (CRL) with one side
    as paraboloid and the other one flat."""

    cl_plist = ("zmax", "focus")
    cl_local_z = """
    float local_z1(float8 cl_plist, float x, float y)
    {
        float res;
        res = 0.25 * (x * x + y * y) / cl_plist.s1;
        if  (res > cl_plist.s0) res = cl_plist.s0;
        return res;
    }

    float local_z2(float8 cl_plist, float x, float y)
    {
      return 0;
    }

    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res=0;
        if (i == 1)
            res = local_z1(cl_plist, x, y);
        if (i == 2)
            res = local_z2(cl_plist, x, y);
        return res;
    }"""
    cl_local_n = """
    float3 local_n1(float8 cl_plist, float x, float y)
    {
        float3 res;
        res.s0 = -x / (2*cl_plist.s1);
        res.s1 = -y / (2*cl_plist.s1);
        float z = (x*x + y*y) / (4*cl_plist.s1);
        if (z > cl_plist.s0)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);
    }

    float3 local_n2(float8 cl_plist, float x, float y)
    {
      return (float3)(0.,0.,1.);
    }

    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res=0;
        if (i == 1)
            res = local_n1(cl_plist, x, y);
        if (i == 2)
            res = local_n2(cl_plist, x, y);
        return res;
    }"""

    def __init__(self, *args, **kwargs):
        u"""
        *focus*: float
            The focal distance of the of paraboloid in mm. The paraboloid is
            then defined by the equation:

            .. math::
                z = (x^2 + y^2) / (4 * focus)

            .. note::

                This is not the focal distance of the lens but of the
                parabola! The former also depends on the refractive index.
                *focus* is only a shape parameter!

        *pitch*: float
            the default value is set to π/2, i.e. to normal incidence.

        *zmax*: float
            If given, limits the *z* coordinate; the object becomes then a
            plate of the thickness *zmax* + *t* with a paraboloid hole at the
            origin.

        *nCRL*: int or tuple (*focalDistance*, *E*)
            If used as CRL (a stack of several lenslets), the number of the
            lenslets nCRL is either given by the user directly or calculated
            for *focalDistance* at energy *E* and then rounded. For
            propagation with *nCRL* > 1 use :meth:`multiple_refract`.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        Plate.__init__(self, *args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        self.focus = kwargs.pop('focus', 1.)
        self.zmax = kwargs.pop('zmax', None)
        self.nCRL = kwargs.pop('nCRL', 1)
        kwargs['pitch'] = kwargs.get('pitch', np.pi/2)
        return kwargs

    def local_z1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        z = (x**2 + y**2) / (4 * self.focus)
        if self.zmax is not None:
            z[z > self.zmax] = self.zmax
        return z

    def local_z2(self, x, y):
        """Determines the surface of OE at (x, y) position."""
        return 0  # just flat

    def local_n1(self, x, y):
        """Determines the normal vector of OE at (x, y) position. If OE is an
        asymmetric crystal, *local_n* must return 2 normals: the 1st one of the
        atomic planes and the 2nd one of the surface."""
        # just flat:
        a = -x / (2*self.focus)  # -dz/dx
        b = -y / (2*self.focus)  # -dz/dy
        if self.zmax is not None:
            z = (x**2 + y**2) / (4*self.focus)
            if isinstance(a, np.ndarray):
                a[z > self.zmax] = 0
            if isinstance(b, np.ndarray):
                b[z > self.zmax] = 0
        c = np.ones_like(x)
        norm = (a**2 + b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]

    def local_n2(self, x, y):
        return self.local_n(x, y)

    def get_nCRL(self, f, E):
        if isinstance(self, DoubleParaboloidLens):
            nFactor = 0.5
        elif isinstance(self, ParabolicCylinderFlatLens):
            nFactor = 2.
        else:
            nFactor = 1.
        return 2 * self.focus / f /\
            (1 - self.material.get_refractive_index(E).real) * nFactor

    def multiple_refract(self, beam, needLocal=False):
        if isinstance(self.nCRL, (int, float)):
            nCRL = self.nCRL
        elif isinstance(self.nCRL, (list, tuple)):
            nCRL = self.get_nCRL(self.nCRL[0], self.nCRL[1])
        else:
            raise ValueError("wrong nCRL value!")
        nCRL = int(round(nCRL))
        if nCRL < 1:
            raise ValueError("wrong nCRL value!")

        if nCRL == 1:
            return self.double_refract(beam, needLocal=needLocal)
        else:
            tempPos = self.center[1]
            beamIn = beam
            for ilens in range(nCRL):
                if isinstance(self, ParabolicCylinderFlatLens):
                    self.roll = -np.pi/4 if ilens % 2 == 0 else np.pi/4
                lglobal, tlocal1, tlocal2 = self.double_refract(
                    beamIn, needLocal=needLocal)
                self.center[1] += self.zmax
                beamIn = lglobal
                if ilens == 0:
                    llocal1, llocal2 = tlocal1, tlocal2
            self.center[1] = tempPos
            return lglobal, llocal1, llocal2


class ParabolicCylinderFlatLens(ParaboloidFlatLens):
    u"""Implements a refractive lens or a stack of lenses (CRL) with one side
    as parabolic cylinder and the other one flat. If used as a CRL, the
    lenslets are arranged such that they alternatively focus in the -45° and
    +45° planes. Therefore the total number of lenslets is doubled as compared
    to ParaboloidFlatLens case."""

    cl_plist = ("zmax", "focus")
    cl_local_z = """
    float local_z1(float8 cl_plist, float x, float y)
    {
        float res;
        res = 0.25 * (x * x + y * y) / cl_plist.s1;
        if  (res > cl_plist.s0) res = cl_plist.s0;
        return res;
    }

    float local_z2(float8 cl_plist, float x, float y)
    {
      return 0;
    }

    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res=0;
        if (i == 1)
            res = local_z1(cl_plist, 0, y);
        if (i == 2)
            res = local_z2(cl_plist, 0, y);
        return res;
    }"""
    cl_local_n = """
    float3 local_n1(float8 cl_plist, float x, float y)
    {
        float3 res;
        res.s0 = -x / (2*cl_plist.s1);
        res.s1 = -y / (2*cl_plist.s1);
        float z = (x*x + y*y) / (4*cl_plist.s1);
        if (z > cl_plist.s0)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);
    }

    float3 local_n2(float8 cl_plist, float x, float y)
    {
      return (float3)(0.,0.,1.);
    }

    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res;
        if (i == 1)
            res = local_n1(cl_plist, 0, y);
        if (i == 2)
            res = local_n2(cl_plist, 0, y);
        return res;
    }"""

    def local_z1(self, x, y):
        return ParaboloidFlatLens.local_z1(self, 0, y)

    def local_n1(self, x, y):
        return ParaboloidFlatLens.local_n1(self, 0, y)


class DoubleParaboloidLens(ParaboloidFlatLens):
    """Implements a refractive lens or a stack of lenses (CRL) with two equal
    paraboloids from both sides."""

    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res;
        res = 0.25 * (x * x + y * y) / cl_plist.s1;
        if (res > cl_plist.s0) res = cl_plist.s0;
        return res;
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res;
        res.s0 = -x / (2*cl_plist.s1);
        res.s1 = -y / (2*cl_plist.s1);
        float z = (x*x + y*y) / (4*cl_plist.s1);
        if (z > cl_plist.s0)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);
    }"""

    def local_z2(self, x, y):
        return self.local_z1(x, y)

    def local_n2(self, x, y):
        return self.local_n1(x, y)

# class ConvConcParaboloidLens(ParaboloidFlatLens):
#    def local_z2(self, x, y):
#        return -self.local_z1(x, y)
#
#    def local_n2(self, x, y):
#        n1 = self.local_n1(x, y)
#        return -n1[0], -n1[1], n1[2]


class SurfaceOfRevolution(OE):
    """Base class for parametric surfaces of revolution. The parameterization
    implements cylindrical coordinates, where *s* is *y* (along the beamline),
    and *phi* and *r* are polar coordinates in planes normal to *s*."""
    def get_surface_limits(self):
        self.isParametric = True
        OE.get_surface_limits(self)

    def xyz_to_param(self, x, y, z):
        return y, np.arctan2(x, z), np.sqrt(x**2 + z**2)  # s, phi, r

    def param_to_xyz(self, s, phi, r):
        return r * np.sin(phi), s, r * np.cos(phi)  # x, y, z


class NormalFZP(OE):
    """Implements a circular Fresnel Zone Plate, as it is described in
    X-Ray Data Booklet, Section 4.4.

    .. warning::

        Do not forget to specify ``kind='FZP'`` in the material!"""
    def __init__(self, *args, **kwargs):
        r"""
        *f*: float
            The focal distance (mm) calculated for waves of energy *E*.

        *E*: float
            Energy (eV) for which *f* is calculated.

        *N*: int
            The number of zones. Is either directly given or calculated from
            *thinnestZone* (mm).

        *thinnestZone*: float
            In mm; can be given to calculate *N*.

        *isCentralZoneBlack*: bool
            if False, the zones are inverted.

        *order*: int or sequence of ints
            Needed diffraction order(s).


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.use_rays_good_gn = True  # use rays_good_gn instead of rays_good

    def __pop_kwargs(self, **kwargs):
        self.f = kwargs.pop('f')
        self.E = kwargs.pop('E')
        self.N = kwargs.pop('N', 1000)
        self.isCentralZoneBlack = kwargs.pop('isCentralZoneBlack', True)
        self.thinnestZone = kwargs.pop('thinnestZone', None)
        self.reset()
        kwargs['limPhysX'] = [-self.rn[-1], self.rn[-1]]
        kwargs['limPhysY'] = [-self.rn[-1], self.rn[-1]]
        return kwargs

    def reset(self):
        lambdaE = CH / self.E * 1e-7
        if self.thinnestZone is not None:
            self.N = lambdaE * self.f / 4. / self.thinnestZone**2
        self.zones = np.arange(self.N+1)
        self.rn = np.sqrt(self.zones*self.f*lambdaE +
                          0.25*(self.zones*lambdaE)**2)
        if _DEBUG:
            print(self.rn)
            print(self.f, self.N)
            print('R(N)={0}, dR(N)={1}'.format(
                  self.rn[-1], self.rn[-1]-self.rn[-2]))
        self.r_to_i = interpolate.interp1d(
            self.rn, self.zones, bounds_error=False, fill_value=0)
        self.i_to_r = interpolate.interp1d(
            self.zones, self.rn, bounds_error=False, fill_value=0)
#        self.drn = self.rn[1:] - self.rn[:-1]

    def rays_good_gn(self, x, y, z):
        """Returns *state* value as inherited from :class:`OE`. The rays that
        fall inside the opaque zones are additionally considered as lost."""
        locState = OE.rays_good(self, x, y)
        r = np.sqrt(x**2 + y**2)
        i = (self.r_to_i(r)).astype(int)
        good = ((i % 2 == int(self.isCentralZoneBlack)) & (r < self.rn[-1]) &
                (locState == 1))
        locState[~good] = self.lostNum
        gz = np.zeros_like(x[good])

        rho = 1./(self.i_to_r(i[good]+1) - self.i_to_r(i[good]-1))
        gx = -x[good] / r[good] * rho
        gy = -y[good] / r[good] * rho
        gn = gx, gy, gz
        return locState, gn


class GeneralFZPin0YZ(OE):
#class GeneralFZPin0YZ(ToroidMirror):
#class GeneralFZPin0YZ(EllipticalMirrorParam):
    """Implements a general Fresnel Zone Plate, where the zones are determined
    by two foci and the surface shape of the OE.

    .. warning::

        Do not forget to specify ``kind='FZP'`` in the material!"""
    def __init__(self, *args, **kwargs):
        """
        *f1* and *f2*: 3- or 4-sequence or str
            The two foci given by 3-sequences representing 3D points in
            _local_ coordinates or 'inf' for infinite position. The 4th member
            in the sequence can be -1 to give the negative sign to the path if
            both foci are on the same side of the FZP.

        *E*: float
            Energy (eV) for which *f* is calculated.

        *N*: int
            The number of zones.

        *grazingAngle*: float
            The angle of the main optical axis to the surface. Defaults to
            self.pitch.

        *phaseShift*: float
            The zones can be phase shifted, which affects the zone structure
            but does not affect the focusing. if *phaseShift* is 0, the central
            zone is at the constructive interference.

        *order*: int or sequence of ints
            Needed diffraction order(s).


        """
        kwargs = self.__pop_kwargs(**kwargs)
        super(GeneralFZPin0YZ, self).__init__(*args, **kwargs)
        self.use_rays_good_gn = True  # use rays_good_gn instead of rays_good
        if self.grazingAngle is None:
            self.grazingAngle = self.pitch
        self.reset()

    def __pop_kwargs(self, **kwargs):
        self.f1 = kwargs.pop('f1')  # in local coordinates!!!
        self.f2 = kwargs.pop('f2')  # in local coordinates!!!
        self.E = kwargs.pop('E')
        self.N = kwargs.pop('N', 1000)
        self.phaseShift = kwargs.pop('phaseShift', 0)
        self.vorticity = kwargs.pop('vorticity', 0)
        self.grazingAngle = kwargs.pop('grazingAngle', None)
        return kwargs

    def reset(self):
        self.lambdaE = CH / self.E * 1e-7
        self.minHalfLambda = None
        self.set_phase_shift(self.phaseShift)

    def set_phase_shift(self, phaseShift):
        self.phaseShift = phaseShift
        if self.phaseShift:
            self.phaseShift /= np.pi

    def rays_good_gn(self, x, y, z):
        locState = super(GeneralFZPin0YZ, self).rays_good(x, y)
        good = locState == 1
        if isinstance(self.f1, str):
            d1 = y[good] * np.cos(self.grazingAngle)
        else:
            d1 = raycing.distance_xyz([x[good], y[good], z[good]], self.f1)
            if len(self.f1) > 3:
                d1 *= self.f1[3]
        if isinstance(self.f2, str):
            d2 = y[good] * np.cos(self.grazingAngle)
        else:
            d2 = raycing.distance_xyz([x[good], y[good], z[good]], self.f2)
            if len(self.f2) > 3:
                d2 *= self.f2[3]
        halfLambda = (d1+d2) / (self.lambdaE/2)
        phi = np.arctan2(y[good]*np.sin(self.grazingAngle), x[good]) / np.pi
        if self.minHalfLambda is None:
            self.minHalfLambda = halfLambda.min()
        halfLambda -= self.minHalfLambda + self.phaseShift - phi*self.vorticity
        zone = np.ones_like(x, dtype=np.int) * (self.N+2)
        zone[good] = np.floor(halfLambda).astype(np.int)
#        N = 0
#        while N < self.N:
#            if (zone == (N+1)).sum() == 0:
#                print("No rays in zone {0}".format(N+1))
#                break
#            N += 1
        N = self.N
        goodN = (zone % 2 == 0) & (zone < N) & good
        badN = ((zone % 2 == 1) | (zone >= N)) & good
        locState[badN] = self.lostNum
#        locState[badN] = 2  # out

        a = np.zeros(N)
        b = np.zeros(N)
        for i in range(1, N+1, 2):
            if (zone == i).sum() == 0:
                continue
            a[i] = max(abs(x[zone == i]))
            b[i] = max(abs(y[zone == i]))
#        print("a", np.diff(a[a > 0]))

        gz = np.zeros_like(x[goodN])
        r = np.sqrt(x[goodN]**2 + y[goodN]**2)
        diva = a[zone[goodN]+1] - a[zone[goodN]-1]
        diva[diva == 0] = 1e20
        divb = b[zone[goodN]+1] - b[zone[goodN]-1]
        divb[divb == 0] = 1e20
        l = (x[goodN]**2/diva + y[goodN]**2/divb) / r**2
        gx = -x[goodN] * l / r
        gy = -y[goodN] * l / r

        gn = gx, gy, gz
        return locState, gn


class BlazedGrating(OE):
    """Implements a grating of triangular shape given by two angles. The front
    side of the triangle (the one looking towards the source) is at *blaze*
    angle to the base plane. The back side is at *antiblaze* angle.

    .. note::

        In contrast to the geometric implementation of the grating diffraction
        when the deflection is calculated by the grating equation, the
        diffraction by :class:`BlazedGrating` is meant to be calculated by the
        wave propagation methods, see :ref:`gallery3`. In those methods, the
        diffraction is not given by the grating equation but by the *surface
        itself* through the calculation of the Kirchhoff integral. Therefore
        the surface material should not have the property ``kind='grating'``
        but rather ``kind='mirror'``.

    A usual optical element (of class :class:`OE`) with such a developed
    surface would have troubles in finding correct intersection points because
    for each ray there are several solutions and we implicitly assume only one.
    The class :class:`BlazedGrating` implements an *ad hoc* method
    :meth:`find_intersection` for selecting the 1st intersection point among
    the several possible ones. The left picture below illustrates the behavior
    of :class:`OE` (the footprint shown by circles is partially in the shadowed
    area). The right picture demonstrates the correct behavior of
    :class:`BlazedGrating` in respect to illumination and shadowing. Notice
    that wave propagation gives the same result for the two cases, apart from a
    small vertical shift. The difference is purely esthetic and caresses our
    perfectionism.

            +-----------------+--------------------+
            |       OE        |   BlazedGrating    |
            +=================+====================+
            | |OE_shadowing|  | |Blazed_shadowing| |
            +-----------------+--------------------+

            .. |OE_shadowing| imagezoom:: _images/1-LEG_profile-default.png
               :scale: 50 %
            .. |Blazed_shadowing| imagezoom:: _images/1-LEG_profile-adhoc.png
               :scale: 50 %
    """

    def __init__(self, *args, **kwargs):
        r"""
        *blaze*, *antiblaze*: float
            Angles in radians

        *rho*: float
            Line density in inverse mm.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.rho_1 = 1. / self.rho
        self.sinBlaze, self.cosBlaze, self.tanBlaze =\
            np.sin(self.blaze), np.cos(self.blaze), np.tan(self.blaze)
        self.sinAntiblaze, self.cosAntiblaze, self.tanAntiblaze =\
            np.sin(self.antiblaze), np.cos(self.antiblaze),\
            np.tan(self.antiblaze)

    def __pop_kwargs(self, **kwargs):
        self.blaze = kwargs.pop('blaze')
        self.antiblaze = kwargs.pop('antiblaze', np.pi*0.4999)
        self.rho = kwargs.pop('rho')
        return kwargs

    def local_z(self, x, y):
        crossingY = self.rho_1 / (1 + self.tanAntiblaze/self.tanBlaze)
        yL = y % self.rho_1
        return np.where(yL > crossingY, (yL - self.rho_1) * self.tanBlaze,
                        -yL * self.tanAntiblaze)

    def local_n(self, x, y):
        crossingY = self.rho_1 / (1 + self.tanAntiblaze/self.tanBlaze)
        yL = y % self.rho_1
        return [np.zeros_like(x),
                np.where(yL > crossingY, -self.sinBlaze, self.sinAntiblaze),
                np.where(yL > crossingY, self.cosBlaze, self.cosAntiblaze)]

    def find_intersection(self, local_f, t1, t2, x, y, z, a, b, c,
                          invertNormal, derivOrder=0):
        b_c = b / c
        n = np.floor((y - b_c*z) / self.rho_1)
        if self.antiblaze == np.pi/2:
            zabl = (self.rho_1 * n - y) / b_c + z
        else:
            zabl = -self.tanAntiblaze * (y - b_c*z - self.rho_1*n) /\
                (1 + self.tanAntiblaze*b_c)
        if self.blaze == np.pi/2:
            zbl = (self.rho_1 * (n+1) - y) / b_c + z
        else:
            zbl = self.tanBlaze * (y - b_c*z - self.rho_1*(n+1)) /\
                (1 - self.tanBlaze*b_c)
        if ((zabl > 0) & (zbl > 0)).any():
            raise
        zabl[zabl > 0] = zbl[zabl > 0] - 1
        zbl[zbl > 0] = zabl[zbl > 0] - 1
        z2 = np.fmax(zabl, zbl)
        z2 = zbl
        y2 = b_c * (z2 - z) + y
        t2 = (y2 - y) / b
        x2 = x + t2 * a
        return t2, x2, y2, z2

    def get_grating_area_fraction(self):
        """This method is used in wave propagation for the calculation of the
        illuminated surface area. It returns the fraction of the longitudinal
        (along *y*) dimension that is illuminated.
        """
        tanPitch = np.tan(abs(self.pitch))
        y1 = self.rho_1 * self.tanBlaze / (self.tanBlaze + tanPitch)
        z1 = -y1 * tanPitch
        y2 = self.rho_1
        z2 = 0
        d = ((y2-y1)**2 + (z2-z1)**2)**0.5
        return d * self.rho


class PlaneGrating(OE):
    """
    Implements a grating of rectangular shape.

    """

    def __init__(self, *args, **kwargs):
        """*blaze* and *antiblaze* are angles in radians. *rho* is the line
        density in inverse mm."""
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.rho_1 = 1. / self.rho  # Period of the grating in [mm]

    def __pop_kwargs(self, **kwargs):
        self.rho = kwargs.pop('rho')
        self.blaze = kwargs.pop('blaze', np.pi*0.4999)
        self.aspect = kwargs.pop('aspect', 0.5)
        self.depth = kwargs.pop('depth', 0.5e-3)
        return kwargs

    def local_z(self, x, y):
        yL = y % self.rho_1
        z = np.array(np.zeros_like(y))
        groove = self.rho_1 * (1.-self.aspect)
        rindex = (yL < groove)
        z[rindex] = -self.depth
        return z

    def local_n(self, x, y):
        yL = y % self.rho_1
        groove = self.rho_1 * (1.-self.aspect)
        norm_x = np.zeros_like(y)
        norm_y = np.zeros_like(y)
        norm_z = np.ones_like(y)
        rindex = (yL == 0)
        norm_y[rindex] = 1
        norm_z[rindex] = 0
        rindex = (yL == groove)
        norm_y[rindex] = -1
        norm_z[rindex] = 0
        return [norm_x, norm_y, norm_z]

    def find_intersection(self, local_f, t1, t2, x, y, z, a, b, c,
                          invertNormal, derivOrder=0):
        # t0 = time.time()
        b_c = b / c
        a_c = a / c
        x2 = np.array(np.zeros_like(y))
        y2 = np.array(np.zeros_like(y))
        z2 = np.array(np.zeros_like(y))
        dy = np.array(np.zeros_like(y))
        dyRel = np.array(np.zeros_like(y))
        y2 = y - z * b_c
        yL = y2 % self.rho_1
        groove = self.rho_1 * (1.-self.aspect)
        x2 = x + z * a_c

        gr = (yL < groove)
        dyRel[gr] = b_c[gr] * self.depth
        dy[gr] = yL[gr] - dyRel[gr]
        bottom = (dy > abs(dyRel)) & (dy < groove-abs(dyRel))
        # bottom = (dy>0) & (dy<groove)
        z2[bottom] = -self.depth
        y2[bottom] += dy[bottom] - yL[bottom]
        x2[bottom] += a_c[bottom] * self.depth

        leftwall = (dy < abs(dyRel))
        # leftwall = (dy < 0)
        z2[leftwall] = (yL[leftwall] / b_c[leftwall])
        y2[leftwall] -= (yL[leftwall])
        x2[leftwall] += (z2[leftwall] * a_c[leftwall])

        rightwall = (dy > groove-abs(dyRel))
        # rightwall = (dy > groove)
        y2[rightwall] += (groove - yL[rightwall])
        z2[rightwall] = (groove - yL[rightwall]) / b_c[rightwall]
        x2[rightwall] += (z2[rightwall] * a_c[rightwall])
        t2 = np.sqrt((x-x2)**2+(y-y2)**2+(z-z2)**2)
        # print "find intersection for", len(y), "rays takes", time.time()-t0
        return t2, x2, y2, z2

    def get_grating_area_fraction(self):
        """
        This method is used in wave propagation for the calculation of the
        illuminated surface area. It returns the fraction of the longitudinal
        (along *y*) dimension that is illuminated.
        """
        return self.aspect


class VLSGrating(OE):
    """
    Implements a grating of rectangular shape with variable period.

    """

    def __init__(self, *args, **kwargs):
        """*coeffs*: list
            Contains the coefficients in the formula defining the period:

            .. math::
                \rho = \rho_0 * (coeffs_0 + 2*coeffs_1*y + 3*coeffs_2*y^2).

            *rho*: float
            The initial line density :math:`\rho_0` in inverse mm."""
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.ticks = []
        p0 = self.limOptY[0]
        while p0 < self.limOptY[1]:
            self.ticks.append(p0)
            p0 += self.__get_period(p0)
        self.ticks = np.array(self.ticks)

    def __get_period(self, coord):
        return 1. / self.rho0 / (self.coeffs[0] + 2. * self.coeffs[1] * -coord +
                                 3. * self.coeffs[2] * coord**2)

    def __pop_kwargs(self, **kwargs):
        self.rho0 = kwargs.pop('rho')
        self.aspect = kwargs.pop('aspect', 0.5)
        self.coeffs = kwargs.pop('coeffs', [1, 0, 0])
        self.depth = kwargs.pop('depth')
        return kwargs

    def local_z(self, x, y):
        z = np.zeros_like(y)
        y0ind = np.searchsorted(self.ticks, y)
        periods = self.ticks[list(y0ind)] - self.ticks[list(y0ind - 1)]
        groove_index = np.where(y - self.ticks[list(y0ind - 1)] < periods *
                                (1. - self.aspect))
        z[groove_index] = -self.depth
        return z

    def local_n(self, x, y):
        y0ind = np.searchsorted(self.ticks[:-1], y)
        periods = self.ticks[list(y0ind)] - self.ticks[list(y0ind - 1)]
        yL = y - self.ticks[list(y0ind - 1)]
        groove = periods * (1. - self.aspect)
        norm_x = np.zeros_like(y)
        norm_y = np.zeros_like(y)
        norm_z = np.ones_like(y)
        rindex = (yL == 0)
        norm_y[rindex] = 1
        norm_z[rindex] = 0
        rindex = (yL == groove)
        norm_y[rindex] = -1
        norm_z[rindex] = 0
        return [norm_x, norm_y, norm_z]

    def find_intersection(self, local_f, t1, t2, x, y, z, a, b, c,
                          invertNormal, derivOrder=0):
        b_c = b / c
        a_c = a / c
        x2 = np.array(np.zeros_like(y))
        y2 = np.array(np.zeros_like(y))
        z2 = np.array(np.zeros_like(y))
        dy = np.array(np.zeros_like(y))
        dyRel = np.array(np.zeros_like(y))
        y2 = y - z * b_c
        y0ind = np.searchsorted(self.ticks, y2)
        periods = self.ticks[list(y0ind)] - self.ticks[list(y0ind - 1)]
        yL = y2 - self.ticks[list(y0ind - 1)]
        groove = periods * (1. - self.aspect)
        x2 = x + z * a_c

        gr = (yL < groove)
        dyRel[gr] = b_c[gr] * self.depth
        dy[gr] = yL[gr] - dyRel[gr]
        bottom = (dy > abs(dyRel)) & (dy < groove-abs(dyRel))
        # bottom = (dy>0) & (dy<groove)
        z2[bottom] = -self.depth
        y2[bottom] += dy[bottom] - yL[bottom]
        x2[bottom] += a_c[bottom] * self.depth

        leftwall = (dy < abs(dyRel))
        # leftwall = (dy < 0)
        z2[leftwall] = (yL[leftwall] / b_c[leftwall])
        y2[leftwall] -= (yL[leftwall])
        x2[leftwall] += (z2[leftwall] * a_c[leftwall])

        rightwall = (dy > groove-abs(dyRel))
        # rightwall = (dy > groove)
        y2[rightwall] += (groove[rightwall] - yL[rightwall])
        z2[rightwall] = (groove[rightwall] - yL[rightwall]) / b_c[rightwall]
        x2[rightwall] += (z2[rightwall] * a_c[rightwall])
        t2 = np.sqrt((x-x2)**2+(y-y2)**2+(z-z2)**2)
        # print "find intersection for", len(y), "rays takes", time.time()-t0
        return t2, x2, y2, z2

    def get_grating_area_fraction(self):
        """
        This method is used in wave propagation for the calculation of the
        illuminated surface area. It returns the fraction of the longitudinal
        (along *y*) dimension that is illuminated.
        """
        return self.aspect
