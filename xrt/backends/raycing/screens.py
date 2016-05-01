﻿# -*- coding: utf-8 -*-
r"""
Screens
-------

Module :mod:`~xrt.backends.raycing.screens` defines a flat screen and a
hemispheric screen that intercept a beam and give its image.

.. autoclass:: xrt.backends.raycing.screens.Screen()
   :members: __init__, expose, prepare_wave

.. autoclass:: xrt.backends.raycing.screens.HemisphericScreen()
   :members: __init__

"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "26 Mar 2016"
__all__ = 'Screen', 'HemisphericScreen'
import numpy as np

from . import sources as rs
from .physconsts import CHBAR

_DEBUG = 20


class Screen(object):
    """Flat screen for beam visualization.
    """
    def __init__(self, bl=None, name='', center=[0, 0, 0], x='auto', z='auto',
                 compressX=None, compressZ=None):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`
            Container for beamline elements. Optical elements are added to its
            `screens` list.

        *name*: str
            User-specified name, can be used for diagnostics output.

        *center*: tuple of 3 floats
            3D point in the global system.

        *x, z*: 3-tuples or 'auto'.
            Normalized 3D vectors in the global system which determine the
            local x and z axes lying in the screen plane. If *x* is 'auto', it
            is horizontal and perpendicular to the beam line. If *z* is 'auto',
            it is vertical.

        *compressX, compressZ*: float
            Multiplicative compression coefficients for the corresponding axes.
            Typically are not needed. Can be useful to account for the viewing
            camera magnification or when the camera sees the screen at an
            angle.


        """
        self.name = name
        self.bl = bl
        if bl is not None:
            bl.screens.append(self)
            self.set_orientation(x, z)
        self.ordinalNum = len(bl.screens)
        self.center = center
        self.compressX = compressX
        self.compressZ = compressZ

    def set_orientation(self, x=None, z=None):
        """Determines the local x, y and z in the global system."""
        if x == 'auto':
            self.x = self.bl.cosAzimuth, -self.bl.sinAzimuth, 0.
        elif x is not None:
            self.x = x
        if z == 'auto':
            self.z = 0., 0., 1.
        elif z is not None:
            self.z = z
        xdotz = np.dot(self.x, self.z)
        assert abs(xdotz) < 1e-14, 'x and z must be orthogonal!'
        self.y = np.cross(self.z, self.x)

    def local_to_global(self, x=0, y=0, z=0):
        xglo = self.center[0] + x*self.x[0] + y*self.y[0] + z*self.z[0]
        yglo = self.center[1] + x*self.x[1] + y*self.y[1] + z*self.z[1]
        zglo = self.center[2] + x*self.x[2] + y*self.y[2] + z*self.z[2]
        return xglo, yglo, zglo

    def expose(self, beam=None):
        """Exposes the screen to the beam. *beam* is in global system, the
        returned beam is in local system of the screen and represents the
        desired image.

        .. .. Returned values: beamLocal
        """
        blo = rs.Beam(copyFrom=beam, withNumberOfReflections=True)  # local
        # Converting the beam to the screen local coordinates
        blo.x[:] = beam.x[:] - self.center[0]
        blo.y[:] = beam.y[:] - self.center[1]
        blo.z[:] = beam.z[:] - self.center[2]

        xyz = blo.x, blo.y, blo.z
        blo.x[:], blo.y[:], blo.z[:] = \
            sum(c*b for c, b in zip(self.x, xyz)),\
            sum(c*b for c, b in zip(self.y, xyz)),\
            sum(c*b for c, b in zip(self.z, xyz))
        abc = beam.a, beam.b, beam.c
        blo.a[:], blo.b[:], blo.c[:] = \
            sum(c*b for c, b in zip(self.x, abc)),\
            sum(c*b for c, b in zip(self.y, abc)),\
            sum(c*b for c, b in zip(self.z, abc))

        with np.errstate(divide='ignore'):
            path = -blo.y / blo.b
        indBad = np.where(np.isnan(path))
        path[indBad] = 0.
        indBad = np.where(np.isinf(path))
        path[indBad] = 0.
        blo.path += path
        blo.x[:] += blo.a * path
        blo.z[:] += blo.c * path
        blo.y[:] = 0.

        if hasattr(blo, 'Es'):
            propPhase = np.exp(1e7j * (blo.E/CHBAR) * path)
            blo.Es *= propPhase
            blo.Ep *= propPhase

        if self.compressX:
            blo.x[:] *= self.compressX
        if self.compressZ:
            blo.z[:] *= self.compressZ
        return blo

    def prepare_wave(self, prevOE, dim1, dim2, dy=0):
        """Creates the beam arrays used in wave diffraction calculations.
        *prevOE* is the diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`.
        *dim1* and *dim2* are *x* and *z* arrays for a flat screen or
        *phi* and *theta* arrays for a hemispheric screen. The two arrays are
        generally of different 1D shapes. They are used to create a 2D mesh by
        ``meshgrid``.
        """
        from . import waves as rw

        d1s, d2s = np.meshgrid(dim1, dim2)
        d1s = d1s.flatten()
        d2s = d2s.flatten()
        if hasattr(dim1, '__getitem__') and hasattr(dim2, '__getitem__'):
            try:
                dS = (dim1[1] - dim1[0]) * (dim2[1] - dim2[0])
            except IndexError:
                dS = 1.
        else:
            dS = 1.
        nrays = len(d1s)

        if isinstance(self, HemisphericScreen):
            xlo, ylo, zlo, xglo, yglo, zglo = self.local_to_global(
                phi=d1s, theta=d2s)
        else:
            xglo, yglo, zglo = self.local_to_global(x=d1s, z=d2s)

        wave = rs.Beam(nrays=nrays, forceState=1, withAmplitudes=True)
        if isinstance(self, HemisphericScreen):
            wave.x[:] = xlo
            wave.y[:] = ylo + dy
            wave.z[:] = zlo
            wave.phi = d1s
            wave.theta = d2s
            dS *= np.abs(np.cos(wave.theta)) * self.R**2
        else:
            wave.x[:] = d1s
            wave.y[:] = np.zeros_like(d1s) + dy
            wave.z[:] = d2s
        wave.dS = dS
        wave.toOE = self
        wave.area = (np.ones_like(d1s) * dS).sum()
        return rw.prepare_wave(prevOE, wave, xglo, yglo+dy, zglo)


class HemisphericScreen(Screen):
    """Hemispheric screen for beam visualization.
    """
    def __init__(self, bl=None, name='', center=[0, 0, 0], R=1000.,
                 x='auto', z='auto', phiOffset=0, thetaOffset=0):
        u"""
        *x, z*: 3-tuples or 'auto'. Normalized 3D vectors in the global system
            which determine the local x and z axes of the hemispheric screen.
            If *x* (the origin of azimuthal angle φ) is 'auto', it coincides
            with the beamline's *y*; if *z* (the polar axis) is 'auto', it is
            coincides with the beamline's *x*. The equator plane is then
            vertical. The polar angle θ is counted from -π/2 to π/2 with 0 at
            the equator and π/2 at the polar axis direction.

        *R*: float
            Radius of the hemisphere in mm.


        """
        self.name = name
        self.bl = bl
        if bl is not None:
            bl.screens.append(self)
            self.ordinalNum = len(bl.screens)
            self.set_orientation(x, z)
        self.center = center
        self.R = R
        self.phiOffset = phiOffset
        self.thetaOffset = thetaOffset

    def set_orientation(self, x=None, z=None):
        """Determines the local x, y and z in the global system."""
        if x == 'auto':
            self.x = self.bl.sinAzimuth, self.bl.cosAzimuth, 0.
        elif x is not None:
            self.x = x
        if z == 'auto':
            self.z = self.bl.cosAzimuth, -self.bl.sinAzimuth, 0.
        elif z is not None:
            self.z = z
        assert np.dot(self.x, self.z) == 0, 'x and z must be orthogonal!'
        self.y = np.cross(self.z, self.x)

    def local_to_global(self, phi, theta):
        thetaO = theta + self.thetaOffset
        phiO = phi + self.phiOffset
        z = np.sin(thetaO) * self.R
        y = np.cos(thetaO) * np.sin(phiO) * self.R
        x = np.cos(thetaO) * np.cos(phiO) * self.R
        xglo, yglo, zglo = Screen.local_to_global(self, x, y, z)
        return x, y, z, xglo, yglo, zglo

    def expose(self, beam=None):
            """Exposes the screen to the beam. *beam* is in global system, the
            returned beam is in local system of the screen and represents the
            desired image.

            .. Returned values: beamLocal
            """
            blo = rs.Beam(copyFrom=beam, withNumberOfReflections=True)  # local
            sqb_2 = (beam.a * (beam.x-self.center[0]) +
                     beam.b * (beam.y-self.center[1]) +
                     beam.c * (beam.z-self.center[2]))
            sqc = ((beam.x-self.center[0])**2 +
                   (beam.y-self.center[1])**2 +
                   (beam.z-self.center[2])**2 - self.R**2)
            path = -sqb_2 + (sqb_2**2 - sqc)**0.5
            blo.path += path
            rx = beam.x + beam.a*path - self.center[0]
            ry = beam.y + beam.b*path - self.center[1]
            rz = beam.z + beam.c*path - self.center[2]
            blo.z = rx*self.z[0] + ry*self.z[1] + rz*self.z[2]
            blo.y = rx*self.y[0] + ry*self.y[1] + rz*self.y[2]
            blo.x = rx*self.x[0] + ry*self.x[1] + rz*self.x[2]
            blo.theta = np.arcsin(blo.z / self.R) - self.thetaOffset
            blo.phi = np.arctan2(blo.y, blo.x) - self.phiOffset
            if hasattr(blo, 'Es'):
                propPhase = np.exp(1e7j * (blo.E / CHBAR) * path)
                blo.Es *= propPhase
                blo.Ep *= propPhase
            return blo
