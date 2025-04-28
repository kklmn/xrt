# -*- coding: utf-8 -*-
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
from .. import raycing
import inspect
import copy

from . import sources as rs
from .physconsts import CHBAR

allArguments = ('bl', 'name', 'center', 'x', 'z', 'compressX',
                'compressZ', 'R', 'phiOffset', 'thetaOffset', 'limPhysX',
                'limPhysY', 'cLimits', 'histShape')

_DEBUG = 20


class Screen(object):
    """Flat screen for beam visualization."""

    def __init__(self, bl=None, name='', center=[0, 0, 0], x='auto', z='auto',
                 compressX=None, compressZ=None, **kwargs):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`
            Container for beamline elements. Optical elements are added to its
            `screens` list.

        *name*: str
            User-specified name, can be used for diagnostics output.

        *center*: tuple of 3 floats
            3D point in the global system.

        *x, z*: 3-sequence or 'auto'.
            Normalized 3D vectors in the global system which determine the
            local x and z axes lying in the screen plane. If *x* is 'auto', it
            is horizontal and perpendicular to the beam line. If *z* is 'auto',
            it is vertical. Both *x* and *z* can also be set as instance
            attributes.

        *compressX, compressZ*: float
            Multiplicative compression coefficients for the corresponding axes.
            Typically are not needed. Can be useful to account for the viewing
            camera magnification or when the camera sees the screen at an
            angle.

        *limPhysX* and *limPhysY*: [*min*, *max*] where *min*, *max* are
            floats or sequences of floats (optional)
            Physical dimension = local coordinate of the corresponding edge.
            Can be given by sequences of the length of *surface*. You do not
            have to provide the limits, although they may help in finding
            intersection points, especially for (strongly) curved surfaces.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.screens:
                bl.screens.append(self)
                self.ordinalNum = len(bl.screens)
                self.lostNum = -self.ordinalNum - 2000
        raycing.set_name(self, name)
        self._x = x
        self._z = z
        self.footprint = []
        self._set_orientation()

        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())
                
        _ = self.__pop_kwargs(**kwargs)
        if np.sum(self.limPhysX) > 0:
            self.image = np.zeros(np.int32(self.histShape))

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.uuid] = [self, 1]

        self.center = center
#        if any([coord == 'auto' for coord in self.center]):
#            self._center = copy.copy(self.center)
        self.compressX = compressX
        self.compressZ = compressZ

    def __pop_kwargs(self, **kwargs):
        self.limPhysX = kwargs.pop('limPhysX', None)
        self.limPhysY = kwargs.pop('limPhysY', None)
        self.cLimits = kwargs.pop('cLimits', None)
        self.histShape = kwargs.pop('histShape', [256, 256])
#        print(self.name, self.limPhysX, self.limPhysY)

    @property
    def limPhysX(self):
        return self._limPhysX

    @limPhysX.setter
    def limPhysX(self, limPhysX):
        if limPhysX is None:
            self._limPhysX = raycing.Limits([0, 0])
        else:
            self._limPhysX = raycing.Limits(limPhysX)

    @property
    def limPhysY(self):
        return self._limPhysY

    @limPhysY.setter
    def limPhysY(self, limPhysY):
        if limPhysY is None:
            self._limPhysY = raycing.Limits([0, 0])
        else:
            self._limPhysY = raycing.Limits(limPhysY)

    @property
    def cLimits(self):
        return self._cLimits

    @cLimits.setter
    def cLimits(self, cLimits):
        if cLimits is None:
            self._cLimits = raycing.Limits([0, 0])
        else:
            self._cLimits = raycing.Limits(cLimits)

    @property
    def histShape(self):
        return self._histShape

    @histShape.setter
    def histShape(self, histShape):
        if histShape is None:
            self._histShape = raycing.Image2D([256, 256])
        else:
            self._histShape = raycing.Image2D(histShape)

    @property
    def center(self):
        return self._center if self._centerVal is None else self._centerVal

    @center.setter
    def center(self, center):
        if any([x == 'auto' for x in center]):
            self._center = copy.copy(center)
            self._centerVal = None
            self._centerInit = center
        else:
            self._centerVal = raycing.Center(center)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = copy.copy(x)
        self._set_orientation()
#        self.update_orientation_quaternion()

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z):
        self._z = copy.copy(z)
        self._set_orientation()
#        self.update_orientation_quaternion()

    def _set_orientation(self):
        """Determines the local x, y and z in the global system."""
        if isinstance(self._x, raycing.basestring):
            self._x = None
        if isinstance(self._z, raycing.basestring):
            self._z = None
        self._x, self.y, self._z = raycing.xyz_from_xz(self, self._x, self._z)

    def set_orientation(self, x=None, z=None):
        """Compatibility method. All calculations moved to setters."""
        self._x = copy.copy(x)
        self._z = copy.copy(z)
        self._set_orientation()

    def local_to_global(self, x=0, y=0, z=0):
        xglo = self.center[0] + x*self.x[0] + y*self.y[0] + z*self.z[0]
        yglo = self.center[1] + x*self.x[1] + y*self.y[1] + z*self.z[1]
        zglo = self.center[2] + x*self.x[2] + y*self.y[2] + z*self.z[2]
        return xglo, yglo, zglo

    def expose_global(self, beam=None, onlyPositivePath=False):
        kwArgsIn = {}
        if self.bl is not None:
            if raycing.is_valid_uuid(beam):
                kwArgsIn['beam'] = beam
                beam = self.bl.beamsDictU[beam]['beamGlobal']
            else:
                kwArgsIn['beam'] = beam.parentId
            self.bl.auto_align(self, beam)
        glo = rs.Beam(copyFrom=beam)  # global
        with np.errstate(divide='ignore'):
            path = ((self.center[0]-beam.x) * self.y[0] +
                    (self.center[1]-beam.y) * self.y[1] +
                    (self.center[2]-beam.z) * self.y[2]) /\
                   (beam.a*self.y[0] + beam.b*self.y[1] + beam.c*self.y[2])

        condBad = np.isnan(path) | np.isinf(path)
        if onlyPositivePath:
            condBad = condBad | (path < 0)
        indBad = np.where(condBad)
        path[indBad] = 0.
        glo.path += path
        glo.state[indBad] = self.lostNum

        glo.x[:] = beam.x + path*beam.a
        glo.y[:] = beam.y + path*beam.b
        glo.z[:] = beam.z + path*beam.c
        return glo

    @raycing.append_to_flow_decorator
    def expose(self, beam=None, onlyPositivePath=False, withHistogram=False):
        """Exposes the screen to the beam. *beam* is in global system, the
        returned beam is in local system of the screen and represents the
        desired image.


        .. .. Returned values: beamLocal
        """
#        kwArgsIn = {'onlyPositivePath': onlyPositivePath,
#                    'withHistogram': withHistogram}
#        if self.bl is not None:
#            if raycing.is_valid_uuid(beam):
#                kwArgsIn['beam'] = beam
#                beam = self.bl.beamsDictU[beam]['beamGlobal']
#            else:
#                kwArgsIn['beam'] = beam.parentId
#            self.bl.auto_align(self, beam)
        blo = rs.Beam(copyFrom=beam, withNumberOfReflections=True)  # local
        xyz = self.x, self.y, self.z
        raycing.global_to_virgin_local(xyz, beam, blo, self.center)

        with np.errstate(divide='ignore'):
            path = -blo.y / blo.b
        condBad = np.isnan(path) | np.isinf(path)
        if onlyPositivePath:
            condBad = condBad | (path < 0)
        indBad = np.where(condBad)
        path[indBad] = 0.
        blo.state[indBad] = self.lostNum

        blo.path += path
        blo.x[:] += blo.a * path
        blo.z[:] += blo.c * path
        blo.y[:] = 0.

        if hasattr(blo, 'Es'):
            propPhase = np.exp(1e7j * (blo.E/CHBAR) * path)
            blo.Es *= propPhase
            blo.Ep *= propPhase
        # Screen size hint for Glow
        good = (blo.state == 1) | (blo.state == 2)
        self.footprint = []
        if len(blo.state[good]) > 0:
            self.footprint.extend([np.hstack((np.min(np.vstack((
                blo.x[good], blo.y[good], blo.z[good])), axis=1),
                np.max(np.vstack((blo.x[good], blo.y[good], blo.z[good])),
                       axis=1))).reshape(2, 3)])

        if self.compressX:
            blo.x[:] *= self.compressX
        if self.compressZ:
            blo.z[:] *= self.compressZ
        raycing.append_to_flow(self.expose, [blo], inspect.currentframe())
        if withHistogram:
#            print(self.limPhysX, self.limPhysY)
            if any([np.sum(np.abs(x)) == 0 for x in [self.limPhysX, self.limPhysY]]):
                print("Using auto limits for histogramming")
                self.limPhysX = raycing.Limits(self.footprint[-1][:, 0].tolist())
                self.limPhysY = raycing.Limits(self.footprint[-1][:, 2].tolist())
#            print(self.limPhysX, self.limPhysY)

            limitsIn = [self.limPhysX if isinstance(self.limPhysX, list) else
                        self.limPhysX.tolist(),
                        self.limPhysY if isinstance(self.limPhysY, list) else
                        self.limPhysY.tolist()]
#            print(limitsIn, self.histShape)
            hist2d, hist2dRGB, limitsOut = raycing.build_hist(
                    blo, limits=limitsIn, isScreen=True, shape=self.histShape,
                    cDataFunc=None, cLimits=None)
            self.image = hist2d
#            print(np.sum(hist2d))
#        blo.parentId = self.uuid
#        self.bl.flowU[self.uuid] = {'method': self.expose,
#                                    'kwArgsIn': kwArgsIn}
#        self.bl.beamsDictU[self.uuid] = {'beamLocal': blo}

        return blo

    def prepare_wave(self, prevOE, dim1, dim2, dy=0, rw=None, condition=None):
        """Creates the beam arrays used in wave diffraction calculations.
        *prevOE* is the diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`.
        *dim1* and *dim2* are *x* and *z* arrays for a flat screen or
        *phi* and *theta* arrays for a hemispheric screen. The two arrays are
        generally of different 1D shapes. They are used to create a 2D mesh by
        ``meshgrid``.

        *condition*: a callable defined in the user script with two flattened
            meshgrid arrays as inputs and outputs. Can be used to select wave
            samples. An example:

            .. code-block:: python

                def condition(d1s, d2s):
                    cond = d1s**2 + d2s**2 <= pinholeDia**2 / 4  # in a pinhole
                    return d1s[cond], d2s[cond]

        """
        if rw is None:
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
        if condition is not None:
            d1s, d2s = condition(d1s, d2s)
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

    def expose_wave(self, wave=None, beam=None, dim1=0, dim2=0):
        """
        Propagates the incoming *wave* through an aperture using the
        Kirchhoff diffraction theorem. Returned global and local beams can be
        used correspondingly for the consequent ray and wave propagation
        calculations.

        *wave*: Beam object
            Local beam on the surface of the previous optical element.

        *beam*: Beam object
            Incident global beam, only used for alignment purpose.

        *nrays*: 'auto' or int
            Dimension of the created wave. If 'auto' - the same as the incoming
            wave.


        .. Returned values: beamLocal
        """
        from . import waves as rw
        prevOE = self.bl.oesDict[wave.parentId]
        if self.bl is not None:
            if beam is not None:
                self.bl.auto_align(self, beam)
            elif 'source' in str(type(prevOE)):
                self.bl.auto_align(self, wave)
            else:
                self.bl.auto_align(self, prevOE.local_to_global(
                    wave, returnBeam=True))

        if isinstance(dim1, int) or isinstance(dim2, int):
            if beam is None:
                if isinstance(prevOE, raycing.oes.DCM):
                    locBeam = self.expose(prevOE.local_to_global(
                        wave, returnBeam=True, is2ndXtal=True))
                else:
                    locBeam = self.expose(prevOE.local_to_global
                                          (wave, returnBeam=True))
            else:
                locBeam = self.expose(beam)
            if isinstance(dim1, int):
                dim1 = np.linspace(np.min(locBeam.x), np.max(locBeam.x), dim1)
            if isinstance(dim2, int):
                dim2 = np.linspace(np.min(locBeam.z), np.max(locBeam.z), dim2)
#        print(dim1, dim2)
        waveOnSelf = self.prepare_wave(prevOE, dim1, dim2, rw=rw)
        if 'source' in str(type(prevOE)):
            prevOE.shine(wave=waveOnSelf)
        else:
            rw.diffract(wave, waveOnSelf)
        waveOnSelf.parentId = self.uuid
        return waveOnSelf


class HemisphericScreen(Screen):
    """Hemispheric screen for beam visualization."""

    def __init__(self, bl=None, name='', center=[0, 0, 0], R=1000.,
                 x='auto', z='auto', phiOffset=0, thetaOffset=0):
        u"""
        *x, z*: 3-tuples or 'auto'. Normalized 3D vectors in the global system
            which determine the local x and z axes of the hemispheric screen.
            If *x* (the origin of azimuthal angle φ) is 'auto', it coincides
            with the beamline's *y*; if *z* (the polar axis) is 'auto', it is
            coincides with the beamline's *x*. The equator plane is then
            vertical. The polar angle θ is counted from -π/2 to π/2 with 0 at
            the equator and π/2 at the polar axis direction. Both *x* and *z*
            can also be set as instance attributes.

        *R*: float
            Radius of the hemisphere in mm.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.screens:
                bl.screens.append(self)
                self.ordinalNum = len(bl.screens)
                self.lostNum = -self.ordinalNum - 2000
        raycing.set_name(self, name)
        self._x = x
        self._z = z
        self._set_orientation()

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.uuid] = [self, 1]

        self.center = center
#        if any([coord == 'auto' for coord in self.center]):
#            self._center = copy.copy(self.center)
        self.R = R
        self.phiOffset = phiOffset
        self.thetaOffset = thetaOffset

    def _set_orientation(self):
        """Determines the local x, y and z in the global system."""
        if isinstance(self._x, (list, tuple, np.ndarray)):
            norm = sum([xc**2 for xc in self._x])**0.5
            self._x = [xc/norm for xc in self._x]
        else:
            self._x = self.bl.sinAzimuth, self.bl.cosAzimuth, 0.

        if isinstance(self._z, (list, tuple, np.ndarray)):
            norm = sum([zc**2 for zc in self._z])**0.5
            self._z = [zc/norm for zc in self._z]
        else:
            self._z = self.bl.cosAzimuth, -self.bl.sinAzimuth, 0.

        xdotz = np.dot(self._x, self._z)
        if abs(xdotz) > 1e-8:
            raycing.colorPrint('x and z must be orthogonal, got xz={0:.4e}'
                               .format(xdotz), 'RED')
        self.y = np.cross(self._z, self._x)

    def local_to_global(self, phi, theta):
        thetaO = theta + self.thetaOffset
        phiO = phi + self.phiOffset
        z = np.sin(thetaO) * self.R
        y = np.cos(thetaO) * np.sin(phiO) * self.R
        x = np.cos(thetaO) * np.cos(phiO) * self.R
        xglo, yglo, zglo = Screen.local_to_global(self, x, y, z)
        return x, y, z, xglo, yglo, zglo

    def expose_global(self, beam=None):
        kwArgsIn = {}
        if self.bl is not None:
            if raycing.is_valid_uuid(beam):
                kwArgsIn['beam'] = beam
                beam = self.bl.beamsDictU[beam]['beamGlobal']
            else:
                kwArgsIn['beam'] = beam.parentId
            self.bl.auto_align(self, beam)
        glo = self.expose(beam)
        _, _, _, glo.x[:], glo.y[:], glo.z[:] = \
            self.local_to_global(glo.phi, glo.theta)
        return glo

    @raycing.append_to_flow_decorator
    def expose(self, beam=None, onlyPositivePath=False):
        """Exposes the screen to the beam. *beam* is in global system, the
        returned beam is in local system of the screen and represents the
        desired image.


        .. Returned values: beamLocal
        """
#        kwArgsIn = {'onlyPositivePath': onlyPositivePath}
#        if self.bl is not None:
#            if raycing.is_valid_uuid(beam):
#                kwArgsIn['beam'] = beam
#                beam = self.bl.beamsDictU[beam]['beamGlobal']
#            else:
#                kwArgsIn['beam'] = beam.parentId
#            self.bl.auto_align(self, beam)
        blo = rs.Beam(copyFrom=beam, withNumberOfReflections=True)  # local
        sqb_2 = (beam.a * (beam.x-self.center[0]) +
                 beam.b * (beam.y-self.center[1]) +
                 beam.c * (beam.z-self.center[2]))
        sqc = ((beam.x-self.center[0])**2 +
               (beam.y-self.center[1])**2 +
               (beam.z-self.center[2])**2 - self.R**2)
        with np.errstate(invalid='ignore'):
            path = -sqb_2 + (sqb_2**2 - sqc)**0.5

        condBad = np.isnan(path) | np.isinf(path)
        if onlyPositivePath:
            condBad = condBad | (path < 0)
        indBad = np.where(condBad)
        path[indBad] = 0.
        blo.state[indBad] = self.lostNum

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
        raycing.append_to_flow(self.expose, [blo],
                               inspect.currentframe())
#        blo.parentId = self.uuid
#        self.bl.flowU[self.uuid] = {'method': self.expose,
#                                    'kwArgsIn': kwArgsIn}
#        self.bl.beamsDictU[self.uuid] = {'beamLocal': blo}

        return blo
