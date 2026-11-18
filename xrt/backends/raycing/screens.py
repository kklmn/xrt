# -*- coding: utf-8 -*-
r"""
Screens
-------

Module :mod:`~xrt.backends.raycing.screens` defines a flat screen and a
hemispheric screen that intercept a beam and give its image.

.. autosummary::
   :signatures: none

   Screen
   HemisphericScreen

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
from .stages import Pose, set_mount, set_mount_offset

allArguments = ('bl', 'name', 'center', 'x', 'z', 'compressX',
                'compressZ', 'R', 'phiOffset', 'thetaOffset', 'limPhysX',
                'limPhysY', 'mount', 'mountOffset')

_DEBUG = 20


class Screen(object):
    """Flat screen for beam visualization."""

    def __init__(self, bl=None, name='', center=[0, 0, 0], x='auto', z='auto',
                 compressX=None, compressZ=None, mount=None,
                 mountOffset=None, **kwargs):
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

        *mount*: :class:`~xrt.backends.raycing.stages.MountingFrame`, str or None
            Frame or stage carrying the screen. While mounted, its center
            and axes are readouts of the composed world pose.

        *mountOffset*: :class:`~xrt.backends.raycing.stages.Pose` or None
            Screen origin and orientation in the parent attachment frame.
            Defaults to identity on attachment.

        *compressX, compressZ*: float
            Multiplicative compression coefficients for the corresponding axes.
            Typically are not needed. Can be useful to account for the viewing
            camera magnification or when the camera sees the screen at an
            angle.

        *limPhysX* and *limPhysY*: [*min*, *max*] where *min*, *max* are floats
            Physical dimensions in mm along the local x and z axes. Used for
            the visible screen size in Glow.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.screens:
                bl.screens.append(self)
                self.ordinalNum = len(bl.screens)
                self.lostNum = -self.ordinalNum - 2000
        raycing.set_name(self, name)
        self.footprint = []
        self._mount = None
        self.set_orientation(x, z)

        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        if bl is not None:
            if self.bl.flowSource != 'Qook0':
                bl.oesDict[self.uuid] = [self, 1]
                bl.oenamesToUUIDs[self.name] = self.uuid

        self.center = center
        set_mount_offset(self, mountOffset, strict=True)
        set_mount(self, mount, strict=True)
        self.compressX = compressX
        self.compressZ = compressZ

        self.limPhysX = kwargs.pop('limPhysX', None)
        self.limPhysY = kwargs.pop('limPhysY', None)

    _unmounted_center = raycing.center_property()

    @property
    def center(self):
        if self.mount is not None:
            return tuple(self.globalPose.center)
        return self._unmounted_center

    @center.setter
    def center(self, value):
        if self.mount is None:
            self._unmounted_center = value

    @property
    def mount(self):
        """Stage carrying this screen, or None for independent placement."""
        return getattr(self, '_mount', None)

    @mount.setter
    def mount(self, value):
        set_mount(self, value)

    @property
    def mountOffset(self):
        """Fixed screen pose relative to the stage's moving attachment."""
        return getattr(self, '_mountOffset', None)

    @mountOffset.setter
    def mountOffset(self, value):
        set_mount_offset(self, value)

    @property
    def globalPose(self):
        """World pose of the screen origin and its local axes."""
        if self.mount is not None:
            return self.mount.mountPose @ self.mountOffset
        return Pose.from_xz(self.center, x=self.x, z=self.z)

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
    def x(self):
        if self.mount is not None:
            return self.globalPose.orientation.x
        return self._x

    @x.setter
    def x(self, x):
        if self.mount is not None:
            return
        self._xInit = copy.copy(x)
        self._x = copy.copy(x)
        self._set_orientation()
#        self.update_orientation_quaternion()

    @property
    def y(self):
        if self.mount is not None:
            return self.globalPose.orientation.matrix[:, 1].tolist()
        return self._y

    @y.setter
    def y(self, value):
        if self.mount is None:
            self._y = value

    @property
    def z(self):
        if self.mount is not None:
            return self.globalPose.orientation.z
        return self._z

    @z.setter
    def z(self, z):
        if self.mount is not None:
            return
        self._zInit = copy.copy(z)
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
        """Set both input axes before resolving the screen orientation."""
        if self.mount is not None:
            return
        self._xInit = copy.copy(x)
        self._zInit = copy.copy(z)
        self._x = copy.copy(x)
        self._z = copy.copy(z)
        self._set_orientation()

    def _ray_frame(self):
        if self.mount is not None:
            pose = self.globalPose
            return pose.center, pose.orientation.matrix.T.tolist()
        return self.center, (self.x, self.y, self.z)

    def local_to_global(self, x=0, y=0, z=0, **kwargs):
        center, (x_axis, y_axis, z_axis) = self._ray_frame()
        xglo = center[0] + x*x_axis[0] + y*y_axis[0] + z*z_axis[0]
        yglo = center[1] + x*x_axis[1] + y*y_axis[1] + z*z_axis[1]
        zglo = center[2] + x*x_axis[2] + y*y_axis[2] + z*z_axis[2]
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
        center, (_, normal, _) = self._ray_frame()
        nx, ny, nz = normal
        with np.errstate(divide='ignore'):
            path = ((center[0]-beam.x) * nx +
                    (center[1]-beam.y) * ny +
                    (center[2]-beam.z) * nz) / (
                        beam.a*nx + beam.b*ny + beam.c*nz)

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
    def expose(self, beam=None, onlyPositivePath=False):
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
        center, xyz = self._ray_frame()
        raycing.global_to_virgin_local(xyz, beam, blo, center)

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
            xlo, ylo, zlo, xglo, yglo, zglo = self.local_to_global_sph(
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
        wave.parentId = prevOE.uuid
        if self.mount is not None and dy != 0:
            nx, ny, nz = self.y
            return rw.prepare_wave(prevOE, wave, xglo+dy*nx, yglo+dy*ny, zglo+dy*nz)
        return rw.prepare_wave(prevOE, wave, xglo, yglo+dy, zglo)

    def expose_wave(self, wave=None, beam=None, dim1=0, dim2=0):
        """
        Propagates the incoming *wave* to the screen using the Kirchhoff
        diffraction theorem. Returns a local wave.

        *wave*: Beam object
            Local beam on the surface of the previous optical element.

        *beam*: Beam object
            Incident global beam, only used for alignment purpose.

        *dim1* and *dim2* are *x* and *z* 1D arrays or their sizes as ints.

        .. Returned values: local wave
        """
        from . import waves as rw
        prevOE = self.bl.oesDict[wave.parentId][0]
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
                 x='auto', z='auto', phiOffset=0, thetaOffset=0, **kwargs):
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

        *mount, mountOffset*:
            Stage and relative pose, as on :class:`Screen`.


        """
#        self.bl = bl
#        if bl is not None:
#            if self not in bl.screens:
#                bl.screens.append(self)
#                self.ordinalNum = len(bl.screens)
#                self.lostNum = -self.ordinalNum - 2000
#        raycing.set_name(self, name)
#        self._x = x
#        self._z = z
#        self._set_orientation()
#
#        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
#            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
#                str(raycing.uuid.uuid4())
#
#        if bl is not None:
#            if self.bl.flowSource != 'Qook0':
#                bl.oesDict[self.uuid] = [self, 1]
#                bl.oenamesToUUIDs[self.name] = self.uuid
#
#        self.center = center
#        if any([coord == 'auto' for coord in self.center]):
#            self._center = copy.copy(self.center)
        super().__init__(bl, name, center, x, z, **kwargs)
        self.R = R
        self.phiOffset = phiOffset
        self.thetaOffset = thetaOffset
#        self.isParametric = True
#        print(self.name, self._x, self.y, self._z)

    def _set_orientation(self):
        """Determines the local x, y and z in the global system."""
        sa = self.bl.sinAzimuth if self.bl is not None else 0.
        ca = self.bl.cosAzimuth if self.bl is not None else 1.
        if isinstance(self._x, (list, tuple, np.ndarray)):
            norm = sum([xc**2 for xc in self._x])**0.5
            self._x = [xc/norm for xc in self._x]
        else:
            self._x = sa, ca, 0.

        if isinstance(self._z, (list, tuple, np.ndarray)):
            norm = sum([zc**2 for zc in self._z])**0.5
            self._z = [zc/norm for zc in self._z]
        else:
            self._z = ca, -sa, 0.

        xdotz = np.dot(self._x, self._z)
        if abs(xdotz) > 1e-8:
            raycing.colorPrint('x and z must be orthogonal, got xz={0:.4e}'
                               .format(xdotz), 'RED')
        self.y = np.cross(self._z, self._x)

    def local_to_global_sph(self, phi, theta, **kwargs):
        thetaO = theta + self.thetaOffset
        phiO = phi + self.phiOffset
        z = np.sin(thetaO) * self.R
        y = np.cos(thetaO) * np.sin(phiO) * self.R
        x = np.cos(thetaO) * np.cos(phiO) * self.R
        xglo, yglo, zglo = self.local_to_global(x, y, z, **kwargs)
        return x, y, z, xglo, yglo, zglo

#    @raycing.append_to_flow_decorator
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
            self.local_to_global_sph(glo.phi, glo.theta)
        return glo

    @raycing.append_to_flow_decorator
    def expose(self, beam=None, onlyPositivePath=False):
        """Exposes the screen to the beam. *beam* is in global system, the
        returned beam is in local system of the screen and represents the
        desired image.


        .. Returned values: beamLocal
        """

        blo = rs.Beam(copyFrom=beam, withNumberOfReflections=True)  # local
        center, (x_axis, y_axis, z_axis) = self._ray_frame()
        dx, dy, dz = beam.x-center[0], beam.y-center[1], beam.z-center[2]
        sqb_2 = beam.a*dx + beam.b*dy + beam.c*dz
        sqc = dx**2 + dy**2 + dz**2 - self.R**2
        with np.errstate(invalid='ignore'):
            path = -sqb_2 + (sqb_2**2 - sqc)**0.5

        condBad = np.isnan(path) | np.isinf(path)
        if onlyPositivePath:
            condBad = condBad | (path < 0)
        indBad = np.where(condBad)
        path[indBad] = 0.
        blo.state[indBad] = self.lostNum

        blo.path += path
        rx = dx + beam.a*path
        ry = dy + beam.b*path
        rz = dz + beam.c*path
        blo.z = rx*z_axis[0] + ry*z_axis[1] + rz*z_axis[2]
        blo.y = rx*y_axis[0] + ry*y_axis[1] + rz*y_axis[2]
        blo.x = rx*x_axis[0] + ry*x_axis[1] + rz*x_axis[2]
        blo.theta = np.arcsin(blo.z / self.R) - self.thetaOffset
        blo.phi = np.arctan2(blo.y, blo.x) - self.phiOffset
        if hasattr(blo, 'Es'):
            propPhase = np.exp(1e7j * (blo.E / CHBAR) * path)
            blo.Es *= propPhase
            blo.Ep *= propPhase
        raycing.append_to_flow(self.expose, [blo],
                               inspect.currentframe())

        return blo
