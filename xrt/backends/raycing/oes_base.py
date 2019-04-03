# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import time
import numpy as np
import inspect
import copy

import matplotlib as mpl
from .. import raycing
from . import sources as rs
from . import myopencl as mcl
from .physconsts import PI2, CH, CHBAR, SQRT2PI, M0C2, R0
from .materials import EmptyMaterial
try:
    import pyopencl as cl  # analysis:ignore
    isOpenCL = True
except ImportError:
    isOpenCL = False

__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "06 Oct 2017"

__fdir__ = os.path.dirname(__file__)

allArguments = ('bl', 'name', 'center', 'bragg', 'pitch', 'roll', 'yaw',
                'positionRoll', 'extraPitch', 'extraRoll', 'extraYaw',
                'rotationSequence', 'extraRotationSequence',
                'surface', 'material', 'material2', 'alpha',
                'limPhysX', 'limOptX', 'limPhysY', 'limOptY',
                'limPhysX2', 'limPhysY2', 'limOptX2', 'limOptY2',
                'isParametric', 'shape', 'gratingDensity', 'order',
                'shouldCheckCenter',
                'dxFacet', 'dyFacet', 'dxGap', 'dyGap', 'Rm',
                'crossSection', 'Rs', 'R', 'r', 'p', 'q',
                'isCylindrical',
                'cryst1roll', 'cryst2roll', 'cryst2pitch', 'alarmLevel',
                'cryst2finePitch', 'cryst2perpTransl', 'cryst2longTransl',
                'fixedOffset', 't', 'focus', 'zmax', 'nCRL', 'f', 'E', 'N',
                'isCentralZoneBlack', 'thinnestZone', 'f1', 'f2',
                'phaseShift', 'vorticity', 'grazingAngle',
                'blaze', 'antiblaze', 'rho', 'aspect', 'depth', 'coeffs',
                'targetOpenCL', 'precisionOpenCL')


def flatten(x):
    if x is None:
        x = [0, 0]
    if isinstance(x, (list, tuple)):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


class OE(object):
    """The main base class for an optical element. It implements a generic flat
    mirror, crystal, multilayer or grating."""
    hiddenMethods = ['multiple_reflect']
    cl_plist = ["center"]
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        return 0.;
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        return (float3)(0.,0.,1.);
    }    """
    cl_local_g = """
    float3 local_g(float8 cl_plist, int i, float x, float y, float rho)
    {
        rho = -100.;
        return (float3)(0.,rho,0.);
    }    """
    cl_xyz_param = """
    float3 xyz_to_param(float8 cl_plist, float x, float y, float z)
    {
        return (float3)(y, atan2(x, z), sqrt(x*x + z*z));
    }"""

    def __init__(
        self, bl=None, name='', center=[0, 0, 0],
        pitch=0, roll=0, yaw=0, positionRoll=0, rotationSequence='RzRyRx',
        extraPitch=0, extraRoll=0, extraYaw=0, extraRotationSequence='RzRyRx',
        alarmLevel=None, surface=None, material=None,
        alpha=None,
        limPhysX=[-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE],
        limOptX=None,
        limPhysY=[-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE],
        limOptY=None, isParametric=False, shape='rect',
        gratingDensity=None, order=None, shouldCheckCenter=False,
            targetOpenCL=None, precisionOpenCL='float64'):
        u"""
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`
            Container for beamline elements. Optical elements are added to its
            `oes` list.

        *name*: str
            User-specified name, occasionally used for diagnostics output.

        *center*: 3-sequence of floats
            3D point in global system. Any two coordinates
            can be 'auto' for automatic alignment.

        *pitch, roll, yaw*: floats
            Rotations Rx, Ry, Rz, correspondingly, defined in the local system.
            If the material belongs to `Crystal`, *pitch* can be
            calculated automatically if alignment energy is given as a single
            element list [energy]. If 'auto',
            the alignment energy will be taken from beamLine.alignE.

        *positionRoll*: float
            A global roll used for putting the OE upside down (=np.pi) or
            at horizontal deflection (=[-]np.pi/2). This parameter does the
            same rotation as *roll*. It is introduced for holding large angles,
            as π or π/2 whereas *roll* is meant for smaller [mis]alignment
            angles.

        *rotationSequence*: str, any combination of 'Rx', 'Ry' and 'Rz'
            Gives the sequence of rotations of the OE around the local axes.
            The sequence is read from left to right (do not consider it as an
            operator). When rotations are more than one, the final position of
            the optical element depends on this parameter.

        *extraPitch, extraRoll, extraYaw, extraRotationSequence*:
            Similar to *pitch, roll, yaw, rotationSequence* but applied after
            them. This is sometimes necessary because rotations do not commute.
            The extra angles were introduced for easier misalignment after the
            initial positioning of the OE.

        *alarmLevel*: float or None
            Allowed fraction of incident rays to be absorbed by OE. If
            exceeded, an alarm output is printed in the console.

        *surface*: None or sequence of str
            If there are several optical surfaces, such as metalized stripes on
            a mirror, these are listed here as names; then also the optical
            limits *must* all be given by sequences of the same length if
            not None.

        *material*: None or sequence of material objects
            The material(s) must have
            :meth:`get_amplitude` or :meth:`get_refractive_index` method. If
            not None, must correspond to *surface*. If None, the reflectivities
            are equal to 1.

        *alpha*: float
            Asymmetry angle for a crystal OE (rad). Positive sign is for the
            atomic planes' normal looking towards positive *y*.

        *limPhysX* and *limPhysY*: [*min*, *max*] where *min*, *max* are
            floats or sequences of floats
            Physical dimension = local coordinate of the corresponding edge.
            Can be given by sequences of the length of *surface*. You do not
            have to provide the limits, although they may help in finding
            intersection points, especially for (strongly) curved surfaces.

        *limOptX* and *limOptY*: [*min*, *max*] where *min*, *max* are
            floats or sequences of floats
            Optical dimension = local coordinate of the corresponding edge.
            Useful when the optical surface is smaller than the whole
            surface, e.g. for metalized stripes on a mirror.

        *isParametric*: bool
            If True, the OE is defined by parametric equations rather than by
            z(*x*, *y*) function. For example, parametric representation is
            useful for describing closed surfaces, such as capillaries. The
            user must supply the transformation functions :meth:`param_to_xyz`
            and :meth:`xyz_to_param` between local (*x*, *y*, *z*) and (*s*,
            *phi*, *r*) and the parametric surface *local_r* dependent on (*s*,
            *phi*). The exact meaning of these three new parameters is up to
            the user because this meaning is self-contained in the above
            mentioned user-supplied functions. For example, these can be viewed
            as cylindrical-like coordinates, where *s* is a running coordinate
            on a 3D axial curve, *phi* and *r* are polar coordinates in planes
            normal to the axial curve and crossing that curve at point *s*.
            Class :class:`SurfaceOfRevolution` gives an example of the
            transformation functions and represents a useful kind of parametric
            surface.
            The methods :meth:`local_n` (surface normal) and :meth:`local_g`
            (grating vector, if used for this OE) return 3D vectors in local
            xyz space but now the two input coordinate parameters
            are *s* and *phi*.
            The limits [*limPhysX*, *limOptX*] and [*limPhysY*, *limOptY*]
            still define, correspondingly, the limits in local *x* and *y*.
            The local beams (footprints) will additionally contain *s*, *phi*
            and *r* arrays.

        *shape*: str or list of [x, y] pairs
            The shape of OE. Supported: 'rect', 'round' or a list of [x, y]
            pairs for an arbitrary shape.

        *gratingDensity*: None or list
            If material *kind* = 'grating', its density can be defined as list
            [axis, ρ\ :sub:`0`, *P*\ :sub:`0`, *P*\ :sub:`1`, *P*\ :sub:`2`],
            where ρ\ :sub:`0` is the constant line density in inverse mm,
            *P*\ :sub:`0` -- *P*\ :sub:`2` are polynom coefficients defining
            the line density variation, so that for a given axis

            .. math::

                \\rho_x = \\rho_0\\cdot(P_0 + 2 P_1 x + 3 P_2 x^2).

            Example: ['y', 800, 1, 0, 0] for the grating with constant
            spacing along the 'y' direction; ['y', 1200, 1, 1e-6, 3.1e-7] for
            the VLS grating.

        *order*: int or sequence of ints
            The order(s) of grating, FZP or Bragg-Fresnel diffraction.

        *shouldCheckCenter*: bool
            This is a leagcy parameter designed to work together with alignment
            stages -- classes in module :mod:`~xrt.backends.raycing.stages` --
            which modify the orientation of an optical element.
            if True, invokes *checkCenter* method for checking whether the oe
            center lies on the original beam line. *checkCenter* implies
            vertical deflection and ignores any difference in height. You
            should override this method for OEs of horizontal deflection.

        *targetOpenCL*: None, str, 2-tuple or tuple of 2-tuples
            pyopencl can accelerate the search for the intersections of rays
            with the OE. If pyopencl is used, *targetOpenCL* is a tuple
            (iPlatform, iDevice) of indices in the lists cl.get_platforms() and
            platform.get_devices(), see the section :ref:`calculations_on_GPU`.
            None, if pyopencl is not wanted. Ignored if pyopencl is not
            installed.

        *precisionOpenCL*: 'float32' or 'float64', only for GPU.
            Single precision (float32) should be enough. So far, we do not see
            any example where double precision is required. The calculations
            with double precision are much slower. Double precision may be
            unavailable on your system.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.oes:  # First init
                bl.oes.append(self)
                self.ordinalNum = len(bl.oes)
                self.lostNum = -self.ordinalNum
        raycing.set_name(self, name)
#        if name not in [None, 'None', '']:
#            self.name = name
#        elif not hasattr(self, 'name'):
#            self.name = '{0}{1}'.format(self.__class__.__name__,
#                                        self.ordinalNum)

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.name] = [self, 1]

        self.shouldCheckCenter = shouldCheckCenter

        self.center = center
        if any([x == 'auto' for x in self.center]):
            self._center = copy.copy(self.center)
        if (bl is not None) and self.shouldCheckCenter:
            self.checkCenter()

        self.pitch = raycing.auto_units_angle(pitch)
        if isinstance(self.pitch, (raycing.basestring, list, tuple)):
            self._pitch = copy.copy(self.pitch)
        self.roll = raycing.auto_units_angle(roll)
        self.yaw = raycing.auto_units_angle(yaw)
        self.rotationSequence = rotationSequence
        self.positionRoll = raycing.auto_units_angle(positionRoll)

        self.extraPitch = raycing.auto_units_angle(extraPitch)
        self.extraRoll = raycing.auto_units_angle(extraRoll)
        self.extraYaw = raycing.auto_units_angle(extraYaw)
        self.extraRotationSequence = extraRotationSequence
        self.alarmLevel = alarmLevel

        self.surface = surface
        self.material = material
        self.set_alpha(raycing.auto_units_angle(alpha))
        self.curSurface = 0
        self.dx = 0
        self.limPhysX = limPhysX
        self.limPhysY = limPhysY
        if self.limPhysX is None:
            self.limPhysX = [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE]
        if self.limPhysY is None:
            self.limPhysY = [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE]
        self.limOptX = limOptX
        self.limOptY = limOptY
        self.isParametric = isParametric
        self.use_rays_good_gn = False  # use rays_good_gn instead of rays_good

        self.shape = shape
        self.gratingDensity = gratingDensity
        if self.gratingDensity is not None:
            self.material = EmptyMaterial()
        self.order = 1 if order is None else order
        self.get_surface_limits()
        self.cl_ctx = None
        self.ucl = None
        self.footprint = []
        if targetOpenCL is not None:
            if not isOpenCL:
                print("pyopencl is not available!")
            else:
                cl_template = os.path.join(__fdir__, r'materials.cl')
                with open(cl_template, 'r') as f:
                    kernelsource = f.read()
                cl_template = os.path.join(__fdir__, r'OE.cl')
                with open(cl_template, 'r') as f:
                    kernelsource += f.read()
                kernelsource = kernelsource.replace('MY_LOCAL_Z',
                                                    self.cl_local_z)
                kernelsource = kernelsource.replace('MY_LOCAL_N',
                                                    self.cl_local_n)
                kernelsource = kernelsource.replace('MY_LOCAL_G',
                                                    self.cl_local_g)
                kernelsource = kernelsource.replace('MY_XYZPARAM',
                                                    self.cl_xyz_param)
                if self.isParametric:
                    kernelsource = kernelsource.replace(
                        'ol isParametric = false', 'ol isParametric = true')
                self.ucl = mcl.XRT_CL(None,
                                      targetOpenCL,
                                      precisionOpenCL,
                                      kernelsource)
                if self.ucl.lastTargetOpenCL is not None:
                    self.cl_precisionF = self.ucl.cl_precisionF
                    self.cl_precisionC = self.ucl.cl_precisionC
                    self.cl_queue = self.ucl.cl_queue
                    self.cl_ctx = self.ucl.cl_ctx
                    self.cl_program = self.ucl.cl_program
                    self.cl_mf = self.ucl.cl_mf
                    self.cl_is_blocking = self.ucl.cl_is_blocking

    def set_alpha(self, alpha):
        """Sets the asymmetry angle *alpha* for a crystal OE. It calculates
        cos(alpha) and sin(alpha) which are then used for rotating the normal
        to the crystal planes."""
        self.alpha = alpha
        if alpha is None:
            return
        self.cosalpha = np.cos(alpha)
        self.sinalpha = np.sin(alpha)
        self.tanalpha = self.sinalpha / self.cosalpha

    def checkCenter(self, misalignmentTolerated=raycing.misalignmentTolerated):
        """Checks whether the oe center lies on the original beam line. If the
        misalignment is bigger than *misalignmentTolerated*, a warning is
        issued. This implementation implies vertical deflection and ignores any
        difference in height. You should override this method for OEs of
        horizontal deflection."""
        a = self.bl.sinAzimuth
        b = self.bl.cosAzimuth
        d = b * (self.center[0]-self.bl.sources[0].center[0])\
            - a * (self.center[1]-self.bl.sources[0].center[1])
        if abs(d) > misalignmentTolerated:
            print("Warning: {0} is off the beamline by {1}".format(
                  self.name, d))
            xc = a * b * (self.center[1]-self.bl.sources[0].center[1])\
                + self.bl.sources[0].center[0] * b**2 + self.center[0] * a**2
            yc = a * b * (self.center[0]-self.bl.sources[0].center[0])\
                + self.bl.sources[0].center[1] * a**2 + self.center[1] * b**2
            print("suggested xc, yc: ", xc, yc)

    def get_orientation(self):
        """To be overridden. Should provide pitch, roll, yaw, height etc. given
        other, possibly newly added variables. Used in conjunction with the
        classes in :mod:`~xrt.backends.raycing.stages`."""
        pass

    def get_Rmer_from_Coddington(self, p, q, pitch=None):
        if pitch is None:
            pitch = self.pitch
        return 2 * p * q / (p+q) / np.sin(abs(pitch))

    def get_rsag_from_Coddington(self, p, q, pitch=None):
        if pitch is None:
            pitch = self.pitch
        return 2 * p * q / (p+q) * np.sin(abs(pitch))

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position. Typically is
        overridden in the derived classes. Must return either a scalar or an
        array of the length of *x* and *y*."""
        return np.zeros_like(y)  # just flat

    def local_z_distorted(self, x, y):
        return

    def local_g(self, x, y, rho=-100.):
        """For a grating, gives the local reciprocal groove vector (without
        2pi!) in 1/mm at (*x*, *y*) position. The vector must lie on the
        surface, i.e. be orthogonal to the normal. Typically is overridden in
        the derived classes or defined in Material class. Returns a 3-tuple of
        floats or of arrays of the length of *x* and *y*."""

        try:
            rhoList = self.gratingDensity
            if rhoList is not None:
                if rhoList[0] == 'x':
                    N = rhoList[1] * (rhoList[2] + 2*rhoList[3]*x +
                                      3*rhoList[4]*x**2)
                    return N, np.zeros_like(N), np.zeros_like(N)
                elif rhoList[0] == 'y':
                    N = rhoList[1] * (rhoList[2] + 2*rhoList[3]*y +
                                      3*rhoList[4]*y**2)
                    return np.zeros_like(N), N, np.zeros_like(N)
        except:
            pass
        return 0, rho, 0  # constant line spacing along y

    def local_n(self, x, y):  # or as (self, s, phi)
        """Determines the normal vector of OE at (*x*, *y*) position. Typically
        is overridden in the derived classes. If OE is an asymmetric crystal,
        *local_n* must return 2 normals as a 6-sequence: the 1st one of the
        atomic planes and the 2nd one of the surface. Note the order!

        If *isParametric* in the constructor is True, :meth:`local_n` still
        returns 3D vector(s) in local xyz space but now the two input
        coordinate parameters are *s* and *phi*.

        The result is a 3-tuple or a 6-tuple. Each element is either a scalar
        or an array of the length of *x* and *y*."""
        # just flat:
        a = 0.  # -dz/dx
        b = 0.  # -dz/dy
        c = 1.
#        norm = (a**2 + b**2 + c**2)**0.5
#        a, b, c = a/norm, b/norm, c/norm
        if self.alpha:
            bAlpha, cAlpha = \
                raycing.rotate_x(b, c, self.cosalpha, -self.sinalpha)
            return [a, bAlpha, cAlpha, a, b, c]
        else:
            return [a, b, c]

    def local_n_distorted(self, x, y):  # or as (self, s, phi)
        """Distortion to the local normal. If *isParametric* in the
        constructor is True, the input arrays are understood as (*s*, *phi*).

        Distortion can be given in two ways and is signaled by the length of
        the returned tuple:

        1) As d_pitch and d_roll rotation angles of the normal (i.e. rotations
           Rx and Ry). A tuple of the two arrays must be returned. This option
           is also suitable for parametric coordinates because the two
           rotations will be around Cartesian axes and the local normal
           (local_n) is also a 3D vector in local xyz space.

        2) As a 3D vector that will be added to the local normal calculated at
           the same coordinates. The returned vector can have any length, not
           necessarily unity. As for local_n, the 3D vector is in local xyz
           space even for a parametric surface. The resulted vector
           `local_n + local_n_distorted` will be normalized internally before
           calculating the reflected beam direction. A tuple of 3 arrays must
           be returned.
        """
        return

    _h = 20.

    def xyz_to_param(self, x, y, z):  # for flat mirror as example
        r = np.sqrt(x**2 + (self._h-z)**2)
        return y, np.arcsin(x / r), r  # s, phi, r

    def param_to_xyz(self, s, phi, r):  # for flat mirror as example
        return r*np.sin(phi), s, self._h - r*np.cos(phi)  # x, y, z

    def local_r(self, s, phi):  # for flat mirror as example
        """Determines the surface of OE at (*s*, *phi*) position. Used when
        *isParametric* in the constructor is True. Typically is overridden in
        the derived classes. Must return either a scalar or an array of the
        length of *s* and *phi*."""
        return self._h / np.cos(phi)

    def local_r_distorted(self, s, phi):
        return

    def find_dz(
            self, local_f, t, x0, y0, z0, a, b, c, invertNormal, derivOrder=0):
        """Returns the z or r difference (in the local system) between the ray
        and the surface. Used for finding the intersection point."""
        x = x0 + a*t
        y = y0 + b*t
        z = z0 + c*t
        if derivOrder == 0:
            if self.isParametric:
                if local_f is None:
                    local_f = self.local_r
                diffSign = -1
            else:
                if local_f is None:
                    local_f = self.local_z
                diffSign = 1
        else:
            if local_f is None:
                local_f = self.local_n

        if self.isParametric:  # s, phi, r =
            x, y, z = self.xyz_to_param(x, y, z)
        surf = local_f(x, y)  # z or r
        if self.isParametric:
            z_distorted = self.local_r_distorted(x, y)
        else:
            z_distorted = self.local_z_distorted(x, y)
        if z_distorted is not None:
            surf += z_distorted

        if derivOrder == 0:
            if surf is None:  # lost
                surf = np.zeros_like(z)
            ind = np.isnan(surf)
            if ind.sum() > 0:
                if raycing._VERBOSITY_ > 0:
                    print('{0} NaNs in the surface!!!'.format(ind.sum()))
                surf[ind] = 0
            dz = (z - surf) * diffSign * invertNormal
        else:
            if surf is None:  # lost
                surf = 0, 0, 1
            dz = (a*surf[-3] + b*surf[-2] + c*surf[-1]) * invertNormal
        return dz, x, y, z

    def find_intersection(self, local_f, t1, t2, x, y, z, a, b, c,
                          invertNormal, derivOrder=0):
        """Finds the ray parameter *t* at the intersection point with the
        surface. Requires *t1* and *t2* as input bracketing. The ray is
        determined by its origin point (*x*, *y*, *z*) and its normalized
        direction (*a*, *b*, *c*). *t* is then the distance between the origin
        point and the intersection point. *derivOrder* tells if minimized is
        the z-difference (=0) or its derivative (=1)."""
        dz1, x1, y1, z1 = self.find_dz(
            local_f, t1, x, y, z, a, b, c, invertNormal, derivOrder)
        dz2, x2, y2, z2 = self.find_dz(
            local_f, t2, x, y, z, a, b, c, invertNormal, derivOrder)
#        tMin = max(t1.min(), 0)
        tMin = t1.min()
        tMax = t2.max()
        ind1 = dz1 <= 0  # lost rays; for them the solution is t1
        ind2 = dz2 >= 0  # over rays; for them the solution is t2
        dz2[ind1 | ind2] = 0
        t2[ind1] = t1[ind1]
        x2[ind1] = x1[ind1]
        y2[ind1] = y1[ind1]
        z2[ind1] = z1[ind1]
        ind = ~(ind1 | ind2)  # good rays
        if abs(dz2).max() > abs(dz1).max()*20:
            t2, x2, y2, z2, numit = self._use_Brent_method(
                local_f, t1, t2, x, y, z, a, b, c, invertNormal, derivOrder,
                dz1, dz2, tMin, tMax, x2, y2, z2, ind)
        else:
            t2, x2, y2, z2, numit = self._use_my_method(
                local_f, t1, t2, x, y, z, a, b, c, invertNormal, derivOrder,
                dz1, dz2, tMin, tMax, x2, y2, z2, ind)
        if numit == raycing.maxIteration and raycing._VERBOSITY_ > 10:
            nn = ind.sum()
            print('maxIteration is reached for {0} ray{1}!!!'.format(
                  nn, 's' if nn > 1 else ''))
        if raycing._VERBOSITY_ > 10:
            print('numit=', numit)
        return t2, x2, y2, z2, ind1

    def find_intersection_CL(self, local_f, t1, t2, x, y, z, a, b, c,
                             invertNormal, derivOrder=0):
        """Finds the ray parameter *t* at the intersection point with the
        surface. Requires *t1* and *t2* as input bracketing. The ray is
        determined by its origin point (*x*, *y*, *z*) and its normalized
        direction (*a*, *b*, *c*). *t* is then the distance between the origin
        point and the intersection point. *derivOrder* tells if minimized is
        the z-difference (=0) or its derivative (=1)."""

        NRAYS = len(x)
        cl_plist = np.squeeze([getattr(self, p) for p in self.cl_plist])
        ext_param = np.zeros(8, dtype=self.cl_precisionF)
        ext_param[:len(cl_plist)] = self.cl_precisionF(cl_plist)

        if local_f is None:
            local_zN = np.int32(0)
        elif ((local_f.__name__)[-1:]).isdigit():
            local_zN = np.int32((local_f.__name__)[-1:])
        else:
            local_zN = np.int32(0)

        scalarArgs = [ext_param,
                      np.int32(invertNormal),
                      np.int32(derivOrder),
                      local_zN,
                      self.cl_precisionF(t1.min()),
                      self.cl_precisionF(t2.max())]

        slicedROArgs = [self.cl_precisionF(t1),  # t1
                        self.cl_precisionF(x),  # x
                        self.cl_precisionF(y),  # y
                        self.cl_precisionF(z),  # z
                        self.cl_precisionF(a),  # a
                        self.cl_precisionF(b),  # b
                        self.cl_precisionF(c)]  # c

        slicedRWArgs = [self.cl_precisionF(t2),  # t2
                        self.cl_precisionF(np.zeros_like(x)),  # x2
                        self.cl_precisionF(np.zeros_like(x)),  # y2
                        self.cl_precisionF(np.zeros_like(x))]  # z2

        t2, x2, y2, z2 = self.ucl.run_parallel(
            'find_intersection', scalarArgs, slicedROArgs, None, slicedRWArgs,
            None, NRAYS)
        return t2, x2, y2, z2

    def _use_my_method(
        self, local_f, t1, t2, x, y, z, a, b, c, invertNormal, derivOrder,
            dz1, dz2, tMin, tMax, x2, y2, z2, ind):
        numit = 2
        while (ind.sum() > 0) and (numit < raycing.maxIteration):
            t = t1[ind]
            dz = dz1[ind]
            t1[ind] = t2[ind]
            dz1[ind] = dz2[ind]
            t2[ind] = t - (t1[ind]-t) * dz / (dz1[ind]-dz)
            swap = t2[ind] < tMin
            t2[np.where(ind)[0][swap]] = tMin
            swap = t2[ind] > tMax
            t2[np.where(ind)[0][swap]] = tMax
            dz2[ind], x2[ind], y2[ind], z2[ind] = self.find_dz(
                local_f, t2[ind], x[ind], y[ind], z[ind],
                a[ind], b[ind], c[ind], invertNormal, derivOrder)
            # swapping using duble slicing:
            swap = np.sign(dz2[ind]) == np.sign(dz1[ind])
            t1[np.where(ind)[0][swap]] = t[swap]
            dz1[np.where(ind)[0][swap]] = dz[swap]
            ind = ind & (abs(dz2) > raycing.zEps)
            numit += 1
# t2 holds the ray parameter at the intersection point
        return t2, x2, y2, z2, numit

    def _use_Brent_method(self, local_f, t1, t2, x, y, z, a, b, c,
                          invertNormal, derivOrder, dz1, dz2, tMin, tMax,
                          x2, y2, z2, ind):
        """Finds the ray parameter *t* at the intersection point with the
        surface. Requires *t1* and *t2* as input bracketing. The rays are
        determined by the origin points (*x*, *y*, *z*) and the normalized
        directions (*a*, *b*, *c*). *t* is then the distance between the origin
        point and the intersection point.

        Uses the classic Brent (1973) method to find a zero of the function
        `dz` on the sign changing interval [*t1*, *t2*]. It is a safe version
        of the secant method that uses inverse quadratic extrapolation. Brent's
        method combines root bracketing, interval bisection, and inverse
        quadratic interpolation.

        A description of the Brent's method can be found at
        http://en.wikipedia.org/wiki/Brent%27s_method.
        """
        swap = abs(dz1[ind]) < abs(dz2[ind])
        if swap.sum() > 0:
            t1[np.where(ind)[0][swap]], t2[np.where(ind)[0][swap]] =\
                t2[np.where(ind)[0][swap]], t1[np.where(ind)[0][swap]]
            dz1[np.where(ind)[0][swap]], dz2[np.where(ind)[0][swap]] =\
                dz2[np.where(ind)[0][swap]], dz1[np.where(ind)[0][swap]]
        t3 = np.copy(t1)  # c:=a
        dz3 = np.copy(dz1)  # f(c)
        t4 = np.zeros_like(t1)  # d
        mflag = np.ones_like(t1, dtype='bool')
        numit = 2
        ind = ind & (abs(dz2) > raycing.zEps)
        while (ind.sum() > 0) and (numit < raycing.maxIteration):
            xa, xb, xc, xd = t1[ind], t2[ind], t3[ind], t4[ind]
            fa, fb, fc = dz1[ind], dz2[ind], dz3[ind]
            mf = mflag[ind]
            xs = np.empty_like(xa)
            inq = (fa != fc) & (fb != fc)
            if inq.sum() > 0:
                xai = xa[inq]
                xbi = xb[inq]
                xci = xc[inq]
                fai = fa[inq]
                fbi = fb[inq]
                fci = fc[inq]
                xs[inq] = \
                    xai * fbi * fci / (fai-fbi) / (fai-fci) + \
                    fai * xbi * fci / (fbi-fai) / (fbi-fci) + \
                    fai * fbi * xci / (fci-fai) / (fci-fbi)
            inx = ~inq
            if inx.sum() > 0:
                xai = xa[inx]
                xbi = xb[inx]
                fai = fa[inx]
                fbi = fb[inx]
                xs[inx] = xbi - fbi * (xbi-xai) / (fbi-fai)

            cond1 = ((xs < (3*xa + xb) / 4.) & (xs < xb) |
                     (xs > (3*xa + xb) / 4.) & (xs > xb))
            cond2 = mf & (abs(xs - xb) >= (abs(xb - xc) / 2.))
            cond3 = (~mf) & (abs(xs - xb) >= (abs(xc - xd) / 2.))
            cond4 = mf & (abs(xb - xc) < raycing.zEps)
            cond5 = (~mf) & (abs(xc - xd) < raycing.zEps)
            conds = cond1 | cond2 | cond3 | cond4 | cond5
            xs[conds] = (xa[conds] + xb[conds]) / 2.
            mf = conds

            fs, x2[ind], y2[ind], z2[ind] = self.find_dz(
                local_f, xs, x[ind], y[ind], z[ind], a[ind], b[ind], c[ind],
                invertNormal, derivOrder)
            xd[:] = xc[:]
            xc[:] = xb[:]
            fc[:] = fb[:]
            fafsNeg = ((fa < 0) & (fs > 0)) | ((fa > 0) & (fs < 0))
            xb[fafsNeg] = xs[fafsNeg]
            fb[fafsNeg] = fs[fafsNeg]
            fafsPos = ~fafsNeg
            xa[fafsPos] = xs[fafsPos]
            fa[fafsPos] = fs[fafsPos]
            swap = abs(fa) < abs(fb)
            xa[swap], xb[swap] = xb[swap], xa[swap]
            fa[swap], fb[swap] = fb[swap], fa[swap]
            t1[ind], t2[ind], t3[ind], t4[ind] = xa, xb, xc, xd
            dz1[ind], dz2[ind], dz3[ind] = fa, fb, fc
            mflag[ind] = mf

            ind = ind & (abs(dz2) > raycing.zEps)
            numit += 1
# t2 holds the ray parameter at the intersection point
        return t2, x2, y2, z2, numit

    def get_surface_limits(self):
        """Returns surface_limits."""
        cs = self.curSurface
        self.surfPhysX = self.limPhysX
        if self.limPhysX is not None:
            try:
                if raycing.is_sequence(self.limPhysX[0]):
                    self.surfPhysX = [self.limPhysX[0][cs],
                                      self.limPhysX[1][cs]]
            except IndexError:
                pass
        self.surfPhysY = self.limPhysY
        if self.limPhysY is not None:
            try:
                if raycing.is_sequence(self.limPhysY[0]):
                    self.surfPhysY = (self.limPhysY[0][cs],
                                      self.limPhysY[1][cs])
            except IndexError:
                pass
        self.surfOptX = self.limOptX
        if self.limOptX is not None:
            try:
                if raycing.is_sequence(self.limOptX[0]):
                    self.surfOptX = (self.limOptX[0][cs], self.limOptX[1][cs])
            except IndexError:
                pass
        self.surfOptY = self.limOptY
        if self.limOptY is not None:
            try:
                if raycing.is_sequence(self.limOptY[0]):
                    self.surfOptY = (self.limOptY[0][cs], self.limOptY[1][cs])
            except IndexError:
                pass

    def assign_auto_material_kind(self, material):
        if self.gratingDensity is not None:
            material.kind = 'grating'
        else:
            material.kind = 'mirror'

    def rays_good(self, x, y, is2ndXtal=False):
        """Returns *state* value for a ray with the given intersection point
        (*x*, *y*) with the surface of OE:
        1: good (intersected)
        2: reflected outside of working area ("out"),
        3: transmitted without intersection ("over"),
        -NN: lost (absorbed) at OE#NN - OE numbering starts from 1 !!!
        """
        if is2ndXtal:
            surfPhysX = self.surfPhysX2
            surfPhysY = self.surfPhysY2
            surfOptX = self.surfOptX2
            surfOptY = self.surfOptY2
        else:
            surfPhysX = self.surfPhysX
            surfPhysY = self.surfPhysY
            surfOptX = self.surfOptX
            surfOptY = self.surfOptY

        locState = np.ones(x.size, dtype=np.int)
        if isinstance(self.shape, raycing.basestring):
            if self.shape.startswith('re'):
                if surfOptX is not None:
                    locState[((surfPhysX[0] <= x) & (x < surfOptX[0])) |
                             ((surfOptX[1] <= x) & (x < surfPhysX[1]))] = 2
                if surfOptY is not None:
                    locState[((surfPhysY[0] <= y) & (y < surfOptY[0])) |
                             ((surfOptY[1] <= y) & (y < surfPhysY[1]))] = 2
                if not hasattr(self, 'overEdge'):
                    self.overEdge = 'yMax'
                ovE = self.overEdge.lower()
                if ovE.startswith('x') and ovE.endswith('in'):
                    locState[x < surfPhysX[0]] = 3
                    locState[(y < surfPhysY[0]) | (y > surfPhysY[1]) |
                             (x > surfPhysX[1])] = self.lostNum
                elif ovE.startswith('x') and ovE.endswith('ax'):
                    locState[x > surfPhysX[1]] = 3
                    locState[(y < surfPhysY[0]) | (y > surfPhysY[1]) |
                             (x < surfPhysX[0])] = self.lostNum
                elif ovE.startswith('y') and ovE.endswith('in'):
                    locState[y < surfPhysY[0]] = 3
                    locState[(x < surfPhysX[0]) | (x > surfPhysX[1]) |
                             (y > surfPhysY[1])] = self.lostNum
                elif ovE.startswith('y') and ovE.endswith('ax'):
                    locState[y > surfPhysY[1]] = 3
                    locState[(x < surfPhysX[0]) | (x > surfPhysX[1]) |
                             (y < surfPhysY[0])] = self.lostNum
            elif self.shape.startswith('ro'):
                centerX = (surfPhysX[0]+surfPhysX[1]) * 0.5
                if np.isnan(centerX):
                    centerX = 0
                radiusX = (surfPhysX[1]-surfPhysX[0]) * 0.5
                if surfPhysY is not None:
                    centerY = (surfPhysY[0]+surfPhysY[1]) * 0.5
                    radiusY = (surfPhysY[1]-surfPhysY[0]) * 0.5
                else:
                    centerY = 0.
                    radiusY = radiusX
                if np.isnan(centerY):
                    centerY = 0
                if not np.isinf(radiusX):
                    locState[((x-centerX)/radiusX)**2 +
                             ((y-centerY)/radiusY)**2 > 1] =\
                        self.lostNum
        elif isinstance(self.shape, list):
            footprint = mpl.path.Path(self.shape)
            locState[:] = footprint.contains_points(np.array(zip(x, y)))
            locState[(locState == 0) & (y < surfPhysY[0])] = self.lostNum
            locState[locState == 0] = 3
        else:
            raise ValueError('Unknown shape of OE {0}!'.format(self.name))
        return locState

    def reflect(self, beam=None, needLocal=True, noIntersectionSearch=False,
                returnLocalAbsorbed=None):
        r"""
        Returns the reflected or transmitted beam as :math:`\vec{out}` in
        global and local (if *needLocal* is true) systems.

        .. rubric:: Mirror [wikiSnell]_:

        .. math::

            \vec{out}_{\rm reflect} &= \vec{in} + 2\cos{\theta_1}\vec{n}\\
            \vec{out}_{\rm refract} &= \frac{n_1}{n_2}\vec{in} +
            \left(\frac{n_1}{n_2}\cos{\theta_1} - \cos{\theta_2}\right)\vec{n},

        where

        .. math::

            \cos{\theta_1} &= -\vec{n}\cdot\vec{in}\\
            \cos{\theta_2} &= sign(\cos{\theta_1})\sqrt{1 -
            \left(\frac{n_1}{n_2}\right)^2\left(1-\cos^2{\theta_1}\right)}.

        .. rubric:: Grating or FZP [SpencerMurty]_:

        For the reciprocal grating vector :math:`\vec{g}` and the :math:`m`\ th
        diffraction order:

        .. math::

            \vec{out} = \vec{in} - dn\vec{n} + \vec{g}m\lambda

        where

        .. math::

            dn = -\cos{\theta_1} \pm \sqrt{\cos^2{\theta_1} -
            2(\vec{g}\cdot\vec{in})m\lambda - \vec{g}^2 m^2\lambda^2}

        .. rubric:: Crystal [SanchezDelRioCerrina]_:

        Crystal (generally asymmetrically cut) is considered a grating with the
        reciprocal grating vector equal to

        .. math::

            \vec{g} = \left(\vec{n_\vec{H}} -
            (\vec{n_\vec{H}}\cdot\vec{n})\vec{n})\right) / d_\vec{H}.

        Note that :math:`\vec{g}` is along the line of the intersection of the
        crystal surface with the plane formed by the two normals
        :math:`\vec{n_\vec{H}}` and :math:`\vec{n}` and its length is
        :math:`|\vec{g}|=\sin{\alpha}/d_\vec{H}`, with :math:`\alpha`
        being the asymmetry angle.

        .. [wikiSnell] http://en.wikipedia.org/wiki/Snell%27s_law .
        .. [SpencerMurty] G. H. Spencer and M. V. R. K. Murty,
           J. Opt. Soc. Am. **52** (1962) 672.
        .. [SanchezDelRioCerrina] M. Sánchez del Río and F. Cerrina,
           Rev. Sci. Instrum. **63** (1992) 936.


        *returnLocalAbsorbed*: None or int
            If not None, returns the absorbed intensity in local beam.

        *noIntersectionSearch*: bool
            Used in wave propagation, normally should be False.


        .. .. Returned values: beamGlobal, beamLocal
        """
        self.footprint = []
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        self.get_orientation()
        # output beam in global coordinates
        gb = rs.Beam(copyFrom=beam)
        if needLocal:
            # output beam in local coordinates
            lb = rs.Beam(copyFrom=beam)
        else:
            lb = gb
        good = beam.state > 0
#        good = beam.state == 1
        if good.sum() == 0:
            return gb, lb
# coordinates in local virgin system:
        pitch = self.pitch
        if hasattr(self, 'bragg'):
            pitch += self.bragg
        raycing.global_to_virgin_local(self.bl, beam, lb, self.center, good)

        self._reflect_local(good, lb, gb,
                            pitch, self.roll+self.positionRoll, self.yaw,
                            self.dx, noIntersectionSearch=noIntersectionSearch,
                            material=self.material)
        goodAfter = (gb.state == 1) | (gb.state == 2)
# in global coordinate system:
        if goodAfter.sum() > 0:
            raycing.virgin_local_to_global(self.bl, gb, self.center, goodAfter)
# not intersected rays remain unchanged except their state:
        notGood = ~goodAfter
        if notGood.sum() > 0:
            rs.copy_beam(gb, beam, notGood)
        if returnLocalAbsorbed is not None:
            absorbedLb = rs.Beam(copyFrom=lb)
            absorbedLb.absorb_intensity(beam)
            lb = absorbedLb
        raycing.append_to_flow(self.reflect, [gb, lb], inspect.currentframe())
        lb.parentId = self.name
        return gb, lb  # in global(gb) and local(lb) coordinates

    def multiple_reflect(
            self, beam=None, maxReflections=1000, needElevationMap=False,
            returnLocalAbsorbed=None):
        """
        Does the same as :meth:`reflect` but with up to *maxReflections*
        reflection on the same surface.

        The returned beam has additional fields: *nRefl* for the number of
        reflections, *elevationD* for the maximum elevation distance between
        the rays and the surface as the ray travels between the impact points,
        *elevationX*, *elevationY*, *elevationZ* for the coordinates of the
        maximum elevation points.

        *returnLocalAbsorbed*: None or int
            If not None, returns the absorbed intensity in local beam.


        .. .. Returned values: beamGlobal, beamLocal
        """
        self.footprint = []
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        self.get_orientation()
# output beam in global coordinates
        gb = rs.Beam(copyFrom=beam)
        lb = gb
        good = beam.state > 0
        if good.sum() == 0:
            return gb, lb
# coordinates in local virgin system:
        raycing.global_to_virgin_local(self.bl, beam, lb, self.center, good)
        iRefl = 0
        isMulti = False
        while iRefl < maxReflections:
            tmpX, tmpY, tmpZ =\
                np.copy(lb.x[good]), np.copy(lb.y[good]), np.copy(lb.z[good])
            if raycing._VERBOSITY_ > 10:
                print('reflection No {0}'.format(iRefl + 1))
            if iRefl == 0:
                if needElevationMap:
                    lb.elevationD = -np.ones_like(lb.x)
                    lb.elevationX = -np.ones_like(lb.x)*raycing.maxHalfSizeOfOE
                    lb.elevationY = -np.ones_like(lb.x)*raycing.maxHalfSizeOfOE
                    lb.elevationZ = -np.ones_like(lb.x)*raycing.maxHalfSizeOfOE
            self._reflect_local(good, lb, gb, self.pitch,
                                self.roll+self.positionRoll, self.yaw,
                                self.dx, material=self.material,
                                needElevationMap=needElevationMap,
                                isMulti=isMulti)
            if iRefl == 0:
                isMulti = True
                lb.nRefl = np.zeros_like(lb.state)
            ov = lb.state[good] == 3
            lb.x[np.where(good)[0][ov]] = tmpX[ov]
            lb.y[np.where(good)[0][ov]] = tmpY[ov]
            lb.z[np.where(good)[0][ov]] = tmpZ[ov]
            good = (lb.state == 1) | (lb.state == 2)
            lb.nRefl[good] += 1
            if iRefl == 0:
                # all local footprints:
                lbN = rs.Beam(copyFrom=lb, withNumberOfReflections=True)
            else:
                lbN.concatenate(lb)
            iRefl += 1
            if raycing._VERBOSITY_ > 10:
                print('iRefl=', iRefl, 'remains=', good.sum())
#                if good.sum() > 0:
#                    print('y min max ', lb.y[good].min(), lb.y[good].max())
            if good.sum() == 0:
                break
#            gc.collect()
# in global coordinate system:
        goodAfter = gb.nRefl > 0
        gb.state[goodAfter] = 1
        if goodAfter.sum() > 0:
            raycing.virgin_local_to_global(self.bl, gb, self.center, goodAfter)
# not intersected rays remain unchanged except their state:
        notGood = ~goodAfter
        if notGood.sum() > 0:
            rs.copy_beam(gb, beam, notGood)
# in global(gb) and local(lbN) coordinates. lbN holds all the reflection spots.
        if returnLocalAbsorbed is not None:
            absorbedLb = rs.Beam(copyFrom=lb)
            absorbedLb.absorb_intensity(beam)
            lbN = absorbedLb
        lbN.parentId = self.name
        raycing.append_to_flow(self.multiple_reflect, [gb, lbN],
                               inspect.currentframe())
        return gb, lbN

    def scatter(self, beam=None, needLocal=True, channels=['inelastic'],
                withSelfAbsorption=True, face='front'):
        r"""

        .. .. Returned values: beamGlobal, beamLocal
        """
        self.footprint = []
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        self.get_orientation()
        # output beam in global coordinates
        gb = rs.Beam(copyFrom=beam)
        if needLocal:
            # output beam in local coordinates
            lb = rs.Beam(copyFrom=beam)
        else:
            lb = gb
        good = beam.state > 0
#        good = beam.state == 1
        if good.sum() == 0:
            return gb, lb
# coordinates in local virgin system:
        pitch = self.pitch
        if hasattr(self, 'bragg'):
            pitch += self.bragg
        raycing.global_to_virgin_local(self.bl, beam, lb, self.center, good)

        fluoLines = []
        for fluoChannel in channels:
            if isinstance(fluoChannel, (list, tuple)):
                for element in self.material.elements:
                    if element.name == fluoChannel[0]:
                        if str(fluoChannel[1]).upper() == 'ALL':
                            for fluoLine in element.flLines:
                                fluoLines.append((fluoChannel[0],
                                                  element.flLines[fluoLine]))
                        else:
                            fluoLines.append((fluoChannel[0],
                                              element.flLines[fluoChannel[1]]))
                        break
            else:
                fluoLines.append(fluoChannel)
        lbf2 = None
        gbf2 = None
        for fluoChannel in fluoLines:
            lbf = rs.Beam(copyFrom=lb)
            gbf = rs.Beam(copyFrom=gb)

            if hasattr(self, 't'):  # Plate
                self._reflect_local(
                    good, lbf, gbf, pitch,  # self.pitch + self.bragg,
                    self.roll + self.positionRoll + self.cryst1roll, self.yaw,
                    self.dx, local_z=self.local_z1, local_n=self.local_n1,
                    material=self.material, fluo=fluoChannel)
            else:
                self._reflect_local(
                    good, lbf, gbf, pitch, self.roll+self.positionRoll,
                    self.yaw, self.dx, material=self.material,
                    fluo=fluoChannel)
            goodAfter = (gbf.state == 1) | (gbf.state == 2)
            notGood = ~goodAfter
            if notGood.sum() > 0:
                rs.copy_beam(gbf, beam, notGood)

            if gbf2 is None:
                lbf2 = lbf
                gbf2 = gbf
            else:
                lbf2.concatenate(lbf)
                gbf2.concatenate(gbf)

        goodAfter = (gbf2.state == 1) | (gbf2.state == 2)
        locCoord = rs.Beam(copyFrom=lbf2)
        if needLocal:
            lbf2 = rs.Beam(copyFrom=gbf2)  # output beam in local coordinates
        else:
            lbf2 = gbf2
        good2 = goodAfter
        if hasattr(self, 't'):  # is instance of Plate
            gbf2.state[~good2] = self.lostNum
        if good2.sum() == 0:
            return gbf2, lbf2

        if withSelfAbsorption:
            if not hasattr(self, 't'):
                face = 'front'

            if face == 'front':
                self._reflect_local(
                    good2, lbf2, gbf2, pitch, self.roll+self.positionRoll,
                    self.yaw, self.dx, material=self.material,
                    fromVacuum=False)  # , is2ndXtal=True)
            else:
                self._reflect_local(
                    good2, lbf2, gbf2,
                    -self.pitch - self.bragg +
                    self.cryst2pitch + self.cryst2finePitch,
                    self.roll + self.cryst2roll - np.pi + self.positionRoll,
                    -self.yaw,
                    -self.dx, self.cryst2longTransl, -self.cryst2perpTransl,
                    local_z=self.local_z2, local_n=self.local_n2,
                    fromVacuum=False, material=self.material2, is2ndXtal=True)

            goodAfter = (gbf2.state == 1) | (gbf2.state == 2)

        lbf2.x = locCoord.x
        lbf2.y = locCoord.y
        lbf2.z = locCoord.z
        if goodAfter.sum() > 0:
            raycing.virgin_local_to_global(self.bl, gbf2, self.center,
                                           goodAfter)
    # not intersected rays remain unchanged except their state:
#            notGood = ~goodAfter
#            if notGood.sum() > 0:
#                rs.copy_beam(gbf2, beam, notGood)

    # in global coordinate system:
        raycing.append_to_flow(self.scatter, [gbf2, lbf2],
                               inspect.currentframe())
        gbf2.parent = self
        return gbf2, lbf2  # in global(gb) and local(lb) coordinates

    def local_to_global(self, lb, returnBeam=False, **kwargs):
        dx, dy, dz = 0, 0, 0
        if isinstance(self, DCM):
            is2ndXtal = kwargs.get('is2ndXtal', False)
            if not is2ndXtal:
                pitch = self.pitch + self.bragg
                roll = self.roll + self.positionRoll + self.cryst1roll
                yaw = self.yaw
                dx = self.dx
            else:
                pitch = -self.pitch - self.bragg + self.cryst2pitch +\
                    self.cryst2finePitch
                roll = self.roll + self.cryst2roll - np.pi + self.positionRoll
                yaw = -self.yaw
                dx = -self.dx
                dy = self.cryst2longTransl
                dz = -self.cryst2perpTransl
        else:
            pitch = self.pitch
            roll = self.roll + self.positionRoll
            yaw = self.yaw

        if dx:
            lb.x += dx
        if dy:
            lb.y += dy
        if dz:
            lb.z += dz

        if self.extraPitch or self.extraRoll or self.extraYaw:
            raycing.rotate_beam(
                lb, rotationSequence='-'+self.extraRotationSequence,
                pitch=self.extraPitch, roll=self.extraRoll,
                yaw=self.extraYaw, **kwargs)

        raycing.rotate_beam(lb, rotationSequence='-'+self.rotationSequence,
                            pitch=pitch, roll=roll, yaw=yaw, **kwargs)

        if self.isParametric:
            s, phi, r = self.xyz_to_param(lb.x, lb.y, lb.z)
            oeNormal = list(self.local_n(s, phi))
        else:
            oeNormal = list(self.local_n(lb.x, lb.y))
        roll = self.roll + self.positionRoll +\
            np.arctan2(oeNormal[-3], oeNormal[-1])
        lb.Jss[:], lb.Jpp[:], lb.Jsp[:] =\
            rs.rotate_coherency_matrix(lb, slice(None), roll)
        if hasattr(lb, 'Es'):
            cosY, sinY = np.cos(roll), np.sin(roll)
            lb.Es[:], lb.Ep[:] = raycing.rotate_y(lb.Es, lb.Ep, cosY, sinY)

        if returnBeam:
            retGlo = rs.Beam(copyFrom=lb)
            raycing.virgin_local_to_global(self.bl, retGlo,
                                           self.center, **kwargs)
            return retGlo
        else:
            raycing.virgin_local_to_global(self.bl, lb, self.center, **kwargs)

    def prepare_wave(self, prevOE, nrays, shape='auto', area='auto', rw=None):
        """Creates the beam arrays used in wave diffraction calculations.
        *prevOE* is the diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`.
        *nrays* of samples are randomly distributed over the surface within
        self.limPhysX limits.
        """
        if rw is None:
            from . import waves as rw

        nrays = int(nrays)
        lb = rs.Beam(nrays=nrays, forceState=1, withAmplitudes=True)
        xy = np.random.rand(nrays, 2)
        if shape == 'auto':
            shape = self.shape
        if shape.startswith('ro'):  # round
            dR = (self.limPhysX[1] - self.limPhysX[0]) / 2
            r = xy[:, 0]**0.5 * dR
            phi = xy[:, 1] * 2*np.pi
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            if area == 'auto':
                area = np.pi * dR**2
        elif shape.startswith('re'):  # rect
            dX = self.limPhysX[1] - self.limPhysX[0]
            dY = self.limPhysY[1] - self.limPhysY[0]
            x = xy[:, 0] * dX + self.limPhysX[0]
            y = xy[:, 1] * dY + self.limPhysY[0]
            if area == 'auto':
                area = dX * dY
        else:
            raise ValueError("unknown shape!")

# this works even for a parametric case because we prepare rays started at the
# center of the previous oe and directed towards this oe (self). The found
# intersection points (by reflect) are exact:
        z = self.local_z(x, y)
        lb.x[:] = x
        lb.y[:] = y
        lb.z[:] = z
        self.local_to_global(lb)
        lb.a[:] = lb.x - prevOE.center[0]
        lb.b[:] = lb.y - prevOE.center[1]
        lb.c[:] = lb.z - prevOE.center[2]
        norm = (lb.a**2 + lb.b**2 + lb.c**2)**0.5
        lb.a /= norm
        lb.b /= norm
        lb.c /= norm
        lb.x[:] = prevOE.center[0]
        lb.y[:] = prevOE.center[1]
        lb.z[:] = prevOE.center[2]

        lbn = rs.Beam(nrays=1)
        lbn.b[:] = 0.
        lbn.c[:] = 1.
        self.local_to_global(lbn)
        a = lbn.x - prevOE.center[0]
        b = lbn.y - prevOE.center[1]
        c = lbn.z - prevOE.center[2]
        norm = (a**2 + b**2 + c**2)**0.5
        areaNormalFact = \
            abs(float((a*lbn.a[0] + b*lbn.b[0] + c*lbn.c[0]) / norm))

        waveGlobal, waveLocal = self.reflect(lb)
        good = (waveLocal.state == 1) | (waveLocal.state == 2)
        waveGlobal.filter_by_index(good)
        waveLocal.filter_by_index(good)
        area *= good.sum() / float(len(good))
        waveLocal.area = area
        waveLocal.areaNormal = area * areaNormalFact
        waveLocal.dS = area / float(len(good))
        waveLocal.toOE = self
#        waveLocal.xGlobal = waveGlobal.x
#        waveLocal.yGlobal = waveGlobal.y
#        waveLocal.zGlobal = waveGlobal.z
        rw.prepare_wave(
            prevOE, waveLocal, waveGlobal.x, waveGlobal.y, waveGlobal.z)
        return waveLocal

    def propagate_wave(self, wave=None, beam=None, nrays='auto'):
        """
        Propagates the incoming *wave* through an optical element using the
        Kirchhoff diffraction theorem. Returnes two Beam objects, one in global
        and one in local coordinate systems, which can be
        used correspondingly for the consequent ray and wave propagation
        calculations.

        *wave*: Beam object
            Local beam on the surface of the previous optical element.

        *beam*: Beam object
            Incident global beam, only used for alignment purpose.

        *nrays*: 'auto' or int
            Dimension of the created wave. If 'auto' - the same as the incoming
            wave.


        .. Returned values: beamGlobal, beamLocal
        """
        from . import waves as rw
        waveSize = len(wave.x) if nrays == 'auto' else int(nrays)
#        if wave is None and beam is not None:
#            wave = beam
        prevOE = self.bl.oesDict[wave.parentId]
        print("Diffract on", self.name, " Prev OE:", prevOE.name)
        if self.bl is not None:
            if beam is not None:
                self.bl.auto_align(self, beam)
            elif 'source' in str(type(prevOE)):
                self.bl.auto_align(self, wave)
            else:
                self.bl.auto_align(self, prevOE.local_to_global(
                    wave, returnBeam=True))
        waveOnSelf = self.prepare_wave(prevOE, waveSize, rw=rw)
        if 'source' in str(type(prevOE)):
            beamToSelf = prevOE.shine(wave=waveOnSelf)
            nIS = False
        else:
            beamToSelf = rw.diffract(wave, waveOnSelf)
            nIS = True
        retGlo, retLoc = self.reflect(beamToSelf, noIntersectionSearch=nIS)
        retLoc.parent = self.name
        return retGlo, retLoc

    def _set_t(self, xyz=None, abc=None, surfPhys=None,
               defSize=raycing.maxHalfSizeOfOE):
        if surfPhys is None:
            limMin = -defSize
            limMax = defSize
        else:
            limMin = surfPhys[0] if surfPhys[0] > -np.inf else -defSize
            limMax = surfPhys[1] if surfPhys[1] < np.inf else defSize
        if abc[0] > 0:
            tMin = (limMin-xyz)/abc - raycing.dt
            tMax = (limMax-xyz)/abc + raycing.dt
        else:
            tMin = (limMax-xyz)/abc - raycing.dt
            tMax = (limMin-xyz)/abc + raycing.dt
        return tMin, tMax

    def _bracketing(self, local_n, x, y, z, a, b, c, invertNormal,
                    is2ndXtal=False, isMulti=False, needElevationMap=False,
                    mainPart=slice(None)):
        if is2ndXtal:
            surfPhysX = self.surfPhysX2
            surfPhysY = self.surfPhysY2
        else:
            surfPhysX = self.surfPhysX
            surfPhysY = self.surfPhysY

        try:
            maxa = np.max(abs(a[mainPart]))
            maxb = np.max(abs(b[mainPart]))
            maxc = np.max(abs(c[mainPart]))
        except ValueError:
            maxa, maxb, maxc = 0, 1, 0
        maxMax = max(maxa, maxb, maxc)

        if maxMax == maxa:
            tMin, tMax = self._set_t(x, a, surfPhysX)
        elif maxMax == maxb:
            tMin, tMax = self._set_t(y, b, surfPhysY)
        else:
            tMin, tMax = self._set_t(z, c, defSize=raycing.maxDepthOfOE)
#        tMin[tMin < 0] = 0
        elevation = None
        if isMulti:
            tMin[:] = 0
            tMaxTmp = np.copy(tMax)
            tMax, _, _, _, _ = self.find_intersection(
                local_n, tMin, tMax, x, y, z, a, b, c, invertNormal,
                derivOrder=1)
            if needElevationMap:
                elevation = \
                    self.find_dz(None, tMax, x, y, z, a, b, c, invertNormal)
            tMin = tMax + raycing.ds
            tMax = tMaxTmp
        else:
            pass
#            if needElevationMap:
#                elevation = \
#                    self.find_dz(None, tMin, x, y, z, a, b, c, invertNormal)
        return tMin, tMax, elevation

    def _grating_deflection(
            self, goodN, lb, g, oeNormal, beamInDotNormal, order=1, sig=None):
        beamInDotG = lb.a[goodN]*g[0] + lb.b[goodN]*g[1] + lb.c[goodN]*g[2]
        G2 = g[0]**2 + g[1]**2 + g[2]**2
        locOrder = order if isinstance(order, int) else \
            np.array(order)[np.random.randint(len(order), size=goodN.sum())]
        lb.order = np.zeros(len(lb.a))
        lb.order[goodN] = locOrder
        orderLambda = locOrder * CH / lb.E[goodN] * 1e-7

        u = beamInDotNormal**2 - 2*beamInDotG*orderLambda - G2*orderLambda**2
#        u[u < 0] = 0
        gs = np.sign(beamInDotNormal) if sig is None else sig
        dn = beamInDotNormal + gs*np.sqrt(abs(u))
        a_out = lb.a[goodN] - oeNormal[-3]*dn + g[0]*orderLambda
        b_out = lb.b[goodN] - oeNormal[-2]*dn + g[1]*orderLambda
        c_out = lb.c[goodN] - oeNormal[-1]*dn + g[2]*orderLambda
        norm = (a_out**2 + b_out**2 + c_out**2)**0.5
        return a_out/norm, b_out/norm, c_out/norm

    def _reportNaN(self, x, strName):
        nanSum = np.isnan(x).sum()
        if nanSum > 0:
            print("{0} NaN rays in array {1} in optical element {2}!".format(
                  nanSum, strName, self.name))

    def local_n_random(self, bLength, theta=0, phi=0, chi=0):
        a = np.zeros(bLength)
        b = np.zeros(bLength)
        c = np.ones(bLength)

        cos_range = np.random.rand(bLength)  # * 2**-0.5
        y_angle = np.arccos(cos_range)
        z_angle = (chi[1]-chi[0]) * np.random.rand(bLength) + chi[0]

        a, c = raycing.rotate_y(a, c, np.cos(y_angle), np.sin(y_angle))
        a, b = raycing.rotate_z(a, b, np.cos(z_angle), np.sin(z_angle))
#        b, c = raycing.rotate_x(a, b, np.cos(x_angle), np.sin(x_angle))
        norm = np.sqrt(a**2 + b**2 + c**2)
        a /= norm
        b /= norm
        c /= norm
        return [a, b, c]

    def _reflect_crystal_cl(self, goodN, lb, matcr, oeNormal):
        DW = self.cl_precisionF(matcr.factDW)
        thickness = self.cl_precisionF(0 if matcr.t is None else matcr.t)
        geometry = np.int32(matcr.geometry)
        if matcr.tK is not None:
            temperature = self.cl_precisionF(matcr.tK)
        else:
            temperature = self.cl_precisionF(0)
        if not np.all(np.array(matcr.atoms) == 14):
            temperature = self.cl_precisionF(0)

        lenGood = np.int32(len(lb.E[goodN]))
        bOnes = np.ones(lenGood)

        iHKL = np.zeros(4, dtype=np.int32)
        iHKL[0:3] = np.array(matcr.hkl, dtype=np.int32)

        Nel = len(matcr.elements)
        elements_in = np.zeros((Nel, 8), dtype=self.cl_precisionF)
        E_in = np.zeros((Nel, 300), dtype=self.cl_precisionF)
        f1_in = np.zeros((Nel, 300), dtype=self.cl_precisionF)
        f2_in = np.zeros((Nel, 300), dtype=self.cl_precisionF)
        f0_in = np.zeros((Nel, 11), dtype=self.cl_precisionF)
        elements_in[:, 0:3] = matcr.atomsXYZ
        elements_in[:, 4] = matcr.atomsFraction

        for iNel in range(Nel):
            f_len = len(matcr.elements[iNel].E)
            elements_in[iNel, 5] = matcr.elements[iNel].Z
            elements_in[iNel, 6] = matcr.elements[iNel].mass
            elements_in[iNel, 7] = self.cl_precisionF(f_len-1)
            E_in[iNel, 0:f_len] = matcr.elements[iNel].E
            f1_in[iNel, 0:f_len] = matcr.elements[iNel].f1
            f2_in[iNel, 0:f_len] = matcr.elements[iNel].f2
            f0_in[iNel, :] = matcr.elements[iNel].f0coeffs

        lattice_in = np.array([matcr.a, matcr.b, matcr.c, 0,
                               matcr.alpha, matcr.beta, matcr.gamma, 0],
                              dtype=self.cl_precisionF)
        calctype = 0
        if matcr.kind == "powder":
            calctype = 5
        elif matcr.kind == "monocrystal":
            calctype = 10 + matcr.Nmax
        elif matcr.kind == "crystal harmonics":
            calctype = 100 + matcr.Nmax

        scalarArgs = [np.int32(calctype), iHKL, DW, thickness, temperature,
                      geometry, np.int32(Nel), lattice_in]

        slicedROArgs = [self.cl_precisionF(lb.a[goodN]),  # a_in
                        self.cl_precisionF(lb.b[goodN]),  # b_in
                        self.cl_precisionF(lb.c[goodN]),  # c_in
                        self.cl_precisionF(lb.E[goodN]),  # Energy
                        self.cl_precisionF(oeNormal[0]*bOnes),  # planeNormalX
                        self.cl_precisionF(oeNormal[1]*bOnes),  # planeNormalY
                        self.cl_precisionF(oeNormal[2]*bOnes),  # planeNormalZ
                        self.cl_precisionF(oeNormal[-3]*bOnes),  # surfNormalX
                        self.cl_precisionF(oeNormal[-2]*bOnes),  # surfNormalY
                        self.cl_precisionF(oeNormal[-1]*bOnes)]  # surfNormalZ

        slicedROArgs.extend([self.cl_precisionF(np.random.rand(lenGood))])

        nonSlicedROArgs = [elements_in.flatten(),  # elements
                           f0_in.flatten(),  # f0
                           E_in.flatten(),   # E_in
                           f1_in.flatten(),  # f1
                           f2_in.flatten()]  # f2

        slicedRWArgs = [np.zeros(lenGood, dtype=self.cl_precisionC),  # reflS
                        np.zeros(lenGood, dtype=self.cl_precisionC),  # reflP
                        np.zeros(lenGood, dtype=self.cl_precisionF),  # a_out
                        np.zeros(lenGood, dtype=self.cl_precisionF),  # b_out
                        np.zeros(lenGood, dtype=self.cl_precisionF)]  # c_out

        curveS, curveP, a_out, b_out, c_out = self.ucl.run_parallel(
            'reflect_crystal', scalarArgs, slicedROArgs, nonSlicedROArgs,
            slicedRWArgs, None, lenGood)
        return a_out, b_out, c_out, curveS, curveP

    def _mosaic_normal(self, mat, oeNormal, beamInDotNormal, lb, goodN):
        E = lb.E[goodN]
        theta = mat.get_Bragg_angle(E) - mat.get_dtheta(E)

        sinTheta = np.sin(theta)
        cosTheta = (1 - sinTheta**2)**0.5

        cosAlpha = np.abs(beamInDotNormal)
        sinAlpha = (1 - cosAlpha**2)**0.5

        # rotate the crystallite normal to meet the Bragg condition
        cn = cosTheta / sinAlpha
        ck = sinTheta + cn*beamInDotNormal
        n1a = cn*oeNormal[0] - ck*lb.a[goodN]
        n1b = cn*oeNormal[1] - ck*lb.b[goodN]
        n1c = cn*oeNormal[2] - ck*lb.c[goodN]

        # this simple solution does the same job as the one in Shadow:
        phi = np.random.normal(0, mat.mosaicity, len(sinTheta))
        # this is the Shadow's solution:
#        import scipy.stats
#        ss = sinAlpha*sinTheta  # sinTheta = cosThetaD
#        cc = cosAlpha*cosTheta  # cosTheta = sinThetaD
#        sinAlphaMinusThetaD = np.abs(ss - cc).min()  # sin(alpha - thetaD)
#        lower = np.arcsin(sinAlphaMinusThetaD)
#        sinAlphaPlusThetaD = np.abs(ss + cc).max()  # sin(alpha + thetaD)
#        if sinAlphaPlusThetaD > 1:
#            sinAlphaPlusThetaD = 1 - 1e-20
#        upper = np.arcsin(sinAlphaPlusThetaD)
#        phi = scipy.stats.truncnorm.rvs(
#            lower/mat.mosaicity, upper/mat.mosaicity,
#            loc=0, scale=mat.mosaicity, size=len(sinTheta))

        cosPhi = np.cos(phi)
        # rotating around in-beam does the same job the Shadow's solution
        cosBeta = cosPhi
        # this is the Shadow's solution:
#        ctanTheta = cosTheta / sinTheta
#        tanAlpha = abs(sinAlpha / cosAlpha)
#        cosBeta = (ctanTheta**2 + tanAlpha**2 - sinTheta**-2 - cosAlpha**-2 +
#                   2*cosPhi/(sinTheta*cosAlpha)) / (2*ctanTheta*tanAlpha)
#        cosBeta[cosBeta > 1] = 1 - 1e-20

        sinBeta = (1 - cosBeta**2)**0.5
        signs = np.random.randint(2, size=len(sinTheta))
        signs[signs == 0] = -1
        sinBeta *= signs
        ocosBeta = 1 - cosBeta

        # en.wikipedia.org/wiki/Rodrigues%27_rotation_formula, made with sympy
        kx, ky, kz = lb.a[goodN], lb.b[goodN], lb.c[goodN]
        nra = n1a*(-ky**2*ocosBeta - kz**2*ocosBeta + 1) +\
            n1b*(kx*ky*ocosBeta - kz*sinBeta) +\
            n1c*(kx*kz*ocosBeta + ky*sinBeta)
        nrb = n1a*(kx*ky*ocosBeta + kz*sinBeta) +\
            n1b*(-kx**2*ocosBeta - kz**2*ocosBeta + 1) +\
            n1c*(-kx*sinBeta + ky*kz*ocosBeta)
        nrc = n1a*(kx*kz*ocosBeta - ky*sinBeta) +\
            n1b*(kx*sinBeta + ky*kz*ocosBeta) +\
            n1c*(-kx**2*ocosBeta - ky**2*ocosBeta + 1)
        beamInDotNormalN = lb.a[goodN]*nra + lb.b[goodN]*nrb + lb.c[goodN]*nrc

        return [nra, nrb, nrc], beamInDotNormalN

    def _mosaic_length(self, mat, beamInDotNormal, lb, goodN):
        Qs, Qp, thetaB = mat.get_kappa_Q(lb.E[goodN])[2:5]  # in cm^-1
        norm = lb.Jss[goodN] + lb.Jpp[goodN]
        norm[norm == 0] = 1.
        Q = (Qs*lb.Jss[goodN] + Qp*lb.Jpp[goodN]) / norm
        delta = np.arcsin(np.abs(beamInDotNormal)) - thetaB
        w = np.exp(-0.5*delta**2 / mat.mosaicity**2) / (SQRT2PI*mat.mosaicity)
        rate = w*Q  # in cm^-1
        rate[rate <= 1e-3] = 1e-3
        length = np.random.exponential(10./rate, size=len(Qs))  # in mm
        if mat.t:
            through = length*beamInDotNormal > mat.t
            length[through] = mat.t / beamInDotNormal[through]
        else:
            through = None
        lb.x[goodN] += lb.olda * length
        lb.y[goodN] += lb.oldb * length
        lb.z[goodN] += lb.oldc * length
        return length, through

    def _reflect_local(
        self, good, lb, vlb, pitch, roll, yaw, dx=None, dy=None, dz=None,
        local_z=None, local_n=None, local_g=None, fromVacuum=True,
        material=None, is2ndXtal=False, needElevationMap=False,
            noIntersectionSearch=False, isMulti=False, fluo=None):
        """Finds the intersection points of rays in the beam *lb* indexed by
        *good* array. *vlb* is the same beam in virgin local system.
        *pitch, roll, yaw* determine the transformation between true local and
        virgin local coordinates.
        *local_n* gives the normal (two normals (h, surface) if for crystal).
        *local_g* for a grating gives the local reciprocal groove vector in
        1/mm. *fromVacuum* tells the beam direction for the vacuum-OE
        interface. *material* is an instance of :class:`Material` or
        :class:`Crystal` or its derivatives. Depending on the geometry used, it
        must have either the method :meth:`get_refractive_index` or the
        :meth:`get_amplitude`."""

        def _get_asymmetric_reflection_grating(
                _gNormal, _oeNormal, _beamInDotSurfaceNormal):
            normalDotSurfNormal = _oeNormal[0]*_oeNormal[-3] +\
                _oeNormal[1]*_oeNormal[-2] + _oeNormal[2]*_oeNormal[-1]
            # note:
            # _oeNormal[0:3] is n_B
            # _oeNormal[-3:] is n_s
            # normalDotSurfNormal is dot(n_B, n_s)
            # |vec(n_B) - dot(n_B, n_s)*vec(n_s)| = sin(alpha)
#            wH = matSur.get_refractive_correction(
#                lb.E[goodN], abs(_beamInDotSurfaceNormal))
            wH = 0
            gNormalCryst = np.asarray((
                (_oeNormal[0]-normalDotSurfNormal*_oeNormal[-3]) * (1 + wH),
                (_oeNormal[1]-normalDotSurfNormal*_oeNormal[-2]) * (1 + wH),
                (_oeNormal[2]-normalDotSurfNormal*_oeNormal[-1]) * (1 + wH)),
                order='F') / (matSur.d * 1e-7)
            if matSur.geom.endswith('Fresnel'):
                if isinstance(self.order, int):
                    locOrder = self.order
                else:
                    locOrder = np.array(self.order)[np.random.randint(
                        len(self.order), size=goodN.sum())]
                if _gNormal is None:
                    _gNormal = local_g(lb.x[goodN], lb.y[goodN])
                _gNormal = np.asarray(_gNormal, order='F') * locOrder
                _gNormal[0] += gNormalCryst[0]
                _gNormal[1] += gNormalCryst[1]
                _gNormal[2] += gNormalCryst[2]
            else:
                _gNormal = gNormalCryst

            sg = 1 if matSur.geom.startswith('Laue') else -1
            res = self._grating_deflection(
                goodN, lb, _gNormal, _oeNormal, _beamInDotSurfaceNormal, 1, sg)
            return res

# rotate the world around the mirror.
# lb is truly local coordinates whereas vlb is in virgin local coordinates:
        if local_n is None:
            local_n = self.local_n
        raycing.rotate_beam(
            lb, good, rotationSequence=self.rotationSequence,
            pitch=-pitch, roll=-roll, yaw=-yaw)
        if self.extraPitch or self.extraRoll or self.extraYaw:
            raycing.rotate_beam(
                lb, good, rotationSequence=self.extraRotationSequence,
                pitch=-self.extraPitch, roll=-self.extraRoll,
                yaw=-self.extraYaw)
        if dx:
            lb.x[good] -= dx
        if dy:
            lb.y[good] -= dy
        if dz:
            lb.z[good] -= dz

# x, y, z:
        if fromVacuum:
            invertNormal = 1
        else:
            invertNormal = -1
#        mainPartForBracketing = lb.state[good] > 0
        mainPartForBracketing = lb.state[good] == 1
        tMin = np.zeros_like(lb.x)
        tMax = np.zeros_like(lb.x)
        tMin[good], tMax[good], elev = self._bracketing(
            local_n, lb.x[good], lb.y[good], lb.z[good], lb.a[good],
            lb.b[good], lb.c[good], invertNormal, is2ndXtal, isMulti=isMulti,
            needElevationMap=needElevationMap, mainPart=mainPartForBracketing)

        if needElevationMap and elev:
            lb.elevationD[good] = elev[0]
            if self.isParametric:
                tX, tY, tZ = self.param_to_xyz(elev[1], elev[2], elev[3])
            else:
                tX, tY, tZ = elev[1], elev[2], elev[3]
            lb.elevationX[good] = tX
            lb.elevationY[good] = tY
            lb.elevationZ[good] = tZ

        _lost = None
        if noIntersectionSearch:
            # lb.x[good], lb.y[good], lb.z[good] unchanged
            tMax[good] = 0.
            if self.isParametric:
                lb.x[good], lb.y[good], lb.z[good] = self.xyz_to_param(
                    lb.x[good], lb.y[good], lb.z[good])
        else:
            if True:  # self.cl_ctx is None:
                res_find = \
                    self.find_intersection(
                        local_z, tMin[good], tMax[good],
                        lb.x[good], lb.y[good], lb.z[good],
                        lb.a[good], lb.b[good], lb.c[good], invertNormal)
                tMax[good], lb.x[good], lb.y[good], lb.z[good] = res_find[:4]
                if len(res_find) > 4:
                    _lost = res_find[4]
            else:  # To be refactored in future versions
                tMax[good], lb.x[good], lb.y[good], lb.z[good] = \
                    self.find_intersection_CL(
                        local_z, tMin[good], tMax[good],
                        lb.x[good], lb.y[good], lb.z[good],
                        lb.a[good], lb.b[good], lb.c[good], invertNormal)
# state:
# the distortion part has moved from here to find_dz
        if self.isParametric:
            # z_distorted = self.local_r_distorted(lb.x[good], lb.y[good])
            tX, tY, tZ = self.param_to_xyz(lb.x[good], lb.y[good], lb.z[good])
        else:
            # z_distorted = self.local_z_distorted(lb.x[good], lb.y[good])
            tX, tY, tZ = lb.x[good], lb.y[good], lb.z[good]
#        if z_distorted is not None:
#            lb.z[good] += z_distorted

        if self.use_rays_good_gn:
            lb.state[good], gNormal = self.rays_good_gn(tX, tY, tZ)
        else:
            gNormal = None
            lb.state[good] = self.rays_good(tX, tY, is2ndXtal)
        if _lost is not None:
            lb.state[np.where(good)[0][_lost]] = self.lostNum

#        goodN = (lb.state == 1) | (lb.state == 2)
        goodN = (lb.state == 1)
# normal at x, y, z:
        goodNsum = goodN.sum()
        needMosaicity = False
        if goodNsum > 0:
            lb.path[goodN] += tMax[goodN]

            toWhere = 0  # 0: reflect, 1: refract, 2: pass straight
            if material is not None:
                if raycing.is_sequence(material):
                    matSur = material[self.curSurface]
                else:
                    matSur = material
                if matSur.kind == 'auto':
                    self.assign_auto_material_kind(matSur)
                if matSur.kind in ('plate', 'lens'):
                    toWhere = 1
                elif matSur.kind in ('crystal', 'multilayer'):
                    if matSur.kind == 'crystal':
                        if matSur.mosaicity:
                            needMosaicity = True
                    if matSur.geom.endswith('transmitted'):
                        toWhere = 2
                elif matSur.kind == 'grating':
                    toWhere = 3
                elif matSur.kind == 'FZP':
                    toWhere = 4
                elif matSur.kind == 'powder':
                    toWhere = 5
                elif matSur.kind == 'monocrystal':
                    toWhere = 6
                elif matSur.kind == 'crystal harmonics':
                    toWhere = 7
            if toWhere == 5:
                oeNormal = list(
                    self.local_n_random(len(lb.E[goodN]), chi=matSur.chi))
#                n = matSur.get_refractive_index(lb.E[goodN])
#                mu = abs(n.imag) * lb.E[goodN] / CHBAR * 2e8  # 1/cm
#                att = np.exp(-mu * tMax[goodN] * 0.1)
                depth = np.random.rand(len(lb.a[goodN])) * matSur.t
                lb.x[goodN] += lb.a[goodN] * depth
                lb.y[goodN] += lb.b[goodN] * depth
                lb.z[goodN] += lb.c[goodN] * depth
            else:
                oeNormal = list(local_n(lb.x[goodN], lb.y[goodN]))

            n_distorted = self.local_n_distorted(lb.x[goodN], lb.y[goodN])
            if n_distorted is not None:
                if len(n_distorted) == 2:
                    cosX, sinX = np.cos(n_distorted[0]), np.sin(n_distorted[0])
                    oeNormal[-2], oeNormal[-1] = raycing.rotate_x(
                        oeNormal[-2], oeNormal[-1], cosX, sinX)
                    cosY, sinY = np.cos(n_distorted[1]), np.sin(n_distorted[1])
                    oeNormal[-3], oeNormal[-1] = raycing.rotate_y(
                        oeNormal[-3], oeNormal[-1], cosY, sinY)
                elif len(n_distorted) == 3:
                    oeNormal[-3] += n_distorted[0]
                    oeNormal[-2] += n_distorted[1]
                    oeNormal[-1] += n_distorted[2]
                    norm = (oeNormal[-3]**2 + oeNormal[-2]**2 +
                            oeNormal[-1]**2)**0.5
                    oeNormal[-3] /= norm
                    oeNormal[-2] /= norm
                    oeNormal[-1] /= norm
                else:
                    raise ValueError(
                        "wrong length returned by 'local_n_distorted'")
            if toWhere < 5:
                isAsymmetric = len(oeNormal) == 6
                oeNormal = np.asarray(oeNormal, order='F')
                beamInDotNormal = lb.a[goodN]*oeNormal[0] +\
                    lb.b[goodN]*oeNormal[1] + lb.c[goodN]*oeNormal[2]
                lb.theta = np.zeros_like(lb.x)
                beamInDotNormal[beamInDotNormal < -1] = -1
                beamInDotNormal[beamInDotNormal > 1] = 1
                lb.theta[goodN] = np.arccos(beamInDotNormal) - np.pi/2

                if isAsymmetric:
                    beamInDotSurfaceNormal = lb.a[goodN]*oeNormal[-3] +\
                        lb.b[goodN]*oeNormal[-2] + lb.c[goodN]*oeNormal[-1]
                else:
                    beamInDotSurfaceNormal = beamInDotNormal

                if needMosaicity:
                    oeNormalN, beamInDotNormalN = self._mosaic_normal(
                        matSur, oeNormal, beamInDotNormal, lb, goodN)
                    if isAsymmetric:
                        o1 = np.ones_like(lb.a[goodN])
                        oeNormalN.extend([
                            oeNormal[-3]*o1, oeNormal[-2]*o1, oeNormal[-1]*o1])
                    oeNormal = np.asarray(oeNormalN, order='F')
                    beamInDotNormalOld = beamInDotNormal
                    beamInDotNormal = beamInDotNormalN
# direction:
            if toWhere in [3, 4]:  # grating, FZP
                if gNormal is None:
                    if self.isParametric:
                        tXN, tYN = self.param_to_xyz(
                            lb.x[goodN], lb.y[goodN], lb.z[goodN])[0:2]
                    else:
                        tXN, tYN = lb.x[goodN], lb.y[goodN]
                    if local_g is None:
                        local_g = self.local_g
                    gNormal = np.asarray(local_g(tXN, tYN), order='F')
                giveSign = 1 if toWhere == 4 else -1
                lb.a[goodN], lb.b[goodN], lb.c[goodN] =\
                    self._grating_deflection(goodN, lb, gNormal, oeNormal,
                                             beamInDotSurfaceNormal,
                                             self.order, giveSign)
            elif toWhere in [0, 2]:  # reflect, straight
                useAsymmetricNormal = False
                if material is not None:
                    if matSur.kind in ('crystal', 'multilayer') and\
                            toWhere == 0 and (not needMosaicity):
                        useAsymmetricNormal = True

                if useAsymmetricNormal:
                    a_out, b_out, c_out = _get_asymmetric_reflection_grating(
                        gNormal, oeNormal, beamInDotSurfaceNormal)
                else:
                    a_out = lb.a[goodN] - oeNormal[0]*2*beamInDotNormal
                    b_out = lb.b[goodN] - oeNormal[1]*2*beamInDotNormal
                    c_out = lb.c[goodN] - oeNormal[2]*2*beamInDotNormal

                calcBorrmann = False
                if material is not None:
                    if matSur.kind in ['crystal'] and \
                        matSur.geom.startswith('Laue') and \
                            matSur.calcBorrmann:
                        calcBorrmann = True
                if calcBorrmann:
                    beamOutDotSurfaceNormal = a_out * oeNormal[-3] + \
                        b_out * oeNormal[-2] + c_out * oeNormal[-1]

                    """
                    if self.crossSection.startswith('para'):

                        This block was used to estimate the influence
                        of the surface curvature on focusing. I will
                        enable it later to preserve the exact phase for
                        the wave propagation.

                        paraA = 0.5/(self.R - matSur.t)
                        paraB = -np.divide(lb.c, lb.b)
                        paraC = -paraB*lb.y - lb.z - matSur.t
                        paraD = np.sqrt(paraB**2 - 4*paraA*paraC)
                        yOut01 = (-paraB + paraD) / 2 / paraA
                        yOut02 = (-paraB - paraD) / 2 / paraA
                        print oeNormal
                        print "yOut01, yOut02", yOut01, yOut02
                        paraB = -np.divide(c_out, b_out)
                        paraC = -paraB*lb.y - lb.z - matSur.t
                        paraD = np.sqrt(paraB**2 - 4*paraA*paraC)
                        yOutH1 = (-paraB + paraD) / 2 / paraA
                        yOutH2 = (-paraB - paraD) / 2 / paraA
                        print "yOutH1, yOutH2",yOutH1, yOutH2
                        xOut0 = lb.x + (yOut02 - lb.y)*lb.a/lb.b
                        xOutH = lb.x + (yOutH2 - lb.y)*a_out/b_out
                        print "xOut0, xOutH", xOut0, xOutH
                        zOut0 = yOut02**2/2./(self.R - matSur.t) +\
                            matSur.t
                        zOutH = yOut02**2/2./(self.R - matSur.t) +\
                            matSur.t
                        print "zOut0, zOutH", zOut0, zOutH
                    """

#                        Getting the thickness projections in forward
#                        and diffracted directions
                    t0 = -matSur.t / beamInDotSurfaceNormal
                    tH = -matSur.t / beamOutDotSurfaceNormal
#                        Find intersection of S0 and output surface
                    point0x = lb.x + lb.a * t0
                    point0y = lb.y + lb.b * t0
#                        Find intersection of Sh and output surface
                    pointHx = lb.x + a_out * tH
                    pointHy = lb.y + b_out * tH

                    pointOnFan =\
                        matSur.get_Borrmann_out(
                            goodN, oeNormal,
                            lb, a_out, b_out, c_out,
                            alphaAsym=self.alpha,
                            Rcurvmm=self.R if 'R' in
                            self.__dict__.keys() else None,
                            ucl=self.ucl,
                            useTT=matSur.useTT)

                    pointOutX = point0x * (1. - pointOnFan) +\
                        pointOnFan * pointHx
                    pointOutY = point0y * (1. - pointOnFan) +\
                        pointOnFan * pointHy

                    lb.x = pointOutX
                    deltaY = point0y * (1. - pointOnFan) +\
                        pointOnFan * lb.y
                    lb.y = pointOutY

                    if matSur.t is not None:
                        tmpR = self.R
                        self.R -= matSur.t
                        lb.z = -self.local_z(lb.x, lb.y) - matSur.t
                        self.R = self.R - (1 - pointOnFan) * matSur.t
                        oeNormalOut = list(self.local_n(lb.x, deltaY))
                        a_out, b_out, c_out = \
                            _get_asymmetric_reflection_grating(
                                gNormal, oeNormalOut, beamInDotSurfaceNormal)
                        self.R = tmpR

                if toWhere == 0:  # reflect
                    if needMosaicity:
                        lb.olda = np.array(lb.a[goodN])
                        lb.oldb = np.array(lb.b[goodN])
                        lb.oldc = np.array(lb.c[goodN])
                    lb.a[goodN] = a_out
                    lb.b[goodN] = b_out
                    lb.c[goodN] = c_out
#                print('after.a', lb.a)
#                print('after.b', lb.b)
#                print('after.c', lb.c)
            elif toWhere == 1:  # refract
                refractive_index = \
                    matSur.get_refractive_index(lb.E[goodN]).real
                if fromVacuum:
                    n1overn2 = 1. / refractive_index
                else:
                    n1overn2 = refractive_index
                signN = np.sign(-beamInDotNormal)
                n1overn2cosTheta1 = -n1overn2 * beamInDotNormal
                cosTheta2 = signN * \
                    np.sqrt(1 - n1overn2**2 + n1overn2cosTheta1**2)
                dn = (n1overn2cosTheta1 - cosTheta2)
                lb.a[goodN] = lb.a[goodN] * n1overn2 + oeNormal[0]*dn
                lb.b[goodN] = lb.b[goodN] * n1overn2 + oeNormal[1]*dn
                lb.c[goodN] = lb.c[goodN] * n1overn2 + oeNormal[2]*dn
            elif toWhere in [5, 6, 7]:  # powder, 'monocrystal', 'harmonics'
                trc0 = time.time()
                aP, bP, cP, rasP, rapP =\
                    self._reflect_crystal_cl(goodN, lb, matSur, oeNormal)
                print('Reflect_crystal completed in {0} s'.format(
                    time.time() - trc0))
#                lb.concatenate(lb)
                lb.a[goodN] = aP
                lb.b[goodN] = bP
                lb.c[goodN] = cP
                goodN = (lb.state == 1) | (lb.state == 2)
#                good = np.append(good, good)
            else:  # pass straight, do nothing
                pass
# flux:
            findReflectivity = False
            if material is not None:
                if hasattr(matSur, 'get_amplitude'):
                    findReflectivity = True
                if toWhere in [5, 6, 7]:  # powder,
                    findReflectivity = True

            # rotate coherency matrix:
            # {np.arctan2: 0./0.: =0, 1./0.: =pi/2}
            rollAngle = roll + np.arctan2(oeNormal[-3], oeNormal[-1])
            localJ = rs.rotate_coherency_matrix(lb, goodN, -rollAngle)
            if hasattr(lb, 'Es'):
                cosY, sinY = np.cos(rollAngle), np.sin(rollAngle)
                lb.Es[goodN], lb.Ep[goodN] = raycing.rotate_y(
                    lb.Es[goodN], lb.Ep[goodN], cosY, -sinY)

            if findReflectivity:
                if toWhere in [5, 6, 7]:  # powder,
                    refl = rasP, rapP
                elif matSur.kind == 'crystal':
                    beamOutDotSurfaceNormal = a_out*oeNormal[-3] + \
                        b_out*oeNormal[-2] + c_out*oeNormal[-1]
                    if needMosaicity:
                        refl = matSur.get_amplitude_mosaic(
                            lb.E[goodN], beamInDotSurfaceNormal,
                            beamOutDotSurfaceNormal, beamInDotNormalOld)
                    elif matSur.useTT:
                        refl = matSur.get_amplitude_TT(
                            lb.E[goodN], beamInDotSurfaceNormal,
                            beamOutDotSurfaceNormal, beamInDotNormal,
                            alphaAsym=self.alpha,
                            Rcurvmm=self.R if 'R' in self.__dict__.keys()
                            else None,
                            ucl=self.ucl)
                    else:
                        refl = matSur.get_amplitude(
                            lb.E[goodN], beamInDotSurfaceNormal,
                            beamOutDotSurfaceNormal, beamInDotNormal)
                elif matSur.kind == 'multilayer':
                    refl = matSur.get_amplitude(
                        lb.E[goodN], beamInDotSurfaceNormal,
                        lb.x[goodN], lb.y[goodN],
                        ucl=self.ucl)
                else:  # 'mirror', 'thin mirror', 'plate', 'lens', 'grating'
                    hasEfficiency = False
                    if hasattr(matSur, 'efficiency'):
                        if (matSur.kind in ('grating', 'FZP')) and\
                                (matSur.efficiency is not None):
                            hasEfficiency = True
                    if hasEfficiency:
                        refl = matSur.get_grating_efficiency(lb, goodN)
                    else:
                        refl = matSur.get_amplitude(
                            lb.E[goodN], beamInDotNormal, fromVacuum)
            else:
                refl = 1., 1.
            ras, rap = refl[0], refl[1]
            nanSum = np.isnan(ras).sum()
            if nanSum > 0:
                ras[np.isnan(ras)] = 0.
#                    self._reportNaN(ras, 'ras')
            nanSum = np.isnan(rap).sum()
            if nanSum > 0:
                rap[np.isnan(rap)] = 0.
#                    self._reportNaN(rap, 'rap')

            lb.Jss[goodN] = (localJ[0] * ras * np.conjugate(ras)).real
            lb.Jpp[goodN] = (localJ[1] * rap * np.conjugate(rap)).real
            lb.Jsp[goodN] = localJ[2] * ras * np.conjugate(rap)
#                self._reportNaN(lb.Jss[goodN], 'lb.Jss[goodN]')
#                self._reportNaN(lb.Jpp[goodN], 'lb.Jpp[goodN]')
#                self._reportNaN(lb.Jsp[goodN], 'lb.Jsp[goodN]')
            if hasattr(lb, 'Es'):
                lb.Es[goodN] *= ras
                lb.Ep[goodN] *= rap

            if fluo:
                beamLen = len(lb.E[goodN])
                depth = np.random.rand(beamLen) * self.t if hasattr(self, 't')\
                    else 0
                lb.x[goodN] -= lb.a[goodN] / lb.c[goodN] * depth
                lb.y[goodN] -= lb.b[goodN] / lb.c[goodN] * depth
                lb.z[goodN] -= depth
#                mu_total = matSur.get_absorption_coefficient(lb.E[goodN])
                mu_total = refl[2]
                rayDepth = np.sqrt(lb.x[goodN]**2 + lb.y[goodN]**2 +
                                   lb.z[goodN]**2)
                att = np.exp(-mu_total*0.1*rayDepth)
                dphi = 10
                dtheta = 10

                phi = np.array([-1, 1]) * np.radians(dphi)  # roll, y_angle
                theta = np.array([-1, 1]) * np.radians(dtheta) +\
                    np.radians(45)  # pitch, x_angle

                fl_c = np.ones(beamLen)
                fl_a = np.zeros_like(fl_c)
                fl_b = np.zeros_like(fl_c)

                y_angle = np.random.uniform(phi[0], phi[1], size=beamLen)
                x_angle = -np.random.uniform(theta[0], theta[1], size=beamLen)

                fl_b, fl_c = raycing.rotate_x(fl_b, fl_c, np.cos(x_angle),
                                              np.sin(x_angle))
                fl_a, fl_c = raycing.rotate_y(fl_a, fl_c, np.cos(y_angle),
                                              np.sin(y_angle))

                normDir = np.sqrt(fl_a*fl_a + fl_b*fl_b + fl_c*fl_c)
                fl_a /= normDir
                fl_b /= normDir
                fl_c /= normDir

                if isinstance(fluo, tuple):  # single fluorescence line
                    # flLine: [energy, edgeName, absIntensity, lineWidth]
                    (flElement, flLine) = fluo
                    mu_pe = matSur.get_photoabsorption_coefficient(
                        lb.E[goodN])[flElement]
                    noFl = lb.E[goodN] < flLine[1]
                    beamLen = len(lb.E[goodN])
                    if isinstance(flLine[3], (list, tuple)):
                        flLine[3] = 0
                    lb.E[goodN] = flLine[0] +\
                        flLine[3] * np.random.standard_cauchy(beamLen)

                    lb.Jss[goodN] *= flLine[2] * mu_pe
                    lb.Jpp[goodN] *= flLine[2] * mu_pe
                    lb.Jsp[goodN] = 0  # flLine[2] * mu_pe
                    lb.state[(lb.E <= flLine[0] - flLine[3]*100) |
                             (lb.E > flLine[0] + flLine[3]*100)] = self.lostNum
                    lb.state[noFl] = self.lostNum
                else:  # 'inelastic' or 'elastic'
                    cosTheta = fl_a*lb.a[goodN] + fl_b*lb.b[goodN] +\
                        fl_c*lb.c[goodN]  # in-plane angle, "vertical" pol.
                    fl_c2 = fl_c*fl_c
                    # cos(dRoll) = fl_c; sin(dRoll) = sqrt(1-fl_c2)
                    # cos(2*dRoll) = 2*fl_c2-1
                    # sin(2*dRoll) = 2*fl_c*sqrt(1-fl_c2)
                    sin2dRoll = 2*fl_c*np.sqrt(1 - fl_c2)
                    cos2dRoll = 2*fl_c2 - 1
                    # converting coherency matrix to Stokes vector
                    S0 = lb.Jss[goodN] + lb.Jpp[goodN]
                    S1 = lb.Jss[goodN] - lb.Jpp[goodN]
                    S2 = 2*lb.Jsp[goodN].real
                    S3 = 2*lb.Jsp[goodN].imag
                    # adjusting for the scattering plane roll
                    S1prime = S2*sin2dRoll + S1*cos2dRoll
                    S2prime = -S1*sin2dRoll + S2*cos2dRoll

                    if fluo == 'inelastic':  # Compton scattering
                        PEprime = 1./(1. +
                                      lb.E[goodN]*(1. - cosTheta)*1e-6/M0C2)
                        PEprimeSum = PEprime + 1./PEprime
                        lb.E[goodN] = lb.E[goodN] * PEprime
                        C0 = 0.25 * np.pi * (PEprime)**2 * R0**2 * 1e-16

                        lb.Jss[goodN] = matSur.sigma_to_mu(
                            0.5 * C0 * (S0 * PEprimeSum + S1prime))
                        lb.Jpp[goodN] = matSur.sigma_to_mu(
                            0.5 * C0 * (S0 * (-2*(1-cosTheta**2)+PEprimeSum) +
                                        S1prime * (1 - 2*cosTheta**2)))
                        lb.Jsp[goodN] = matSur.sigma_to_mu(
                            C0 * (S2prime * cosTheta + S3 * (
                                cosTheta + 0.5 * cosTheta * (PEprimeSum - 2))))
                        if False:  # analytic solution for total cross-section
                            eRel = lb.E[goodN][0] / 0.511 * 1e-6
                            eRelP1 = eRel + 1
                            eRel2P1 = 2*eRel + 1
                            sigmaKNtotal = (0.75*eRelP1*eRel**-3*(
                                    2*eRel*eRelP1/eRel2P1 - np.log(eRel2P1)) +
                                    0.5/eRel*np.log(eRel2P1) -
                                    (1 + 3*eRel)*eRel2P1**-2) * R0**2 * 1e-16
                            print("sigmaKNtotal", sigmaKNtotal,
                                  "muKNtotal",
                                  matSur.sigma_to_mu(sigmaKNtotal))
                    else:  # fluo == 'elastic':
                        ats = np.zeros_like(lb.E[goodN])
                        for el, xi in zip(matSur.elements, matSur.quantities):
                            ats += np.abs(el.get_f1f2(lb.E[goodN]))**2 * xi
                        C0 = 0.75 * np.pi * R0**2 * 1e-16
                        lb.Jss[goodN] = matSur.sigma_to_mu(C0 * cosTheta**2 *
                                                           (S0 + S1prime))
                        lb.Jpp[goodN] = matSur.sigma_to_mu(C0 * (S0 - S1prime))
                        lb.Jsp[goodN] = matSur.sigma_to_mu(C0 * cosTheta *
                                                           (S2prime + S3))

                lb.Jss[goodN] *= att
                lb.Jpp[goodN] *= att
                lb.Jsp[goodN] *= att
                lb.a[goodN] = fl_a
                lb.b[goodN] = fl_b
                lb.c[goodN] = fl_c

            if (not fromVacuum) and fluo is None and\
                    not (matSur.kind in ('crystal', 'multilayer')):
                # tMax in mm, refl[2]=mu0 in 1/cm
                att = np.exp(-refl[2] * tMax[goodN] * 0.1)
                lb.Jss[goodN] *= att
                lb.Jpp[goodN] *= att
                lb.Jsp[goodN] *= att
                if hasattr(lb, 'Es'):
                    # refl[3] = n.real * k in 1/cm
                    mPh = att**0.5 * np.exp(0.1j * refl[3] * tMax[goodN])
                    lb.Es[goodN] *= mPh
                    lb.Ep[goodN] *= mPh
            else:
                if hasattr(lb, 'Es'):
                    mPh = np.exp(1e7j * lb.E[goodN]/CHBAR * tMax[goodN])
                    lb.Es[goodN] *= mPh
                    lb.Ep[goodN] *= mPh

        if self.isParametric:
            lb.s = np.copy(lb.x)
            lb.phi = np.copy(lb.y)
            lb.r = np.copy(lb.z)
            lb.x[good], lb.y[good], lb.z[good] = self.param_to_xyz(
                lb.x[good], lb.y[good], lb.z[good])

        if goodNsum > 0:
            if needMosaicity:  # secondary extinction and attenuation
                length, through = self._mosaic_length(
                    matSur, beamInDotSurfaceNormal, lb, goodN)
                n = matSur.get_refractive_index(lb.E[goodN])
                if through is not None:
                    lb.a[goodN][through] = lb.olda[through]
                    lb.b[goodN][through] = lb.oldb[through]
                    lb.c[goodN][through] = lb.oldc[through]
                    att = np.exp(-abs(n.imag) * lb.E[goodN] / CHBAR * 2e8 *
                                 tMax[goodN] * 0.1)
                    lb.Jss[goodN][through] *= att[through]
                    lb.Jpp[goodN][through] *= att[through]
                    lb.Jsp[goodN][through] *= att[through]
                if hasattr(lb, 'Es'):
                    nk = n.real * lb.E[goodN] / CHBAR * 1e8  # [1/cm]
                    mPh = np.exp(1j * nk * 0.2*length)  # *2: in and out
                    if through is not None:
                        mPh[through] =\
                            (att**0.5 * np.exp(1j * nk * 0.1*length))[through]
                    lb.Es[goodN] *= mPh
                    lb.Ep[goodN] *= mPh

# rotate coherency matrix back:
            vlb.Jss[goodN], vlb.Jpp[goodN], vlb.Jsp[goodN] =\
                rs.rotate_coherency_matrix(lb, goodN, rollAngle)
            if hasattr(lb, 'Es'):
                vlb.Es[goodN], vlb.Ep[goodN] = raycing.rotate_y(
                    lb.Es[goodN], lb.Ep[goodN], cosY, sinY)

        if vlb is not lb:
            # includeJspEsp=False because Jss, Jpp, Jsp, Es and Ep are in vlb
            # already:
            rs.copy_beam(vlb, lb, good, includeState=True, includeJspEsp=False)
# rotate the world back for the virgin local beam:
        if dx:
            vlb.x[good] += dx
        if dy:
            vlb.y[good] += dy
        if dz:
            vlb.z[good] += dz
        if self.extraPitch or self.extraRoll or self.extraYaw:
            raycing.rotate_beam(
                vlb, good, rotationSequence='-'+self.extraRotationSequence,
                pitch=self.extraPitch, roll=self.extraRoll,
                yaw=self.extraYaw)
        raycing.rotate_beam(vlb, good,
                            rotationSequence='-'+self.rotationSequence,
                            pitch=pitch, roll=roll, yaw=yaw)
        self.footprint.extend([np.hstack((np.min(np.vstack((
            lb.x[good], lb.y[good], lb.z[good])), axis=1),
            np.max(np.vstack((lb.x[good], lb.y[good], lb.z[good])),
                   axis=1))).reshape(2, 3)])

#        print len(self.footprint)
        if self.alarmLevel is not None:
            raycing.check_alarm(self, good, vlb)


class DCM(OE):
    """Implements a Double Crystal Monochromator with flat crystals."""

    hiddenMethods = ['reflect', 'multiple_reflect', 'propagate_wave']

    def __init__(self, *args, **kwargs):
        u"""
        *bragg*: float, str, list
            Bragg angle in rad. Can be calculated automatically if alignment
            energy is given as a single element list [energy]. If 'auto',
            the alignment energy will be taken from beamLine.alignE.

        *cryst1roll*, *cryst2roll*, *cryst2pitch*, *cryst2finePitch*: float
            Misalignment angles in rad.

        *cryst2perpTransl*, *cryst2longTransl*: float
            perpendicular and longitudinal translations of the 2nd crystal in
            respect to the 1st one.

        *limPhysX2*, *limPhysY2*, *limOptX2*, *limOptY2*, *material2*:
            refer to the 2nd crystal and are similar to the same parameters
            of the parent class :class:`OE` without the trailing "2".

        *fixedOffset*: float
            Offset between the incoming and outcoming beams in mm. If not None
            or zero the value of *cryst2perpTransl* is replaced by
            *fixedOffset*/2/cos(*bragg*)


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.energyMin = rs.defaultEnergy - 5.
        self.energyMax = rs.defaultEnergy + 5.

    def __pop_kwargs(self, **kwargs):
        self.bragg = raycing.auto_units_angle(kwargs.pop('bragg', 0))
        if isinstance(self.bragg, (raycing.basestring, list, tuple)):
            self._bragg = copy.copy(self.bragg)
        self.cryst1roll = raycing.auto_units_angle(kwargs.pop('cryst1roll', 0))
        self.cryst2roll = raycing.auto_units_angle(kwargs.pop('cryst2roll', 0))
        self.cryst2pitch = raycing.auto_units_angle(
            kwargs.pop('cryst2pitch', 0))
        self.cryst2finePitch = raycing.auto_units_angle(
            kwargs.pop('cryst2finePitch', 0))
        self.cryst2perpTransl = kwargs.pop('cryst2perpTransl', 0)
        self.cryst2longTransl = kwargs.pop('cryst2longTransl', 0)
        self.limPhysX2 = kwargs.pop(
            'limPhysX2', [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE])
        if self.limPhysX2 is None:
            self.limPhysX2 = [-raycing.maxHalfSizeOfOE,
                              raycing.maxHalfSizeOfOE]
        self.limPhysY2 = kwargs.pop(
            'limPhysY2', [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE])
        if self.limPhysY2 is None:
            self.limPhysY2 = [-raycing.maxHalfSizeOfOE,
                              raycing.maxHalfSizeOfOE]
        self.limOptX2 = kwargs.pop('limOptX2', None)
        self.limOptY2 = kwargs.pop('limOptY2', None)
        self.material = kwargs.get('material', None)
        self.material2 = kwargs.pop('material2', None)
        self.fixedOffset = kwargs.pop('fixedOffset', None)
        return kwargs

    def get_surface_limits(self):
        """Returns surface_limits."""
        OE.get_surface_limits(self)
        cs = self.curSurface
        self.surfPhysX2 = self.limPhysX2
        if self.limPhysX2 is not None:
            if raycing.is_sequence(self.limPhysX2[0]):
                self.surfPhysX2 = (self.limPhysX2[0][cs],
                                   self.limPhysX2[1][cs])
        self.surfPhysY2 = self.limPhysY2
        if self.limPhysY2 is not None:
            if raycing.is_sequence(self.limPhysY2[0]):
                self.surfPhysY = (self.limPhysY2[0][cs], self.limPhysY2[1][cs])
        self.surfOptX2 = self.limOptX2
        if self.limOptX2 is not None:
            if raycing.is_sequence(self.limOptX2[0]):
                self.surfOptX = (self.limOptX2[0][cs], self.limOptX2[1][cs])
        self.surfOptY2 = self.limOptY2
        if self.limOptY2 is not None:
            if raycing.is_sequence(self.limOptY2[0]):
                self.surfOptY = (self.limOptY2[0][cs], self.limOptY2[1][cs])

    def local_z1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # just flat:
        return self.local_z(x, y)

    def local_z2(self, x, y):
        return self.local_z1(x, y)

    def local_n1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # just flat:
        return self.local_n(x, y)

    def local_n2(self, x, y):
        res = self.local_n1(x, y)
        if self.alpha:
            res[1] *= -1
        return res

    def get_orientation(self):
        if self.fixedOffset not in [0, None]:
            self.cryst2perpTransl = self.fixedOffset/2./np.cos(self.bragg)

    def double_reflect(self, beam=None, needLocal=True,
                       fromVacuum1=True, fromVacuum2=True,
                       returnLocalAbsorbed=None):
        """
        Returns the reflected beam in global and two local (if *needLocal*
        is true) systems.

        *returnLocalAbsorbed*: None or int
            If not None, returns the absorbed intensity in local beam. If
            equals zero, total absorbed intensity is return in the last local
            beam, otherwise the N-th local beam returns the
            absorbed intensity on N-th surface of the optical element.


        .. Returned values: beamGlobal, beamLocal1, beamLocal2
        """
        self.footprint = []
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        self.get_orientation()
        gb = rs.Beam(copyFrom=beam)  # output beam in global coordinates
        if needLocal:
            lo1 = rs.Beam(copyFrom=beam)  # output beam in local coordinates
        else:
            lo1 = gb

        good1 = beam.state > 0
        if good1.sum() == 0:
            return gb, lo1, lo1
        raycing.global_to_virgin_local(self.bl, beam, lo1, self.center, good1)
        self._reflect_local(
            good1, lo1, gb, self.pitch + self.bragg,
            self.roll + self.positionRoll + self.cryst1roll, self.yaw, self.dx,
            local_z=self.local_z1, local_n=self.local_n1,
            fromVacuum=fromVacuum1, material=self.material)
        goodAfter1 = (gb.state == 1) | (gb.state == 2)
# not intersected rays remain unchanged except their state:
        notGood = ~goodAfter1
        if notGood.sum() > 0:
            rs.copy_beam(gb, beam, notGood)

        gb2 = rs.Beam(copyFrom=gb)
        if needLocal:
            lo2 = rs.Beam(copyFrom=gb2)  # output beam in local coordinates
        else:
            lo2 = gb2
        good2 = goodAfter1
        if hasattr(self, 't'):  # is instance of Plate
            gb2.state[~good2] = self.lostNum
        if good2.sum() == 0:
            return gb2, lo1, lo2
        self._reflect_local(
            good2, lo2, gb2,
            -self.pitch - self.bragg + self.cryst2pitch + self.cryst2finePitch,
            self.roll + self.cryst2roll - np.pi + self.positionRoll, -self.yaw,
            -self.dx, self.cryst2longTransl, -self.cryst2perpTransl,
            local_z=self.local_z2, local_n=self.local_n2,
            fromVacuum=fromVacuum2, material=self.material2, is2ndXtal=True)
        goodAfter2 = (gb2.state == 1) | (gb2.state == 2)
# in global coordinate system:
        raycing.virgin_local_to_global(self.bl, gb2, self.center, goodAfter2)
# not intersected rays remain unchanged except their state:
        notGood = ~goodAfter2
        if hasattr(self, 't'):  # is instance of Plate
            gb2.state[notGood] = self.lostNum
        if notGood.sum() > 0:
            rs.copy_beam(gb2, beam, notGood)

        if returnLocalAbsorbed is not None:
            if returnLocalAbsorbed == 0:
                absorbedLb = rs.Beam(copyFrom=lo2)
                absorbedLb.absorb_intensity(beam)
                lo2 = absorbedLb
            elif returnLocalAbsorbed == 1:
                absorbedLb = rs.Beam(copyFrom=lo1)
                absorbedLb.absorb_intensity(beam)
                lo1 = absorbedLb
            elif returnLocalAbsorbed == 1:
                absorbedLb = rs.Beam(copyFrom=lo2)
                absorbedLb.absorb_intensity(lo1)
                lo2 = absorbedLb
        lo2.parentId = self.name
        raycing.append_to_flow(self.double_reflect, [gb2, lo1, lo2],
                               inspect.currentframe())

        return gb2, lo1, lo2  # in global and local(lo1 and lo2) coordinates
