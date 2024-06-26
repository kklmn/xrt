# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import time
import numpy as np
import inspect
import copy
from scipy.spatial.transform import Rotation as scprot

import matplotlib as mpl
from .. import raycing
from . import sources as rs
from . import myopencl as mcl
from .physconsts import PI2, CH, CHBAR, SQRT2PI
from .materials import EmptyMaterial
# try:
#     import pyopencl as cl  # analysis:ignore
#     isOpenCL = True
# except ImportError:
#     isOpenCL = False

if mcl.isOpenCL or mcl.isZMQ:
    isOpenCL = True
else:
    isOpenCL = False

__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "06 Oct 2017"

__fdir__ = os.path.dirname(__file__)

allArguments = ['bl', 'name', 'center', 'bragg', 'pitch', 'roll', 'yaw',
                'positionRoll', 'extraPitch', 'extraRoll', 'extraYaw',
                'rotationSequence', 'extraRotationSequence',
                'surface', 'material', 'material2', 'alpha',
                'limPhysX', 'limOptX', 'limPhysY', 'limOptY',
                'limPhysX2', 'limPhysY2', 'limOptX2', 'limOptY2',
                'isParametric', 'shape', 'gratingDensity', 'order',
                'shouldCheckCenter',
                'dxFacet', 'dyFacet', 'dxGap', 'dyGap', 'Rm',
                'crossSection', 'Rs', 'R', 'r', 'p', 'q',
                'isCylindrical', 'L0', 'theta', 'r0', 'ellipseA', 'ellipseB',
                'workingDistance', 'wedgeAngle',
                'cryst1roll', 'cryst2roll', 'cryst2pitch', 'alarmLevel',
                'cryst2finePitch', 'cryst2perpTransl', 'cryst2longTransl',
                'fixedOffset', 't', 'focus', 'zmax', 'nCRL', 'f', 'E', 'N',
                'isCentralZoneBlack', 'thinnestZone', 'f1', 'f2',
                'phaseShift', 'vorticity', 'grazingAngle',
                'blaze', 'antiblaze', 'rho', 'aspect', 'depth', 'coeffs',
                'targetOpenCL', 'precisionOpenCL']


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
        r"""
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
            [axis, ρ\ :sub:`0`, *P*\ :sub:`0`, *P*\ :sub:`1`, ...],
            where ρ\ :sub:`0` is the constant line density in inverse mm,
            *P*\ :sub:`0` -- *P*\ :sub:`n` are polynom coefficients defining
            the line density variation, so that for a given axis

            .. math::

                \rho_x = \rho_0\cdot(P_0 + 2 P_1 x + 3 P_2 x^2 + ...).

            Example: ['y', 800, 1] for the grating with constant
            spacing along the 'y' direction; ['y', 1200, 1, 1e-6, 3.1e-7] for
            a VLS grating. The length of the list determines the polynomial
            order.

            .. note::

                Redefining :meth:`local_g` is the most flexible way to define
                a VLS grating.

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
#        if any([x == 'auto' for x in self.center]):
#            self._center = copy.copy(self.center)
        if (bl is not None) and self.shouldCheckCenter:
            self.checkCenter()

        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        self.rotationSequence = rotationSequence
        self.positionRoll = positionRoll

        self.extraPitch = extraPitch
        self.extraRoll = extraRoll
        self.extraYaw = extraYaw
        self.extraRotationSequence = extraRotationSequence
        self.alarmLevel = alarmLevel

        self.surface = surface
        self.material = material
        self.alpha = alpha
        self.curSurface = 0
        self.dx = 0
        self.limPhysX = limPhysX
        self.limPhysY = limPhysY
        self.limOptX = limOptX
        self.limOptY = limOptY
        self.isParametric = isParametric
        self.use_rays_good_gn = False  # use rays_good_gn instead of rays_good

        self.shape = shape
        self.gratingDensity = gratingDensity
        self.order = order
        self.get_surface_limits()
        self.cl_ctx = None
        self.ucl = None
        self.footprint = []
        if targetOpenCL is not None:
            if not isOpenCL:
                raycing.colorPrint("pyopencl is not available!", "RED")
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

    @property
    def center(self):
        return self._center if self._centerVal is None else self._centerVal

    @center.setter
    def center(self, center):
        if any([x == 'auto' for x in center]):
            self._center = copy.copy(center)
            self._centerVal = None
            self._centerInit = copy.copy(center)  # For glow auto-recognition
        else:
            self._centerVal = center
#            if not hasattr(self, '_center'):
#                self._center = None

    @property
    def pitch(self):
        # _pitch can only contain unprocessed input for auto-calculation
        # _pitchVal is None before auto-calculation, a number after it
        return self._pitch if self._pitchVal is None else self._pitchVal

    @pitch.setter
    def pitch(self, pitch):
        pitch = raycing.auto_units_angle(pitch)
        if isinstance(pitch, (raycing.basestring, list, tuple)):
            self._pitch = copy.copy(pitch)
            self._pitchVal = None
            self._pitchInit = copy.copy(pitch)  # For glow auto-recognition
        else:  # also after auto-calculation
            self._pitchVal = pitch
            self.update_orientation_quaternion()

        if hasattr(self, '_reset_pq'):
            self._reset_pq()

        if hasattr(self, '_RPQ'):
            if self._RPQ is not None:
                self._R = self.get_Rmer_from_Coddington(self._RPQ[0],
                                                        self._RPQ[1])
        if hasattr(self, '_rPQ'):
            if self._rPQ is not None:
                self._r = self.get_rsag_from_Coddington(self._rPQ[0],
                                                        self._rPQ[1])
        if hasattr(self, '_RmPQ'):
            if self._RmPQ is not None:
                self._Rm = self.get_Rmer_from_Coddington(self._RmPQ[0],
                                                         self._RmPQ[1])
        if hasattr(self, '_RsPQ'):
            if self._RsPQ is not None:
                self._Rs = self.get_rsag_from_Coddington(self._RsPQ[0],
                                                         self._RsPQ[1])

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, roll):
        self._roll = raycing.auto_units_angle(roll)
        self.update_orientation_quaternion()
        if hasattr(self, '_reset_pq'):
            self._reset_pq()

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, yaw):
        self._yaw = raycing.auto_units_angle(yaw)
        self.update_orientation_quaternion()
        if hasattr(self, '_reset_pq'):
            self._reset_pq()

    @property
    def extraPitch(self):
        return self._extraPitch

    @extraPitch.setter
    def extraPitch(self, extraPitch):
        self._extraPitch = raycing.auto_units_angle(extraPitch)
        self.update_orientation_quaternion()

    @property
    def extraRoll(self):
        return self._extraRoll

    @extraRoll.setter
    def extraRoll(self, extraRoll):
        self._extraRoll = raycing.auto_units_angle(extraRoll)
        self.update_orientation_quaternion()

    @property
    def extraYaw(self):
        return self._extraYaw

    @extraYaw.setter
    def extraYaw(self, extraYaw):
        self._extraYaw = raycing.auto_units_angle(extraYaw)
        self.update_orientation_quaternion()

    @property
    def positionRoll(self):
        return self._positionRoll

    @positionRoll.setter
    def positionRoll(self, positionRoll):
        self._positionRoll = raycing.auto_units_angle(positionRoll)
        self.update_orientation_quaternion()
        if hasattr(self, '_reset_pq'):
            self._reset_pq()

    @property
    def limPhysX(self):
        return self._limPhysX

    @limPhysX.setter
    def limPhysX(self, limPhysX):
        if limPhysX is None:
            self._limPhysX = [-raycing.maxHalfSizeOfOE,
                              raycing.maxHalfSizeOfOE]
        else:
            self._limPhysX = limPhysX

    @property
    def limPhysY(self):
        return self._limPhysY

    @limPhysY.setter
    def limPhysY(self, limPhysY):
        if limPhysY is None:
            self._limPhysY = [-raycing.maxHalfSizeOfOE,
                              raycing.maxHalfSizeOfOE]
        else:
            self._limPhysY = limPhysY

    @property
    def gratingDensity(self):
        return self._gratingDensity

    @gratingDensity.setter
    def gratingDensity(self, gratingDensity):
        self._gratingDensity = gratingDensity
        if gratingDensity is not None and self.material is None and \
                not hasattr(self, 'get_grating_area_fraction'):
            self.material = EmptyMaterial()
        if hasattr(self, 'reset'):
            self.reset()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = 1 if order is None else order

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """Sets the asymmetry angle *alpha* for a crystal OE. It calculates
        cos(alpha) and sin(alpha) which are then used for rotating the normal
        to the crystal planes."""
        self._alpha = raycing.auto_units_angle(alpha)
        if self.alpha is not None:
            self.cosalpha = np.cos(self.alpha)
            self.sinalpha = np.sin(self.alpha)
            self.tanalpha = self.sinalpha / self.cosalpha

    def set_alpha(self, alpha):
        """Same as alpha.setter, left for compatibility"""
        self.alpha = alpha

    def update_orientation_quaternion(self):
        """Will be used with xrtGlow for fast orientation tracking"""
        try:  # Experimental
            if all([hasattr(self, angle) for angle in ['pitch', 'roll', 'yaw'
                    'extraPitch', 'extraRoll', 'extraYaw', 'positionRoll']]):
                rotAx = {'x': self.pitch,
                         'y': self.roll+self.positionRoll,
                         'z': self.yaw}
                extraRotAx = {'x': self.extraPitch,
                              'y': self.extraRoll,
                              'z': self.extraYaw}
                rotSeq = self.rotationSequence(slice(1, None, 2))
                extraRotSeq = self.extraRotationSequence(slice(1, None, 2))
                rotation = (scprot.from_euler(
                        rotSeq, [rotAx[i] for i in rotSeq])).as_quat()
                extraRot = (scprot.from_euler(
                    extraRotSeq,
                    [extraRotAx[i] for i in extraRotSeq])).as_quat()
                self.orientationQuat = raycing.multiply_quats(rotation,
                                                              extraRot)
        except Exception as e:
            print(e)

    def _update_bounding_box(self):
        pass

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
            raycing.colorPrint("Warning: {0} is off the beamline by {1}"
                               .format(self.name, d), "YELLOW")
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
            if hasattr(self, '_pitchVal'):
                if self._pitchVal is not None:
                    pitch = self._pitchVal
                else:
                    return None
            else:
                return None
        return 2 * p * q / (p+q) / np.sin(abs(pitch))

    def get_rsag_from_Coddington(self, p, q, pitch=None):
        if pitch is None:
            if hasattr(self, '_pitchVal'):
                if self._pitchVal is not None:
                    pitch = self._pitchVal
                else:
                    return None
            else:
                return None
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
        floats or of arrays of the length of *x* and *y*.

        .. note::

            The sign of the returned vector depends on the user's definition
            of the diffraction order sign.

        """

        try:
            rhoList = self.gratingDensity
            if rhoList is not None:
                coord = x if rhoList[0] == 'x' else y
                poly = 0.
                for ic, coeff in enumerate(rhoList[2:]):
                    poly += (ic+1) * coeff * coord**ic
                N = rhoList[1] * poly

                if rhoList[0] == 'x':
                    return N, np.zeros_like(N), np.zeros_like(N)
                elif rhoList[0] == 'y':
                    return np.zeros_like(N), N, np.zeros_like(N)
        except:  # noqa
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
                    raycing.colorPrint('{0} NaNs in the surface!!!'
                                       .format(ind.sum()), 'RED')
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
            raycing.colorPrint('maxIteration is reached for {0} ray{1}!!!'
                               .format(nn, 's' if nn > 1 else ''), 'RED')
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
            # swapping using double slicing:
            # stackoverflow.com/questions/1687566/why-does-an
            # -assignment-for-double-sliced-numpy-arrays-not-work
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

    def rays_good(self, x, y, z, is2ndXtal=False):
        """Returns *state* value for a ray with the given intersection point
        (*x*, *y*) with the surface of OE:
        1: good (intersected)
        2: reflected outside of working area ("out"),
        3: transmitted without intersection ("over"),
        -NN: lost (absorbed) at OE#NN - OE numbering starts from 1 !!!

        Note, *x*, *y*, *z* are local Cartesian coordinates, even for a
        parametric OE.
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

        locState = np.ones(x.size, dtype=np.int32)
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
            Used in wave propagation, normally should be False. Certainly
            should be False if the OE is distorted.


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
        if hasattr(beam, 'createdByDiffract'):
            goodAfter = gb.state == 1
        else:
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

    def local_to_global(self, lb, returnBeam=False, **kwargs):
        dx, dy, dz = 0, 0, 0
        extraAnglesSign = 1.  # only for pitch and yaw
        if isinstance(self, DCM):
            is2ndXtal = kwargs.get('is2ndXtal', False)
            if is2ndXtal:
                pitch = -self.pitch - self.bragg + self.cryst2pitch +\
                    self.cryst2finePitch
                roll = self.roll + self.cryst2roll + self.positionRoll
                yaw = -self.yaw
                dx = -self.dx
                dy = self.cryst2longTransl
                dz = -self.cryst2perpTransl
                extraAnglesSign = -1.
            else:
                pitch = self.pitch + self.bragg
                roll = self.roll + self.positionRoll + self.cryst1roll
                yaw = self.yaw
                dx = self.dx
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
                pitch=extraAnglesSign*self.extraPitch, roll=self.extraRoll,
                yaw=extraAnglesSign*self.extraYaw, **kwargs)

        raycing.rotate_beam(lb, rotationSequence='-'+self.rotationSequence,
                            pitch=pitch, roll=roll, yaw=yaw, **kwargs)
        if isinstance(self, DCM):
            if is2ndXtal:
                raycing.rotate_beam(lb, roll=np.pi)

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
        *nrays*: if int, specifies the number of randomly distributed samples
        the surface within ``self.limPhysX`` limits; if 2-tuple of ints,
        specifies (nx, ny) sizes of a uniform mesh of samples.
        """

        if rw is None:
            from . import waves as rw

        if isinstance(nrays, (int, float)):
            nsamples = int(nrays)
        elif isinstance(nrays, (list, tuple)):
            nsamples = nrays[0] * nrays[1]
        else:
            raise ValueError('wrong type of `nrays`!')

        lb = rs.Beam(nrays=nsamples, forceState=1, withAmplitudes=True)
        if isinstance(nrays, (int, float)):
            xy = np.random.rand(nsamples, 2)
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
        elif isinstance(nrays, (list, tuple)):
            if shape.startswith('ro'):  # round
                raise ValueError("must be rectangular")
            xx = np.linspace(*self.limPhysX, nrays[0])
            yy = np.linspace(*self.limPhysY, nrays[1])
            X, Y = np.meshgrid(xx, yy)
            x = X.ravel()
            y = Y.ravel()
            if area == 'auto':
                dX = self.limPhysX[1] - self.limPhysX[0]
                dY = self.limPhysY[1] - self.limPhysY[0]
                area = dX * dY
        else:
            raise ValueError('wrong type of `nrays`!')

        # These are approximate samples (exact for undistorted and
        # non-parametric cases). This works even for a parametric case because
        # `reflect()` (that follows `diffract()`) will make it exact. Make sure
        # that `noIntersectionSearch=False` in `reflect()`.
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

        # this line is important for cases when the previous reflection points
        # (the ray heads) are close, e.g. in Montel mirror without setting
        # physical surface limits. This solution is not fully studied and it
        # may break `reflect` after `diffract` (the factor 1e6 is to play with)
        tMin[tMin < -1e6*raycing.zEps] = -1e6*raycing.zEps

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
            raycing.colorPrint(
                "{0} NaN rays in array {1} in optical element {2}!".format(
                    nanSum, strName, self.name), "RED")

    def local_n_random(self, bLength, chi):
        a = np.zeros(bLength)
        b = np.zeros(bLength)
        c = np.ones(bLength)

        cos_range = np.random.rand(bLength)  # * 2**-0.5
        y_angle = np.arccos(cos_range)
        z_angle = (chi[1]-chi[0]) * np.random.rand(bLength) + chi[0]

        a, c = raycing.rotate_y(a, c, np.cos(y_angle), np.sin(y_angle))
        a, b = raycing.rotate_z(a, b, np.cos(z_angle), np.sin(z_angle))
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
        beamInDotNormalAbs = np.abs(beamInDotNormal)
        delta = np.arcsin(beamInDotNormalAbs) - thetaB
        w = np.exp(-0.5*delta**2 / mat.mosaicity**2) / (SQRT2PI*mat.mosaicity)
        rate = w*Q  # in cm^-1
        rate[rate <= 1e-3] = 1e-3
        length = np.random.exponential(10./rate, size=len(Qs))  # in mm
        if mat.t:
            through = length*beamInDotNormalAbs > mat.t
            length[through] = mat.t / beamInDotNormalAbs[through]
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
            noIntersectionSearch=False, isMulti=False):
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
                _gNormal, _oeNormal, _beamInDotSurfaceNormal,
                _beamInDotNormal, xd=None, yd=None):
            normalDotSurfNormal = _oeNormal[0]*_oeNormal[-3] +\
                _oeNormal[1]*_oeNormal[-2] + _oeNormal[2]*_oeNormal[-1]
            bdn = _beamInDotNormal.sum() / len(_beamInDotNormal)
            sgbdn = 1 if bdn < 0 else -1
            # note:
            # _oeNormal[0:3] is n_B
            # _oeNormal[-3:] is n_s
            # normalDotSurfNormal is dot(n_B, n_s)
            # |vec(n_B) - dot(n_B, n_s)*vec(n_s)| = sin(alpha)
#            wH = matSur.get_refractive_correction(
#                lb.E[goodN], abs(_beamInDotSurfaceNormal))
            wH = 0
            if hasattr(matSur, 'get_d') and xd is not None and yd is not None:
                crystd = matSur.get_d(xd, yd)
            else:
                crystd = matSur.d

            wHd = (1 + wH) / (crystd * 1e-7)
            gNormalCryst = np.asarray((
                (_oeNormal[0]-normalDotSurfNormal*_oeNormal[-3]) * wHd,
                (_oeNormal[1]-normalDotSurfNormal*_oeNormal[-2]) * wHd,
                (_oeNormal[2]-normalDotSurfNormal*_oeNormal[-1]) * wHd),
                order='F') * sgbdn
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
        extraAnglesSign = 1.  # only for pitch and yaw
        if is2ndXtal:
            raycing.rotate_beam(lb, good, roll=-np.pi)
            extraAnglesSign = -1.  # only for pitch and yaw
        raycing.rotate_beam(
            lb, good, rotationSequence=self.rotationSequence,
            pitch=-pitch, roll=-roll, yaw=-yaw)
        if self.extraPitch or self.extraRoll or self.extraYaw:
            raycing.rotate_beam(
                lb, good, rotationSequence=self.extraRotationSequence,
                pitch=-extraAnglesSign*self.extraPitch, roll=-self.extraRoll,
                yaw=-extraAnglesSign*self.extraYaw)
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
            tX, tY, tZ = self.param_to_xyz(lb.x[good], lb.y[good], lb.z[good])
        else:
            tX, tY, tZ = lb.x[good], lb.y[good], lb.z[good]

        if self.use_rays_good_gn:
            lb.state[good], gNormal = self.rays_good_gn(tX, tY, tZ)
        else:
            gNormal = None
            lb.state[good] = self.rays_good(tX, tY, tZ, is2ndXtal)
        if _lost is not None:
            lb.state[np.where(good)[0][_lost]] = self.lostNum

#        goodN = (lb.state == 1) | (lb.state == 2)
        goodN = (lb.state == 1)
# normal at x, y, z:
        goodNsum = goodN.sum()
        needMosaicity = False
        crystalVD = False
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
                        if hasattr(matSur, 'volumetricDiffraction'):
                            if matSur.volumetricDiffraction:
                                crystalVD = True
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
                    self.local_n_random(len(lb.E[goodN]), matSur.chi))
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
                    # The following code will consider finite thickness of
                    # the Laue crystal
                    if crystalVD:
                        # Max depth for flat crystal.
                        thMax = -matSur.t / beamInDotSurfaceNormal
                        # Point of diffraction along the path is approximated
                        # with uniform distribution.
                        dpth = np.random.rand(len(lb.a[goodN])) * thMax
                        # dpth = np.linspace(0, 1, len(lb.a[goodN])) * thMax
                        lb.x[goodN] += lb.a[goodN] * dpth
                        lb.y[goodN] += lb.b[goodN] * dpth
                        lb.z[goodN] += lb.c[goodN] * dpth
                        # Local orientation of the plane inside the crystal
                        deepNormal = list(self.local_n_depth(
                                lb.x[goodN], lb.y[goodN], lb.z[goodN]))
                        oeNormal[0:3] = deepNormal[0:3]
                        beamInDotNormal =\
                            lb.a[goodN]*oeNormal[0] +\
                            lb.b[goodN]*oeNormal[1] +\
                            lb.c[goodN]*oeNormal[2]
                        lb.theta = np.zeros_like(lb.x)
                        beamInDotNormal[beamInDotNormal < -1] = -1
                        beamInDotNormal[beamInDotNormal > 1] = 1
                        lb.theta[goodN] = np.arccos(beamInDotNormal)\
                            - np.pi/2
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
                    gN = (oeNormal[-3]*gNormal[0] + oeNormal[-2]*gNormal[1] +
                          oeNormal[-1]*gNormal[2])
                    if not np.allclose(gN, np.zeros_like(gN)):
                        print("Warning: local_g is not orthogonal to local_n!")

                giveSign = 1 if toWhere == 4 else -1
                lb.a[goodN], lb.b[goodN], lb.c[goodN] =\
                    self._grating_deflection(goodN, lb, gNormal, oeNormal,
                                             beamInDotSurfaceNormal,
                                             self.order, giveSign)
            elif toWhere in [0, 2]:  # reflect, straight
                useAsymmetricNormal = False
                if material is not None:
                    if matSur.kind in ('crystal', 'multilayer') and\
                            toWhere == 0 and (not needMosaicity) and\
                            (not crystalVD):
                        useAsymmetricNormal = True

                if useAsymmetricNormal:
                    a_out, b_out, c_out = _get_asymmetric_reflection_grating(
                        gNormal, oeNormal, beamInDotSurfaceNormal,
                        beamInDotNormal, lb.x[goodN], lb.y[goodN])
                else:
                    a_out = lb.a[goodN] - oeNormal[0]*2*beamInDotNormal
                    b_out = lb.b[goodN] - oeNormal[1]*2*beamInDotNormal
                    c_out = lb.c[goodN] - oeNormal[2]*2*beamInDotNormal

                if toWhere == 0:  # reflect
                    if needMosaicity:
                        lb.olda = np.array(lb.a[goodN])
                        lb.oldb = np.array(lb.b[goodN])
                        lb.oldc = np.array(lb.c[goodN])
                        lb.oldJss = np.array(lb.Jss[goodN])
                        lb.oldJpp = np.array(lb.Jpp[goodN])
                        lb.oldJsp = np.array(lb.Jsp[goodN])
                        if hasattr(lb, 'Es'):
                            lb.oldEs = np.array(lb.Es[goodN])
                            lb.oldEp = np.array(lb.Ep[goodN])

                    lb.a[goodN] = a_out
                    lb.b[goodN] = b_out
                    lb.c[goodN] = c_out
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
                        Ry = self.R if hasattr(self, 'R') else self.Rm \
                            if hasattr(self, 'Rm') else None
                        lcname = self.__class__.__name__.lower()
                        if 'johansson' in lcname or 'ground' in lcname:
                            Ry *= 2
                        Rx = self.Rs if hasattr(self, 'Rs') else None
                        refl = matSur.get_amplitude_pytte(
                            lb.E[goodN], beamInDotSurfaceNormal,
                            beamOutDotSurfaceNormal, beamInDotNormal,
                            alphaAsym=self.alpha,
                            Ry=Ry, Rx=Rx, ucl=self.ucl)

                        # if '_R' in self.__dict__.keys():
                        #     Ry = self.R
                        # elif '_Rm' in self.__dict__.keys():
                        #     Ry = self.Rm
                        # else:
                        #     Ry = None
                        # refl = matSur.get_amplitude_pytte(
                        #     lb.E[goodN], beamInDotSurfaceNormal,
                        #     beamOutDotSurfaceNormal, beamInDotNormal,
                        #     alphaAsym=self.alpha,
                        #     Ry=Ry, Rx=self.Rs if 'Rs' in self.__dict__.keys()
                        #     else None,
                        #     ucl=self.ucl)
                    else:
                        refl = matSur.get_amplitude(
                            lb.E[goodN], beamInDotSurfaceNormal,
                            beamOutDotSurfaceNormal, beamInDotNormal,
                            lb.x[goodN], lb.y[goodN])
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

            if (not fromVacuum) and\
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
                if through is not None:  # if mat.t
                    # using double slicing, see
                    # stackoverflow.com/questions/1687566/why-does-an
                    # -assignment-for-double-sliced-numpy-arrays-not-work
                    lb.a[np.where(goodN)[0][through]] = lb.olda[through]
                    lb.b[np.where(goodN)[0][through]] = lb.oldb[through]
                    lb.c[np.where(goodN)[0][through]] = lb.oldc[through]
                    att = np.exp(-abs(n.imag) * lb.E[goodN] / CHBAR * 2e8 *
                                 length * 0.1)
                    lb.Jss[np.where(goodN)[0][through]] =\
                        lb.oldJss[through] * att[through]
                    lb.Jpp[np.where(goodN)[0][through]] = \
                        lb.oldJpp[through] * att[through]
                    lb.Jsp[np.where(goodN)[0][through]] = \
                        lb.oldJsp[through] * att[through]
                    if hasattr(lb, 'Es'):
                        lb.Es[np.where(goodN)[0][through]] = lb.oldEs[through]
                        lb.Ep[np.where(goodN)[0][through]] = lb.oldEp[through]
                if hasattr(lb, 'Es'):
                    nk = n.real * lb.E[goodN] / CHBAR * 1e8  # [1/cm]
                    mPh = np.exp(1j * nk * 0.2*length)  # *2: in and out
                    if through is not None:
                        mPh[through] =\
                            (att**0.5 * np.exp(1j*nk*0.1*length))[through]
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
                pitch=extraAnglesSign*self.extraPitch, roll=self.extraRoll,
                yaw=extraAnglesSign*self.extraYaw)
        raycing.rotate_beam(vlb, good,
                            rotationSequence='-'+self.rotationSequence,
                            pitch=pitch, roll=roll, yaw=yaw)
        if is2ndXtal:
            raycing.rotate_beam(vlb, good, roll=np.pi)
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
        self.bragg = kwargs.pop('bragg', 0)
        self.cryst1roll = kwargs.pop('cryst1roll', 0)
        self.cryst2roll = kwargs.pop('cryst2roll', 0)
        self.cryst2pitch = kwargs.pop('cryst2pitch', 0)
        self.cryst2finePitch = kwargs.pop('cryst2finePitch', 0)
        self.cryst2perpTransl = kwargs.pop('cryst2perpTransl', 0)
        self.cryst2longTransl = kwargs.pop('cryst2longTransl', 0)
        self.limPhysX2 = kwargs.pop(
            'limPhysX2', [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE])
        self.limPhysY2 = kwargs.pop(
            'limPhysY2', [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE])
        self.limOptX2 = kwargs.pop('limOptX2', None)
        self.limOptY2 = kwargs.pop('limOptY2', None)
        self.material = kwargs.get('material', None)
        self.material2 = kwargs.pop('material2', None)
        self.fixedOffset = kwargs.pop('fixedOffset', None)
        return kwargs

    @property
    def bragg(self):
        return self._bragg if self._braggVal is None else self._braggVal

    @bragg.setter
    def bragg(self, bragg):
        bragg = raycing.auto_units_angle(bragg)
        if isinstance(bragg, (raycing.basestring, list, tuple)):
            self._bragg = copy.copy(bragg)
            self._braggVal = None
            self._braggInit = copy.copy(bragg)  # For glow auto-recognition
        else:
            self._braggVal = raycing.auto_units_angle(bragg)
            self.update_orientation_quaternion()

    @property
    def cryst1roll(self):
        return self._cryst1roll

    @cryst1roll.setter
    def cryst1roll(self, cryst1roll):
        self._cryst1roll = raycing.auto_units_angle(cryst1roll)
        self.update_orientation_quaternion()

    @property
    def cryst2roll(self):
        return self._cryst2roll

    @cryst2roll.setter
    def cryst2roll(self, cryst2roll):
        self._cryst2roll = raycing.auto_units_angle(cryst2roll)
        self.update_orientation_quaternion()

    @property
    def cryst2pitch(self):
        return self._cryst2pitch

    @cryst2pitch.setter
    def cryst2pitch(self, cryst2pitch):
        self._cryst2pitch = raycing.auto_units_angle(cryst2pitch)
        self.update_orientation_quaternion()

    @property
    def cryst2finePitch(self):
        return self._cryst2finePitch

    @cryst2finePitch.setter
    def cryst2finePitch(self, cryst2finePitch):
        self._cryst2finePitch = raycing.auto_units_angle(cryst2finePitch)
        self.update_orientation_quaternion()

    @property
    def limPhysX2(self):
        return self._limPhysX2

    @limPhysX2.setter
    def limPhysX2(self, limPhysX2):
        if limPhysX2 is None:
            self._limPhysX2 = [-raycing.maxHalfSizeOfOE,
                               raycing.maxHalfSizeOfOE]
        else:
            self._limPhysX2 = limPhysX2

    @property
    def limPhysY2(self):
        return self._limPhysY2

    @limPhysY2.setter
    def limPhysY2(self, limPhysY2):
        if limPhysY2 is None:
            self._limPhysY2 = [-raycing.maxHalfSizeOfOE,
                               raycing.maxHalfSizeOfOE]
        else:
            self._limPhysY2 = limPhysY2

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
            self.roll + self.cryst2roll + self.positionRoll, -self.yaw,
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
