# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import numpy as np
import copy
from scipy.spatial.transform import Rotation as scprot

import matplotlib as mpl

from ... import raycing
from .. import sources as rs
from .. import myopencl as mcl
from ..physconsts import CH, CHBAR, SQRT2PI
from ..materials import EmptyMaterial
from .reflect import OEMainMethods

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
__date__ = "1 Feb 2026"

__fdir__ = os.path.dirname(__file__)

allArguments = ['bl', 'name', 'center', 'bragg', 'pitch', 'roll', 'yaw',
                'positionRoll', 'extraPitch', 'extraRoll', 'extraYaw',
                'rotationSequence', 'extraRotationSequence',
                'surface', 'material', 'material2', 'alpha',
                'limPhysX', 'limOptX', 'limPhysY', 'limOptY',
                'limPhysX2', 'limPhysY2', 'limOptX2', 'limOptY2',
                'isParametric', 'shape', 'gratingDensity', 'order',
                'shouldCheckCenter', 'braggOffset',
                'dxFacet', 'dyFacet', 'dxGap', 'dyGap', 'Rm',
                'crossSection', 'Rs', 'R', 'r', 'p', 'q', 'isCylindrical',
                'isClosed', 'L0', 'theta', 'r0', 'ellipseA', 'ellipseB',
                'workingDistance', 'wedgeAngle',
                'cryst1roll', 'cryst2roll', 'cryst2pitch', 'alarmLevel',
                'cryst2finePitch', 'cryst2perpTransl', 'cryst2longTransl',
                'fixedOffset', 't', 'focus', 'zmax', 'nCRL', 'f', 'E', 'N',
                'isCentralZoneBlack', 'thinnestZone', 'f1', 'f2', 'pAxis',
                'parabolaAxis', 'phaseShift', 'vorticity', 'grazingAngle',
                'blaze', 'antiblaze', 'rho', 'aspect', 'depth', 'coeffs',
                'targetOpenCL', 'precisionOpenCL', 'fileName', 'recenter',
                'orientation', 'figureError']


def flatten(x):
    if x is None:
        x = [0, 0]
    if isinstance(x, (list, tuple)):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


class OE(OEMainMethods):
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
        alarmLevel=None, surface=None, material=None, figureError=None,
        alpha=None,
        limPhysX=[-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE],
        limOptX=None,
        limPhysY=[-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE],
        limOptY=None, isParametric=False, shape='rect',
        gratingDensity=None, order=None, shouldCheckCenter=False,
            targetOpenCL=None, precisionOpenCL='float64', **kwargs):
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

        *figureError*: None or FigureError object.


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
            pairs for an arbitrary shape. *shape* refers to the geometric shape
            of the XY projection. 'round' shape makes a circular disk, not a
            capillary optical element. The latter can be made as a parametric
            surface, see e.g. :class:`SurfaceOfRevolution` or
            :class:`EllipticalMirrorParam`.

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

        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        if bl is not None:
            if self.bl.flowSource != 'Qook0':  # should work everywhere
                bl.oesDict[self.uuid] = [self, 1]

        self.shouldCheckCenter = shouldCheckCenter
        self.needReCenter = False
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
        self.material = material  # can be uuid
        self.figureError = figureError
        self.alpha = alpha
        self.curSurface = 0
        self.dx = 0
        self.limOptX = limOptX
        self.limOptY = limOptY
        self.limPhysX = limPhysX
        self.limPhysY = limPhysY
        self.isParametric = isParametric
        self.use_rays_good_gn = False  # use rays_good_gn instead of rays_good

        self.shape = shape
        self.gratingDensity = gratingDensity
        self.order = order
#        self.get_surface_limits()
        self.cl_ctx = None
        self.ucl = None
        self.footprint = []
        if targetOpenCL is not None:
            if not isOpenCL:
                raycing.colorPrint("pyopencl is not available!", "RED")
            else:
                cl_template = os.path.join(__fdir__, r'../cl/materials.cl')
                with open(cl_template, 'r') as f:
                    kernelsource = f.read()
                cl_template = os.path.join(__fdir__, r'../cl/OE.cl')
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

    center = raycing.center_property()

    @property
    def pitch(self):
        # _pitch can only contain unprocessed input for auto-calculation
        # _pitchVal is None before auto-calculation, a number after it
        return self._pitch if self._pitchVal is None else self._pitchVal

    @pitch.setter
    def pitch(self, pitch):
        if isinstance(pitch, (raycing.basestring, list, tuple)):
            self._pitchInit = copy.deepcopy(pitch)  # For glow auto-recognition

        pitch = raycing.auto_units_angle(pitch)
        if isinstance(pitch, (raycing.basestring, list, tuple)):
            self._pitch = copy.deepcopy(pitch)
            self._pitchVal = None
        else:  # also after auto-calculation
            self._pitchVal = pitch
            self._pitch = None

        if hasattr(self, '_reset_pq'):
            self._reset_pq()

        if hasattr(self, '_R') and isinstance(self._R, (tuple, list)):
            if hasattr(self, '_braggVal'):
                pass
            elif hasattr(self, '_pitchVal') and self._pitchVal != 0:
                self._RVal = self.get_Rmer_from_Coddington(
                        self._R[0], self._R[1], self._pitchVal)
            else:
                self._RVal = np.inf

        if hasattr(self, '_r') and isinstance(self._r, (list, tuple)):
            self._rVal = self.get_rsag_from_Coddington(*self._r)

        if hasattr(self, '_Rm') and isinstance(self._Rm, (tuple, list)):
            if hasattr(self, '_braggVal'):
                pass
            elif hasattr(self, '_pitchVal') and self._pitchVal != 0:
                self._RmVal = self.get_Rmer_from_Coddington(
                        self._Rm[0], self._Rm[1], self._pitchVal)
            else:
                self._RmVal = np.inf

        if hasattr(self, '_Rs') and isinstance(self._Rs, (tuple, list)):
            if hasattr(self, '_braggVal'):
                pass
            elif hasattr(self, '_pitchVal') and self._pitchVal != 0:
                self._RsVal = self.get_rsag_from_Coddington(
                        self._Rs[0], self._Rs[1], self._pitchVal)
            else:
                self._RsVal = np.inf

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, roll):
        self._roll = raycing.auto_units_angle(roll)
        if hasattr(self, '_reset_pq'):
            self._reset_pq()

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, yaw):
        self._yaw = raycing.auto_units_angle(yaw)
        if hasattr(self, '_reset_pq'):
            self._reset_pq()

    @property
    def extraPitch(self):
        return self._extraPitch

    @extraPitch.setter
    def extraPitch(self, extraPitch):
        self._extraPitch = raycing.auto_units_angle(extraPitch)

    @property
    def extraRoll(self):
        return self._extraRoll

    @extraRoll.setter
    def extraRoll(self, extraRoll):
        self._extraRoll = raycing.auto_units_angle(extraRoll)

    @property
    def extraYaw(self):
        return self._extraYaw

    @extraYaw.setter
    def extraYaw(self, extraYaw):
        self._extraYaw = raycing.auto_units_angle(extraYaw)

    @property
    def positionRoll(self):
        return self._positionRoll

    @positionRoll.setter
    def positionRoll(self, positionRoll):
        self._positionRoll = raycing.auto_units_angle(positionRoll)
        if hasattr(self, '_reset_pq'):
            self._reset_pq()

    @property
    def material(self):
        def resolve(mat):
            if not raycing.is_valid_uuid(mat):
                return mat

            if self.bl is None:
                print(f"Material with UUID {mat} doesn't exist!")
                return None

            return self.bl.materialsDict.get(mat)

        m = self._material

        if raycing.is_sequence(m):
            return [resolve(x) for x in m]
        else:
            return resolve(m)

    @material.setter
    def material(self, material):
        self._material = material

    @property
    def figureError(self):
        if raycing.is_valid_uuid(self._figureError):
            if self.bl is not None:
                fe = self.bl.fesDict.get(self._figureError)
            else:
                fe = None
                print(f"Figure Error instance with UUID {self._material} doesn't exist!")
        else:
            fe = self._figureError
        return fe

    @figureError.setter
    def figureError(self, figureError):
        self._figureError = figureError

    @property
    def limPhysX(self):
        return self._limPhysX

    @limPhysX.setter
    def limPhysX(self, limPhysX):
        if limPhysX is None:
            self._limPhysX = raycing.Limits([-raycing.maxHalfSizeOfOE,
                                             raycing.maxHalfSizeOfOE])
        else:
            self._limPhysX = raycing.Limits(limPhysX)
        self.get_surface_limits()

    @property
    def limPhysY(self):
        return self._limPhysY

    @limPhysY.setter
    def limPhysY(self, limPhysY):
        if limPhysY is None:
            self._limPhysY = raycing.Limits([-raycing.maxHalfSizeOfOE,
                                             raycing.maxHalfSizeOfOE])
        else:
            self._limPhysY = raycing.Limits(limPhysY)
        self.get_surface_limits()

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

    def get_orientation_quaternion(self):
        """Will be used with xrtGlow for fast orientation tracking"""
        try:  # Experimental
            if all([hasattr(self, angle) for angle in ['pitch', 'roll', 'yaw',
                    'extraPitch', 'extraRoll', 'extraYaw', 'positionRoll']]):
                rotAx = {'x': self.pitch,
                         'y': self.roll+self.positionRoll,
                         'z': self.yaw}
                extraRotAx = {'x': self.extraPitch,
                              'y': self.extraRoll,
                              'z': self.extraYaw}
                rotSeq = (self.rotationSequence[slice(1, None, 2)])[::-1]
                extraRotSeq = (
                    self.extraRotationSequence[slice(1, None, 2)])[::-1]
                rotation = scprot.from_euler(
                        rotSeq, [rotAx[i] for i in rotSeq]).as_quat()
                extraRot = scprot.from_euler(
                    extraRotSeq,
                    [extraRotAx[i] for i in extraRotSeq]).as_quat()
                rotation = \
                    [rotation[-1], rotation[0], rotation[1], rotation[2]]
                extraRot = \
                    [extraRot[-1], extraRot[0], extraRot[1], extraRot[2]]

                orientationQ = raycing.multiply_quats(rotation, extraRot)

                # new_norm = raycing.quat_vec_rotate(
                #     np.array([0, 0, 1]), rotation)

                return orientationQ
        except Exception as e:
            # raise
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
        elif isinstance(pitch, str):
            pitch = raycing.auto_units_angle(pitch)
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
        elif isinstance(pitch, str):
            pitch = raycing.auto_units_angle(pitch)
        return 2 * p * q / (p+q) * np.sin(abs(pitch))

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position. Typically is
        overridden in the derived classes. Must return either a scalar or an
        array of the length of *x* and *y*."""
        return np.zeros_like(y)  # just flat

    def local_z_distorted(self, x, y):
        if self.figureError is not None and hasattr(
                self.figureError, 'local_z_distorted'):
            return self.figureError.local_z_distorted(x, y)
        else:
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
        if self.figureError is not None and hasattr(
                self.figureError, 'local_n_distorted'):
            if self.isParametric:
                r = self.local_r(x, y)
                x, y, _ = self.param_to_xyz(x, y, r)
            return self.figureError.local_n_distorted(x, y)
        else:
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
        if self.figureError is not None and hasattr(
                self.figureError, 'local_z_distorted'):
            r = self.local_r(s, phi)
            x, y, z = self.param_to_xyz(s, phi, r)
            z += self.figureError.local_z_distorted(x, y)
            _, _, r1 = self.xyz_to_param(x, y, z)
            return r1 - r  # TODO: Check if this is correct
        else:
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

        if not all([hasattr(self, arg) for arg in [
                'curSurface', 'limPhysX', 'limPhysY', 'limOptX', 'limOptY']]):
            return
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

    def local_to_global(self, lb, returnBeam=False, **kwargs):
        dx, dy, dz = 0, 0, 0
        extraAnglesSign = 1.  # only for pitch and yaw
        # if isinstance(self, DCM):
        if hasattr(self, 'cryst2pitch'):
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
        # if isinstance(self, DCM):
        if hasattr(self, 'cryst2pitch'):
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
