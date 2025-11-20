# -*- coding: utf-8 -*-
import numpy as np

from .. import raycing
from . import sources as rs
from .oes_base import OE


class EllipticalMirrorParam(OE):
    """The elliptical mirror is implemented as a parametric surface. The
    parameterization is the following: *s* - is local coordinate along the
    major axis with origin at the ellipse center. *phi* and *r* are local polar
    coordinates in planes normal to the major axis at every point *s*. The
    polar axis is upwards.

    The *center* of this OE lies on the mirror surface and its *pitch* is Rx
    at this point.

    If *isCylindrical* is True (default is False), the figure is an elliptical
    cylinder being flat in the lateral direction, otherwise it is an ellipsoid
    of revolution around the major axis.

    If *isClosed* is True (default is False), the mirror is a complete surface
    of revolution. Otherwise the mirror is open, i.e. only its lower half is
    effective. If you want a closed mirror, compare this OE with
    :class:`EllipsoidCapillaryMirror` that can produce the same surface, just
    with another meaning of *center* and *pitch* parameters.

    .. note::

        If the mirror is a closed surface, *pitch* should still be non-zero
        even if the major axis lies on the optical axis. A *center* must be
        defined somewhere on the surface and *pitch* is referred to that point.

    The user supplies two foci either by focal distances *p* and *q* (both are
    positive distances from the mirror center to the focal points) or as *f1*
    and *f2* points in the global coordinate system (3-sequences). Any
    combination of (*p* or *f1*) and (*q* or *f2*) is allowed. If *p* is
    supplied, not *f1*, the incoming optical axis is assumed to be along the
    global Y axis. For a general orientation of the ellipse axes, *f1* or
    *pAxis* -- the *p* arm direction in global coordinates -- should be
    supplied.

    .. note::

        Any of *p*, *q*, *f1*, *f2* or *pAxis* can be set as instance
        attributes of this mirror object; the ellipse parameters will be
        recalculated automatically.

    Values of the ellipse semi-major and semi-minor axes lengths can be
    accessed after init as *ellipseA* and *ellipseB* respectively.

    The usage is exemplified in `test_param_mirror.py` and
    `test_ellipsoid_tube_mirror.py`. Both test scripts can produce a 3D view by
    xrtGlow if a corresponding option at the top of the script is enabled.

    """

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

        *f1* and *f2*: 3-sequence
            Focal points in the global coordinate system. Alternatives for,
            correspondingly, *p* and *q*.

        *pAxis*: 3-sequence
            Used with *p*, the *p* arm direction in global coordinates,
            defaults to the global Y axis.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.isParametric = True
        self._reset_pq()  # self.p, self.q, self.f1, self.f2, self.pAxis)

    def _to_global(self, lb):
        # if self.extraPitch or self.extraRoll or self.extraYaw:
        #     raycing.rotate_beam(
        #         lb, rotationSequence='-'+self.extraRotationSequence,
        #         pitch=self.extraPitch, roll=self.extraRoll,
        #         yaw=self.extraYaw, skip_xyz=True)
        raycing.rotate_beam(lb, rotationSequence='-'+self.rotationSequence,
                            pitch=self.pitch, roll=self.roll+self.positionRoll,
                            yaw=self.yaw, skip_xyz=True)
        raycing.virgin_local_to_global(self.bl, lb, self.center, skip_xyz=True)

    def reset_pqpitch(self, p=None, q=None, pitch=None):
        """Compatibility method. To pass pitch is not needed any longer."""
        self._reset_pq()

    def reset_pq(self, p=None, q=None, f1=None, f2=None, pAxis=None):
        """Compatibility method. All calculations moved to setters."""
        self._reset_pq()

    def _reset_pq(self):  # , p=None, q=None, f1=None, f2=None, pAxis=None):
        """This method allows re-assignment of *p*, *q*, *pitch*, *f1* and *f2*
        from outside of the constructor.
        """
        if not all([hasattr(self, v) for v in
                    ['_p', '_q', '_f1', '_f2', '_pAxis',
                     '_pitchVal', '_roll', '_yaw',
                     '_positionRoll', 'rotationSequence']]):
            return
        lbn = rs.Beam(nrays=1)
        lbn.a[:], lbn.b[:], lbn.c[:] = 0, 0, 1
        self._to_global(lbn)
        normal = lbn.a[0], lbn.b[0], lbn.c[0]

        if self.f1 is not None:
            p = (sum((x-y)**2 for x, y in zip(self.center, self.f1)))**0.5
            axis = [c-f for c, f in zip(self.center, self.f1)]
            self._p = p
        else:
            axis = self.pAxis if self.pAxis else [0, 1, 0]

        norm = sum([a**2 for a in axis])**0.5
        sintheta = sum([a*n for a, n in zip(axis, normal)]) / norm
        absPitch = abs(np.arcsin(sintheta))

        if self.f2 is not None:
            q = (sum((x-y)**2 for x, y in zip(self.center, self.f2)))**0.5
            self._q = q

        # gamma is angle between the major axis and the mirror surface
        if self.p and self.q:
            gamma = np.arctan2((self.p - self.q) * np.sin(absPitch),
                               (self.p + self.q) * np.cos(absPitch))
            self.cosGamma = np.cos(gamma)
            self.sinGamma = np.sin(gamma)
            # (y0, z0) is the ellipse center in local coordinates
            self.y0 = (self.q - self.p)/2. * np.cos(absPitch)
            self.z0 = (self.q + self.p)/2. * np.sin(absPitch)
            self.ellipseA = (self.q + self.p)/2.
            self.ellipseB = np.sqrt(self.q * self.p) * np.sin(absPitch)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p
        self._reset_pq()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q
        self._reset_pq()

    @property
    def f1(self):
        return self._f1

    @f1.setter
    def f1(self, f1):
        self._f1 = f1
        self._reset_pq()

    @property
    def f2(self):
        return self._f2

    @f2.setter
    def f2(self, f2):
        self._f2 = f2
        self._reset_pq()

    @property
    def pAxis(self):
        return self._pAxis

    @pAxis.setter
    def pAxis(self, pAxis):
        self._pAxis = pAxis
        self._reset_pq()

    def __pop_kwargs(self, **kwargs):
        self.f1 = kwargs.pop('f1', None)
        self.f2 = kwargs.pop('f2', None)
        self.pAxis = kwargs.pop('pAxis', None)
        self.p = kwargs.pop('p', 1000)  # source-to-mirror
        self.q = kwargs.pop('q', 1000)  # mirror-to-focus
        self.isCylindrical = kwargs.pop('isCylindrical', False)
        self.isClosed = kwargs.pop('isClosed', False)
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
        r = self.ellipseB * np.sqrt(abs(1 - s**2 / self.ellipseA**2))
        if self.isCylindrical:
            r /= abs(np.cos(phi))
        if self.isClosed:
            return r
        return np.where(abs(phi) > np.pi/2, r, np.ones_like(phi)*1e20)

    def local_n(self, s, phi):
        A2s2 = np.array(self.ellipseA**2 - s**2)
        A2s2[A2s2 <= 0] = 1e22  # this rays will be lost
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


EllipticalMirror = EllipticalMirrorParam


class ParabolicalMirrorParam(OE):
    """The parabolical mirror is implemented as a parametric surface. The
    parameterization is the following: *s* - is local coordinate along the
    paraboloid axis with origin at the focus. *phi* and *r* are local polar
    coordinates in planes normal to the axis at every point *s*. The polar axis
    is upwards.

    The user supplies one (and only one) focal distance *p* or *q* as a
    positive value. Alternatively, instead of *p* one can specify *f1*
    (3-sequence) as a 3D point in the global coordinate system and instead of
    *q* -- *f2*. If *p* or *q* is supplied, the paraboloid axis isassumed to be
    along the global Y axis, otherwise supply *parabolaAxis* as a vector in
    global coordinates.

    If *isCylindrical* is True, the figure is an
    parabolical cylinder, otherwise it is a paraboloid of revolution around the
    major axis.

    If *isClosed* is True (default is False), the mirror is a complete surface
    of revolution. Otherwise the mirror is open, i.e. only its lower half is
    effective. If you want a closed mirror, compare this OE with
    :class:`ParaboloidCapillaryMirror` that can produce the same surface, just
    with another meaning of *center* and *pitch* parameters.

    .. note::

        Any of *p*, *q*, *f1*, *f2* or *parabolaAxis* can be set as instance
        attributes of this mirror object; the ellipsoid parameters will be
        recalculated automatically.

    The usage is exemplified in `test_param_mirror.py`.

    """

    def __init__(self, *args, **kwargs):
        """
        *p* or *q*: float
            *p* and *q* arms of the mirror, both are positive. One and only one
            of them must be given.

        *f1* and *f2*: 3-sequence
            Focal points in the global coordinate system. Alternatives for,
            correspondingly, *p* and *q*. Only one of them must be given.

        *parabolaAxis*: 3-sequence
            Used with *p* or *q*, the parabola axis in global coordinates,
            defaults to the global Y axis.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.isParametric = True
        self._reset_pq()

    def _to_global(self, lb):
        raycing.rotate_beam(lb, rotationSequence='-'+self.rotationSequence,
                            pitch=self.pitch, roll=self.roll+self.positionRoll,
                            yaw=self.yaw, skip_xyz=True)
        raycing.virgin_local_to_global(self.bl, lb, self.center, skip_xyz=True)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p
        self._reset_pq()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q
        self._reset_pq()

    @property
    def f1(self):
        return self._f1

    @f1.setter
    def f1(self, f1):
        self._f1 = f1
        self._reset_pq()

    @property
    def f2(self):
        return self._f2

    @f2.setter
    def f2(self, f2):
        self._f2 = f2
        self._reset_pq()

    @property
    def parabolaAxis(self):
        return self._parabolaAxis

    @parabolaAxis.setter
    def parabolaAxis(self, parabolaAxis):
        self._parabolaAxis = parabolaAxis
        self._reset_pq()

    def reset_pq(self, p=None, q=None, f1=None, f2=None, parabolaAxis=None):
        """Compatibility method. All calculations moved to setters."""
        self._reset_pq()

    def _reset_pq(self):
        """This method allows re-assignment of *p*, *q* and *pitch* from
        outside of the constructor.
        """
        if not all([hasattr(self, v) for v in
                    ['_p', '_q', '_f1', '_f2', '_parabolaAxis',
                     '_pitchVal', '_roll', '_yaw',
                     '_positionRoll', 'rotationSequence']]):
            return

        lbn = rs.Beam(nrays=1)
        lbn.a[:], lbn.b[:], lbn.c[:] = 0, 0, 1
        self._to_global(lbn)
        normal = lbn.a[0], lbn.b[0], lbn.c[0]
        p = None
        q = None
        if self.f1 is not None:
            p = (sum((x-y)**2 for x, y in zip(self.center, self.f1)))**0.5
            axis = [c-f for c, f in zip(self.center, self.f1)]
        elif self.f2 is not None:
            q = (sum((x-y)**2 for x, y in zip(self.center, self.f2)))**0.5
            axis = [c-f for c, f in zip(self.center, self.f2)]
        else:
            axis = self.parabolaAxis if self.parabolaAxis else [0, 1, 0]

        norm = sum([a**2 for a in axis])**0.5
        sintheta = sum([a*n for a, n in zip(axis, normal)]) / norm
        absPitch = abs(np.arcsin(sintheta))

        if p is not None:
            self._p = p
        if q is not None:
            self._q = q
        if ((self.p is not None) and (self.q is not None)) or\
                ((self.p is None) and (self.q is None)):
            print('p={0}, q={1}'.format(self.p, self.q))
            raise ValueError('One and only one of p (or f1) or q (or f2)'
                             ' must be None!')
        # (y0, z0) is the focus point in local coordinates
        # gamma is angle between the parabola axis and the mirror surface
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
        self.f1 = kwargs.pop('f1', None)
        self.f2 = kwargs.pop('f2', None)
        self.p = kwargs.pop('p', None)  # source-to-mirror
        self.q = kwargs.pop('q', None)  # mirror-to-focus
        self.parabolaAxis = kwargs.pop('parabolaAxis', None)
        self.isCylindrical = kwargs.pop('isCylindrical', False)
        self.isClosed = kwargs.pop('isClosed', False)
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
        r2 = self.parabParam*s + self.parabParam**2
        r2[r2 < 0] = 0
        r = 2 * r2**0.5
        if self.isCylindrical:
            r /= abs(np.cos(phi))
        if self.isClosed:
            return r
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


ParabolicMirror = ParabolicalMirrorParam


class HyperbolicMirrorParam(OE):
    """The hyperbolic mirror is implemented as a parametric surface. The
    parameterization is the following: *s* - is local coordinate along the
    major axis with origin at the hyperbola center. *phi* and *r* are local
    polar coordinates in planes normal to the major axis at every point *s*.
    The polar axis is upwards.

    Unlike EllipticalMirrorParam, reflective is the *outer* surface by default.
    If the inner surface is wanted, the instance variable `invertNormal` should
    be set equal to 1 after the class instantiation (alternatively, change it
    in a subclass).

    .. note::

        In both cases -- outer or inner surface -- foci *f1* and *f2* are
        numbered in the sense of propagation direction: from *f1* to the mirror
        and then to imaginary focus (i.e. away from the focus) *f2* after the
        reflection. In the outer case, *f1* is the farther focus, while in the
        inner case it is the closer focus. See the example
        `test_param_mirror.py`, also in 3D.

    The *center* of this OE lies on the mirror surface and its *pitch* is Rx
    at this point.

    If *isCylindrical* is True (default is False), the figure is a hyperbolic
    cylinder being flat in the lateral direction, otherwise it is a hyperboloid
    of revolution around the major axis.

    If *isClosed* is True (default is False), the mirror is a complete surface
    of revolution. Otherwise the mirror is open, i.e. only its one half is
    effective. If you want a closed mirror, compare this OE with
    :class:`HyperboloidCapillaryMirror` that can produce the same surface, just
    with another meaning of *center* and *pitch* parameters.

    .. note::

        If the mirror is a closed surface, *pitch* should still be non-zero
        even if the major axis lies on the optical axis. A *center* must be
        defined somewhere on the surface and *pitch* is referred to that point.

    The user supplies two foci either by focal distances *p* and *q* (both are
    positive distances from the mirror center to the focal points) or as *f1*
    and *f2* points in the global coordinate system (3-sequences). Any
    combination of (*p* or *f1*) and (*q* or *f2*) is allowed. If *p* is
    supplied, not *f1*, the incoming optical axis is assumed to be along the
    global Y axis. For a general orientation of the hyperbola axes *f1* or
    *pAxis* -- the *p* arm direction in global coordinates -- should be
    supplied.

    .. note::

        Any of *p*, *q*, *f1*, *f2* or *pAxis* can be set as instance
        attributes of this mirror object; the hyperbola parameters will be
        recalculated automatically.

    Values of the hyperbola semi-major and semi-minor axes lengths can be
    accessed after init as *hyperbolaA* and *hyperbolaB* respectively.

    The usage is exemplified in `test_param_mirror.py` and
    `test_hyperboloid_tube_mirror.py`. Both test scripts can produce a 3D view
    by xrtGlow if a corresponding option at the top of the script is enabled.

    """

    def __init__(self, *args, **kwargs):
        """
        *p* and *q*: float
            *p* and *q* arms of the mirror -- positive distances from the
            mirror center to the foci.

        *f1* and *f2*: 3-sequence
            Focal points in the global coordinate system. Alternatives for,
            correspondingly, *p* and *q*.

        *pAxis*: 3-sequence
            Used with *p*, the *p* arm direction in global coordinates,
            defaults to the global Y axis.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.isParametric = True
        self.invertNormal = -1  # the outer surface is reflective
        self._reset_pq()  # self.p, self.q, self.f1, self.f2, self.pAxis)

    def _to_global(self, lb):
        # if self.extraPitch or self.extraRoll or self.extraYaw:
        #     raycing.rotate_beam(
        #         lb, rotationSequence='-'+self.extraRotationSequence,
        #         pitch=self.extraPitch, roll=self.extraRoll,
        #         yaw=self.extraYaw, skip_xyz=True)
        raycing.rotate_beam(lb, rotationSequence='-'+self.rotationSequence,
                            pitch=self.pitch, roll=self.roll+self.positionRoll,
                            yaw=self.yaw, skip_xyz=True)
        raycing.virgin_local_to_global(self.bl, lb, self.center, skip_xyz=True)

    def reset_pqpitch(self, p=None, q=None, pitch=None):
        """Compatibility method. To pass pitch is not needed any longer."""
        self._reset_pq()

    def reset_pq(self, p=None, q=None, f1=None, f2=None, pAxis=None):
        """Compatibility method. All calculations moved to setters."""
        self._reset_pq()

    def _reset_pq(self):  # , p=None, q=None, f1=None, f2=None, pAxis=None):
        """This method allows re-assignment of *p*, *q*, *pitch*, *f1* and *f2*
        from outside of the constructor.
        """
        if not all([hasattr(self, v) for v in
                    ['_p', '_q', '_f1', '_f2', '_pAxis',
                     '_pitchVal', '_roll', '_yaw',
                     '_positionRoll', 'rotationSequence']]):
            return
        lbn = rs.Beam(nrays=1)
        lbn.a[:], lbn.b[:], lbn.c[:] = 0, 0, 1
        self._to_global(lbn)
        normal = lbn.a[0], lbn.b[0], lbn.c[0]

        if self.f1 is not None:
            p = (sum((x-y)**2 for x, y in zip(self.center, self.f1)))**0.5
            axis = [c-f for c, f in zip(self.center, self.f1)]
            self._p = p
        else:
            axis = self.pAxis if self.pAxis else [0, 1, 0]

        norm = sum([a**2 for a in axis])**0.5
        sintheta = sum([a*n for a, n in zip(axis, normal)]) / norm
        absPitch = abs(np.arcsin(sintheta))

        if self.f2 is not None:
            q = (sum((x-y)**2 for x, y in zip(self.center, self.f2)))**0.5
            self._q = q

        # gamma is angle between the major axis and the mirror surface
        if self.p and self.q:
            gamma = np.arctan2((self.p + self.q) * np.sin(absPitch),
                               (self.p - self.q) * np.cos(absPitch))
            self.cosGamma = np.cos(gamma)
            self.sinGamma = np.sin(gamma)
            # (y0, z0) is the hyperbola center in local coordinates
            self.y0 = -(self.p + self.q)/2. * np.cos(absPitch)
            self.z0 = (self.p - self.q)/2. * np.sin(absPitch)
            self.hyperbolaA = abs(self.p - self.q)/2.
            self.hyperbolaB = np.sqrt(self.p*self.q) * np.sin(absPitch)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p
        self._reset_pq()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q
        self._reset_pq()

    @property
    def f1(self):
        return self._f1

    @f1.setter
    def f1(self, f1):
        self._f1 = f1
        self._reset_pq()

    @property
    def f2(self):
        return self._f2

    @f2.setter
    def f2(self, f2):
        self._f2 = f2
        self._reset_pq()

    @property
    def pAxis(self):
        return self._pAxis

    @pAxis.setter
    def pAxis(self, pAxis):
        self._pAxis = pAxis
        self._reset_pq()

    def __pop_kwargs(self, **kwargs):
        self.f1 = kwargs.pop('f1', None)
        self.f2 = kwargs.pop('f2', None)
        self.pAxis = kwargs.pop('pAxis', None)
        self.p = kwargs.pop('p', 1000)  # source-to-mirror
        self.q = kwargs.pop('q', 1000)  # mirror-to-focus
        self.isCylindrical = kwargs.pop('isCylindrical', False)
        self.isClosed = kwargs.pop('isClosed', False)
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
        r = self.hyperbolaB * np.sqrt(abs(s**2/self.hyperbolaA**2 - 1))
        if self.isCylindrical:
            r /= abs(np.cos(phi))
        if self.isClosed:
            return r
        return np.where(abs(phi) < np.pi/2, r, np.ones_like(phi)*1e20)

    def local_n(self, s, phi):
        A2s2 = np.array(s**2 - self.hyperbolaA**2)
        A2s2[A2s2 <= 0] = 1e22  # this rays will be lost
        nr = -self.hyperbolaB / self.hyperbolaA * s / np.sqrt(A2s2)
        norm = np.sqrt(nr**2 + 1)
        b = nr / norm
        if self.isCylindrical:
            a = np.zeros_like(phi)
            c = 1. / norm
        else:
            a = np.sin(phi) / norm
            c = np.cos(phi) / norm
        bNew, cNew = raycing.rotate_x(b, c, self.cosGamma, -self.sinGamma)
        return [a, bNew, cNew]


HyperbolicMirror = HyperbolicMirrorParam


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


class ParaboloidCapillaryMirror(SurfaceOfRevolution):
    """Paraboloid of revolution a.k.a. Mirror Lens. By default will be oriented
    for focusing. Set yaw to 180deg for collimation."""

    def __init__(self, *args, **kwargs):
        r"""
        *q*: float
            Distance from the center of the element to focus.

        *r0*: float
            Radius at the center of the element.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        super().__init__(*args, **kwargs)

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q
        self.reset_focus()

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, r0):
        self._r0 = r0
        self.reset_focus()

    def reset_focus(self, q=None, r0=None):
        if not all([hasattr(self, v) for v in
                    ['_q', '_r0']]):
            return
        self.focus = -0.5*(self.q-(self.q**2+self.r0**2)**0.5)
        self.s0 = self.focus + self.q

    def __pop_kwargs(self, **kwargs):
        self.q = kwargs.pop('q', 500.)  # Distance to parabola focus
        self.r0 = kwargs.pop('r0', 2.5)
        return kwargs

    def local_r(self, s, phi):
        return 2*np.sqrt((self.s0-s)*self.focus)

    def local_n(self, s, phi):
        a = -np.sin(phi)
        b = -np.sqrt(self.focus/(self.s0-s))
        c = -np.cos(phi)
        norm = np.sqrt(a**2 + b**2 + c**2)
        return a/norm, b/norm, c/norm


class EllipsoidCapillaryMirror(SurfaceOfRevolution):
    """Ellipsoid of revolution a.k.a. Mirror Lens. Do not forget to set
    reasonable limPhysY."""

    def __init__(self, *args, **kwargs):
        r"""
        The center is on major axis in the middle of the capillary. *pitch* is
        zero if the capillary axis is parallel to the optical axis.

        *ellipseA*: float
            Semi-major axis.

        *ellipseB*: float
            Semi-minor axis. Do not confuse with the radius of the capillary!

        *workingDistance*: float
            Distance between the end face of the capillary tube and the focal
            point. Mind the length of the optical element for proper
            positioning.

        The usage is exemplified in `test_ellipsoid_tube_mirror.py`. There, one
        can find expressions of ellipse semi-axes based on focal lengths and
        capillary radius.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        super().__init__(*args, **kwargs)

    @property
    def ellipseA(self):
        return self._ellipseA

    @ellipseA.setter
    def ellipseA(self, ellipseA):
        self._ellipseA = ellipseA
        self.reset_curvature()

    @property
    def ellipseB(self):
        return self._ellipseB

    @ellipseB.setter
    def ellipseB(self, ellipseB):
        self._ellipseB = ellipseB
        self.reset_curvature()

    @property
    def workingDistance(self):
        return self._workingDistance

    @workingDistance.setter
    def workingDistance(self, workingDistance):
        self._workingDistance = workingDistance
        self.reset_curvature()

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
        self.reset_curvature()

    def reset_curvature(self):
        if not all([hasattr(self, v) for v in [
                '_ellipseA', '_ellipseB', '_workingDistance', '_limPhysY']]):
            return
        c = (self.ellipseA**2 - self.ellipseB**2)**0.5
        self.ctd = c - self.workingDistance -\
            0.5*np.abs(self.limPhysY[-1]-self.limPhysY[0])

    def __pop_kwargs(self, **kwargs):
        self.ellipseA = kwargs.pop('ellipseA', 10000)  # Semi-major axis
        self.ellipseB = kwargs.pop('ellipseB', 2.5)  # Semi-minor axis
        self.workingDistance = kwargs.pop('workingDistance', 17.)
        return kwargs

    def local_r(self, s, phi):
        r = self.ellipseB * np.sqrt(abs(1 - (self.ctd+s)**2/self.ellipseA**2))
        return r

    def local_n(self, s, phi):
        A2s2 = np.array(self.ellipseA**2 - (self.ctd+s)**2)
        A2s2[A2s2 <= 0] = 1e22  # this rays will be lost
        nr = -self.ellipseB / self.ellipseA * (self.ctd+s) / np.sqrt(A2s2)
        norm = np.sqrt(nr**2 + 1.)
        b = nr / norm
        a = -np.sin(phi) / norm
        c = -np.cos(phi) / norm
        return a, b, c


class HyperboloidCapillaryMirror(SurfaceOfRevolution):
    """Hyperboloid of revolution a.k.a. Mirror Lens. Unlike
    EllipsoidCapillaryMirror, reflective is the *outer* surface. Do not
    forget to set reasonable limPhysY."""

    def __init__(self, *args, **kwargs):
        r"""
        The center is on major axis in the middle of the capillary. *pitch* is
        zero if the capillary axis is parallel to the optical axis.

        *hyperbolaA*: float
            Semi-major axis.

        *hyperbolaB*: float
            Semi-minor axis. Do not confuse with the radius of the capillary!

        *workingDistance*: float
            Distance between the imaginary focus and the front face of the
            capillary tube. Mind the length of the optical element for proper
            positioning.

        The usage is exemplified in `test_hyperboloid_tube_mirror.py`. There,
        one can find expressions of hyperbola semi-axes based on focal lengths
        and capillary radius.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        self.invertNormal = -1  # the outer surface is reflective
        super().__init__(*args, **kwargs)

    @property
    def hyperbolaA(self):
        return self._hyperbolaA

    @hyperbolaA.setter
    def hyperbolaA(self, hyperbolaA):
        self._hyperbolaA = hyperbolaA
        self.reset_curvature()

    @property
    def hyperbolaB(self):
        return self._hyperbolaB

    @hyperbolaB.setter
    def hyperbolaB(self, hyperbolaB):
        self._hyperbolaB = hyperbolaB
        self.reset_curvature()

    @property
    def workingDistance(self):
        return self._workingDistance

    @workingDistance.setter
    def workingDistance(self, workingDistance):
        self._workingDistance = workingDistance
        self.reset_curvature()

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
        self.reset_curvature()

    def reset_curvature(self):
        if not all([hasattr(self, v) for v in [
                '_hyperbolaA', '_hyperbolaB', '_workingDistance',
                '_limPhysY']]):
            return
        c = (self.hyperbolaA**2 + self.hyperbolaB**2)**0.5
        self.ctd = c + self.workingDistance +\
            0.5*np.abs(self.limPhysY[-1]-self.limPhysY[0])

    def __pop_kwargs(self, **kwargs):
        self.hyperbolaA = kwargs.pop('hyperbolaA', 10000)  # Semi-major axis
        self.hyperbolaB = kwargs.pop('hyperbolaB', 2.5)  # Semi-minor axis
        self.workingDistance = kwargs.pop('workingDistance', 17.)
        return kwargs

    def local_r(self, s, phi):
        ss = self.ctd + s
        r = self.hyperbolaB * np.sqrt(abs(ss**2/self.hyperbolaA**2 - 1))
        return r

    def local_n(self, s, phi):
        ss = self.ctd + s
        A2s2 = np.array(ss**2 - self.hyperbolaA**2)
        A2s2[A2s2 <= 0] = 1e22  # this rays will be lost
        nr = -self.hyperbolaB / self.hyperbolaA * ss / np.sqrt(A2s2)
        norm = np.sqrt(nr**2 + 1)
        b = nr / norm
        a = np.sin(phi) / norm
        c = np.cos(phi) / norm
        return a, b, c
