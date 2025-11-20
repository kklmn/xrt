# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy import interpolate, ndimage

from .. import raycing
from .physconsts import CH
from .oes_base import OE


class NormalFZP(OE):
    """Implements a circular Fresnel Zone Plate, as it is described in
    X-Ray Data Booklet, Section 4.4. The zones lie on the same flat plane, they
    have a zero thickness and have transmittivity zero and one. The optical
    axis is the local Z axis.

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

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        self._f = f
        self.reset()

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, E):
        self._E = E
        self.reset()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        self.reset()

    @property
    def thinnestZone(self):
        return self._thinnestZone

    @thinnestZone.setter
    def thinnestZone(self, thinnestZone):
        self._thinnestZone = thinnestZone
        self.reset()

    def __pop_kwargs(self, **kwargs):
        self.f = kwargs.pop('f')
        self.E = kwargs.pop('E')
        self.N = kwargs.pop('N', 1000)
        self.isCentralZoneBlack = kwargs.pop('isCentralZoneBlack', True)
        self.thinnestZone = kwargs.pop('thinnestZone', None)
#        self.reset()
        kwargs['limPhysX'] = [-self.rn[-1], self.rn[-1]]
        kwargs['limPhysY'] = [-self.rn[-1], self.rn[-1]]
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'FZP'

    def reset(self):
        if all([hasattr(self, val) for val in ['f', 'E', 'N',
                                               'thinnestZone']]):
            lambdaE = CH / self.E * 1e-7
            if self.thinnestZone is not None:
                self._N = lambdaE * self.f / 4. / self.thinnestZone**2
            self.zones = np.arange(self.N+1)
            self.rn = np.sqrt(self.zones*self.f*lambdaE +
                              0.25*(self.zones*lambdaE)**2)
            if raycing._VERBOSITY_ > 10:
                print(self.rn)
                print(self.f, self.N)
                print('R(N)={0}, dR(N)={1}'.format(
                      self.rn[-1], self.rn[-1]-self.rn[-2]))
            self.r_to_i = interpolate.interp1d(
                self.rn, self.zones, bounds_error=False, fill_value=0)
            self.i_to_r = interpolate.interp1d(
                self.zones, self.rn, bounds_error=False, fill_value=0)
            self.limPhysX = [-self.rn[-1], self.rn[-1]]
            self.limPhysY = [-self.rn[-1], self.rn[-1]]

#            self.drn = self.rn[1:] - self.rn[:-1]

    def rays_good_gn(self, x, y, z):
        """Returns *state* value as inherited from :class:`OE`. The rays that
        fall inside the opaque zones are additionally considered as lost."""
        locState = OE.rays_good(self, x, y, z)
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


# class GeneralFZPin0YZ(ToroidMirror):
# class GeneralFZPin0YZ(EllipticalMirrorParam):
class GeneralFZPin0YZ(OE):
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
        super().__init__(*args, **kwargs)
        self.use_rays_good_gn = True  # use rays_good_gn instead of rays_good
        if self.grazingAngle is None:
            self.grazingAngle = self.pitch
        self.reset()

    @property
    def f1(self):
        return self._f1

    @f1.setter
    def f1(self, f1):
        self._f1 = f1
        self.reset()

    @property
    def f2(self):
        return self._f2

    @f2.setter
    def f2(self, f2):
        self._f2 = f2
        self.reset()

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, E):
        self._E = E
        self.reset()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        self.reset()

    @property
    def phaseShift(self):
        return self._phaseShift

    @phaseShift.setter
    def phaseShift(self, phaseShift):
        self._phaseShift = phaseShift
        self.reset()

    @property
    def grazingAngle(self):
        return self._grazingAngle

    @grazingAngle.setter
    def grazingAngle(self, grazingAngle):
        self._grazingAngle = raycing.auto_units_angle(grazingAngle)

    def __pop_kwargs(self, **kwargs):
        self.f1 = kwargs.pop('f1')  # in local coordinates!!!
        self.f2 = kwargs.pop('f2')  # in local coordinates!!!
        self.E = kwargs.pop('E')
        self.N = kwargs.pop('N', 1000)
        self.phaseShift = kwargs.pop('phaseShift', 0)
        self.vorticity = kwargs.pop('vorticity', 0)
        self.grazingAngle = kwargs.pop('grazingAngle', None)
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'FZP'

    def reset(self):
        if all([hasattr(self, val) for val in ['E', 'phaseShift']]):
            self.lambdaE = CH / self.E * 1e-7
            self.minHalfLambda = None
            self.set_phase_shift(self.phaseShift)

    def set_phase_shift(self, phaseShift):
        self._phaseShift = phaseShift
        if self.phaseShift:
            self._phaseShift /= np.pi

    def rays_good_gn(self, x, y, z):
        locState = super().rays_good(x, y, z)
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
        zone = np.ones_like(x, dtype=np.int32) * (self.N+2)
        zone[good] = np.floor(halfLambda).astype(np.int32)
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

        gz = np.zeros_like(x[goodN])
        r = np.sqrt(x[goodN]**2 + y[goodN]**2)
        diva = a[zone[goodN]+1] - a[zone[goodN]-1]
        diva[diva == 0] = 1e20
        divb = b[zone[goodN]+1] - b[zone[goodN]-1]
        divb[divb == 0] = 1e20
        xy = (x[goodN]**2/diva + y[goodN]**2/divb) / r**2
        gx = -x[goodN] * xy / r
        gy = -y[goodN] * xy / r

        gn = gx, gy, gz
        return locState, gn


class BlazedGrating(OE):
    r"""Implements a grating of triangular shape given by two angles. The front
    side of the triangle (the one looking towards the source) is at *blaze*
    angle to the base plane. The back side is at *antiblaze* angle.

    .. note::

        In contrast to the geometric implementation of the grating diffraction
        when the deflection is calculated by the grating equation, the
        diffraction by :class:`BlazedGrating` **is meant to be used by the wave
        propagation methods**\ , see :ref:`gallery3`. In those methods, the
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
    small vertical shift. The difference is purely esthetic.

        +-----------------+--------------------+
        |       OE        |   BlazedGrating    |
        +=================+====================+
        | |OE_shadowing|  | |Blazed_shadowing| |
        +-----------------+--------------------+

        .. |OE_shadowing| imagezoom:: _images/1-LEG_profile-default.png
           :scale: 50 %
        .. |Blazed_shadowing| imagezoom:: _images/1-LEG_profile-adhoc.png
           :loc: upper-right-corner
           :scale: 50 %
    """

    def __init__(self, *args, **kwargs):
        r"""
        *blaze*, *antiblaze*: float
            Angles in radians

        *rho*: float
            Constant line density in inverse mm. If the density is variable,
            use *gratingDensity* from the parental class :class:`OE` with the
            1st argument 'y' (i.e. along y-axis).


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.reset()

    @property
    def rho0(self):
        return self._rho0

    @rho0.setter
    def rho0(self, rho0):
        self._rho0 = rho0
        self.reset()

    @property
    def blaze(self):
        return self._blaze

    @blaze.setter
    def blaze(self, blaze):
        self._blaze = blaze
        self.reset()

    @property
    def antiblaze(self):
        return self._antiblaze

    @antiblaze.setter
    def antiblaze(self, antiblaze):
        self._antiblaze = antiblaze
        self.reset()

    @property
    def gratingDensity(self):
        return self._gratingDensity

    @gratingDensity.setter
    def gratingDensity(self, gratingDensity):
        self._gratingDensity = gratingDensity
        self.reset()

    def __pop_kwargs(self, **kwargs):
        self.blaze = raycing.auto_units_angle(kwargs.pop('blaze'))
        self.antiblaze = raycing.auto_units_angle(
            kwargs.pop('antiblaze', np.pi*0.4999))
        self.rho0 = kwargs.pop('rho', 1)
        return kwargs

    def reset(self):
        if all([hasattr(self, field) for field in ['rho0', 'gratingDensity',
                                                   'blaze', 'antiblaze']]):
            if self.gratingDensity is not None:
                self.rho0 = self.gratingDensity[1]
                self.coeffs = self.gratingDensity[2:]
                self.ticks = []
                lim = self.limOptY if self.limOptY is not None else \
                    self.limPhysY
                self.ticksN = int(round(
                    self._get_groove(lim[1]) - self._get_groove(lim[0])))
                y = lim[0]
                while y < lim[1]:
                    self.ticks.append(y)
                    y += self._get_period(y)
                self.ticks = np.array(self.ticks)
                print("tick len {0}, integrated as {1}".format(
                    len(self.ticks), self.ticksN))

            self.rho_1 = 1. / self.rho0

            self.sinBlaze, self.cosBlaze, self.tanBlaze =\
                np.sin(self.blaze), np.cos(self.blaze), np.tan(self.blaze)
            self.sinAntiblaze, self.cosAntiblaze, self.tanAntiblaze =\
                np.sin(self.antiblaze), np.cos(self.antiblaze), \
                np.tan(self.antiblaze)

    def _get_period(self, coord):
        poly = 0.
        for ic, coeff in enumerate(self.coeffs):
            poly += (ic+1) * coeff * coord**ic
        dy = 1. / self.rho0 / poly
        # if type(dy) == float:
        #     assert dy > 0, "wrong coefficients: negative groove density"
        return abs(dy)

    def _get_groove(self, coord):
        poly = 0.
        for ic, coeff in enumerate(self.coeffs):
            poly += coeff * coord**(ic+1)
        return self.rho0 * poly

    def assign_auto_material_kind(self, material):
        material.kind = 'mirror'  # to be used with wave propagation

    def local_pre(self, x, y):
        if self.gratingDensity is not None:
            y0ind = np.searchsorted(self.ticks[:-1], y) - 1
            y0 = self.ticks[y0ind]
            y1 = self.ticks[y0ind+1]
            yL = y - y0
        else:
            y0 = (y // self.rho_1) * self.rho_1
            y1 = y0 + self.rho_1
            yL = y % self.rho_1
            y0ind = 0
        yC = (y1-y0) / (1 + self.tanAntiblaze/self.tanBlaze)
        return y0ind, y0, y1, yC, yL

    def local_z(self, x, y):
        y0ind, y0, y1, yC, yL = self.local_pre(x, y)
        z = np.where(yL > yC, -(y1-y) * self.tanBlaze, -yL * self.tanAntiblaze)
        if self.gratingDensity is not None:
            z[(y0ind < 1) | (y0ind > len(self.ticks)-2)] = 0
        return z

    def local_n(self, x, y):
        y0ind, y0, y1, yC, yL = self.local_pre(x, y)
        n = [np.zeros_like(x),
             np.where(yL > yC, -self.sinBlaze, self.sinAntiblaze),
             np.where(yL > yC, self.cosBlaze, self.cosAntiblaze)]
        if self.gratingDensity is not None:
            n[1][(y0ind < 1) | (y0ind > len(self.ticks)-2)] = 0.
            n[2][(y0ind < 1) | (y0ind > len(self.ticks)-2)] = 1.
        return n

    def find_intersection(self, local_f, t1, t2, x, y, z, a, b, c,
                          invertNormal, derivOrder=0):
        b_c = b / c
        if self.gratingDensity is not None:
            y0ind = np.searchsorted(self.ticks[:-1], y - b_c*z) - 1
            y0 = self.ticks[y0ind]
            y1 = self.ticks[y0ind+1]
        else:
            n = np.floor((y - b_c*z) / self.rho_1)
            y0 = self.rho_1 * n
            y1 = y0 + self.rho_1

        if self.antiblaze == np.pi/2:
            zabl = (y0-y) / b_c + z
        else:
            zabl = -self.tanAntiblaze * (y - b_c*z - y0) /\
                (1 + self.tanAntiblaze*b_c)
        if self.blaze == np.pi/2:
            zbl = (y1-y) / b_c + z
        else:
            zbl = self.tanBlaze * (y - b_c*z - y1) / (1 - self.tanBlaze*b_c)
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
        return d * self.rho0


class LaminarGrating(OE):
    """
    Implements a grating of rectangular profile.

    """

    def __init__(self, *args, **kwargs):
        """
        *rho*: float
            Lines density in inverse mm.

        *aspect*: float
            Top-to-period ratio of the groove.

        *depth*: float
            Depth of the groove in mm.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
#        self.rho_1 = 1. / self.rho  # Period of the grating in [mm]
        self.illuminatedGroove = 0

    @property
    def rho0(self):
        return self._rho0

    @rho0.setter
    def rho0(self, rho0):
        self._rho0 = rho0
        self.rho_1 = 1. / self.rho0

    def __pop_kwargs(self, **kwargs):
        self.rho = kwargs.pop('rho')
        self.aspect = kwargs.pop('aspect', 0.5)
        self.depth = kwargs.pop('depth', 1e-3)
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'mirror'  # to be used with wave propagation

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
        rindex = (yL < raycing.zEps)
        norm_y[rindex] = 1
        norm_z[rindex] = 0
        rindex = (np.abs(yL - groove) < raycing.zEps)
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

        if len(dy[bottom]) > 0:
            ilG = (np.max(dy[bottom]) - np.min(dy[bottom])) / self.rho_1
            if ilG > self.illuminatedGroove:
                self.illuminatedGroove = ilG

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
        return self.aspect + self.illuminatedGroove


class VLSLaminarGrating(OE):
    """
    Implements a grating of rectangular profile with variable period.

    """

    def __init__(self, *args, **kwargs):
        r"""
        *aspect*: float
            Top-to-period ratio of the groove.

        *depth*: float
            Depth of the groove in mm.

        For the VLS density, use *gratingDensity* of the parental class
        :class:`OE` with the 1st argument 'y' (i.e. along y-axis).


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    def reset(self):
        if self.gratingDensity is not None:
            self.rho0 = self.gratingDensity[1]
            self.coeffs = self.gratingDensity[2:]
        self.ticks = []
        p0 = self.limOptY[0]
        while p0 < self.limOptY[1]:
            self.ticks.append(p0)
            p0 += self._get_period(p0)
        self.ticks = np.array(self.ticks)
        self.illuminatedGroove = 0
        self.rho_1 = 1. / self.rho0

    def _get_period(self, coord):
        poly = 0.
        for ic, coeff in enumerate(self.coeffs):
            poly += (ic+1) * coeff * coord**ic
        dy = 1. / self.rho0 / poly
        # if type(dy) == float:
        #     assert dy > 0, "wrong coefficients: negative groove density"
        return abs(dy)

    def __pop_kwargs(self, **kwargs):
        self.rho0 = kwargs.pop('rho', None)
        self.aspect = kwargs.pop('aspect', 0.5)
        self.coeffs = kwargs.pop('coeffs', [1, 0, 0])
        self.depth = kwargs.pop('depth', 1e-3)  # 1 micron depth
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'mirror'  # to be used with wave propagation

    def local_z(self, x, y):
        z = np.zeros_like(y)
        y0ind = np.searchsorted(self.ticks[:-1], y)
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
        rindex = (yL < raycing.zEps)
        norm_y[rindex] = 1
        norm_z[rindex] = 0
        rindex = (np.abs(yL - groove) < raycing.zEps)
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
        y0ind = np.searchsorted(self.ticks[:-1], y2)
        periods = self.ticks[list(y0ind)] - self.ticks[list(y0ind - 1)]
        yL = y2 - self.ticks[list(y0ind - 1)]
        groove = periods * (1. - self.aspect)
        x2 = x + z * a_c

        gr = (yL < groove)
        dyRel[gr] = b_c[gr] * self.depth
        dy[gr] = yL[gr] - dyRel[gr]
        bottom = (dy > abs(dyRel)) & (dy < groove-abs(dyRel))

        if len(dy[bottom]) > 0:
            ilG = np.max(dy[bottom] / periods[bottom]) -\
                np.min(dy[bottom] / periods[bottom])

            if ilG > self.illuminatedGroove:
                self.illuminatedGroove = ilG
            print("ilG", self.illuminatedGroove)
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
        return self.aspect + self.illuminatedGroove


VLSGrating = VLSLaminarGrating
