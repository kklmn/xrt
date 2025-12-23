# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "03 Jul 2016"
import numpy as np
import scipy as sp
import inspect

from . import run as rr
from .. import raycing
from .sources_beams import Beam, defaultEnergy, allArguments
from .physconsts import PI2, CHBAR

# _DEBUG replaced with raycing._VERBOSITY_


def make_energy(
        distE, energies, nrays, filamentBeam=False, energyWeights=None):
    """Creates energy distributions with the distribution law given by *distE*.
    *energies* either determine the limits or is a sequence of discrete
    energies. If *distE* is 'lines', *energyWeights* can define the relative
    weights of the lines.
    """
    locnrays = 1 if filamentBeam else int(nrays)
    if distE == 'normal':
        try:
            E = np.random.normal(energies[0], energies[1], locnrays)
        except ValueError:
            E = np.zeros(locnrays)
    elif distE == 'flat':
        E = np.random.uniform(energies[0], energies[1], locnrays)
    elif distE == 'lines':
#        E = np.array(energies)[np.random.randint(len(energies), size=locnrays)]
        E = np.random.choice(energies, size=locnrays, p=energyWeights)
    return E


def make_polarization(polarization, bo, nrays=raycing.nrays):
    r"""Initializes the coherency matrix. The following polarizations are
    supported:

        1) horizontal (*polarization* is a string started with 'h'):

           .. math::

              J = \left( \begin{array}{ccc}1 & 0 \\ 0 & 0\end{array} \right)

        2) vertical (*polarization* is a string started with 'v'):

           .. math::

              J = \left( \begin{array}{ccc}0 & 0 \\ 0 & 1\end{array} \right)

        3) at +45º (*polarization* = '+45'):

           .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & 1 \\ 1 & 1\end{array} \right)

        4) at -45º (*polarization* = '-45'):

           .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & -1 \\ -1 & 1\end{array} \right)

        5) right (*polarization* is a string started with 'r'):

          .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & i \\ -i & 1\end{array} \right)

        5) left (*polarization* is a string started with 'l'):

           .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & -i \\ i & 1\end{array} \right)

        7) unpolarized (*polarization* is None):

           .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & 0 \\ 0 & 1\end{array} \right)

        8) user-defined (*polarization* is 4-sequence with):

           .. math::

              J = \left( \begin{array}{ccc}
              {\rm polarization[0]} &
              {\rm polarization[2]} + i * {\rm polarization[3]} \\
              {\rm polarization[2]} - i * {\rm polarization[3]} &
              {\rm polarization[1]}\end{array} \right)

        """
    def _fill_beam(Jss, Jpp, Jsp, Es, Ep):
        bo.Jss.fill(Jss)
        bo.Jpp.fill(Jpp)
        bo.Jsp.fill(Jsp)
        if hasattr(bo, 'Es'):
            bo.Es.fill(Es)
            if isinstance(Ep, str):
                bo.Ep[:] = np.random.uniform(size=int(nrays)) * 2**(-0.5)
            else:
                bo.Ep.fill(Ep)

    if (polarization is None) or (polarization.startswith('un')):
        _fill_beam(0.5, 0.5, 0, 2**(-0.5), 'random phase')
    elif isinstance(polarization, tuple):
        if len(polarization) != 4:
            raise ValueError('wrong coherency matrix: must be a 4-tuple!')
        bo.Jss.fill(polarization[0])
        bo.Jpp.fill(polarization[1])
        bo.Jsp.fill(polarization[2] + 1j*polarization[3])
    else:
        if polarization.startswith('h'):
            _fill_beam(1, 0, 0, 1, 0)
        elif polarization.startswith('v'):
            _fill_beam(0, 1, 0, 0, 1)
        elif polarization == '+45':
            _fill_beam(0.5, 0.5, 0.5, 2**(-0.5), 2**(-0.5))
        elif polarization == '-45':
            _fill_beam(0.5, 0.5, -0.5, 2**(-0.5), -2**(-0.5))
        elif polarization.startswith('r'):
            _fill_beam(0.5, 0.5, 0.5j, 2**(-0.5), -1j * 2**(-0.5))
        elif polarization.startswith('l'):
            _fill_beam(0.5, 0.5, -0.5j, 2**(-0.5), 1j * 2**(-0.5))
        else:
            raise ValueError('wrong polarization!')


class GeometricSource(object):
    """Implements a geometric source - a source with the ray origin,
    divergence and energy sampled with the given distribution laws."""

    def __init__(
        self, bl=None, name='', center=(0, 0, 0), nrays=raycing.nrays,
        distx='normal', dx=0.32, disty=None, dy=0, distz='normal', dz=0.018,
        distxprime='normal', dxprime=1e-3, distzprime='normal', dzprime=1e-4,
        distE='lines', energies=(defaultEnergy,), energyWeights=None,
        polarization='horizontal', filamentBeam=False,
            uniformRayDensity=False, pitch=0, roll=0, yaw=0, **kwargs):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *name*: str

        *center*: tuple of 3 floats
            3D point in global system

        *nrays*: int

        *distx*, *disty*, *distz*, *distxprime*, *distzprime*:
            Linear (*distx*, *disty*, *distz*) and
            angular (*distxprime*, *distzprime*) source distributions.
            Accepted values: 'normal', 'flat', 'annulus' or None.
            If is None, the corresponding arrays remain with the values got at
            the instantiation of :class:`Beam`.
            'annulus' sets a uniform distribution for (x and z) or for (xprime
            and zprime) pairs. You can assign 'annulus' to only one member in
            the pair.

        *dx*, *dy*, *dz*, *dxprime*, *dzprime*: float
            Linear (*dx*, *dy*, *dz*) and
            angular (*dxprime*, *dzprime*) source sizes.
            For normal distribution is sigma or (sigma, cut_limit), for flat
            is full width or tuple (min, max), for annulus is tuple
            (rMin, rMax), otherwise is ignored.

        *distE*: 'normal', 'flat', 'lines', None

        *energies*: all in eV. (centerE, sigmaE) for *distE* = 'normal',
            (minE, maxE) for *distE* = 'flat', a sequence of E values for
            *distE* = 'lines'

        *energyWeights*: 1-D array-like
            Can be used together with *distE* = 'lines' to specify the weight
            of each line. Must be of the shape of *energies*.

        *polarization*:
            'h[orizontal]', 'v[ertical]', '+45', '-45', 'r[ight]', 'l[eft]',
            None, custom. In the latter case the polarization is given by a
            tuple of 4 components of the coherency matrix:
            (Jss, Jpp, Re(Jsp), Im(Jsp)).

        *filamentBeam*: bool
            If True the source generates coherent monochromatic
            wavefronts. Required for the wave propagation calculations.

        *uniformRayDensity*: bool
            If True, the radiation is sampled uniformly but with varying
            amplitudes, otherwise with the density proportional to intensity
            and with constant amplitudes. Required as True for wave propagation
            calculations. False is usual for ray-tracing. This parameter
            only affects normal distributions, as for flat and annulus
            distributions the density is already uniform. If you set it True,
            the size parameter (*dx* or *dz*) must be given as
            (sigma, cut_limit).

        *pitch*, *roll*, *yaw*: float
            rotation angles around x, y and z axes. Useful for canted sources.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.sources:
                bl.sources.append(self)
                self.ordinalNum = len(bl.sources)
        raycing.set_name(self, name)
#        if name in [None, 'None', '']:
#            self.name = '{0}{1}'.format(self.__class__.__name__,
#                                        self.ordinalNum)
#        else:
#            self.name = name
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        self.center = center
        self.nrays = np.int64(nrays)

        self.distx = distx
        self.dx = dx
        self.disty = disty
        self.dy = dy
        self.distz = distz
        self.dz = dz
        self.distxprime = distxprime
        self.dxprime = raycing.auto_units_angle(dxprime)
        self.distzprime = distzprime
        self.dzprime = raycing.auto_units_angle(dzprime)
        self.distE = distE
        if self.distE == 'lines':
            self.energies = np.array(energies)
        else:
            self.energies = energies
        self.energyWeights = energyWeights

        if bl is not None:
            if self.bl.flowSource != 'Qook0':
                bl.oesDict[self.uuid] = [self, 0]

        self.polarization = polarization
        self.filamentBeam = filamentBeam
        self.uniformRayDensity = uniformRayDensity
        self.pitch = raycing.auto_units_angle(pitch)
        self.roll = raycing.auto_units_angle(roll)
        self.yaw = raycing.auto_units_angle(yaw)

    center = raycing.center_property()

    def _apply_distribution(self, axis, distaxis, daxis, bo=None):
        if distaxis == 'normal':
            if self.uniformRayDensity:
                if not isinstance(daxis, (list, tuple)):
                    raise ValueError("Wrong distribution size!")
                axis[:] = np.random.uniform(-daxis[1], daxis[1], self.nrays)
                amp = np.exp(-axis**2 / daxis[0]**2 / 2) /\
                    PI2**0.5 / daxis[0] * 2 * daxis[1]
                bo.Jss *= amp
                bo.Jpp *= amp
                bo.Jsp *= amp
                amp = amp**0.5
                bo.Es *= amp
                bo.Ep *= amp
            else:
                sigma = daxis[0] if isinstance(daxis, (list, tuple)) else daxis
                try:
                    axis[:] = np.random.normal(0, sigma, self.nrays)
                except ValueError:
                    axis[:] = np.zeros(self.nrays)
        elif (distaxis == 'flat'):
            if raycing.is_sequence(daxis):
                aMin, aMax = daxis[0], daxis[1]
            else:
                if daxis <= 0:
                    return
                aMin, aMax = -daxis*0.5, daxis*0.5
            axis[:] = np.random.uniform(aMin, aMax, self.nrays)
#        else:
#            axis[:] = 0

    def _set_annulus(self, axis1, axis2, rMin, rMax, phiMin, phiMax):
        if rMax > rMin:
            A = 2. / (rMax**2 - rMin**2)
            r = np.sqrt(2*np.random.uniform(0, 1, self.nrays)/A + rMin**2)
        else:
            r = rMax
        phi = np.random.uniform(phiMin, phiMax, self.nrays)
        axis1[:] = r * np.cos(phi)
        axis2[:] = r * np.sin(phi)

    @raycing.append_to_flow_decorator
    def shine(self, toGlobal=True, withAmplitudes=False, accuBeam=None):
        u"""
        Returns the source beam. If *toGlobal* is True, the output is in
        the global system. If *withAmplitudes* is True, the resulted beam
        contains arrays Es and Ep with the *s* and *p* components of the
        electric field.


        .. Returned values: beamGlobal
        """

#        kwArgsIn = {'toGlobal': toGlobal,
#                    'withAmplitudes': withAmplitudes}

        if self.bl is not None:
            try:
                self.bl._alignE = float(self.bl.alignE)
            except ValueError:
                if self.distE in ['lines', 'normal']:
                    self.bl._alignE = self.energies[0]
                elif self.distE in ['flat']:
                    self.bl._alignE = 0.5 * (self.energies[0] +
                                             self.energies[-1])
                else:
                    self.bl._alignE = self.energies

#            if accuBeam is None:
#                kwArgsIn['accuBeam'] = accuBeam
#            else:
#                if raycing.is_valid_uuid(accuBeam):
#                    kwArgsIn['accuBeam'] = accuBeam
#                    accuBeam = self.bl.beamsDictU[accuBeam][
#                            'beamGlobal' if toGlobal else 'beamLocal']
#                else:
#                    kwArgsIn['accuBeam'] = accuBeam.parentId

        if self.uniformRayDensity:
            withAmplitudes = True
        bo = Beam(self.nrays, withAmplitudes=withAmplitudes)  # beam-out
        bo.state[:] = 1

        make_polarization(self.polarization, bo, self.nrays)
# in local coordinate system:
        self._apply_distribution(bo.y, self.disty, self.dy)

        isAnnulus = False
        if (self.distx == 'annulus') or (self.distz == 'annulus'):
            isAnnulus = True
            if raycing.is_sequence(self.dx):
                rMin, rMax = self.dx
            else:
                isAnnulus = False
            if raycing.is_sequence(self.dz):
                phiMin, phiMax = self.dz
            else:
                phiMin, phiMax = 0, PI2
        if isAnnulus:
            self._set_annulus(bo.x, bo.z, rMin, rMax, phiMin, phiMax)
        else:
            self._apply_distribution(bo.x, self.distx, self.dx, bo)
            self._apply_distribution(bo.z, self.distz, self.dz, bo)

        isAnnulus = False
        if (self.distxprime == 'annulus') or (self.distzprime == 'annulus'):
            isAnnulus = True
            if raycing.is_sequence(self.dxprime):
                rMin, rMax = self.dxprime
            else:
                isAnnulus = False
            if raycing.is_sequence(self.dzprime):
                phiMin, phiMax = self.dzprime
            else:
                phiMin, phiMax = 0, PI2
        if isAnnulus:
            self._set_annulus(bo.a, bo.c, rMin, rMax, phiMin, phiMax)
        else:
            self._apply_distribution(bo.a, self.distxprime, self.dxprime)
            self._apply_distribution(bo.c, self.distzprime, self.dzprime)

# normalize (a,b,c):
        ac = bo.a**2 + bo.c**2
        if sum(ac > 1) > 0:
            bo.b[:] = (ac + 1)**0.5
            bo.a[:] /= bo.b
            bo.c[:] /= bo.b
            bo.b[:] = 1.0 / bo.b
        else:
            bo.b[:] = (1 - ac)**0.5
        if self.distE is not None:
            if accuBeam is None:
                bo.E[:] = make_energy(
                    self.distE, self.energies, self.nrays, self.filamentBeam,
                    energyWeights=self.energyWeights)
            else:
                bo.E[:] = accuBeam.E[:]

        if self.pitch or self.roll or self.yaw:
            raycing.rotate_beam(
                bo, pitch=self.pitch, roll=self.roll, yaw=self.yaw)
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
#            self.bl.beamsDictU[self.uuid] = {'beamGlobal': bo}
#        else:
#            self.bl.beamsDictU[self.uuid] = {'beamLocal': bo}

        raycing.append_to_flow(self.shine, [bo],
                               inspect.currentframe())

#        self.bl.flowU[self.uuid] = {'method': self.shine,
#                                    'kwArgsIn': kwArgsIn}

        return bo


class GaussianBeam(object):
    r"""Implements a Gaussian beam https://en.wikipedia.org/wiki/Gaussian_beam.
    It *must* be used for an already available set of 3D points which are
    obtained by :meth:`prepare_wave` of a slit, oe or screen. See a usage
    example in ``\tests\raycing\laguerre_hermite_gaussian_beam.py``."""

    def __init__(
        self, bl=None, name='', center=(0, 0, 0), w0=0.1,
        distE='lines', energies=(defaultEnergy,), energyWeights=None,
            polarization='horizontal', pitch=0, roll=0, yaw=0, **kwargs):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *name*: str

        *center*: tuple of 3 floats
            3D point in global system

        *w0*: float or 2-sequence
            Gaussian beam waist size. If a 2-sequence, the sizes refer to
            the horizontal and the vertical axes.

        *distE*: 'normal', 'flat', 'lines', None

        *energies*: all in eV. (centerE, sigmaE) for *distE* = 'normal',
            (minE, maxE) for *distE* = 'flat', a sequence of E values for
            *distE* = 'lines'

        *energyWeights*: 1-D array-like
            Can be used together with *distE* = 'lines' to specify the weight
            of each line. Must be of the shape of *energies*.

        *polarization*:
            'h[orizontal]', 'v[ertical]', '+45', '-45', 'r[ight]', 'l[eft]',
            None, custom. In the latter case the polarization is given by a
            tuple of 4 components of the coherency matrix:
            (Jss, Jpp, Re(Jsp), Im(Jsp)).

        *pitch*, *roll*, *yaw*: float
            rotation angles around x, y and z axes. Useful for canted sources.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.sources:
                bl.sources.append(self)
                self.ordinalNum = len(bl.sources)
        raycing.set_name(self, name)
#        if name in [None, 'None', '']:
#            self.name = '{0}{1}'.format(self.__class__.__name__,
#                                        self.ordinalNum)
#        else:
#            self.name = name
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        self.center = center
        self.w0 = w0
        if raycing.is_sequence(self.w0):
            if len(self.w0) != 2:
                raise ValueError('wrong length of w0')
        self.distE = distE
        if self.distE == 'lines':
            self.energies = np.array(energies)
        else:
            self.energies = energies
        self.energyWeights = energyWeights

        if bl is not None:
            if self.bl.flowSource != 'Qook0':
                bl.oesDict[self.uuid] = [self, 0]

        self.polarization = polarization
        self.vortex = None
        self.tem = None
        self.pitch = raycing.auto_units_angle(pitch)
        self.roll = raycing.auto_units_angle(roll)
        self.yaw = raycing.auto_units_angle(yaw)

    center = raycing.center_property()

    def rayleigh_range(self, E, w0=None):
        if w0 is None:
            w0 = self.w0[0] if raycing.is_sequence(self.w0) else self.w0
        k = E / CHBAR * 1e7  # mm^-1
        return k/2 * w0**2

    def w(self, y, E=None, yR=None, w0=None):
        if w0 is None:
            w0 = self.w0[0] if raycing.is_sequence(self.w0) else self.w0
        if yR is None:
            yR = self.rayleigh_range(E, w0)
        return w0 * (1 + (y/yR)**2)**0.5

    @raycing.append_to_flow_decorator
    def shine(self, toGlobal=True, wave=None, accuBeam=None):
        u"""
        Returns the source beam. If *toGlobal* is True, the output is in
        the global system.

        .. note::
            You must run :meth:`prepare_wave` before shine() as it needs a wave
            object for which the intensities will be calculated!


        .. Returned values: beamGlobal
        """
        if self.bl is not None:
            try:
                self.bl._alignE = float(self.bl.alignE)
            except ValueError:
                if self.distE in ['lines', 'normal']:
                    self.bl._alignE = self.energies[0]
                elif self.distE in ['flat']:
                    self.bl._alignE = 0.5 * (self.energies[0] +
                                             self.energies[-1])
                else:
                    self.bl._alignE = self.energies

        try:
            mcRays = len(wave.rDiffr)
        except AttributeError:
            raise ValueError("run a `prepare_wave` before shine!")
        if self.distE is not None:
            if accuBeam is None:
                wave.E[:] = make_energy(
                    self.distE, self.energies, mcRays, filamentBeam=False,
                    energyWeights=self.energyWeights)
            else:
                wave.E[:] = accuBeam.E[:]
        make_polarization(self.polarization, wave, mcRays)

        if self.vortex is not None:
            l, p = self.vortex
            gouy = abs(l) + 2*p
        elif self.tem is not None:
            m, n = self.tem
            gouy = m + n
        else:
            gouy = 0
        k = wave.E / CHBAR * 1e7  # mm^-1
        if raycing.is_sequence(self.w0):
            amp = (2/np.pi)**0.5 * np.exp(1j*k*wave.yDiffr)
            for iw, w0 in enumerate(self.w0):
                yR = k/2 * w0**2
                invR = wave.yDiffr / (wave.yDiffr**2 + yR**2)
                psi = (gouy + 1) * np.arctan2(wave.yDiffr, yR) * 0.5
                w = self.w(wave.yDiffr, yR=yR, w0=w0)
                if iw == 0:
                    wx = w
                    rSquare = wave.xDiffr**2
                elif iw == 1:
                    wz = w
                    rSquare = wave.zDiffr**2
                amp *= w**(-0.5) * np.exp(
                    -rSquare/w**2 + 0.5j*k*rSquare*invR - 1j*psi)
        else:
            yR = k/2 * self.w0**2
            invR = wave.yDiffr / (wave.yDiffr**2 + yR**2)
            psi = (gouy + 1) * np.arctan2(wave.yDiffr, yR)
            w = self.w(wave.yDiffr, yR=yR)
            wx = wz = w
            rSquare = wave.xDiffr**2 + wave.zDiffr**2
            amp = (2/np.pi)**0.5 / w * np.exp(
                -rSquare/w**2 + 1j*k*(wave.yDiffr + 0.5*rSquare*invR) - 1j*psi)

        if self.vortex is not None:
            phi = np.arctan2(wave.zDiffr, wave.xDiffr)
            clp = (sp.special.factorial(p)*1. /
                   sp.special.factorial(abs(l)+p))**0.5
            amp *= clp * ((rSquare*2)**0.5/w)**abs(l) * np.exp(1j*l*phi)
            if p > 0:
                lg = sp.special.eval_genlaguerre(p, abs(l), 2*rSquare/w**2)
                amp *= lg
        elif self.tem is not None:
            clp = (2**(m+n)*sp.special.factorial(m) *
                   sp.special.factorial(n))**(-0.5)
            amp *= clp
            if m > 0:
                hm = sp.special.eval_hermite(m, 2**0.5*wave.xDiffr/wx)
                amp *= hm
            if n > 0:
                hn = sp.special.eval_hermite(n, 2**0.5*wave.zDiffr/wz)
                amp *= hn

        amp *= wave.dS**0.5
        wave.Es *= amp
        wave.Ep *= amp
        amp2 = np.abs(amp)**2
        wave.Jss *= amp2
        wave.Jpp *= amp2
        wave.Jsp *= amp2

        wave.a[:] = wave.xDiffr
        wave.c[:] = wave.zDiffr
        with np.errstate(divide='ignore'):
            wave.b[:] = 1/invR
        wave.b[invR == 0] = 1e20
        wave.b[:] = (wave.b**2 - wave.a**2 - wave.c**2)**0.5
# normalize (a,b,c):
        norm = (wave.a**2 + wave.b**2 + wave.c**2)**0.5
        wave.a /= norm
        wave.b /= norm
        wave.c /= norm
        bo = Beam(copyFrom=wave)
        bo.x[:] = wave.xDiffr
        bo.y[:] = wave.yDiffr
        bo.z[:] = wave.zDiffr
        bo.path = (wave.xDiffr**2 + wave.yDiffr**2 + wave.zDiffr**2)**0.5
        if self.pitch or self.roll or self.yaw:
            raycing.rotate_beam(
                bo, pitch=self.pitch, roll=self.roll, yaw=self.yaw)
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        raycing.append_to_flow(self.shine, [bo], inspect.currentframe())
        return bo


class LaguerreGaussianBeam(GaussianBeam):
    r"""Implements Laguerre-Gaussian beam
    https://en.wikipedia.org/wiki/Gaussian_beam.
    It must be used for an already available set of 3D points which are
    obtained by :meth:`prepare_wave` of a slit, oe or screen. See a usage
    example in ``\tests\raycing\laguerre_hermite_gaussian_beam.py``."""

    def __init__(self, *args, **kwargs):
        """
        *vortex*: None or tuple(l, p)
            specifies Laguerre-Gaussian beam with *l* the azimuthal index and
            *p* >= 0 the radial index.


        """
        vortex = kwargs.pop('vortex', None)
        GaussianBeam.__init__(self, *args, **kwargs)
        if raycing.is_sequence(self.w0):
            raise ValueError('w0 must be a value, not a sequence')
        self.vortex = vortex


class HermiteGaussianBeam(GaussianBeam):
    r"""Implements Hermite-Gaussian beam
    https://en.wikipedia.org/wiki/Gaussian_beam.
    It must be used for an already available set of 3D points which are
    obtained by :meth:`prepare_wave` of a slit, oe or screen. See a usage
    example in ``\tests\raycing\laguerre_hermite_gaussian_beam.py``."""

    def __init__(self, *args, **kwargs):
        """
        *TEM*: None or tuple(m, n)
            specifies Hermite-Gaussian beam of order (m, n) referring to the x
            and y directions.


        """
        tem = kwargs.pop('TEM', None)
        GaussianBeam.__init__(self, *args, **kwargs)
        self.tem = tem


class MeshSource(object):
    """Implements a point source representing a rectangular angular mesh of
    rays. Primarily, it is meant for internal usage for matching the maximum
    divergence to the optical sizes of optical elements."""

    def __init__(
        self, bl=None, name='', center=(0, 0, 0),
        minxprime=-1e-4, maxxprime=1e-4,
        minzprime=-1e-4, maxzprime=1e-4, nx=11, nz=11,
        distE='lines', energies=(defaultEnergy,), energyWeights=None,
        polarization='horizontal', withCentralRay=True,
            autoAppendToBL=False, **kwargs):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *name*: str

        *center*: tuple of 3 floats
            3D point in global system

        *minxprime*, *maxxprime*, *minzprime*, *maxzprime*: float
            limits for the ungular distributions

        *nx*, *nz*: int
            numbers of points in x and z dircetions

        *distE*: 'normal', 'flat', 'lines', None

        *energies*, all in eV: (centerE, sigmaE) for *distE* = 'normal',
            (minE, maxE) for *distE* = 'flat', a sequence of E values for
            *distE* = 'lines'

        *energyWeights*: 1-D array-like
            Can be used together with *distE* = 'lines' to specify the weight
            of each line. Must be of the shape of *energies*.

        *polarization*: 'h[orizontal]', 'v[ertical]', '+45', '-45', 'r[ight]',
            'l[eft]', None, custom. In the latter case the polarization is
            given by a tuple of 4 components of the coherency matrix:
            (Jss, Jpp, Re(Jsp), Im(Jsp)).

        *withCentralRay*: bool
            if True, the 1st ray in the beam is along the nominal beamline
            direction

        *autoAppendToBL*: bool
            if True, the source is added to the list of beamline sources.
            Otherwise the user must manually start it with :meth:`shine`.

        """
        self.bl = bl
        if autoAppendToBL:
            if bl is not None:
                if self not in bl.sources:
                    bl.sources.append(self)
                    self.ordinalNum = len(bl.sources)
        raycing.set_name(self, name)
        self.withCentralRay = withCentralRay
#        if name in [None, 'None', '']:
#            self.name = '{0}{1}'.format(self.__class__.__name__,
#                                        self.ordinalNum)
#        else:
#            self.name = name
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        self.center = center
        self.minxprime = raycing.auto_units_angle(minxprime)
        self.maxxprime = raycing.auto_units_angle(maxxprime)
        self.minzprime = raycing.auto_units_angle(minzprime)
        self.maxzprime = raycing.auto_units_angle(maxzprime)
        self.nx = nx
        self.nz = nz
        self.nrays = self.nx * self.nz + int(withCentralRay)
        self.distE = distE
        if self.distE == 'lines':
            self.energies = np.array(energies)
        else:
            self.energies = energies
        self.energyWeights = energyWeights

        if bl is not None:
            if self.bl.flowSource != 'Qook0':
                bl.oesDict[self.uuid] = [self, 0]

        self.polarization = polarization

    center = raycing.center_property()

    @raycing.append_to_flow_decorator
    def shine(self, toGlobal=True):
        u"""
        Returns the source. If *toGlobal* is True, the output is in the global
        system.


        .. Returned values: beamGlobal
        """
        if self.bl is not None:
            try:
                self.bl._alignE = float(self.bl.alignE)
            except ValueError:
                if self.distE in ['lines', 'normal']:
                    self.bl._alignE = self.energies[0]
                elif self.distE in ['flat']:
                    self.bl._alignE = 0.5 * (self.energies[0] +
                                             self.energies[-1])
                else:
                    self.bl._alignE = self.energies

        self.dxprime = (self.maxxprime-self.minxprime) / (self.nx-1)
        self.dzprime = (self.maxzprime-self.minzprime) / (self.nz-1)
        bo = Beam(self.nrays)  # beam-out
        bo.state[:] = 1
# in local coordinate system:
        xx, zz = np.meshgrid(
            np.linspace(self.minxprime, self.maxxprime, self.nx),
            np.linspace(self.minzprime, self.maxzprime, self.nz))
        zz = np.flipud(zz)
        bo.a[int(self.withCentralRay):] = xx.flatten()
        bo.c[int(self.withCentralRay):] = zz.flatten()
# normalize (a,b,c):
        bo.b[:] = (bo.a**2 + 1.0 + bo.c**2)**0.5
        bo.a[:] /= bo.b
        bo.c[:] /= bo.b
        bo.b[:] = 1.0 / bo.b
        if self.distE is not None:
            bo.E[:] = make_energy(self.distE, self.energies, self.nrays,
                                  energyWeights=self.energyWeights)
        make_polarization(self.polarization, bo, self.nrays)
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        raycing.append_to_flow(self.shine, [bo],
                               inspect.currentframe())
        return bo


class NESWSource(MeshSource):
    """Implements a point source with 4 rays: N(ord), E(ast), S(outh), W(est).
    Used internally for matching the maximum divergence to the optical sizes of
    optical elements.
    """

    @raycing.append_to_flow_decorator
    def shine(self, toGlobal=True):
        u"""
        Returns the source. If *toGlobal* is True, the output is in the global
        system.


        .. Returned values: beamGlobal
        """
        bo = Beam(4)  # beam-out
        bo.state[:] = 1
# in local coordinate system:
        bo.a[0] = 0
        bo.a[1] = self.maxxprime
        bo.a[2] = 0
        bo.a[3] = self.minxprime
        bo.c[0] = self.maxzprime
        bo.c[1] = 0
        bo.c[2] = self.minzprime
        bo.c[3] = 0
# normalize (a,b,c):
        bo.b[:] = (bo.a**2 + 1.0 + bo.c**2)**0.5
        bo.a[:] /= bo.b
        bo.c[:] /= bo.b
        bo.b[:] = 1.0 / bo.b
        bo.z[:] += 0.05

        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        raycing.append_to_flow(self.shine, [bo],
                               inspect.currentframe())
        return bo


class CollimatedMeshSource(object):
    """Implements a source representing a mesh of collimated rays. Is similar
    to :class:`MeshSource`.
    """

    def __init__(
        self, bl=None, name='', center=(0, 0, 0), dx=1., dz=1., nx=11, nz=11,
        distE='lines', energies=(defaultEnergy,), energyWeights=None,
        polarization='horizontal', withCentralRay=True,
        autoAppendToBL=False, **kwargs):
        self.bl = bl
        if autoAppendToBL:
            if bl is not None:
                if self not in bl.sources:
                    bl.sources.append(self)
                    self.ordinalNum = len(bl.sources)
        raycing.set_name(self, name)
        self.withCentralRay = withCentralRay
#        if name in [None, 'None', '']:
#            self.name = '{0}{1}'.format(self.__class__.__name__,
#                                        self.ordinalNum)
#        else:
#            self.name = name
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        self.center = center
        self.dx = dx
        self.dz = dz
        self.nx = nx
        self.nz = nz
        self.nrays = self.nx * self.nz + int(withCentralRay)
        self.distE = distE
        if self.distE == 'lines':
            self.energies = np.array(energies)
        else:
            self.energies = energies
        self.energyWeights = energyWeights

        if bl is not None:
            if self.bl.flowSource != 'Qook0':
                bl.oesDict[self.uuid] = [self, 0]

        self.polarization = polarization

    center = raycing.center_property()

    @raycing.append_to_flow_decorator
    def shine(self, toGlobal=True):
        u"""
        Returns the source beam. If *toGlobal* is True, the output is in the
        global system.


        .. Returned values: beamGlobal
        """
        if self.bl is not None:
            try:
                self.bl._alignE = float(self.bl.alignE)
            except ValueError:
                if self.distE in ['lines', 'normal']:
                    self.bl._alignE = self.energies[0]
                elif self.distE in ['flat']:
                    self.bl._alignE = 0.5 * (self.energies[0] +
                                             self.energies[-1])
                else:
                    self.bl._alignE = self.energies

        bo = Beam(self.nrays)  # beam-out
        bo.state[:] = 1
# in local coordinate system:
        xx, zz = np.meshgrid(
            np.linspace(-self.dx/2., self.dx/2., self.nx),
            np.linspace(-self.dz/2., self.dz/2., self.nz))
        zz = np.flipud(zz)
        bo.x[int(self.withCentralRay):] = xx.flatten()
        bo.z[int(self.withCentralRay):] = zz.flatten()
        if self.distE is not None:
            bo.E[:] = make_energy(self.distE, self.energies, self.nrays,
                                  energyWeights=self.energyWeights)
        make_polarization(self.polarization, bo, self.nrays)
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        raycing.append_to_flow(self.shine, [bo],
                               inspect.currentframe())
        return bo


class BeamFromFile():
    r"""Convenience class to simulate beam generation from a previously saved
    beam object. Can be used as a reproducible source or to save time on
    synchrotron source beam generation. Provides the shine() method for
    compatibility with other sources.

         *fileName*: str
            Path to the the *.npy file with previously exported `Beam`


    """

    def __init__(self, bl=None, name='', center=(0, 0, 0),
                 nrays=raycing.nrays, fileName=None, **kwargs):
        super().__init__()
        self.bl = bl
        if bl is not None:
            if self not in bl.sources:
                bl.sources.append(self)
                self.ordinalNum = len(bl.sources)
        raycing.set_name(self, name)

        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        if bl is not None:
            if self.bl.flowSource != 'Qook0':
                bl.oesDict[self.uuid] = [self, 0]

        self.center = center
        self.nrays = np.int64(nrays)
        self.fileName = fileName

    center = raycing.center_property()

    @property
    def fileName(self):
        return self._fileName

    @fileName.setter
    def fileName(self, fileName):
        self._fileName = fileName
        if fileName is not None:
            self.fbeam = Beam(copyFrom=fileName)
        else:
            print("No filename provided, using an empty beam")
            self.fbeam = Beam()


    @raycing.append_to_flow_decorator
    def shine(self):
        u"""
        Returns the beam loaded from file.

        .. Returned values: beamGlobal
        """
#        self.fbeam.parentId = self.uuid
        return self.fbeam


def shrink_source(beamLine, beams, minxprime, maxxprime, minzprime, maxzprime,
                  nx, nz):
    """Utility function that does ray tracing with a mesh source and shrinks
    its divergence until the footprint beams match to the optical surface.
    Parameters:

        *beamline*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *beams*: tuple of str

            Dictionary keys in the result of
            :func:`~xrt.backends.raycing.run.run_process()` corresponding to
            the wanted footprints.

        *minxprime, maxxprime, minzprime, maxzprime*: float

            Determines the size of the mesh source. This size can only be
            shrunk, not expanded. Therefore, you should provide it sufficiently
            big for your needs. Typically, min values are negative and max
            values are positive.

        *nx, nz*: int

            Sizes of the 2D mesh grid in *x* and *z* direction.

    Returns an instance of :class:`MeshSource` which can be used then for
    getting the divergence values.
    """
    if not isinstance(beams, tuple):
        beams = (beams,)
    storeSource = beamLine.sources[0]  # store the current 1st source
    for ibeam in beams:
        # discover which side of the footprint corresponds to which side of
        # divergence
        neswSource = NESWSource(
            beamLine, 'NESW', storeSource.center, minxprime*0.1, maxxprime*0.1,
            minzprime*0.1, maxzprime*0.1)
        beamLine.sources[0] = neswSource
        raycing_output = rr.run_process(beamLine, shineOnly1stSource=True)
        beam = raycing_output[ibeam]
        badNum = beam.state != 1
        if badNum.sum() > 0:
            print("cannot shrink the source!")
            raise
        sideDict = {'left': np.argmin(beam.x), 'right': np.argmax(beam.x),
                    'bottom': np.argmin(beam.y), 'top': np.argmax(beam.y)}
        checkSides = set(i for key, i in sideDict.iteritems())
        if len(checkSides) != 4:
            print("cannot shrink the source!")
            raise
        sideList = ['', '', '', '']
        for k, v in sideDict.iteritems():
            sideList[v] = k
# end of discover which side of the footprint ...
        meshSource = MeshSource(
            beamLine, 'mesh', storeSource.center, minxprime, maxxprime,
            minzprime, maxzprime, nx, nz)
        beamLine.sources[0] = meshSource
        raycing_output = rr.run_process(beamLine, shineOnly1stSource=True)
        beam = raycing_output[ibeam]
        rectState = beam.state[1:].reshape((meshSource.nz, meshSource.nx))
#        badNum = (rectState < 0) | (rectState > 1)
        badNum = rectState != 1
        nxLeft, nxRight, nzBottom, nzTop = 0, 0, 0, 0
        while badNum.sum() > 0:
            badNumRow = badNum.sum(axis=1)
            badNumCol = badNum.sum(axis=0)
            badNumRowMax = 2*badNumRow.max() - badNum.shape[1]
            badNumColMax = 2*badNumCol.max() - badNum.shape[0]
            if badNumRowMax >= badNumColMax:
                izDel, = np.where(badNumRow == badNumRow.max())
                izDel = max(izDel)
                if izDel < meshSource.nz/2:
                    nzTop += 1
                else:
                    nzBottom += 1
                badNum = np.delete(badNum, izDel, axis=0)
            else:
                ixDel, = np.where(badNumCol == badNumCol.max())
                ixDel = max(ixDel)
                if ixDel < meshSource.nx/2:
                    nxLeft += 1
                else:
                    nxRight += 1
                badNum = np.delete(badNum, ixDel, axis=1)
        if nxLeft > 1:
            nxLeft += 1
        if nxRight > 1:
            nxRight += 1
        if nzBottom > 1:
            nzBottom += 1
        if nzTop > 1:
            nzTop += 1
        cutDict = {
            'left': nxLeft, 'right': nxRight, 'bottom': nzBottom, 'top': nzTop}
        maxzprime -= cutDict[sideList[0]] * meshSource.dzprime
        maxxprime -= cutDict[sideList[1]] * meshSource.dxprime
        minzprime += cutDict[sideList[2]] * meshSource.dzprime
        minxprime += cutDict[sideList[3]] * meshSource.dxprime
        meshSource.maxzprime = maxzprime
        meshSource.maxxprime = maxxprime
        meshSource.minzprime = minzprime
        meshSource.minxprime = minxprime
    beamLine.sources[0] = storeSource  # restore the 1st source
    beamLine.alarms = []
    return meshSource
