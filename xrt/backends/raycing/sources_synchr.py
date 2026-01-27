# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "12 Aug 2021"
import sys
import numpy as np
from scipy import special
from scipy.interpolate import interp1d, UnivariateSpline
import inspect

from .. import raycing
from .sources_beams import Beam, allArguments
from .physconsts import E0, C, M0, EV2ERG, K2B, SIE0, EMC,\
    SIM0, FINE_STR, PI, PI2, SQ3, E2W, E2WC, CHeVcm

from .sources_sybase import SourceBase, IntegratedSource

# _DEBUG replaced with raycing._VERBOSITY_


class BendingMagnet(SourceBase):
    u"""
    Bending magnet source. The computation is reasonably fast and thus a GPU
    is not required and is not implemented.
    """

    def __init__(self, *args, **kwargs):
        u"""
        *B0*: float
            Magnetic field (T). Alternatively, specify *rho*.

        *rho*: float
            Curvature radius (m). Alternatively, specify *B0*.


        """
        B0 = kwargs.pop('B0', 1.)
        rho = kwargs.pop('rho', None)
        super(BendingMagnet, self).__init__(*args, **kwargs)

        if isinstance(self, Wiggler):
            self.B = K2B * self.K / self.L0
            self.ro = M0 * C**2 * self.gamma / self.B / E0 / 1e6
            self.X0 = 0.5 * self.K * self.L0 / self.gamma / PI
            self.isMPW = True
            self._xPrimeMaxAutoReduce = True
        else:
            self.Np = 0.5
            self.B = B0
            self.ro = rho
            if self.ro:
                if not self.B:
                    self.B = M0 * C**2 * self.gamma / self.ro / E0 / 1e6
            elif self.B:
                self.ro = M0 * C**2 * self.gamma / self.B / E0 / 1e6
            self.isMPW = False

    @property
    def B0(self):
        return self.B

    @B0.setter
    def B0(self, B):
        self.B = float(B)
        self.ro = M0 * C**2 * self.gamma / B / E0 / 1e6
        if hasattr(self, 'L0'):
            self._K = B * self.L0 / K2B  # Only for the Wiggler
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def rho(self):
        return self.ro

    @rho.setter
    def rho(self, rho):
        self.ro = rho
        self.B = M0 * C**2 * self.gamma / rho / E0 / 1e6
        if hasattr(self, 'L0'):
            self._K = self.B * self.L0 / K2B
        self.needReset = True
        # Need to recalculate the integration parameters

    def prefix_save_name(self):
        return '3-BM-xrt'

    def build_I_map(self, dde, ddtheta, ddpsi, harmonic=None, dg=None):
        if self.needReset:
            self.reset()
        np.seterr(invalid='ignore')
        np.seterr(divide='ignore')
        gamma = self.gamma
        if self.eEspread > 0:
            if np.array(dde).shape:
                if dde.shape[0] > 1:
                    gamma += np.random.normal(0, gamma*self.eEspread,
                                              dde.shape)
            gamma2 = gamma**2
        else:
            gamma2 = self.gamma2

        w_cr = 1.5 * gamma2 * self.B * SIE0 / SIM0
        if self.isMPW:
            w_cr *= np.sin(np.arccos(ddtheta * gamma / self.K))
        w_cr = np.where(np.isfinite(w_cr), w_cr, 0.)

        gammapsi = gamma * ddpsi
        gamma2psi2p1 = gammapsi**2 + 1
        eta = 0.5 * dde * E2W / w_cr * gamma2psi2p1**1.5

        ampSP = -0.5j * SQ3 / PI * gamma * dde * E2W / w_cr * gamma2psi2p1
        ampS = ampSP * special.kv(2./3., eta)
        ampP = 1j * gammapsi * ampSP * special.kv(1./3., eta) /\
            np.sqrt(gamma2psi2p1)

        ampS = np.where(np.isfinite(ampS), ampS, 0.)
        ampP = np.where(np.isfinite(ampP), ampP, 0.)

        bwFact = 0.001 if self.distE == 'BW' else 1./dde
        Amp2Flux = FINE_STR * bwFact * self.eI / SIE0 * 2 * self.Np

        np.seterr(invalid='warn')
        np.seterr(divide='warn')

        return (Amp2Flux * (np.abs(ampS)**2 + np.abs(ampP)**2),
                np.sqrt(Amp2Flux) * ampS,
                np.sqrt(Amp2Flux) * ampP)

    @raycing.append_to_flow_decorator
    def shine(self, toGlobal=True, withAmplitudes=True, fixedEnergy=False,
              accuBeam=None):
        u"""
        Returns the source beam. If *toGlobal* is True, the output is in
        the global system. If *withAmplitudes* is True, the resulted beam
        contains arrays Es and Ep with the *s* and *p* components of the
        electric field.


        .. Returned values: beamGlobal
        """
        if self.needReset:
            self.reset()

#        kwArgsIn = {'toGlobal': toGlobal,
#                    'withAmplitudes': withAmplitudes,
#                    'fixedEnergy': fixedEnergy}

        if self.bl is not None:
            try:
                self.bl._alignE = float(self.bl.alignE)
            except ValueError:
                self.bl._alignE = 0.5 * (self.eMin + self.eMax)

#            if accuBeam is None:
#                kwArgsIn['accuBeam'] = accuBeam
#            else:
#                if raycing.is_valid_uuid(accuBeam):
#                    kwArgsIn['accuBeam'] = accuBeam
#                    accuBeam = self.bl.beamsDictU[accuBeam][
#                            'beamGlobal' if toGlobal else 'beamLocal']
#                else:
#                    kwArgsIn['accuBeam'] = accuBeam.parentId


        if self.uniformRayDensity:  # Force withAmplitudes=True
            withAmplitudes = True

        bo = None
        length = 0
        seeded = np.int64(0)
        seededI = 0.
        np.seterr(invalid='warn')
        np.seterr(divide='warn')
        mcRays = np.int64(self.nrays * 1.2) if not self.uniformRayDensity else\
            self.nrays
        if self.filamentBeam:
            if accuBeam is None:
                rE = np.random.random_sample() *\
                    float(self.E_max - self.E_min) + self.E_min
                if self.isMPW:
                    sigma_r2 = 2 * (CHeVcm/rE*10*self.L0*self.Np) / PI2**2
                    sourceSIGMAx = self.dx
                    sourceSIGMAz = self.dz
                    rTheta0 = np.random.random_sample() *\
                        (self.Theta_max - self.Theta_min) + self.Theta_min
                    ryNp = 0.5 * self.L0 *\
                        (np.arccos(rTheta0 * self.gamma / self.K) / PI) +\
                        0.5 * self.L0 *\
                        np.random.random_integers(0, int(2*self.Np - 1))
                    rY = ryNp - 0.5*self.L0*self.Np
                    if (ryNp - 0.25*self.L0 <= 0):
                        rY += self.L0*self.Np
                    rX = self.X0 * np.sin(PI2 * rY / self.L0) +\
                        sourceSIGMAx * np.random.standard_normal()
                    rY -= 0.25 * self.L0
                    rZ = sourceSIGMAz * np.random.standard_normal()
                else:
                    rZ = self.dz * np.random.standard_normal()
                    rTheta0 = np.random.random_sample() *\
                        (self.Theta_max - self.Theta_min) + self.Theta_min
                    R1 = self.dx * np.random.standard_normal() +\
                        self.ro * 1000.
                    rX = -R1 * np.cos(rTheta0) + self.ro*1000.
                    rY = R1 * np.sin(rTheta0)
                dtheta = self.dxprime * np.random.standard_normal()
                dpsi = self.dzprime * np.random.standard_normal()
            else:
                rE = accuBeam.E[0]
                rX = accuBeam.x[0]
                rY = accuBeam.y[0]
                rZ = accuBeam.z[0]
                dtheta = accuBeam.filamentDtheta
                dpsi = accuBeam.filamentDpsi
        if fixedEnergy:
            rE = fixedEnergy

        nrep = 0
        rep_condition = True
#        while length < self.nrays:
        while rep_condition:
            """Preparing 4 columns of random numbers
            0: Energy
            1: Theta / horizontal
            2: Psi / vertical
            3: Monte-Carlo discriminator"""
            rnd_r = np.random.rand(mcRays, 4)
            seeded += mcRays
            if self.filamentBeam:
                rThetaMin = np.max((self.Theta_min, rTheta0 - 1. / self.gamma))
                rThetaMax = np.min((self.Theta_max, rTheta0 + 1. / self.gamma))
                rTheta = (rnd_r[:, 1]) * (rThetaMax - rThetaMin) +\
                    rThetaMin
                rE *= np.ones(mcRays)
            else:
                rE = rnd_r[:, 0] * float(self.E_max - self.E_min) +\
                    self.E_min
                rTheta = (rnd_r[:, 1]) * (self.Theta_max - self.Theta_min) +\
                    self.Theta_min
            rPsi = rnd_r[:, 2] * (self.Psi_max - self.Psi_min) +\
                self.Psi_min
            Intensity, mJss, mJpp = self.build_I_map(rE, rTheta, rPsi)

            if self.uniformRayDensity:
                seededI += self.nrays * self.xzE
            else:
                seededI += Intensity.sum() * self.xzE

            tmp_max = np.max(Intensity)
            if tmp_max > self.Imax:
                self.Imax = tmp_max
                self.fluxConst = self.Imax * self.xzE
                if raycing._VERBOSITY_ > 10:
                    imax = np.argmax(Intensity)
                    print(self.Imax, imax, rE[imax], rTheta[imax], rPsi[imax])
            if self.uniformRayDensity:
                I_pass = slice(None)
                npassed = mcRays
            else:
                I_pass =\
                    np.where(self.Imax * rnd_r[:, 3] < Intensity)[0]
                npassed = len(I_pass)
            if npassed == 0:
                print('No good rays in this seed!'
                      ' {0} of {1} rays in total so far...'.format(
                          length, self.nrays))
                continue

            bot = Beam(npassed, withAmplitudes=withAmplitudes)
            bot.state[:] = 1  # good

            bot.E[:] = rE[I_pass]

            Theta0 = rTheta[I_pass]
            Psi0 = rPsi[I_pass]

            if not self.filamentBeam:
                if self.dxprime > 0:
                    dtheta = np.random.normal(0, self.dxprime, npassed)
                else:
                    dtheta = 0
                if not self.isMPW:
                    dtheta += np.random.normal(0, 1/self.gamma, npassed)

                if self.dzprime > 0:
                    dpsi = np.random.normal(0, self.dzprime, npassed)
                else:
                    dpsi = 0

            bot.a[:] = np.tan(Theta0 + dtheta)
            bot.c[:] = np.tan(Psi0 + dpsi)

            intensS = (mJss[I_pass] * np.conj(mJss[I_pass])).real
            intensP = (mJpp[I_pass] * np.conj(mJpp[I_pass])).real
            if self.uniformRayDensity:
                sSP = 1.
            else:
                sSP = intensS + intensP
            # as by Walker and by Ellaume; SPECTRA's value is twice smaller:

            if self.isMPW:
                sigma_r2 = 2 * (CHeVcm/bot.E*10 * self.L0*self.Np) / PI2**2
                bot.sourceSIGMAx = np.sqrt(self.dx**2 + sigma_r2)
                bot.sourceSIGMAz = np.sqrt(self.dz**2 + sigma_r2)
                if self.filamentBeam:
                    bot.z[:] = rZ
                    bot.x[:] = rX
                    bot.y[:] = rY
                else:
                    bot.y[:] = ((np.arccos(Theta0*self.gamma/self.K) / PI) +
                                np.random.randint(
                                    -int(self.Np), int(self.Np), npassed) -
                                0.5) * 0.5 * self.L0
                    bot.x[:] = self.X0 * np.sin(PI2 * bot.y / self.L0) +\
                        np.random.normal(0., bot.sourceSIGMAx, npassed)
                    bot.z[:] = np.random.normal(0., bot.sourceSIGMAz, npassed)
                bot.Jsp[:] = np.zeros(npassed)
            else:
                if self.filamentBeam:
                    bot.z[:] = rZ
                    bot.x[:] = rX
                    bot.y[:] = rY
                else:
                    if self.dz > 0:
                        bot.z[:] = np.random.normal(0., self.dz, npassed)
                    if self.dx > 0:
                        R1 = np.random.normal(self.ro*1e3, self.dx, npassed)
                    else:
                        R1 = self.ro * 1e3
                    bot.x[:] = -R1 * np.cos(Theta0) + self.ro*1000.
                    bot.y[:] = R1 * np.sin(Theta0)

                bot.Jsp[:] = np.array(
                    np.where(sSP,
                             mJss[I_pass] * np.conj(mJpp[I_pass]) / sSP,
                             sSP), dtype=complex)

            bot.Jss[:] = np.where(sSP, intensS / sSP, sSP)
            bot.Jpp[:] = np.where(sSP, intensP / sSP, sSP)

            if withAmplitudes:
                bot.Es[:] = mJss[I_pass]
                bot.Ep[:] = mJpp[I_pass]

            if bo is None:
                bo = bot
            else:
                bo.concatenate(bot)
            length = len(bo.a)
            if raycing._VERBOSITY_ > 0:
                print("{0} rays of {1}".format(length, self.nrays))
                try:
                    if self.bl is not None:
                        if self.bl.flowSource == 'Qook' and\
                                self.bl.statusSignal is not None:
                            ptg = (self.bl.statusSignal[1] +
                                   float(length) / float(self.nrays)) /\
                                      self.bl.statusSignal[2]
                            self.bl.statusSignal[0].emit(
                                (ptg, self.bl.statusSignal[3]))
                except Exception:
                    pass
            if self.filamentBeam:
                nrep += 1
                rep_condition = nrep < self.nrepmax
            else:
                rep_condition = length < self.nrays
            if self.uniformRayDensity:
                rep_condition = False
            if raycing._VERBOSITY_ > 0:
                sys.stdout.flush()

        if length >= self.nrays:
            bo.accepted = length * self.fluxConst
            bo.acceptedE = bo.E.sum() * self.fluxConst * SIE0
            bo.seeded = seeded
            bo.seededI = seededI
        if length > self.nrays and not self.filamentBeam:
            bo.filter_by_index(slice(0, np.int64(self.nrays)))
        if self.filamentBeam:
            bo.filamentDtheta = dtheta
            bo.filamentDpsi = dpsi
        norm = np.sqrt(bo.a**2 + 1.0 + bo.c**2)
        bo.a /= norm
        bo.b /= norm
        bo.c /= norm

        bo.basis = np.identity(3)
        bo.parentId = self.uuid

        if self.pitch or self.yaw:
            raycing.rotate_beam(bo, pitch=self.pitch, yaw=self.yaw)
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


class Wiggler(BendingMagnet):
    u"""
    Wiggler source. The computation is reasonably fast and thus a GPU
    is not required and is not implemented.
    """

    hiddenParams = getattr(BendingMagnet, 'hiddenParams', set()) | {'B0', 'rho'}

    def __init__(self, *args, **kwargs):
        u"""Parameters are the same as in BendingMagnet except *B0* and *rho*
        which are not required and additionally:

        *K*: float
            Deflection parameter

        *period*: float
            period length in mm.

        *n*: int
            Number of periods.


        """
        self._K = kwargs.pop('K', 8.446)
        self._Ky = self._K
        self.L0 = kwargs.pop('period', 50)
        self.Np = kwargs.pop('n', 40)
        name = kwargs.pop('name', 'wiggler')
        kwargs['name'] = name
        super(Wiggler, self).__init__(*args, **kwargs)
        self.needReset = True

    @property
    def period(self):
        return self.L0

    @period.setter
    def period(self, period):
        self.L0 = float(period)
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def n(self):
        return self.Np

    @n.setter
    def n(self, n):
        self.Np = float(n)
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, K):
        self._K = float(K)
        self._Ky = float(K)
        self._B = K2B * K / self.L0
        self.ro = M0 * C**2 * self.gamma / self.B / E0 / 1e6
        self.X0 = 0.5 * K * self.L0 / self.gamma / PI
        self.needReset = True
        # Need to recalculate the integration parameters

    def prefix_save_name(self):
        return '2-Wiggler-xrt'

    def power_vs_K(self, energy, theta, psi, Ks):
        u"""
        Calculates *power curve* -- total power in W at given K values (*Ks*).
        The power is calculated through the aperture defined by *theta* and
        *psi* opening angles within the *energy* range.

        Returns a 1D array corresponding to *Ks*.
        """
        try:
            dtheta, dpsi, dE = \
                theta[1] - theta[0], psi[1] - psi[0], energy[1] - energy[0]
        except TypeError:
            dtheta, dpsi, dE = 1, 1, 1
        tmpK = self.K
        powers = []
        for iK, K in enumerate(Ks):
            if raycing._VERBOSITY_ > 10:
                print("K={0}, {1} of {2}".format(K, iK+1, len(Ks)))
            self.K = K
            self.reset()
            I0 = self.intensities_on_mesh(energy, theta, psi)[0]
            if self.distE == 'BW':
                I0 *= 1e3
            else:  # 'eV'
                I0 *= energy[:, np.newaxis, np.newaxis]
            power = I0.sum() * dtheta * dpsi * dE * EV2ERG * 1e-7  # [W]
            powers.append(power)
        self.K = tmpK
        return np.array(powers)


class SourceFromField(IntegratedSource):
    """Dedicated class for the sources based on a custom field table."""

    def __init__(self, *args, **kwargs):
        """
        *customField*: float or str or tuple(fileName, kwargs) or numpy array.
            If float, adds a constant longitudinal field.
            If str or tuple, expects table of field samples given as an Excel
            file or as text file. If given as a tuple or list, the 2nd member
            is a key word dictionary for reading Excel by
            :meth:`pandas.read_excel()` or reading text file by
            :meth:`numpy.loadtxt()`, e.g. ``dict(skiprows=4)`` for skipping the
            file header. The file must contain the columns with longitudinal
            coordinate in mm, {B_hor,} B_ver {, B_long}, all in T. The field
            can be provided as a numpy array with the same structure as the
            table from file.

        """
        customField = kwargs.pop('customField', None)
        super(SourceFromField, self).__init__(*args, **kwargs)

        self.spl_kw = {'kind': 'cubic',
                       'bounds_error': False,
                       'fill_value': 'extrapolate'}
        self.periodicTest = False
        self._customField = customField
        self.deviceLength = 0
        if customField is not None:
            if isinstance(customField, (tuple, list)):
                fname = customField[0]
                readkw = customField[1]
            elif isinstance(customField, np.ndarray):
                self.customFieldData = customField
                fname = None
            else:
                fname = customField
                readkw = {}
            if fname:
                self.customFieldData = self.read_custom_field(fname, readkw)
        else:  # Test with periodic field
            self.Kx = 0.
            self.Ky = 4.4  # 17.274 #1.7
            self.phase = 0
            self.L0 = 53.96  # 100 #10.
            self.Np = 41  # 70 #30
            self.quadm = 50
            self.gIntervals = 2  # *self.Np
            self.wtGrid = np.linspace(-self.L0*self.Np*0.5,
                                      self.L0*self.Np*0.5,
                                      1000*self.Np)  # 1000 points per period
            self.customFieldData = None
            Bx, By, Bz = self._magnetic_field_periodic(self.wtGrid)
            self.customFieldData = np.vstack((self.wtGrid, Bx, By, Bz)).T

        self.needReset = True

    @property
    def customField(self):
        return self._customField

    @customField.setter
    def customField(self, customField):
        self._customField = customField
        if customField is not None:
            if isinstance(customField, (tuple, list)):
                fname = customField[0]
                kwargs = customField[1]
            elif isinstance(customField, np.ndarray):
                self.customFieldData = customField
            else:
                fname = customField
                kwargs = {}
            if fname:
                self.customFieldData = self.read_custom_field(fname, kwargs)
        else:
            self.customFieldData = None
        self.needReset = True

    def prefix_save_name(self):
        return '5-SFF-xrt'

    def read_custom_field(self, fname, kwargs={}):
        print(f'reading custom field from {fname}')
        if fname.endswith('.xls') or fname.endswith('.xlsx'):
            from pandas import read_excel
            data = read_excel(fname, **kwargs).values
        else:
            data = np.loadtxt(fname, **kwargs)
        z = data[:, 0]
        B = np.abs(data[:, 1:]).max(axis=1)
        self.deviceLength = self._fwhm(z, B)
        return data

    def _fwhm(self, z, a):
        dz = z[1] - z[0]
        args = np.argwhere(a >= a.max()*0.5)
        return z[np.max(args)] - z[np.min(args)] + dz

    def get_sigma_r02(self, E):  # linear size
        """Squared sigma_{r0} as by Walker and by Ellaume and
        Tanaka and Kitamura J. Synchrotron Rad. 16 (2009) 380–386 (see the
        text after Eq(23))"""
        return 2 * CHeVcm/E*10 * self.deviceLength / PI2**2

    def get_sigmaP_r02(self, E):  # angular size
        """Squared sigmaP_{r0}"""
        return CHeVcm/E*10 / (2 * self.deviceLength)

    def get_SIGMA(self, E, onlyOddHarmonics=True):
        sigma_r2 = self.get_sigma_r02(E)
        return ((self.dx**2 + sigma_r2)**0.5,
                (self.dz**2 + sigma_r2)**0.5)

    def _magnetic_field(self, grid=None):
        dataz = self.customFieldData[:, 0]
        if grid is None:
            lenmm = np.abs(dataz[-1] - dataz[0])
            self.wtGrid = np.linspace(dataz[0], dataz[-1], int(lenmm*10))
            self.BGrid = np.linspace(dataz[0], dataz[-1], 2*len(self.wtGrid)-1)
            z = self.BGrid  # 'z' in mm
        else:
            z = grid

        dataShape = self.customFieldData.shape
        if dataShape[1] == 2:
#            By = UnivariateSpline(dataz, self.customFieldData[:, 1], s=1e-6)(z)
            By = interp1d(dataz, self.customFieldData[:, 1], **self.spl_kw)(z)
            Bx = np.zeros_like(By)
            Bz = np.zeros_like(By)
        elif dataShape[1] == 3:
#            Bx = UnivariateSpline(dataz, self.customFieldData[:, 1], s=1e-6)(z)
#            By = UnivariateSpline(dataz, self.customFieldData[:, 2], s=1e-6)(z)
            Bx = interp1d(dataz, self.customFieldData[:, 1], **self.spl_kw)(z)
            By = interp1d(dataz, self.customFieldData[:, 2], **self.spl_kw)(z)
            Bz = np.zeros_like(By)
        elif dataShape[1] == 4:
#            Bx = UnivariateSpline(dataz, self.customFieldData[:, 1], s=1e-6)(z)
#            By = UnivariateSpline(dataz, self.customFieldData[:, 2], s=1e-6)(z)
#            Bz = UnivariateSpline(dataz, self.customFieldData[:, 3], s=1e-6)(z)
            Bx = interp1d(dataz, self.customFieldData[:, 1], **self.spl_kw)(z)
            By = interp1d(dataz, self.customFieldData[:, 2], **self.spl_kw)(z)
            Bz = interp1d(dataz, self.customFieldData[:, 3], **self.spl_kw)(z)
        else:
            print("Unknown file structure.")
            raise
        return Bx, By, Bz

    def _magnetic_field_periodic(self, grid=None):
        if grid is None:
            dataz = self.customFieldData[:, 0]
            lenmm = np.abs(dataz[-1] - dataz[0])
            self.wtGrid = np.linspace(dataz[0], dataz[-1], int(lenmm*10))
            self.BGrid = np.linspace(dataz[0], dataz[-1], 2*len(self.wtGrid)-1)
            z = self.BGrid
        else:
            z = grid
        self.B0x = K2B * self.Kx / self.L0
        self.B0y = K2B * self.Ky / self.L0
        self.B0z = 0
        z = 2*np.pi*z/self.L0
        Bx = self.B0x*np.sin(z + self.phase)
        By = self.B0y*np.sin(z)
        Bz = self.B0z*np.ones_like(Bx)
        return Bx, By, Bz

    def _sp(self, dim, emcg, w, gamma, ddphi, ddpsi, Bx, By, Bz,
            betax, betay, betam, trajx, trajy, trajz, R0=None):
        lengamma = 1 if len(np.array(gamma).shape) == 0 else len(gamma)
        gS = gamma
        if dim == 0:
            wS = w
            ddphiS = ddphi
            ddpsiS = ddpsi
        elif dim == 1:
            wS = w[:, np.newaxis]
            ddphiS = ddphi[:, np.newaxis]
            ddpsiS = ddpsi[:, np.newaxis]
            if lengamma > 1:
                gS = gamma[:, np.newaxis]
#        elif dim == 3:
#            wS = w[:, :, :, np.newaxis]
#            ddphiS = ddphi[:, :, :, np.newaxis]
#            ddpsiS = ddpsi[:, :, :, np.newaxis]
#            if lengamma > 1:
#                gS = gamma[:, :, :, np.newaxis]

        dirx = ddphiS
        diry = ddpsiS
        dirz = np.sqrt(1. - ddphiS**2 - ddpsiS**2)
#        dirz = 1. - 0.5*(ddphiS**2 + ddpsiS**2)
        revgamma2 = 1./gS**2

        if self.filamentBeam:
            wc = wS * E2WC / betam
            betax_ = betax
            betay_ = betay
            trajx_ = trajx
            trajy_ = trajy
            trajz_ = trajz
        else:
            wc = wS*E2WC/(1. + (betam*EMC**2 - 0.5)*revgamma2)
            betax_ = emcg*betax
            betay_ = emcg*betay
            trajx_ = emcg*trajx
            trajy_ = emcg*trajy
            trajz_ = self.tg*(1.-0.5*revgamma2) + EMC**2*revgamma2*trajz
        rloc = np.array([trajx_, trajy_, trajz_])

        if R0 is not None:
            if len(R0.shape) < 2:
                R0 = np.expand_dims(R0, 1)
            dr = R0 - rloc
            dist = np.linalg.norm(dr, axis=0)
            sinr0z = np.sin(wc*R0[2, :])
            cosr0z = np.cos(wc*R0[2, :])
            rdrz = 1./dr[2, :]
            drs = (dr[0, :]**2+dr[1, :]**2)*rdrz

            LRS = 0.5*drs - 0.125*drs**2*rdrz + 0.0625*drs**3*rdrz**2
            sinzloc = np.sin(wc * (self.tg - trajz_))
            coszloc = np.cos(wc * (self.tg - trajz_))

            sindrs = np.sin(wc * LRS)
            cosdrs = np.cos(wc * LRS)

            eucosx = (-sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -
                      cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs)
            eucosy = (-sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +
                      cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs)
        else:
            phz = wc*(self.tg - dirz*trajz_)
            phxy = wc*(dirx*trajx_ + diry*trajy_)
            sinphz, cosphz = np.sin(phz), np.cos(phz)
            sinphxy, cosphxy = np.sin(phxy), np.cos(phxy)
            eucosx = sinphz*cosphxy - cosphz*sinphxy
            eucosy = cosphz*cosphxy + sinphz*sinphxy

        eucos = eucosx + 1j*eucosy

        if R0 is not None:
            direction = dr/dist
            dirx = direction[0, :]
            diry = direction[1, :]
            dirz = direction[2, :]

        smTerm = 1./gS**2 + betax_**2 + betay_**2
        betaz = 1 - 0.5*smTerm - 0.125*smTerm**2 - 0.0625*smTerm**3

        betaPx = betay_*Bz - betaz*By
        betaPy = -betax_*Bz + betaz*Bx
        betaPz = betax_*By - betay_*Bx

        rkrel = 1./(1. - dirx*betax_ - diry*betay_ - dirz*betaz)

        eucos *= self.ag * rkrel**2
        bnx = dirx - betax_
        bny = diry - betay_
        bnz = dirz - betaz

        dirDotBetaP = dirx*betaPx + diry*betaPy + dirz*betaPz
        dirDotDmB = dirx*bnx + diry*bny + dirz*bnz

        Bsr = np.sum(eucos*emcg*(bnx*dirDotBetaP - betaPx*dirDotDmB), axis=dim)
        Bpr = np.sum(eucos*emcg*(bny*dirDotBetaP - betaPy*dirDotDmB), axis=dim)

        return Bsr, Bpr

#    @profile
    def _sp_sum(self, emcg, w, gamma, ddphi, ddpsi, Bx, By, Bz,
                betax, betay, betam, trajx, trajy, trajz, R0=None):

        Bsr = np.complex128(0)
        Bpr = np.complex128(0)

        gamma_ = gamma[0] if self.filamentBeam else gamma
        dirx = ddphi
        diry = ddpsi
#        dirz = 1. - 0.5*(ddphi**2 + ddpsi**2)
        dirz = np.sqrt(1. - ddphi**2 - ddpsi**2)
        revgamma2 = 1./gamma_**2

        wc = w * E2WC / (1. + (betam*EMC**2 - 0.5)*revgamma2) if\
            self.filamentBeam else w * E2WC / betam

        if R0 is not None:
            sinr0z, cosr0z = np.sin(wc*R0[2, :]), np.cos(wc*R0[2, :])

        for i in range(len(self.tg)):
            if self.filamentBeam:
                betax_ = betax[i]
                betay_ = betay[i]
                trajx_ = trajx[i]
                trajy_ = trajy[i]
                trajz_ = trajz[i]
            else:
                betax_ = emcg*betax[i]
                betay_ = emcg*betay[i]
                trajx_ = emcg*trajx[i]
                trajy_ = emcg*trajy[i]
                trajz_ = self.tg[i]*(1.-0.5*revgamma2) +\
                    EMC**2*revgamma2*trajz[i]

            if R0 is not None:
                rloc = np.array([trajx_, trajy_, trajz_])
                if len(rloc.shape) < 2:
                    rloc = np.expand_dims(rloc, 1)
                dr = R0 - rloc
                dist = np.linalg.norm(dr, axis=0)
                rdrz = 1./dr[2, :]
                drs = (dr[0, :]**2+dr[1, :]**2)*rdrz
                LRS = 0.5*drs - 0.125*drs**2*rdrz + 0.0625*drs**3*rdrz**2
                sinzloc = np.sin(wc * (self.tg[i] - trajz_))
                coszloc = np.cos(wc * (self.tg[i] - trajz_))
                sindrs = np.sin(wc * LRS)
                cosdrs = np.cos(wc * LRS)
                eucosx = (-sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -
                          cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs)
                eucosy = (-sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +
                          cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs)
            else:
                phz = wc*(self.tg[i] - dirz*trajz_)
                phxy = wc*(dirx*trajx_ + diry*trajy_)
                sinphz, cosphz = np.sin(phz), np.cos(phz)
                sinphxy, cosphxy = np.sin(phxy), np.cos(phxy)
                eucosx = sinphz*cosphxy - cosphz*sinphxy
                eucosy = cosphz*cosphxy + sinphz*sinphxy

            eucos = eucosx + 1j*eucosy

            if R0 is not None:
                direction = dr/dist
                dirx = direction[0, :]
                diry = direction[1, :]
                dirz = direction[2, :]

            smTerm = revgamma2 + betax_**2 + betay_**2
            betaz = 1. - 0.5*smTerm - 0.125*smTerm**2 - 0.0625*smTerm**3

            betaPx = betay_*Bz[i] - betaz*By[i]
            betaPy = -betax_*Bz[i] + betaz*Bx[i]
            betaPz = betax_*By[i] - betay_*Bx[i]
            rkrel = 1./(1. - dirx*betax_ - diry*betay_ - dirz*betaz)
            eucos *= self.ag[i] * rkrel**2

            bnx = dirx - betax_
            bny = diry - betay_
            bnz = dirz - betaz

            dirDotBetaP = dirx*betaPx + diry*betaPy + dirz*betaPz
            dirDotDmB = dirx*bnx + diry*bny + dirz*bnz

            Bsr += eucos*(bnx*dirDotBetaP - betaPx*dirDotDmB)
            Bpr += eucos*(bny*dirDotBetaP - betaPy*dirDotDmB)
        return Bsr*emcg, Bpr*emcg

    def build_I_map(self, w, ddtheta, ddpsi, dh=None, dg=None):
        if self.needReset:
            self.reset()
        useCL = False
        if isinstance(w, np.ndarray):
            if w.shape[0] > 1:
                useCL = True
        if (self.cl_ctx is None) or not useCL:
            return \
                self._build_I_map_custom_field_conv(w, ddtheta, ddpsi, dh, dg)
        else:
            return self._build_I_map_custom_field_CL(w, ddtheta, ddpsi, dh, dg)

    def _build_integration_grid(self):
        quad_rule = np.polynomial.legendre.leggauss if self._useGauLeg else\
            self._clenshaw_curtis
        tg_n, ag_n = quad_rule(self.quadm)

        if isinstance(self.customFieldData, (float, int)) or \
                self.customFieldData is None:  # TODO: this is not 100% correct
            dataz = [-0.5*self.L0*self.Np, 0.5*self.L0*self.Np]
        else:
            dataz = self.customFieldData[:, 0]
        dstep = (dataz[-1] - dataz[0]) / float(self.gIntervals)
        dI = np.arange(0.5 * dstep + dataz[0], dataz[-1], dstep)

        self.tg = (dI[:, None]+0.5*dstep*tg_n).ravel()
        self.ag = (dI[:, None]*0+ag_n).ravel()
        self.dstep = dstep

    def build_trajectory(self, Bx, By, Bz, gamma=None):
        if self.cl_ctx is None:
            return self._build_trajectory_conv(Bx, By, Bz, gamma)
        else:
            return self._build_trajectory_CL(Bx, By, Bz, gamma)

    def _build_trajectory_CL(self, Bx, By, Bz, gamma=None):
        if gamma is None:
            gamma = self.gamma
        scalarArgs = [np.int32(len(self.wtGrid))]  # jend
        if self.filamentBeam:
            scalarArgs.extend([self.cl_precisionF(gamma)])

        nonSlicedROArgs = [self.cl_precisionF(self.wtGrid),  # Integration grid
                           self.cl_precisionF(Bx),  # Mangetic field
                           self.cl_precisionF(By),  # components on the
                           self.cl_precisionF(Bz)]  # Runge-Kutta grid

        nonSlicedRWArgs = [np.zeros_like(self.wtGrid),  # beta.x
                           np.zeros_like(self.wtGrid),  # beta.y
                           np.zeros_like(self.wtGrid),  # beta.z average
                           np.zeros_like(self.wtGrid),  # traj.x
                           np.zeros_like(self.wtGrid),  # traj.y
                           np.zeros_like(self.wtGrid)]  # traj.z

        clKernel = 'get_trajectory_filament' if self.filamentBeam\
            else 'get_trajectory'

        betax, betay, betazav, trajx, trajy, trajz = self.ucl.run_parallel(
            clKernel, scalarArgs, None, nonSlicedROArgs,
            None, nonSlicedRWArgs, 1)

#        betaxTg = UnivariateSpline(self.wtGrid, betax, s=1e-6)(self.tg)
#        betayTg = UnivariateSpline(self.wtGrid, betay, s=1e-6)(self.tg)
#        trajxTg = UnivariateSpline(self.wtGrid, trajx, s=1e-6)(self.tg)
#        trajyTg = UnivariateSpline(self.wtGrid, trajy, s=1e-6)(self.tg)
#        trajzTg = UnivariateSpline(self.wtGrid, trajz, s=1e-6)(self.tg)
        betaxTg = interp1d(self.wtGrid, betax, **self.spl_kw)(self.tg)
        betayTg = interp1d(self.wtGrid, betay, **self.spl_kw)(self.tg)
        trajxTg = interp1d(self.wtGrid, trajx, **self.spl_kw)(self.tg)
        trajyTg = interp1d(self.wtGrid, trajy, **self.spl_kw)(self.tg)
        trajzTg = interp1d(self.wtGrid, trajz, **self.spl_kw)(self.tg)
        return betaxTg, betayTg, [betazav[-1]], trajxTg, trajyTg, trajzTg

    def _build_trajectory_conv(self, Bx, By, Bz, gamma=None):
        def f_beta(B, beta):
            return emcg*np.array((beta[1]*B[2]-B[1], B[0] - beta[0]*B[2]))

        def f_traj(beta):
            if self.filamentBeam:
                smTerm = 1./gamma**2 + beta[0]**2 + beta[1]**2
                betaz = 1. - 0.5*smTerm - 0.125*smTerm**2 - 0.0625*smTerm**3
            else:
                betaz = -0.5*(beta[0]**2 + beta[1]**2)
            return np.array((beta[0], beta[1], betaz))

        def next_beta_rk(iB, beta):
            k1beta = rkStep * f_beta([Bx[iB], By[iB], Bz[iB]],
                                     beta)
            k2beta = rkStep * f_beta([Bx[iB+1], By[iB+1], Bz[iB+1]],
                                     beta + 0.5*k1beta)
            k3beta = rkStep * f_beta([Bx[iB+1], By[iB+1], Bz[iB+1]],
                                     beta + 0.5*k2beta)
            k4beta = rkStep * f_beta([Bx[iB+2], By[iB+2], Bz[iB+2]],
                                     beta + k3beta)
            return beta + (k1beta + 2*k2beta + 2*k3beta + k4beta)/6.

        def next_traj_rk(iB, beta, traj):
            k1beta = rkStep * f_beta([Bx[iB], By[iB], Bz[iB]], beta)
            k1traj = rkStep * f_traj(beta)
            k2beta = rkStep * f_beta([Bx[iB+1], By[iB+1], Bz[iB+1]],
                                     beta + 0.5*k1beta)
            k2traj = rkStep * f_traj(beta + 0.5*k1beta)
            k3beta = rkStep * f_beta([Bx[iB+1], By[iB+1], Bz[iB+1]],
                                     beta + 0.5*k2beta)
            k3traj = rkStep * f_traj(beta + 0.5*k2beta)
            k4beta = rkStep * f_beta([Bx[iB+2], By[iB+2], Bz[iB+2]],
                                     beta + k3beta)
            k4traj = rkStep * f_traj(beta + k3beta)
            return (beta + (k1beta + 2*k2beta + 2*k3beta + k4beta)/6.,
                    traj + (k1traj + 2*k2traj + 2*k3traj + k4traj)/6.)

        if gamma is None:
            gamma = np.array(self.gamma)
        emcg = SIE0 / SIM0 / C / 10. / gamma if self.filamentBeam else 1.
        beta_next = np.zeros(2)
        beta0 = np.zeros(2)
        betam_int = 0

        for i in range(len(self.wtGrid)-1):
            rkStep = self.wtGrid[i+1] - self.wtGrid[i]
            beta_next = next_beta_rk(2*i, beta_next)
            beta0 += rkStep * beta_next

        beta0 /= -(self.wtGrid[-1] - self.wtGrid[0])
        beta_next = np.copy(beta0)
        traj_next = np.zeros(3)
        traj0 = np.zeros(3)

        for i in range(len(self.wtGrid)-1):
            rkStep = self.wtGrid[i+1] - self.wtGrid[i]
            beta_next, traj_next = next_traj_rk(2*i, beta_next, traj_next)
            traj0 += rkStep * traj_next
            if self.filamentBeam:
                betam_int += rkStep * np.sqrt(
                        1. - 1./gamma**2 - beta_next[0]**2 - beta_next[1]**2)
            else:
                betam_int += beta_next[0]**2 + beta_next[1]**2

        traj0 /= -(self.wtGrid[-1] - self.wtGrid[0])
        beta_next = np.copy(beta0)
        traj_next = np.copy(traj0)
        if self.filamentBeam:
            betam_int /= -(self.wtGrid[-1] - self.wtGrid[0])
        else:
            betam_int *= -0.5/(len(self.wtGrid)-1)

        betax = [beta0[0]]
        betay = [beta0[1]]
        trajx = [traj0[0]]
        trajy = [traj0[1]]
        trajz = [traj0[2]]

        for i in range(len(self.wtGrid)-1):
            rkStep = self.wtGrid[i+1] - self.wtGrid[i]
            beta_next, traj_next = next_traj_rk(2*i, beta_next, traj_next)
            betax.append(beta_next[0])
            betay.append(beta_next[1])
            trajx.append(traj_next[0])
            trajy.append(traj_next[1])
            trajz.append(traj_next[2])

#        betaxTg = UnivariateSpline(self.wtGrid, betax, s=1e-6)(self.tg)
#        betayTg = UnivariateSpline(self.wtGrid, betay, s=1e-6)(self.tg)
#        trajxTg = UnivariateSpline(self.wtGrid, trajx, s=1e-6)(self.tg)
#        trajyTg = UnivariateSpline(self.wtGrid, trajy, s=1e-6)(self.tg)
#        trajzTg = UnivariateSpline(self.wtGrid, trajz, s=1e-6)(self.tg)
        betaxTg = interp1d(self.wtGrid, betax, **self.spl_kw)(self.tg)
        betayTg = interp1d(self.wtGrid, betay, **self.spl_kw)(self.tg)
        trajxTg = interp1d(self.wtGrid, trajx, **self.spl_kw)(self.tg)
        trajyTg = interp1d(self.wtGrid, trajy, **self.spl_kw)(self.tg)
        trajzTg = interp1d(self.wtGrid, trajz, **self.spl_kw)(self.tg)
        return betaxTg, betayTg, [betam_int], trajxTg, trajyTg, trajzTg

    def build_trajectory_periodic(self, Bx, By, Bz, gamma=None):
        if gamma is None:
            gamma = self.gamma
        gamma2 = gamma**2
        betam = 1. - (1. + 0.5*self.Kx**2 + 0.5*self.Ky**2) / 2. / gamma2

        self.wu = PI / self.L0 / gamma2 * \
            (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2WC

        z = 2*np.pi*self.tg/self.L0
        tgw = self.L0 / 2. / np.pi
        betax = self.Ky / gamma * np.cos(z)
        betay = -self.Kx / gamma * np.cos(z + self.phase)
        trajx = tgw * self.Ky / gamma * np.sin(z)
        trajy = -tgw * self.Kx / gamma * np.sin(z + self.phase)
        trajz = tgw * (betam*z - 0.125/gamma**2 *
                       (self.Ky**2 * np.sin(2*z) +
                        self.Kx**2 * np.sin(2*(z + self.phase))))
        return betax, betay, [betam], trajx, trajy, trajz

    def _build_I_map_custom_field_CL(
            self, w, ddtheta, ddpsi, harmonic=None, dgamma=None):
        NRAYS = 1 if len(np.array(w).shape) == 0 else len(w)
        gamma = self.gamma

        if self.eEspread > 0:
            if dgamma is not None:
                gamma += dgamma
            else:
                sz = 1 if self.filamentBeam else NRAYS
                gamma += gamma * self.eEspread * np.random.normal(size=sz)

        gamma = gamma * np.ones(NRAYS, dtype=self.cl_precisionF)
        scalarArgs = []  # R0
        R0 = self.R0 if self.R0 is not None else 0

        if self.customFieldData is not None:
            Bx, By, Bz = self._magnetic_field()
        else:
            Bx, By, Bz = self._magnetic_field_periodic()

        if self.customFieldData is not None:
            betax, betay, betazav, trajx, trajy, trajz =\
                self.build_trajectory(Bx, By, Bz)
            Bxt, Byt, Bzt = self._magnetic_field(self.tg)
        else:
            betax, betay, betazav, trajx, trajy, trajz =\
                self.build_trajectory_periodic(Bx, By, Bz)
            Bxt, Byt, Bzt = self._magnetic_field_periodic(self.tg)

        emcg0 = EMC/gamma[0]

        if self.filamentBeam:
            self.beta = [betax, betay]
            self.trajectory = [trajx, trajy, trajz]
        else:
            self.beta = [betax*emcg0, betay*emcg0]
            self.trajectory = [
                trajx*emcg0, trajy*emcg0,
                self.tg*(1.-0.5/gamma[0]**2) + trajz*EMC**2/gamma[0]**2]

        betam = betazav[-1]
        ab = 0.5 / np.pi / betam

        if self.filamentBeam:
            scalarArgsTest = [np.int32(len(self.tg)),
                              self.cl_precisionF(emcg0),
                              self.cl_precisionF(1./gamma[0]**2),
                              self.cl_precisionF(R0),
                              self.cl_precisionF(w[0]*E2WC/betam)]

            slicedROArgs = [self.cl_precisionF(ddtheta),  # Theta
                            self.cl_precisionF(ddpsi)]  # Psi

            nonSlicedROArgs = [self.cl_precisionF(self.tg),  # Integration grid
                               self.cl_precisionF(self.ag),  # Intn weights
                               self.cl_precisionF(Bxt),  # Mangetic field
                               self.cl_precisionF(Byt),  # components on the
                               self.cl_precisionF(Bzt),  # CC grid
                               self.cl_precisionF(betax),  # Components of the
                               self.cl_precisionF(betay),  # velosity and
                               self.cl_precisionF(trajx),  # trajectory of the
                               self.cl_precisionF(trajy),  # electron on the
                               self.cl_precisionF(trajz)]  # CC grid

            slicedRWArgs = [np.zeros(NRAYS, dtype=self.cl_precisionC),  # Is
                            np.zeros(NRAYS, dtype=self.cl_precisionC)]  # Ip

            clKernel = 'custom_field_filament'

            Is_local, Ip_local = self.ucl.run_parallel(
                clKernel, scalarArgsTest, slicedROArgs, nonSlicedROArgs,
                slicedRWArgs, None, NRAYS)

        else:
            ab = 0.5 / np.pi / (1. - 0.5/gamma**2 + betam*EMC**2/gamma**2)

            scalarArgs.extend([np.int32(len(self.tg)),  # jend
                               self.cl_precisionF(betam),
                               self.cl_precisionF(self.R0)])

            slicedROArgs = [self.cl_precisionF(gamma),  # gamma
                            self.cl_precisionF(w),  # Energy
                            self.cl_precisionF(ddtheta),  # Theta
                            self.cl_precisionF(ddpsi)]  # Psi

            nonSlicedROArgs = [self.cl_precisionF(self.tg),  # Integration grid
                               self.cl_precisionF(self.ag),  # Intn weights
                               self.cl_precisionF(Bxt),  # Mangetic field
                               self.cl_precisionF(Byt),  # components on the
                               self.cl_precisionF(Bzt),  # CC grid
                               self.cl_precisionF(betax),  # Components of the
                               self.cl_precisionF(betay),  # velosity and
                               self.cl_precisionF(trajx),  # trajectory of the
                               self.cl_precisionF(trajy),  # electron on the
                               self.cl_precisionF(trajz)]  # CC grid

            slicedRWArgs = [np.zeros(NRAYS, dtype=self.cl_precisionC),  # Is
                            np.zeros(NRAYS, dtype=self.cl_precisionC)]  # Ip

            clKernel = 'custom_field'

            Is_local, Ip_local = self.ucl.run_parallel(
                clKernel, scalarArgs, slicedROArgs, nonSlicedROArgs,
                slicedRWArgs, None, NRAYS)

        bwFact = 0.001 if self.distE == 'BW' else 1./w
        Amp2Flux = FINE_STR * bwFact * self.eI / SIE0

        integralField = np.abs(Is_local)**2 + np.abs(Ip_local)**2
        if self.convergenceSearchFlag:
            return np.abs(np.sqrt(integralField) * 0.5 * self.dstep)
        else:
            return (Amp2Flux * 0.25 * self.dstep**2 * ab**2 * integralField,
                    np.sqrt(Amp2Flux) * Is_local * 0.5 * self.dstep * ab,
                    np.sqrt(Amp2Flux) * Ip_local * 0.5 * self.dstep * ab)

    def _build_I_map_custom_field_conv(self, w, ddtheta, ddpsi, harmonic=None,
                                       dgamma=None):
        NRAYS = 1 if len(np.array(w).shape) == 0 else len(w)
        gamma = self.gamma
        if self.eEspread > 0:
            if dgamma is not None:
                gamma += dgamma
            else:
                sz = 1 if self.filamentBeam else NRAYS
                gamma += gamma * self.eEspread * np.random.normal(size=sz)
        gamma = gamma * np.ones(NRAYS)

        R0 = self.R0 if self.R0 is not None else 0

        if self.customFieldData is not None and not self.periodicTest:
            Bx, By, Bz = self._magnetic_field()
        else:
            Bx, By, Bz = self._magnetic_field_periodic()

        if self.customFieldData is not None and not self.periodicTest:
            betax, betay, betazav, trajx, trajy, trajz =\
                self.build_trajectory(Bx, By, Bz)
#                betax3, betay3, betazav3, trajx3, trajy3, trajz3 =\
#                    self.build_trajectory_conv(Bx, By, Bz, gamma[0])
            Bxt, Byt, Bzt = self._magnetic_field(self.tg)

        else:
            betax, betay, betazav, trajx, trajy, trajz =\
                self.build_trajectory_periodic(Bx, By, Bz)
            Bxt, Byt, Bzt = self._magnetic_field_periodic(self.tg)

        betam = betazav[-1]
        ab = 0.5 / np.pi / betam if self.filamentBeam else\
            0.5 / np.pi / (1. - 0.5/gamma**2 + betam*EMC**2/gamma**2)
        emcg = SIE0 / SIM0 / C / 10. / gamma

        if self.filamentBeam:
            self.beta = [betax, betay]
            self.trajectory = [trajx, trajy, trajz]
        else:
            self.beta = [betax*emcg[0], betay*emcg[0]]
            self.trajectory = [
                trajx*emcg[0], trajy*emcg[0],
                self.tg*(1.-0.5/gamma[0]**2) + trajz*EMC**2/gamma[0]**2]

        if self.R0:
            R0v = np.array(
                (np.tan(ddtheta), np.tan(ddpsi), np.ones_like(ddpsi)))
#            R0n = np.linalg.norm(R0v, axis=0)  # Only for spherical screen
            R0v *= R0  # /R0n
        else:
            R0v = None

        if NRAYS > 10:  # sum along the integration grid in a loop
            Is_local, Ip_local = self._sp_sum(
                    emcg, w, gamma, ddtheta, ddpsi, Bxt, Byt, Bzt,
                    betax, betay, betam, trajx, trajy, trajz, R0v)
        else:  # Convergence only
            dim = len(np.array(w).shape)
            Is_local, Ip_local = self._sp(
                    dim, emcg, w, gamma, ddtheta, ddpsi, Bxt, Byt, Bzt,
                    betax, betay, betam, trajx, trajy, trajz, R0v)

        bwFact = 0.001 if self.distE == 'BW' else 1./w
        Amp2Flux = FINE_STR * bwFact * self.eI / SIE0

        integralField = np.abs(Is_local)**2 + np.abs(Ip_local)**2
        if self.convergenceSearchFlag:
            return np.abs(np.sqrt(integralField) * 0.5 * self.dstep)
        else:
            return (Amp2Flux * 0.25 * self.dstep**2 * ab**2 * integralField,
                    np.sqrt(Amp2Flux) * Is_local * 0.5 * self.dstep * ab,
                    np.sqrt(Amp2Flux) * Ip_local * 0.5 * self.dstep * ab)


class Undulator(IntegratedSource):
    u"""
    Undulator source. The computation is volumnous and thus a decent GPU is
    highly recommended.
    """

    def __init__(self, *args, **kwargs):
        """
        *period*, *n*:
            Magnetic period (mm) length and number of periods.

        *K*, *Kx*, *Ky*: float
            Deflection parameter for the vertical field or for an elliptical
            undulator.

        *B0x*, *B0y*: float
            Maximum magnetic field. If both K and B provided at the init, K
            value will be used.

        *phaseDeg*: float
            Phase difference between horizontal and vertical magnetic arrays in
            degrees. Used in the elliptical case where it should be equal
            to 90 or -90.

        *taper*: tuple(dgap(mm), gap(mm))
            Linear variation in undulator gap. None if tapering is not used.
            Pyopencl is recommended for tapering.

        *targetE*: a tuple (Energy, harmonic{, isElliptical})
            Can be given for automatic calculation of the deflection parameter.
            If isElliptical is not given, it is assumed as False (as planar).
            *isElliptical* here can also be an angle in radians between the
            resulting K and Kx (=np.pi/4 for a circular case).

        *xPrimeMaxAutoReduce*, *zPrimeMaxAutoReduce*: bool
            Whether to reduce too large angular ranges down to the feasible
            values in order to improve efficiency. It is highly recommended to
            keep them True.


        """
        period = kwargs.pop('period', 50)
        n = kwargs.pop('n', 50)
        K = kwargs.pop('K', 0)
        Kx = kwargs.pop('Kx', 0)
        Ky = kwargs.pop('Ky', 0)
        B0x = kwargs.pop('B0x', 0)
        B0y = kwargs.pop('B0y', 0)
        phaseDeg = kwargs.pop('phaseDeg', 0)
        taper = kwargs.pop('taper', None)
        targetE = kwargs.pop('targetE', None)
        xPrimeMaxAutoReduce = kwargs.pop('xPrimeMaxAutoReduce', True)
        zPrimeMaxAutoReduce = kwargs.pop('zPrimeMaxAutoReduce', True)
        super(Undulator, self).__init__(*args, **kwargs)

        self.L0 = period
        self.Np = n
        self.taper = taper
        self.phaseDeg = phaseDeg

        if targetE is not None:
            self._targetE = targetE
            Ky = np.sqrt(targetE[1] * 8 * PI * self.gamma2 /
                         period / targetE[0] / E2WC - 2)
            if raycing._VERBOSITY_ > 10:
                print("K = {0}".format(Ky))
            if np.isnan(Ky):
                raise ValueError("Cannot calculate K, try to increase the "
                                 "undulator harmonic number")
            if len(targetE) > 2:
                elliptical = targetE[2]
                if elliptical:
                    if isinstance(elliptical, float):
                        Kx = Ky * np.cos(elliptical)
                        Ky = Ky * np.sin(elliptical)
                        if raycing._VERBOSITY_ > 10:
                            print("Kx = {0}, Ky = {1}".format(Kx, Ky))
                    else:
                        Kx = Ky = Ky / 2**0.5
                        if raycing._VERBOSITY_ > 10:
                            print("Kx = Ky = {0}".format(Kx))

        self.Kbase = True

        if Kx == 0 and Ky == 0:
            if abs(K) > 0:
                self.Kx = 0
                self.K = K
            elif B0x == 0 and B0y == 0:
                self.Kx = 0
                self.K = 1
                raise ValueError("Please define either K or B0!")
            else:
                self.Kbase = False
                self.B0y = B0y
                self.B0x = B0x
        else:
            self.Kx = Kx
            self.Ky = Ky

        self.xPrimeMaxAutoReduce = xPrimeMaxAutoReduce
        self.zPrimeMaxAutoReduce = zPrimeMaxAutoReduce
        if self.R0 is not None:  # Required for convergence
            self.xPrimeMaxAutoReduce = True
            self.zPrimeMaxAutoReduce = True

    @property
    def xPrimeMaxAutoReduce(self):
        return self._xPrimeMaxAutoReduce

    @xPrimeMaxAutoReduce.setter
    def xPrimeMaxAutoReduce(self, xPrimeMaxAutoReduce):
        self._xPrimeMaxAutoReduce = xPrimeMaxAutoReduce
        self.needReset = True

    @property
    def zPrimeMaxAutoReduce(self):
        return self._zPrimeMaxAutoReduce

    @zPrimeMaxAutoReduce.setter
    def zPrimeMaxAutoReduce(self, zPrimeMaxAutoReduce):
        self._zPrimeMaxAutoReduce = zPrimeMaxAutoReduce
        self.needReset = True

    @property
    def phaseDeg(self):
        return self._phaseDeg

    @phaseDeg.setter
    def phaseDeg(self, phaseDeg):
        phaseDeg = np.degrees(raycing.auto_units_angle(phaseDeg)) if\
            isinstance(phaseDeg, raycing.basestring) else phaseDeg
        self._phaseDeg = phaseDeg
        self.phase = np.radians(phaseDeg)
        self.needReset = True

    @property
    def period(self):
        return self.L0

    @period.setter
    def period(self, period):
        self.L0 = float(period)
        if hasattr(self, 'Kbase'):
            if self.Kbase:
                self._B0x = K2B * self.Kx / self.L0
                self._B0y = K2B * self.Ky / self.L0
            else:
                self._Kx = self.B0x * self.L0 / K2B
                self._Ky = self.B0y * self.L0 / K2B
        self.report_E1()
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def n(self):
        return self.Np

    @n.setter
    def n(self, n):
        self.Np = float(n)
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def targetE(self):
        return self._targetE

    @targetE.setter
    def targetE(self, targetE):
        self._targetE = targetE
        Ky = np.sqrt(targetE[1] * 8 * PI * self.gamma2 /
                     self.L0 / targetE[0] / E2WC - 2)
        Kx = 0
        if np.isnan(Ky):
            raise ValueError("Cannot calculate K, try to increase the "
                             "undulator harmonic number")
        if len(targetE) > 2:
            isElliptical = targetE[2]
            if isElliptical:
                Kx = Ky = Ky / 2**0.5
                if raycing._VERBOSITY_ > 10:
                    print("Kx = Ky = {0}".format(Kx))
        self._Kx = Kx
        self._Ky = Ky
        if raycing._VERBOSITY_ > 10:
            if Kx == 0:
                print("K = {0}".format(Ky))
            else:
                print("Kx = {0}, Ky = {1}".format(Kx, Ky))
        self.report_E1()
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def taper(self):
        return self._taperVal

    @taper.setter
    def taper(self, taper):
        if isinstance(taper, (list, tuple)) and len(taper) == 2:
            self._taper = taper
            self._taperVal = taper[0] / self.Np / self.L0 / taper[1]
            self.gap = taper[1]
        else:
            self._taperVal = None
            self._taper = None
        self.needReset = True

    @property
    def Kx(self):
        return self._Kx

    @Kx.setter
    def Kx(self, Kx):
        self._Kx = float(Kx)
        self._B0x = K2B * Kx / self.L0
        if hasattr(self, '_Ky'):
            self.report_E1()
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def Ky(self):
        return self._Ky

    @Ky.setter
    def Ky(self, Ky):
        self._Ky = float(Ky)
        self._B0y = K2B * Ky / self.L0
        if hasattr(self, '_Kx'):
            self.report_E1()
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def K(self):
        return self._Ky

    @K.setter
    def K(self, K):
        self._Ky = float(K)
        self._B0y = K2B * K / self.L0
        if hasattr(self, '_Kx'):
            self.report_E1()
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def B0x(self):
        return self._B0x

    @B0x.setter
    def B0x(self, B0x):
        self._B0x = float(B0x)
        self._Kx = B0x * self.L0 / K2B
        if hasattr(self, '_Ky'):
            self.report_E1()
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def B0y(self):
        return self._B0y

    @B0y.setter
    def B0y(self, B0y):
        self._B0y = float(B0y)
        self._Ky = B0y * self.L0 / K2B
        if hasattr(self, '_Kx'):
            self.report_E1()
        self.needReset = True
        # Need to recalculate the integration parameters

    def report_E1(self):
        wu = PI / self.L0 / self.gamma2 * \
            (2*self.gamma2 - 1. - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2WC

        E1 = 2*wu*self.gamma2 / (1 + 0.5*self.Kx**2 + 0.5*self.Ky**2)
        if raycing._VERBOSITY_ >= 10:
            print("E1 = {0}".format(E1))
            print("E3 = {0}".format(E1*3))
            print("E5 = {0}".format(E1*5))
            print("E7 = {0}".format(E1*7))
            print("E9 = {0}".format(E1*9))
            print("E11 = {0}".format(E1*11))
            print("B0 = {0}".format(self.B0y))
            if self.taper is not None:
                print("dB/dx/B = {0}".format(
                    -PI * self.gap * self.taper / self.L0 * 1e3))
        self.E1 = E1

    def prefix_save_name(self):
        if self.Kx > 0:
            return '4-elu-xrt'
        else:
            return '1-und-xrt'

    def tuning_curves(self, energy, theta, psi, harmonics, Ks):
        """Calculates *tuning curves* -- maximum flux of given *harmomonics* at
        given K values (*Ks*). The flux is calculated through the aperture
        defined by *theta* and *psi* opening angles (1D arrays).

        Returns two 2D arrays: energy in keV and flux values in ph/s/0.1%bw.
        The rows correspond to *Ks*, the colums correspond to *harmomonics*.
        """
        try:
            dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
        except TypeError:
            dtheta, dpsi = 1, 1
        tunesE, tunesF = [], []
        tmpKy = self.Ky
        for iK, K in enumerate(Ks):
            if raycing._VERBOSITY_ >= 10:
                print("\nCalculation {1} of {2}, K={0:.3f}".format(
                    K, iK+1, len(Ks)))
            self.Ky = K
            # all energies can be calculated at once but for big theta and psi
            # arrays the I0 array may become too big to fit into memory:
            # I0 = self.intensities_on_mesh(energy, theta, psi, harmonics)[0]
            # flux = I0.sum(axis=(1, 2)) * dtheta * dpsi
            # therefore, energy axis is split into a loop:
            flux = None
            for ie, e in enumerate(energy):
                if ie % 100 == 0:
                    print("Calculation at E={0}, K={1:.3f}".format(e, K))
                I0 = self.intensities_on_mesh([e], theta, psi, harmonics)[0]
                iflux = I0.sum(axis=(1, 2)) * dtheta * dpsi
                if flux is None:
                    flux = iflux
                else:
                    flux = np.vstack((flux, iflux))
            argm = np.argmax(flux, axis=0)
            fluxm = np.max(flux, axis=0)
            tunesE.append(energy[argm] / 1000.)
            tunesF.append(fluxm)
        self.Ky = tmpKy

        return np.array(tunesE).T, np.array(tunesF).T

    def power_vs_K(self, energy, theta, psi, harmonics, Ks):
        """Calculates *power curve* -- total power in W for all *harmomonics*
        at given K values (*Ks*). The power is calculated through the aperture
        defined by *theta* and *psi* opening angles within the *energy* range.

        The result of this numerical integration depends on the used angular
        and energy meshes; you should check convergence. Internally, electron
        beam energy spread is also sampled by adding another dimension to the
        intensity array and making it 5-dimensional. You therefore may want to
        set energy spread to zero, it doesn’t affect the resulting power
        anyway.

        Returns a 1D array corresponding to *Ks*.
        """
        try:
            dtheta, dpsi, dE = \
                theta[1] - theta[0], psi[1] - psi[0], energy[1] - energy[0]
        except TypeError:
            dtheta, dpsi, dE = 1, 1, 1
        tmpKy = self.Ky
        powers = []
        for iK, K in enumerate(Ks):
            if raycing._VERBOSITY_ >= 10:
                print("Calculation {1} of {2}, K={0}".format(K, iK+1, len(Ks)))
            self.Ky = K
            I0 = self.intensities_on_mesh(energy, theta, psi, harmonics)[0]
            if self.distE == 'BW':
                I0 *= 1e3
            else:  # 'eV'
                I0 *= energy[:, np.newaxis, np.newaxis, np.newaxis]
            power = I0.sum() * dtheta * dpsi * dE * EV2ERG * 1e-7  # [W]
            powers.append(power)
        self.Ky = tmpKy
        return np.array(powers)

    def _build_integration_grid(self):
        quad_rule = np.polynomial.legendre.leggauss if self._useGauLeg else\
            self._clenshaw_curtis
        self.tg_n, self.ag_n = quad_rule(self.quadm)

        dstep = 2 * PI / float(self.gIntervals)
        dI = np.arange(-PI + 0.5 * dstep, PI, dstep)

        self.tg = (dI[:, None]+0.5*dstep*self.tg_n).ravel()
        self.ag = (dI[:, None]*0+self.ag_n).ravel()
        self.sintg = np.sin(self.tg)
        self.costg = np.cos(self.tg)
        self.sintgph = np.sin(self.tg + self.phase)
        self.costgph = np.cos(self.tg + self.phase)
        self.dstep = dstep

#    @profile
    def _sp(self, dim, ww1, w, wu, gamma, ddphi, ddpsi, R0=None):
        lengamma = 1 if len(np.array(gamma).shape) == 0 else len(gamma)
        gS = gamma
        if dim == 0:
            ww1S = ww1
            wS, wuS = w, wu
            ddphiS = ddphi
            ddpsiS = ddpsi
        elif dim == 1:
            ww1S = ww1[:, np.newaxis]
            wS = w[:, np.newaxis]
            wuS = wu[:, np.newaxis]
            ddphiS = ddphi[:, np.newaxis]
            ddpsiS = ddpsi[:, np.newaxis]
            if lengamma > 1:
                gS = gamma[:, np.newaxis]
#        elif dim == 3:
#            ww1S = ww1[:, :, :, np.newaxis]
#            wS, wuS = w[:, :, :, np.newaxis], wu[:, :, :, np.newaxis]
#            ddphiS = ddphi[:, :, :, np.newaxis]
#            ddpsiS = ddpsi[:, :, :, np.newaxis]
#            if lengamma > 1:
#                gS = gamma[:, :, :, np.newaxis]

        taperC = 1
        alphaS = 0

        if (R0 is not None) or (self.taper is not None):
            dI = np.arange(0.5*self.dstep - PI*self.Np, PI*self.Np, self.dstep)
            tg = (dI[:, None] + 0.5*self.dstep*self.tg_n).ravel()
            ag = np.tile(self.ag, self.Np)
            sinx = np.tile(self.sintg, self.Np)
            cosx = np.tile(self.costg, self.Np)
            sinxph = np.tile(self.sintgph, self.Np)
            cosxph = np.tile(self.costgph, self.Np)
        else:
            tg = self.tg
            ag = self.ag
            sinx = self.sintg
            cosx = self.costg
            sinxph = self.sintgph
            cosxph = self.costgph

        sin2x = 2*sinx*cosx
        sin2xph = 2*sinxph*cosxph
        revgamma = 1./gS
        revgamma2 = revgamma**2
        betam = 1. - (1. + 0.5*self.Kx**2 + 0.5*self.Ky**2)*0.5*revgamma2
        wwuS = wS/wuS
        dirx = ddphiS
        diry = ddpsiS
        dirz = 1. - 0.5*(ddphiS**2 + ddpsiS**2)
        if self.taper is not None:
            alphaS = self.taper/E2WC
            taperC = 1 - alphaS*tg/wuS
            ucos = ww1S*tg + \
                wwuS*revgamma*(
                    -self.Ky*dirx*(sinx + alphaS/wuS*(1 - cosx - tg*sinx)) +
                    self.Kx*diry*sinx + 0.125*revgamma*(
                        self.Kx**2 * sin2xph + self.Ky**2 *
                        (sin2x - 2*alphaS/wuS*(tg**2 + cosx**2 + tg*sin2x))))
            eucos = np.exp(1j*ucos)
        elif R0 is not None:
            sinr0z = np.sin(wwuS*R0[-1])
            cosr0z = np.cos(wwuS*R0[-1])

            zterm = 0.5*(self.Ky**2*sin2x +
                         self.Kx**2*sin2xph)*revgamma
            rloc = np.array([self.Ky*sinx*revgamma,
                             self.Kx*sinxph*revgamma,
                             betam*tg-0.25*zterm*revgamma])
            dr = R0 - rloc

            dist = np.linalg.norm(dr, axis=0)
            ucos = wwuS*(tg + dist)
            drs = 0.5*(dr[0, :]**2+dr[1, :]**2)/dr[2, :]

            sinzloc = np.sin(wwuS * tg*(1.-betam))
            coszloc = np.cos(wwuS * tg*(1.-betam))
            sindrs = np.sin(wwuS * (drs + 0.25 * zterm * revgamma))
            cosdrs = np.cos(wwuS * (drs + 0.25 * zterm * revgamma))

            eucosx = (-sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -
                      cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs)
            eucosy = (-sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +
                      cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs)
            eucos = eucosx + 1j*eucosy

            direction = dr/dist
            dirx = direction[0, :]
            diry = direction[1, :]
            dirz = direction[2, :]
        else:
            ucos = ww1S*tg + wwuS*revgamma*(
                -self.Ky*ddphiS*sinx + self.Kx*ddpsiS*sinxph +
                0.125*revgamma*(self.Ky**2 * sin2x + self.Kx**2 * sin2xph))
            eucos = np.exp(1j*ucos)

        betax = taperC*self.Ky*revgamma*cosx
        betay = -self.Kx*revgamma*cosxph
        betaz = 1. - 0.5*(revgamma2 + betax*betax + betay*betay)

        betaPx = -self.Ky*(alphaS*cosx + taperC*sinx)
        betaPy = self.Kx*sinxph
        betaPz = 0.5*revgamma*(
            self.Ky**2 * taperC*(alphaS*cosx**2 + taperC*sin2x) +
            self.Kx**2 * sin2xph)

        rkrel = 1./(1. - dirx*betax - diry*betay - dirz*betaz)
        eucos *= ag * rkrel**2
        bnx = dirx - betax
        bny = diry - betay
        bnz = dirz - betaz

        dirDotBetaP = dirx*betaPx + diry*betaPy + dirz*betaPz
        dirDotDmB = dirx*bnx + diry*bny + dirz*bnz

        Bsr = np.sum(eucos*wuS*revgamma*(bnx*dirDotBetaP - betaPx*dirDotDmB),
                     axis=dim)
        Bpr = np.sum(eucos*wuS*revgamma*(bny*dirDotBetaP - betaPy*dirDotDmB),
                     axis=dim)

        return Bsr, Bpr

#    @profile
    def _sp_sum(self, ww1S, wS, wuS, gS, ddphiS, ddpsiS, R0=None):

        taperC = 1
        alphaS = 0

        sin2x = 2.*self.sintg*self.costg
        sin2xph = 2.*self.sintgph*self.costgph
        revgamma = 1. / gS
        revgamma2 = revgamma**2
        betam = 1. - (1. + 0.5*self.Kx**2 + 0.5*self.Ky**2)*0.5*revgamma2
        wwuS = wS/wuS

        Bsr = np.complex128(0)
        Bpr = np.complex128(0)

        dirx = ddphiS
        diry = ddpsiS
        dirz = 1. - 0.5*(ddphiS**2 + ddpsiS**2)
        Nmx = self.Np if (R0 is not None or self.taper is not None) else 1

        if R0 is not None:
            sinr0z = np.sin(R0[-1])
            cosr0z = np.cos(R0[-1])

        for Nperiod in range(Nmx):
            if raycing._VERBOSITY_ > 80 and (self.taper is not None or
                                             R0 is not None):
                print("Period {} out of {}".format(Nperiod+1, Nmx))
            for i in range(len(self.tg)):
                if self.taper is not None:
                    zloc = -(Nmx-1)*np.pi + Nperiod*PI2 + self.tg[i]
                    alphaS = self.taper/E2WC
                    taperC = 1. - alphaS*zloc/wuS
                    ucos = ww1S*zloc +\
                        wwuS*revgamma *\
                        (-self.Ky*dirx*(self.sintg[i] + alphaS/wuS *
                                        (1 - self.costg[i] -
                                         zloc*self.sintg[i])) +
                         self.Kx*diry*self.sintg[i] + 0.125*revgamma *
                         (self.Kx**2 * sin2xph[i] + self.Ky**2 * (sin2x[i] -
                          2*alphaS/wuS*(zloc**2 + self.costg[i]**2 +
                                        zloc*sin2x[i]))))
                    eucos = np.exp(1j*ucos)
                elif R0 is not None:
                    zterm = 0.5*(self.Ky**2*sin2x[i] +
                                 self.Kx**2*sin2xph[i])*revgamma
                    zloc = -(Nmx-1)*np.pi + Nperiod*PI2 + self.tg[i]
                    rloc = np.array([self.Ky*self.sintg[i]*revgamma,
                                     self.Kx*self.sintgph[i]*revgamma,
                                     betam*zloc-0.25*zterm*revgamma])

                    dr = R0 - rloc
                    dist = np.linalg.norm(dr, axis=0)

                    drs = 0.5*(dr[0, :]**2+dr[1, :]**2) / dr[2, :]

                    sinzloc = np.sin(wwuS * zloc*(1.-betam))
                    coszloc = np.cos(wwuS * zloc*(1.-betam))
                    sindrs = np.sin(wwuS * (drs + 0.25*zterm*revgamma))
                    cosdrs = np.cos(wwuS * (drs + 0.25*zterm*revgamma))

                    eucosx = (-sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -
                              cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs)
                    eucosy = (-sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +
                              cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs)
                    eucos = eucosx + 1j*eucosy

                    direction = dr/dist
                    dirx = direction[0, :]
                    diry = direction[1, :]
                    dirz = direction[2, :]

                else:
                    ucos = ww1S*self.tg[i] + wwuS*revgamma *\
                        (-self.Ky*ddphiS*self.sintg[i] +
                         self.Kx*ddpsiS*self.sintgph[i] +
                         0.125*revgamma*(self.Ky**2 * sin2x[i] +
                                         self.Kx**2 * sin2xph[i]))
                    eucos = np.exp(1j*ucos)
#                print("eucos.shape", eucos.shape)
                betax = taperC*self.Ky*revgamma*self.costg[i]
                betay = -self.Kx*revgamma*self.costgph[i]
                betaz = 1. - 0.5*(revgamma2 + betax*betax + betay*betay)

                betaPx = -self.Ky*(alphaS*self.costg[i] + taperC*self.sintg[i])
                betaPy = self.Kx*self.sintgph[i]
                betaPz = 0.5*revgamma *\
                    (self.Ky**2 * taperC*(
                        alphaS*self.costg[i]**2 + taperC*sin2x[i]) +
                     self.Kx**2 * sin2xph[i])

                rkrel = 1./(1. - dirx*betax - diry*betay - dirz*betaz)
                eucos *= self.ag[i] * rkrel**2

                bnx = dirx - betax
                bny = diry - betay
                bnz = dirz - betaz

                dirDotBetaP = dirx*betaPx + diry*betaPy + dirz*betaPz
                dirDotDmB = dirx*bnx + diry*bny + dirz*bnz

                Bsr += eucos*(bnx*dirDotBetaP - betaPx*dirDotDmB)
                Bpr += eucos*(bny*dirDotBetaP - betaPy*dirDotDmB)

        return wuS*revgamma*Bsr, wuS*revgamma*Bpr

    def build_I_map(self, w, ddtheta, ddpsi, harmonic=None, dg=None):
        if self.needReset:
            self.reset()
        useCL = False
        if isinstance(w, np.ndarray):
            if w.shape[0] > 1:
                useCL = True
        if (self.cl_ctx is None) or not useCL:
            return self._build_I_map_conv(w, ddtheta, ddpsi, harmonic, dg)
        else:
            return self._build_I_map_CL(w, ddtheta, ddpsi, harmonic, dg)

    def _build_I_map_conv(self, w, ddtheta, ddpsi, harmonic, dgamma=None):
        #        np.seterr(invalid='ignore')
        #        np.seterr(divide='ignore')
        NRAYS = 1 if len(np.array(w).shape) == 0 else len(w)

        gamma = self.gamma
        if self.eEspread > 0:
            if dgamma is not None:
                gamma += dgamma
            else:
                sz = 1 if self.filamentBeam else NRAYS
                gamma += gamma * self.eEspread * np.random.normal(size=sz)
        gamma = gamma * np.ones(NRAYS)
        gamma2 = gamma**2

        wu = PI / self.L0 / gamma2 * np.ones_like(w) *\
            (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2WC
        ww1 = w * ((1. + 0.5*self.Kx**2 + 0.5*self.Ky**2) +
                   gamma2 * (ddtheta**2 + ddpsi**2)) / (2. * gamma2 * wu)

        if (self.taper is not None) or (self.R0 is not None):
            ab = 1. / PI2 / wu
        else:
            ab = 1. / PI2 / wu * np.sin(PI * self.Np * ww1) / np.sin(PI * ww1)

        if self.R0:
            R0v = np.array((np.tan(ddtheta),
                            np.tan(ddpsi),
                            np.ones_like(ddpsi)))
            # R0n = np.linalg.norm(R0v, axis=0)  # TODO: Only for spher. screen
            R0v *= self.R0*np.pi*2/self.L0  # /R0n
        else:
            R0v = None

        if NRAYS > 10:
            # if self.filamentBeam:
            #     gamma = self.gamma * np.ones(NRAYS)
            Is_local, Ip_local = self._sp_sum(
                    ww1, w, wu, gamma, ddtheta, ddpsi, R0v)
        else:  # Convergence only
            dim = len(np.array(w).shape)
            Is_local, Ip_local = self._sp(
                dim, ww1, w, wu, gamma, ddtheta, ddpsi, R0v)

        bwFact = 0.001 if self.distE == 'BW' else 1./w
        Amp2Flux = FINE_STR * bwFact * self.eI / SIE0

        if harmonic is not None:
            Is_local[ww1 > harmonic+0.5] = 0
            Ip_local[ww1 > harmonic+0.5] = 0
            Is_local[ww1 < harmonic-0.5] = 0
            Ip_local[ww1 < harmonic-0.5] = 0

        #        np.seterr(invalid='warn')
        #        np.seterr(divide='warn')
        integralField = np.abs(Is_local)**2 + np.abs(Ip_local)**2
        if self.convergenceSearchFlag:
            return np.abs(np.sqrt(integralField) * 0.5 * self.dstep)
        else:
            return (Amp2Flux * ab**2 * 0.25 * self.dstep**2 * integralField,
                    np.sqrt(Amp2Flux) * ab * Is_local * 0.5 * self.dstep,
                    np.sqrt(Amp2Flux) * ab * Ip_local * 0.5 * self.dstep)

    def _build_I_map_CL(self, w, ddtheta, ddpsi, harmonic, dgamma=None):
        NRAYS = 1 if len(np.array(w).shape) == 0 else len(w)
        gamma = self.gamma
        if self.eEspread > 0:
            if dgamma is not None:
                gamma += dgamma
            else:
                sz = 1 if self.filamentBeam else NRAYS
                gamma += gamma * self.eEspread * np.random.normal(size=sz)
        gamma = gamma * np.ones(NRAYS, dtype=self.cl_precisionF)
        gamma2 = gamma**2

        wu = PI / self.L0 / gamma2 *\
            (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2WC
#        wu = 2*PI / self.L0 / E2WC * np.ones_like(gamma)
        ww1 = w * ((1. + 0.5 * self.Kx**2 + 0.5 * self.Ky**2) +
                   gamma2 * (ddtheta * ddtheta + ddpsi * ddpsi)) /\
            (2. * gamma2 * wu)

        if (self.taper is not None) or (self.R0 is not None):
            ab = 1. / PI2 / wu
        else:
            ab = 1. / PI2 / wu * np.sin(PI * self.Np * ww1) / np.sin(PI * ww1)

        scalarArgs = [self.cl_precisionF(0.)]

        if self.R0 is not None:
            scalarArgs = [self.cl_precisionF(self.R0*np.pi*2/self.L0)]
        elif self.taper:
            scalarArgs = [self.cl_precisionF(self.taper)]

        scalarArgs.extend([self.cl_precisionF(self.Kx),  # Kx
                           self.cl_precisionF(self.Ky),  # Ky
                           np.int32(len(self.tg))])  # jend
        if (self.taper is not None) or (self.R0 is not None):
            scalarArgs.extend([np.int32(self.Np)])

        slicedROArgs = [self.cl_precisionF(gamma),  # gamma
                        self.cl_precisionF(wu),  # Wund
                        self.cl_precisionF(w),  # Energy
                        self.cl_precisionF(ww1),  # Energy/Eund(0)
                        self.cl_precisionF(ddtheta),  # Theta
                        self.cl_precisionF(ddpsi)]  # Psi

        nonSlicedROArgs = [self.cl_precisionF(self.tg),  # Integration grid
                           self.cl_precisionF(self.ag),  # Integration weights
                           self.cl_precisionF(self.sintg),  # Move outside
                           self.cl_precisionF(self.costg),
                           self.cl_precisionF(self.sintgph),
                           self.cl_precisionF(self.costgph)]
        slicedRWArgs = [np.zeros(NRAYS, dtype=self.cl_precisionC),  # Is
                        np.zeros(NRAYS, dtype=self.cl_precisionC)]  # Ip

        if self.taper is not None:
            clKernel = 'undulator_taper'
        elif self.R0 is not None:
            clKernel = 'undulator_nf'
#            if self.full:
#                clKernel = 'undulator_nf_full'
#        elif self.full:
#            clKernel = 'undulator_full'
        else:
            clKernel = 'undulator'

        Is_local, Ip_local = self.ucl.run_parallel(
            clKernel, scalarArgs, slicedROArgs, nonSlicedROArgs,
            slicedRWArgs, dimension=NRAYS)

        bwFact = 0.001 if self.distE == 'BW' else 1./w
        Amp2Flux = FINE_STR * bwFact * self.eI / SIE0

        if harmonic is not None:
            Is_local[ww1 > harmonic+0.5] = 0
            Ip_local[ww1 > harmonic+0.5] = 0
            Is_local[ww1 < harmonic-0.5] = 0
            Ip_local[ww1 < harmonic-0.5] = 0

        integralField = np.abs(Is_local)**2 + np.abs(Ip_local)**2
        if self.convergenceSearchFlag:
            return np.abs(np.sqrt(integralField) * 0.5 * self.dstep)
        else:
            return (Amp2Flux * ab**2 * 0.25 * self.dstep**2 * integralField,
                    np.sqrt(Amp2Flux) * Is_local * ab * 0.5 * self.dstep,
                    np.sqrt(Amp2Flux) * Ip_local * ab * 0.5 * self.dstep)

#    def _reportNaN(self, x, strName):
#        nanSum = np.isnan(x).sum()
#        if nanSum > 0:
#            print("{0} NaN rays in {1}!".format(nanSum, strName))

    def get_sigma_r02(self, E):  # linear size
        """Squared sigma_{r0} as by Walker and by Ellaume and
        Tanaka and Kitamura J. Synchrotron Rad. 16 (2009) 380–386 (see the
        text after Eq(23))"""
        return 2 * CHeVcm/E*10 * self.L0*self.Np / PI2**2

    def get_sigmaP_r02(self, E):  # angular size
        """Squared sigmaP_{r0}"""
        return CHeVcm/E*10 / (2 * self.L0*self.Np)

    def get_sigma_r2(self, E, onlyOddHarmonics=True, with0eSpread=False):
        """Squared sigma_{r} as by
        Tanaka and Kitamura J. Synchrotron Rad. 16 (2009) 380–386
        that also depends on energy spread."""
        sigma_r02 = self.get_sigma_r02(E)
        if self.eEspread == 0 or with0eSpread:
            return sigma_r02
        harmonic = np.floor_divide(E, self.E1)
#        harmonic[harmonic < 1] = 1
        if onlyOddHarmonics:
            harmonic += harmonic % 2 - 1
        eEspread_norm = PI2 * harmonic * self.Np * self.eEspread
        Qa2 = self.tanaka_kitamura_Qa2(eEspread_norm/4.)  # note 1/4
        return sigma_r02 * Qa2**(2/3.)

    def get_sigmaP_r2(self, E, onlyOddHarmonics=True, with0eSpread=False):
        """Squared sigmaP_{r} as by
        Tanaka and Kitamura J. Synchrotron Rad. 16 (2009) 380–386
        that also depends on energy spread."""
        sigmaP_r02 = self.get_sigmaP_r02(E)
        if self.eEspread == 0 or with0eSpread:
            return sigmaP_r02
        harmonic = np.floor_divide(E, self.E1)
#        harmonic[harmonic < 1] = 1
        if onlyOddHarmonics:
            harmonic += harmonic % 2 - 1
        eEspread_norm = PI2 * harmonic * self.Np * self.eEspread
        Qa2 = self.tanaka_kitamura_Qa2(eEspread_norm)
        return sigmaP_r02 * Qa2

    def get_SIGMA(self, E, onlyOddHarmonics=True, with0eSpread=False):
        """Calculates total linear source size, also including the effect of
        electron beam energy spread. Uses Tanaka and Kitamura, J. Synchrotron
        Rad. 16 (2009) 380–6.

        *E* can be a value or an array. Returns a 2-tuple with x and y sizes.
        """
        sigma_r2 = self.get_sigma_r2(E, onlyOddHarmonics, with0eSpread)
        return ((self.dx**2 + sigma_r2)**0.5,
                (self.dz**2 + sigma_r2)**0.5)

    def get_SIGMAP(self, E, onlyOddHarmonics=True, with0eSpread=False):
        """Calculates total angular source size, also including the effect of
        electron beam energy spread. Uses Tanaka and Kitamura, J. Synchrotron
        Rad. 16 (2009) 380–6.

        *E* can be a value or an array. Returns a 2-tuple with x and y sizes.
        """
        sigmaP_r2 = self.get_sigmaP_r2(E, onlyOddHarmonics, with0eSpread)
        return ((self.dxprime**2 + sigmaP_r2)**0.5,
                (self.dzprime**2 + sigmaP_r2)**0.5)
