# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "20 Sep 2016"
import os
import sys
#import pickle
import numpy as np
from scipy import optimize
from scipy import special
import inspect

from .. import raycing
from . import myopencl as mcl
from .sources_beams import Beam, allArguments
from .physconsts import E0, C, M0, EV2ERG, K2B, SIE0,\
    SIM0, FINE_STR, PI, PI2, SQ2, SQ3, SQPI, E2W, CHeVcm, CHBAR

try:
    import pyopencl as cl  # analysis:ignore
    isOpenCL = True
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
except ImportError:
    isOpenCL = False

# _DEBUG replaced with raycing._VERBOSITY_


class BendingMagnet(object):
    u"""
    Bending magnet source. The computation is reasonably fast and thus a GPU
    is not required and is not implemented.
    """
    def __init__(self, bl=None, name='BM', center=(0, 0, 0),
                 nrays=raycing.nrays,
                 eE=3.0, eI=0.5, eEspread=0., eSigmaX=None, eSigmaZ=None,
                 eEpsilonX=1., eEpsilonZ=0.01, betaX=9., betaZ=2.,
                 B0=1., rho=None, filamentBeam=False, uniformRayDensity=False,
                 eMin=5000., eMax=15000., eN=51, distE='eV',
                 xPrimeMax=0.5, zPrimeMax=0.5, nx=25, nz=25, pitch=0, yaw=0):
        u"""
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`
            Container for beamline elements. Sourcess are added to its
            `sources` list.

        *name*: str
            User-specified name, can be used for diagnostics output.

        *center*: tuple of 3 floats
            3D point in global system.

        *nrays*: int
            The number of rays sampled in one iteration.

        *eE*: float
            Electron beam energy (GeV).

        *eI*: float
            Electron beam current (A).

        *eEspread*: float
            Energy spread relative to the beam energy.

        *eSigmaX*, *eSigmaZ*: float
            rms horizontal and vertical electron beam sizes (µm).
            Alternatively, betatron functions can be specified instead of the
            electron beam sizes.

        *eEpsilonX*, *eEpsilonZ*: float
            Horizontal and vertical electron beam emittance (nm rad).

        *betaX*, *betaZ*: float
            Betatron function (m). Alternatively, beam size can be specified.

        *B0*: float
            Magnetic field (T). Alternatively, specify *rho*.

        *rho*: float
            Curvature radius (m). Alternatively, specify *B0*.

        *eMin*, *eMax*: float
            Minimum and maximum photon energy (eV).

        *eN*: int
            Number of photon energy intervals, used only in the test suit,
            not required in ray tracing

        *distE*: 'eV' or 'BW'
            The resulted flux density is per 1 eV or 0.1% bandwidth. For ray
            tracing 'eV' is used.

        *xPrimeMax*, *zPrimeMax*: float
            Horizontal and vertical acceptance (mrad).

        *nx*, *nz*: int
            Number of intervals in the horizontal and vertical directions,
            used only in the test suit, not required in ray tracing.

        *filamentBeam*: bool
            If True the source generates coherent monochromatic wavefronts.
            Required for the wave propagation calculations.

        *pitch*, *yaw*: float
            rotation angles around x and z axis. Useful for canted sources and
            declined electron beams.


        """
        self.Ee = eE
        self.gamma = self.Ee * 1e9 * EV2ERG / (M0 * C**2)
        if isinstance(self, Wiggler):
            self.B = K2B * self.K / self.L0
            self.ro = M0 * C**2 * self.gamma / self.B / E0 / 1e6
            self.X0 = 0.5 * self.K * self.L0 / self.gamma / PI
            self.isMPW = True
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

        self.center = center  # 3D point in global system
        self.nrays = np.long(nrays)
        self.dx = eSigmaX * 1e-3 if eSigmaX else None
        self.dz = eSigmaZ * 1e-3 if eSigmaZ else None
        self.eEpsilonX = eEpsilonX
        self.eEpsilonZ = eEpsilonZ
        self.I0 = eI
        self.eEspread = eEspread
        self.eMin = float(eMin)
        self.eMax = float(eMax)

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.name] = [self, 0]

        xPrimeMax = raycing.auto_units_angle(xPrimeMax) * 1e3 if\
            isinstance(xPrimeMax, raycing.basestring) else xPrimeMax
        zPrimeMax = raycing.auto_units_angle(zPrimeMax) * 1e3 if\
            isinstance(zPrimeMax, raycing.basestring) else zPrimeMax
        self.xPrimeMax = xPrimeMax * 1e-3 if xPrimeMax else None
        self.zPrimeMax = zPrimeMax * 1e-3 if zPrimeMax else None
        self.betaX = betaX
        self.betaZ = betaZ
        self.eN = eN + 1
        self.nx = 2*nx + 1
        self.nz = 2*nz + 1
        self.xs = np.linspace(-self.xPrimeMax, self.xPrimeMax, self.nx)
        self.zs = np.linspace(-self.zPrimeMax, self.zPrimeMax, self.nz)
        self.energies = np.linspace(eMin, eMax, self.eN)
        self.distE = distE
        self.mode = 1
        self.uniformRayDensity = uniformRayDensity
        self.filamentBeam = filamentBeam
        self.pitch = raycing.auto_units_angle(pitch)
        self.yaw = raycing.auto_units_angle(yaw)

        if (self.dx is None) and (self.betaX is not None):
            self.dx = np.sqrt(self.eEpsilonX * self.betaX * 0.001)
        elif (self.dx is None) and (self.betaX is None):
            print("Set either eSigmaX or betaX!")
        if (self.dz is None) and (self.betaZ is not None):
            self.dz = np.sqrt(self.eEpsilonZ * self.betaZ * 0.001)
        elif (self.dz is None) and (self.betaZ is None):
            print("Set either eSigmaZ or betaZ!")

        dxprime, dzprime = None, None
        if dxprime:
            self.dxprime = dxprime
        else:
            self.dxprime = 1e-6 * self.eEpsilonX /\
                self.dx if self.dx > 0 else 0.  # [rad]
        if dzprime:
            self.dzprime = dzprime
        else:
            self.dzprime = 1e-6 * self.eEpsilonZ /\
                self.dz if self.dx > 0 else 0.  # [rad]

        self.gamma2 = self.gamma**2
        """" K2B: Conversion of Deflection parameter to magnetic field [T]
                        for the period in [mm]"""
        # self.c_E = 0.0075 * HPLANCK * C * self.gamma**3 / PI / EV2ERG
        # self.c_3 = 40. * PI * E0 * EV2ERG * self.I0 /\
        #    (np.sqrt(3) * HPLANCK * HPLANCK * C * self.gamma2) * \
        #    200. * EV2ERG / (np.sqrt(3) * HPLANCK * C * self.gamma2)

        mE = self.eN
        mTheta = self.nx
        mPsi = self.nz

        if self.isMPW:  # xPrimeMaxAutoReduce
            xPrimeMaxTmp = self.K / self.gamma
            if self.xPrimeMax > xPrimeMaxTmp:
                print("Reducing xPrimeMax from {0} down to {1} mrad".format(
                      self.xPrimeMax * 1e3, xPrimeMaxTmp * 1e3))
                self.xPrimeMax = xPrimeMaxTmp

        self.Theta_min = float(-self.xPrimeMax)
        self.Psi_min = float(-self.zPrimeMax)
        self.Theta_max = float(self.xPrimeMax)
        self.Psi_max = float(self.zPrimeMax)
        self.E_min = float(np.min(self.energies))
        self.E_max = float(np.max(self.energies))

        self.dE = (self.E_max-self.E_min) / float(mE-1)
        self.dTheta = (self.Theta_max-self.Theta_min) / float(mTheta-1)
        self.dPsi = (self.Psi_max-self.Psi_min) / float(mPsi-1)

        """Trying to find real maximum of the flux density"""
        E0fit = 0.5 * (self.E_max+self.E_min)

        precalc = True
        rMax = self.nrays
        if precalc:
            rE = np.random.uniform(self.E_min, self.E_max, rMax)
            rTheta = np.random.uniform(0., self.Theta_max, rMax)
            rPsi = np.random.uniform(0., self.Psi_max, rMax)
            DistI = self.build_I_map(rE, rTheta, rPsi)[0]
            f_max = np.amax(DistI)
            a_max = np.argmax(DistI)
            NZ = np.ceil(np.max(rPsi[np.where(DistI > 0)[0]]) / self.dPsi) *\
                self.dPsi
            self.zPrimeMax = min(self.zPrimeMax, NZ)
            self.Psi_max = float(self.zPrimeMax)
            initial_x = [(rE[a_max]-E0fit) * 1e-5,
                         rTheta[a_max] * 1e3, rPsi[a_max] * 1e3]
        else:
            xE, xTheta, xPsi = np.mgrid[
                self.E_min:self.E_max + 0.5*self.dE:self.dE,
                self.Theta_min:self.Theta_max + 0.5*self.dTheta:self.dTheta,
                self.Psi_min:self.Psi_max + 0.5*self.dPsi:self.dPsi]
            DistI = self.build_I_map(xE, xTheta, xPsi)[0]
            f_max = np.amax(DistI)
            initial_x = [
                (self.E_min + 0.6 * mE * self.dE - E0fit) * 1e-5,
                (self.Theta_min + 0.6 * mTheta * self.dTheta) * 1e3,
                (self.Psi_min + 0.6 * self.dPsi * mPsi) * 1e3]

        bounds_x = [
            ((self.E_min - E0fit) * 1e-5, (self.E_max - E0fit) * 1e-5),
            (0, self.Theta_max * 1e3),
            (0, self.Psi_max * 1e3)]

        def int_fun(x):
            return -1. * (self.build_I_map(x[0] * 1e5 + E0fit,
                                           x[1] * 1e-3,
                                           x[2] * 1e-3)[0]) / f_max
        res = optimize.fmin_slsqp(int_fun, initial_x,
                                  bounds=bounds_x,
                                  acc=1e-12,
                                  iter=1000,
                                  epsilon=1.e-8,
                                  full_output=1,
                                  iprint=0)
        self.Imax = max(-1 * int_fun(res[0]) * f_max, f_max)

        if self.filamentBeam:
            self.nrepmax = np.floor(rMax / len(np.where(
                self.Imax * np.random.rand(rMax) < DistI)[0]))

        """Preparing to calculate the total flux integral"""
        self.xzE = 4 * (self.E_max-self.E_min) * self.Theta_max * self.Psi_max
        self.fluxConst = self.Imax * self.xzE

    def prefix_save_name(self):
        return '3-BM-xrt'

    def build_I_map(self, dde, ddtheta, ddpsi):
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
        Amp2Flux = FINE_STR * bwFact * self.I0 / SIE0 * 2 * self.Np

        np.seterr(invalid='warn')
        np.seterr(divide='warn')

        return (Amp2Flux * (np.abs(ampS)**2 + np.abs(ampP)**2),
                np.sqrt(Amp2Flux) * ampS,
                np.sqrt(Amp2Flux) * ampP)

    def intensities_on_mesh(self, energy='auto', theta='auto', psi='auto'):
        if isinstance(energy, str):
            energy = np.mgrid[self.E_min:self.E_max + 0.5*self.dE:self.dE]
        if isinstance(theta, str):
            theta = np.mgrid[
                self.Theta_min:self.Theta_max + 0.5*self.dTheta:self.dTheta]
        if isinstance(psi, str):
            psi = np.mgrid[self.Psi_min:self.Psi_max + 0.5*self.dPsi:self.dPsi]
        xE, xTheta, xPsi = np.meshgrid(energy, theta, psi, indexing='ij')
        self.Itotal, ampS, ampP = self.build_I_map(xE, xTheta, xPsi)
        self.Is = (ampS * np.conj(ampS)).real
        self.Ip = (ampP * np.conj(ampP)).real
        self.Isp = ampS * np.conj(ampP)
        s0 = self.Is + self.Ip
        with np.errstate(divide='ignore'):
            Pol1 = np.where(s0, (self.Is - self.Ip) / s0, s0)
            Pol3 = np.where(s0, 2. * self.Isp / s0, s0)
        return (self.Itotal, Pol1, self.Is*0., Pol3)

    def shine(self, toGlobal=True, withAmplitudes=True, fixedEnergy=False,
              accuBeam=None):
        u"""
        Returns the source beam. If *toGlobal* is True, the output is in
        the global system. If *withAmplitudes* is True, the resulted beam
        contains arrays Es and Ep with the *s* and *p* components of the
        electric field.


        .. Returned values: beamGlobal
        """
        if self.bl is not None:
            try:
                self.bl._alignE = float(self.bl.alignE)
            except ValueError:
                self.bl._alignE = 0.5 * (self.eMin + self.eMax)

        if self.uniformRayDensity:
            withAmplitudes = True

        bo = None
        length = 0
        seeded = np.long(0)
        seededI = 0.
        np.seterr(invalid='warn')
        np.seterr(divide='warn')
        mcRays = np.long(self.nrays * 1.2) if not self.uniformRayDensity else\
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
#                print(self.Theta_min, rTheta0 - 1. / self.gamma)
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
            # as by Walker and by Ellaume; SPECTRA's value is two times
            # smaller:

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
                except:
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
            bo.filter_by_index(slice(0, np.long(self.nrays)))
        if self.filamentBeam:
            bo.filamentDtheta = dtheta
            bo.filamentDpsi = dpsi
        norm = np.sqrt(bo.a**2 + 1.0 + bo.c**2)
        bo.a /= norm
        bo.b /= norm
        bo.c /= norm
        if self.pitch or self.yaw:
            raycing.rotate_beam(bo, pitch=self.pitch, yaw=self.yaw)
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        raycing.append_to_flow(self.shine, [bo],
                               inspect.currentframe())
        return bo


class Wiggler(BendingMagnet):
    u"""
    Wiggler source. The computation is reasonably fast and thus a GPU
    is not required and is not implemented.
    """

    hiddenParams = ['B0', 'rho']

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
        self.K = kwargs.pop('K', 8.446)
        self.L0 = kwargs.pop('period', 50)
        self.Np = kwargs.pop('n', 40)
        name = kwargs.pop('name', 'wiggler')
        kwargs['name'] = name
        super(Wiggler, self).__init__(*args, **kwargs)
        self.reset()

    def prefix_save_name(self):
        return '2-Wiggler-xrt'

    def reset(self):
        """Needed for changing *K* after instantiation."""
        self.B = K2B * self.K / self.L0
        self.ro = M0 * C**2 * self.gamma / self.B / E0 / 1e6
        self.X0 = 0.5 * self.K * self.L0 / self.gamma / PI

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


class Undulator(object):
    u"""
    Undulator source. The computation is volumnous an thus requires a GPU.
    """
    def __init__(self, bl=None, name='und', center=(0, 0, 0),
                 nrays=raycing.nrays,
                 eE=6.0, eI=0.1, eEspread=0., eSigmaX=None, eSigmaZ=None,
                 eEpsilonX=1., eEpsilonZ=0.01, betaX=20., betaZ=5.,
                 period=50, n=50, K=10., Kx=0, Ky=0., phaseDeg=0,
                 taper=None, R0=None, targetE=None, customField=None,
                 eMin=5000., eMax=15000., eN=51, distE='eV',
                 xPrimeMax=0.5, zPrimeMax=0.5, nx=25, nz=25,
                 xPrimeMaxAutoReduce=True, zPrimeMaxAutoReduce=True,
                 gp=1e-2, gIntervals=1, nRK=30,
                 uniformRayDensity=False, filamentBeam=False,
                 targetOpenCL=raycing.targetOpenCL,
                 precisionOpenCL=raycing.precisionOpenCL,
                 pitch=0, yaw=0):
        u"""
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`
            Container for beamline elements. Sourcess are added to its
            `sources` list.

        *name*: str
            User-specified name, can be used for diagnostics output.

        *center*: tuple of 3 floats
            3D point in global system.

        *nrays*: int
            The number of rays sampled in one iteration.

        *eE*: float
            Electron beam energy (GeV).

        *eI*: float
            Electron beam current (A).

        *eEspread*: float
            Energy spread relative to the beam energy, rms.

        *eSigmaX*, *eSigmaZ*: float
            rms horizontal and vertical electron beam sizes (µm).
            Alternatively, betatron functions can be specified instead of the
            electron beam sizes.

        *eEpsilonX*, *eEpsilonZ*: float
            Horizontal and vertical electron beam emittance (nm rad).

        *betaX*, *betaZ*:
            Betatron function (m). Alternatively, beam size can be specified.

        *period*, *n*:
            Magnetic period (mm) length and number of periods.

        *K*, *Kx*, *Ky*: float
            Deflection parameter for the vertical field or for an elliptical
            undulator.

        *phaseDeg*: float
            Phase difference between horizontal and vertical magnetic arrays.
            Used in the elliptical case where it should be equal to 90 or -90.

        *taper*: tuple(dgap(mm), gap(mm))
            Linear variation in undulator gap. None if tapering is not used.
            Tapering should be used only with pyopencl.

        *R0*: float
            Distance center-to-screen for the near field calculations (mm).
            If None, the far field approximation (i.e. "usual" calculations) is
            used. Near field calculations should be used only with pyopencl.
            Here, a GPU can be much faster than a CPU.

        *targetE*: a tuple (Energy, harmonic{, isElliptical})
            Can be given for automatic calculation of the deflection parameter.
            If isElliptical is not given, it is assumed as False (as planar).

        *customField*: float or str or tuple(fileName, kwargs)
            If given, adds a constant longitudinal field or a table of field
            samples given as an Excel file or as a column text file. If given
            as a tuple or list, the 2nd member is a key word dictionary for
            reading Excel by :meth:`pandas.read_excel()` or reading text file
            by :meth:`numpy.loadtxt()`, e.g. ``dict(skiprows=4)`` for skipping
            the file header. The file must contain the columns with
            longitudinal coordinate in mm, B_hor, B_ver {, B_long}, all in T.

        *nRK*: int
            Size of the Runge-Kutta integration grid per each interval between
            Gauss-Legendre integration nodes (only valid if customField is not
            None).

        *eMin*, *eMax*: float
            Minimum and maximum photon energy (eV). Used as band width for flux
            calculation.

        *eN*: int
            Number of photon energy intervals, used only in the test suit, not
            required in ray tracing.

        *distE*: 'eV' or 'BW'
            The resulted flux density is per 1 eV or 0.1% bandwidth. For ray
            tracing 'eV' is used.

        *xPrimeMax*, *zPrimeMax*:
            Horizontal and vertical acceptance (mrad).

            .. note::
                The Monte Carlo sampling of the rays having their density
                proportional to the beam intensity can be extremely inefficient
                for sharply peaked distributions, like the undulator angular
                density distribution. It is therefore very important to
                restrict the sampled angular acceptance down to very small
                angles. Use this source only with reasonably small *xPrimeMax*
                and *zPrimeMax*!

        *nx*, *nz*: int
            Number of intervals in the horizontal and vertical directions,
            used only in the test suit, not required in ray tracing.

        *xPrimeMaxAutoReduce*, *zPrimeMaxAutoReduce*: bool
            Whether to reduce too large angular ranges down to the feasible
            values in order to improve efficiency. It is highly recommended to
            keep them True.

        *gp*: float
            Defines the precision of the Gauss integration.

        *gIntervals*: int
            Integral of motion is divided by gIntervals to reduce the order of
            Gauss-Legendre quadrature. Default value of 1 is usually enough for
            a conventional undulator. For extreme cases (wigglers, near field,
            wide angles) this value can be set to the order of few hundreds to
            achieve the convergence of the integral. Large values can
            significantly increase the calculation time and RAM consumption
            especially if OpenCL is not used.

        *uniformRayDensity*: bool
            If True, the radiation is sampled uniformly but with varying
            amplitudes, otherwise with the density proportional to intensity
            and with constant amplitudes. Required as True for wave propagation
            calculations. False is usual for ray-tracing.

        *filamentBeam*: bool
            If True the source generates coherent monochromatic wavefronts.
            Required as True for the wave propagation calculations in partially
            coherent regime.

        *targetOpenCL*:  None, str, 2-tuple or tuple of 2-tuples
            assigns the device(s) for OpenCL accelerated calculations. Accepts
            the following values:
            1) a tuple (iPlatform, iDevice) of indices in the
            lists cl.get_platforms() and platform.get_devices(), see the
            section :ref:`calculations_on_GPU`. None, if pyopencl is not
            wanted. Ignored if pyopencl is not installed.
            2) a tuple of tuples ((iP1, iD1),..,(iPn, iDn)) to assign specific
            devices from one or multiple platforms.
            3) int iPlatform - assigns all devices found at the given platform.
            4) 'GPU' - lets the program scan the system and select all found
            GPUs.
            5) 'CPU' - similar to 'GPU'. If one CPU exists in multiple
            platforms the program tries to select the vendor-specific driver.
            6) 'other' - similar to 'GPU', used for Intel PHI and other OpenCL-
            capable accelerator boards.
            7) 'all' - lets the program scan the system and assign all found
            devices. Not recommended, since the performance will be limited by
            the slowest device.
            8) 'auto' - lets the program scan the system and make an assignment
            according to the priority list: 'GPU', 'other', 'CPU' or None if no
            devices were found. Used by default.

            .. warning::
                A good graphics or dedicated accelerator card is highly
                recommended! Special cases as wigglers by the undulator code,
                near field, wide angles and tapering are hardly doable on CPU.

            .. note::
                Consider the :ref:`warnings and tips <usage_GPU_warnings>` on
                using xrt with GPUs.

        *precisionOpenCL*: 'float32' or 'float64', only for GPU.
            Single precision (float32) should be enough in most cases. The
            calculations with doube precision are much slower. Double precision
            may be unavailable on your system.

        *pitch*, *yaw*: float
            rotation angles around x and z axis. Useful for canted sources.


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

        self.center = center  # 3D point in global system
        self.nrays = np.long(nrays)
        self.gp = gp
        self.dx = eSigmaX * 1e-3 if eSigmaX else None
        self.dz = eSigmaZ * 1e-3 if eSigmaZ else None
        self.eEpsilonX = eEpsilonX * 1e-6
        self.eEpsilonZ = eEpsilonZ * 1e-6
        self.Ee = float(eE)
        self.eEspread = eEspread
        self.I0 = float(eI)
        self.eMin = float(eMin)
        self.eMax = float(eMax)
        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.name] = [self, 0]
        xPrimeMax = raycing.auto_units_angle(xPrimeMax) * 1e3 if\
            isinstance(xPrimeMax, raycing.basestring) else xPrimeMax
        zPrimeMax = raycing.auto_units_angle(zPrimeMax) * 1e3 if\
            isinstance(zPrimeMax, raycing.basestring) else zPrimeMax
        self.xPrimeMax = xPrimeMax * 1e-3  # if xPrimeMax else None
        self.zPrimeMax = zPrimeMax * 1e-3  # if zPrimeMax else None
        self.betaX = betaX * 1e3
        self.betaZ = betaZ * 1e3
        self.eN = eN + 1
        self.nx = 2*nx + 1
        self.nz = 2*nz + 1
        self.xs = np.linspace(-self.xPrimeMax, self.xPrimeMax, self.nx)
        self.zs = np.linspace(-self.zPrimeMax, self.zPrimeMax, self.nz)
        self.energies = np.linspace(eMin, eMax, self.eN)
        self.distE = distE
        self.uniformRayDensity = uniformRayDensity
        self.filamentBeam = filamentBeam
        self.pitch = raycing.auto_units_angle(pitch)
        self.yaw = raycing.auto_units_angle(yaw)
        self.gIntervals = gIntervals
        self.L0 = period
        self.R0 = R0 if R0 is None else R0 + self.L0*0.25
        self.nRK = nRK
        self.trajectory = None
        fullLength = False  # NOTE maybe a future input parameter
        self.full = fullLength
        if fullLength:
            # self.filamentBeam = True
            self.theta0 = 0
            self.psi0 = 0

        self.cl_ctx = None
        if (self.R0 is not None):
            precisionOpenCL = 'float64'
        if targetOpenCL is not None:
            if not isOpenCL:
                print("pyopencl is not available!")
            else:
                self.ucl = mcl.XRT_CL(
                    r'undulator.cl', targetOpenCL, precisionOpenCL)
                if self.ucl.lastTargetOpenCL is not None:
                    self.cl_precisionF = self.ucl.cl_precisionF
                    self.cl_precisionC = self.ucl.cl_precisionC
                    self.cl_queue = self.ucl.cl_queue
                    self.cl_ctx = self.ucl.cl_ctx
                    self.cl_program = self.ucl.cl_program
                    self.cl_mf = self.ucl.cl_mf
                    self.cl_is_blocking = self.ucl.cl_is_blocking

#        self.mode = 1

        if (self.dx is None) and (self.betaX is not None):
            self.dx = np.sqrt(self.eEpsilonX*self.betaX)
        elif (self.dx is None) and (self.betaX is None):
            print("Set either dx or betaX!")
        if (self.dz is None) and (self.betaZ is not None):
            self.dz = np.sqrt(self.eEpsilonZ*self.betaZ)
        elif (self.dz is None) and (self.betaZ is None):
            print("Set either dz or betaZ!")
        dxprime, dzprime = None, None
        if dxprime:
            self.dxprime = dxprime
        else:
            self.dxprime = self.eEpsilonX / self.dx if self.dx > 0\
                else 0.  # [rad]
        if dzprime:
            self.dzprime = dzprime
        else:
            self.dzprime = self.eEpsilonZ / self.dz if self.dz > 0\
                else 0.  # [rad]
        if raycing._VERBOSITY_ > 10:
            print('dx = {0} mm'.format(self.dx))
            print('dz = {0} mm'.format(self.dz))
            print('dxprime = {0} rad'.format(self.dxprime))
            print('dzprime = {0} rad'.format(self.dzprime))
        self.gamma = self.Ee * 1e9 * EV2ERG / (M0 * C**2)
        self.gamma2 = self.gamma**2

        if targetE is not None:
            K = np.sqrt(targetE[1] * 8 * PI * C * 10 * self.gamma2 /
                        period / targetE[0] / E2W - 2)
            if raycing._VERBOSITY_ > 10:
                print("K = {0}".format(K))
            if np.isnan(K):
                raise ValueError("Cannot calculate K, try to increase the "
                                 "undulator harmonic number")
            if len(targetE) > 2:
                isElliptical = targetE[2]
                if isElliptical:
                    Kx = Ky = K / 2**0.5
                    if raycing._VERBOSITY_ > 10:
                        print("Kx = Ky = {0}".format(Kx))

        self.Kx = Kx
        self.Ky = Ky
        self.K = K
        phaseDeg = np.degrees(raycing.auto_units_angle(phaseDeg)) if\
            isinstance(phaseDeg, raycing.basestring) else phaseDeg
        self.phase = np.radians(phaseDeg)

        self.Np = n

        if taper is not None:
            self.taper = taper[0] / self.Np / self.L0 / taper[1]
            self.gap = taper[1]
        else:
            self.taper = None
        if self.Kx == 0 and self.Ky == 0:
            self.Ky = self.K

        self.B0x = K2B * self.Kx / self.L0
        self.B0y = K2B * self.Ky / self.L0
        self.customField = customField

        if customField is not None:
            self.gIntervals *= 2
            if isinstance(customField, (tuple, list)):
                fname = customField[0]
                kwargs = customField[1]
            elif isinstance(customField, (float, int)):
                fname = None
                self.customFieldData = customField
                if customField > 0:
                    betaL = 2 * M0*C**2*self.gamma / customField / E0 / 1e6
                    print("Larmor betatron function = {0} m".format(betaL))
            else:
                fname = customField
                kwargs = {}
            if fname:
                self.customFieldData = self.read_custom_field(fname, kwargs)
        else:
            self.customFieldData = None

        if xPrimeMaxAutoReduce:
            xPrimeMaxTmp = self.Ky / self.gamma
            if self.xPrimeMax > xPrimeMaxTmp:
                print("Reducing xPrimeMax from {0} down to {1} mrad".format(
                      self.xPrimeMax * 1e3, xPrimeMaxTmp * 1e3))
                self.xPrimeMax = xPrimeMaxTmp
        if zPrimeMaxAutoReduce:
            K0 = self.Kx if self.Kx > 0 else 1.
            zPrimeMaxTmp = K0 / self.gamma
            if self.zPrimeMax > zPrimeMaxTmp:
                print("Reducing zPrimeMax from {0} down to {1} mrad".format(
                      self.zPrimeMax * 1e3, zPrimeMaxTmp * 1e3))
                self.zPrimeMax = zPrimeMaxTmp

        self.reset()

    def read_custom_field(self, fname, kwargs={}):
        def my_sin(x, a, k, ph, c):
            return a * np.cos(k * x + ph) + c
        from scipy.optimize import curve_fit

        if fname.endswith('.xls') or fname.endswith('.xlsx'):
            import pandas
            data = pandas.read_excel(fname, **kwargs).values
        else:
            data = np.loadtxt(fname)

        datalen4 = data.shape[0] // 10
        minBx = data[datalen4:-datalen4, 1].min()
        maxBx = data[datalen4:-datalen4, 1].max()
        minBy = data[datalen4:-datalen4, 2].min()
        maxBy = data[datalen4:-datalen4, 2].max()
        p0 = [(maxBx-minBx)/2., PI2/self.L0, 0.3, 1e-4]
        poptx, pcovx = curve_fit(my_sin,
                                 data[datalen4:-datalen4, 0],
                                 data[datalen4:-datalen4, 1],
                                 p0=p0)
        if poptx[0] < 0:
            poptx[0] *= -1
            poptx[2] += PI
        p0 = [(maxBy-minBy)/2., PI2/self.L0, 0.3, 1e-4]
        popty, pcovy = curve_fit(my_sin,
                                 data[datalen4:-datalen4, 0],
                                 data[datalen4:-datalen4, 2],
                                 p0=p0)
        if popty[0] < 0:
            popty[0] *= -1
            popty[2] += PI
        print(poptx)
        print(popty)
        B0x = poptx[0]
        B0y = popty[0]
        Kx = B0x * self.L0 / K2B
        Ky = B0y * self.L0 / K2B
        lambdaUx = PI2 / poptx[1]
        lambdaUy = PI2 / popty[1]
        phase = poptx[2] - popty[2]
        phaseDeg = phase / PI * 180
        print("field data in {0}:".format(fname))
        print("B0x={0:.3f}T, B0y={1:.3f}T".format(B0x, B0y))
        print("Kx={0:.3f}, Ky={1:.3f}".format(Kx, Ky))
        print(u"λ_Ux={0:.3f}mm, λ_Uy={1:.3f}mm".format(lambdaUx, lambdaUy))
        print(u"phase difference = {0:.3f}deg".format(phaseDeg))
        return data

    def magnetic_field(self, z):  # 'z' in radians
        if isinstance(self.customField, (float, int)):
            Bx = self.B0x * np.sin(z + self.phase)
            By = self.B0y * np.sin(z)
            Bz = self.customFieldData * np.ones_like(z)
        else:
            dataz = self.customFieldData[:, 0] / self.L0 * PI2
            Bx = np.interp(z, dataz, self.customFieldData[:, 1])
            By = np.interp(z, dataz, self.customFieldData[:, 2])
            if self.customFieldData.shape[1] > 3:
                Bz = np.interp(z, dataz, self.customFieldData[:, 3])
            else:
                Bz = np.zeros_like(Bx)
        return (Bx, By, Bz)

    def reset(self):
        """This method must be invoked after any changes in the undulator
        parameters."""
        self.wu = PI * (0.01 * C) / self.L0 / 1e-3 / self.gamma2 * \
            (2*self.gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2W
        # wnu = 2 * PI * (0.01 * C) / self.L0 / 1e-3 / E2W
        self.E1 = 2*self.wu*self.gamma2 / (1 + 0.5*self.Kx**2 + 0.5*self.Ky**2)
        if raycing._VERBOSITY_ > 10:
            print("E1 = {0}".format(self.E1))
            print("E3 = {0}".format(3*self.E1))
            print("B0 = {0}".format(self.Ky / 0.09336 / self.L0))
            if self.taper is not None:
                print("dB/dx/B = {0}".format(
                    -PI * self.gap * self.taper / self.L0 * 1e3))
        mE = self.eN
        mTheta = self.nx
        mPsi = self.nz

        if not self.xPrimeMax:
            print("No Theta range specified, using default 1 mrad")
            self.xPrimeMax = 1e-3

        self.Theta_min = -float(self.xPrimeMax)
        self.Theta_max = float(self.xPrimeMax)
        self.Psi_min = -float(self.zPrimeMax)
        self.Psi_max = float(self.zPrimeMax)

        self.energies = np.linspace(self.eMin, self.eMax, self.eN)
        self.E_min = float(np.min(self.energies))
        self.E_max = float(np.max(self.energies))

        self.dE = (self.E_max - self.E_min) / float(mE - 1)
        self.dTheta = (self.Theta_max - self.Theta_min) / float(mTheta - 1)
        self.dPsi = (self.Psi_max - self.Psi_min) / float(mPsi - 1)

        """Adjusting the number of points for Gauss integration"""
        # self.gp = 1
        quad_int_error = self.gp * 10.
        self.quadm = 0
        tmpeEspread = self.eEspread
        self.eEspread = 0
        mstart = 5
        m = mstart
        while quad_int_error >= self.gp:
            m += 1
            self.quadm = int(1.5**m)
            if self.cl_ctx is not None:
                # sE = np.linspace(self.E_min, self.E_max, self.eN)
                sE = self.E_max * np.ones(1)
                sTheta_max = self.Theta_max * np.ones(1)
                sPsi_max = self.Psi_max * np.ones(1)
                In = self.build_I_map(sE, sTheta_max, sPsi_max)[0][0]
            else:
                In = self.build_I_map(
                    self.E_max, self.Theta_max, self.Psi_max)[0]
            if m == mstart+1:
                I2 = In
                continue
            else:
                I1 = I2
                I2 = In
            quad_int_error = np.abs((I2 - I1)/I2)
            if raycing._VERBOSITY_ > 10:
                print("G = {0}".format(
                    [self.gIntervals, self.quadm, quad_int_error, I2,
                     2-self.ag_n.sum()]))
            if self.quadm > 400:
                self.gIntervals *= 2
                m = mstart
                quad_int_error = self.gp * 10.
                if self.gIntervals > 100:
                    break
                continue
        """end of Adjusting the number of points for Gauss integration"""
        self.eEspread = tmpeEspread
        if raycing._VERBOSITY_ > 10:
            print("Done with Gaussian optimization, {0} points will be used"
                  " in {1} interval{2}".format(
                      self.quadm, self.gIntervals,
                      's' if self.gIntervals > 1 else ''))

        if self.filamentBeam:
            rMax = self.nrays
            rE = np.random.uniform(self.E_min, self.E_max, rMax)
            rTheta = np.random.uniform(self.Theta_min, self.Theta_max, rMax)
            rPsi = np.random.uniform(self.Psi_min, self.Psi_max, rMax)
            tmpEspread = self.eEspread
            self.eEspread = 0
            DistI = self.build_I_map(rE, rTheta, rPsi)[0]
            self.Imax = np.max(DistI) * 1.2
            self.nrepmax = np.floor(rMax / len(np.where(
                self.Imax * np.random.rand(rMax) < DistI)[0]))
            self.eEspread = tmpEspread
        else:
            self.Imax = 0.
        """Preparing to calculate the total flux integral"""
        self.xzE = (self.E_max - self.E_min) *\
            (self.Theta_max - self.Theta_min) *\
            (self.Psi_max - self.Psi_min)
        self.fluxConst = self.Imax * self.xzE

    def prefix_save_name(self):
        if self.Kx > 0:
            return '4-elu-xrt'
        else:
            return '1-und-xrt'

    def tuning_curves(self, energy, theta, psi, harmonics, Ks):
        """Calculates *tuning curves* -- maximum flux of given *harmomonics* at
        given K values (*Ks*). The flux is calculated through the aperture
        defined by *theta* and *psi* opening angles.

        Returns two 2D arrays: energy positions and flux values. The rows
        correspond to *Ks*, the colums correspond to *harmomonics*.
        """
        try:
            dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
        except TypeError:
            dtheta, dpsi = 1, 1
        tunesE, tunesF = [], []
        tmpKy = self.Ky
        for iK, K in enumerate(Ks):
            if raycing._VERBOSITY_ > 10:
                print("K={0}, {1} of {2}".format(K, iK+1, len(Ks)))
            self.Ky = K
            self.reset()
            I0 = self.intensities_on_mesh(energy, theta, psi, harmonics)[0]
            flux = I0.sum(axis=(1, 2)) * dtheta * dpsi
            argm = np.argmax(flux, axis=0)
            fluxm = np.max(flux, axis=0)
            tunesE.append(energy[argm] / 1000.)
            tunesF.append(fluxm)
        self.Ky = tmpKy
        self.reset()
        return np.array(tunesE).T, np.array(tunesF).T

    def power_vs_K(self, energy, theta, psi, harmonics, Ks):
        """Calculates *power curve* -- total power in W for all *harmomonics*
        at given K values (*Ks*). The power is calculated through the aperture
        defined by *theta* and *psi* opening angles within the *energy* range.

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
            if raycing._VERBOSITY_ > 10:
                print("K={0}, {1} of {2}".format(K, iK+1, len(Ks)))
            self.Ky = K
            self.reset()
            I0 = self.intensities_on_mesh(energy, theta, psi, harmonics)[0]
            if self.distE == 'BW':
                I0 *= 1e3
            else:  # 'eV'
                I0 *= energy[:, np.newaxis, np.newaxis, np.newaxis]
            power = I0.sum() * dtheta * dpsi * dE * EV2ERG * 1e-7  # [W]
            powers.append(power)
        self.Ky = tmpKy
        self.reset()
        return np.array(powers)

    def multi_electron_stack(self, energy='auto', theta='auto', psi='auto',
                             harmonic=None, withElectronDivergence=True):
        """Returns Es and Ep in the shape (energy, theta, psi, [harmonic]).
        Along the 0th axis (energy) are stored "macro-electrons" that emit at
        the proton energy given by *energy* (constant or variable) onto the
        angular mesh given by *theta* and *psi*. The transverse field from each
        macro-electron gets individual random angular offsets dtheta and dpsi
        within the emittance distribution if *withElectronDivergence* is True
        and an individual random shift to gamma within the energy spread. The
        parameter self.filamentBeam is irrelevant for this method."""
        if isinstance(energy, str):  # i.e. if 'auto'
            energy = np.mgrid[self.E_min:self.E_max + 0.5*self.dE:self.dE]
        nmacroe = 1 if len(np.array(energy).shape) == 0 else len(energy)

        if isinstance(theta, str):
            theta = np.mgrid[
                self.Theta_min:self.Theta_max + 0.5*self.dTheta:self.dTheta]

        if isinstance(psi, str):
            psi = np.mgrid[self.Psi_min:self.Psi_max + 0.5*self.dPsi:self.dPsi]

        if harmonic is None:
            xH = None
            tomesh = energy, theta, psi
        else:
            tomesh = energy, theta, psi, harmonic
        mesh = np.meshgrid(*tomesh, indexing='ij')
        if withElectronDivergence and self.dxprime > 0:
            dthe = np.random.normal(0, self.dxprime, nmacroe)
            if harmonic is None:
                mesh[1][:, ...] += dthe[:, np.newaxis, np.newaxis]
            else:
                mesh[1][:, ...] += dthe[:, np.newaxis, np.newaxis, np.newaxis]
        if withElectronDivergence and self.dzprime > 0:
            dpsi = np.random.normal(0, self.dzprime, nmacroe)
            if harmonic is None:
                mesh[2][:, ...] += dpsi[:, np.newaxis, np.newaxis]
            else:
                mesh[2][:, ...] += dpsi[:, np.newaxis, np.newaxis, np.newaxis]

        if self.eEspread > 0:
            spr = np.random.normal(0, self.eEspread, nmacroe) * self.gamma
            dgamma = np.zeros_like(mesh[0])
            if harmonic is None:
                dgamma[:, ...] = spr[:, np.newaxis, np.newaxis]
            else:
                dgamma[:, ...] = spr[:, np.newaxis, np.newaxis, np.newaxis]
            xdGamma = dgamma.ravel()
        else:
            xdGamma = 0

        xE, xTheta, xPsi = mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()
        if harmonic is not None:
            xH = mesh[3].ravel()

        if harmonic is None:
            sh = nmacroe, len(theta), len(psi)
        else:
            sh = nmacroe, len(theta), len(psi), len(harmonic)
        res = self.build_I_map(xE, xTheta, xPsi, xH, xdGamma)
        Es = res[1].reshape(sh)
        Ep = res[2].reshape(sh)
        return Es, Ep

    def intensities_on_mesh(self, energy='auto', theta='auto', psi='auto',
                            harmonic=None):
        if isinstance(energy, str):  # i.e. if 'auto'
            energy = np.mgrid[self.E_min:self.E_max + 0.5*self.dE:self.dE]

        if isinstance(theta, str):
            theta = np.mgrid[
                self.Theta_min:self.Theta_max + 0.5*self.dTheta:self.dTheta]

        if isinstance(psi, str):
            psi = np.mgrid[self.Psi_min:self.Psi_max + 0.5*self.dPsi:self.dPsi]

        tomesh = [energy, theta, psi]
        if harmonic is not None:
            tomesh.append(harmonic)
            iharmonic = len(tomesh)-1
        else:
            iharmonic = None
        if self.eEspread > 0:
            spr = np.linspace(-3, 3, 13)
            dgamma = self.gamma * spr * self.eEspread
            wspr = np.exp(-0.5 * spr**2)
            wspr /= wspr.sum()
            tomesh.append(dgamma)
            ispread = len(tomesh)-1
        else:
            ispread = None

        mesh = np.meshgrid(*tomesh, indexing='ij')
        xE, xTheta, xPsi = mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()
        sh = [len(energy), len(theta), len(psi)]
        if iharmonic:
            xH = mesh[iharmonic].ravel()
            sh.append(len(harmonic))
        else:
            xH = None
        if ispread:
            xG = mesh[ispread].ravel()
            sh.append(len(dgamma))
        else:
            xG = None

        res = self.build_I_map(xE, xTheta, xPsi, xH, xG)
        Es = res[1].reshape(sh)
        Ep = res[2].reshape(sh)
        if ispread:
            if iharmonic:
                ws = wspr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
            else:
                ws = wspr[np.newaxis, np.newaxis, np.newaxis, :]
            Is = ((Es*np.conj(Es)).real * ws).sum(axis=ispread)
            Ip = ((Ep*np.conj(Ep)).real * ws).sum(axis=ispread)
            Isp = (Es*np.conj(Ep) * ws).sum(axis=ispread)
        else:
            Is = (Es*np.conj(Es)).real
            Ip = (Ep*np.conj(Ep)).real
            Isp = Es*np.conj(Ep)
        self.Is = Is.astype(float)
        self.Ip = Ip.astype(float)
        self.Isp = Isp.astype(complex)

        s0 = self.Is + self.Ip
        s1 = self.Is - self.Ip
        s2 = 2. * np.real(self.Isp)
        s3 = -2. * np.imag(self.Isp)

        if self.dxprime > 0 or self.dzprime > 0:
            from scipy.ndimage.filters import gaussian_filter
            Sx = self.dxprime / (theta[1] - theta[0])
            Sz = self.dzprime / (psi[1] - psi[0])
            for ie, ee in enumerate(energy):
                if harmonic is None:
                    s0[ie, :, :] = gaussian_filter(s0[ie, :, :], [Sx, Sz])
                    s1[ie, :, :] = gaussian_filter(s1[ie, :, :], [Sx, Sz])
                    s2[ie, :, :] = gaussian_filter(s2[ie, :, :], [Sx, Sz])
                    s3[ie, :, :] = gaussian_filter(s3[ie, :, :], [Sx, Sz])
                else:
                    for ih, hh in enumerate(harmonic):
                        s0[ie, :, :, ih] = gaussian_filter(
                            s0[ie, :, :, ih], [Sx, Sz])
                        s1[ie, :, :, ih] = gaussian_filter(
                            s1[ie, :, :, ih], [Sx, Sz])
                        s2[ie, :, :, ih] = gaussian_filter(
                            s2[ie, :, :, ih], [Sx, Sz])
                        s3[ie, :, :, ih] = gaussian_filter(
                            s3[ie, :, :, ih], [Sx, Sz])

        with np.errstate(divide='ignore'):
            return (s0,
                    np.where(s0, s1 / s0, s0),
                    np.where(s0, s2 / s0, s0),
                    np.where(s0, s3 / s0, s0))

    def _sp(self, dim, x, ww1, w, wu, gamma, ddphi, ddpsi):
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
        elif dim == 3:
            ww1S = ww1[:, :, :, np.newaxis]
            wS, wuS = w[:, :, :, np.newaxis], wu[:, :, :, np.newaxis]
            ddphiS = ddphi[:, :, :, np.newaxis]
            ddpsiS = ddpsi[:, :, :, np.newaxis]
            if lengamma > 1:
                gS = gamma[:, :, :, np.newaxis]
        taperC = 1
        alphaS = 0
        sinx = np.sin(x)
        cosx = np.cos(x)
        sin2x = 2*sinx*cosx
        if self.taper is not None:
            alphaS = self.taper * C * 10 / E2W
            taperC = 1 - alphaS * x / wuS
            ucos = ww1S * x +\
                wS / gS / wuS *\
                (-self.Ky * ddphiS * (sinx + alphaS / wuS *
                                      (1 - cosx - x * sinx)) +
                 self.Kx * ddpsiS * np.sin(x + self.phase) +
                 0.125 / gS *
                 (self.Kx**2 * np.sin(2 * (x + self.phase)) +
                 self.Ky**2 * (sin2x - 2 * alphaS / wuS *
                               (x**2 + cosx**2 + x * sin2x))))
        elif self.R0 is not None:
            betam = 1 - (1 + 0.5 * self.Kx**2 + 0.5 * self.Ky**2) / 2. / gS**2
            WR0 = self.R0 / 10 / C * E2W
            ddphiS = -ddphiS
            drx = WR0 * np.tan(ddphiS) - self.Ky / wuS / gS * sinx
            dry = WR0 * np.tan(ddpsiS) + self.Kx / wuS / gS * np.sin(
                x + self.phase)
            drz = WR0 * np.cos(np.sqrt(ddphiS**2+ddpsiS**2)) -\
                betam * x / wuS + 0.125 / wuS / gS**2 *\
                (self.Ky**2 * sin2x +
                 self.Kx**2 * np.sin(2 * (x + self.phase)))
            ucos = wS * (x / wuS + np.sqrt(drx**2 + dry**2 + drz**2))
        else:
            ucos = ww1S * x + wS / gS / wuS *\
                (-self.Ky * ddphiS * sinx +
                 self.Kx * ddpsiS * np.sin(x + self.phase) +
                 0.125 / gS * (self.Ky**2 * sin2x +
                               self.Kx**2 * np.sin(2. * (x + self.phase))))

        nz = 1 - 0.5*(ddphiS**2 + ddpsiS**2)
        betax = taperC * self.Ky / gS * cosx
        betay = -self.Kx / gS * np.cos(x + self.phase)
        betaz = 1 - 0.5*(1./gS**2 + betax**2 + betay**2)

        betaPx = -wuS * self.Ky / gS * (alphaS * cosx + taperC * sinx)
        betaPy = wuS * self.Kx / gS * np.sin(x + self.phase)
        betaPz = 0.5 * wuS / gS**2 *\
            (self.Ky**2 * taperC * (alphaS*cosx**2 + taperC * sin2x) +
             self.Kx**2 * np.sin(2. * (x + self.phase)))
        krel = 1. - ddphiS*betax - ddpsiS*betay - nz*betaz
        eucos = np.exp(1j * ucos) / krel**2

        bnx = betax - ddphiS
        bny = betay - ddpsiS
        bnz = betaz - nz
        primexy = betaPx*bny - betaPy*bnx

        return ((nz*(betaPx*bnz - betaPz*bnx) + ddpsiS*primexy) * eucos,
                (nz*(betaPy*bnz - betaPz*bny) - ddphiS*primexy) * eucos)

    def build_I_map(self, w, ddtheta, ddpsi, harmonic=None, dg=None):
        useCL = False
        if isinstance(w, np.ndarray):
            if w.shape[0] > 32:
                useCL = True
        if (self.cl_ctx is None) or not useCL:
            return self._build_I_map_conv(w, ddtheta, ddpsi, harmonic, dg)
        elif self.customField is not None:
            return self._build_I_map_custom(w, ddtheta, ddpsi, harmonic, dg)
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

        wu = PI * C * 10 / self.L0 / gamma2 * np.ones_like(w) *\
            (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2W
        ww1 = w * ((1. + 0.5*self.Kx**2 + 0.5*self.Ky**2) +
                   gamma2 * (ddtheta**2 + ddpsi**2)) / (2. * gamma2 * wu)
        tg_n, ag_n = np.polynomial.legendre.leggauss(self.quadm)
        self.tg_n, self.ag_n = tg_n, ag_n

        if (self.taper is not None) or (self.R0 is not None):
            AB = 1. / PI2 / wu
            dstep = 2 * PI / float(self.gIntervals)
            dI = np.arange(0.5 * dstep - PI * self.Np, PI * self.Np, dstep)
        else:
            AB = 1. / PI2 / wu * np.sin(PI * self.Np * ww1) / np.sin(PI * ww1)
            dstep = 2 * PI / float(self.gIntervals)
            dI = np.arange(-PI + 0.5 * dstep, PI, dstep)

        tg = (dI[:, None] + 0.5*dstep*tg_n).ravel()  # + PI/2
        ag = (dI[:, None]*0 + ag_n).ravel()
        # Bsr = np.zeros_like(w, dtype='complex')
        # Bpr = np.zeros_like(w, dtype='complex')
        dim = len(np.array(w).shape)
        sp3res = self._sp(dim, tg, ww1, w, wu, gamma, ddtheta, ddpsi)
        Bsr = np.sum(ag * sp3res[0], axis=dim)
        Bpr = np.sum(ag * sp3res[1], axis=dim)

        bwFact = 0.001 if self.distE == 'BW' else 1./w
        Amp2Flux = FINE_STR * bwFact * self.I0 / SIE0

        if harmonic is not None:
            Bsr[ww1 > harmonic+0.5] = 0
            Bpr[ww1 > harmonic+0.5] = 0
            Bsr[ww1 < harmonic-0.5] = 0
            Bpr[ww1 < harmonic-0.5] = 0

        #        np.seterr(invalid='warn')
        #        np.seterr(divide='warn')
        return (Amp2Flux * AB**2 * 0.25 * dstep**2 *
                (np.abs(Bsr)**2 + np.abs(Bpr)**2),
                np.sqrt(Amp2Flux) * AB * Bsr,
                np.sqrt(Amp2Flux) * AB * Bpr)

    def _build_I_map_custom(self, w, ddtheta, ddpsi, harmonic, dgamma=None):
        # time1 = time.time()
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

        wu = PI * C * 10 / self.L0 / gamma2 *\
            (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2W
        ww1 = w * ((1. + 0.5 * self.Kx**2 + 0.5 * self.Ky**2) +
                   gamma2 * (ddtheta * ddtheta + ddpsi * ddpsi)) /\
            (2. * gamma2 * wu)
        scalarArgs = []  # R0
        R0 = self.R0 if self.R0 is not None else 0

        Np = np.int32(self.Np)

        tg_n, ag_n = np.polynomial.legendre.leggauss(self.quadm)
        self.tg_n, self.ag_n = tg_n, ag_n

        ab = 1. / PI2 / wu
        dstep = 2 * PI / float(self.gIntervals)
        dI = np.arange(0.5 * dstep - PI * Np, PI * Np, dstep)

        tg = np.array([-PI*Np + PI/2.])
        ag = [0]
        tg = self.cl_precisionF(
            np.concatenate((tg, (dI[:, None]+0.5*dstep*tg_n).ravel() + PI/2.)))
        ag = self.cl_precisionF(np.concatenate(
            (ag, (dI[:, None]*0+ag_n).ravel())))

        nwt = self.nRK
        wtGrid = []
        for itg in range(len(tg) - 1):
            tmppr, tmpstp = np.linspace(tg[itg],
                                        tg[itg+1],
                                        2*nwt,
                                        endpoint=False, retstep=True)
            wtGrid.extend(np.linspace(tg[itg],
                                      tg[itg+1],
                                      2*nwt,
                                      endpoint=False))
        wtGrid.append(tg[-1])

        # print("Custom magnetic field: Bx={0}. By={1}, Bz={2}".format(
        #    self.B0x, self.B0y, self.B0z))
        Bx, By, Bz = self.magnetic_field(wtGrid)

        if self.filamentBeam:
            emcg = self.L0 * SIE0 / SIM0 / C / 10. / gamma[0] / PI2
            scalarArgsTraj = [np.int32(len(tg)),  # jend
                              np.int32(nwt),
                              self.cl_precisionF(emcg),
                              self.cl_precisionF(gamma[0])]

            nonSlicedROArgs = [tg,  # Gauss-Legendre grid
                               self.cl_precisionF(Bx),  # Mangetic field
                               self.cl_precisionF(By),  # components on the
                               self.cl_precisionF(Bz)]  # Runge-Kutta grid

            nonSlicedRWArgs = [np.zeros_like(tg),  # beta.x
                               np.zeros_like(tg),  # beta.y
                               np.zeros_like(tg),  # beta.z average
                               np.zeros_like(tg),  # traj.x
                               np.zeros_like(tg),  # traj.y
                               np.zeros_like(tg)]  # traj.z

            clKernel = 'get_trajectory'

            betax, betay, betazav, trajx, trajy, trajz = self.ucl.run_parallel(
                clKernel, scalarArgsTraj, None, nonSlicedROArgs,
                None, nonSlicedRWArgs, 1)
            self.beta = [betax, betay]
            self.trajectory = [trajx[1:-1] * self.L0 / PI2,
                               trajy[1:-1] * self.L0 / PI2,
                               trajz[1:-1] * self.L0 / PI2]
            wuAv = PI2 * C * 10. * betazav[-1] / self.L0 / E2W

            scalarArgsTest = [np.int32(len(tg)),
                              np.int32(nwt),
                              self.cl_precisionF(emcg),
                              self.cl_precisionF(gamma[0]**2),
                              self.cl_precisionF(wuAv),
                              self.cl_precisionF(self.L0),
                              self.cl_precisionF(R0)]

            slicedROArgs = [self.cl_precisionF(w),  # Energy
                            self.cl_precisionF(ddtheta),  # Theta
                            self.cl_precisionF(ddpsi)]  # Psi

            nonSlicedROArgs = [tg,  # Gauss-Legendre grid
                               ag,   # Gauss-Legendre weights
                               self.cl_precisionF(Bx),  # Mangetic field
                               self.cl_precisionF(By),  # components on the
                               self.cl_precisionF(Bz),  # Runge-Kutta grid
                               self.cl_precisionF(betax),  # Components of the
                               self.cl_precisionF(betay),  # velosity and
                               self.cl_precisionF(trajx),  # trajectory of the
                               self.cl_precisionF(trajy),  # electron on the
                               self.cl_precisionF(trajz)]  # Gauss grid

            slicedRWArgs = [np.zeros(NRAYS, dtype=self.cl_precisionC),  # Is
                            np.zeros(NRAYS, dtype=self.cl_precisionC)]  # Ip

            clKernel = 'undulator_custom_filament'

            Is_local, Ip_local = self.ucl.run_parallel(
                clKernel, scalarArgsTest, slicedROArgs, nonSlicedROArgs,
                slicedRWArgs, None, NRAYS)
        else:
            scalarArgs.extend([np.int32(len(tg)),  # jend
                               np.int32(nwt),
                               self.cl_precisionF(self.L0)])

            slicedROArgs = [self.cl_precisionF(gamma),  # gamma
                            self.cl_precisionF(w),  # Energy
                            self.cl_precisionF(ddtheta),  # Theta
                            self.cl_precisionF(ddpsi)]  # Psi

            nonSlicedROArgs = [tg,  # Gauss-Legendre grid
                               ag,   # Gauss-Legendre weights
                               self.cl_precisionF(Bx),  # Mangetic field
                               self.cl_precisionF(By),  # components on the
                               self.cl_precisionF(Bz)]  # Runge-Kutta grid

            slicedRWArgs = [np.zeros(NRAYS, dtype=self.cl_precisionC),  # Is
                            np.zeros(NRAYS, dtype=self.cl_precisionC)]  # Ip

            clKernel = 'undulator_custom'

            Is_local, Ip_local = self.ucl.run_parallel(
                clKernel, scalarArgs, slicedROArgs, nonSlicedROArgs,
                slicedRWArgs, None, NRAYS)

        bwFact = 0.001 if self.distE == 'BW' else 1./w
        Amp2Flux = FINE_STR * bwFact * self.I0 / SIE0

        if harmonic is not None:
            Is_local[ww1 > harmonic+0.5] = 0
            Ip_local[ww1 > harmonic+0.5] = 0
            Is_local[ww1 < harmonic-0.5] = 0
            Ip_local[ww1 < harmonic-0.5] = 0

        return (Amp2Flux * ab**2 * 0.25 * dstep**2 *
                (np.abs(Is_local)**2 + np.abs(Ip_local)**2),
                np.sqrt(Amp2Flux) * Is_local * ab * 0.5 * dstep,
                np.sqrt(Amp2Flux) * Ip_local * ab * 0.5 * dstep)

    def _build_I_map_CL(self, w, ddtheta, ddpsi, harmonic, dgamma=None):
        # time1 = time.time()
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

        wu = PI * C * 10 / self.L0 / gamma2 *\
            (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2W
        ww1 = w * ((1. + 0.5 * self.Kx**2 + 0.5 * self.Ky**2) +
                   gamma2 * (ddtheta * ddtheta + ddpsi * ddpsi)) /\
            (2. * gamma2 * wu)
        scalarArgs = [self.cl_precisionF(0.)]

        if self.R0 is not None:
            scalarArgs = [self.cl_precisionF(self.R0),  # R0
                          self.cl_precisionF(self.L0)]
        elif self.taper:
            scalarArgs = [self.cl_precisionF(self.taper)]

        Np = np.int32(self.Np)

        tg_n, ag_n = np.polynomial.legendre.leggauss(self.quadm)
        self.tg_n, self.ag_n = tg_n, ag_n

        dstep = 2 * PI / float(self.gIntervals)
        if (self.taper is not None) or (self.R0 is not None) or self.full:
            ab = 1. / PI2 / wu
            dI = np.arange(0.5 * dstep - PI * Np, PI * Np, dstep)
        else:
            ab = 1. / PI2 / wu * np.sin(PI * Np * ww1) / np.sin(PI * ww1)
            dI = np.arange(-PI + 0.5*dstep, PI, dstep)

        extra = PI/2*0
        tg = self.cl_precisionF((dI[:, None]+0.5*dstep*tg_n).ravel()) + extra
        ag = self.cl_precisionF((dI[:, None]*0+ag_n).ravel())

        scalarArgs.extend([self.cl_precisionF(self.Kx),  # Kx
                           self.cl_precisionF(self.Ky),  # Ky
                           self.cl_precisionF(self.phase),  # phase
                           np.int32(len(tg))])  # jend

        slicedROArgs = [self.cl_precisionF(gamma),  # gamma
                        self.cl_precisionF(wu),  # Eund
                        self.cl_precisionF(w),  # Energy
                        self.cl_precisionF(ww1),  # Energy/Eund(0)
                        self.cl_precisionF(ddtheta),  # Theta
                        self.cl_precisionF(ddpsi)]  # Psi
        if self.full:
            if isinstance(self.theta0, np.ndarray):
                slicedROArgs.extend([self.cl_precisionF(self.theta0),
                                     self.cl_precisionF(self.psi0)])
            else:
                slicedROArgs.extend([self.cl_precisionF(
                                        self.theta0*np.ones_like(w)),
                                     self.cl_precisionF(
                                        self.psi0*np.ones_like(w))])

        nonSlicedROArgs = [tg,  # Gauss-Legendre grid
                           ag]  # Gauss-Legendre weights

        slicedRWArgs = [np.zeros(NRAYS, dtype=self.cl_precisionC),  # Is
                        np.zeros(NRAYS, dtype=self.cl_precisionC)]  # Ip

        if self.taper is not None:
            clKernel = 'undulator_taper'
        elif self.R0 is not None:
            clKernel = 'undulator_nf'
            if self.full:
                clKernel = 'undulator_nf_full'
        elif self.full:
            clKernel = 'undulator_full'
        else:
            clKernel = 'undulator'

        Is_local, Ip_local = self.ucl.run_parallel(
            clKernel, scalarArgs, slicedROArgs, nonSlicedROArgs,
            slicedRWArgs, dimension=NRAYS)

        bwFact = 0.001 if self.distE == 'BW' else 1./w
        Amp2Flux = FINE_STR * bwFact * self.I0 / SIE0

        if harmonic is not None:
            Is_local[ww1 > harmonic+0.5] = 0
            Ip_local[ww1 > harmonic+0.5] = 0
            Is_local[ww1 < harmonic-0.5] = 0
            Ip_local[ww1 < harmonic-0.5] = 0

        # print("Build_I_Map completed in {0} s".format(time.time() - time1))
        return (Amp2Flux * ab**2 * 0.25 * dstep**2 *
                (np.abs(Is_local)**2 + np.abs(Ip_local)**2),
                np.sqrt(Amp2Flux) * Is_local * ab * 0.5 * dstep,
                np.sqrt(Amp2Flux) * Ip_local * ab * 0.5 * dstep)

#    def _reportNaN(self, x, strName):
#        nanSum = np.isnan(x).sum()
#        if nanSum > 0:
#            print("{0} NaN rays in {1}!".format(nanSum, strName))

    def real_photon_source_sizes(
            self, energy='auto', theta='auto', psi='auto'):
        """Returns energy dependent arrays: flux, (dx')², (dz')², dx², dz².
        Depending on *distE* being 'eV' or 'BW', the flux is either in ph/s or
        in ph/s/0.1%BW, being integrated over the specified theta and psi
        ranges. The squared angular and linear photon source sizes are
        variances, i.e. squared sigmas. The latter two (linear sizes) are in
        mm**2.
        """
        if isinstance(energy, str):  # i.e. if 'auto'
            energy = np.mgrid[self.E_min:self.E_max + 0.5*self.dE:self.dE]

        if isinstance(theta, str):
            theta = np.mgrid[
                self.Theta_min:self.Theta_max + 0.5*self.dTheta:self.dTheta]

        if isinstance(psi, str):
            psi = np.mgrid[self.Psi_min:self.Psi_max + 0.5*self.dPsi:self.dPsi]

        tomesh = [energy, theta, psi]
        sh = [len(energy), len(theta), len(psi)]
        if self.eEspread > 0:
            spr = np.linspace(-3, 3, 13)
            dgamma = self.gamma * spr * self.eEspread
            wspr = np.exp(-0.5 * spr**2)
            wspr /= wspr.sum()
            tomesh.append(dgamma)
            sh.append(len(dgamma))

        mesh = np.meshgrid(*tomesh, indexing='ij')
        xE, xTheta, xPsi = mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()
        xG = mesh[3].ravel() if self.eEspread > 0 else None

        res = self.build_I_map(xE, xTheta, xPsi, dg=xG)
        Es = res[1].reshape(sh)
        Ep = res[2].reshape(sh)
        if self.eEspread > 0:
            ws = wspr[np.newaxis, np.newaxis, np.newaxis, :]
            Is = ((Es*np.conj(Es)).real * ws).sum(axis=3)
            Ip = ((Ep*np.conj(Ep)).real * ws).sum(axis=3)
        else:
            Is = (Es*np.conj(Es)).real
            Ip = (Ep*np.conj(Ep)).real
        I0 = (Is.astype(float) + Ip.astype(float)) *\
            (theta.max() - theta.min()) * (psi.max() - psi.min())
        dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
        flux = I0.sum(axis=(1, 2))
        theta2 = (I0 * (theta**2)[np.newaxis, :, np.newaxis]).sum(
            axis=(1, 2)) / flux
        psi2 = (I0 * (psi**2)[np.newaxis, np.newaxis, :]).sum(
            axis=(1, 2)) / flux

        EsFT = np.fft.fftshift(np.fft.fft2(Es, axes=(1, 2))) * dtheta * dpsi
        EpFT = np.fft.fftshift(np.fft.fft2(Ep, axes=(1, 2))) * dtheta * dpsi
        thetaFT = np.fft.fftshift(np.fft.fftfreq(len(theta), dtheta))
        psiFT = np.fft.fftshift(np.fft.fftfreq(len(psi), dpsi))
        if self.eEspread > 0:
            ws = wspr[np.newaxis, np.newaxis, np.newaxis, :]
            IsFT = ((EsFT*np.conj(EsFT)).real * ws).sum(axis=3)
            IpFT = ((EpFT*np.conj(EpFT)).real * ws).sum(axis=3)
        else:
            IsFT = (EsFT*np.conj(EsFT)).real
            IpFT = (EpFT*np.conj(EpFT)).real
        I0FT = (IsFT.astype(float) + IpFT.astype(float)) *\
            (thetaFT.max() - thetaFT.min()) * (psiFT.max() - psiFT.min())
#        fluxFT = I0FT.sum(axis=(1, 2))  # it is equal to flux, checked
        k = energy / CHBAR * 1e7  # in 1/mm
        dx2 = (I0FT * (thetaFT**2)[np.newaxis, :, np.newaxis]).sum(
            axis=(1, 2)) / flux * k**(-2)
        dz2 = (I0FT * (psiFT**2)[np.newaxis, np.newaxis, :]).sum(
            axis=(1, 2)) / flux * k**(-2)

        return flux, theta2, psi2, dx2, dz2

    def tanaka_kitamura_Qa2(self, x, eps=1e-6):
        """Squared Q_a function from Tanaka and Kitamura J. Synchrotron Rad. 16
        (2009) 380–386, Eq(17). The argument is normalized energy spread by
        Eq(13)."""
        ret = np.ones_like(x, dtype=float)
        xarr = np.array(x)
#        ret[x <= eps] = 1  # ret already holds ones
        y = SQ2 * xarr[xarr > eps]
        y2 = y**2
        ret[x > eps] = y2 / (np.exp(-y2) + SQPI*y*special.erf(y) - 1)
        return ret

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
        harmonic[harmonic < 1] = 1
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
        harmonic[harmonic < 1] = 1
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

    def shine(self, toGlobal=True, withAmplitudes=True, fixedEnergy=False,
              wave=None, accuBeam=None):
        u"""
        Returns the source beam. If *toGlobal* is True, the output is in
        the global system. If *withAmplitudes* is True, the resulted beam
        contains arrays Es and Ep with the *s* and *p* components of the
        electric field.

        *fixedEnergy* is either None or a value in eV. If *fixedEnergy* is
        specified, the energy band is not 0.1%BW relative to *fixedEnergy*, as
        probably axpected but is given by (eMax - eMin) of the constructor.

        *wave* and *accuBeam* are used in wave diffraction. *wave* is a Beam
        object and determines the positions of the wave samples. It must be
        obtained by a previous `prepare_wave` run. *accuBeam* is only needed
        with *several* repeats of diffraction integrals when the parameters of
        the filament beam must be preserved for all the repeats.


        .. Returned values: beamGlobal
        """
        if self.bl is not None:
            try:
                self.bl._alignE = float(self.bl.alignE)
            except ValueError:
                self.bl._alignE = 0.5 * (self.eMin + self.eMax)

        if wave is not None:
            if not hasattr(wave, 'rDiffr'):
                raise ValueError("If you want to use a `wave`, run a" +
                                 " `prepare_wave` before shine!")
            self.uniformRayDensity = True
            mcRays = len(wave.a)
        else:
            mcRays = self.nrays

        if self.uniformRayDensity:
            withAmplitudes = True
        if not self.uniformRayDensity:
            if raycing._VERBOSITY_ > 0:
                print("Rays generation")
        bo = None
        length = 0
        seeded = np.long(0)
        seededI = 0.
        np.seterr(invalid='warn')
        np.seterr(divide='warn')
        if self.filamentBeam:
            if accuBeam is None:
                rsE = np.random.random_sample() * \
                    float(self.E_max - self.E_min) + self.E_min
                rX = self.dx * np.random.standard_normal()
                rZ = self.dz * np.random.standard_normal()
                dtheta = self.dxprime * np.random.standard_normal()
                dpsi = self.dzprime * np.random.standard_normal()
            else:
                rsE = accuBeam.E[0]
                rX = accuBeam.filamentDX
                rZ = accuBeam.filamentDZ
                dtheta = accuBeam.filamentDtheta
                dpsi = accuBeam.filamentDpsi
                seeded = accuBeam.seeded
                seededI = accuBeam.seededI
        if self.full:
            if self.filamentBeam:
                self.theta0 = dtheta
                self.psi0 = dpsi
            else:
                self.theta0 = np.random.normal(0, self.dxprime, mcRays)
                self.psi0 = np.random.normal(0, self.dzprime, mcRays)

        if fixedEnergy:
            rsE = fixedEnergy
            if (self.E_max-self.E_min) > fixedEnergy*1.1e-3:
                print("Warning: the bandwidth seem too big. "
                      "Specify it by giving eMin and eMax in the constructor.")
        nrep = 0
        rep_condition = True
#        while length < self.nrays:
        while rep_condition:
            seeded += mcRays
            # start_time = time.time()
            if self.filamentBeam or fixedEnergy:
                rE = rsE * np.ones(mcRays)
            else:
                rndg = np.random.rand(mcRays)
                rE = rndg * float(self.E_max - self.E_min) + self.E_min

            if wave is not None:
                self.xzE = (self.E_max - self.E_min)
                if self.filamentBeam:
                    shiftX = rX
                    shiftZ = rZ
                else:
                    shiftX = np.random.normal(
                        0, self.dx, mcRays) if self.dx > 0 else 0
                    shiftZ = np.random.normal(
                        0, self.dz, mcRays) if self.dz > 0 else 0
                x = wave.xDiffr + shiftX
                y = wave.yDiffr
                z = wave.zDiffr + shiftZ
                rDiffr = (x**2 + y**2 + z**2)**0.5
                rTheta = x / rDiffr
                rPsi = z / rDiffr
                if self.filamentBeam:
                    rTheta += dtheta
                    rPsi += dpsi
                else:
                    if self.dxprime > 0:
                        rTheta += np.random.normal(0, self.dxprime, mcRays)
                    if self.dzprime > 0:
                        rPsi += np.random.normal(0, self.dzprime, mcRays)
            else:
                rndg = np.random.rand(mcRays)
                rTheta = rndg * (self.Theta_max - self.Theta_min) +\
                    self.Theta_min
                rndg = np.random.rand(mcRays)
                rPsi = rndg * (self.Psi_max - self.Psi_min) + self.Psi_min

            Intensity, mJs, mJp = self.build_I_map(rE, rTheta, rPsi)

            if self.uniformRayDensity:
                seededI += mcRays * self.xzE
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
                rndg = np.random.rand(mcRays)
                I_pass = np.where(self.Imax * rndg < Intensity)[0]
                npassed = len(I_pass)
            if npassed == 0:
                if raycing._VERBOSITY_ > 0:
                    print('No good rays in this seed!', length, 'of',
                          self.nrays, 'rays in total so far...')
                    print(self.Imax, self.E_min, self.E_max,
                          self.Theta_min, self.Theta_max,
                          self.Psi_min, self.Psi_max)
                continue

            if wave is not None:
                bot = wave
            else:
                bot = Beam(npassed, withAmplitudes=withAmplitudes)
            bot.state[:] = 1  # good
            bot.E[:] = rE[I_pass]

            if self.filamentBeam:
                dxR = rX
                dzR = rZ
#                sigma_r2 = self.get_sigma_r2(bot.E)
#                dxR += np.random.normal(0, sigma_r2**0.5, npassed)
#                dzR += np.random.normal(0, sigma_r2**0.5, npassed)
            else:
                if self.full:
                    bot.sourceSIGMAx = self.dx
                    bot.sourceSIGMAz = self.dz
                    dxR = np.random.normal(0, bot.sourceSIGMAx, npassed)
                    dzR = np.random.normal(0, bot.sourceSIGMAz, npassed)
                else:
                    bot.sourceSIGMAx, bot.sourceSIGMAz = self.get_SIGMA(
                        bot.E, onlyOddHarmonics=False)
                    dxR = np.random.normal(0, bot.sourceSIGMAx, npassed)
                    dzR = np.random.normal(0, bot.sourceSIGMAz, npassed)

            if wave is not None:
                wave.rDiffr = ((wave.xDiffr - dxR)**2 + wave.yDiffr**2 +
                               (wave.zDiffr - dzR)**2)**0.5
                wave.path[:] = 0
                wave.a[:] = (wave.xDiffr - dxR) / wave.rDiffr
                wave.b[:] = wave.yDiffr / wave.rDiffr
                wave.c[:] = (wave.zDiffr - dzR) / wave.rDiffr
            else:
                bot.x[:] = dxR
                bot.z[:] = dzR
                bot.a[:] = rTheta[I_pass]
                bot.c[:] = rPsi[I_pass]

                if not self.full:
                    if self.filamentBeam:
                        bot.a[:] += dtheta
                        bot.c[:] += dpsi
                    else:
                        if self.dxprime > 0:
                            bot.a[:] += np.random.normal(
                                0, self.dxprime, npassed)
                        if self.dzprime > 0:
                            bot.c[:] += np.random.normal(
                                0, self.dzprime, npassed)

            mJs = mJs[I_pass]
            mJp = mJp[I_pass]
            if wave is not None:
                area = wave.areaNormal if hasattr(wave, 'areaNormal') else\
                    wave.area
                norm = area**0.5 / wave.rDiffr
                mJs *= norm
                mJp *= norm
            mJs2 = (mJs * np.conj(mJs)).real
            mJp2 = (mJp * np.conj(mJp)).real

            if self.uniformRayDensity:
                sSP = 1.
            else:
                sSP = mJs2 + mJp2

            bot.Jsp[:] = np.where(sSP, mJs * np.conj(mJp) / sSP, 0)
            bot.Jss[:] = np.where(sSP, mJs2 / sSP, 0)
            bot.Jpp[:] = np.where(sSP, mJp2 / sSP, 0)

            if withAmplitudes:
                bot.Es[:] = mJs
                bot.Ep[:] = mJp

            if bo is None:
                bo = bot
            else:
                bo.concatenate(bot)
            length = len(bo.a)
            if not self.uniformRayDensity:
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
                    except:
                        pass
            if self.filamentBeam:
                nrep += 1
                rep_condition = nrep < self.nrepmax
            else:
                rep_condition = length < self.nrays
            if self.uniformRayDensity:
                rep_condition = False

            bo.accepted = length * self.fluxConst
            bo.acceptedE = bo.E.sum() * self.fluxConst * SIE0
            bo.seeded = seeded
            bo.seededI = seededI
            if raycing._VERBOSITY_ > 0:
                sys.stdout.flush()

        if length > self.nrays and not self.filamentBeam and wave is None:
            bo.filter_by_index(slice(0, self.nrays))
        if self.filamentBeam:
            bo.filamentDtheta = dtheta
            bo.filamentDpsi = dpsi
            bo.filamentDX = rX
            bo.filamentDZ = rZ

        norm = (bo.a**2 + bo.b**2 + bo.c**2)**0.5
        bo.a /= norm
        bo.b /= norm
        bo.c /= norm

#        if raycing._VERBOSITY_ > 10:
#            self._reportNaN(bo.Jss, 'Jss')
#            self._reportNaN(bo.Jpp, 'Jpp')
#            self._reportNaN(bo.Jsp, 'Jsp')
#            self._reportNaN(bo.E, 'E')
#            self._reportNaN(bo.x, 'x')
#            self._reportNaN(bo.y, 'y')
#            self._reportNaN(bo.z, 'z')
#            self._reportNaN(bo.a, 'a')
#            self._reportNaN(bo.b, 'b')
#            self._reportNaN(bo.c, 'c')
        if self.pitch or self.yaw:
            raycing.rotate_beam(bo, pitch=self.pitch, yaw=self.yaw)
        bor = Beam(copyFrom=bo)
        if wave is not None:
            bor.x[:] = dxR
            bor.y[:] = 0
            bor.z[:] = dzR
            bor.path[:] = 0
            mPh = np.exp(1e7j * wave.E/CHBAR * wave.rDiffr)
            wave.Es *= mPh
            wave.Ep *= mPh

        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bor, self.center)
        bor.parentId = self.name
        raycing.append_to_flow(self.shine, [bor],
                               inspect.currentframe())
        return bor
