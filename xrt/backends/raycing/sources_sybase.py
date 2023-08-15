# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "12 Aug 2021"
import os
import sys
import numpy as np
from scipy import special
import inspect
import uuid
from .. import raycing
from . import myopencl as mcl
from .sources_beams import Beam
from .physconsts import C, M0, EV2ERG, SIE0, SQ2, SQPI, CH, CHBAR

# try:
#     import pyopencl as cl  # analysis:ignore
#     isOpenCL = True
#     os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
# except ImportError:
#     isOpenCL = False

if mcl.isOpenCL or mcl.isZMQ:
    isOpenCL = True
else:
    isOpenCL = False
# _DEBUG replaced with raycing._VERBOSITY_


class SourceBase:
    """Base class for the Synchrotron Sources. Not to be called explicitly."""

    hiddenParams = ['eN', 'nx', 'nz']

    def __init__(self, bl=None, name='GenericSource', center=(0, 0, 0),
                 nrays=raycing.nrays,
                 eE=6.0, eI=0.1, eEspread=0., eSigmaX=None, eSigmaZ=None,
                 eEpsilonX=1., eEpsilonZ=0.01, betaX=9., betaZ=2.,
                 eMin=5000., eMax=15000., distE='eV',
                 xPrimeMax=0.5, zPrimeMax=0.5, R0=None,
                 uniformRayDensity=False, filamentBeam=False,
                 pitch=0, yaw=0, eN=51, nx=25, nz=25):
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

        *R0*: float
            Distance center-to-screen for the near field calculations (mm).
            If None, the far field approximation (i.e. "usual" calculations) is
            used.

        *eMin*, *eMax*: float
            Minimum and maximum photon energy (eV). Used as band width for flux
            calculation.

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

        *uniformRayDensity*: bool
            If True, the radiation is sampled uniformly but with varying
            amplitudes, otherwise with the density proportional to intensity
            and with constant amplitudes. Required as True for wave propagation
            calculations. False is usual for ray-tracing.

        *filamentBeam*: bool
            If True the source generates coherent monochromatic wavefronts.
            Required as True for the wave propagation calculations in partially
            coherent regime.

        *pitch*, *yaw*: float
            rotation angles around x and z axis. Useful for canted sources.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.sources:
                bl.sources.append(self)
                self.ordinalNum = len(bl.sources)
        raycing.set_name(self, name)

        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = uuid.uuid4()

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.uuid] = [self, 0]

        self.center = center  # 3D point in global system
        self._pitch = raycing.auto_units_angle(pitch)
        self._yaw = raycing.auto_units_angle(yaw)
        self.nrays = np.int64(nrays)

        self.R0 = R0
        self.distE = distE
        self.uniformRayDensity = uniformRayDensity
        self.filamentBeam = filamentBeam

        self._eE = float(eE)
        self.gamma = self._eE * 1e9 * EV2ERG / (M0 * C**2)
        self.gamma2 = self.gamma**2
        self.eEspread = eEspread
        self.eI = float(eI)

        self._eEpsilonX = eEpsilonX * 1e-6  # input in nmrad
        self._eEpsilonZ = eEpsilonZ * 1e-6  # input in nmrad
        self.dx = eSigmaX * 1e-3 if eSigmaX else None  # input in mkm
        self.dz = eSigmaZ * 1e-3 if eSigmaZ else None  # input in mkm
        self._eMin = float(eMin)
        self._eMax = float(eMax)

        # Beam size and divergence conversion
        if isinstance(xPrimeMax, (tuple, list)):
            # if units are not provided, we expect mrad here
            xPrimeMax = [raycing.auto_units_angle(xPrimeMax[0],
                                                  defaultFactor=1e-3),
                         raycing.auto_units_angle(xPrimeMax[-1],
                                                  defaultFactor=1e-3)]
            self._xPrimeMin, self._xPrimeMax = min(xPrimeMax), max(xPrimeMax)
        elif isinstance(xPrimeMax, raycing.basestring):
            self._xPrimeMax = abs(raycing.auto_units_angle(xPrimeMax))
            self._xPrimeMin = -self._xPrimeMax
        else:
            self._xPrimeMax = abs(xPrimeMax) * 1e-3
            self._xPrimeMin = -self._xPrimeMax

        if isinstance(zPrimeMax, (tuple, list)):
            # if units are not provided, we expect mrad here
            zPrimeMax = [raycing.auto_units_angle(zPrimeMax[0],
                                                  defaultFactor=1e-3),
                         raycing.auto_units_angle(zPrimeMax[-1],
                                                  defaultFactor=1e-3)]
            self._zPrimeMin, self._zPrimeMax = min(zPrimeMax), max(zPrimeMax)
        elif isinstance(zPrimeMax, raycing.basestring):
            self._zPrimeMax = abs(raycing.auto_units_angle(zPrimeMax))
            self._zPrimeMin = -self._zPrimeMax
        else:
            self._zPrimeMax = abs(zPrimeMax) * 1e-3
            self._zPrimeMin = -self._zPrimeMax

        self._betaX = betaX * 1e3 if betaX else None  # input in m
        self._betaZ = betaZ * 1e3 if betaX else None  # input in m
        if (self.dx is not None) and (self._betaX is None):
            self._betaX = self.dx**2 / self._eEpsilonX if self._eEpsilonX\
                else 0.
        if (self.dz is not None) and (self._betaZ is None):
            self._betaZ = self.dz**2 / self._eEpsilonZ if self._eEpsilonZ\
                else 0.

        if (self.dx is None) and (self._betaX is not None):
            self.dx = np.sqrt(self._eEpsilonX*self._betaX)
        elif (self.dx is None) and (self._betaX is None):
            raycing.colorPrint("Set either dx or betaX!", "RED")
        if (self.dz is None) and (self._betaZ is not None):
            self.dz = np.sqrt(self._eEpsilonZ*self._betaZ)
        elif (self.dz is None) and (self._betaZ is None):
            raycing.colorPrint("Set either dz or betaZ!", "RED")

        dxprime, dzprime = None, None
        if dxprime:
            self.dxprime = dxprime
        else:
            self.dxprime = self._eEpsilonX / self.dx if self.dx > 0\
                else 0.  # [rad]
        if dzprime:
            self.dzprime = dzprime
        else:
            self.dzprime = self._eEpsilonZ / self.dz if self.dz > 0\
                else 0.  # [rad]
        if raycing._VERBOSITY_ > 10:
            print('Beam horz. size dx = {0} mm'.format(self.dx))
            print('Beam vert. size dz = {0} mm'.format(self.dz))
            print('Beam horz. diverg. dxprime = {0} rad'.format(self.dxprime))
            print('Beam vert. diverg. dzprime = {0} rad'.format(self.dzprime))

        # Left here for compatibility
        self.eN = eN + 1
        self.nx = 2*nx + 1
        self.nz = 2*nz + 1

        self.needReset = True

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        self._pitch = raycing.auto_units_angle(pitch)

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, yaw):
        self._yaw = raycing.auto_units_angle(yaw)

    @property
    def eSigmaX(self):
        return self.dx * 1e3  # returns in mkm

    @eSigmaX.setter
    def eSigmaX(self, eSigmaX):
        self.dx = eSigmaX * 1e-3  # conversion from mkm to mm

    @property
    def eSigmaZ(self):
        return self.dz * 1e3  # returns in mkm

    @eSigmaZ.setter
    def eSigmaZ(self, eSigmaZ):
        self.dz = eSigmaZ * 1e-3  # conversion from mkm to mm

    @property
    def eEpsilonX(self):
        return self._eEpsilonX * 1e6  # returns in nmrad

    @eEpsilonX.setter
    def eEpsilonX(self, eEpsilonX):
        self._eEpsilonX = eEpsilonX * 1e-6  # conversion from nmrad to mmrad
        self.dx = np.sqrt(self._eEpsilonX * self._betaX)
        self.dxprime = self._eEpsilonX / self.dx if self.dx > 0 else 0

    @property
    def eEpsilonZ(self):
        return self._eEpsilonZ * 1e6  # returns in nmrad

    @eEpsilonZ.setter
    def eEpsilonZ(self, eEpsilonZ):
        self._eEpsilonZ = eEpsilonZ * 1e-6  # conversion from nmrad to mmrad
        self.dz = np.sqrt(self._eEpsilonZ * self._betaZ)
        self.dzprime = self._eEpsilonZ / self.dz if self.dz > 0 else 0

    @property
    def betaX(self):
        return self._betaX * 1e-3  # returns in m

    @betaX.setter
    def betaX(self, betaX):
        self._betaX = betaX * 1e3  # conversion from m to mm
        self.dx = np.sqrt(self._eEpsilonX * self._betaX)
        self.dxprime = self._eEpsilonX / self.dx if self.dx > 0 else 0

    @property
    def betaZ(self):
        return self._betaZ * 1e-3  # returns in m

    @betaZ.setter
    def betaZ(self, betaZ):
        self._betaZ = betaZ * 1e3  # conversion from m to mm
        self.dz = np.sqrt(self._eEpsilonZ * self._betaZ)
        self.dzprime = self._eEpsilonZ / self.dz if self.dz > 0 else 0

    @property
    def eMin(self):
        return self._eMin

    @eMin.setter
    def eMin(self, eMin):
        self._eMin = eMin
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def eMax(self):
        return self._eMax

    @eMax.setter
    def eMax(self, eMax):
        self._eMax = eMax
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def xPrimeMax(self):
        return self._xPrimeMax * 1e3  # return in mrad

    @xPrimeMax.setter
    def xPrimeMax(self, xPrimeMax):
        if isinstance(xPrimeMax, (tuple, list)):
            # if units are not provided, we expect mrad here
            xPrimeMax = [raycing.auto_units_angle(xPrimeMax[0],
                                                  defaultFactor=1e-3),
                         raycing.auto_units_angle(xPrimeMax[-1],
                                                  defaultFactor=1e-3)]
            self._xPrimeMin, self._xPrimeMax = min(xPrimeMax), max(xPrimeMax)
        elif isinstance(xPrimeMax, raycing.basestring):
            self._xPrimeMax = abs(raycing.auto_units_angle(xPrimeMax))
            self._xPrimeMin = -self._xPrimeMax
        else:
            self._xPrimeMax = abs(xPrimeMax) * 1e-3
            self._xPrimeMin = -self._xPrimeMax

    @property
    def zPrimeMax(self):
        return self._zPrimeMax * 1e3  # return in mrad

    @zPrimeMax.setter
    def zPrimeMax(self, zPrimeMax):
        if isinstance(zPrimeMax, (tuple, list)):
            # if units are not provided, we expect mrad here
            zPrimeMax = [raycing.auto_units_angle(zPrimeMax[0],
                                                  defaultFactor=1e-3),
                         raycing.auto_units_angle(zPrimeMax[-1],
                                                  defaultFactor=1e-3)]
            self._zPrimeMin, self._zPrimeMax = min(zPrimeMax), max(zPrimeMax)
        elif isinstance(zPrimeMax, raycing.basestring):
            self._zPrimeMax = abs(raycing.auto_units_angle(zPrimeMax))
            self._zPrimeMin = -self._zPrimeMax
        else:
            self._zPrimeMax = abs(zPrimeMax) * 1e-3
            self._zPrimeMin = -self._zPrimeMax

    @property
    def eE(self):
        return self._eE

    @eE.setter
    def eE(self, eE):
        self._eE = float(eE)
        self.gamma = self._eE * 1e9 * EV2ERG / (M0 * C**2)
        self.gamma2 = self.gamma**2
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def R0(self):
        return self._R0

    @R0.setter
    def R0(self, R0):
        self._R0 = R0
        self.needReset = True
        # Need to recalculate the integration parameters

    def _reset_limits(self):
        if not self._xPrimeMax:
            print("No Theta range specified, using default +/- 1 mrad")
            self._xPrimeMax = 1e-3
            self._xPrimeMin = -1e-3
        if not self._zPrimeMax:
            print("No Psi range specified, using default +/- 1 mrad")
            self._zPrimeMax = 1e-3
            self._zPrimeMin = -1e-3

        # Limits corrected for divergence
#        print(self._xPrimeMax, self.dxprime)
        self.Theta_min = float(self._xPrimeMin-self.dxprime)
        self.Theta_max = float(self._xPrimeMax+self.dxprime)
        self.Psi_min = float(self._zPrimeMin-self.dzprime)
        self.Psi_max = float(self._zPrimeMax+self.dzprime)
        self.E_min = float(min(self.eMin, self.eMax))
        self.E_max = float(max(self.eMin, self.eMax))

        try:  # Left here for compatibility
            self.dE = (self.E_max - self.E_min) / float(self.eN-1)
            self.dTheta = (self.Theta_max - self.Theta_min) / float(self.nx-1)
            self.dPsi = (self.Psi_max - self.Psi_min) / float(self.nz-1)
        except Exception:
            pass

    def _reset_integration_grid(self):
        """To be redefined in the subclass"""
        pass

    def reset(self):
        """This method is invoked after certain changes in the source
        parameters."""

        self.needReset = False
        self._reset_limits()
        self._reset_integration_grid()

        if self.filamentBeam and not hasattr(self, 'dimExy'):
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

    def build_I_map(self):
        """Used to calculate the intensity. To be redefined in the subclass"""
        raise NotImplementedError

    def real_photon_source_sizes(
            self, energy='auto', theta='auto', psi='auto', method='rms'):
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
        dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
        I0 = (Is.astype(float) + Ip.astype(float))
        flux = I0.sum(axis=(1, 2)) * dtheta * dpsi
        theta2, psi2 = self._get_2D_sizes(
            I0, flux, theta, psi, dtheta, dpsi, method)

        EsFT = np.fft.fftshift(np.fft.fft2(Es), axes=(1, 2)) * dtheta * dpsi
        EpFT = np.fft.fftshift(np.fft.fft2(Ep), axes=(1, 2)) * dtheta * dpsi
        thetaFT = np.fft.fftshift(np.fft.fftfreq(len(theta), d=dtheta))
        psiFT = np.fft.fftshift(np.fft.fftfreq(len(psi), d=dpsi))
        dthetaFT, dpsiFT = thetaFT[1] - thetaFT[0], psiFT[1] - psiFT[0]
        if self.eEspread > 0:
            ws = wspr[np.newaxis, np.newaxis, np.newaxis, :]
            IsFT = ((EsFT*np.conj(EsFT)).real * ws).sum(axis=3)
            IpFT = ((EpFT*np.conj(EpFT)).real * ws).sum(axis=3)
        else:
            IsFT = (EsFT*np.conj(EsFT)).real
            IpFT = (EpFT*np.conj(EpFT)).real
        I0FT = (IsFT.astype(float) + IpFT.astype(float))
        fluxFT = I0FT.sum(axis=(1, 2)) * dthetaFT * dpsiFT
        # flux equals fluxFT, check it:
#        print(flux)
#        print(fluxFT)
        k = energy / CH * 1e7  # in 1/mm
        dx2, dz2 = self._get_2D_sizes(
            I0FT, fluxFT, thetaFT, psiFT, dthetaFT, dpsiFT, method, k)

        return flux, theta2, psi2, dx2, dz2

    def _get_2D_sizes(
            self, I0, flux, theta, psi, dtheta, dpsi, method, k=None):
        if method == 'rms':
            theta2 = (I0 * (theta[np.newaxis, :, np.newaxis])**2).sum(
                axis=(1, 2)) * dtheta * dpsi / flux
            psi2 = (I0 * (psi[np.newaxis, np.newaxis, :])**2).sum(
                axis=(1, 2)) * dtheta * dpsi / flux
        elif isinstance(method, float):  # 0 < method < 1
            theta2 = self._get_1D_size(I0, flux, theta, dtheta, 1, method)
            psi2 = self._get_1D_size(I0, flux, psi, dpsi, 2, method)
        else:
            raise ValueError('unknown method!')
        if k is not None:
            theta2 *= k**(-2)
            psi2 *= k**(-2)
        return theta2, psi2

    def _get_1D_size(self, I0, flux, ang, dang, axis, method):
        ang2 = np.zeros(I0.shape[0])
        if axis == 1:
            angCutI0 = I0[:, I0.shape[1]//2:, I0.shape[2]//2].squeeze()
        elif axis == 2:
            angCutI0 = I0[:, I0.shape[1]//2, I0.shape[2]//2:].squeeze()
        angCumFlux = (angCutI0*ang[np.newaxis, len(ang)//2:]).cumsum(axis=1)\
            * 2*np.pi * dang
        for ie, ee in enumerate(flux):
            try:
                argBorder = np.argwhere(angCumFlux[ie, :] > ee*method)[0][0]
            except IndexError:
                ang2[ie] = 0
                continue
            r2a = ang[len(ang)//2+argBorder-1]**2
            va = angCumFlux[ie, argBorder-1]
            r2b = ang[len(ang)//2+argBorder]**2
            vb = angCumFlux[ie, argBorder]
            r2m = (ee*method - va) * (r2b-r2a) / (vb-va) + r2a
            ang2[ie] = r2m
        return ang2

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

    def multi_electron_stack(self, energy='auto', theta='auto', psi='auto',
                             harmonic=None, withElectronDivergence=True):
        """Returns Es and Ep in the shape (energy, theta, psi, [harmonic]).
        Along the 0th axis (energy) are stored "macro-electrons" that emit at
        the photon energy given by *energy* (constant or variable) onto the
        angular mesh given by *theta* and *psi*. The transverse field from each
        macro-electron gets individual random angular offsets dtheta and dpsi
        within the emittance distribution if *withElectronDivergence* is True
        and an individual random shift to gamma within the energy spread.
        The parameter self.filamentBeam is irrelevant for this method."""
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

    def intensities_on_mesh(
            self, energy='auto', theta='auto', psi='auto', harmonic=None,
            eSpreadSigmas=3.5, eSpreadNSamples=36, mode='constant',
            resultKind='Stokes'):
        """Returns
        Stokes parameters (as a 4-list of arrays, when resultKind == 'Stokes')
        or intensities and OAM (Orbital Angular Momentum) matrix elements (as
        [Is, Ip, OAMs, OAMp, Es, Ep], when resultKind == 'vortex')in the shape
        (energy, theta, psi, [harmonic]), with *theta* being the horizontal
        mesh angles and *psi* the vertical mesh angles. Each one of the input
        arrays is a 1D array of an individually selectable length. Energy
        spread is sampled by a normal distribution and the resulting field
        values are averaged over it. *eSpreadSigmas* is sigma value of the
        distribution; *eSpreadNSamples* sets the number of samples. The
        resulted transverse field is convolved with angular spread by means of
        scipy.ndimage.filters.gaussian_filter. *mode* is a synonymous parameter
        of that filter that controls its behaviour at the borders.

        .. note::
           This method provides incoherent averaging over angular and energy
           spread of electron beam. The photon beam phase is lost here!

        .. note::
           We do not provide any internal mesh optimization, as mesh functions
           are not our core objectives. In particular, the angular meshes must
           be wider than the electron beam divergences in order to convolve the
           field distribution with the electron distribution. A warning will be
           printed (new in version 1.3.4) if the requested meshes are too
           narrow.

        """
        assert resultKind in ('Stokes', 'vortex')

        if self.needReset:
            self.reset()
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
            spr = np.linspace(-eSpreadSigmas, eSpreadSigmas, eSpreadNSamples)
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

        Is = (Es*np.conj(Es)).real.astype(float)
        Ip = (Ep*np.conj(Ep)).real.astype(float)
        if resultKind == 'Stokes':
            Isp = Es*np.conj(Ep).astype(complex)
        elif resultKind == 'vortex':
            dEsdtheta, dEsdpsi = np.gradient(Es, theta, psi, axis=(1, 2))
            dEpdtheta, dEpdpsi = np.gradient(Ep, theta, psi, axis=(1, 2))

            # lsy = 1j*(dEsdtheta*psi[:] - dEsdpsi*theta[:, None])
            # lpy = 1j*(dEpdtheta*psi[:] - dEpdpsi*theta[:, None])
            # https://stackoverflow.com/a/62655664/2696065
            thetaShape = np.swapaxes(dEsdpsi, dEsdpsi.ndim-1, 1).shape
            theta_brc = np.broadcast_to(theta, thetaShape)
            theta_brc = np.swapaxes(theta_brc, dEsdpsi.ndim-1, 1)

            psiShape = np.swapaxes(dEsdtheta, dEsdtheta.ndim-1, 2).shape
            psi_brc = np.broadcast_to(psi, psiShape)
            psi_brc = np.swapaxes(psi_brc, dEsdtheta.ndim-1, 2)

            lsy = 1j*(dEsdtheta*psi_brc - dEsdpsi*theta_brc)
            lpy = 1j*(dEpdtheta*psi_brc - dEpdpsi*theta_brc)
            OAMs = (Es.conj()*lsy).real.astype(float)
            OAMp = (Ep.conj()*lpy).real.astype(float)

        if ispread:
            if iharmonic:
                ws = wspr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
            else:
                ws = wspr[np.newaxis, np.newaxis, np.newaxis, :]
            Is = (Is * ws).sum(axis=ispread)
            Ip = (Ip * ws).sum(axis=ispread)
            if resultKind == 'Stokes':
                Isp = (Isp * ws).sum(axis=ispread)
            elif resultKind == 'vortex':
                OAMs = (OAMs * ws).sum(axis=ispread)
                OAMp = (OAMp * ws).sum(axis=ispread)
                Es = (Es * ws).sum(axis=ispread)
                Ep = (Ep * ws).sum(axis=ispread)

        self.Is = Is
        self.Ip = Ip
        if resultKind == 'Stokes':
            self.Isp = Isp

        if resultKind == 'Stokes':
            s0 = Is + Ip
            s1 = Is - Ip
            s2 = 2. * np.real(Isp)
            s3 = -2. * np.imag(Isp)
            ss = [s0, s1, s2, s3]
        elif resultKind == 'vortex':
            ss = [Is, Ip, OAMs, OAMp, Es, Ep]
        else:
            raise ValueError("Unknown resultKind {0}".format(resultKind))

        if (self.dxprime > 0 or self.dzprime > 0) and \
                len(theta) > 1 and len(psi) > 1:
            from scipy.ndimage.filters import gaussian_filter
            Sx = self.dxprime / (theta[1] - theta[0])
            Sz = self.dzprime / (psi[1] - psi[0])
            # print("self.dxprime, theta[-1] - theta[0], Sx, len(theta)")
            # print(self.dxprime, theta[-1] - theta[0], Sx, len(theta))
            # print("self.dzprime, psi[-1] - psi[0], Sz, len(psi)")
            # print(self.dzprime, psi[-1] - psi[0], Sz, len(psi))
            if Sx > len(theta)//4:  # ±2σ
                print("************* Warning ***********************")
                print("Your theta mesh is too narrow!")
                print("It must be wider than the electron beam width")
                print("*********************************************")
            if self.xPrimeMax < theta.max():
                print("************* Warning ****************************")
                print("Your xPrimeMax is too small!")
                print("It must be bigger than theta.max()")
                if hasattr(self, 'xPrimeMaxAutoReduce'):
                    if self.xPrimeMaxAutoReduce:
                        print("You probably need to set "
                              "xPrimeMaxAutoReduce=False")
                print("**************************************************")
            if Sz > len(psi)//4:  # ±2σ
                print("************* Warning ************************")
                print("Your psi mesh is too narrow!")
                print("It must be wider than the electron beam height")
                print("**********************************************")
            if self.zPrimeMax < psi.max():
                print("************* Warning ****************************")
                print("Your zPrimeMax is too small!")
                print("It must be bigger than psi.max()")
                if hasattr(self, 'zPrimeMaxAutoReduce'):
                    if self.zPrimeMaxAutoReduce:
                        print("You probably need to set "
                              "zPrimeMaxAutoReduce=False")
                print("**************************************************")
            # mode = 'reflect'  # default in gaussian_filter
            for ie, ee in enumerate(energy):
                if harmonic is None:
                    for arr in ss:
                        arr[ie, :, :] = gaussian_filter(
                            arr[ie, :, :], [Sx, Sz], mode=mode)
                else:
                    for ih, hh in enumerate(harmonic):
                        for arr in ss:
                            arr[ie, :, :, ih] = gaussian_filter(
                                arr[ie, :, :, ih], [Sx, Sz], mode=mode)

        if resultKind == 'Stokes':
            with np.errstate(divide='ignore'):
                return [s0,
                        np.where(s0, s1/s0, s0),
                        np.where(s0, s2/s0, s0),
                        np.where(s0, s3/s0, s0)]
        elif resultKind == 'vortex':
            return ss


class IntegratedSource(SourceBase):
    """Base class for the Sources with numerically integrated amplitudes:
    :class:`SourceFromField` and :class:`Undulator`.
    Not to be called explicitly."""

    hiddenParams = ['gIntervals']

    def __init__(self, *args, **kwargs):
        """
        *gp*: float
            Defines the relative precision of the integration (last
            significant digit). Undulator model converges down to 1e-6 and
            below. Custom field calculation may require setting the precision
            of 1e-3.

        *gNodes*: int
            Number of integration nodes in each of the integration intervals.
            If not provided at init, will be defined automatically.

        *targetOpenCL*:  None, str, 2-tuple or tuple of 2-tuples
            assigns the device(s) for OpenCL accelerated calculations. None,
            if pyopencl is not wanted. Ignored if pyopencl is not installed.
            Accepts the following values:

            1) a tuple (iPlatform, iDevice) of indices in the
               lists ``cl.get_platforms()`` and ``platform.get_devices()``, see
               the section :ref:`calculations_on_GPU`.

            2) a tuple of tuples ((iP1, iD1), ..., (iPn, iDn)) to assign
               specific devices from one or multiple platforms.

            3) int iPlatform - assigns all devices found at the given platform.

            4) 'GPU' - lets the program scan the system and select all found
               GPUs.

            5) 'CPU' - similar to 'GPU'. If one CPU exists in multiple
               platforms the program tries to select the vendor-specific
               driver.

            6) 'other' - similar to 'GPU', used for Intel PHI and other OpenCL-
               capable accelerator boards.

            7) 'all' - lets the program scan the system and assign all found
               devices. Not recommended, since the performance will be limited
               by the slowest device.

            8) 'auto' - lets the program scan the system and make an assignment
               according to the priority list: 'GPU', 'other', 'CPU' or None if
               no devices were found. Used by default.

            9) 'SERVER_ADRESS:PORT' - calculations will be run on remote
               server. See ``tests/raycing/RemoteOpenCLCalculation``.

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
            Tapering and Near Field calculations require double precision.


        """
        gp = kwargs.pop('gp', 1e-6)
        gIntervals = kwargs.pop('gIntervals', 2)
        gNodes = kwargs.pop('gNodes', None)
        targetOpenCL = kwargs.pop('targetOpenCL', raycing.targetOpenCL)
        precisionOpenCL = kwargs.pop(
            'precisionOpenCL', raycing.precisionOpenCL)

        super(IntegratedSource, self).__init__(*args, **kwargs)
        # Integration routine-related init
        try:
            self.gIntervals = int(gIntervals)
            self.quadm = int(gNodes)
            self.needConvergence = False
        except TypeError:
            self.needConvergence = True
        self.gp = gp
        self.madBoundary = 20
        self.convergence_finder = 'mixed'  # , "mad"
        self._useGauLeg = False
        self.maxIntegrationSteps = 9000  # Up to 511000 nodes
        self.convergenceSearchFlag = False
        self.trajectory = None

        # OpenCL-related init
        self.cl_ctx = None
        if (self.R0 is not None):
            precisionOpenCL = 'float64'
        if targetOpenCL is not None:
            if not isOpenCL:
                raycing.colorPrint("pyopencl is not available!", "RED")
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

    @property
    def gNodes(self):
        try:
            return self.quadm
        except AttributeError as e:
            raise Exception('First run the method test_convergence()!') from e

    @gNodes.setter
    def gNodes(self, gNodes):
        self.quadm = int(gNodes)
        self._build_integration_grid()

        # Need to recalculate the integration parameters

    def _clenshaw_curtis(self, n):
        """
        Adopted from quadpy https://github.com/nschloe/quadpy
        Fixed python 2 compatibilty
        """
        points = -np.cos((np.pi * np.arange(n)) / (n - 1))

        if n == 2:
            weights = np.array([1.0, 1.0])
            return (points, weights)

        n -= 1
        N = np.arange(1, n, 2)
        length = len(N)
        m = n - length
        v0 = np.concatenate(
            [2.0 / N / (N - 2), np.array([1.0 / N[-1]]), np.zeros(m)]
        )
        v2 = -v0[:-1] - v0[:0:-1]
        g0 = -np.ones(n)
        g0[length] += n
        g0[m] += n
        g = g0 / (n ** 2 - 1 + (n % 2))

        w = np.fft.ihfft(v2 + g)
        assert max(w.imag) < 1.0e-15
        w = w.real

        if n % 2 == 1:
            weights = np.concatenate([w, w[::-1]])
        else:
            weights = np.concatenate([w, w[len(w)-2::-1]])

        return (points, weights)

    def _find_convergence_thrsh(self, testMode=False):  # Obsolete
        mstart = 5
        m = mstart
        quad_int_error = self.gp * 10.
        converged = True
        if testMode:
            xm = []
            pltout = []
            statOut = []
        while quad_int_error >= self.gp:
            m += 1
            self.quadm = int(1.5**m)
            self._build_integration_grid()
            if self.cl_ctx is not None:
                sE = self.E_max * np.ones(2)
                sTheta_max = self.Theta_max * np.ones(2)
                sPsi_max = self.Psi_max * np.ones(2)
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
            if testMode:
                xm.append(self.quadm*self.gIntervals)
                pltout.append(In)
                statOut.append(quad_int_error)
            if raycing._VERBOSITY_ > 10:
                print("G = {0}".format(
                    [self.gIntervals, self.quadm, quad_int_error, I2]))
            if self.quadm > 400000:
                self.gIntervals *= 2
                m = mstart
                quad_int_error = self.gp * 10.
                if self.gIntervals > 100:
                    converged = False
                    break
                continue
        if testMode:
            return converged, (np.array(xm), np.array(pltout),
                               np.array(statOut), np.array(statOut))
        else:
            return converged, (0,)

    def _find_convergence_mixed(self, testMode=False):
        if raycing._VERBOSITY_ > 0:
            print("Estimating convergence")
        mstart = 3
        m = mstart
        quad_int_error = self.gp * 10.
        converged = True
        if testMode:
            xm = []
            pltout = []
            statOut = []
        # PHASE 1: Find convergence, very rough and fast
        if raycing._VERBOSITY_ > 0:
            print("Phase 1. Exponential / rough")
        step_stat = 5
        while m < 10000:
            m += 1
            self.quadm = int(2**m)
            mad, dimad = self._get_mad()
            if testMode:
                xm.append(self.quadm*self.gIntervals)
                pltout.append(mad)
                statOut.append(quad_int_error)
            if raycing._VERBOSITY_ > 10:
                print("G = {0}".format(
                    [self.gIntervals, self.quadm, mad, dimad]))
            if (dimad < self.gp) or (mad < self.gp):
                break
            if self.quadm > 400000:
                break

        # PHASE 2: Bisection last interval, locate the threshold precizely
        ph2start = int(2**(m-1))
        ph2end = self.quadm
        jmax = int(np.log2((ph2end-ph2start) / (4*step_stat)))
        if raycing._VERBOSITY_ > 0:
            print("Phase 2. Bisection / precize. {} steps".format(jmax))
        for j in range(jmax):
            self.quadm = int(0.5*(ph2end+ph2start))
            mad, dimad = self._get_mad()

            if (dimad < self.gp) or (mad < self.gp):
                ph2end = self.quadm
            else:
                ph2start = self.quadm
        self.quadm = ph2end
        if raycing._VERBOSITY_ > 0:
            print("Done estimating convergence")

        if testMode:
            return converged, (np.array(xm), np.array(pltout),
                               np.array(statOut), np.array(statOut))
        else:
            return converged, (0,)

    def _get_mad(self):
        def _mad(vin):
            med = np.median(vin)
            return np.median(np.abs(vin - med))

        tmp_quadm = self.quadm
        tmp_GI = self.gIntervals

        m_step = 1
        stat_step = 5
        m_start = self.quadm - int(0.5*stat_step)
        m = 0
        k = m_start
        pltout = []
        dIout = []
        sE = self.E_max * np.ones(1)
        sTheta_max = self.Theta_max * np.ones(1)
        sPsi_max = self.Psi_max * np.ones(1)

        for m in range(stat_step):
            k += m_step
            self.quadm = k
            self._build_integration_grid()
            Inew = self.build_I_map(sE, sTheta_max, sPsi_max)[0]

            if m == 0:
                Iold = Inew
                continue
            pltout.append(Inew)
            dIout.append(np.abs(Inew-Iold)/Inew)
            Iold = Inew

        mad = _mad(np.abs(np.array(pltout))[:])
        dIMAD = np.median(dIout[:])
        if raycing._VERBOSITY_ > 10:
            print(self.quadm, mad, dIMAD)

        self.quadm = tmp_quadm
        self.gIntervals = tmp_GI

        return mad, dIMAD

    def test_convergence(self, nMax=500000, withPlots=True, overStep=100):
        u"""
        This function evaluates the length of the integration grid required for
        convergence.

        *nMax*: int
            Maximum number of nodes.
        *withPlots*: bool
            Enables visualization.
        *overStep*: int
            Defines the number of extra points to calculate when the
            convergence is found. If None, calculation will proceed
            till *nMax*.


        """

        def _mad(vin):
            med = np.median(vin)
            return np.median(np.abs(vin - med))

        self.convergenceSearchFlag = True
        self.needReset = False
        self._reset_limits()
        mStart = 10
        mStep = 1
        statStep = 5
        m = 0
        k = mStart
        converged = False
        postConv = 0
        pltout = []
        dIout = []
        Iold = 0
        sE = self.E_max * np.ones(1)
        sTheta_max = self.Theta_max * np.ones(1)
        sPsi_max = self.Psi_max * np.ones(1)

        statOut = []
        dIOut = []
        xm = []

        outQuad = 0
        outInt = 0
        if withPlots:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(8, 6))

            ax0 = fig.add_axes([0.1, 0.65, 0.8, 0.3])
            ax0.xaxis.set_visible(False)
            ax0.set_ylabel('Relative intensity $I$', color='C0')
            ampLine, = ax0.semilogy([], [], 'C0')

            ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.55])
            ax1.set_xlabel('Number of nodes')
            ax1.set_ylabel('Median absolute deviation of $I$', color='C1')
            madLine, = ax1.semilogy([], [], 'C1')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Median $dI/I$', color='C2')
            relmadLine, = ax2.semilogy([], [], 'C2')
        else:
            fig = None

        while True:
            m += 1
            if m % 1000 == 0:
                mStep *= 2
                if True:  # raycing._VERBOSITY_ > 10:
                    # print("INSUFFICIENT CONVERGENCE RANGE:", k, "NODES")
                    print("INCREASING CONVERGENCE STEP. NEW STEP", mStep)

            k += mStep
            self.quadm = k
            self._build_integration_grid()
            xm.append(k*self.gIntervals)
            Inew = self.build_I_map(sE, sTheta_max, sPsi_max)[0]
            pltout.append(Inew)
            dIout.append(np.abs(Inew-Iold)/Inew)
            if m == 1:
                Iold = Inew
                continue
            Iold = Inew

            if withPlots:
                ampLine.set_xdata(xm)
                relInt = np.array(pltout)
                relInt /= relInt.max()
                ampLine.set_ydata(relInt)
                new_y_min = np.floor(np.log10(relInt.min()))
                ax0.set_xlim([0, xm[-1]+5])
                ax0.set_ylim([10**(new_y_min+0.1), 1.1])

            if converged:
                postConv += 1
            if m > statStep:
                mad = _mad(np.abs(np.array(pltout))[m-statStep:m])
                dIMAD = np.median(dIout[m-statStep:m])

                statOut.append(mad)
                dIOut.append(dIMAD)

                if ((dIMAD < self.gp) or (mad < self.gp)) and not converged:
                    convPoint = k*self.gIntervals
                    outQuad = k
                    outInt = self.gIntervals
                    if True:  # raycing._VERBOSITY_ > 10:
                        print("CONVERGENCE THRESHOLD REACHED AT "
                              "{0} NODES, {1} INTERVALS.".format(
                                  k, self.gIntervals))
                        print("INTEGRATION GRID LENGTH IS {} POINTS".format(
                                convPoint))
                    converged = True
                    if withPlots:
                        label = 'True convergence: {0} nodes, {1} interval{2}'\
                            .format(self.quadm, self.gIntervals,
                                    '' if self.gIntervals == 1 else 's')
                        axvlineDict = dict(x=convPoint, color='r', label=label)
                        ax0.axvline(**axvlineDict)
                        ax1.axvline(**axvlineDict)
                if withPlots:
                    new_y_max = np.ceil(np.log10(max(statOut)))
                    new_y_min = np.floor(np.log10(min(statOut)))
                    ax1.set_xlim([0, xm[-1]+5])
                    ax1.set_ylim([10**new_y_min, 10**(new_y_max-0.1)])
                    madLine.set_xdata(xm[statStep:])
                    madLine.set_ydata(statOut)
                    relmadLine.set_xdata(xm[statStep:])
                    relmadLine.set_ydata(dIOut)
                    new_y_max = np.ceil(np.log10(max(dIOut)))
                    new_y_min = np.floor(np.log10(min(dIOut)))
                    ax2.set_xlim([0, xm[-1]+5])
                    ax2.set_ylim([10**new_y_min, 10**new_y_max])
                    fig.canvas.draw()
                    plt.pause(0.001)

            if xm[-1] > nMax:
                if not converged:
                    print("PROBLEM WITH CONVERGENCE. INCREASE nMax.")
                break

            if overStep is not None:
                if postConv > overStep:
                    break

        convRes, stats = self._find_convergence_mixed()
        print("CONVERGENCE TEST COMPLETED.")
        self.needReset = True
        if withPlots:
            label = 'Auto-finder: {0} nodes, {1} interval{2}'.format(
                self.quadm, self.gIntervals,
                '' if self.gIntervals == 1 else 's')
            axvlineDict = dict(x=self.quadm*self.gIntervals, color='m',
                               linestyle='--', label=label)
            ax0.axvline(**axvlineDict)
            ax1.axvline(**axvlineDict)
            ax1.legend()
            fig.canvas.draw()
            plt.pause(0.1)
        return converged, outQuad, outInt, fig

    def _build_integration_grid(self):
        """To be redefined in subclasses"""
        pass

    def _reset_integration_grid(self):
        """Adjusting the integration grid length"""
        if self.needConvergence:
            self.quadm = 0
            tmpeEspread = self.eEspread
            self.eEspread = 0
            self.convergenceSearchFlag = True
            convRes, stats = self._find_convergence_mixed()
            self.convergenceSearchFlag = False
            self.eEspread = tmpeEspread
        self._build_integration_grid()
        if raycing._VERBOSITY_ > 0:
            print("Done with integration optimization, {0} points will be used"
                  " in {1} interval{2}".format(
                      self.quadm, self.gIntervals,
                      's' if self.gIntervals > 1 else ''))

    def shine(self, toGlobal=True, withAmplitudes=True, fixedEnergy=False,
              wave=None, accuBeam=None):
        u"""
        Returns the source beam. If *toGlobal* is True, the output is in
        the global system. If *withAmplitudes* is True, the resulted beam
        contains arrays Es and Ep with the *s* and *p* components of the
        electric field.

        *fixedEnergy* is either None or a value in eV. If *fixedEnergy* is
        specified, the energy band is not 0.1%BW relative to *fixedEnergy*, as
        probably expected but is given by (eMax - eMin) of the constructor.

        *wave* and *accuBeam* are used in wave diffraction. *wave* is a Beam
        object and determines the positions of the wave samples. It must be
        obtained by a previous ``prepare_wave`` run. *accuBeam* is only needed
        with *several* repeats of diffraction integrals when the parameters of
        the filament beam must be preserved for all the repeats.


        .. Returned values: beamGlobal
        """
        if self.needReset:
            self.reset()
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
        seeded = np.int64(0)
        seededI = 0.
        np.seterr(invalid='warn')
        np.seterr(divide='warn')
        dgamma = None
        if self.filamentBeam:
            if accuBeam is None:
                rsE = np.random.random_sample() * \
                    float(self.E_max - self.E_min) + self.E_min
                rX = self.dx * np.random.standard_normal()
                rZ = self.dz * np.random.standard_normal()
                dtheta = self.dxprime * np.random.standard_normal()
                dpsi = self.dzprime * np.random.standard_normal()
                if self.eEspread > 0:
                    dgamma = self.gamma * self.eEspread * \
                        np.random.standard_normal()
            else:
                rsE = accuBeam.E[0]
                rX = accuBeam.filamentDX
                rZ = accuBeam.filamentDZ
                dtheta = accuBeam.filamentDtheta
                dpsi = accuBeam.filamentDpsi
                dgamma = accuBeam.filamentDgamma
                seeded = accuBeam.seeded
                seededI = accuBeam.seededI

#        if self.full:
#            if self.filamentBeam:
#                self.theta0 = dtheta
#                self.psi0 = dpsi
#            else:
#                self.theta0 = np.random.normal(0, self.dxprime, mcRays)
#                self.psi0 = np.random.normal(0, self.dzprime, mcRays)

        if fixedEnergy:
            rsE = fixedEnergy
            if (self.E_max-self.E_min) > fixedEnergy*1.1e-3:
                raycing.colorPrint(
                    "Warning: the bandwidth seems too big. "
                    "Specify it by giving eMin and eMax in the constructor.",
                    "YELLOW")
        nrep = 0
        rep_condition = True

        while rep_condition:
            seeded += mcRays
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
                rDiffr = np.sqrt((x**2 + y**2 + z**2))
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

            Intensity, mJs, mJp = self.build_I_map(rE, rTheta, rPsi, dg=dgamma)

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
                bot.sourceSIGMAx, bot.sourceSIGMAz = self.get_SIGMA(
                    bot.E, onlyOddHarmonics=False)
                dxR = np.random.normal(0, bot.sourceSIGMAx, npassed)
                dzR = np.random.normal(0, bot.sourceSIGMAz, npassed)

            if wave is not None:
                wave.rDiffr = np.sqrt(
                    ((wave.xDiffr - dxR)**2 + wave.yDiffr**2 +
                     (wave.zDiffr - dzR)**2))
                wave.path[:] = 0
                wave.a[:] = (wave.xDiffr - dxR) / wave.rDiffr
                wave.b[:] = wave.yDiffr / wave.rDiffr
                wave.c[:] = (wave.zDiffr - dzR) / wave.rDiffr
            else:
                bot.x[:] = dxR
                bot.z[:] = dzR
                bot.a[:] = rTheta[I_pass]
                bot.c[:] = rPsi[I_pass]

                if True:  # not self.full:
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
                if self.uniformRayDensity:
                    bot.Es[:] = mJs
                    bot.Ep[:] = mJp
                else:
                    bot.Es[:] = mJs / mJs2**0.5
                    bot.Ep[:] = mJp / mJp2**0.5

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
                    except Exception:
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
            bo.filamentDgamma = dgamma

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
        if wave is not None and self.R0 is None:
            bor.x[:] = dxR
            bor.y[:] = 0
            bor.z[:] = dzR
            bor.path[:] = 0
            mPh = np.exp(1e7j * wave.E/CHBAR * wave.rDiffr)
            wave.Es *= mPh
            wave.Ep *= mPh

        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bor, self.center)
        bor.parentId = self.uuid
        raycing.append_to_flow(self.shine, [bor],
                               inspect.currentframe())
        return bor
