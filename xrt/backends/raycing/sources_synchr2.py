# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "12 Aug 2021"
import os
import sys
#import pickle
import numpy as np
from scipy import optimize
from scipy import special
from scipy.interpolate import interp1d
import inspect
import time

from .. import raycing
from . import myopencl as mcl
from .sources_beams import Beam, allArguments
from .physconsts import E0, C, M0, EV2ERG, K2B, SIE0, EMC,\
    SIM0, FINE_STR, PI, PI2, SQ2, SQ3, SQPI, E2W, E2WC, CHeVcm, CH, CHBAR

try:
    import pyopencl as cl  # analysis:ignore
    isOpenCL = True
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
except ImportError:
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
        .. warning::
            If you change any undulator parameter outside of the constructor,
            invoke ``your_undulator_instance.reset()``.

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

            .. warning::
                If you change these parameters outside of the constructor,
                interpret them in *rad*; in the constructor they are given in
                *mrad*. This awkwardness is kept for version compatibility.

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

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.name] = [self, 0]

        self.center = center  # 3D point in global system
        self._pitch = raycing.auto_units_angle(pitch)
        self._yaw = raycing.auto_units_angle(yaw)
        self.nrays = np.long(nrays)

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
            print("Set either dx or betaX!")
        if (self.dz is None) and (self._betaZ is not None):
            self.dz = np.sqrt(self._eEpsilonZ*self._betaZ)
        elif (self.dz is None) and (self._betaZ is None):
            print("Set either dz or betaZ!")

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
            self.dE = (self.E_max - self.E_min) / float(self.eN - 1)
            self.dTheta = (self.Theta_max - self.Theta_min) / float(self.nx - 1)
            self.dPsi = (self.Psi_max - self.Psi_min) / float(self.nz - 1)
        except:
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
#        print(self.Imax, self.xzE, self.fluxConst, self.nrepmax)

    def build_I_map(self):
        """Used to calculate the intensity. To be redefined in the subclass"""
        pass

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

    def intensities_on_mesh(self, energy='auto', theta='auto', psi='auto',
                            harmonic=None,
                            eSpreadSigmas=3.5, eSpreadNSamples=36):
        """Returns the Stokes parameters in the shape (energy, theta, psi,
        [harmonic]), with *theta* being the horizontal mesh angles and *psi*
        the vertical mesh angles. Each one of the input parameters is a 1D
        array of an individually selectable length.

        .. note::
           We do not provide any internal mesh optimization, as mesh functions
           are not our core objectives. In particular, the angular meshes must
           be wider than the electron beam divergences in order to convolve the
           field distribution with the electron distribution. A warning will be
           printed (new in version 1.3.4) if the requested meshes are too
           narrow.

        """
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

        if (self.dxprime > 0 or self.dzprime > 0) and \
                len(theta) > 1 and len(psi) > 1:
            from scipy.ndimage.filters import gaussian_filter
            Sx = self.dxprime / (theta[1] - theta[0])
            Sz = self.dzprime / (psi[1] - psi[0])
#            print(self.dxprime, theta[-1] - theta[0], Sx, len(theta))
#            print(self.dzprime, psi[-1] - psi[0], Sz, len(psi))
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

    def get_SIGMA(self, E, onlyOddHarmonics=False):
        return self.dx, self.dz

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
        obtained by a previous `prepare_wave` run. *accuBeam* is only needed
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
                print("Warning: the bandwidth seems too big. "
                      "Specify it by giving eMin and eMax in the constructor.")
        nrep = 0
        rep_condition = True

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
#                if self.full:
#                    bot.sourceSIGMAx = self.dx
#                    bot.sourceSIGMAz = self.dz
#                    dxR = np.random.normal(0, bot.sourceSIGMAx, npassed)
#                    dzR = np.random.normal(0, bot.sourceSIGMAz, npassed)
#                else:
                bot.sourceSIGMAx, bot.sourceSIGMAz = self.get_SIGMA(
                    bot.E, onlyOddHarmonics=False)
                dxR = np.random.normal(0, bot.sourceSIGMAx, npassed)
                dzR = np.random.normal(0, bot.sourceSIGMAz, npassed)

            if wave is not None:
                wave.rDiffr = np.sqrt(((wave.xDiffr - dxR)**2 + wave.yDiffr**2 +
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

                if True: #not self.full:
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
        bor.parentId = self.name
        raycing.append_to_flow(self.shine, [bor],
                               inspect.currentframe())
        return bor


class IntegratedSource(SourceBase):
    """Base class for the Sources with numerically integrated amplitudes.
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
            9) 'SERVER_ADRESS:PORT' - calculations will be run on remote
            server. See corresponding example.

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
        precisionOpenCL = kwargs.pop('precisionOpenCL', raycing.precisionOpenCL)

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
        self.convergence_finder = 'mixed' #, "mad"
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
                print("pyopencl is not available!")
            else:
                print("1", targetOpenCL)
                self.ucl = mcl.XRT_CL(
                    r'undulator2.cl', targetOpenCL, precisionOpenCL)
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
        return self.quadm

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
            weights = np.concatenate([w, w[len(w) - 2 :: -1]])

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
            return converged, (np.array(xm), np.array(pltout), np.array(statOut),
                   np.array(statOut))
        else:
            return converged, (0,)

    def _find_convergence_mixed(self, testMode=False):
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
        print("Phase 1. Exponential / rough")
        step_stat = 5
        while m<10000:
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
        print("Phase 2. Bisection / precize. {} steps".format(jmax))
        for j in range(jmax):
            self.quadm = int(0.5*(ph2end+ph2start))
            mad, dimad = self._get_mad()

            if (dimad < self.gp) or (mad < self.gp):
                ph2end = self.quadm
            else:
                ph2start = self.quadm
        self.quadm = ph2end

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

    def test_convergence(self, nMax=500000, interactive=True, autoStop=True):
        def _mad(vin):
            med = np.median(vin)
            return np.median(np.abs(vin - med))

        self.convergenceSearchFlag = True
        self.needReset = False
        self._reset_limits()
        mStart = 3
        mStep = 1
        statStep = 5
        m = 0
        k = mStart
        converged = False
        overStep = 120
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
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Total nodes', fontsize=14)
#        ax.set_ylabel('Electric field amplitude', fontsize=24)
        ax.set_ylabel('MAD I', fontsize=14)
        ax.tick_params(axis='y', labelcolor='b')
        madLine, = ax.semilogy([], [], label='MAD Amp')

        ax2 = ax.twinx()
        ax2.set_xlabel('Total nodes', fontsize=14)
        ax2.set_ylabel('Median dI/I', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='g')
        relmadLine, = ax2.semilogy([], [], 'g')

        fig2 = plt.figure(figsize=(8,5))
        axt = fig2.add_subplot(111)
        axt.set_xlabel('Total nodes', fontsize=14)
        axt.set_ylabel('Electric field amplitude', fontsize=14)
        ampLine, = axt.semilogy([], [], label='Amp')

        while True:
            m += 1
            if m % 1000 == 0:
                mStep *= 2
                if True: #raycing._VERBOSITY_ > 10:
                    print("INSUFFICIENT CONVERGENCE RANGE:", k, "NODES")
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

            ampLine.set_xdata(xm)
            ampLine.set_ydata(pltout)
            new_y_max = np.ceil(np.log10(max(pltout)))
            new_y_min = np.floor(np.log10(min(pltout)))
            axt.set_ylim([10**new_y_min, 10**new_y_max])
            axt.set_xlim([0, xm[-1]+5])
            fig2.canvas.draw()
            plt.pause(0.001)

            if converged:
                postConv += 1
            if m > statStep:
                mad = _mad(np.abs(np.array(pltout))[m-statStep:m])
                dIMAD = np.median(dIout[m-statStep:m])

                statOut.append(mad)
                dIOut.append(dIMAD)

                if ((dIMAD < self.gp) or (mad < self.gp)) and not converged:
                    convPoint = k*self.gIntervals
                    if True: #raycing._VERBOSITY_ > 10:
                        print("CONVERGENCE THRESHOLD REACHED AT", convPoint)
                    converged = True
                    ax.axvline(x=convPoint, color='r')
                    axt.axvline(x=convPoint, color='r')
                new_y_max = np.ceil(np.log10(max(statOut)))
                new_y_min = np.floor(np.log10(min(statOut)))
                ax.set_ylim([10**new_y_min, 10**new_y_max])
                ax.set_xlim([0, xm[-1]+5])
                madLine.set_xdata(xm[statStep:])
                madLine.set_ydata(statOut)
                relmadLine.set_xdata(xm[statStep:])
                relmadLine.set_ydata(dIOut)

                new_y_max = np.ceil(np.log10(max(dIOut)))
                new_y_min = np.floor(np.log10(min(dIOut)))
                ax2.set_ylim([10**new_y_min, 10**new_y_max])

                fig.canvas.draw()
                plt.pause(0.001)

            if xm[-1] > nMax or postConv > overStep:
#                if converged:
#                    if raycing._VERBOSITY_ > 10:
#                        print("SUCCESSFULLY CONVERGED AT", convPoint)
#                else:
#                    if raycing._VERBOSITY_ > 10:
#                        print("PROBLEM WITH CONVERGENCE. USING MAX NNODES")
#                    raise("PROBLEM WITH CONVERGENCE. PLEASE INCREASE maxIntegrationSteps")
                break

        convRes, stats = self._find_convergence_mixed()
        ax.axvline(x=self.quadm*self.gIntervals, color='m', linestyle='--')
        axt.axvline(x=self.quadm*self.gIntervals, color='m', linestyle='--')
        fig.canvas.draw()
        plt.pause(0.001)
        fig2.canvas.draw()
        plt.pause(0.001)
        plt.show()

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
        if raycing._VERBOSITY_ > 10:
            print("Done with integration optimization, {0} points will be used"
                  " in {1} interval{2}".format(
                      self.quadm, self.gIntervals,
                      's' if self.gIntervals > 1 else ''))



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

        if self.isMPW:  # xPrimeMaxAutoReduce
            xPrimeMaxTmp = self.K / self.gamma
            if abs(self._xPrimeMax) > xPrimeMaxTmp:
                print("Reducing xPrimeMax from {0} down to {1} mrad".format(
                      self.xPrimeMax * 1e3, xPrimeMaxTmp * 1e3))
                self._xPrimeMax = xPrimeMaxTmp
            if abs(self._xPrimeMin) > abs(xPrimeMaxTmp):
                print("Reducing xPrimeMin from {0} down to {1} mrad".format(
                      self._xPrimeMin * 1e3, xPrimeMaxTmp * 1e3))
                self._xPrimeMin = np.sign(self._xPrimeMin) * xPrimeMaxTmp

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
        self._K = kwargs.pop('K', 8.446)
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
    """Dedicated class for the sources based on custom field table."""
    def __init__(self, *args, **kwargs):
        """
        *customField*: float or str or tuple(fileName, kwargs) or numpy array.
            If float, adds a constant longitudinal field.
            If str or tuple, expects table of field
            samples given as an Excel file or as text file. If given
            as a tuple or list, the 2nd member is a key word dictionary for
            reading Excel by :meth:`pandas.read_excel()` or reading text file
            by :meth:`numpy.loadtxt()`, e.g. ``dict(skiprows=4)`` for skipping
            the file header. The file must contain the columns with
            longitudinal coordinate in mm, {B_hor,} B_ver {, B_long}, all in T.
            The field can be provided as a numpy array with the same structure
            as the table from file.

        """
        customField = kwargs.pop('customField', None)
        super(SourceFromField, self).__init__(*args, **kwargs)

        self.spl_kw = {'kind': 'cubic',
                       'bounds_error': False,
                       'fill_value': 'extrapolate'}
        self.periodicTest = False
        self._customField = customField
        if customField is not None:
            if isinstance(customField, (tuple, list)):
                fname = customField[0]
                kwargs = customField[1]
            elif isinstance(customField, np.ndarray):
                self.customFieldData = customField
                fname = None
            else:
                fname = customField
                kwargs = {}
            if fname:
                self.customFieldData = self.read_custom_field(fname, kwargs)
        else:  # Test with periodic field
            self.Kx = 0.
            self.Ky = 4.4 #17.274 #1.7
            self.phase = 0
            self.L0 = 53.96 #100 #10.
            self.Np = 41 #70 #30
            self.quadm = 50
            self.gIntervals = 2 #*self.Np
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
        if fname.endswith('.xls') or fname.endswith('.xlsx'):
            from pandas import read_excel
            data = read_excel(fname, **kwargs).values
        else:
            data = np.loadtxt(fname, **kwargs)
        return data

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
            By = interp1d(dataz, self.customFieldData[:, 1], **self.spl_kw)(z)
            Bx = np.zeros_like(By)
            Bz = np.zeros_like(By)
        elif dataShape[1] == 3:
            Bx = interp1d(dataz, self.customFieldData[:, 1], **self.spl_kw)(z)
            By = interp1d(dataz, self.customFieldData[:, 2], **self.spl_kw)(z)
            Bz = np.zeros_like(By)
        elif dataShape[1] == 4:
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
        dirz = 1. - 0.5*(ddphiS**2 + ddpsiS**2)
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
            R0 = np.expand_dims(R0, axis=1)
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

            eucosx = -sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -\
                       cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs
            eucosy = -sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +\
                       cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs
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
        betaz = 1 - 0.5*smTerm + 0.125*smTerm**2

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

        Bsr = np.complex(0)
        Bpr = np.complex(0)

        gamma_ = gamma[0] if self.filamentBeam else gamma
        dirx = ddphi
        diry = ddpsi
        dirz = 1. - 0.5*(ddphi**2 + ddpsi**2)
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
                dr = R0 - np.expand_dims(rloc, 1)
                dist = np.linalg.norm(dr, axis=0)
                rdrz = 1./dr[2, :]
                drs = (dr[0, :]**2+dr[1, :]**2)*rdrz
                LRS = 0.5*drs - 0.125*drs**2*rdrz + 0.0625*drs**3*rdrz**2
                sinzloc = np.sin(wc * (self.tg[i] - trajz_))
                coszloc = np.cos(wc * (self.tg[i] - trajz_))
                sindrs = np.sin(wc * LRS)
                cosdrs = np.cos(wc * LRS)
                eucosx = -sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -\
                           cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs
                eucosy = -sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +\
                           cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs
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
            betaz = 1. - 0.5*smTerm + 0.125*smTerm**2

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
            return self._build_I_map_custom_field_conv(w, ddtheta, ddpsi, dh, dg)
        else:
            return self._build_I_map_custom_field_CL(w, ddtheta, ddpsi, dh, dg)

    def _build_integration_grid(self):
        quad_rule = np.polynomial.legendre.leggauss if self._useGauLeg else\
            self._clenshaw_curtis
        tg_n, ag_n = quad_rule(self.quadm)

        if isinstance(self.customFieldData, (float, int)) or\
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
                betaz = 1.-0.5*smTerm-0.125*smTerm**2
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
            k1beta = rkStep * f_beta([Bx[iB], By[iB], Bz[iB]],
                                        beta)
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
                betam_int +=  beta_next[0]**2 + beta_next[1]**2

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
        betam = 1. - (1. + 0.5 * self.Kx**2 + 0.5*self.Ky**2) / 2. / gamma2

        self.wu = PI  / self.L0 / gamma2 * \
            (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2WC

        z = 2*np.pi*self.tg/self.L0
        tgw = self.L0 / 2. / np.pi
        betax = self.Ky / gamma * np.cos(z)
        betay =-self.Kx / gamma * np.cos(z + self.phase)
        trajx = tgw * self.Ky / gamma * np.sin(z)
        trajy =-tgw * self.Kx / gamma * np.sin(z + self.phase)
        trajz = tgw * (betam * z - 0.125 / gamma**2 *
                 (self.Ky**2 * np.sin(2*z) +
                  self.Kx**2 * np.sin(2*(z + self.phase))))
        return betax, betay, [betam], trajx, trajy, trajz

    def _build_I_map_custom_field_CL(self, w, ddtheta, ddpsi,
                                  harmonic=None, dgamma=None):
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
            self.trajectory = [trajx*emcg0,
                               trajy*emcg0,
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
                               self.cl_precisionF(self.ag),   # Integration weights
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
                               self.cl_precisionF(self.ag),   # Integration weights
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

        self.beta = [betax, betay]
        self.trajectory = [trajx, trajy, trajz]

        betam = betazav[-1]
        ab = 0.5 / np.pi / betam if self.filamentBeam else\
            0.5 / np.pi / (1. - 0.5/gamma**2 + betam*EMC**2/gamma**2)
        emcg = SIE0 / SIM0 / C / 10. / gamma

        if self.R0:
            R0v = np.array((np.tan(ddtheta), np.tan(ddpsi), np.ones_like(ddpsi)))
#            R0n = np.linalg.norm(R0v, axis=0)  # Only for spherical screen
            R0v *= R0  # /R0n
        else:
            R0v=None

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
    Undulator source. The computation is volumnous and thus decent GPU is
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
            Phase difference between horizontal and vertical magnetic arrays.
            Used in the elliptical case where it should be equal to 90 or -90.

        *taper*: tuple(dgap(mm), gap(mm))
            Linear variation in undulator gap. None if tapering is not used.
            Pyopencl is recommended for tapering.

        *targetE*: a tuple (Energy, harmonic{, isElliptical})
            Can be given for automatic calculation of the deflection parameter.
            If isElliptical is not given, it is assumed as False (as planar).

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
                isElliptical = targetE[2]
                if isElliptical:
                    Kx = Ky / 2**0.5
                    if raycing._VERBOSITY_ > 10:
                        print("Kx = Ky = {0}".format(Kx))

        phaseDeg = np.degrees(raycing.auto_units_angle(phaseDeg)) if\
            isinstance(phaseDeg, raycing.basestring) else phaseDeg
        self.phase = np.radians(phaseDeg)

        if taper is not None:
            self.taper = taper[0] / self.Np / self.L0 / taper[1]
            self.gap = taper[1]
        else:
            self.taper = None

        if Kx == 0 and Ky == 0:
            if abs(K) > 0:
                Ky = K
            elif abs(B0y) > 0:
                    self.B0y = B0y
            elif abs(B0x) > 0:
                    self.B0x = B0x
            else:
                self.K = 1
                raise("Please define either K or B0!")

        self.Kx = Kx
        self.Ky = Ky

        if self.Kx == 0 and B0x > 0:
            self.B0x = B0x

#        self.B0x = K2B * self.Kx / self.L0
#        self.B0y = K2B * self.Ky / self.L0

        self.xPrimeMaxAutoReduce = xPrimeMaxAutoReduce
        self.zPrimeMaxAutoReduce = zPrimeMaxAutoReduce
        if self.R0 is not None:  # Required for convergence
            self.xPrimeMaxAutoReduce = True
            self.zPrimeMaxAutoReduce = True

        if xPrimeMaxAutoReduce:
            K0 = self.Ky if abs(self.Ky) > 0 else 2.
            xPrimeMaxTmp = K0 / self.gamma
            if abs(self._xPrimeMax) > abs(xPrimeMaxTmp):
                print("Reducing xPrimeMax from {0} down to {1} mrad".format(
                      self._xPrimeMax * 1e3, xPrimeMaxTmp * 1e3))
                self._xPrimeMax = xPrimeMaxTmp
            if abs(self._xPrimeMin) > abs(xPrimeMaxTmp):
                print("Reducing xPrimeMin from {0} down to {1} mrad".format(
                      self._xPrimeMin * 1e3, xPrimeMaxTmp * 1e3))
                self._xPrimeMin = np.sign(self._xPrimeMin) * xPrimeMaxTmp
        if zPrimeMaxAutoReduce:
            K0 = self.Kx if abs(self.Kx) > 0 else 2.
            zPrimeMaxTmp = K0 / self.gamma
            if abs(self._zPrimeMax) > abs(zPrimeMaxTmp):
                print("Reducing zPrimeMax from {0} down to {1} mrad".format(
                      self._zPrimeMax * 1e3, zPrimeMaxTmp * 1e3))
                self._zPrimeMax = zPrimeMaxTmp
            if abs(self._zPrimeMin) > abs(zPrimeMaxTmp):
                print("Reducing xPrimeMin from {0} down to {1} mrad".format(
                      self._zPrimeMin * 1e3, zPrimeMaxTmp * 1e3))
                self._zPrimeMin = np.sign(self._zPrimeMin) * zPrimeMaxTmp

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
    def targetE(self):
        return self._targetE

    @targetE.setter
    def targetE(self, targetE):
        self._targetE = targetE
        Ky = np.sqrt(targetE[1] * 8 * PI * self.gamma2 /
                    self.L0 / targetE[0] / E2WC - 2)
        Kx = 0
        if raycing._VERBOSITY_ > 10:
            print("K = {0}".format(Ky))
        if np.isnan(Ky):
            raise ValueError("Cannot calculate K, try to increase the "
                             "undulator harmonic number")
        if len(targetE) > 2:
            isElliptical = targetE[2]
            if isElliptical:
                Kx = Ky / 2**0.5
                if raycing._VERBOSITY_ > 10:
                    print("Kx = Ky = {0}".format(Kx))
        self._Kx = Kx
        self._Ky = Ky
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def Kx(self):
        return self._Kx

    @Kx.setter
    def Kx(self, Kx):
        self._Kx = float(Kx)
        self._B0x = K2B * Kx / self.L0
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def Ky(self):
        return self._Ky

    @Ky.setter
    def Ky(self, Ky):
        self._Ky = float(Ky)
        self._K = float(Ky)
        self._B0y = K2B * Ky / self.L0
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
        self.needReset = True
        # Need to recalculate the integration parameters

    @property
    def B0y(self):
        return self._B0y

    @B0y.setter
    def B0y(self, B0y):
        self._B0y = float(B0y)
        self._Ky = B0y * self.L0 / K2B
        self.needReset = True
        # Need to recalculate the integration parameters

    def report_E1(self):
        wu = PI / self.L0 / self.gamma2 * \
            (2*self.gamma2 - 1. - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2WC

        E1 = 2*wu*self.gamma2 / (1 + 0.5*self.Kx**2 + 0.5*self.Ky**2)
        if raycing._VERBOSITY_ > 10:

            print("E1 = {0}".format(E1))
            print("E3 = {0}".format(3*E1))
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
            I0 = self.intensities_on_mesh(energy, theta, psi, harmonics)[0]
            flux = I0.sum(axis=(1, 2)) * dtheta * dpsi
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
            if raycing._VERBOSITY_ > 10:
                print("K={0}, {1} of {2}".format(K, iK+1, len(Ks)))
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

        if (self.R0 is not None) or (self.taper is not None):
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
            ucos = ww1S*tg +\
                wwuS*revgamma*\
                (-self.Ky*dirx*(sinx + alphaS/wuS*
                                (1 - cosx - tg*sinx)) +
                 self.Kx*diry*sinx + 0.125*revgamma*
                 (self.Kx**2 * sin2xph + self.Ky**2 * (sin2x -
                  2*alphaS/wuS*(tg**2 + cosx**2 + tg*sin2x))))
            eucos = np.exp(1j*ucos)
        elif R0 is not None:
            sinr0z = np.sin(wwuS*R0)
            cosr0z = np.cos(wwuS*R0)
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
            sindrs = np.sin(wwuS *(drs + 0.25 * zterm * revgamma))
            cosdrs = np.cos(wwuS *(drs + 0.25 * zterm * revgamma))

            eucosx = -sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -\
                       cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs
            eucosy = -sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +\
                       cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs
            eucos = eucosx + 1j*eucosy

            direction = dr/dist
            dirx = direction[0, :]
            diry = direction[1, :]
            dirz = direction[2, :]
        else:
            ucos = ww1S*tg + wwuS*revgamma*\
                (-self.Ky*ddphiS*sinx + self.Kx*ddpsiS*sinxph +
                 0.125*revgamma*(self.Ky**2 * sin2x +
                               self.Kx**2 * sin2xph))
            eucos = np.exp(1j*ucos)

        betax = taperC*self.Ky*revgamma*cosx
        betay = -self.Kx*revgamma*cosxph
        betaz = 1. - 0.5*(revgamma2 + betax*betax + betay*betay)

        betaPx = -self.Ky*(alphaS*cosx + taperC*sinx)
        betaPy = self.Kx*sinxph
        betaPz = 0.5*revgamma*\
            (self.Ky**2 * taperC*(alphaS*cosx**2 + taperC*sin2x)+
             self.Kx**2 * sin2xph)

        rkrel = 1./(1. - dirx*betax - diry*betay - dirz*betaz)
        eucos *= ag * rkrel**2
        bnx = dirx - betax
        bny = diry - betay
        bnz = dirz - betaz

        dirDotBetaP = dirx*betaPx + diry*betaPy + dirz*betaPz
        dirDotDmB = dirx*bnx + diry*bny + dirz*bnz

        Bsr = np.sum(eucos*wuS*revgamma*(bnx*dirDotBetaP - betaPx*dirDotDmB), axis=dim)
        Bpr = np.sum(eucos*wuS*revgamma*(bny*dirDotBetaP - betaPy*dirDotDmB), axis=dim)

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

        Bsr = np.complex(0)
        Bpr = np.complex(0)

        dirx = ddphiS
        diry = ddpsiS
        dirz = 1. - 0.5*(ddphiS**2 + ddpsiS**2)
        Nmx = self.Np if (R0 is not None or self.taper is not None) else 1
        if R0 is not None:
            sinr0z = np.sin(R0)
            cosr0z = np.cos(R0)

        for Nperiod in range(Nmx):
            if raycing._VERBOSITY_ > 80 and (self.taper is not None or\
                                             R0 is not None):
                print("Period {} out of {}".format(Nperiod+1, Nmx))
            for i in range(len(self.tg)):
                if self.taper is not None:
                    zloc = -(Nmx-1)*np.pi + Nperiod*PI2 + self.tg[i]
                    alphaS = self.taper/E2WC
                    taperC = 1. - alphaS*zloc/wuS
                    ucos = ww1S*zloc +\
                        wwuS*revgamma*\
                        (-self.Ky*dirx*(self.sintg[i] + alphaS/wuS*
                                        (1 - self.costg[i] -
                                         zloc*self.sintg[i])) +
                         self.Kx*diry*self.sintg[i] + 0.125*revgamma*
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
                    dr = R0 - np.expand_dims(rloc, 1)
                    dist = np.linalg.norm(dr, axis=0)

                    drs = 0.5*(dr[0, :]**2+dr[1, :]**2)/dr[2, :];

                    sinzloc = np.sin(wwuS * zloc*(1.-betam))
                    coszloc = np.cos(wwuS * zloc*(1.-betam))
                    sindrs = np.sin(wwuS *(drs + 0.25 * zterm * revgamma))
                    cosdrs = np.cos(wwuS *(drs + 0.25 * zterm * revgamma))

                    eucosx = -sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -\
                               cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs;
                    eucosy = -sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +\
                               cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs;
                    eucos = eucosx + 1j*eucosy

                    direction = dr/dist
                    dirx = direction[0, :]
                    diry = direction[1, :]
                    dirz = direction[2, :]

                else:
                    ucos = ww1S*self.tg[i] + wwuS*revgamma*\
                        (-self.Ky*ddphiS*self.sintg[i] + self.Kx*ddpsiS*self.sintgph[i] +
                         0.125*revgamma*(self.Ky**2 * sin2x[i] +
                                       self.Kx**2 * sin2xph[i]))
                    eucos = np.exp(1j*ucos)

                betax = taperC*self.Ky*revgamma*self.costg[i]
                betay = -self.Kx*revgamma*self.costgph[i]
                betaz = 1. - 0.5*(revgamma2 + betax*betax + betay*betay)

                betaPx = -self.Ky*(alphaS*self.costg[i] + taperC*self.sintg[i])
                betaPy = self.Kx*self.sintgph[i]
                betaPz = 0.5*revgamma*\
                    (self.Ky**2 * taperC*(alphaS*self.costg[i]**2 + taperC*sin2x[i])+
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
#            R0n = np.linalg.norm(R0v, axis=0)  # TODO: Only for spherical screen
            R0v *= self.R0*np.pi*2/self.L0  # /R0n
        else:
            R0v=None

        if NRAYS > 10:
            if self.filamentBeam:
                gamma = self.gamma
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
            scalarArgs = [self.cl_precisionF(self.R0*np.pi*2/self.L0)] #,  # R0
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
                           self.cl_precisionF(self.sintg), # Move outside
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
