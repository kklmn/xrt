# -*- coding: utf-8 -*-
"""
Surface Roughness
----------------

Basic containers for surface roughness generators.

"""

from __future__ import print_function
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "29 Nov 2025"

import numpy as np
from scipy import interpolate
from .. import raycing


class FigureError():
    """ Base class for distorted surfaces. Provides the functionality for
    height maps generation and diagnostics. Height map calculation function
    must be implemented in a subclass.
    
        *name*: str
            Attribute to store instance name

        *base*: None or instance of `FigureError` subclass
            Can be used to create complex maps (see examples).

        *gridStep*: float.
            Grid step in [mm]. Used indirectly to control the grid size. Real
            number of nodes will be picked as the next power of 2 in order to
            accelerate Fourier Transforms widely used for map generation. TBD

        *limPhysX* and *limPhysY*: [*min*, *max*] where *min*, *max* are
            floats floats. All in [mm].
            Physical dimension = local coordinate of the corresponding edge.
            Same as in `OE`. Ideally should be the same as in the base optical
            element, or at least not smaller than the expected beam footprint.

        *fileName*: str, path.
            path to surface distortion map file.
            
    
    """
    def __init__(self, name='', base=None,
                 limPhysX=None, limPhysY=None, gridStep=0.5, fileName=None,
                 **kwargs):
        self.name = name
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        self._base = base
        self._gridStep = gridStep  # [mm]
        self._limPhysX = limPhysX
        self._limPhysY = limPhysY
        self._fileName = fileName
        self.build_spline()

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, base):
        self._base = base
        self.build_spline()

    @property
    def gridStep(self):
        return self._gridStep

    @gridStep.setter
    def gridStep(self, gridStep):
        self._gridStep = gridStep
        self.build_spline()

    @property
    def limPhysX(self):
        return self._limPhysX

    @limPhysX.setter
    def limPhysX(self, limPhysX):
        self._limPhysX = limPhysX
        self.build_spline()

    @property
    def limPhysY(self):
        return self._limPhysY

    @limPhysY.setter
    def limPhysY(self, limPhysY):
        self._limPhysY = limPhysY
        self.build_spline()

    def get_rms(self):
        z = self.get_z_grid()
        zAvg = z - z.mean()
        rms = np.sqrt((zAvg**2).mean())
        return rms

    def get_rms_slope(self):
        xm, ym = self.get_grids()
        a, b, c = self.local_n(xm, ym)
        rms_slope = np.sqrt((b**2).mean())
        return rms_slope

    def get_z_grid(self):
        xm, ym = self.get_grids()
#        xf = xm.flatten()
#        yf = ym.flatten()
        return self.local_z(xm, ym)

    def get_dimensions(self):
        xlength = np.abs(self.limPhysX[-1] - self.limPhysX[0])
        ylength = np.abs(self.limPhysY[-1] - self.limPhysY[0])
        nxi = xlength / self.gridStep
        nyi = ylength / self.gridStep
        self.nx = max(self.next_pow2(nxi), 128)
        self.ny = max(self.next_pow2(nyi), 128)
        self.dx = xlength / self.nx
        self.dy = ylength / self.ny
        return

    def get_grids(self):
        self.get_dimensions()
        xgrid = np.linspace(np.min(self.limPhysX),
                            np.max(self.limPhysX),
                            self.nx)
        ygrid = np.linspace(np.min(self.limPhysY),
                            np.max(self.limPhysY),
                            self.ny)
        xm, ym = np.meshgrid(xgrid, ygrid)
        return xm, ym

    def get_psd(self):  # TODO: untested
        z = self.get_z_grid()
        nx, ny = z.shape
        zAvg = z - z.mean()
        H = np.fft.fft2(zAvg)
        H = np.fft.fftshift(H)
        PSD = (np.abs(H)**2) / (nx * ny)
        dx = np.abs(self.limPhysX[-1] - self.limPhysX[0]) / nx
        dy = np.abs(self.limPhysY[-1] - self.limPhysY[0]) / ny
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        return KX, KY, PSD

    def generate_profile(self):
        x, y = self.get_grids()
        return np.zeros_like(x)

    def build_spline(self):
        x, y = self.get_grids()
        z = self.generate_profile()

        xgrid = np.linspace(np.min(self.limPhysX),
                            np.max(self.limPhysX),
                            self.nx)
        ygrid = np.linspace(np.min(self.limPhysY),
                            np.max(self.limPhysY),
                            self.ny)
        self.local_z_spline = interpolate.RectBivariateSpline(
                ygrid, xgrid, z)

    def local_z(self, x, y):
        spl = 'RBS'  # 'SBS'
        shape = None

        if not hasattr(x, 'shape'):
            x = np.array(x)
            y = np.array(y)

        if len(x.shape) > 1:
            shape = x.shape
            x = x.flatten()
            y = y.flatten()

        if spl == 'RBS':
            z = self.local_z_spline.ev(y, x)
        else:
            z = self.local_z_spline.ev(x, y)

        if shape is not None:
            z = z.reshape(shape)

        return z * 1e-6  # convert from nm to mm

    def local_n(self, x, y):
        spl = 'RBS'  # 'SBS'
        shape = None

        if not hasattr(x, 'shape'):
            x = np.array(x)
            y = np.array(y)

        if len(x.shape) > 1:
            shape = x.shape
            x = x.flatten()
            y = y.flatten()
        # z spline is built in nm, x, y in mm
        if spl == 'RBS':
            a = self.local_z_spline.ev(y, x, dx=1, dy=0) * 1e-6
            b = self.local_z_spline.ev(y, x, dx=0, dy=1) * 1e-6
        else:
            a = self.local_z_spline.ev(x, y, dx=1, dy=0) * 1e-6
            b = self.local_z_spline.ev(x, y, dx=0, dy=1) * 1e-6

        if shape is not None:
            a = a.reshape(shape)
            b = b.reshape(shape)

        norm = np.sqrt(a**2+b**2+1.)
        return [-a/norm, -b/norm, 1./norm]

    def next_pow2(self, n):
        return 1 << int(np.ceil(np.log2(n)))


class RandomRoughness(FigureError):
    """Random roughness map.
    
        *rms*: float
            Root Mean Square roughness in [nm]

        *corrLength*: float
            Correlation length in [mm].

        *seed*: None or int.
            Seed number for numpy random number generator. Any number
            0 < seed < 2^128-1
    
    """
    def __init__(self, rms=1., corrLength=None, seed=None, **kwargs):
        self._rms = rms
        self._corrLength = corrLength
        if seed is None:
            seed = np.random.SeedSequence().entropy
        self._seed = seed
        super().__init__(**kwargs)

    @property
    def rms(self):
        return self._rms

    @rms.setter
    def rms(self, rms):
        self._rms = rms
        self.build_spline()

    @property
    def corrLength(self):
        return self._corrLength

    @corrLength.setter
    def corrLength(self, corrLength):
        self._corrLength = corrLength
        self.build_spline()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        if seed is None:
            seed = np.random.SeedSequence().entropy
        self._seed = seed
        self.build_spline()

    def generate_profile(self):
        rng = np.random.default_rng(self.seed)
        xg, yg = self.get_grids()

        base_z = np.zeros_like(xg)

        if self.base is not None and hasattr(self.base, 'local_z'):
            base_z = self.base.local_z(yg, xg) * 1e6  # local_z returns z in mm

        z = rng.normal(loc=0.0, scale=1.0, size=(self.ny, self.nx))
        if self.corrLength is not None:
            Z = np.fft.rfft2(z)
            kx = 2 * np.pi * np.fft.rfftfreq(self.nx, d=self.dx)
            ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
            KX, KY = np.meshgrid(kx, ky, indexing='xy')
            K2 = KX**2 + KY**2
            filter_k = np.exp(-0.5*K2*self.corrLength**2)
            zf = Z * filter_k
            z = np.fft.irfft2(zf, s=(self.ny, self.nx))

        z -= z.mean()
        current_rms = np.sqrt((z**2).mean())
        if current_rms > 0:
            z *= (self.rms / current_rms)

        return z + base_z


class GaussianBump(FigureError):
    """Height profile defined by the Gaussian function.
    
        *bumpHeight*: float
            Hight at the peak in [nm]

        *sigmaX*, *sigmaY*: float
            Standard deviation along X and Y axes in [mm].
    
    """
    def __init__(self, bumpHeight=100., sigmaX=1., sigmaY=1., **kwargs):
        self._bumpHeight = bumpHeight
        self._sigmaX = sigmaX
        self._sigmaY = sigmaY
        super().__init__(**kwargs)

    @property
    def bumpHeight(self):
        return self._bumpHeight

    @bumpHeight.setter
    def bumpHeight(self, bumpHeight):
        self._bumpHeight = bumpHeight
        self.build_spline()

    @property
    def sigmaX(self):
        return self._sigmaX

    @sigmaX.setter
    def sigmaX(self, sigmaX):
        self._sigmaX = sigmaX
        self.build_spline()

    @property
    def sigmaY(self):
        return self._sigmaY

    @sigmaY.setter
    def sigmaY(self, sigmaY):
        self._sigmaY = sigmaY
        self.build_spline()

    def generate_profile(self):
        x, y = self.get_grids()

        base_z = np.zeros_like(x)
        if self.base is not None and hasattr(self.base, 'local_z'):
            base_z = self.base.local_z(x, y) * 1e6  # local_z returns z in mm

        z = self.bumpHeight *\
            np.exp(-x**2/self.sigmaX**2 - y**2/self.sigmaY**2)
        return z + base_z


class Waviness(FigureError):
    """Surface waviness following 2d cosine law.
    
        *amplitude*: float
            Cosine amplitude in [nm]

        *xWaveLength*, *yWaveLength*: float
            Wave period along the X and Y axes in [mm].
    
    """
    def __init__(self, amplitude=10., xWaveLength=20., yWaveLength=50.,
                 **kwargs):
        self.amplitude = amplitude
        self.xWaveLength = xWaveLength
        self.yWaveLength = yWaveLength
        super().__init__(**kwargs)

    def generate_profile(self):
        x, y = self.get_grids()

        base_z = np.zeros_like(x)
        if self.base is not None and hasattr(self.base, 'local_z'):
            base_z = self.base.local_z(x, y) * 1e6   # local_z returns z in mm

        z = self.amplitude * np.cos(2*np.pi*x/self.xWaveLength) *\
            np.cos(2*np.pi*y/self.yWaveLength)
        return z + base_z
