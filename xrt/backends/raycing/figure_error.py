# -*- coding: utf-8 -*-
"""
Surface Roughness
----------------

Basic containers for surface roughness generators.

"""

from __future__ import print_function
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "29 Nov 2025"
__all__ = ('RandomRoughness', 'GaussianBump', 'Waviness',
           'FigureErrorImported')


import numpy as np
from scipy import interpolate
from pathlib import Path
from .. import raycing

maxFeHalfSize = 100  # Intentionally smaller size than OE to speed up spline


class FigureErrorBase():
    """
    Base class for distorted surfaces. Provides the functionality for
    height maps generation and diagnostics. Height map calculation function
    must be implemented in a subclass.

    """
    def __init__(self, name='', baseFE=None,
                 limPhysX=None, limPhysY=None, gridStep=0.5,
                 **kwargs):
        """
        *name*: str
            Attribute to store instance name

        *baseFE*: None or instance of `FigureError` subclass
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


        """
        self.name = name
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())
        self.bl = kwargs.get('bl')
        self._baseFE = baseFE
        self._gridStep = gridStep  # [mm]
        if limPhysX is None:
            self._limPhysX = raycing.Limits([-maxFeHalfSize,
                                             maxFeHalfSize])
        else:
            self._limPhysX = raycing.Limits(limPhysX)
        if limPhysY is None:
            self._limPhysY = raycing.Limits([-maxFeHalfSize,
                                             maxFeHalfSize])
        else:
            self._limPhysY = raycing.Limits(limPhysY)
        self.build_spline()

    @property
    def baseFE(self):
        if raycing.is_valid_uuid(self._baseFE) and self.bl is not None:
            fe = self.bl.fesDict.get(self._baseFE)
        else:
            fe = self._baseFE
        return fe

    @baseFE.setter
    def baseFE(self, baseFE):
        self._baseFE = baseFE
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
        if limPhysX is None:
            self._limPhysX = raycing.Limits([-maxFeHalfSize,
                                             maxFeHalfSize])
        else:
            self._limPhysX = raycing.Limits(limPhysX)
        self.build_spline()

    @property
    def limPhysY(self):
        return self._limPhysY

    @limPhysY.setter
    def limPhysY(self, limPhysY):
        if limPhysY is None:
            self._limPhysY = raycing.Limits([-maxFeHalfSize,
                                             maxFeHalfSize])
        else:
            self._limPhysY = raycing.Limits(limPhysY)
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
        """This function must be overriden in subclass."""
        x, y = self.get_grids()
        return np.zeros_like(x)

    def build_spline(self):
#        x, y = self.get_grids()
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
        # z spline is built in nm; x, y in mm
        if spl == 'RBS':
            a = -self.local_z_spline.ev(y, x, dx=0, dy=1) * 1e-6
            b = -self.local_z_spline.ev(y, x, dx=1, dy=0) * 1e-6
#        else:
#            a = self.local_z_spline.ev(x, y, dx=1, dy=0) * 1e-6
#            b = self.local_z_spline.ev(x, y, dx=0, dy=1) * 1e-6

        if shape is not None:
            a = a.reshape(shape)
            b = b.reshape(shape)

#        return [np.zeros_like(a), 1e-3*np.ones_like(b)]
        return [np.arctan(b), np.arctan(a)]  # [d_pitch, d_roll]

    def next_pow2(self, n):
        return 1 << int(np.ceil(np.log2(n)))


class FigureErrorImported(FigureErrorBase):
    """Figure error from file."""

    def __init__(self, fileName=None, recenter=False, orientation="XYZ",
                 **kwargs):
        """
        *fileName*: str, path.
            path to surface distortion map file.
        *recenter*: bool
            moves center to (0, 0)
        *orientation*: str
            redefines the order of columns in the input file. default "XYZ".


        """
        self.surfArrays = {}
        self.zGrid = None
        super().__init__(**kwargs)
        self._recenter = recenter
        self._orientation = orientation
        self.fileName = fileName

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation
        if self.surfArrays:
            self.align_arrays()
            self.build_spline()

    @property
    def recenter(self):
        return self._recenter

    @recenter.setter
    def recenter(self, recenter):
        self._recenter = recenter
        if self.surfArrays:
            self.align_arrays()
            self.build_spline()

    @property
    def fileName(self):
        return self._fileName

    @fileName.setter
    def fileName(self, fileName):
        self._fileName = fileName
        if fileName is not None:
            self.read_file(fileName)
            self.align_arrays()
            self.build_spline()

    def read_file(self, fileName):
        path = Path(fileName)
        if not path.is_file():
            return

        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 3:
            return

        try:
            xL, yL, zL = np.loadtxt(path, unpack=True)
        except Exception as e:  # TODO: better exceptions
            print(e)
            return

        self.surfArrays = {'x': xL, 'y': yL, 'z': zL}

    def align_arrays(self):
        orientlc = str(self.orientation).lower()
        axes = {'x': orientlc[0], 'y': orientlc[1], 'z': orientlc[-1]}
        xL = self.surfArrays.get(axes['x'])
        yL = self.surfArrays.get(axes['y'])
        zL = self.surfArrays.get(axes['z'])

        dx = dy = 0
        xmin, xmax = np.min(xL), np.max(xL)
        ymin, ymax = np.min(yL), np.max(yL)
        if self.recenter:
            dx = -0.5*(xmin+xmax)
            dy = -0.5*(ymin+ymax)

        self._limPhysX = [xmin+dx, xmax+dx]
        self._limPhysY = [ymin+dy, ymax+dy]

        tol = None  # rounding for real-world positions. try 1e-3 for noisy xs
        if tol is not None:
            x_rounded = np.round(xL/tol)*tol
            y_rounded = np.round(yL/tol)*tol
            ux = np.unique(x_rounded)
            uy = np.unique(y_rounded)
        else:
            ux = np.unique(xL)
            uy = np.unique(yL)

        nx, ny = len(ux), len(uy)

        if nx*ny != len(xL):
            print("Input data does not form a grid")
            return

        self.nx = nx
        self.ny = ny

        row_maj = np.all(np.diff(xL[:nx]) > 0) and np.all(yL[:nx] == yL[0])

        if row_maj:
            zm = zL.reshape((ny, nx))
        else:
            zm = zL.reshape((nx, ny)).T

        self.zGrid = zm

    def get_dimensions(self):
        xlength = np.abs(self.limPhysX[-1] - self.limPhysX[0])
        ylength = np.abs(self.limPhysY[-1] - self.limPhysY[0])
        nxi = xlength / self.gridStep
        nyi = ylength / self.gridStep
        if not hasattr(self, 'nx'):
            self.nx = max(self.next_pow2(nxi), 128)
        if not hasattr(self, 'ny'):
            self.ny = max(self.next_pow2(nyi), 128)
        self.dx = xlength / self.nx
        self.dy = ylength / self.ny
        return

    def generate_profile(self):
        """This function must be overriden in subclass."""
        xg, yg = self.get_grids()
        base_z = np.zeros_like(xg)
        if self.baseFE is not None and hasattr(self.baseFE, 'local_z'):
            base_z = self.baseFE.local_z(xg, yg) * 1e6  # local_z returns z in mm

        if self.surfArrays and self.zGrid is not None:
            z = self.zGrid
        else:  # fallback to flat
            z = np.zeros_like(xg)
        return z + base_z

class RandomRoughness(FigureErrorBase):
    """Random roughness map."""

    def __init__(self, rms=1., corrLength=None, seed=None, **kwargs):
        """
        *rms*: float
            Root Mean Square roughness in [nm]

        *corrLength*: float
            Correlation length in [mm].

        *seed*: None or int.
            Seed number for numpy random number generator. Any number
            0 < seed < 2^128-1


        """
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
        if self.baseFE is not None and hasattr(self.baseFE, 'local_z'):
            base_z = self.baseFE.local_z(xg, yg) * 1e6  # local_z returns z in mm

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


class GaussianBump(FigureErrorBase):
    """Height profile defined by the Gaussian function."""

    def __init__(self, bumpHeight=100., sigmaX=1., sigmaY=1., **kwargs):
        """
        *bumpHeight*: float
            Hight at the peak in [nm]

        *sigmaX*, *sigmaY*: float
            Standard deviation along X and Y axes in [mm].


        """
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

        if self.baseFE is not None and hasattr(self.baseFE, 'local_z'):
            base_z = self.baseFE.local_z(x, y) * 1e6  # local_z returns z in mm

        z = self.bumpHeight *\
            np.exp(-x**2/self.sigmaX**2 - y**2/self.sigmaY**2)
        return z + base_z


class Waviness(FigureErrorBase):
    """Surface waviness following 2d cosine law."""

    def __init__(self, amplitude=10., xWaveLength=20., yWaveLength=50.,
                 **kwargs):
        """
        *amplitude*: float
            Cosine amplitude in [nm]

        *xWaveLength*, *yWaveLength*: float
            Wave period along the X and Y axes in [mm].


        """
        self._amplitude = amplitude
        self._xWaveLength = xWaveLength
        self._yWaveLength = yWaveLength
        super().__init__(**kwargs)

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        self._amplitude = amplitude
        self.build_spline()

    @property
    def xWaveLength(self):
        return self._xWaveLength

    @xWaveLength.setter
    def xWaveLength(self, xWaveLength):
        self._xWaveLength = xWaveLength
        self.build_spline()

    @property
    def yWaveLength(self):
        return self._yWaveLength

    @yWaveLength.setter
    def yWaveLength(self, yWaveLength):
        self._yWaveLength = yWaveLength
        self.build_spline()

    def generate_profile(self):
        x, y = self.get_grids()
        base_z = np.zeros_like(x)

        if self.baseFE is not None and hasattr(self.baseFE, 'local_z'):
            base_z = self.baseFE.local_z(x, y) * 1e6   # local_z returns z in mm

        z = self.amplitude * np.cos(2*np.pi*x/self.xWaveLength) *\
            np.cos(2*np.pi*y/self.yWaveLength)
        return z + base_z
