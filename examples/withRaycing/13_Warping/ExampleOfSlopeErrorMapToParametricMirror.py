# -*- coding: utf-8 -*-
__date__ = "20 Jan 2024"
# import sys
# sys.path.append(r"C:\Ray-tracing")
import numpy as np
from scipy import ndimage
# import pickle

import xrt.backends.raycing.oes as roe


class EllipticalMirrorParamNOM(roe.EllipticalMirrorParam):
    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        super().__init__(*args, **kwargs)
        self.nom_read()

    def __pop_kwargs(self, **kwargs):
        self.waviness = kwargs.pop('figureError')  # file name
        return kwargs

    def nom_read(self):
        # here, the file self.waviness has this structure:
        # x[mm]  y[mm]  z[nm]
        # 30.00  29.75  -31.498373
        # 30.50  29.75  -32.992258
        # 31.00  29.75  -33.864061
        # 31.50  29.75  -34.240630
        # ...
        xL, yL, zL = np.loadtxt(self.waviness, unpack=True)
        nX = (yL == yL[0]).sum()
        nY = (xL == xL[0]).sum()
        x = xL[:nX]
        y = yL[::nX]
        z = zL.reshape((nY, nX))
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        b, a = np.gradient(z)  # NOM x is along (our y) and y is across (our x)
        # a, b = np.gradient(z)  # NOM x is across and y is along
        a = np.arctan(a/dy)
        b = np.arctan(b/dx)
        self.nom_rmsA = ((a**2).sum() / (nX * nY))**0.5
        self.nom_rmsB = ((b**2).sum() / (nX * nY))**0.5
        # self.nom_splineZ = ndimage.spline_filter(z.T) * 1e-6  # mm to nm
        # self.nom_splineA = ndimage.spline_filter(a.T) * 1e-6  # rad to µrad
        # self.nom_splineB = ndimage.spline_filter(b.T) * 1e-6  # rad to µrad
        self.nom_splineZ = z.T * 1e-6  # mm to nm
        self.nom_splineA = a.T * 1e-6  # rad to µrad
        self.nom_splineB = b.T * 1e-6  # rad to µrad
        self.nom_nX = nX
        self.nom_nY = nY
        self.nom_x = x
        self.nom_y = y

    def local_r_distorted(self, s, phi):
        r = self.local_r(s, phi)
        x, y, z = self.param_to_xyz(s, phi, r)
        # if NOM x is along (our y) and y is across (our x):
        coords = np.array([
            (y/(self.nom_x[-1]-self.nom_x[0]) + 0.5) * (self.nom_nX-1),
            (x/(self.nom_y[-1]-self.nom_y[0]) + 0.5) * (self.nom_nY-1)])
        # coords.shape = (2, self.nrays)
        # z += ndimage.map_coordinates(self.nom_splineZ, coords, prefilter=True)
        z += ndimage.map_coordinates(self.nom_splineZ, coords, order=1)
        s1, phi1, r1 = self.xyz_to_param(x, y, z)
        return r1 - r

    def local_n_distorted(self, s, phi):
        r = self.local_r(s, phi)
        x, y, z = self.param_to_xyz(s, phi, r)
        # if NOM x is along (our y) and y is across (our x):
        coords = np.array([
            (y/(self.nom_x[-1]-self.nom_x[0]) + 0.5) * (self.nom_nX-1),
            (x/(self.nom_y[-1]-self.nom_y[0]) + 0.5) * (self.nom_nY-1)])
        # coords.shape = (2, self.nrays)
        # a = ndimage.map_coordinates(self.nom_splineA, coords, prefilter=True)
        # b = ndimage.map_coordinates(self.nom_splineB, coords, prefilter=True)
        a = ndimage.map_coordinates(self.nom_splineA, coords, order=1)
        b = ndimage.map_coordinates(self.nom_splineB, coords, order=1)
        return -a, -b
