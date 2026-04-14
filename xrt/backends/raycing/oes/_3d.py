# -*- coding: utf-8 -*-
#import copy
import numpy as np
from scipy.interpolate import griddata, RectBivariateSpline
from collections import defaultdict, deque

from .base import OE

try:
    from stl import mesh
    isSTLsupported = True
except ImportError:
    isSTLsupported = False


class MeshOE(OE):
    def __init__(self, *args, **kwargs):
        u"""
        Optical element defined by an STL mesh.

        The top surface is identified by selecting triangles whose surface
        normals have a positive (and typically largest) z-component. The
        corresponding vertices are extracted and used to reconstruct a
        continuous surface z = f(x, y).

        Depending on *surfaceHint*, the surface is approximated either by a
        polynomial fit or by a spline-based interpolation. In the spline mode,
        the scattered data are first interpolated onto a regular grid and then
        fitted with `scipy.interpolate.RectBivariateSpline` to obtain a smooth
        surface suitable for ray propagation and visualization.

        *fileName*: str
            Path to the STL file.

        *orientation*: str
            Axis-remapping string for converting STL coordinates into the xrt
            coordinate system. (X right-left, Y forward-backward, Z top-down).
            Default 'XYZ'.

        *recenter*: bool
            If True, the mesh is recentered so that the local origin
            corresponds to the geometric center of the top surface of the
            optical element.

        *surfaceHint*: str
            Hint for the surface reconstruction method:
                - 'flat'      : plane surface
                - 'quad'      : fit a 2nd-order polynomial surface
                - 'spline'    : use cubic spline interpolation on a regular
                                grid


        """

        fileName = kwargs.pop('fileName', None)
        orientation = kwargs.pop('orientation', 'XYZ')
        recenter = kwargs.pop('recenter', True)
        surfaceHint = kwargs.pop('surfaceHint', 'quad')
        super().__init__(*args, **kwargs)
        self.stl_mesh = None
        self.orientation = orientation
        self.recenter = recenter
        self.surfaceHint = surfaceHint

        self.fileName = fileName

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation
        if self.stl_mesh is not None:
            self.fit_surface()

    @property
    def recenter(self):
        return self._recenter

    @recenter.setter
    def recenter(self, recenter):
        self._recenter = recenter
        if self.stl_mesh is not None:
            self.fit_surface()

    @property
    def surfaceHint(self):
        return self._surfaceHint

    @surfaceHint.setter
    def surfaceHint(self, surfaceHint):
        self._surfaceHint = surfaceHint
        if self.stl_mesh is not None:
            self.fit_surface()

    @property
    def fileName(self):
        return self._fileName

    @fileName.setter
    def fileName(self, fileName):
        if isSTLsupported:
            try:
                self.read_file(fileName)
                self._fileName = fileName
                self.fit_surface()
            except Exception as e:
                raise e
                print("STL file import error", e)
        else:
            print("numpy-stl must be installed to work with STL models")

    def read_file(self, filename):
        self.stl_mesh = mesh.Mesh.from_file(filename)

    def fit_surface(self):

        def pkey(p, ndigits=8):
            return tuple(np.round(p, ndigits))

        if self.stl_mesh is None:
            return
        normals = np.array(self.stl_mesh.normals)
        faces = self.stl_mesh.data
        xrt_ax = {'X': 0, 'Y': 1, 'Z': 2}
        # TODO: catch exception
        z_ax = xrt_ax[self.orientation[2].upper()]

        x_arr = getattr(self.stl_mesh, self.orientation[0].lower())
        y_arr = getattr(self.stl_mesh, self.orientation[1].lower())
        z_arr = getattr(self.stl_mesh, self.orientation[2].lower())

        topSurfIndex = np.where(normals[:, z_ax] > 0.1)[0]
        # we take z-coord of the last point in triangle. arbitrary choice
        z_coordinates = np.array(z_arr[topSurfIndex, 2])
        izmax = topSurfIndex[np.argmax(z_coordinates)]

        tri_keys = [[pkey(p) for p in face[1]] for face in faces]

        point_to_triangles = defaultdict(set)
        for ti, pts in enumerate(tri_keys):
            for pt in pts:
                point_to_triangles[pt].add(ti)

        candidate_set = set(topSurfIndex.tolist())

        topSurfIndexArr = [izmax]
        allowed = candidate_set - {izmax}
        queue = deque([izmax])

        while queue:
            tsi = queue.popleft()

            for pt in tri_keys[tsi]:
                for nei in point_to_triangles[pt]:
                    if nei in allowed:
                        allowed.remove(nei)
                        topSurfIndexArr.append(nei)
                        queue.append(nei)

        xs = np.array(x_arr[topSurfIndexArr]).flatten()
        ys = np.array(y_arr[topSurfIndexArr]).flatten()
        zs = np.array(z_arr[topSurfIndexArr]).flatten()

        self.limPhysX = np.array([np.min(xs), np.max(xs)])
        self.limPhysY = np.array([np.min(ys), np.max(ys)])

        if self.recenter:  # first stage. use original grid
            self.dcx = 0.5*(self.limPhysX[-1]+self.limPhysX[0])
            self.dcy = 0.5*(self.limPhysY[-1]+self.limPhysY[0])
            xs -= self.dcx
            ys -= self.dcy
            self.limPhysX -= self.dcx
            self.limPhysY -= self.dcy
            zs0 = np.min(zs)
            zs -= zs0

        self.dcz = 0
        dcz = 0

        planeCoords = np.vstack((xs, ys)).T

        uxy, ui = np.unique(planeCoords, axis=0, return_index=True)
        ux = uxy[:, 0]
        uy = uxy[:, 1]
        uz = zs[ui]

        if self.surfaceHint == 'quad':
            A = np.c_[ux**2, uy**2, ux*uy, ux, uy, np.ones_like(ux)]
            self.cpoly, *_ = np.linalg.lstsq(A, uz, rcond=None)
            dcz = self.cpoly[5]
            self.z_spline = None
            Rmer = 0.5 / self.cpoly[1]
            Rsag = 0.5 / self.cpoly[0]
            print(f'{Rmer=}, {Rsag=}')
        elif self.surfaceHint == 'spline':
            gridsizeX = int(10 * (self.limPhysX[-1] - self.limPhysX[0]))
            gridsizeY = int(10 * (self.limPhysY[-1] - self.limPhysY[0]))

            xgrid = np.linspace(self.limPhysX[0], self.limPhysX[-1],
                                gridsizeX)
            ygrid = np.linspace(self.limPhysY[0], self.limPhysY[-1],
                                gridsizeY)
            xmesh, ymesh = np.meshgrid(xgrid, ygrid, indexing='ij')
            zmesh = griddata((ux, uy), uz, (xmesh, ymesh),
                                         method='cubic')

            mask = np.isnan(zmesh)
            if np.any(mask):
                zmesh[mask] = np.nanmean(zmesh)
            self.z_spline = RectBivariateSpline(xgrid, ygrid, zmesh, s=1e-6)
            dcz = np.min(zmesh)
            self.cpoly = None

        if self.recenter:
            self.dcz = dcz

        self.points = np.array(self.stl_mesh.vectors).reshape(-1, 3) -\
            np.array([self.dcx, self.dcy, self.dcz + zs0])
        self.normals = np.repeat(self.stl_mesh.normals, 3, axis=0)

    def local_z(self, x, y):
        if getattr(self, 'z_spline', None) is not None:
            z = self.z_spline.ev(x, y) - self.dcz
        elif getattr(self, 'cpoly', None) is not None:
            z = self.cpoly[0]*x**2 + self.cpoly[1]*y**2 + self.cpoly[2]*x*y +\
                self.cpoly[3]*x + self.cpoly[4]*y + self.cpoly[5] - self.dcz
        else:  # flat
            z = np.zeros_like(x)
        return z

    def local_n(self, x, y):
        if getattr(self, 'z_spline', None) is not None:
            a = self.z_spline.ev(y, x, dx=0, dy=1)
            b = self.z_spline.ev(y, x, dx=1, dy=0)
        elif getattr(self, 'cpoly', None) is not None:
            a = 2*self.cpoly[0]*x + self.cpoly[2]*y + self.cpoly[3]
            b = 2*self.cpoly[1]*y + self.cpoly[2]*x + self.cpoly[4]
        else:  # flat
            a = b = np.zeros_like(x)

        norm = np.sqrt(a**2+b**2+1.)
        return [-a/norm, -b/norm, 1./norm]
