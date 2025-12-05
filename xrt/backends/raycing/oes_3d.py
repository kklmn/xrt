# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy import interpolate

from .oes_base import OE

try:
    from stl import mesh
    isSTLsupported = True
except ImportError:
    isSTLsupported = False


class MeshOE(OE):
    def __init__(self, *args, **kwargs):
        u"""
        Optical element loaded from an STL mesh. The algorithm identifies the
        “top” surface by selecting triangles whose surface normals have the
        largest z-component. The extracted points are then fitted with
        `scipy.interpolate.SmoothBivariateSpline` on a 10x10 µm² grid to obtain
        a smooth optical surface suitable for propagation and visualization.

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


        """

        fileName = kwargs.pop('fileName', None)
        orientation = kwargs.pop('orientation', 'XYZ')
        recenter = kwargs.pop('recenter', True)
        super().__init__(*args, **kwargs)
        self.stl_mesh = None
        self.orientation = orientation
        self.recenter = recenter

        self.fileName = fileName

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation
        if self.stl_mesh is not None:
            self.build_surface_spline()

    @property
    def recenter(self):
        return self._recenter

    @recenter.setter
    def recenter(self, recenter):
        self._recenter = recenter
        if self.stl_mesh is not None:
            self.build_surface_spline()

    @property
    def fileName(self):
        return self._fileName

    @fileName.setter
    def fileName(self, fileName):
        if isSTLsupported:
            try:
                self.read_file(fileName)
                self._fileName = fileName
                self.build_surface_spline()
            except Exception as e:
                print("STL file improt error", e)
        else:
            print("numpy-stl must be installed to work with STL models")

    def read_file(self, filename):
        self.stl_mesh = mesh.Mesh.from_file(filename)

    def build_surface_spline(self):
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

        topSurfIndex = np.where(normals[:, z_ax] > 0.01)[0]
        # we take z-coord of the last point in triangle. arbitrary choice
        z_coordinates = np.array(z_arr[topSurfIndex, 2])
        izmax = topSurfIndex[np.argmax(z_coordinates)]
        topSurfIndexArr = [izmax]
        topSurfCoords = faces[izmax][1].tolist()

        tmptsi = copy.copy(topSurfIndex.tolist())
        isNrPtsInc = True

        while isNrPtsInc:
            isNrPtsInc = False
            for tsi in tmptsi:
                for point in faces[tsi][1]:
                    if list(point) in topSurfCoords:
                        topSurfIndexArr.append(tsi)
                        topSurfCoords.extend(faces[tsi][1].tolist())
                        tmptsi.remove(tsi)
                        isNrPtsInc = True
                        break

        xs = np.array(x_arr[topSurfIndexArr]).flatten()
        ys = np.array(y_arr[topSurfIndexArr]).flatten()
        zs = np.array(z_arr[topSurfIndexArr]).flatten()

        self.limPhysX = np.array([np.min(xs), np.max(xs)])
        self.limPhysY = np.array([np.min(ys), np.max(ys)])

        if self.recenter:  # first stage
            self.dcx = 0.5*(self.limPhysX[-1]+self.limPhysX[0])
            self.dcy = 0.5*(self.limPhysY[-1]+self.limPhysY[0])
            xs -= self.dcx
            ys -= self.dcy
            self.limPhysX -= self.dcx
            self.limPhysY -= self.dcy
            zs -= np.min(zs)

        self.dcz = 0

        planeCoords = np.vstack((xs, ys)).T

        uxy, ui = np.unique(planeCoords, axis=0, return_index=True)
        uz = zs[ui]

        self.z_spline = interpolate.SmoothBivariateSpline(
                uxy[:, 0], uxy[:, 1], uz, s=len(uz) * 1e-6)

        self.gridsizeX = int(10 * (self.limPhysX[-1] - self.limPhysX[0]))
        self.gridsizeY = int(10 * (self.limPhysY[-1] - self.limPhysY[0]))

        xgrid = np.linspace(self.limPhysX[0], self.limPhysX[-1],
                            self.gridsizeX)
        ygrid = np.linspace(self.limPhysY[0], self.limPhysY[-1],
                            self.gridsizeY)
        xmesh, ymesh = np.meshgrid(xgrid, ygrid, indexing='ij')
        zgrid = self.z_spline.ev(xmesh, ymesh)

        if self.recenter:
            self.dcz = np.min(zgrid)

        self.points = np.array(self.stl_mesh.vectors).reshape(-1, 3) - np.array(
                [self.dcx, self.dcy, self.dcz])
        self.normals = np.repeat(self.stl_mesh.normals, 3, axis=0)

    def local_z(self, x, y):
        if hasattr(self, 'z_spline'):
            z = self.z_spline.ev(x, y) - self.dcz
        else:
            z = np.zeros_like(x)
        return z

    def local_n(self, x, y):
        if hasattr(self, 'z_spline'):
            a = self.z_spline.ev(x, y, dx=1, dy=0)
            b = self.z_spline.ev(x, y, dx=0, dy=1)
        else:
            a = b = np.zeros_like(x)

        norm = np.sqrt(a**2+b**2+1.)
        return [-a/norm, -b/norm, 1./norm]
