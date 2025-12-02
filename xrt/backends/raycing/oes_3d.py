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


class OEfrom3DModel(OE):
    def __init__(self, *args, **kwargs):
        """
        *filename*: str
            Path to STL file.

        *orientation*: str
            Sequence of axes to match xrt standard (X right-left,
            Y forward-backward, Z top-down). Default 'XYZ'.

        *recenter*: bool
            Parameter defines whether to move local origin to the center of OE


        """

        filename = kwargs.pop('filename', None)
        orientation = kwargs.pop('orientation', 'XYZ')
        recenter = kwargs.pop('recenter', True)
        super().__init__(*args, **kwargs)

        self.filename = filename
        self.orientation = orientation
        self.recenter = recenter
        if isSTLsupported:
            self.load_STL(filename)
        else:
            raise ImportError(
                "numpy-stl must be installed to work with STL models")

    def load_STL(self, filename):
        stl_mesh = mesh.Mesh.from_file(filename)

        normals = np.array(stl_mesh.normals)
        faces = stl_mesh.data
        xrt_ax = {'X': 0, 'Y': 1, 'Z': 2}
        # TODO: catch exception
        z_ax = xrt_ax[self.orientation[2].upper()]

        x_arr = getattr(stl_mesh, self.orientation[0].lower())
        y_arr = getattr(stl_mesh, self.orientation[1].lower())
        z_arr = getattr(stl_mesh, self.orientation[2].lower())

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

        # TODO: catch exceptions
#        self.z_spline = interpolate.RBFInterpolator(uxy, uz, kernel='cubic')
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

        self.points = np.array(stl_mesh.vectors).reshape(-1, 3) - np.array(
                [self.dcx, self.dcy, self.dcz])
        self.normals = np.repeat(stl_mesh.normals, 3, axis=0)

    def local_z(self, x, y):
        z = self.z_spline.ev(x, y) - self.dcz
        return z

    def local_n(self, x, y):
        a = self.z_spline.ev(x, y, dx=1, dy=0)
        b = self.z_spline.ev(x, y, dx=0, dy=1)

        norm = np.sqrt(a**2+b**2+1.)
        return [-a/norm, -b/norm, 1./norm]
