# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy import interpolate, ndimage

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

        self.orientation = orientation
        self.recenter = recenter
        if isSTLsupported:
            self.load_STL(filename)
        else:
            raise ImportError(
                "numpy-stl must be installed to work with STL models")

    def load_STL(self, filename):
        self.stl_mesh = mesh.Mesh.from_file(filename)

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

        if self.recenter:
            self.dcx = 0.5*(self.limPhysX[-1]+self.limPhysX[0])
            self.dcy = 0.5*(self.limPhysY[-1]+self.limPhysY[0])
            xs -= self.dcx
            ys -= self.dcy
            self.limPhysX -= self.dcx
            self.limPhysY -= self.dcy
            zs -= np.min(zs)

        planeCoords = np.vstack((xs, ys)).T

        uxy, ui = np.unique(planeCoords, axis=0, return_index=True)
        uz = zs[ui]

        # TODO: catch exception
        self.z_spline = interpolate.RBFInterpolator(uxy, uz, kernel='cubic')

        self.gridsizeX = int(10 * (self.limPhysX[-1] - self.limPhysX[0]))
        self.gridsizeY = int(10 * (self.limPhysY[-1] - self.limPhysY[0]))

        xgrid = np.linspace(self.limPhysX[0], self.limPhysX[-1],
                            self.gridsizeX)
        ygrid = np.linspace(self.limPhysY[0], self.limPhysY[-1],
                            self.gridsizeY)
        xmesh, ymesh = np.meshgrid(xgrid, ygrid, indexing='ij')

        xygrid = np.vstack((xmesh.flatten(), ymesh.flatten())).T
        zgrid = self.z_spline(xygrid).reshape(self.gridsizeX, self.gridsizeY)

        self.x_grad, self.y_grad = np.gradient(zgrid)
        # self.a_spline = ndimage.spline_filter(x_grad/(xgrid[1]-xgrid[0]))
        # self.b_spline = ndimage.spline_filter(y_grad/(ygrid[1]-ygrid[0]))

    def local_z(self, x, y):
        pnt = np.array((x, y)).T
        z = self.z_spline(pnt)
        return z

    def local_n(self, x, y):
        coords = np.array(
            [(x-self.limPhysX[0]) /
             (self.limPhysX[-1]-self.limPhysX[0]) * (self.gridsizeX-1),
             (y-self.limPhysY[0]) /
             (self.limPhysY[-1]-self.limPhysY[0]) * (self.gridsizeY-1)])
        a = ndimage.map_coordinates(self.x_grad, coords, order=1)
        b = ndimage.map_coordinates(self.y_grad, coords, order=1)
        norm = np.sqrt(a**2+b**2+1.)
        return [-a/norm, -b/norm, 1./norm]
