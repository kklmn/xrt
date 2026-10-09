# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.optimize import least_squares
from collections import defaultdict, deque
from pathlib import Path
from .base import OE

try:
    from stl import mesh
    isSTLsupported = True
except ImportError:
    isSTLsupported = False


class MeshOE(OE):
    """Optical element defined by an STL mesh."""
    hiddenParams = ['surfaceHint']
    fitDiagnostics = ('RsagFit', 'RmerFit', 'conicXFit', 'conicYFit',
                      'fitRmsError')
    _axisHints = ('flat', 'parabolic', 'circular', 'elliptical', 'spline')

    def __init__(self, *args, **kwargs):
        u"""
        The top surface is the connected, uppermost set of triangles whose
        unit normals have a significant z-component, regardless of triangle
        winding. The corresponding vertices are extracted and used to
        reconstruct a continuous surface z = f(x, y).

        The fitted surface uses independent X/Y conic profiles. Matching
        parabolic hints include the quadratic cross term; matching circular
        hints use an exact torus; matching elliptical hints use a biconic.
        Different hints use additive profiles without cross terms.

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

        *surfaceHintX*: str
            Sagittal (X) profile: 'flat', 'parabolic', 'circular', or
            'elliptical'. Default 'parabolic'. 'spline' selects 2D cubic
            interpolation and sets both axis hints to 'spline'.

        *surfaceHintY*: str
            Meridional (Y) profile: 'flat', 'parabolic', 'circular', or
            'elliptical'. Default 'parabolic'. 'spline' selects 2D cubic
            interpolation and sets both axis hints to 'spline'.


        Diagnostics: *RsagFit* and *RmerFit* are fitted vertex radii in mm
        (infinite for flat axes). *conicXFit* and *conicYFit* are fitted conic
        constants. *fitRmsError* is the RMS height residual in micrometres,
        evaluated at the selected, unique STL surface vertices.


        """

        fileName = kwargs.pop('fileName', None)
        orientation = kwargs.pop('orientation', 'XYZ')
        recenter = kwargs.pop('recenter', True)
        surfaceHint = kwargs.pop('surfaceHint', None)
        explicitX = 'surfaceHintX' in kwargs
        explicitY = 'surfaceHintY' in kwargs
        surfaceHintX = kwargs.pop('surfaceHintX', 'parabolic')
        surfaceHintY = kwargs.pop('surfaceHintY', 'parabolic')
        super().__init__(*args, **kwargs)
        self.stl_mesh = None
        self.fileName = None
        self.orientation = orientation
        self.recenter = recenter
        self._surfaceHintX = self._surfaceHintY = 'parabolic'
        self.surfaceHint = surfaceHint
        if explicitX:
            self.surfaceHintX = surfaceHintX
        if explicitY:
            self.surfaceHintY = surfaceHintY

        self.fileName = fileName

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        previousValue = getattr(self, '_orientation', None)
        self._orientation = orientation
        if self.stl_mesh is not None and not self.fit_surface():
            self._orientation = previousValue

    @property
    def recenter(self):
        return self._recenter

    @recenter.setter
    def recenter(self, recenter):
        previousValue = getattr(self, '_recenter', None)
        self._recenter = recenter
        if self.stl_mesh is not None and not self.fit_surface():
            self._recenter = previousValue

    @property
    def surfaceHint(self):
        if self.surfaceHintX != self.surfaceHintY:
            return None
        return {'parabolic': 'quad', 'circular': 'toroid'}.get(
            self.surfaceHintX, self.surfaceHintX)

    @surfaceHint.setter
    def surfaceHint(self, value):
        if value is None:
            return
        hint = {'quad': 'parabolic', 'toroid': 'circular'}.get(value, value)
        self._set_surface_hints(hint, hint)

    @property
    def surfaceHintX(self):
        return self._surfaceHintX

    @surfaceHintX.setter
    def surfaceHintX(self, value):
        other = (value if 'spline' in (value, self.surfaceHintY) else
                 self.surfaceHintY)
        self._set_surface_hints(value, other)

    @property
    def surfaceHintY(self):
        return self._surfaceHintY

    @surfaceHintY.setter
    def surfaceHintY(self, value):
        other = (value if 'spline' in (value, self.surfaceHintX) else
                 self.surfaceHintX)
        self._set_surface_hints(other, value)

    def _set_surface_hints(self, xhint, yhint):
        if xhint not in self._axisHints or yhint not in self._axisHints:
            print("STL surface fit error: unknown axis hint", xhint, yhint)
            return False
        if 'spline' in (xhint, yhint):
            xhint = yhint = 'spline'
        previous = self._surfaceHintX, self._surfaceHintY
        self._surfaceHintX, self._surfaceHintY = xhint, yhint
        if self.stl_mesh is not None and not self.fit_surface():
            self._surfaceHintX, self._surfaceHintY = previous
            return False
        return True

    def _fit_curvature(self, axis):
        if self.conicFit is not None:
            return self.conicFit[axis]
        if self.toroidFit is not None:
            return 1. / self.toroidFit[1-axis]
        if self.cpoly is not None:
            return 2. * self.cpoly[axis]
        return None

    @property
    def RsagFit(self):
        curvature = self._fit_curvature(0)
        return (None if curvature is None else
                1. / curvature if curvature != 0 else np.inf)

    @property
    def RmerFit(self):
        curvature = self._fit_curvature(1)
        return (None if curvature is None else
                1. / curvature if curvature != 0 else np.inf)

    @property
    def conicXFit(self):
        if self.surfaceHintX == 'flat':
            return None
        if self.conicFit is not None:
            return self.conicFit[2]
        return {'parabolic': -1., 'circular': 0.}.get(self.surfaceHintX) \
            if self.stl_mesh is not None else None

    @property
    def conicYFit(self):
        if self.surfaceHintY == 'flat':
            return None
        if self.conicFit is not None:
            return self.conicFit[3]
        return {'parabolic': -1., 'circular': 0.}.get(self.surfaceHintY) \
            if self.stl_mesh is not None else None

    @property
    def fitRmsError(self):
        return self._fitRmsError

    @property
    def fileName(self):
        return self._fileName

    @fileName.setter
    def fileName(self, fileName):
        if not fileName:
            self._fileName = None
            self.stl_mesh = None
            self.points = None
            self.normals = None
            self.cpoly = None
            self.z_spline = None
            self.toroidFit = self.conicFit = None
            self._fitRmsError = None
            self.dcx = self.dcy = self.dcz = 0.
            return

        if not isSTLsupported:
            print("numpy-stl must be installed to work with STL models")
            return

        path = Path(fileName)
        if not path.is_file():
            print("STL file does not exist:", fileName)
            return

        previousState = self.__dict__.copy()
        try:
            self.read_file(path)
            if not self.fit_surface():
                self.__dict__.clear()
                self.__dict__.update(previousState)
                return
        except Exception as e:
            self.__dict__.clear()
            self.__dict__.update(previousState)
            print("STL file import error:", e)
            return

        self._fileName = fileName

    def read_file(self, filename):
        self.stl_mesh = mesh.Mesh.from_file(filename)

    def fit_surface(self):
        """Refit the mesh safely; return whether the fit succeeded."""
        if self.stl_mesh is None:
            return False

        previousState = self.__dict__.copy()
        try:
            if self._fit_surface():
                return True
        except Exception as e:
            print("STL surface fit error:", e)

        self.__dict__.clear()
        self.__dict__.update(previousState)
        return False

    def _fit_surface(self):

        def pkey(p, ndigits=8):
            return tuple(np.round(p, ndigits))

        normals = np.array(self.stl_mesh.normals)
        faces = self.stl_mesh.data
        xrt_ax = {'X': 0, 'Y': 1, 'Z': 2}
        z_ax = xrt_ax[self.orientation[2].upper()]

        x_arr = getattr(self.stl_mesh, self.orientation[0].lower())
        y_arr = getattr(self.stl_mesh, self.orientation[1].lower())
        z_arr = getattr(self.stl_mesh, self.orientation[2].lower())

        normalLengths = np.linalg.norm(normals, axis=1)
        normalZ = np.divide(
            normals[:, z_ax], normalLengths,
            out=np.zeros_like(normalLengths), where=normalLengths > 0)
        topSurfIndex = np.flatnonzero(np.abs(normalZ) > 0.1)
        if not len(topSurfIndex):
            print(
                "STL surface fit error: no non-degenerate surface along the "
                f"{self.orientation[2]} axis")
            return False
        z_coordinates = np.max(z_arr[topSurfIndex], axis=1)
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

        self.dcx = self.dcy = 0.
        zs0 = 0.
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
        ux = uxy[:, 0].astype(float)
        uy = uxy[:, 1].astype(float)
        uz = zs[ui].astype(float)

        self.cpoly = self.z_spline = self.toroidFit = self.conicFit = None
        hints = self.surfaceHintX, self.surfaceHintY
        A = np.c_[ux**2, uy**2, ux*uy, ux, uy, np.ones_like(ux)]
        seed, *_ = np.linalg.lstsq(A, uz, rcond=None)
        if all(h in ('flat', 'parabolic') for h in hints) or \
                hints == ('circular', 'circular'):
            self.cpoly = seed
            if hints != ('circular', 'circular'):
                columns = [5]
                if hints[0] == 'parabolic':
                    columns += [0, 3]
                if hints[1] == 'parabolic':
                    columns += [1, 4]
                if hints == ('parabolic', 'parabolic'):
                    columns += [2]
                self.cpoly = np.zeros(6)
                self.cpoly[columns], *_ = np.linalg.lstsq(
                    A[:, columns], uz, rcond=None)
            dcz = self.cpoly[5]
            if hints == ('circular', 'circular'):
                if len(uz) < 6:
                    print("STL toroid fit error: at least six vertices needed")
                    return False
                scale = max(np.ptp(ux), np.ptp(uy), 1.)
                threshold = np.finfo(float).eps / scale
                signs = [np.sign(c) or 1. for c in self.cpoly[:2]]
                radii = [0.5 / abs(c) if abs(c) > threshold else 1e12 * scale
                         for c in self.cpoly[:2]]
                cx = (-self.cpoly[3] / (2. * self.cpoly[0])
                      if abs(self.cpoly[0]) > threshold else 0.)
                cy = (-self.cpoly[4] / (2. * self.cpoly[1])
                      if abs(self.cpoly[1]) > threshold else 0.)
                xmax = np.max(np.abs(ux - cx))
                ymax = np.max(np.abs(uy - cy))
                rsag = max(radii[0], xmax + 1e-3 * scale)
                sagmax = xmax*xmax / (
                    rsag + np.sqrt(rsag*rsag - xmax*xmax))
                initial = [
                    np.log(rsag - xmax),
                    np.log(max(radii[1] - sagmax - ymax, 1e-3 * scale)),
                    cx, cy, dcz]

                def residual(parameters):
                    cx, cy, z0 = parameters[2:]
                    xmax = np.max(np.abs(ux - cx))
                    ymax = np.max(np.abs(uy - cy))
                    # Positive margins keep trial radii outside the patch.
                    rsag = xmax + np.exp(parameters[0])
                    sagmax = xmax*xmax / (
                        rsag + np.sqrt(rsag*rsag - xmax*xmax))
                    rmer = sagmax + ymax + np.exp(parameters[1])
                    self.toroidFit = np.array(
                        [signs[1]*rmer, signs[0]*rsag, cx, cy, z0])
                    return self.local_z(ux, uy) - uz

                result = least_squares(
                    residual, initial, x_scale='jac', max_nfev=200,
                    bounds=([np.log(1e-9 * scale)]*2 + [-np.inf]*3,
                            [np.log(1e13 * scale)]*2 + [np.inf]*3))
                if not result.success or not np.all(np.isfinite(result.fun)):
                    print("STL toroid fit error:", result.message)
                    return False
                residual(result.x)
                Rmer, Rsag = self.toroidFit[:2]
                dcz = self.local_z(0., 0.)
                self.cpoly = None
        elif hints == ('spline', 'spline'):
            gridsizeX = max(4, int(10 * np.ptp(self.limPhysX)))
            gridsizeY = max(4, int(10 * np.ptp(self.limPhysY)))

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
            self.z_spline = RectBivariateSpline(xgrid, ygrid, zmesh, s=0.)
            dcz = np.min(zmesh)
            self.cpoly = None
        else:
            if not self._fit_conics(ux, uy, uz, seed):
                return False
            dcz = self.local_z(0., 0.)

        if self.recenter:
            self.dcz = dcz

        error = self.local_z(ux, uy) + self.dcz - uz
        if not np.all(np.isfinite(error)):
            print("STL surface fit error: non-finite fitted surface")
            return False
        self._fitRmsError = float(np.sqrt(np.mean(error**2)) * 1e3)
        print(f'{self.RmerFit=}, {self.RsagFit=}, {self.fitRmsError=} um')

        self.points = np.array(self.stl_mesh.vectors).reshape(-1, 3) -\
            np.array([self.dcx, self.dcy, self.dcz + zs0])
        self.normals = np.repeat(self.stl_mesh.normals, 3, axis=0)
        return True

    @staticmethod
    def _conic_sag(x, y, cx, cy, kx, ky):
        """Shared conic sag and analytic derivatives; zero curvature is flat."""
        x, y = np.asarray(x), np.asarray(y)
        ax, ay = (1. + kx)*cx*cx, (1. + ky)*cy*cy
        with np.errstate(invalid='ignore', divide='ignore'):
            root = np.sqrt(1. - ax*x*x - ay*y*y)
            denominator = 1. + root
            numerator = cx*x*x + cy*y*y
            z = numerator / denominator
            dx = 2.*cx*x/denominator + z*ax*x/(root*denominator)
            dy = 2.*cy*y/denominator + z*ay*y/(root*denominator)
        return z, dx, dy

    def _conic_values(self, x, y):
        cx, cy, kx, ky, x0, y0, z0, coupled = self.conicFit
        x, y = np.asarray(x)-x0, np.asarray(y)-y0
        if coupled:
            z, dx, dy = self._conic_sag(x, y, cx, cy, kx, ky)
        else:
            zx, dx, _ = self._conic_sag(x, 0., cx, 0., kx, 0.)
            zy, _, dy = self._conic_sag(0., y, 0., cy, 0., ky)
            z = zx + zy
        return z + z0 - self.dcz, dx, dy

    def _fit_conics(self, ux, uy, uz, seed):
        hints = self.surfaceHintX, self.surfaceHintY
        coupled = hints == ('elliptical', 'elliptical')
        scale = max(np.ptp(ux), np.ptp(uy), 1.)
        centers = np.zeros(2)
        curvatures = np.zeros(2)
        q = np.ones(2)  # q = 1 + conic constant; ellipse q > 0.
        active = [i for i, h in enumerate(hints) if h != 'flat']
        signs = np.sign(seed[:2])
        signs[signs == 0] = 1.
        for i in active:
            curvatures[i] = max(abs(2.*seed[i]), 1e-12/scale)
            centers[i] = -seed[3+i]/(2.*seed[i]) if seed[i] != 0 else 0.
            q[i] = {'parabolic': 0., 'circular': 1.,
                    'elliptical': 0.5}[hints[i]]
        extents = np.maximum(
            [np.max(np.abs(ux-centers[0])),
             np.max(np.abs(uy-centers[1]))], 1e-9*scale)
        domain = q*(curvatures*extents)**2
        if coupled:
            if np.sum(domain) >= 0.9:
                curvatures *= np.sqrt(0.9/np.sum(domain))
                domain = q*(curvatures*extents)**2
            raw = curvatures / np.sqrt(1.-np.sum(domain))
        else:
            curvatures /= np.sqrt(np.maximum(domain/0.9, 1.))
            domain = q*(curvatures*extents)**2
            raw = curvatures / np.sqrt(1.-domain)
        initial, lower, upper = [], [], []
        for i in active:
            initial += [np.log(raw[i]*scale), centers[i]/scale]
            lower += [-30., -np.inf]
            upper += [30., np.inf]
            if hints[i] == 'elliptical':
                initial += [np.log(q[i])]
                lower += [-30.]
                upper += [30.]
        initial += [np.mean(uz)/scale]
        lower += [-np.inf]
        upper += [np.inf]
        if len(uz) < len(initial):
            print("STL conic fit error: insufficient surface vertices")
            return False

        def residual(parameters):
            raw, centers, q = np.zeros(2), np.zeros(2), np.ones(2)
            pos = 0
            for i in active:
                raw[i] = signs[i]*np.exp(parameters[pos])/scale
                centers[i] = parameters[pos+1]*scale
                pos += 2
                if hints[i] == 'elliptical':
                    q[i] = np.exp(parameters[pos])
                    pos += 1
                elif hints[i] == 'parabolic':
                    q[i] = 0.
            extents = [np.max(np.abs(ux-centers[0])),
                       np.max(np.abs(uy-centers[1]))]
            domain = q*(raw*np.asarray(extents))**2
            divisor = np.sqrt(1.+np.sum(domain) if coupled else 1.+domain)
            curvature = raw/divisor*(1.-1e-10)
            self.conicFit = (*curvature, *(q-1.), *centers,
                             parameters[-1]*scale, coupled)
            return self.local_z(ux, uy) - uz

        initial[-1] -= np.mean(residual(initial))/scale
        result = least_squares(residual, initial, bounds=(lower, upper),
                               x_scale='jac', max_nfev=500)
        if not result.success or not np.all(np.isfinite(result.fun)):
            print("STL conic fit error:", result.message)
            return False
        residual(result.x)
        return True

    def local_z(self, x, y):
        if self.conicFit is not None:
            return self._conic_values(x, y)[0]
        if self.toroidFit is not None:
            Rmer, Rsag, cx, cy, z0 = self.toroidFit
            x, y = np.asarray(x)-cx, np.asarray(y)-cy
            hx, _, _ = self._conic_sag(x, 0., 1./Rsag, 0., 0., 0.)
            hy, _, _ = self._conic_sag(0., y, 0., 1./(Rmer-hx), 0., 0.)
            z = hx + hy + z0 - self.dcz
        elif getattr(self, 'z_spline', None) is not None:
            z = self.z_spline.ev(x, y) - self.dcz
        elif getattr(self, 'cpoly', None) is not None:
            z = self.cpoly[0]*x**2 + self.cpoly[1]*y**2 + self.cpoly[2]*x*y +\
                self.cpoly[3]*x + self.cpoly[4]*y + self.cpoly[5] - self.dcz
        else:  # flat
            z = np.zeros_like(x)
        return z

    def local_n(self, x, y):
        if self.conicFit is not None:
            _, a, b = self._conic_values(x, y)
        elif self.toroidFit is not None:
            Rmer, Rsag, cx, cy, _ = self.toroidFit
            x, y = np.asarray(x)-cx, np.asarray(y)-cy
            hx, a, _ = self._conic_sag(x, 0., 1./Rsag, 0., 0., 0.)
            _, _, b = self._conic_sag(0., y, 0., 1./(Rmer-hx), 0., 0.)
            with np.errstate(invalid='ignore', divide='ignore'):
                a /= np.sqrt(1.-(y/(Rmer-hx))**2)
        elif getattr(self, 'z_spline', None) is not None:
            a = self.z_spline.ev(x, y, dx=1, dy=0)
            b = self.z_spline.ev(x, y, dx=0, dy=1)
        elif getattr(self, 'cpoly', None) is not None:
            a = 2*self.cpoly[0]*x + self.cpoly[2]*y + self.cpoly[3]
            b = 2*self.cpoly[1]*y + self.cpoly[2]*x + self.cpoly[4]
        else:  # flat
            a = b = np.zeros_like(x)

        norm = np.sqrt(a**2+b**2+1.)
        return [-a/norm, -b/norm, 1./norm]
