# -*- coding: utf-8 -*-
import numpy as np

from ..physconsts import CHBAR
from .material import Material


class TXMMaterial(Material):
    """
    A file-backed indexed-volume transmission sample material.

    The HDF5 file must use the following layout::

        /indexGrid       integer dataset, shape (nz, ny, nx)
        /limits/x        [xmin, xmax] in mm
        /limits/y        [ymin, ymax] in mm
        /limits/z        [zmin, zmax] in mm

    The grid axis order is always ``(z, y, x)``. Each value stored in
    ``/indexGrid`` is an integer key in ``materialsIndex``. The number of
    voxels along each axis is taken from the shape of ``/indexGrid``.

    The optional ``/indexGrid`` attribute ``backgroundIndex`` identifies the
    material used when spatial coordinates are not supplied. It defaults to
    0. If the optional ``axisOrder`` attribute is present, it must be
    ``"zyx"``.

    An instance with an empty ``fileName`` or ``materialsIndex`` is a
    transparent vacuum placeholder. This allows GUI editors to create and
    configure the material incrementally. A failed reload of a non-empty
    configuration is recorded in ``loadError`` and leaves the last
    successfully loaded volume active.

    For example, a compatible file can be created with::

        with h5py.File("sample.h5", "w") as h5:
            grid = h5.create_dataset(
                "indexGrid", data=indexGrid, dtype="u1")
            grid.attrs["axisOrder"] = "zyx"
            grid.attrs["backgroundIndex"] = 0

            limits = h5.create_group("limits")
            limits.create_dataset("x", data=[-0.025, 0.025])
            limits.create_dataset("y", data=[-0.025, 0.025])
            limits.create_dataset("z", data=[0.0, 0.050])
    """

    needsSpatialAmplitude = True
    hiddenParams = (
        'kind', 'elements', 'quantities', 'rho', 't', 'table', 'efficiency',
        'efficiencyFile', 'refractiveIndex')

    def __init__(self, fileName='', materialsIndex=None, name='', **kwargs):
        r"""
        *fileName*: str
            Path to the indexed-volume HDF5 file.

        *materialsIndex*: dict or sequence
            Maps the integer values stored in ``/indexGrid`` to material
            instances or material names. A sequence is interpreted as a
            zero-based mapping. For example::

                materialsIndex = {
                    0: water,
                    1: rocksalt,
                    2: air,
                }

        *name*: str
            User-specified name.
        """
        kind = kwargs.pop('kind', 'plate')
        if kind != 'plate':
            raise ValueError('TXMMaterial kind is always "plate"')
        for ignored in self.hiddenParams:
            if ignored in kwargs:
                del kwargs[ignored]
        Material.__init__(
            self, elements=(), quantities=(), kind='plate', rho=0, t=None,
            name=name, refractiveIndex=1., **kwargs)
        self._fileName = ''
        self._materialsIndex = {}
        self._materialsIndexError = None
        self._activeMaterialsIndex = {}
        self.isLoaded = False
        self.loadError = None
        self.fileName = fileName
        self.materialsIndex = materialsIndex

    @property
    def kind(self):
        return 'plate'

    @kind.setter
    def kind(self, kind):
        if kind != 'plate':
            raise ValueError('TXMMaterial kind is always "plate"')
        self._kind = 'plate'

    @property
    def fileName(self):
        return self._fileName

    @fileName.setter
    def fileName(self, fileName):
        self._fileName = fileName or ''
        if hasattr(self, '_materialsIndex'):
            self.reload()

    @property
    def materialsIndex(self):
        return self._materialsIndex

    @materialsIndex.setter
    def materialsIndex(self, materialsIndex):
        try:
            normalized = self._normalize_materials_index(materialsIndex)
        except Exception as error:
            self._materialsIndexError = error
            self.loadError = error
            return
        self._materialsIndexError = None
        self._materialsIndex = normalized
        if hasattr(self, '_fileName'):
            self.reload()

    def _normalize_materials_index(self, materialsIndex):
        if materialsIndex is None:
            return {}
        if isinstance(materialsIndex, dict):
            return {
                int(k): self._resolve_materials_index_entry(v)
                for k, v in materialsIndex.items()}
        if not isinstance(materialsIndex, (list, tuple)):
            raise TypeError('materialsIndex must be a dict or sequence')
        return {
            i: self._resolve_materials_index_entry(v)
            for i, v in enumerate(materialsIndex)}

    def _resolve_materials_index_entry(self, material):
        if not isinstance(material, str):
            return material

        if self.bl is not None:
            matId = self.bl.matnamesToUUIDs.get(material, material)
            resolved = self.bl.materialsDict.get(matId)
            if resolved is not None:
                return resolved

        return material

    def reload(self, strict=False):
        """
        Reload and validate the configured indexed volume.

        A missing file name or material index leaves this instance as a
        transparent vacuum placeholder. For a malformed non-empty
        configuration, the previous successfully loaded volume remains active.

        *strict*: bool
            If True, raise an exception for missing or invalid configuration.
        """
        if not self.fileName or not self.materialsIndex:
            self.isLoaded = False
            self.loadError = None
            if strict:
                raise ValueError(
                    'TXMMaterial requires fileName and materialsIndex')
            return False
        if self._materialsIndexError is not None:
            self.loadError = self._materialsIndexError
            if strict:
                raise self._materialsIndexError
            return False

        try:
            volume = self._read_volume_file(self.fileName)
            self._validate_materials_index(
                volume['indexGrid'], volume['backgroundIndex'],
                self.materialsIndex)
        except Exception as error:
            self.loadError = error
            if strict:
                raise
            return False

        for name, value in volume.items():
            setattr(self, name, value)
        self._activeMaterialsIndex = self.materialsIndex.copy()
        self.isLoaded = True
        self.loadError = None
        return True

    def _read_volume_file(self, fileName):
        try:
            import h5py
        except ImportError:
            raise ImportError(
                'TXMMaterial requires h5py to load indexed volume files')

        with h5py.File(fileName, 'r') as h5:
            indexGrid = np.asarray(h5['indexGrid'][:])
            axisOrder = h5['indexGrid'].attrs.get('axisOrder', 'zyx')
            if isinstance(axisOrder, bytes):
                axisOrder = axisOrder.decode()
            if axisOrder.lower() != 'zyx':
                raise ValueError(
                    'TXMMaterial expects /indexGrid axisOrder="zyx"')

            limits = h5['limits']
            xLimits = np.asarray(limits['x'][:], dtype=float)
            yLimits = np.asarray(limits['y'][:], dtype=float)
            zLimits = np.asarray(limits['z'][:], dtype=float)
            backgroundIndex = int(
                h5['indexGrid'].attrs.get('backgroundIndex', 0))

        if indexGrid.ndim != 3:
            raise ValueError('/indexGrid must be a 3D integer dataset')
        if not np.issubdtype(indexGrid.dtype, np.integer):
            raise ValueError('/indexGrid must be an integer dataset')

        for name, limits in (
                ('x', xLimits), ('y', yLimits), ('z', zLimits)):
            if limits.shape != (2,) or limits[0] >= limits[1]:
                raise ValueError(
                    '/limits/{0} must contain [min, max] in mm'.format(name))

        nz, ny, nx = indexGrid.shape
        dx = (xLimits[1] - xLimits[0]) / nx
        dy = (yLimits[1] - yLimits[0]) / ny
        dz = (zLimits[1] - zLimits[0]) / nz
        zEdges = np.linspace(zLimits[0], zLimits[1], nz + 1)
        return {
            'indexGrid': indexGrid,
            'xLimits': xLimits,
            'yLimits': yLimits,
            'zLimits': zLimits,
            'backgroundIndex': backgroundIndex,
            'nz': nz, 'ny': ny, 'nx': nx,
            'dx': dx, 'dy': dy, 'dz': dz,
            'zEdges': zEdges}

    def _validate_materials_index(
            self, indexGrid, backgroundIndex, materialsIndex):
        required = set(np.unique(indexGrid))
        required.add(backgroundIndex)
        missing = required - set(materialsIndex)
        if missing:
            raise ValueError(
                'materialsIndex has no entries for indices {0}'.format(
                    sorted(int(v) for v in missing)))
        for index, material in materialsIndex.items():
            if not hasattr(material, 'get_refractive_index'):
                raise TypeError(
                    'materialsIndex[{0}] has no get_refractive_index method'
                    .format(index))

    def _xyz_to_index(self, x, y, z):
        ix = np.floor((x - self.xLimits[0]) / self.dx).astype(np.intp)
        iy = np.floor((y - self.yLimits[0]) / self.dy).astype(np.intp)
        iz = np.floor((z - self.zLimits[0]) / self.dz).astype(np.intp)
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        iz = np.clip(iz, 0, self.nz - 1)
        return ix, iy, iz

    def get_material_indices(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        ix, iy, iz = self._xyz_to_index(x, y, z)
        return self.indexGrid[iz, iy, ix]

    def _get_refractive_index_by_indices(self, E, indices):
        E, indices = np.broadcast_arrays(E, indices)
        flatE = E.ravel()
        flatIndices = indices.ravel()
        res = np.empty(flatE.shape, dtype=np.complex128)
        for index in np.unique(flatIndices):
            mask = flatIndices == index
            material = self._activeMaterialsIndex[int(index)]
            res[mask] = material.get_refractive_index(flatE[mask])
        return res.reshape(E.shape)

    def get_refractive_index(self, E, x=None, y=None, z=None):
        if not self.isLoaded:
            return np.ones_like(E, dtype=np.complex128)
        if x is None or y is None or z is None:
            material = self._activeMaterialsIndex[self.backgroundIndex]
            return material.get_refractive_index(E)

        indices = self.get_material_indices(x, y, z)
        return self._get_refractive_index_by_indices(E, indices)

    def get_absorption_coefficient(self, E, x=None, y=None, z=None):
        if not self.isLoaded:
            return np.zeros_like(E, dtype=float)
        n = self.get_refractive_index(E, x, y, z)
        return abs(n.imag) * E / CHBAR * 2e8

    def _vacuum_amplitude(self, E, beamInDotNormal):
        E, beamInDotNormal = np.broadcast_arrays(E, beamInDotNormal)
        ones = np.ones(E.shape, dtype=np.complex128)
        zeros = np.zeros(E.shape, dtype=float)
        nk = E * 1e8 / CHBAR
        return ones, ones.copy(), zeros, nk

    def _plate_amplitude_from_n(self, E, beamInDotNormal, fromVacuum, n):
        E, beamInDotNormal, n = np.broadcast_arrays(E, beamInDotNormal, n)
        if fromVacuum:
            n1 = 1.
            n2 = n
        else:
            n1 = n
            n2 = 1.

        cosAlpha = abs(beamInDotNormal)
        sinAlpha2 = 1 - beamInDotNormal**2
        if isinstance(sinAlpha2, np.ndarray):
            sinAlpha2[sinAlpha2 < 0] = 0
        n1cosAlpha = n1 * cosAlpha
        cosBeta = np.sqrt(1 - (n1/n2)**2*sinAlpha2)
        n2cosBeta = n2 * cosBeta
        tf = np.sqrt(
            (n2cosBeta * np.conjugate(n1)).real / cosAlpha) / abs(n1)
        rs = 2 * n1cosAlpha / (n1cosAlpha + n2cosBeta) * tf
        rp = 2 * n1cosAlpha / (n2*cosAlpha + n1*cosBeta) * tf
        return (rs, rp, abs(n.imag) * E / CHBAR * 2e8,
                n.real * E / CHBAR * 1e8)

    def _volume_integrals(self, E, x, y, z, a, b, c, tMax):
        E, x, y, z, a, b, c, tMax = np.broadcast_arrays(
            E, x, y, z, a, b, c, tMax)
        shape = E.shape
        flatE = E.ravel()
        flatX = x.ravel()
        flatY = y.ravel()
        flatZ = z.ravel()
        flatA = a.ravel()
        flatB = b.ravel()
        flatC = c.ravel()
        flatT = np.maximum(tMax.ravel(), 0)

        tau = np.zeros(flatE.shape, dtype=float)
        phase = np.zeros(flatE.shape, dtype=float)
        validC = np.abs(flatC) > 1e-15
        cSafe = np.where(validC, flatC, 1.)

        for iz in range(self.nz):
            s0 = (self.zEdges[iz] - flatZ) / cSafe
            s1 = (self.zEdges[iz + 1] - flatZ) / cSafe
            slow = np.minimum(s0, s1)
            shigh = np.maximum(s0, s1)
            seg0 = np.maximum(slow, 0)
            seg1 = np.minimum(shigh, flatT)
            active = validC & (seg1 > seg0)
            if not np.any(active):
                continue

            pos = np.where(active)[0]
            mid = 0.5 * (seg0[pos] + seg1[pos])
            xm = flatX[pos] + flatA[pos] * mid
            ym = flatY[pos] + flatB[pos] * mid
            zm = flatZ[pos] + flatC[pos] * mid
            indices = self.get_material_indices(xm, ym, zm).ravel()
            segCm = (seg1[pos] - seg0[pos]) * 0.1

            for index in np.unique(indices):
                mask = indices == index
                material = self._activeMaterialsIndex[int(index)]
                n = material.get_refractive_index(flatE[pos][mask])
                mu = abs(n.imag) * flatE[pos][mask] / CHBAR * 2e8
                nk = n.real * flatE[pos][mask] / CHBAR * 1e8
                tau[pos[mask]] += mu * segCm[mask]
                phase[pos[mask]] += nk * segCm[mask]

        flatMu = np.zeros(flatE.shape, dtype=float)
        flatNk = np.zeros(flatE.shape, dtype=float)
        hasPath = flatT > 0
        pathCm = flatT[hasPath] * 0.1
        flatMu[hasPath] = tau[hasPath] / pathCm
        flatNk[hasPath] = phase[hasPath] / pathCm
        return flatMu.reshape(shape), flatNk.reshape(shape)

    def get_amplitude(
            self, E, beamInDotNormal, fromVacuum=True, x=None, y=None, z=None,
            a=None, b=None, c=None, tMax=None):
        if not self.isLoaded:
            return self._vacuum_amplitude(E, beamInDotNormal)
        if x is None or y is None or z is None:
            n = self.get_refractive_index(E)
            return self._plate_amplitude_from_n(
                E, beamInDotNormal, fromVacuum, n)

        if (not fromVacuum) and tMax is not None and\
                all(v is not None for v in (a, b, c)):
            xExit = x + a * tMax
            yExit = y + b * tMax
            zExit = z + c * tMax
            nSurface = self.get_refractive_index(E, xExit, yExit, zExit)
            rs, rp, _, _ = self._plate_amplitude_from_n(
                E, beamInDotNormal, fromVacuum, nSurface)
            mu, nk = self._volume_integrals(E, x, y, z, a, b, c, tMax)
            return rs, rp, mu, nk

        nSurface = self.get_refractive_index(E, x, y, z)
        return self._plate_amplitude_from_n(
            E, beamInDotNormal, fromVacuum, nSurface)
