# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "03 Jul 2016"
import os
import sys
import inspect
# import copy
# import math
import numpy as np
from scipy import ndimage
import pickle
import time
from multiprocessing import Pool, cpu_count

import gzip
from .. import raycing
from .sources_beams import Beam
from .physconsts import M0C2, K2B, SIE0, SIC, PI, PI2, CHeVcm

_DEBUG = 20  # if non-zero, some diagnostics is printed out

# You should better modify the paths to XOP here, otherwise you have to give
# the path as a parameter of UndulatorUrgent, WigglerWS or BendingMagnetWS.
if os.name == 'posix':
    xopBinDir = r'/home/konkle/xop2.3/bin.linux'
else:
    xopBinDir = r'c:\XOP\bin.x86'


def run_one(path, tmpwd, infile, msg=None):
    from subprocess import Popen, PIPE
    if _DEBUG:
        if msg is not None:
            if os.name == 'posix':
                sys.stdout.write("\r\x1b[K "+msg)
            else:
                # sys.stdout.write("\r" + "    ")
                print(msg + ' ')
            sys.stdout.flush()
    with open(os.devnull, 'w') as fn:
        cproc = Popen(path, stdin=PIPE, stdout=fn, cwd=tmpwd)
        cproc.communicate(infile)


def gzip_output(tmpwd, outName, msg=None):
    if _DEBUG:
        if msg is not None:
            if os.name == 'posix':
                sys.stdout.write("\r\x1b[K "+msg)
            else:
                # sys.stdout.write("\r" + "    ")
                print(msg + ' ')
            sys.stdout.flush()
    fname = os.path.join(tmpwd, outName)
# for Python 2.7+:
#    with open(fname, 'rb') as txtFile:
#        with gzip.open(fname + '.gz', 'wb') as zippedFile:
#            zippedFile.writelines(txtFile)
# for Python 2.7-:
    txtFile = open(fname, 'rb')
    zippedFile = gzip.open(fname + '.gz', 'wb')
    try:
        zippedFile.writelines(txtFile)
    finally:
        txtFile.close()
        zippedFile.close()
    os.remove(fname)


def read_output(tmpwd, outName, skiprows, usecols, comments, useZip, msg=None):
    if _DEBUG:
        if msg is not None:
            if os.name == 'posix':
                sys.stdout.write("\r\x1b[K "+msg)
            else:
                # sys.stdout.write("\r" + "    ")
                print(msg + ' ')
            sys.stdout.flush()
    try:
        return np.loadtxt(
            os.path.join(tmpwd, outName+('.gz' if useZip else '')),
            skiprows=skiprows, unpack=True, usecols=usecols, comments=comments,
            converters={2: lambda s: s.replace(b'D', b'e').replace(b'd', b'e')}
            )
    except:
        pass


class UndulatorUrgent(object):
    u"""
    Undulator source that uses the external code Urgent. It has some drawbacks,
    as demonstrated in the section :ref:`comparison-synchrotron-sources`, but
    nonetheless can be used for comparison purposes. If you are going to use
    it, the code is freely available as part of XOP package.
    """
    def __init__(
        self, bl=None, name='UrgentU', center=(0, 0, 0), nrays=raycing.nrays,
        period=32., K=2.668, Kx=0., Ky=0., n=12, eE=6., eI=0.1,
        eSigmaX=134.2, eSigmaZ=6.325, eEpsilonX=1., eEpsilonZ=0.01,
        uniformRayDensity=False,
        eMin=1500, eMax=81500, eN=1000, eMinRays=None, eMaxRays=None,
        xPrimeMax=0.25, zPrimeMax=0.1, nx=25, nz=25, path=None,
            mode=4, icalc=1, useZip=True, order=3, processes='auto'):
        u"""
        The 1st instantiation of this class runs the Urgent code and saves
        its output into a ".pickle" file. The temporary directory "tmp_urgent"
        can then be deleted. If any of the Urgent parameters has changed since
        the previous run, the Urgent code is forced to redo the calculations.

        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`
            Container for beamline elements. Sourcess are added to its
            `sources` list.

        *name*: str
            User-specified name, can be used for diagnostics output.

        *center*: tuple of 3 floats
            3D point in global system

        *nrays*: int
            the number of rays sampled in one iteration

        *period*: float
            Magnet period (mm).

        *K* or *Ky*: float
            Magnet deflection parameter (Ky) in the vertical field.

        *Kx*: float
            Magnet deflection parameter in the horizontal field.

        *n*: int
            Number of magnet periods.

        *eE*: float
            Electron beam energy (GeV).

        *eI*: float
            Electron beam current (A).

        *eSigmaX*, *eSigmaZ*: float
            rms horizontal and vertical electron beam sizes (µm).

        *eEpsilonX*, *eEpsilonZ*: float
            Horizontal and vertical electron beam emittance (nm rad).

        *eMin*, *eMax*: float
            Minimum and maximum photon energy (eV) used by Urgent.

        *eN*: int
            Number of photon energy intervals used by Urgent.

        *eMinRays*, *eMaxRays*: float
            The range of energies for rays. If None, are set equal to *eMin*
            and *eMax*. These two parameters are useful for playing with the
            energy axis without having to force Urgent to redo the
            calculations each time.

        *xPrimeMax*, *zPrimeMax*: float
            Half of horizontal and vertical acceptance (mrad).

        *nx*, *nz*: int
            Number of intervals in the horizontal and vertical directions from
            zero to maximum.

        *path*: str
            Full path to Urgent executable. If None, it is set automatically
            from the module variable *xopBinDir*.

        *mode*: 1, 2 or 4
            the MODE parameter of Urgent. If =1, UndulatorUrgent scans energy
            and reads the xz distribution from Urgent. If =2 or 4,
            UndulatorUrgent scans x and z and reads energy spectrum (angular
            density for 2 or flux through a window for 4) from Urgent. The
            meshes for x, z, and E are restricted in Urgent: nx,nz<50 and
            nE<5000. You may overcome these restrictions if you scan the
            corresponding quantities outside of Urgent, i.e. inside of this
            class UndulatorUrgent. *mode* = 4 is by far most preferable.

        *icalc*: int
            The ICALC parameter of Urgent.

        *useZip*: bool
            Use gzip module to compress the output files of Urgent. If True,
            the temporary storage takes much less space but a slightly bit
            more time.

        *order*: 1 or 3
            the order of the spline interpolation. 3 is recommended.

        *processes*: int or any other type as 'auto'
            the number of worker processes to use. If the type is not int then
            the number returned by multiprocessing.cpu_count()/2 is used.


        """
        # patch for starting a script with processes>1 from Spyder console:
        try:
            frm = inspect.stack()[2]
            mod = inspect.getmodule(frm[0])
            if not hasattr(mod, "__spec__"):
                mod.__spec__ = None
        except:
            pass

        self.bl = bl
        if bl is not None:
            bl.sources.append(self)
            self.ordinalNum = len(bl.sources)
        self.name = name
        self.center = center  # 3D point in global system
        self.nrays = int(nrays)

        self.period = period
        self.K = K if Ky == 0 else Ky
        self.Kx = Kx
        self.n = n
        self.eE = eE
        self.gamma = eE / M0C2 * 1e3
        self.eI = eI
        self.eSigmaX = eSigmaX
        self.eSigmaZ = eSigmaZ
        self.eEpsilonX = eEpsilonX
        self.eEpsilonZ = eEpsilonZ
        self.eMin = eMin
        self.eMax = eMax
        self.eN = eN
        if eMinRays is None:
            self.eMinRays = eMin
        else:
            self.eMinRays = eMinRays
        if eMaxRays is None:
            self.eMaxRays = eMax
        else:
            self.eMaxRays = eMaxRays
        self.logeMinRays = np.log(self.eMinRays)
        self.logeMaxRays = np.log(self.eMaxRays)
        self.xPrimeMax = xPrimeMax
        self.zPrimeMax = zPrimeMax
        self.nx = nx
        self.nz = nz
        self.path = path
        self.mode = mode
        self.icalc = icalc
        self.useZip = useZip
        if isinstance(processes, int):
            self.processes = processes
            pp = processes
        else:
            self.processes = None
            pp = cpu_count() // 2
        if _DEBUG:
            print('{0} process{1} will be requested'.format(
                  pp, ('' if pp == 1 else 'es')))
        self.xpads = len(str(self.nx))
        self.zpads = len(str(self.nz))
        self.Epads = len(str(self.eN))
# extra rows and columns to the negative part (reflect from the 1st quadrant)
# in order to have good spline coefficients. Otherwise the spline may have
# discontinuity at the planes x=0 and z=0.
        self.extraRows = 0
        self.order = order
        self.prefilter = self.order == 1
        self.run_and_save(pp)
        self.xzE = 4e3 * self.xPrimeMax * self.zPrimeMax *\
            (self.logeMaxRays-self.logeMinRays)  # =2[-Max to +Max]*2*(0.1%)
        self.fluxConst = self.Imax * self.xzE
        self.uniformRayDensity = uniformRayDensity

    def run_and_save(self, pp):
        tstart = time.time()
        self.run()
        if self.needRecalculate:
            if _DEBUG:
                print('. Finished after {0} seconds'.format(
                      time.time() - tstart))
        tstart = time.time()
        self.splines, self.Imax = self.make_spline_arrays(
            skiprows=32, cols1=(2, 3, 4, 5), cols2=(0, 2, 6, 7, 8))
        if _DEBUG:
            print('. Finished after {0} seconds'.format(time.time() - tstart))

    def code_name(self):
        return 'urgent'

    def comment_strings(self):
        return [" MAXIMUM", " TOTAL"]

    def prefix_save_name(self):
        if self.Kx > 0:
            return '4-elu-{0}'.format(self.code_name())
        else:
            return '1-und-{0}'.format(self.code_name())

    def make_input(self, x, z, E):
        # 1) ITYPE, PERIOD, KX, KY, PHASE, N
        # 2) EMIN, EMAX, NE
        # 3) ENERGY, CUR, SIGX, SIGY, SIGX1, SIGY1
        # 4) D, XPC, YPC, XPS, YPS, NXP, NYP
        # 5) MODE, ICALC, IHARM
        # 6) NPHI, NSIG, NALPHA, DALPHA, NOMEGA, DOMEGA
        infile = ''
        infile += '1 {0} {1} {2} 0. {3}\n'.format(
            self.period*1e-3, self.Kx, self.K, self.n)
        infile += '{0} {1} {2}\n'.format(E, self.eMax, self.eN)
        infile += '{0} {1} {2} {3} {4:.7f} {5:.7f}\n'.format(
            self.eE, self.eI, self.eSigmaX*1e-3, self.eSigmaZ*1e-3,
            self.eEpsilonX/self.eSigmaX, self.eEpsilonZ/self.eSigmaZ)
        if self.mode == 1:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                0, 0, 2*self.xPrimeMax, 2*self.zPrimeMax, self.nx, self.nz)
        elif self.mode == 2:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                x, z, self.xPrimeMax, self.zPrimeMax, self.nx, self.nz)
        elif self.mode == 4:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                x, z, self.xPrimeMax/self.nx, self.zPrimeMax/self.nz, 11, 11)
# ICALC=1 non-zero emittance, finite N
# ICALC=2 non-zero emittance, infinite N
# ICALC=3 zero emittance, finite N
        infile += '{0} {1} -1\n'.format(self.mode, self.icalc)
        infile += '80 7 4 0 0 0\n'
        return infile

    def tmp_wd_xz(self, cwd, ix, iz):
        return os.path.join(cwd, 'tmp_'+self.code_name(), 'x{0}z{1}'.format(
            (str(ix)).zfill(self.xpads), (str(iz)).zfill(self.zpads)))

    def tmp_wd_E(self, cwd, iE):
        return os.path.join(cwd, 'tmp_'+self.code_name(), 'E{0}'.format(
            (str(iE)).zfill(self.Epads)))

    def msg_xz(self, ix, iz):
        return '{0} of {1}, {2} of {3}'.format(
            (str(ix+1)).zfill(self.xpads),
            (str(len(self.xs))).zfill(self.xpads),
            (str(iz+1)).zfill(self.zpads),
            (str(len(self.zs))).zfill(self.zpads))

    def msg_E(self, iE):
        return '{0} of {1}'.format(
            (str(iE+1)).zfill(self.Epads),
            (str(len(self.Es))).zfill(self.Epads))

    def run(self, forceRecalculate=False, iniFileForEachDirectory=False):
        self.xs = np.linspace(0, self.xPrimeMax, self.nx+1)
        self.zs = np.linspace(0, self.zPrimeMax, self.nz+1)
        self.Es = np.linspace(self.eMin, self.eMax, self.eN+1)
        self.energies = self.Es
        cwd = os.getcwd()
        inpName = os.path.join(cwd, self.prefix_save_name()+'.inp')
        infile = self.make_input(0, 0, self.eMin)
        self.needRecalculate = True
        if os.path.exists(inpName):
            saved = ""
            with open(inpName, 'r') as f:
                for line in f:
                    saved += line
            self.needRecalculate = saved != infile
        if self.needRecalculate:
            with open(inpName, 'w') as f:
                f.write(infile)
        cwd = os.getcwd()
        pickleName = os.path.join(cwd, self.prefix_save_name()+'.pickle')
        if not os.path.exists(pickleName):
            if self.mode == 1:
                for iE, E in enumerate(self.Es):
                    tmpwd = self.tmp_wd_E(cwd, iE)
                    outName = os.path.join(tmpwd, self.code_name() + '.out' +
                                           ('.gz' if self.useZip else ''))
                    if not os.path.exists(outName):
                        self.needRecalculate = True
                        break
            elif self.mode in (2, 4):
                for iz, z in enumerate(self.zs):
                    for ix, x in enumerate(self.xs):
                        tmpwd = self.tmp_wd_xz(cwd, ix, iz)
                        outName = os.path.join(
                            tmpwd, self.code_name() + '.out' +
                            ('.gz' if self.useZip else ''))
                        if not os.path.exists(outName):
                            self.needRecalculate = True
                            break
            else:
                raise ValueError("mode must be 1, 2 or 4!")
        if (not self.needRecalculate) and (not forceRecalculate):
            return

        if self.path is None:
            self.path = os.path.join(
                xopBinDir, self.code_name() +
                ('.exe' if os.name == 'nt' else ''))
        if not os.path.exists(self.path):
            raise ImportError("The file {0} does not exist!".format(self.path))
        pool = Pool(self.processes)
        if _DEBUG:
            print('calculating with {0} ... '.format(self.code_name()))
        if self.mode == 1:
            for iE, E in enumerate(self.Es):
                tmpwd = self.tmp_wd_E(cwd, iE)
                if not os.path.exists(tmpwd):
                    os.makedirs(tmpwd)
                infile = self.make_input(0, 0, E)
                if iniFileForEachDirectory:
                    inpName = os.path.join(tmpwd, self.code_name()+'.inp')
                    with open(inpName, 'w') as f:
                        f.write(infile)
                msg = self.msg_E(iE) if iE % 10 == 0 else None
                pool.apply_async(run_one, (self.path, tmpwd, infile, msg))
        elif self.mode in (2, 4):
            for iz, z in enumerate(self.zs):
                for ix, x in enumerate(self.xs):
                    tmpwd = self.tmp_wd_xz(cwd, ix, iz)
                    if not os.path.exists(tmpwd):
                        os.makedirs(tmpwd)
                    infile = self.make_input(x, z, self.eMin)
                    if iniFileForEachDirectory:
                        inpName = os.path.join(tmpwd, self.code_name()+'.inp')
                        with open(inpName, 'w') as f:
                            f.write(infile)
                    msg = self.msg_xz(ix, iz) if ix % 10 == 0 else None
                    pool.apply_async(run_one, (self.path, tmpwd, infile, msg))
        else:
            raise ValueError("mode must be 1, 2 or 4!")
        pool.close()
        pool.join()
        if _DEBUG:
            print()
        if self.useZip:
            if _DEBUG:
                print('zipping ... ')
            poolz = Pool(self.processes)
            if self.mode == 1:
                for iE, E in enumerate(self.Es):
                    tmpwd = self.tmp_wd_E(cwd, iE)
                    msg = self.msg_E(iE) if iE % 10 == 0 else None
                    poolz.apply_async(gzip_output, (
                        tmpwd, self.code_name() + '.out', msg))
            elif self.mode in (2, 4):
                for iz, z in enumerate(self.zs):
                    for ix, x in enumerate(self.xs):
                        tmpwd = self.tmp_wd_xz(cwd, ix, iz)
                        msg = self.msg_xz(ix, iz) if ix % 10 == 0 else None
                        poolz.apply_async(gzip_output, (
                            tmpwd, self.code_name() + '.out', msg))
            poolz.close()
            poolz.join()

    def make_spline_arrays(self, skiprows, cols1, cols2):
        cwd = os.getcwd()
        pickleName = os.path.join(cwd, self.prefix_save_name()+'.pickle')
        if self.needRecalculate or (not os.path.exists(pickleName)):
            if _DEBUG:
                print('reading ... ')
            if self.mode == 1:
                I = np.zeros(
                    (self.Es.shape[0], self.xs.shape[0], self.zs.shape[0]))
                l1 = np.zeros_like(I)
                l2 = np.zeros_like(I)
                l3 = np.zeros_like(I)
                for iE, E in enumerate(self.Es):
                    tmpwd = self.tmp_wd_E(cwd, iE)
                    msg = self.msg_E(iE) if iE % 10 == 0 else None
                    res = read_output(
                        tmpwd, self.code_name()+'.out', skiprows,
                        cols1, self.comment_strings()[0], self.useZip, msg)
                    if res is not None:
                        It, l1t, l2t, l3t = res
                    else:
                        pass
#                        raise ValueError('Error in the calculation at ' +
#                                         'i={0}, E={1}'.format(iE, E))
                    if res is not None:
                        try:
                            I[iE, :, :] = \
                                np.reshape(It, (self.nx+1, self.nz+1))
                            l1[iE, :, :] = \
                                np.reshape(l1t, (self.nx+1, self.nz+1))
                            l2[iE, :, :] = \
                                np.reshape(l2t, (self.nx+1, self.nz+1))
                            l3[iE, :, :] = \
                                np.reshape(l3t, (self.nx+1, self.nz+1))
                        except:
                            pass
            elif self.mode in (2, 4):
                I = None
                for iz, z in enumerate(self.zs):
                    for ix, x in enumerate(self.xs):
                        tmpwd = self.tmp_wd_xz(cwd, ix, iz)
                        msg = self.msg_xz(ix, iz) if ix % 10 == 0 else None
                        res = read_output(
                            tmpwd, self.code_name()+'.out', skiprows,
                            cols2, self.comment_strings()[1], self.useZip, msg)
                        if res is not None:
                            self.Es, It, l1t, l2t, l3t = res
                            if self.mode == 4:
                                It /= self.xPrimeMax / (self.nx+0.5) *\
                                    self.zPrimeMax / (self.nz+0.5)
                        else:
                            pass
                        if I is None:
                            I = np.zeros((self.Es.shape[0], self.xs.shape[0],
                                          self.zs.shape[0]))
                            l1 = np.zeros_like(I)
                            l2 = np.zeros_like(I)
                            l3 = np.zeros_like(I)
                            I[:, ix, iz] = It
                            l1[:, ix, iz] = l1t
                            l2[:, ix, iz] = l2t
                            l3[:, ix, iz] = l3t
                        else:
                            if res is not None:
                                I[:, ix, iz], l1[:, ix, iz], l2[:, ix, iz],\
                                    l3[:, ix, iz] = It, l1t, l2t, l3t
            splines, Imax = self.save_spline_arrays(
                pickleName, (I, l1, l2, l3))
        else:
            if _DEBUG:
                print('restoring arrays ... ')
        splines, Imax = self.restore_spline_arrays(pickleName)
        if _DEBUG:
            print('shape={0}, max={1}'.format(splines[0].shape, Imax))
        return splines, Imax

    def save_spline_arrays(self, pickleName, what):
        if _DEBUG:
            print('. Pickling splines to\n{0}'.format(pickleName))
        splines = []
        for ia, a in enumerate(what):
            a = np.concatenate((a[:, self.extraRows:0:-1, :], a), axis=1)
            a = np.concatenate((a[:, :, self.extraRows:0:-1], a), axis=2)
            if self.order == 3:
                spline = ndimage.spline_filter(a)
            else:
                spline = a
            splines.append(spline)
        Imax = np.max(what[0])
        with open(pickleName, 'wb') as f:
            pickle.dump((Imax, splines), f, protocol=2)
        return splines, Imax

    def restore_spline_arrays(
            self, pickleName, findNewImax=True, IminCutOff=1e-50):
        if sys.version_info < (3, 1):
            kw = {}
        else:
            kw = dict(encoding='latin1')
        with open(pickleName, 'rb') as f:
            Imax, savedSplines = pickle.load(f, **kw)
        try:
            if findNewImax:
                ind = [i for i in range(self.Es.shape[0]) if
                       self.eMinRays <= self.Es[i] <= self.eMaxRays]
                if len(ind) == 0:
                    fact = self.eN / (self.eMax-self.eMin)
                    ind = [(self.eMinRays-self.eMin) * fact,
                           (self.eMaxRays-self.eMin) * fact]
                elif len(ind) == 1:
                    ind = [ind[0], ind[0]]
                coords = np.mgrid[ind[0]:ind[-1],
                                  self.extraRows:self.nx+1+self.extraRows,
                                  self.extraRows:self.nz+1+self.extraRows]

                I = ndimage.map_coordinates(
                    savedSplines[0], coords, order=self.order,
                    prefilter=self.prefilter)
                Imax = I.max()
                _eMinRays = self.Es[min(np.nonzero(I > Imax*IminCutOff)[0])]
                if _eMinRays > self.eMinRays:
                    self.eMinRays = _eMinRays
                    print('eMinRays has been corrected up to {0}'.format(
                        _eMinRays))
                _eMaxRays = self.Es[max(np.nonzero(I > Imax*IminCutOff)[0])]
                if _eMaxRays < self.eMaxRays:
                    self.eMaxRays = _eMaxRays
                    print('eMaxRays has been corrected down to {0}'.format(
                        _eMaxRays))
        except ValueError:
            pass
        return savedSplines, Imax

    def intensities_on_mesh(self):
        Is = []
        coords = np.mgrid[0:self.Es.shape[0],
                          self.extraRows:self.nx+1+self.extraRows,
                          self.extraRows:self.nz+1+self.extraRows]
        for a in self.splines:
            aM = ndimage.map_coordinates(a, coords, order=self.order,
                                         prefilter=self.prefilter)
            Is.append(aM)
        return Is

    def find_electron_path(self, vec, K, npassed):
        anorm = vec * self.gamma / K
        phase = np.empty_like(anorm)
        a1 = np.where(abs(anorm) <= 1)[0]
        phase[a1] = np.arcsin(anorm[a1])
        a1 = np.where(abs(anorm) > 1)[0]
        phase[a1] = np.sign(
            anorm[a1]) * np.random.normal(PI/2, PI/2/K, len(anorm[a1]))
        phase[::2] = np.sign(phase[::2]) * PI - phase[::2]
        phase -= np.sign(phase) * PI *\
            np.random.random_integers(-self.n+1, self.n, npassed)
        y = self.period / PI2 * phase
        x = K * self.period / PI2 / self.gamma * np.cos(phase)
        a = K / self.gamma * np.sin(phase)
        return y, x, a

    def shine(self, toGlobal=True):
        u"""
        Returns the source beam. If *toGlobal* is True, the output is in the
        global system."""
        bo = None
        length = 0
        seeded = np.long(0)
        seededI = 0.
        while length < self.nrays:
            bot = Beam(self.nrays)  # beam-out
            seeded += self.nrays
            bot.state[:] = 1  # good
            bot.E = np.exp(np.random.uniform(self.logeMinRays,
                                             self.logeMaxRays, self.nrays))
#            bot.E = np.random.uniform(
#                self.eMinRays, self.eMaxRays, self.nrays)
# mrad:
            bot.a = np.tan(
                np.random.uniform(-1, 1, self.nrays)*self.xPrimeMax * 1e-3)
            bot.c = np.tan(
                np.random.uniform(-1, 1, self.nrays)*self.zPrimeMax * 1e-3)
            coords = np.array(
                [(bot.E - self.eMin)/(self.eMax - self.eMin) * self.eN,
                 np.abs(bot.a)/(self.xPrimeMax*1e-3)*self.nx + self.extraRows,
                 np.abs(bot.c)/(self.zPrimeMax*1e-3)*self.nz + self.extraRows])
# coords.shape = (3, self.nrays)
            Icalc = ndimage.map_coordinates(
                self.splines[0], coords, order=self.order,
                prefilter=self.prefilter)
            seededI += Icalc.sum() * self.xzE
            if self.uniformRayDensity:
                npassed = self.nrays
                Icalc[Icalc < 0] = 0
                I0 = Icalc * 4 * self.xPrimeMax * self.zPrimeMax
            else:
                I = np.random.uniform(0, 1, self.nrays)
                passed = np.where(I * self.Imax < Icalc)[0]
                npassed = len(passed)
                if npassed == 0:
                    print('No good rays in this seed!'
                          ' {0} of {1} rays in total so far...'.format(
                              length, self.nrays))
                    continue
                I0 = 1.
                coords = coords[:, passed]
                bot.filter_by_index(passed)

            l1 = ndimage.map_coordinates(self.splines[1], coords,
                                         order=self.order,
                                         prefilter=self.prefilter)
            l2 = ndimage.map_coordinates(self.splines[2], coords,
                                         order=self.order,
                                         prefilter=self.prefilter)
            l3 = ndimage.map_coordinates(self.splines[3], coords,
                                         order=self.order,
                                         prefilter=self.prefilter)
            if self.Kx == 0:
                l3[bot.c < 0] *= -1.
            if self.order == 3:
                l1[l1 < -1] = -1.
                l1[l1 > 1] = 1.
                l2[l2 < -1] = -1.
                l2[l2 > 1] = 1.
                l3[l3 < -1] = -1.
                l3[l3 > 1] = 1.
            bot.Jss[:] = (1 + l1) / 2. * I0
            bot.Jpp[:] = (1 - l1) / 2. * I0
            sign = 1 if isinstance(self, WigglerWS) else -1
            bot.Jsp[:] = sign * (l2 + 1j*l3) / 2. * I0
# origin coordinates:
            if isinstance(self, BendingMagnetWS):
                bot.y[:] = -bot.a * self.rho
                bot.x[:] = bot.a**2 * self.rho / 2
            elif isinstance(self, WigglerWS):
                if self.Kx > 0:
                    bot.y[:], bot.z[:], bot.c[:] = \
                        self.find_electron_path(bot.c, self.Kx, npassed)
                if self.K > 0:
                    bot.y[:], bot.x[:], bot.a[:] = \
                        self.find_electron_path(bot.a, self.K, npassed)
            else:
                pass

# as by Walker and by Ellaume; SPECTRA's value is two times smaller:
            sigma_r2 = 2 * (CHeVcm / bot.E * 10 * self.period*self.n) / PI2**2
            bot.sourceSIGMAx = ((self.eSigmaX*1e-3)**2 + sigma_r2)**0.5
            bot.sourceSIGMAz = ((self.eSigmaZ*1e-3)**2 + sigma_r2)**0.5
            bot.x[:] += np.random.normal(0, bot.sourceSIGMAx, npassed)
            bot.z[:] += np.random.normal(0, bot.sourceSIGMAz, npassed)

            if bo is None:
                bo = bot
            else:
                bo.concatenate(bot)
            length = len(bo.a)
        if length >= self.nrays:
            bo.accepted = length * self.fluxConst
            bo.acceptedE = bo.E.sum() * self.fluxConst * SIE0
            bo.seeded = seeded
            bo.seededI = seededI
        if length > self.nrays:
            bo.filter_by_index(slice(0, self.nrays))
# normalize (a,b,c):
        norm = (bo.a**2 + 1.0 + bo.c**2)**0.5
        bo.a /= norm
        bo.b /= norm
        bo.c /= norm
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        return bo


class WigglerWS(UndulatorUrgent):
    u"""
    Wiggler source that uses the external code ws. It has some drawbacks,
    as demonstrated in the section :ref:`comparison-synchrotron-sources`, but
    nonetheless can be used for comparison purposes. If you are going to use
    it, the code is freely available as part of XOP package.
    """
    def __init__(self, *args, **kwargs):
        u"""Uses WS code. All the parameters are the same as in
        UndulatorUrgent."""
        kwargs['name'] = kwargs.pop('name', 'WSwiggler')
        kwargs['mode'] = kwargs.pop('mode', 1)
        UndulatorUrgent.__init__(self, *args, **kwargs)

    def run_and_save(self, pp):
        tstart = time.time()
        self.run(iniFileForEachDirectory=True)
        if self.needRecalculate:
            if _DEBUG:
                print('. Finished after {0} seconds'.format(
                      time.time() - tstart))
        tstart = time.time()
        self.splines, self.Imax = self.make_spline_arrays(
            skiprows=18, cols1=(2, 3, 4, 5), cols2=(0, 1, 2, 3, 4))
        if _DEBUG:
            print('. Finished after {0} seconds'.format(time.time() - tstart))

    def code_name(self):
        return 'ws'

    def comment_strings(self):
        return ["#", "#"]

    def prefix_save_name(self):
        return '2-wig-{0}'.format(self.code_name())

    def make_input(self, x, z, E, isBM=False):
        # 1) Name
        # 2) RING-ENERGY CURRENT
        # 3) PERIOD N KX KY
        # 4) EMIN EMAX NE
        # 5) D XPC YPC XPS YPS NXP XYP
        # 6) MODE
        if isBM:
            xxx = self.xPrimeMax / 50.
        else:
            if self.mode == 1:
                xxx = 2 * self.xPrimeMax
            elif self.mode == 2:
                xxx = self.xPrimeMax / self.nx
        infile = ''
        infile += self.name+'\n'
        infile += '{0} {1}\n'.format(self.eE, self.eI*1e3)
        infile += '{0} {1} 0. {2}\n'.format(self.period/10., self.n, self.K)
        infile += '{0} {1} {2}\n'.format(E, self.eMax, self.eN)
        if self.mode == 1:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                0, 0, xxx, 2*self.zPrimeMax, self.nx, self.nz)
        elif self.mode == 2:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                x, z, xxx, self.zPrimeMax/self.nz, self.nx, self.nz)
        infile += '{0}\n'.format(self.mode)
        return infile


class BendingMagnetWS(WigglerWS):
    u"""
    Bending magnet source that uses the external code ws. It has some
    drawbacks, as demonstrated in the section
    :ref:`comparison-synchrotron-sources`, but nonetheless can be used for
    comparison purposes. If you are going to use it, the code is freely
    available as parts of XOP package.
    """
    def __init__(self, *args, **kwargs):
        u"""Uses WS code.

        *B0*: float
            Field in Tesla.

        *K*, *n*, *period* and *nx*:
            Are set internally.

        The other parameters are the same as in UndulatorUrgent.


        """
        kwargs['K'] = 50.
        kwargs['n'] = 0.5
        kwargs['name'] = kwargs.pop('name', 'WSmagnet')
        kwargs['mode'] = kwargs.pop('mode', 1)
        self.B0 = kwargs.pop('B0')
        # kwargs['period'] = kwargs['K'] / (93.36 * self.B0)
        kwargs['period'] = K2B * kwargs['K'] / self.B0
        kwargs['nx'] = 1
        UndulatorUrgent.__init__(self, *args, **kwargs)
        self.rho = 1e9 / SIC * self.eE / self.B0 * 1e3  # mm

    def make_input(self, x, z, E):
        return WigglerWS.make_input(self, 0, z, E, True)

    def prefix_save_name(self):
        return '3-BM-{0}'.format(self.code_name())
