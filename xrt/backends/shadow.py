# -*- coding: utf-8 -*-
"""
This is a deprecated backend. :mod:`~xrt.backends.raycing` is much more
functional. Module :mod:`~xrt.backends.shadow` works with shadow input files,
starts the ray-tracing and gets its output.

Description of shadow
---------------------

... can be found in the manual pages of your shadow distribution. In connection
with using shadow in xrt, it is important to understand the naming of
the output beam files and the naming of parameters in the setup files.

Preparation for a shadow run
----------------------------

.. note:: on shadow under Windows Vista and 7:

    Under Windows Vista and 7 shadow does not work out of the box because of
    ``epath`` (a part of shadow) reporting an error. There is a workaround
    consisting of simply stopping the Windows’ Error Reporting Service.

Create a folder where you will store your ray-tracing script and the output
images. Make it as Python's working directory. Create there a sub-folder
``tmp0``. Put there your shadow project file along with all necessary data
files (reflectivities, crystal parameters etc.). Run shadow and make sure it
correctly produces output files you want to accumulate (like ``star.01``).

Now you need to generate two command files that run shadow source and shadow
trace. These are system-specific and also differ for different shadow sources.
Under Windows, this can be done as follows: set the working directory of
shadowVUI as your ``tmp0``, run Source in shadowVUI and rename the produced
``shadowvui.bat`` to ``shadow-source.bat``; then run Trace and rename the
produced ``shadowvui.bat`` to ``shadow-trace.bat``.

Try to run the generated command files in order to check their validity.

If you want to use multi-threading then copy ``tmp0`` to ``tmp1``, ``tmp2``
etc. (in total as many directories as you have threads).

.. _scriptingShadow:

Scripting in python
-------------------

The simplest script consists of 4 lines::

    import xrt.runner as xrtr
    import xrt.plotter as xrtp
    plot1 = xrtp.XYCPlot('star.01')
    xrtr.run_ray_tracing(plot1, repeats=40, updateEvery=2, threads=1)
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "10 Apr 2015"

import os
import time
import numpy as np
import subprocess

# import psyco  #!!!psyco speeds it up!!!
# psyco.full()

_sourceAsciiFile = 'start.00'
# _DEBUG = True


def read_input(fileName, vtype, *getlines):
    """
    reads a shadow text input file (like ``start.NN``) which consists of lines
    ``field = value``.

    Parameters:
        *fileName*: str
        *vtype*: {int|str|float}
            Type of the returned value.
        *getlines*: list of strings

    Returns:
        *results*: list
            a list of values which correspond to the list *getlines* if
            successful, otherwise -1.

    Example:
        >>> fPolar = read_input('start.00', int, 'f_polar')[0]

    """
    lines = []
    f = None
    try:
        f = open(fileName, 'rU')
        for line in f:
            lines.append(line.split())
    except IOError:
        print("The file ", fileName, " does not exist or corrupt!")
        return -1
    finally:
        if f:
            f.close()

    results = []
    for el in getlines:
        for line in lines:
            if line[0].lower() == el.lower():
                results.append(vtype(line[2]))
                break

    if len(results) == 0:
        raise Exception(
            "The parameter(s) %s cannot be found in %s" % (getlines, fileName))
    return results


def modify_input(fileNameList, *editlines):
    """
    modifies a shadow text input file (like ``start.NN``) which consists of
    lines ``field = value``.

    Parameters:
        *fileNameList*: str or list of str
            A list of file names is useful when executing shadow in parallel in
            several directories.

        *editlines*: list of tuples of strings (field, value).

    Returns:
        0 if successful, otherwise -1.

    Example:
        >>> modify_input('start.00',('istar1',str(seed)))  #change seed

    """
    if isinstance(fileNameList, str):
        locFileNameList = [fileNameList, ]
    elif isinstance(fileNameList, (tuple, list)):
        locFileNameList = fileNameList
    else:
        print(type(fileNameList))
        print("Wrong fileNameList parameter!")
        return -1

    for fileName in locFileNameList:
        lines = []
        f = None
        try:
            f = open(fileName, 'rU')
            for line in f:
                lines.append(line.split())
        except IOError:
            print("The file ", fileName, " does not exist or corrupt!")
            return -1
        finally:
            if f:
                f.close()

        for el in editlines:
            for line in lines:
                if line[0].lower() == el[0].lower():
                    line[2] = el[1]
                    if len(line) > 2:  # otherwise trapped by "none specified"
                        del line[3:]
                    break

        f = None
        try:
            f = open(fileName, 'w')
            for line in lines:
                f.write(' '.join(line) + '\n')
        finally:
            if f:
                f.close()
    return 0


def modify_xsh_input(fileNameList, *editlines):
    """
    modifies a shadow xsh text input file (like ``xsh_nphoton_tmp.inp``) which
    consist of lines of values, one value per line.

    Parameters:
        *fileNameList*: str or list of str
            A list of file names is useful when executing shadow in parallel in
            several directories.
        *editlines*: list of tuples (*fieldNo*: int, *value*: str)
            *fieldNo* is zero-based index of the modified parameter.

    Returns:
        0 if successful, otherwise -1.

    Example:
        >>> modify_xsh_input('xsh_nphoton_tmp.inp', (2, energyRange[0]),
                             (3, energyRange[1]))

    """
    if isinstance(fileNameList, str):
        locFileNameList = [fileNameList, ]
    elif isinstance(fileNameList, (tuple, list)):
        locFileNameList = fileNameList
    else:
        print("Wrong fileNameList parameter!")
        return -1

    for fileName in locFileNameList:
        lines = []
        tryAgain = True
        f = None
        while tryAgain:
            # several attempts instead of file locking, which is
            # easier because is not system dependent.
            try:
                f = open(fileName, 'rU')
                for line in f:
                    lines.append(line)
            except IOError:
                print("The file ", fileName, " does not exist or corrupt!")
                return -1
            finally:
                if f:
                    f.close()
            try:
                for el in editlines:
                    lines[el[0]] = str(el[1]) + '\n'
            except:
                time.sleep(0.1)
                continue

            f = None
            try:
                f = open(fileName, 'w')
                for line in lines:
                    f.write(line)
            except:
                time.sleep(0.1)
                continue
            finally:
                if f:
                    f.close()

            f = None
            try:
                f = open(fileName, 'rU')
                fsize = os.path.getsize(fileName)
                if fsize <= 0:
                    raise IOError()
            except IOError:
                time.sleep(0.1)
                continue
            else:
                tryAgain = False
            finally:
                if f:
                    f.close()
    return 0


def read_bin_file(binFileName, _f_polar, _blockNRays, lostRayFlag=1):
    """
    Reads a binary shadow output file (like ``star.NN``, ``mirr.NN`` or
    ``screen.NNMM``).

    Parameters:
        *binFileName*: str
        *f_polar*: int
           determines the number of columns stored in binFileName.
        *lostRayFlag*: 1=only good, 0=only lost, 2=all rays

    Returns:
        *d*: ndarray, shape(*NumberOfRays*, *NumberOfColumns*)
            *NumberOfRays* is determined by *lostRayFlag* field.
            *NumberOfColumns* is determined by the field *f_polar* in the
            source input file.
        *intensity*: ndarray, shape(*NumberOfRays*,)
            The array of intensity for each ray.
    """
    if _f_polar == 1:
        shadowColums = 19
    else:
        shadowColums = 14
    tryAgain = True
    while tryAgain:
        # several attempts instead of file locking, which is
        # easier because is not system dependent.
        f = None
        try:
            f = open(binFileName, 'rb')
    #        header =
            np.fromfile(f, dtype=np.uint32, count=6)
# shadow writes rays in a weird way (see putrays.pro).
# It writes a dummy structure tmp = byte([12,0,0,0]) twice in
# every write operator (why?):
#   writeu,Unit,tmp,a.ncol,a.npoint,0L,tmp      ;write the header information
#   writeu,Unit,tmp,ray,tmp
# The array of rays (25000x14) or (25000x19) appears to start at the offset
# of 24 bytes but the file size is 4 bytes less (why?).
# Therefore I add one zero field at the end in order to be able to reshape.
            d = np.fromfile(f, dtype=np.float64)
        except IOError:
            print("The file ", binFileName, " does not exist or corrupt!")
            return -1
        finally:
            if f:
                f.close()
        try:
            d = np.concatenate((d, [0, ]))
            d = d.reshape(-1, shadowColums)
            locNrays = d.shape[0]
            if locNrays != _blockNRays:
                raise ValueError()
        except ValueError:
            time.sleep(0.1)
            continue
        else:
            tryAgain = False

    if lostRayFlag == 1:
        d = d[d[:, 9] == 1]
    elif lostRayFlag == 0:
        d = d[d[:, 9] < 0]

# energy in shadow is in cm^-1:
    d[:, 10] /= 50676.89778440964400  # 1/ch constant [eV cm]^-1

    intensity = np.square(d[:, 6]) + np.square(d[:, 7]) + np.square(d[:, 8])
    if _f_polar == 1:
        intensity += \
            np.square(d[:, 15]) + np.square(d[:, 16]) + np.square(d[:, 17])

# it returns a matrix with the columns listed in class XYCPlot
    return d, intensity, locNrays


def check_shadow_dirs(processes, cwd):
    """
    Assures that tmp0, tmp1 etc. exist
    """
    nonExistingDirs = []
    for iprocess in range(processes):
        tmpDir = cwd + os.sep + 'tmp' + str(iprocess)
        if not os.path.exists(tmpDir):
            nonExistingDirs.append(tmpDir)
    if len(nonExistingDirs) > 0:
        if len(nonExistingDirs) == 1:
            raise Exception("directory %s must exist!" % nonExistingDirs[0])
        else:
            raise Exception("directories %s must exist!" % nonExistingDirs)


def init_shadow(processes, cwd, energyRange):
    """
    Initializes the work of shadow: determines the source type, the number of
    columns and the number of rays.
    """
    tmpDir = 'tmp0' + os.sep
    # determines the source type, which determines the input type
    fWiggler = read_input(tmpDir + _sourceAsciiFile, int, 'f_wiggler')
    if fWiggler == -1:
        return
    _fWiggler = fWiggler[0]
    # reads f_polar field, which determines the number of columns
    fPolar = read_input(tmpDir + _sourceAsciiFile, int, 'f_polar')
    if fPolar == -1:
        return
    _fPolar = fPolar[0]
    # reads the number of rays in each shadow run, usually =25000.
    blockNRays = read_input(tmpDir + _sourceAsciiFile, int, 'npoint')
    if blockNRays == -1:
        return
    _blockNRays = blockNRays[0]

    for iprocess in range(processes):
        tmpDir = cwd + os.sep + 'tmp' + str(iprocess)
        init_process(tmpDir, energyRange, _fWiggler)
#    print("init shadow finished")
    return _fWiggler, _fPolar, _blockNRays


def init_process(runDir, energyRange, _fWiggler):
    """
    Sets the energy range.
    """
    if energyRange is not None:  # change Emin and Emax
        if _fWiggler == 1:
            if modify_xsh_input(
                runDir + os.sep + 'xsh_nphoton_tmp.inp',
                    (2, energyRange[0]), (3, energyRange[1])) == -1:
                return
        elif _fWiggler == 2:
            if modify_xsh_input(
                runDir + os.sep + 'xsh_undul_set_tmp.inp',
                    (7, energyRange[0]), (8, energyRange[1])) == -1:
                return
        if modify_input(runDir + os.sep + _sourceAsciiFile,
                        ('ph1', str(energyRange[0])),
                        ('ph2', str(energyRange[1]))) == -1:
            return


def files_in_tmp_subdirs(fileName, processes=1):
    """
    Creates and returns a list of full file names of copies of a given file
    located in the process directories. This list is needed for reading and
    writing to several versions of one file (one for each process) in one go.
    Useful in user's scripts.

    Example:
        >>> start01 = shadow.files_in_tmp_subdirs('start.01', processes=4)
        >>> shadow.modify_input(start01, ('THICK(1)', str(thick * 1e-4)))
    """
    filesInTmpSubdirsList = []
    for iprocess in range(processes):
        fName = 'tmp' + str(iprocess) + os.sep + fileName
        filesInTmpSubdirsList.append(fName)
    return filesInTmpSubdirsList


def run_process(args, _fWiggler, runDir):
    """
    Changes the seed for shadow Source and runs shadow.

    Parameters
    ----------
    *args*: str
        What to run: 'source' or 'trace'
    """
    if args == 'source':
        # np.random.seed(0)
        seed = 2 * np.random.random_integers(50, 5e4-1) + 1
        modify_input(runDir + os.sep + _sourceAsciiFile,
                     ('istar1', str(seed)))  # change seed
        if _fWiggler != 0:
            # change seed
            modify_xsh_input(
                runDir + os.sep + 'xsh_input_source_tmp.inp', (3, seed))

    if os.name == 'nt':
        genStr = ''.join(['shadow-', args, '.bat'])
        close_fds = False
    elif os.name == 'posix':
        genStr = ''.join(['./', 'shadow-', args, '.sh'])
        close_fds = True
    else:
        print("not supported OS")
        return -10

    errptr = None
    outptr = None
    # Create output log file
    outFile = os.path.join(runDir, ''.join(['output_', args, '.log']))
    outptr = open(outFile, "w")
    # Create error log file
    errFile = os.path.join(runDir, ''.join(['error_', args, '.log']))
    errptr = open(errFile, "w")
    try:
        retcode = subprocess.call(genStr, shell=True, close_fds=close_fds,
                                  stdout=outptr, stderr=errptr, cwd=runDir)
        if retcode != 0:
            errData = errptr.read()
            raise Exception("Error executing command: " + repr(errData))
    except Exception as inst:
        print(inst.args, ", ignored...")
        return retcode
    finally:
        # Close log handles
        if errptr:
            errptr.close()
        if outptr:
            outptr.close()
    return retcode


def get_output(plot, _fPolar, _blockNRays, dir=None):
    """
    Returns the plotting arrays from the shadow output.
    """
    if dir:
        fName = ''.join([dir, os.sep, plot.beam])
    else:
        fName = plot.beam
    raysArray, intensity, locNrays = read_bin_file(fName, _fPolar, _blockNRays,
                                                   plot.rayFlag)
    if isinstance(plot.xaxis.data, int):
        x = raysArray[:, plot.xaxis.data] * plot.xaxis.factor
    else:
        x = plot.xaxis.data * plot.xaxis.factor
    if isinstance(plot.yaxis.data, int):
        y = raysArray[:, plot.yaxis.data] * plot.yaxis.factor
    else:
        y = plot.yaxis.data * plot.yaxis.factor
    if isinstance(plot.caxis.data, int):
        cData = raysArray[:, plot.caxis.data] * plot.caxis.factor
    else:
        cData = plot.caxis.data * plot.caxis.factor
    locNraysNeeded = raysArray.shape[0]
    return x, y, intensity, cData, locNrays, locNraysNeeded
