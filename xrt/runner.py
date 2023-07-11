# -*- coding: utf-8 -*-
"""
Module :mod:`runner` defines the entry point of xrt - :func:`run_ray_tracing`,
containers for job properties and functions for running the processes or
threads and accumulating the resulting histograms.
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "26 Mar 2016"

import os
import sys
import time
import inspect
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
import errno
import threading
if sys.version_info < (3, 1):
    import Queue
else:
    import queue
    Queue = queue
import uuid  #  is needed on some platforms with pyopencl  # analysis:ignore

from . import multipro
from .backends import raycing

# _DEBUG = True
__fdir__ = os.path.abspath(os.path.dirname(__file__))
runCardVals = None
runCardProcs = None
_plots = []


def retry_on_eintr(function, *args, **kw):
    """
    Suggested in:
    http://mail.python.org/pipermail/python-list/2011-February/1266462.html
    as a solution for `IOError: [Errno 4] Interrupted system call` in Linux.
    """
    while True:
        try:
            return function(*args, **kw)
        except IOError as e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise


class RunCardVals(object):
    """
    Serves as a global container for a sub-set of run properties passed by the
    user to :func:`run_ray_tracing`. The sub-set is limited to pickleable
    objects for passing it to job processes or threads.
    """
    def __init__(self, threads, processes, repeats, updateEvery, pickleEvery,
                 backend, globalNorm, runfile):
        if threads >= processes:
            self.Event = threading.Event
            self.Queue = Queue.Queue
        else:
            self.Event = multiprocessing.Event
            self.Queue = multiprocessing.Queue

        self.stop_event = self.Event()
        self.finished_event = self.Event()
        self.stop_event.clear()
        self.finished_event.clear()

        self.threads = threads
        self.processes = processes
        self.repeats = repeats
        self.updateEvery = updateEvery
        self.pickleEvery = pickleEvery
        self.backend = backend
        self.globalNorm = globalNorm
        self.runfile = runfile
        self.passNo = 0
        self.savedResults = []
        self.iteration = 0
        self.lastRunsPickleName = os.path.join(__fdir__, 'lastRuns.pickle')
        self.lastRuns = []
        try:
            with open(self.lastRunsPickleName, 'rb') as f:
                self.lastRuns = pickle.load(f)
        except:  # analysis:ignore
            pass
        if self.lastRuns:
            raycing.colorPrint("The last {0} run{1}".format(
                len(self.lastRuns), 's' if len(self.lastRuns) > 1 else ''),
                fcolor='GREEN')
        for lastRun in self.lastRuns:
            if len(lastRun) > 3:
                print("{0}::".format(lastRun[3]))
            st0 = time.strftime("%a, %d %b %Y %H:%M:%S", lastRun[0])
            if (time.strftime("%a, %d %b %Y", lastRun[0]) ==
                    time.strftime("%a, %d %b %Y", lastRun[1])):
                st1 = time.strftime("%H:%M:%S", lastRun[1])
            else:
                st1 = time.strftime("%a, %d %b %Y %H:%M:%S", lastRun[1])
            print("start: {0}; stop: {1}; duration: {2:.1f} s".format(
                st0, st1, lastRun[2]))


class RunCardProcs(object):
    """
    Serves as a global container for a sub-set of run properties passed by the
    user to :func:`run_ray_tracing` limited to functions. These cannot be
    passed to job processes or threads (because are not pickleable) and have to
    be executed by the job server (this module).
    """
    def __init__(self, afterScript, afterScriptArgs, afterScriptKWargs):
        self.afterScript = afterScript
        self.afterScriptArgs = afterScriptArgs
        self.afterScriptKWargs = afterScriptKWargs
        self.generatorNorm = None
        self.generatorPlot = None


def set_repeats(repeats=0):
    if runCardVals is not None:
        runCardVals.repeats = repeats


def _simple_generator():
    """
    The simplest generator for running only one ray-tracing study. Search
    examples for generators that run complex ray-tracing studies.
    """
    yield


def start_jobs():
    """
    Restores the plots if requested and if the persistent files exist and
    starts the qt timer of the 1st plot.
    """
    for plot in _plots:
        if plot.persistentName:
            plot.restore_plots()
        try:
            plot.fig.canvas.manager.set_window_title(plot.title)
        except AttributeError:
            pass

    runCardVals.iteration = np.int64(0)
    noTimer = len(_plots) == 0 or\
        (plt.get_backend().lower() in (x.lower() for x in
                                       mpl.rcsetup.non_interactive_bk))
    if noTimer:
        print("The job is running... ")
        while True:
            sys.stdout.flush()
            res = dispatch_jobs()
            tFromStart = time.time() - runCardVals.tstart
            msg = '{0} of {1} in {2:.1f} s'.format(
                runCardVals.iteration, runCardVals.repeats, tFromStart)
            if os.name == 'posix':
                sys.stdout.write("\r\x1b[K " + msg)
            else:
                sys.stdout.write("\r  ")
                print(msg+' ')
            if res:
                return
    else:
        plot = _plots[0]
        plot.areProcessAlreadyRunning = False
        plot.timer = plot.fig.canvas.new_timer()
        plot.timer.add_callback(plot.timer_callback)
        plot.timer.start()


def dispatch_jobs():
    """Runs the jobs in separate processes or threads and collects the resulted
    histograms from the output queues. One cannot run this function in a loop
    because the redrawing will not work. Instead, it is started from a timer
    event handler of a qt-graph."""
    if (runCardVals.iteration >= runCardVals.repeats) or \
            runCardVals.stop_event.is_set():
        on_finish()
        return True
    one_iteration()
    if (runCardVals.iteration >= runCardVals.repeats) or \
            runCardVals.stop_event.is_set():
        on_finish()
        return True
    if runCardVals.iteration % runCardVals.updateEvery == 0:
        for plot in _plots:
            plot.plot_plots()
    if runCardVals.pickleEvery:
        if runCardVals.iteration % runCardVals.pickleEvery == 0:
            for plot in _plots:
                plot.store_plots()
    if len(_plots) > 0:
        _plots[0].areProcessAlreadyRunning = False


def one_iteration():
    """The body of :func:`dispatch_jobs`."""
    plots2Pickle = [plot.card_copy() for plot in _plots]
    outPlotQueues = [runCardVals.Queue() for plot in _plots]
    alarmQueue = runCardVals.Queue()

# in the 1st iteration the plots may require some of x, y, e limits to be
# calculated and thus this case is special:
    cpus = max(runCardVals.threads, runCardVals.processes)

    if runCardVals.iteration == 0:
        runCardVals.uniqueFirstRun = False
        if hasattr(runCardVals, 'beamLine'):
            bl = runCardVals.beamLine
            bl.forceAlign = False
            for oe in bl.oes + bl.slits + bl.screens:
                if raycing.is_auto_align_required(oe):
                    bl.forceAlign = True
                    runCardVals.uniqueFirstRun = True
                    break

        if not runCardVals.uniqueFirstRun:
            for plot in _plots:
                xLimitsDefined = (plot.xaxis.limits is not None) and\
                    (not isinstance(plot.xaxis.limits, str))
                yLimitsDefined = (plot.yaxis.limits is not None) and\
                    (not isinstance(plot.yaxis.limits, str))
                cLimitsDefined = (plot.caxis.limits is not None) and\
                    (not isinstance(plot.caxis.limits, str)) or plot.ePos == 0
                if not (xLimitsDefined and yLimitsDefined and cLimitsDefined):
                    runCardVals.uniqueFirstRun = True
                    break

        if runCardVals.uniqueFirstRun:
            cpus = 1

    elif runCardVals.iteration == 1:
        if runCardVals.uniqueFirstRun:  # balances the 1st iteration
            cpus -= 1

    if cpus < 1:
        cpus = 1

    if runCardVals.backend.startswith('raycing'):
        runCardVals.beamLine.alarms = []

    if runCardVals.threads >= runCardVals.processes or cpus == 1:
        BackendOrProcess = multipro.BackendThread
    else:
        BackendOrProcess = multipro.BackendProcess
    processes = [BackendOrProcess(runCardVals, plots2Pickle, outPlotQueues,
                                  alarmQueue, icpu) for icpu in range(cpus)]
#    print('top process:', os.getpid())
    for pid, p in enumerate(processes):
        p.ppid = pid + runCardVals.iteration
        p.start()

    for p in processes:
        if runCardVals.backend.startswith('raycing'):
            runCardVals.beamLine.alarms = retry_on_eintr(alarmQueue.get)
            for alarm in runCardVals.beamLine.alarms:
                raycing.colorPrint(alarm, 'RED')
        outList = [0, ]
        for plot, aqueue in zip(_plots, outPlotQueues):
            outList = retry_on_eintr(aqueue.get)

            if len(outList) == 0:
                continue
            if (runCardVals.iteration >= runCardVals.repeats) or \
                    runCardVals.stop_event.is_set():
                continue

            plot.nRaysAll += outList[13]
            if runCardVals.backend.startswith('shadow'):
                plot.nRaysNeeded += outList[14]
            elif runCardVals.backend.startswith('raycing'):
                nRaysVarious = outList[14]
                plot.nRaysAlive += nRaysVarious[0]
                plot.nRaysGood += nRaysVarious[1]
                plot.nRaysOut += nRaysVarious[2]
                plot.nRaysOver += nRaysVarious[3]
                plot.nRaysDead += nRaysVarious[4]
                plot.nRaysAccepted += nRaysVarious[5]
                plot.nRaysAcceptedE += nRaysVarious[6]
                plot.nRaysSeeded += nRaysVarious[7]
                plot.nRaysSeededI += nRaysVarious[8]
                plot.displayAsAbsorbedPower = outList[15]

            for iaxis, axis in enumerate(
                    [plot.xaxis, plot.yaxis, plot.caxis]):
                if (iaxis == 2) and (not plot.ePos):
                    continue
                axis.total1D += outList[0+iaxis*3]
                axis.total1D_RGB += outList[1+iaxis*3]
                if runCardVals.iteration == 0:
                    axis.binEdges = outList[2+iaxis*3]
            plot.total2D += outList[9]
            plot.total2D_RGB += outList[10]
            if plot.fluxKind.lower().endswith('4d'):
                plot.total4D += outList[11]
            elif plot.fluxKind.lower().endswith('pca'):
                plot.total4D.append(outList[11])
            plot.intensity += outList[12]

            if runCardVals.iteration == 0:  # needed for multiprocessing
                plot.set_axes_limits(*outList.pop())

            tFromStart = time.time() - runCardVals.tstart
            plot.textStatus.set_text(
                "{0} of {1} in {2:.1f} s (right click to stop)".format(
                    runCardVals.iteration+1, runCardVals.repeats, tFromStart))
#            aqueue.task_done()

        if len(outList) > 0:
            runCardVals.iteration += 1
    for p in processes:
        p.join(60.)
    if hasattr(runCardVals, 'beamLine'):
        bl = runCardVals.beamLine
        bl.forceAlign = False
        if bl.flowSource == 'legacy':
            bl.flowSource = 'done_once'


def on_finish():
    """Executed on exit from the ray-tracing iteration loop."""
    if len(_plots) > 0:
        plot = _plots[0]
        if plt.get_backend().lower() not in (
                x.lower() for x in mpl.rcsetup.non_interactive_bk):
            plot.timer.stop()
            plot.timer.remove_callback(plot.timer_callback)
        plot.areProcessAlreadyRunning = False
    for plot in _plots:
        if plot.fluxKind.startswith('E') and \
                plot.fluxKind.lower().endswith('pca'):
            xbin, zbin = plot.xaxis.bins, plot.yaxis.bins
            plot.total4D = np.concatenate(plot.total4D).reshape(-1, xbin, zbin)
            plot.field3D = plot.total4D
        plot.textStatus.set_text('')
        plot.fig.canvas.mpl_disconnect(plot.cidp)
        plot.plot_plots()
        plot.save()
    runCardVals.tstop = time.time()
    runCardVals.tstopLong = time.localtime()
    raycing.colorPrint('The ray tracing with {0} iteration{1} took {2:0.1f} s'
                       .format(runCardVals.iteration,
                               's' if runCardVals.iteration > 1 else '',
                               runCardVals.tstop-runCardVals.tstart),
                       fcolor="GREEN")
    runCardVals.finished_event.set()
    for plot in _plots:
        if runCardVals.globalNorm or plot.persistentName:
            plot.store_plots()
    if runCardVals.stop_event.is_set():
        raycing.colorPrint('Interrupted by user after iteration {0}'.format(
            runCardVals.iteration), fcolor='YELLOW')
        return
    try:
        if runCardProcs.generatorPlot is not None:
            if sys.version_info < (3, 1):
                runCardProcs.generatorPlot.next()
            else:
                next(runCardProcs.generatorPlot)
    except StopIteration:
        pass
    else:
        for plot in _plots:
            plot.clean_plots()
        start_jobs()
        return

    if runCardVals.globalNorm:
        aSavedResult = -1
        print('normalizing ...')
        for aRenormalization in runCardProcs.generatorNorm:
            for plot in _plots:
                aSavedResult += 1
                saved = runCardVals.savedResults[aSavedResult]
                plot.clean_plots()
                saved.restore(plot)
                try:
                    plot.fig.canvas.manager.set_window_title(plot.title)
                except AttributeError:
                    pass
                for runCardVals.passNo in [1, 2]:
                    plot.plot_plots()
                    plot.save('_norm' + str(runCardVals.passNo))

    print('finished')

    runCardVals.lastRuns.append([runCardVals.tstartLong, runCardVals.tstopLong,
                                 runCardVals.tstop-runCardVals.tstart,
                                 runCardVals.runfile])
    try:
        with open(runCardVals.lastRunsPickleName, 'wb') as f:
            pickle.dump(runCardVals.lastRuns[-10:], f, protocol=2)
    except OSError:  # Read-only file system
        pass  # no history tracking of last 10 runs

#    plt.close('all')
    if runCardProcs.afterScript:
        runCardProcs.afterScript(
            *runCardProcs.afterScriptArgs, **runCardProcs.afterScriptKWargs)


def normalize_sibling_plots(plots):
    print('normalization started')
    max1Dx = 0
    max1Dy = 0
    max1Dc = 0
    max1Dx_RGB = 0
    max1Dy_RGB = 0
    max1Dc_RGB = 0
    max2D_RGB = 0
    for plot in plots:
        if max1Dx < plot.xaxis.max1D:
            max1Dx = plot.xaxis.max1D
        if max1Dy < plot.yaxis.max1D:
            max1Dy = plot.yaxis.max1D
        if max1Dc < plot.caxis.max1D:
            max1Dc = plot.caxis.max1D
        if max1Dx_RGB < plot.xaxis.max1D_RGB:
            max1Dx_RGB = plot.xaxis.max1D_RGB
        if max1Dy_RGB < plot.yaxis.max1D_RGB:
            max1Dy_RGB = plot.yaxis.max1D_RGB
        if max1Dc_RGB < plot.caxis.max1D_RGB:
            max1Dc_RGB = plot.caxis.max1D_RGB
        if max2D_RGB < plot.max2D_RGB:
            max2D_RGB = plot.max2D_RGB

    for plot in plots:
        plot.xaxis.globalMax1D = max1Dx
        plot.yaxis.globalMax1D = max1Dy
        plot.caxis.globalMax1D = max1Dc
        plot.xaxis.globalMax1D_RGB = max1Dx_RGB
        plot.yaxis.globalMax1D_RGB = max1Dy_RGB
        plot.caxis.globalMax1D_RGB = max1Dc_RGB
        plot.globalMax2D_RGB = max2D_RGB

    for runCardVals.passNo in [1, 2]:
        for plot in plots:
            plot.plot_plots()
            plot.save('_norm' + str(runCardVals.passNo))
    print('normalization finished')


def run_ray_tracing(
    plots=[], repeats=1, updateEvery=1, pickleEvery=None, energyRange=None,
    backend='raycing', beamLine=None, threads=1, processes=1,
    generator=None, generatorArgs=[], generatorKWargs='auto', globalNorm=0,
        afterScript=None, afterScriptArgs=[], afterScriptKWargs={}):
    u"""
    This function is the entry point of xrt.
    Parameters are all optional except the 1st one. Please use them as keyword
    arguments because the list of parameters may change in future versions.

        *plots*: instance of :class:`~xrt.plotter.XYCPlot` or a sequence of
            instances or an empty sequence if no graphical output is wanted.

        *repeats*: int
            The number of ray tracing runs. It should be stressed that
            accumulated are not rays, which would be limited by the physical
            memory, but rather the histograms from each run are summed up. In
            this way the number of rays is unlimited.

        *updateEvery*: int
            Redrawing rate. Redrawing happens when the current iteration index
            is divisible by *updateEvery*.

        *pickleEvery*: int
            Saving rate. Applicable to plots with a defined *persistentName*.
            If None, the pickling will happen once at the end.

        *energyRange*: [*eMin*: float, *eMax*: float]
            Only in `shadow` backend: If not None, sets the energy range of
            shadow source. Alternatively, this can be done directly inside
            the *generator*.

        *backend*: str
            so far supported: {'shadow' | 'raycing' | 'dummy'}

        *beamLine*: instance of :class:`~xrt.backends.raycing.BeamLine`, used
            with `raycing` backend.

        *threads*, *processes*: int or str
            The number of parallel threads or processes, should not be greater
            than the number of cores in your computer, otherwise it gives no
            gain. The bigger of the two will be used as a signal for using
            either :mod:`threading` or :mod:`multiprocessing`. If they are
            equal, :mod:`threading` is used. See also
            :ref:`performance tests<tests>`. If 'all' is given then the number
            returned by multiprocessing.cpu_count() will be used.

            .. warning::
                You cannot use multiprocessing in combination with OpenCL
                because the resources (CPU or GPU) are already shared by
                OpenCL. You will get an error if *processes* > 1. You can still
                use *threads* > 1 but with a little gain.

            .. note::
                For the :mod:`shadow` backend you must create ``tmp0``,
                ``tmp1`` etc. directories (counted by *threads* or *processes*)
                in your working directory. Even if the execution is not
                parallelized, there must be ``tmp0`` with the shadow files
                prepared in it.

        *generator*: generator object
            A generator for running complex ray-tracing studies. It must modify
            the optics, specify the graph limits, define the output file names
            etc. in a loop and return to xrt by ``yield``.
            See the supplied examples.

        *generatorArgs*, *generatorKWargs*: list and (dictionary or 'auto')
            If *generatorKWargs* is 'auto', the following keyword dictionary
            will be used for the generator: kwargs = {} if *generator* is
            defined within the caller of :func:`run_ray_tracing` or if
            *generatorArgs* is not empty, otherwise
            kwargs = {'plots'=pots, 'beamLine'=beamLine}.

        .. _globalNorm:

        *globalNorm*: bool
            If True, the intensity of the histograms will be normalized to the
            global maximum throughout the series of graphs. There are two
            flavors of normalization:

            1) only the heights of 1D histograms are globally normalized while
               the brightness is kept with the normalization to the local
               maximum (i.e. the maximum in the given graph).
            2) both the heights of 1D histograms and the brightness of 1D and
               2D histograms are globally normalized.

            The second way is physically more correct but sometimes is less
            visual: some of the normalized pictures may become too dark, e.g.
            when you compare focused and strongly unfocused images. Both
            normalizations are saved with suffixes ``_norm1`` and ``_norm2``
            for you to select the better one.

            Here is a normalization example where the intensity maximum was
            found throughout a series of images for filters of different
            thickness. The brightest image was for the case of no filter (not
            shown here) and the normalization shown below was done relative to
            that image:

            +------------------+-----------------------------------------+
            | normalized       |                                         |
            | to local maximum |              |image_nonorm|             |
            +------------------+-----------------------------------------+
            | global           |                                         |
            | normalization,   |                                         |
            | type 1           |              |image_norm1|              |
            +------------------+-----------------------------------------+
            | global           |                                         |
            | normalization,   |                                         |
            | type 2           |              |image_norm2|              |
            +------------------+-----------------------------------------+

            .. |image_nonorm| imagezoom:: _images/filterFootprint2_I400mum.png
               :scale: 50 %
            .. |image_norm1| imagezoom:: _images/filterFootprint2_I400mum_norm1.png
               :scale: 50 %
            .. |image_norm2| imagezoom:: _images/filterFootprint2_I400mum_norm2.png
               :scale: 50 %

        *afterScript*: function object
            This function is executed at the end of the current script. For
            example, it may run the next ray-tracing script.

        *afterScriptArgs*, *afterScriptKWargs*: list and dictionary
            args and kwargs for *afterScript*.


    """
    global runCardVals, runCardProcs, _plots
    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    runfile = mod.__file__
    # patch for starting a script with processes>1 from Spyder console
    if not hasattr(mod, "__spec__"):
        mod.__spec__ = None

    if isinstance(plots, (list, tuple)):
        _plots = plots
    else:
        _plots = [plots, ]
    for plot in _plots:
        if backend == 'raycing':
            if plot.caxis.useCategory:
                plot.caxis.limits = [raycing.hueMin, raycing.hueMax]
            if isinstance(plot.rayFlag, int):
                plot.rayFlag = plot.rayFlag,
    if updateEvery < 1:
        updateEvery = 1
    if (repeats > 1) and (updateEvery > repeats):
        updateEvery = repeats
    cpuCount = multiprocessing.cpu_count()
    if isinstance(processes, str):
        if processes.startswith('a'):  # all
            processes = cpuCount
        else:
            processes = max(cpuCount // 2, 1)
    if isinstance(threads, str):
        if threads.startswith('a'):  # all
            threads = cpuCount
        else:
            threads = max(cpuCount // 2, 1)
    runCardVals = RunCardVals(threads, processes, repeats, updateEvery,
                              pickleEvery, backend, globalNorm, runfile)
    runCardProcs = RunCardProcs(
        afterScript, afterScriptArgs, afterScriptKWargs)

    runCardVals.cwd = os.getcwd()
    if backend.startswith('shadow'):
        from .backends import shadow
        cpuCount = max(processes, threads)
        shadow.check_shadow_dirs(cpuCount, runCardVals.cwd)
        runCardVals.fWiggler, runCardVals.fPolar, runCardVals.blockNRays = \
            shadow.init_shadow(cpuCount, runCardVals.cwd, energyRange)
    elif backend == 'raycing':
        runCardVals.beamLine = beamLine

    if generator is None:
        runCardProcs.generatorPlot = _simple_generator()
    else:
        if generatorKWargs == 'auto':
            if (generator.__name__ in sys._getframe(1).f_locals) or\
                    len(generatorArgs) > 0:
                # generator is defined within the caller function
                kwargs = {}
            else:
                # outside the caller
                kwargs = {'plots': plots, 'beamLine': beamLine}
        else:
            kwargs = generatorKWargs
        runCardProcs.generatorPlot = generator(*generatorArgs, **kwargs)
        if globalNorm:
            runCardProcs.generatorNorm = generator(*generatorArgs, **kwargs)

    if runCardProcs.generatorPlot is not None:
        if sys.version_info < (3, 1):
            runCardProcs.generatorPlot.next()
        else:
            next(runCardProcs.generatorPlot)

    runCardVals.tstart = time.time()
    runCardVals.tstartLong = time.localtime()
    start_jobs()
    plt.show()
