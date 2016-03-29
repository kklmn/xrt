# -*- coding: utf-8 -*-
"""
Module :mod:`multipro` defines :class:`BackendProcess` as a subclass of
``multiprocessing.Process`` or ``threading.Thread``. You can opt between
deriving from :mod:`multiprocessing` or :mod:`threading` by selecting the
corresponding parameter in :func:`~xrt.runner.run_ray_tracing`. The
multiprocessing is normally faster than multithreading but has an inconvenience
when the user aborts the execution: the processes have to be killed manually.
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "26 Mar 2016"

import os
import time
from multiprocessing import Process
from threading import Thread
import numpy as np
from . import kde
import matplotlib as mpl

from .backends import shadow
from .backends import dummy
from .backends import raycing

try:
    import pyopencl as cl
    isOpenCL = True
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
except ImportError:
    isOpenCL = False
__dir__ = os.path.dirname(__file__)


_DEBUG = 1


class GenericProcessOrThread(object):
    """
    Defines a ray tracing process or thread that can run in parallel execution.
    If the backend is 'shadow', the working directory of the process or the
    thread is changed to the corresponding 'tmpNN' directory (see
    mod:`shadow`).
    """
    def __init__(self, locCard, plots, outPlotQueues, alarmQueue, idLoc):
        self.status = -1
        if locCard.backend.startswith('shadow'):
            self.runDir = locCard.cwd + os.sep + 'tmp' + str(idLoc)
        self.idN = idLoc
        self.status = 0
        self.plots = plots
        self.outPlotQueues = outPlotQueues
        self.alarmQueue = alarmQueue
        self.card = locCard
        isOpenCL = False
        self.cl_ctx = None
        if isOpenCL:
            iDevice = None
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    if device.type == 2:
                        iDevice = device
                        break
                if iDevice is not None:
                    break
            if iDevice is not None:
                self.cl_ctx = cl.Context(devices=[iDevice])
                self.cl_queue = cl.CommandQueue(self.cl_ctx)
                cl_file = os.path.join(__dir__, r'hist.cl')
                with open(cl_file, 'r') as f:
                    kernelsource = f.read()
                self.cl_program = cl.Program(self.cl_ctx, kernelsource).build()
                self.cl_mf = cl.mem_flags

    def hist1d_cl(self, x, bins, range, weights):
        cl_weights = np.float64(np.array(weights))
        cl_x = np.float64(np.array(x))
        cl_bins = np.int32(bins)
        hist_out = np.zeros(bins, dtype=np.float64)
        locker = np.zeros(bins, dtype=np.int32)
        r_max = np.max(range)
        r_min = np.min(range)

        cl_a = np.float64(np.float64(bins) / (r_max - r_min))
        cl_b = np.float64(np.float64(bins) * r_min / (r_max - r_min))

        cl_bin_edges = np.linspace(r_min, r_max, bins + 1)

        x_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                          self.cl_mf.COPY_HOST_PTR, hostbuf=cl_x)
        weights_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                                self.cl_mf.COPY_HOST_PTR, hostbuf=cl_weights)
        hist_out_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE |
                                 self.cl_mf.COPY_HOST_PTR, hostbuf=hist_out)
        locker_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE |
                               self.cl_mf.COPY_HOST_PTR, hostbuf=locker)

        global_size = cl_x.shape
        local_size = None
        self.cl_program.hist_1d(self.cl_queue,
                                global_size,
                                local_size,
                                cl_a, cl_b, cl_bins,
                                x_buf, weights_buf, hist_out_buf,
                                locker_buf
                                ).wait()
        cl.enqueue_read_buffer(self.cl_queue,
                               hist_out_buf,
                               hist_out).wait()
        return hist_out, cl_bin_edges

    def hist2d_cl(self, y, x, bins, range, weights):
        cl_weights = np.float64(np.array(weights))
        cl_x = np.float64(np.array(x))
        cl_y = np.float64(np.array(y))

        cl_xbins = np.int32(bins[1])
        cl_ybins = np.int32(bins[0])

        hist_out = np.zeros(bins[1]*bins[0])
        locker = np.zeros(bins[1]*bins[0], dtype=np.int32)

        rx_max = np.max(np.array(range)[1, :])
        ry_max = np.max(np.array(range)[0, :])

        rx_min = np.min(np.array(range)[1, :])
        ry_min = np.min(np.array(range)[0, :])

        cl_ax = np.float64(bins[1] / (rx_max - rx_min))
        cl_ay = np.float64(bins[0] / (ry_max - ry_min))

        cl_bx = np.float64(bins[1] * rx_min / (rx_max - rx_min))
        cl_by = np.float64(bins[0] * ry_min / (ry_max - ry_min))

        cl_xbin_edges = np.linspace(rx_min, rx_max, bins[1] + 1)
        cl_ybin_edges = np.linspace(ry_min, ry_max, bins[0] + 1)

        x_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                          self.cl_mf.COPY_HOST_PTR, hostbuf=cl_x)
        y_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                          self.cl_mf.COPY_HOST_PTR, hostbuf=cl_y)
        weights_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                                self.cl_mf.COPY_HOST_PTR, hostbuf=cl_weights)
        hist_out_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE |
                                 self.cl_mf.COPY_HOST_PTR, hostbuf=hist_out)
        locker_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE |
                               self.cl_mf.COPY_HOST_PTR, hostbuf=locker)

        global_size = cl_x.shape
        local_size = None

        self.cl_program.hist_2d(self.cl_queue,
                                global_size,
                                local_size,
                                cl_ax, cl_bx, cl_ay, cl_by,
                                cl_xbins, cl_ybins,
                                x_buf, y_buf,
                                weights_buf,
                                hist_out_buf,
                                locker_buf
                                ).wait()
        cl.enqueue_read_buffer(self.cl_queue,
                               hist_out_buf,
                               hist_out).wait()
        return hist_out.reshape(bins[0], bins[1]), cl_ybin_edges, cl_xbin_edges

    def do_hist1d(self, x, intensity, cDataRGB, axis):
        """
        Calculates the specified 1D histogram.
        *x, intensity*: ndarray, shape(NumberOfRays,)
            arrays of position and intensity
        *cDataRGB*: ndarray, shape(NumberOfRays, 3)
            used for weighing the histogram in order to colorize it
        *axis*: XYCAxis instance
            the abscissa of the 1D histogram."""
        hist1dRGB = np.zeros((axis.bins, 3))
        if axis.density.lower() == 'kde':
            if axis.limits is None:
                binEdges = np.linspace(x.min(), x.max(), axis.bins+1)
            else:
                binEdges = np.linspace(
                    axis.limits[0], axis.limits[1], axis.bins+1)
            binCenters = (binEdges[:-1] + binEdges[1:]) / 2.

            kdeobj = kde.Gaussian_kde(x, weights=intensity)
            hist1d = kdeobj(binCenters)
            if cDataRGB is not None:
                for i in range(3):  # over RGB components
                    kdeobj = kde.Gaussian_kde(x, weights=cDataRGB[:, i])
                    norm = cDataRGB[:, i].sum()
                    hist1dRGB[:, i] = kdeobj(binCenters)
                    hist1dRGB[:, i] *= norm / hist1dRGB[:, i].sum()
        else:
            if self.cl_ctx is None:
                histogram = np.histogram
            else:
                histogram = self.hist1d_cl

            hist1d, binEdges = histogram(
                x, bins=axis.bins, range=axis.limits, weights=intensity)
            if cDataRGB is not None:
                for i in range(3):  # over RGB components
                    hist1dRGB[:, i], binEdges = histogram(
                        x, bins=axis.bins, range=axis.limits,
                        weights=cDataRGB[:, i])
        return hist1d, hist1dRGB, binEdges

    def do_histXXZZ(self, x, intensity, cDataRGB, axis):
        """
        Used for 2D mutual intensity functions X1X2 or Y1Y2.
        """
        hist1dr, hist1dRGB, binEdges =\
            self.do_hist1d(x, intensity.real, None, axis)
        hist1di, hist1dRGB, binEdges =\
            self.do_hist1d(x, intensity.imag, cDataRGB, axis)
        xs = hist1dr + 1j*hist1di
        hist2d = np.outer(xs, xs.conjugate())

        hist2dRGB = np.zeros((axis.bins, axis.bins, 3))
        for i in range(3):  # over RGB components
            hist2dRGB[:, :, i] = np.outer(hist1dRGB[:, i], hist1dRGB[:, i])
        return hist2d, hist2dRGB

    def do_hist2d(self, x, y, intensity, cDataRGB, plot):
        """
        Calculates the 2D histogram.
        *x, y, intensity*: ndarray, shape(NumberOfRays,)
            arrays of positions and intensity
        *cDataRGB*: ndarray, shape(NumberOfRays, 3)
            used for weighing the histogram in order to colorize it
        *plot* instance of :class:`XYCPlot`: the plot hosting the 2D histogram.

        If *plot.fluxKind* starts with 'E' then the mutual intensity is
        calculated:

            - If *plot.fluxKind* ends with 'xx' or 'zz', the corresponding 2D
              cuts are done. The *plot* must have equal axes.

            - If without these endings, one of the two points in the the mutual
              intensity is fixed to the center of the image.

            - If *plot.fluxKind* ends with '4D', the complete mutual intensity
              is calculated in stored in stored in *plot.hist4d*.

            .. warning::

                Be cautious with the size of the mutual intensity object, it is
                four-dimensional!

        """
        hist4d = None
        if self.cl_ctx is None:
            histogram2d = np.histogram2d
        else:
            histogram2d = self.hist2d_cl

        xyrange = [plot.yaxis.limits, plot.xaxis.limits]
        if not (raycing.is_sequence(plot.xaxis.limits) and
                raycing.is_sequence(plot.yaxis.limits)):
            raise ValueError()
        xybins = [plot.yaxis.bins, plot.xaxis.bins]

        if plot.fluxKind.startswith('E'):
            if plot.fluxKind.lower().endswith('xx'):
                return self.do_histXXZZ(x, intensity, cDataRGB, plot.xaxis)
            elif plot.fluxKind.lower().endswith('zz') \
                    or plot.fluxKind.lower().endswith('yy'):
                return self.do_histXXZZ(y, intensity, cDataRGB, plot.yaxis)

#            if not (raycing.is_sequence(plot.xaxis.limits) and
#                raycing.is_sequence(plot.yaxis.limits)):
#                hist2dr, yedges, xedges = histogram2d(
#                    y, x, bins=xybins, range=xyrange, weights=intensity.real)
#                xyrange = [[yedges[0], yedges[-1]], [xedges[0], xedges[-1]]]
            hist2dr, t1, t2 = histogram2d(
                y, x, bins=xybins, range=xyrange, weights=intensity.real)
            hist2di, t1, t2 = histogram2d(
                y, x, bins=xybins, range=xyrange, weights=intensity.imag)
            hist2d = hist2dr + 1j*hist2di

            size2D = plot.yaxis.bins * plot.xaxis.bins
            if plot.fluxKind.lower().endswith('4d'):
                hist4d = np.outer(hist2d, hist2d.conjugate())
            elif plot.fluxKind.lower().endswith('pca'):
                hist4d = np.zeros((size2D, size2D), dtype=np.complex128)
                if self.ppid < size2D:
                    hist4d[:, self.ppid] = hist2d.flatten()
                else:
                    print('Warning: too many images (repeats) to save for PCA!'
                          'The next repeats will be ignored.')

# equivalent to np.outer(hist2d.flatten(), hist2d.flatten().conjugate())
            fl = hist2d.flatten()
            central = fl[len(fl)//2]
            hist2d *= central.conjugate()
        else:
            hist2d, yedges, xedges = histogram2d(
                y, x, bins=xybins, range=xyrange, weights=intensity)

        hist2dRGB = np.zeros((xybins[0], xybins[1], 3))
        if len(x) > 0:
            for i in range(3):  # over RGB components
                hist2dRGB[:, :, i], yedges, xedges = histogram2d(
                    y, x, bins=xybins, range=xyrange, weights=cDataRGB[:, i])
        return hist2d, hist2dRGB, hist4d

    def update_limits(self, axis, x):
        """
        Updates the *axis* limits given the data in *x*. Used at the 1st
        iteration."""
        if (axis.limits is None) or isinstance(axis.limits, str):
            if len(x) > 1:
                xmin, xmax = np.min(x), np.max(x)
                dx = axis.extraMargin * (xmax-xmin) / axis.bins
                xmin -= dx
                xmax += dx
                if xmin == xmax:
                    xmin -= 1.
                    xmax += 1.
            else:
                xmin, xmax = 1., 10.
            if isinstance(axis.limits, str):
                xmm = max(abs(xmin), abs(xmax))
                xmin, xmax = -xmm, xmm
            axis.limits = [xmin, xmax]
        else:
            xmin, xmax = axis.limits[0], axis.limits[1]
        return xmin, xmax

    def equalize_xy(self, plot, leadingLimits):
        """
        Updates the limits of *xaxis* and *yaxis* according to the given
        *aspect*.
        """
        if plot.aspect == 'equal':
            plot.aspect = 1.0
        if not isinstance(plot.aspect, float):
            return
        xaxis = plot.xaxis
        yaxis = plot.yaxis
        aspect = plot.aspect * xaxis.pixels / float(yaxis.pixels)
        dx = xaxis.limits[1] - xaxis.limits[0]
        dy = yaxis.limits[1] - yaxis.limits[0]
        if aspect == 1.0 and dx == dy:
            return

        if leadingLimits is None:
            if dx > (dy * aspect):
                leadingLimits = 'x'
            else:
                leadingLimits = 'y'
        if leadingLimits == 'x':
            yMid = (yaxis.limits[1]+yaxis.limits[0]) / 2.
            dy2 = dx / aspect / 2
            yaxis.limits = [yMid-dy2, yMid+dy2]
        else:
            xMid = (xaxis.limits[1]+xaxis.limits[0]) / 2.
            dx2 = dy * aspect / 2
            xaxis.limits = [xMid-dx2, xMid+dx2]
        return xaxis.limits[0], xaxis.limits[1], yaxis.limits[0],\
            yaxis.limits[1]

    def run(self):
        """
        Starts the chosen ray-tracing backend, invokes the 1D and 2D
        histogramming routines and puts them into the output queue.
        """
        seed = int(time.time()) ^ (os.getpid()+self.idN)
#        random.seed(seed) - has no effect!
        np.random.seed(seed)
        if _DEBUG > 2:
            print(seed)
        if _DEBUG > 2:
            print('parent process id:{0}, process id{1}'.format(
                  os.getppid(), os.getpid()))
        if self.card.backend.startswith('shadow'):
            self.alarmQueue.put([])
            ret = shadow.run_process(
                'source', self.card.fWiggler, self.runDir)
            if ret != 0:
                for queue in self.outPlotQueues:
                    queue.put([])
                return
            if self.card.backend.startswith('shadow'):
                time.sleep(0.1)
            if not self.card.backend.startswith('shadow0'):
                ret = shadow.run_process(
                    'trace', self.card.fWiggler, self.runDir)
                if ret != 0:
                    for queue in self.outPlotQueues:
                        queue.put([])
                    return
                if self.card.backend.startswith('shadow'):
                    time.sleep(0.1)
        elif self.card.backend.startswith('dummy'):
            dummy_output = dummy.run_process()
            self.alarmQueue.put([])
        elif self.card.backend.startswith('raycing'):
            raycing_output = raycing.run.run_process(self.card.beamLine)
            self.alarmQueue.put(self.card.beamLine.alarms)

        for plot, queue in zip(self.plots, self.outPlotQueues):
            displayAsAbsorbedPower = False
            if self.card.backend.startswith('shadow'):
                x, y, intensity, cData, locNrays, locNraysNeeded = \
                    shadow.get_output(
                        plot, self.card.fPolar, self.card.blockNRays,
                        self.runDir)
            elif self.card.backend.startswith('raycing'):
                x, y, intensity, cData, locNrays, locAlive, locGood, locOut,\
                    locOver, locDead, locAccepted, locAcceptedE, locSeeded,\
                    locSeededI = raycing.get_output(plot, raycing_output)
                if hasattr(plot, 'displayAsAbsorbedPower'):
                    displayAsAbsorbedPower = True
            elif self.card.backend.startswith('dummy'):
                x, y, intensity, cData, locNrays = dummy_output

            if self.card.iteration == 0:
                leadingLimits = None
                xLimitsDefined = (plot.xaxis.limits is not None) and \
                    (not isinstance(plot.xaxis.limits, str))
                yLimitsDefined = (plot.yaxis.limits is not None) and \
                    (not isinstance(plot.yaxis.limits, str))
                if xLimitsDefined and (not yLimitsDefined):
                    leadingLimits = 'x'
                elif yLimitsDefined and (not xLimitsDefined):
                    leadingLimits = 'y'
                xmin, xmax = self.update_limits(plot.xaxis, x)
                ymin, ymax = self.update_limits(plot.yaxis, y)
                emin, emax = self.update_limits(plot.caxis, cData)
                if plot.aspect == 'equal' or isinstance(plot.aspect,
                                                        (int, float)):
                    xyeq = self.equalize_xy(plot, leadingLimits)
                    if xyeq is not None:
                        xmin, xmax, ymin, ymax = xyeq

            limits = plot.caxis.limits
            cData01 = ((cData - limits[0]) * plot.colorFactor /
                       (limits[1] - limits[0])).reshape(-1, 1)
            cData01[cData01 < 0] = 0.
            cData01[cData01 > 1] = 1.
            if plot.invertColorMap:
                cData01 -= 0.5
                cData01[cData01 < 0] += 1

            if intensity.dtype == np.complex128:
                flux = intensity.real**2 + intensity.imag**2
            else:
                flux = intensity
            cDataHSV = np.dstack(
                (cData01, np.ones_like(cData01) * plot.colorSaturation,
                 flux.reshape(-1, 1)))
            cDataRGB = (mpl.colors.hsv_to_rgb(cDataHSV)).reshape(-1, 3)
# 1D x, y and cData histograms
            xh, xhRGB, xbe = self.do_hist1d(x, flux, cDataRGB, plot.xaxis)
            yh, yhRGB, ybe = self.do_hist1d(y, flux, cDataRGB, plot.yaxis)
            if plot.ePos:
                eh, ehRGB, ebe = self.do_hist1d(
                    cData, flux, cDataRGB, plot.caxis)
            else:
                eh, ehRGB, ebe = None, None, None
# 2D histogram
            res = self.do_hist2d(x, y, intensity, cDataRGB, plot)
            xyh, xyhRGB = res[0], res[1]
            is4d = (plot.fluxKind.lower().endswith('4d') or
                    plot.fluxKind.lower().endswith('pca'))
            xyh4 = res[2] if is4d else None

            if plot.fluxKind.endswith('log'):
                xh = np.log10(xh)
                xh[np.where(np.isnan(xh))] = 0
                xhRGB = np.log10(xhRGB)
                xhRGB[np.where(np.isnan(xhRGB))] = 0
                yh = np.log10(yh)
                yh[np.where(np.isnan(yh))] = 0
                yhRGB = np.log10(yhRGB)
                yhRGB[np.where(np.isnan(yhRGB))] = 0
                if plot.ePos:
                    eh = np.log10(eh)
                    eh[np.where(np.isnan(eh))] = 0
                    ehRGB = np.log10(ehRGB)
                    ehRGB[np.where(np.isnan(ehRGB))] = 0
                xyh = np.log10(xyh)
                xyh[np.where(np.isnan(xyh))] = 0
                xyhRGB = np.log10(xyhRGB)
                xyhRGB[np.where(np.isnan(xyhRGB))] = 0

            outList = [xh, xhRGB, xbe, yh, yhRGB, ybe,
                       eh, ehRGB, ebe, xyh, xyhRGB, xyh4, sum(flux), locNrays]
            if self.card.backend.startswith('shadow'):
                outList.append(locNraysNeeded)
            elif self.card.backend.startswith('raycing'):
                outList.append((locAlive, locGood, locOut, locOver, locDead,
                                locAccepted, locAcceptedE, locSeeded,
                                locSeededI))
            outList.append(displayAsAbsorbedPower)
            if self.card.iteration == 0:  # needed for multiprocessing
                outList.append((xmin, xmax, ymin, ymax, emin, emax))
            queue.put(outList)


class BackendProcess(GenericProcessOrThread, Process):
    def __init__(self, locCard, plots, outPlotQueues, alarmQueue, idLoc):
        Process.__init__(self)
        GenericProcessOrThread.__init__(self, locCard, plots, outPlotQueues,
                                        alarmQueue, idLoc)


class BackendThread(GenericProcessOrThread, Thread):
    def __init__(self, locCard, plots, outPlotQueues, alarmQueue, idLoc):
        Thread.__init__(self)
        GenericProcessOrThread.__init__(self, locCard, plots, outPlotQueues,
                                        alarmQueue, idLoc)
