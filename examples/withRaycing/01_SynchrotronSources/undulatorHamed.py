# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"

#import matplotlib
#matplotlib.use('agg')

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import pickle
import time
#import matplotlib
#matplotlib.use("Agg")
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False

suffix = ''
R0 = 25000
xPrimeMax = 0.6
zPrimeMax = 0.6
repeats = 100
#repeats = 1

#sheet, prefix = 'EPU_HP_mode', '1'
#sheet, prefix = 'EPU_VP_mode', '3'
sheet, prefix = 'QEPU_HP_mode', '2'
#sheet, prefix = 'QEPU_VP_mode', '4'

prefix += sheet

#prefix += '-1-band'
#prefix += '-2-1stHarmonic'
#prefix += '-3-mono1stHarmonic'
#prefix += '-4-2ndHarmonic'
#prefix += '-5-mono2ndHarmonic'
#prefix += '-6-3rdHarmonic'
prefix += '-7-mono3rdHarmonic'
#prefix += '-8-5thHarmonic'
#prefix += '-9-mono5thHarmonic'

fixedEnergy = False
filamentBeam = False
if 'VP' in prefix:
    eMinRays, eMaxRays = 3., 60.
else:
    eMinRays, eMaxRays = 3., 40.
if 'mono' in prefix:
    if '1st' in prefix:
        if 'VP' in prefix:
            if 'QEPU' in prefix:
                fixedEnergy = 11.3
            else:
                fixedEnergy = 10.65
        else:
            if 'QEPU' in prefix:
                fixedEnergy = 7.60
            else:
                fixedEnergy = 7.15
    elif '2nd' in prefix:
        if 'VP' in prefix:
            if 'QEPU' in prefix:
                fixedEnergy = 21.9
            else:
                fixedEnergy = 20.7
        else:
            if 'QEPU' in prefix:
                fixedEnergy = 14.65
            else:
                fixedEnergy = 13.9
    elif '3rd' in prefix:
        if 'VP' in prefix:
            if 'QEPU' in prefix:
                fixedEnergy = 34.0
            else:
                fixedEnergy = 32.0
        else:
            if 'QEPU' in prefix:
#                fixedEnergy = 20.65
                fixedEnergy = 20.5
            else:
                fixedEnergy = 21.45
    elif '5th' in prefix:
        if 'VP' in prefix:
            if 'QEPU' in prefix:
                fixedEnergy = 53
            else:
                fixedEnergy = 53.5
        else:
            if 'QEPU' in prefix:
                fixedEnergy = 35.85
            else:
                fixedEnergy = 35.75
    else:
        raise ValueError('unknown harmonic')
    prefix += '-E={0:.2f}eV'.format(fixedEnergy)
#    filamentBeam = True
elif 'Harm' in prefix:
    if '1st' in prefix:
        if 'VP' in prefix:
            if 'QEPU' in prefix:
                eMinRays, eMaxRays = 8.5, 12.
            else:
                eMinRays, eMaxRays = 8., 11.5
        else:
            if 'QEPU' in prefix:
                eMinRays, eMaxRays = 6.4, 7.9
            else:
                eMinRays, eMaxRays = 6., 7.5
    elif '2nd' in prefix:
        if 'VP' in prefix:
            if 'QEPU' in prefix:
                eMinRays, eMaxRays = 16., 23.
            else:
                eMinRays, eMaxRays = 16.5, 22.
        else:
            if 'QEPU' in prefix:
                eMinRays, eMaxRays = 12., 16.
            else:
                eMinRays, eMaxRays = 12., 14.5
    elif '3rd' in prefix:
        if 'VP' in prefix:
            if 'QEPU' in prefix:
                eMinRays, eMaxRays = 24., 34.5
            else:
                eMinRays, eMaxRays = 25., 33.
        else:
            if 'QEPU' in prefix:
                eMinRays, eMaxRays = 18., 22.
            else:
                eMinRays, eMaxRays = 18., 22.
    elif '5th' in prefix:
        if 'VP' in prefix:
            if 'QEPU' in prefix:
                eMinRays, eMaxRays = 43., 55.
            else:
                eMinRays, eMaxRays = 43., 55.
        else:
            if 'QEPU' in prefix:
                eMinRays, eMaxRays = 30.2, 36.2
            else:
                eMinRays, eMaxRays = 30., 36.
    else:
        raise ValueError('unknown harmonic')

if 'mono' in prefix:
    bins = 64  # Number of bins in the plot histogram
    ppb = 4  # Number of pixels per histogram bin
else:
    bins = 256  # Number of bins in the plot histogram
    ppb = 1  # Number of pixels per histogram bin

Source = rs.Undulator
kwargs = dict(
    eE=1.5, eI=0.5, eEspread=8e-4,
    eEpsilonX=6.0, eEpsilonZ=0.06, betaX=5.66, betaZ=2.85,
    period=84., n=36,
    xPrimeMax=xPrimeMax, zPrimeMax=zPrimeMax,
    xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
#    targetOpenCL='CPU',
    filamentBeam=filamentBeam)
xlimits = [-xPrimeMax*R0*1e-3, xPrimeMax*R0*1e-3]
zlimits = [-zPrimeMax*R0*1e-3, zPrimeMax*R0*1e-3]
kwargs['customField'] = ['B-Hamed.xlsx', dict(sheetname=sheet, skiprows=0)]
#kwargs['customField'] = 10.

if False:  # zero source size:
    kwargs['eEpsilonX'] = 0
    kwargs['eEpsilonZ'] = 0
    eEpsilonC = '0'
else:
    eEpsilonC = 'n'

eUnit = 'eV'
kwargs['eMin'] = eMinRays
kwargs['eMax'] = eMaxRays


def build_beamline():
    beamLine = raycing.BeamLine()
    beamLine.source = Source(beamLine, **kwargs)
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', (0, R0, 0))
    return beamLine


def run_process(beamLine):
    startTime = time.time()
    waveOnScreen = beamLine.fsm1.prepare_wave(
        beamLine.source, beamLine.fsmExpX, beamLine.fsmExpZ)
    beamSource = beamLine.source.shine(wave=waveOnScreen,
                                       fixedEnergy=fixedEnergy)
#    beamSource = beamLine.source.shine(fixedEnergy=fixedEnergy)
    print('shine time = {0}s'.format(time.time() - startTime))
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    outDict = {'beamSource': beamSource,
               'beamFSM1': beamFSM1}
    if showIn3D:
        beamLine.prepare_flow()
    return outDict

rr.run_process = run_process


def define_plots(beamLine):
    plots = []
    plotsE = []

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='total flux')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '1totalFlux' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='s', aspect='auto', title='horizontal polarization flux')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '2horizFlux' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='p', aspect='auto', title='vertical polarization flux')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '3vertFlux' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis,
        caxis=xrtp.XYCAxis('circular polarization rate', '',
                           data=raycing.get_circular_polarization_rate,
                           limits=[-1, 1], bins=bins, ppb=ppb),
        aspect='auto', title='circular polarization rate')
    plot.saveName = prefix + '4circPolRate' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)

    complexPlotsPCA = []
    if 'mono' in prefix:
        xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
        yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
        caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                             bins=bins, ppb=ppb)
        plot = xrtp.XYCPlot(
            'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis,
            fluxKind='EsPCA',
            aspect='auto', title='EsPCA')
        plot.saveName = prefix + '5EsPCA' + suffix + '.png'
        plot.caxis.fwhmFormatStr = None
        plots.append(plot)
        plotsE.append(plot)
        complexPlotsPCA.append(plot)

        xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
        yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
        caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                             bins=bins, ppb=ppb)
        plot = xrtp.XYCPlot(
            'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis,
            fluxKind='EpPCA',
            aspect='auto', title='EpPCA')
        plot.saveName = prefix + '6EpPCA' + suffix + '.png'
        plot.caxis.fwhmFormatStr = None
        plots.append(plot)
        plotsE.append(plot)
        complexPlotsPCA.append(plot)

    ax = plot.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    ax = plot.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

    for plot in plotsE:
        f = plot.caxis.factor
        plot.caxis.limits = eMinRays*f, eMaxRays*f
    for plot in plots:
        plot.xaxis.fwhmFormatStr = '%.2f'
        plot.yaxis.fwhmFormatStr = '%.2f'
        plot.fluxFormatStr = '%.2p'
    return plots, plotsE, complexPlotsPCA


def afterScript(plots, complexPlotsPCA, beamLine):
    if beamLine.source.trajectory is not None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        tr = beamLine.source.trajectory
        ax.plot(tr[2]/1e3, tr[0]*1e3, tr[1]*1e3)
        ax.set_xlabel(r"$z$, m")
        ax.set_ylabel(u"$x$, µm")
        ax.set_zlabel(u"$y$, µm")
        plt.title("trajectory 3D")
        plt.savefig(prefix + '0trajectory' + suffix + '.png')
        plt.show()

    if not complexPlotsPCA:
        return
    if not ('mono' in prefix):
        return

    import scipy.linalg as spl
    dump = []
    start = time.time()
    for complexPlotPCA in complexPlotsPCA:
        k = complexPlotPCA.size2D
        wPCA, vPCA, outPCA = None, None, None
        x = complexPlotPCA.xaxis.binCenters
        y = complexPlotPCA.yaxis.binCenters
        if repeats >= 4:
            pE = complexPlotPCA.total4D
            cEr = pE[:, :repeats]
            cE = np.dot(cEr.T.conjugate(), cEr)
            cE /= np.diag(cE).sum()
            kwargs = dict(eigvals=(repeats-4, repeats-1))
            wPCA, vPCA = spl.eigh(cE, **kwargs)
            print(wPCA)
            outPCA = np.zeros((k, repeats), dtype=np.complex128)
            for i in range(4):
                mPCA = np.outer(vPCA[:, -1-i], vPCA[:, -1-i].T.conjugate())
                outPCA[:, -1-i] = np.dot(cEr, mPCA)[:, 0]
            print("repeats={0}; PCA problem has taken {1} s".format(
                  repeats, time.time()-start))
            dump.append([repeats, x, y, wPCA, outPCA])

    pickleName = '{0}-{1}repeats.pickle'.format(prefix, repeats)
    with open(pickleName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)
    print("Done")


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.fsmExpX = np.linspace(xlimits[0], xlimits[1], bins+1)
        beamLine.fsmExpZ = np.linspace(zlimits[0], xlimits[1], bins+1)
        beamLine.glow()
    else:
        plots, plotsE, complexPlotsPCA = define_plots(beamLine)
        xrtr.run_ray_tracing(plots, repeats=repeats,
                             afterScript=afterScript, afterScriptArgs=[
                                 plots, complexPlotsPCA, beamLine],
                             beamLine=beamLine)


def plotPCA():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    cmap = cm.get_cmap('cubehelix')

    pickleName = '{0}-{1}repeats.pickle'.format(prefix, repeats)
    with open(pickleName, 'rb') as f:
        dump = pickle.load(f)

    for i, pol in enumerate(['s', 'p']):
        repeatsS, x, y, wPCA, outPCA = dump[i]
        extent = xlimits + zlimits
        if wPCA is not None:
            print(wPCA)
            norm = (abs(outPCA[:, -4:])**2).sum(axis=0)
            outPCA[:, -4:] /= norm**0.5
            figMs = plt.figure(figsize=(8, 8))
            figMs.suptitle('principal components of one-electron images\n' +
                           pol + '-polarization',
                           fontsize=14)
            p1, p2 = 0.1-0.02, 0.505-0.02
            rect2d = [p1, p2, 0.4, 0.4]
            ax0 = figMs.add_axes(rect2d, aspect=1)
            rect2d = [p2, p2, 0.4, 0.4]
            ax1 = figMs.add_axes(rect2d, aspect=1)
            rect2d = [p1, p1, 0.4, 0.4]
            ax2 = figMs.add_axes(rect2d, aspect=1)
            rect2d = [p2, p1, 0.4, 0.4]
            ax3 = figMs.add_axes(rect2d, aspect=1)
            for ax in [ax0, ax1, ax2, ax3]:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.tick_params(axis='x', colors='grey')
                ax.tick_params(axis='y', colors='grey')
            for ax in [ax0, ax1]:
                ax.xaxis.tick_top()
            for ax in [ax1, ax3]:
                ax.yaxis.tick_right()
            im = (outPCA[:, -1]).reshape(len(y), len(x))
            ax0.imshow(im.real**2 + im.imag**2, extent=extent, cmap=cmap)
            modeName = '{0}-component'.format(pol)
            plt.text(0, extent[-1]-2,
                     '0th (coherent) {0}: w={1:.3f}'.format(
                         modeName, wPCA[-1]),
                     transform=ax0.transData,
                     ha='center', va='top', color='w')
            im = (outPCA[:, -2]).reshape(len(y), len(x))
            ax1.imshow(im.real**2 + im.imag**2, extent=extent, cmap=cmap)
            plt.text(0, extent[-1]-2,
                     '1st residual {0}: w={1:.3f}'.format(modeName, wPCA[-2]),
                     transform=ax1.transData, ha='center', va='top', color='w')
            im = (outPCA[:, -3]).reshape(len(y), len(x))
            ax2.imshow(im.real**2 + im.imag**2, extent=extent, cmap=cmap)
            plt.text(0, extent[-1]-2,
                     '2nd residual {0}: w={1:.3f}'.format(modeName, wPCA[-3]),
                     transform=ax2.transData, ha='center', va='top', color='w')
            im = (outPCA[:, -4]).reshape(len(y), len(x))
            ax3.imshow(im.real**2 + im.imag**2, extent=extent, cmap=cmap)
            plt.text(0, extent[-1]-2,
                     '3rd residual {0}: w={1:.3f}'.format(modeName, wPCA[-4]),
                     transform=ax3.transData, ha='center', va='top', color='w')

            figMs.savefig('Components-{0}-{1}-{2}repeats.png'.format(
                pol, prefix, repeatsS))

    print("Done")
    plt.show()

if __name__ == '__main__':
    main()
#    plotPCA()
