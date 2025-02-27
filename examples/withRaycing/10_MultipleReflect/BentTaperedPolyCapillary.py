# -*- coding: utf-8 -*-
"""
!!! select one of the two functions to run at the very bottom !!!
"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.screens as rsc

#mGold = rm.Material('Au', rho=19.3)
mGlass = rm.Material(('Si', 'O'), quantities=(1, 2), rho=2.2)


class BentCapillary(roe.OE):
    def __init__(self, *args, **kwargs):
        self.rSample = kwargs.pop('rSample')
        self.entranceAlpha = kwargs.pop('entranceAlpha')
        self.f = kwargs.pop('f')
        self.r0in = kwargs.pop('rIn')
        self.r0out = kwargs.pop('rOut')
        roe.OE.__init__(self, *args, **kwargs)

        s0 = self.f - self.rSample * np.cos(self.entranceAlpha)
        self.a0 = -np.tan(self.entranceAlpha) / 2 / s0
        self.b0 = self.rSample * np.sin(self.entranceAlpha) - self.a0 * s0**2
        self.s0 = s0
        self.ar = (self.r0out-self.r0in) / s0**2
        self.br = self.r0in
        self.isParametric = True

    def local_x0(self, s):  # axis of capillary, x(s)
        return self.a0 * s**2 + self.b0

    def local_x0Prime(self, s):
        return 2 * self.a0 * s

    def local_r0(self, s):  # radius of capillary (s)
        return self.ar * (s-self.s0)**2 + self.br

    def local_r0Prime(self, s):
        return self.ar * 2 * (s-self.s0)

    def local_r(self, s, phi):
        den = np.cos(np.arctan(self.local_x0Prime(s)))**2
        return self.local_r0(s) / (np.cos(phi)**2/den + np.sin(phi)**2)

    def local_n(self, s, phi):
        a = -np.sin(phi)
        b = -np.sin(phi)*self.local_x0Prime(s) - self.local_r0Prime(s)
        c = -np.cos(phi)
        norm = np.sqrt(a**2 + b**2 + c**2)
        return a/norm, b/norm, c/norm

    def xyz_to_param(self, x, y, z):
        """ *s*, *r*, *phi* are cynindrc-like coordinates of the capillary.
        *s* is along y in the reverse direction, starting at the exit,
        *r* is measured from the capillary axis x0(s)
        *phi* is the polar angle measured from the z (vertical) direction."""
        s = self.f - y
        phi = np.arctan2(x - self.local_x0(s), z)
        r = np.sqrt((x-self.local_x0(s))**2 + z**2)
        return s, phi, r

    def param_to_xyz(self, s, phi, r):
        x = self.local_x0(s) + r*np.sin(phi)
        y = self.f - s
        z = r * np.cos(phi)
        return x, y, z


E0 = 9000.
rSample = 100.
f = 500.
r0 = 0.1
wall = 0.02
layers = 12  # number of hexagonal layers
nRefl = 12
nReflDisp = 12
xzPrimeMax = 3.


def build_beamline(nrays=1000):
    beamLine = raycing.BeamLine(height=0)
    rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0), nrays=nrays,
        dx=0., dz=0., distxprime='annulus',
        distE='lines', energies=(E0,), polarization='horizontal')
    beamLine.fsm1 = rsc.Screen(beamLine, 'DiamondFSM1', (0, rSample, 0))
    beamLine.capillaries = []
    beamLine.firstInLayer = []
    beamLine.xzMax = 0
    for n in range(layers):
        if n > 0:
            ms = range(n)
            i6 = range(6)
        else:
            ms = 0,
            i6 = 0,
        beamLine.firstInLayer.append(len(beamLine.capillaries))
        for i in i6:
            for m in ms:
                x = 2 * (r0+wall) * (n**2 + m**2 - n*m)**0.5
                alpha = np.arcsin(x / rSample)
                roll1 = -np.arctan2(np.sqrt(3)*m, 2*n - m)
                roll = roll1 + i*np.pi/3.
                capillary = BentCapillary(
                    beamLine, 'BentCapillary', [0, 0, 0], roll=roll,
                    material=mGlass, limPhysY=[rSample*np.cos(alpha), f],
                    f=f, rSample=rSample, entranceAlpha=alpha, rIn=r0, rOut=r0)
                beamLine.capillaries.append(capillary)
                if beamLine.xzMax < capillary.b0:
                    beamLine.xzMax = capillary.b0
    print('max divergence =', alpha)
    beamLine.xzMax += 2 * r0
    print('{0} capillaries built'.format(len(beamLine.capillaries)))
    print('indices of first capillaries in each layer:', beamLine.firstInLayer)
    beamLine.sources[0].dxprime = 0, np.arcsin((2*n+1) * (r0+wall) / rSample)
#    beamLine.sources[0].dxprime = (np.arcsin((2*n-3) * (r0+wall) / rSample),
#        np.arcsin((2*n+1) * (r0+wall) / rSample))
#    beamLine.sources[0].dxprime = 0, np.arcsin(r0 / rSample)
    beamLine.fsm2 = rsc.Screen(beamLine, 'DiamondFSM2', (0, f, 0))
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
#    raycing.rotate_beam(
#        beamSource, yaw=-beamLine.capillaries[0].entranceAlpha)
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1}
    beamCapillaryGlobalTotal = None
    for i, capillary in enumerate(beamLine.capillaries):
        beamCapillaryGlobal, beamCapillaryLocalN =\
            capillary.multiple_reflect(beamSource, maxReflections=nRefl)
        beamCapillaryLocalN.phi /= np.pi
        if beamCapillaryGlobalTotal is None:
            beamCapillaryGlobalTotal = beamCapillaryGlobal
        else:
            good = ((beamCapillaryGlobal.state == 1) |
                    (beamCapillaryGlobal.state == 3))
            rs.copy_beam(beamCapillaryGlobalTotal, beamCapillaryGlobal,
                         good, includeState=True)
        outDict['beamCapillaryLocalN{0:02d}'.format(i)] = beamCapillaryLocalN
    outDict['beamCapillaryGlobalTotal'] = beamCapillaryGlobalTotal
    beamFSM2 = beamLine.fsm2.expose(beamCapillaryGlobalTotal)
    outDict['beamFSM2'] = beamFSM2
    return outDict
rr.run_process = run_process


def plot2D():
    beamLine = build_beamline()
    fig1 = plt.figure(1, figsize=(8, 6))
#    ax1 = plt.subplot(111, aspect='equal', label='1')
    ax1 = plt.subplot(111, aspect=50, label='1')
    ax1.set_title('Cross-section of polycapillary at $z$=0')
    ax1.set_xlabel(r'$y$ (mm)', fontsize=14)
    ax1.set_ylabel(r'$x$ (mm)', fontsize=14)
    seq = [2, 5, 12, 5]
    for i in beamLine.firstInLayer:
        capillary = beamLine.capillaries[i]
        s = np.linspace(0, capillary.s0, 200)
        x = capillary.local_x0(s)
        r = capillary.local_r0(s)
        ax1.plot([0, f-s[-1]], [0, x[-1]], 'k-', lw=0.5)
        line = ax1.plot(f-s, x, 'k-.', lw=0.5)
        line[0].set_dashes(seq)
        ax1.plot(f-s, x-r, 'r-', lw=2)
        ax1.plot(f-s, x+r, 'r-', lw=2)
    ax1.set_xlim(0, f)
    ax1.set_ylim(-2*capillary.r0in, capillary.local_x0(0) + 2*capillary.r0out)
    fig1.savefig('PolycapillaryZ0crosssection.png')
    plt.show()


def define_plots(beamLine):
    fwhmFormatStr3 = '%.3f'
    plots = []
#    PlotClass = xrtp.XYCPlotWithNumerOfReflections
    PlotClass = xrtp.XYCPlot

    for ibins, bins in enumerate([128, 256]):
        plot = xrtp.XYCPlot(
            'beamFSM1', (1, 3, -1),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=bins, ppb=2),
            yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=bins, ppb=2),
            caxis='category', beamState='beamFSM2', title='FSM1_Cat')
        plot.baseName = 'NCapillaries-a-FSM1Cat{0}'.format(ibins)
        plots.append(plot)

        toShow = [0, 5, 11]
        # for i in [beamLine.firstInLayer[ind] for ind in toShow]:
        for i in toShow:
            beamLocal = 'beamCapillaryLocalN{0:02d}'.format(i)

            plot = PlotClass(
                beamLocal, (1,), aspect='auto',
                xaxis=xrtp.XYCAxis(r'$\phi$', 'rad', bins=bins, ppb=2),
                yaxis=xrtp.XYCAxis(r'$s$', 'mm', bins=bins, ppb=2),
                caxis=xrtp.XYCAxis('number of reflections', '',
                                   bins=nReflDisp+1, ppb=16,
                                   data=raycing.get_reflection_number),
                ePos=1, title='N (s, phi)')
            plot.xaxis.fwhmFormatStr = '%.2f' + r'$ \pi$'
            plot.xaxis.limits = [-1, 1]  # limits are in units of pi!
            plot.yaxis.limits = [0, f]
            plot.caxis.fwhmFormatStr = None
            plot.caxis.limits = [-0.5, nReflDisp+0.5]
            formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
            plot.ax2dHist.xaxis.set_major_formatter(formatter)
            plot.baseName = 'NCapillaries-b-Local{0}N{1}'.format(i, ibins)
            plots.append(plot)

            plot = xrtp.XYCPlot(
                beamLocal, (1,), aspect='auto',
                xaxis=xrtp.XYCAxis(r'$\phi$', 'rad', bins=bins, ppb=2),
                yaxis=xrtp.XYCAxis(r'$s$', 'mm', bins=bins, ppb=2),
                caxis=xrtp.XYCAxis(r'incidence angle $\theta$', 'mrad'),
                title='theta (s, phi)')
            plot.xaxis.fwhmFormatStr = '%.2f' + r'$ \pi$'
            plot.xaxis.limits = [-1, 1]  # limits are in units of pi!
            plot.yaxis.limits = [0, f]
            plot.caxis.fwhmFormatStr = None
            plot.caxis.limits = [0, xzPrimeMax*0.5]
            formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
            plot.ax2dHist.xaxis.set_major_formatter(formatter)
            plot.baseName = 'NCapillaries-c-Local{0}Theta{1}'.format(i, ibins)
            plots.append(plot)

        plot = PlotClass(
            'beamFSM2', (1, 3),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=bins, ppb=2),
            yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=bins, ppb=2),
            caxis='category', title='FSM2_xzCat')
        plot.xaxis.fwhmFormatStr = fwhmFormatStr3
        plot.yaxis.fwhmFormatStr = fwhmFormatStr3
        plot.xaxis.limits = [-beamLine.xzMax, beamLine.xzMax]
        plot.yaxis.limits = [-beamLine.xzMax, beamLine.xzMax]
    #    plot.fluxFormatStr = '%.2e'
        plot.baseName = 'NCapillaries-d-FSM2-xzCat{0}'.format(ibins)
        plots.append(plot)

        plot = PlotClass(
            'beamFSM2', (1, 3),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=bins, ppb=2),
            yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=bins, ppb=2),
            caxis=xrtp.XYCAxis('number of reflections', '',  bins=nReflDisp+1,
                               ppb=16, data=raycing.get_reflection_number),
            title='FSM2_xzN')
        plot.xaxis.fwhmFormatStr = fwhmFormatStr3
        plot.yaxis.fwhmFormatStr = fwhmFormatStr3
        plot.xaxis.limits = [-beamLine.xzMax, beamLine.xzMax]
        plot.yaxis.limits = [-beamLine.xzMax, beamLine.xzMax]
        plot.caxis.fwhmFormatStr = None
        plot.caxis.limits = [-0.5, nReflDisp+0.5]
    #    plot.fluxFormatStr = '%.2e'
        plot.baseName = 'NCapillaries-e-FSM2-xzN{0}'.format(ibins)
        plots.append(plot)

        plot = PlotClass(
            'beamFSM2', (1, 3), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=bins, ppb=2),
            yaxis=xrtp.XYCAxis(r"$x'$", 'mrad', bins=bins, ppb=2),
            caxis=xrtp.XYCAxis('number of reflections', '', bins=nReflDisp+1,
                               ppb=16, data=raycing.get_reflection_number),
            title='FSM2_xxPrimeN')
        plot.xaxis.fwhmFormatStr = fwhmFormatStr3
        plot.yaxis.fwhmFormatStr = fwhmFormatStr3
        plot.xaxis.limits = [-beamLine.xzMax, beamLine.xzMax]
        plot.yaxis.limits = [-xzPrimeMax, xzPrimeMax]
        plot.caxis.fwhmFormatStr = None
        plot.caxis.limits = [-0.5, nReflDisp+0.5]
    #    plot.fluxFormatStr = '%.2e'
        plot.baseName = 'NCapillaries-f-FSM2-xPhaseSpaceN{0}'.format(ibins)
        plots.append(plot)

        plot = PlotClass(
            'beamCapillaryGlobalTotal', (1, 3), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=bins, ppb=2),
            yaxis=xrtp.XYCAxis(r"$x$", 'mm', bins=bins, ppb=2),
            caxis=xrtp.XYCAxis(r"$x'$", 'mrad', bins=bins, ppb=2),
            title='beamCapillaryGlobalTotal_yx')
        plot.xaxis.fwhmFormatStr = fwhmFormatStr3
        plot.yaxis.fwhmFormatStr = fwhmFormatStr3
        plot.yaxis.limits = 'symmetric'
        plot.caxis.fwhmFormatStr = None
        plot.caxis.limits = [-xzPrimeMax, xzPrimeMax]
    #    plot.fluxFormatStr = '%.2e'
        plot.baseName = 'NCapillaries-g-CapillaryOut-depthX{0}'.format(ibins)
        plots.append(plot)

        plot = PlotClass(
            'beamCapillaryGlobalTotal', (1, 3), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=bins, ppb=2),
            yaxis=xrtp.XYCAxis(r"$z$", 'mm', bins=bins, ppb=2),
            caxis=xrtp.XYCAxis(r"$z'$", 'mrad', bins=bins, ppb=2),
            title='beamCapillaryGlobalTotal_yz')
        plot.xaxis.fwhmFormatStr = fwhmFormatStr3
        plot.yaxis.fwhmFormatStr = fwhmFormatStr3
        plot.yaxis.limits = 'symmetric'
        plot.caxis.fwhmFormatStr = None
        plot.caxis.limits = [-xzPrimeMax, xzPrimeMax]
    #    plot.fluxFormatStr = '%.2e'
        plot.baseName = 'NCapillaries-h-CapillaryOut-depthZ{0}'.format(ibins)
        plots.append(plot)

    for plot in plots:
        plot.invertColorMap = True
        plot.negative = True
        plot.saveName = plot.baseName + '.png'
#        plot.persistentName = plot.baseName + '.pickle'
    return plots


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=1000*16, beamLine=beamLine,
                         processes='half')


# this is necessary to use multiprocessing in Windows, otherwise the new Python
# contexts cannot be initialized:
if __name__ == '__main__':
    main()
#    plot2D()
