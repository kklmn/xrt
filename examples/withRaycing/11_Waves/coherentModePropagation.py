# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "8 Dec 2021"
# import matplotlib as mpl
# mpl.use('agg')

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.waves as rw
import xrt.backends.raycing.modes as rm

# caseE = u'4keV'
caseE = u'7keV'
# caseE = u'1Å'

if caseE == u'4keV':
    E0, harmE0, dE0 = 4040., 1, 0.
elif caseE == u'7keV':
    E0, harmE0, dE0 = 7100., 3, 0.
elif caseE == u'1Å':
    E0, harmE0, dE0 = 12398., 7, 0.
else:
    raise ValueError("unknown energy case")

slitFEPos = 20000.
p, q = 40000., 40000.
pitch = 3.5e-3
sin2Pitch, cos2Pitch = np.sin(2*pitch), np.cos(2*pitch)

caxisUnit = 'eV'

BW = 3.6e-4
eMinRays, eMaxRays = E0*(1-BW/2.), E0*(1+BW/2.)

eEpsilonX = 310e-12  # mrad
eEpsilonZ = 5.5e-12  # mrad
betaX = 9.539
betaZ = 1.982
accMax = 0.04, 0.02  # mrad

kwargs = dict(
    name='CoSAXS U19',
    eE=3., eI=0.25, eEspread=8e-4,
    eEpsilonX=eEpsilonX*1e9, eEpsilonZ=eEpsilonZ*1e9,
    betaX=betaX, betaZ=betaZ,
    period=19.3, n=101, targetE=(E0+dE0, harmE0),
    filamentBeam=True,
    uniformRayDensity=True,  # not strictly necessary
    distE='BW',
    # targetOpenCL='GPU',
    # R0=slitFEPos,
    xPrimeMax=accMax[0], zPrimeMax=accMax[1],
    xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
    eMin=eMinRays, eMax=eMaxRays)

nElectrons = 500
nModes = 10
nsamples = 512*256

bins = 128
ppb = 1
focusLim = 64, 16  # xmax, ymax in µm


def build_beamline():
    bl = raycing.BeamLine()
    bl.source = rs.Undulator(bl, **kwargs)

    opening = [s*slitFEPos*1e-3 for s in
               (-accMax[0], accMax[0], -accMax[1], accMax[1])]
    bl.slitFE = ra.RectangularAperture(
        bl, 'FE slit', [0, slitFEPos, 0],
        ('left', 'right', 'bottom', 'top'), opening)

    bl.limPhysXmax = accMax[0]*1e-3*1.2 * p
    bl.limPhysZmax = accMax[1]*1e-3*1.2 * p
    bl.limPhysYmax = bl.limPhysZmax / np.sin(pitch)
    focusPos = [0, p + q*cos2Pitch, q*sin2Pitch]
    bl.oe = roe.EllipticalMirrorParam(
        bl, 'M1', center=[0, p, 0], pitch=pitch, p=p, q=q,
        limPhysX=[-bl.limPhysXmax, bl.limPhysXmax],
        limPhysY=[-bl.limPhysYmax, bl.limPhysYmax])

    bl.fsmF = rsc.Screen(bl, 'inFocus', focusPos, z=(0, -sin2Pitch, cos2Pitch))
    return bl


def run_process_wave(bl):
    bl.iBeam = bl.iBeam+1 if hasattr(bl, 'iBeam') else 0
    waveOElocal = bl.oe.prepare_wave(bl.slitFE, nsamples)
    waveScreenF = bl.fsmF.prepare_wave(bl.oe, bl.fsmFX, bl.fsmFZ)
    waveFElocal = bl.savedBeams[bl.iBeam]
    beamToOE = rw.diffract(waveFElocal, waveOElocal)
    beamOEglobal, beamOElocal = bl.oe.reflect(
        beamToOE, noIntersectionSearch=True)
    rw.diffract(beamOElocal, waveScreenF)
    outDict = {'beamFElocal': waveFElocal,
               'beamScreenF': waveScreenF}
    return outDict


def run_process_hybr(bl):
    bl.iBeam = bl.iBeam+1 if hasattr(bl, 'iBeam') else 0
    beamSource = bl.savedBeams[bl.iBeam]
    waveScreenF = bl.fsmF.prepare_wave(bl.oe, bl.fsmFX, bl.fsmFZ)
    beamFElocal = bl.slitFE.propagate(beamSource)
    beamOEglobal, beamOElocal = bl.oe.reflect(beamSource)
    rw.diffract(beamOElocal, waveScreenF)
    outDict = {'beamFElocal': beamFElocal,
               'beamScreenF': waveScreenF}
    return outDict


def run_process_rays(bl):
    bl.iBeam = bl.iBeam+1 if hasattr(bl, 'iBeam') else 0
    beamSource = bl.savedBeams[bl.iBeam]
    beamFElocal = bl.slitFE.propagate(beamSource)
    beamOEglobal, beamOElocal = bl.oe.reflect(beamSource)
    beamScreenF = bl.fsmF.expose(beamOEglobal)
    outDict = {'beamFElocal': beamFElocal,
               'beamScreenF': beamScreenF}
    return outDict


def run_process_view(bl):
    beamSource = bl.sources[0].shine(fixedEnergy=E0)
    beamFElocal = bl.slitFE.propagate(beamSource)
    beamOEglobal, beamOElocal = bl.oe.reflect(beamSource)
    beamScreenF = bl.fsmF.expose(beamOEglobal)
    outDict = {'beamSource': beamSource,
               'beamFElocal': beamFElocal,
               'beamScreenF': beamScreenF}
    bl.beams = outDict
    bl.prepare_flow()
    return outDict


def add_plot(plots, plot, what, basename):
    plots.append(plot)
    plot.baseName = '{0}-{1}-{2}-E={3:05.0f}'.format(
        what, basename, plot.title, E0)
    plot.saveName = [plot.baseName + '.png', ]


class MyXYCPlot(xrtp.XYCPlot):
    def update_user_elements(self):
        if not hasattr(self, 'iMode'):
            self.iMode = 0
        else:
            self.iMode += 1
        N, what = self.textPanelParams
        if what.endswith('modes'):
            if self.iMode == 0:
                self.textPanel.set_text('mode 0 of {0}'.format(N))
            else:
                self.textPanel.set_text('modes 0 to {0} of {1}'.format(
                    self.iMode, N))
            self.save(suffix='-{0}'.format(self.iMode))
        else:
            if self.iMode == 0:
                self.textPanel.set_text('beam 1 of {0}'.format(N))
            else:
                self.textPanel.set_text('beams 1 to {0} of {1}'.format(
                    self.iMode+1, N))
            self.save(suffix='-{0}'.format(self.iMode+1))


def define_plots(bl, what, nEs, basename):
    plots = []

    ratio1 = int(accMax[0] / accMax[1])
    plot = MyXYCPlot(
        'beamFElocal', (1,),
        xaxis=xrtp.XYCAxis('x', 'mm', bins=bins*ratio1, ppb=ppb),
        yaxis=xrtp.XYCAxis('z', 'mm', bins=bins, ppb=ppb),
        caxis=xrtp.XYCAxis('energy', caxisUnit, bins=bins, ppb=ppb))
    plot.xaxis.limits = bl.slitFE.opening[:2]
    plot.yaxis.limits = bl.slitFE.opening[2:]
    plot.xaxis.fwhmFormatStr = '%.2f'
    plot.yaxis.fwhmFormatStr = '%.2f'
    add_plot(plots, plot, what, basename)
    plot.textPanel = plot.fig.text(1.02, 0.4, '',
                                   transform=plot.ax1dHistX.transAxes, size=10,
                                   color='r', ha='left')
    plot.textPanelParams = nEs, what

    ratio2 = int(focusLim[0] / focusLim[1])
    plot = MyXYCPlot(
        'beamScreenF', (1,),
        xaxis=xrtp.XYCAxis('x', u'µm', bins=bins*ratio2, ppb=ppb),
        yaxis=xrtp.XYCAxis('z', u'µm', bins=bins, ppb=ppb),
        caxis=xrtp.XYCAxis('energy', caxisUnit, bins=bins, ppb=ppb),
        fluxKind='EsPCA'
        )
    plot.xaxis.limits = [-focusLim[0], focusLim[0]]
    plot.yaxis.limits = [-focusLim[1], focusLim[1]]
    plot.xaxis.fwhmFormatStr = '%.1f'
    plot.yaxis.fwhmFormatStr = '%.1f'
    add_plot(plots, plot, what, basename)
    ax = plot.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    bl.fsmFX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    ax = plot.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    bl.fsmFZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    print('{0} points on the screen Exp'.format(len(bl.fsmFX)*len(bl.fsmFZ)))
    plot.textPanel = plot.fig.text(1.02, 0.4, '',
                                   transform=plot.ax1dHistX.transAxes, size=10,
                                   color='r', ha='left')
    plot.textPanelParams = nEs, what

    caxisFactor = 1
    for plot in plots:
        if plot.caxis.label.startswith('energy'):
            plot.caxis.limits = eMinRays*caxisFactor, eMaxRays*caxisFactor
            plot.caxis.offset = E0
            # plot.caxis.fwhmFormatStr = '%.2f'
            plot.caxis.fwhmFormatStr = None
            plot.fluxFormatStr = '%.1p'

    return plots


def show_bl(basename):
    bl = build_beamline()
    rr.run_process = run_process_view
    bl.glow(centerAt='M1')
    # bl.glow(scale=[5e3, 10, 5e3], centerAt='xtal1')


def make_modes(basename):
    bl = build_beamline()
    limitsOrigin = [-focusLim[0]*1e-3, focusLim[0]*1e-3,
                    -focusLim[1]*1e-3, focusLim[1]*1e-3]
    rm.make_and_save_modes(
        bl, nsamples, nElectrons, nModes, nModes, E0, output='all',
        basename=basename, limitsOrigin=limitsOrigin)


def use_modes(basename, what):
    def get_flux(beam):
        res = beam.Jss.sum() + beam.Jpp.sum()
        print(res)
        return res

    bl = build_beamline()
    bl.savedBeams, wAll, totalFlux = rm.use_saved(what, basename)
    if what.endswith('fields'):
        bl.savedBeams.sort(key=get_flux)
    plots = define_plots(bl, what, len(wAll), basename)
    if what.startswith('wave'):
        rr.run_process = run_process_wave
    elif what.startswith('hybr'):
        rr.run_process = run_process_hybr
    elif what.startswith('rays'):
        rr.run_process = run_process_rays
    else:
        raise ValueError('unknown mode of propagation')
    xrtr.run_ray_tracing(plots, repeats=nModes, beamLine=bl)


def main():
    step = 1  # 0 to 2

    basename = 'atFE-5000'
    # basename = 'atFE'
    if step == 0:
        show_bl(basename)
    elif step == 1:
        make_modes(basename)
    elif step == 2:
        # what = 'wave-fields'
        what = 'wave-modes'
        # what = 'hybr-fields'
        # what = 'hybr-modes'
        # what = 'rays-fields'
        # what = 'rays-modes'
        use_modes(basename, what)


if __name__ == '__main__':
    main()
