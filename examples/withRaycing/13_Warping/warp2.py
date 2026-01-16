# -*- coding: utf-8 -*-
r"""
.. !!! select one of the two functions to run at the very bottom !!!
.. !!! select 'rays' or 'wave' below !!!
.. !!! select a desired prefix below !!!

.. _warping:

Defocusing by a distorted mirror
--------------------------------

The images below are produced by
``\examples\withRaycing\13_Warping\warp.py``.

This example has two objectives:

1. to demonstrate how one can add a functional or measured figure error to an
   ideal optical element and
2. to study the influence of various figure errors onto image non-uniformity in
   focused and defocused cases. The study will be done in ray tracing and wave
   propagation, the latter being calculated in partial coherence with the
   actual emittance of the MAX IV 3 GeV ring.

Here, a toroidal mirror focuses an undulator source in 1:1 magnification. The
sagittal radius of the torus was determined for p = q = 25 m and pitch = 4
mrad. Defocusing in horizontal is done by going to a smaller pitch angle, here
2.2 mrad, and in vertical by unbending the meridional figure.

Three distorted surfaces are of Gaussian, waviness and as measured shapes, see
below. They were normalized such that the meridional slope error be 1 µrad rms.
The surfaces are determined on a 2D mesh. Interpolation splines for the height
and the normal vector are found at the time of mirror instantiation and used in
two special methods: ``local_z_distorted`` and ``local_n_distorted``, see
Section :ref:`distorted`. If the distorted shape is known analytically, as for
waviness, the two methods may directly invoke the corresponding functions
without interpolation. The scattered circles in the figures are random samples
where the height is calculated by interpolation (cf. the color (height) of the
circles with the color of the surface) together with the interpolated normals
(white arrows as projected onto the xy plane).

+------------------------+------------------------+------------------------+
|        Gaussian        |        waviness        |  mock NOM measurement  |
+========================+========================+========================+
|        |warp_G|        |        |warp_w|        |          |warp_N|      |
+------------------------+------------------------+------------------------+

.. |warp_G| imagezoom:: _images/warp_surf_gaussian.*
.. |warp_w| imagezoom:: _images/warp_surf_waviness.*
.. |warp_N| imagezoom:: _images/warp_surf_mock_NOM.*
   :loc: upper-right-corner

Defocused images reveal horizontal stripes seen both by ray tracing and wave
propagation. Notice that wave propagation 'sees' less distortion in the best
focusing case.

+----------+--------------------------+--------------------------+
|          |       ray tracing        |     wave propagation     |
+==========+==========================+==========================+
| ideal    |        |warp_rt0|        |        |warp_wp0|        |
+----------+--------------------------+--------------------------+
| Gaussian |        |warp_rtG|        |        |warp_wpG|        |
+----------+--------------------------+--------------------------+
| waviness |        |warp_rtw|        |        |warp_wpw|        |
+----------+--------------------------+--------------------------+
| mock NOM |        |warp_rtN|        |        |warp_wpN|        |
+----------+--------------------------+--------------------------+

.. |warp_rt0| animation:: _images/warp_rt0
.. |warp_rtG| animation:: _images/warp_rtG
.. |warp_rtw| animation:: _images/warp_rtw
.. |warp_rtN| animation:: _images/warp_rtN
.. |warp_wp0| animation:: _images/warp_wp0
   :loc: upper-right-corner
.. |warp_wpG| animation:: _images/warp_wpG
   :loc: upper-right-corner
.. |warp_wpw| animation:: _images/warp_wpw
   :loc: upper-right-corner
.. |warp_wpN| animation:: _images/warp_wpN
   :loc: lower-right-corner

"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.figure_error as rfe
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.waves as rw

import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False

source_dX = 25e-3
source_dZ = 5e-3
p = 25000.
q = 25000.
pitch = 2.2e-3
pitchr = 4e-3  # for which the sagittal curvature is calculated
rdefocus = 2 * p*q/(p+q) * np.sin(pitchr)
Rnom = 2 * p*q/(p+q) / np.sin(pitch)
if showIn3D:
    Rs = []
else:
    Rs = [Rnom*(1.5)**iRf for iRf in range(4)]
Rs.append(1e20)

E0 = 12398  # 7th
dE = 2.
K = 1.79  # gap=4.612mm
K *= 1.002

eEpsilonX = 0.263e-9
eEpsilonZ = 0.008e-9
betaX = 9.539
betaZ = 1.982

what = 'rays'
# what = 'wave'

if what == 'rays':
    prefix = 'rays-'
    emittanceFactor = 1
    nrays = 1e5
    repeats = 10
elif what == 'wave':
    is0emittance = True
    nrays = 1e6
    if is0emittance:
        prefix = 'wave-0emit-'
        emittanceFactor = 0
        repeats = 1
    else:
        emittanceFactor = 1
        prefix = 'wave-non0e-'
        repeats = 100

# prefix += 'gaussian-'
# prefix += 'waviness-'
prefix += 'NOM-'


def gaussian_bump():
    xmax, ymax = 5., 125.
    nX, nY = 101, 201
    x = np.linspace(-xmax, xmax, nX)
    y = np.linspace(-ymax, ymax, nY)
    z = 2.32e-4 * np.exp(-x[:, np.newaxis]**2/20**2 - y**2/150**2)
#    z += ((y > 10) & (x[:, np.newaxis] > 5))*0.002
    return x, y, z, 'gaussian bump'


def waviness():
    xmax, ymax = 5., 125.
    xWaveLength, yWaveLength = 20., 100.
    nX, nY = 101, 201
    x = np.linspace(-xmax, xmax, nX)
    y = np.linspace(-ymax, ymax, nY)
    z = 1.61e-5 * np.cos(x[:, np.newaxis]/xWaveLength*2*np.pi) *\
        np.cos(y/yWaveLength*2*np.pi)
    return x, y, z, 'waviness'


def read_NOM():
    xL, yL, zL = np.loadtxt('mock_surface.dat', unpack=True)
    nX = (yL == yL[0]).sum()
    nY = (xL == xL[0]).sum()
    x = xL[:nX]
    y = yL[::nX]
    z = zL.reshape((nY, nX))
    # z *= 1e-6 / 1.12
    z *= 1e-6
    x -= (np.min(x) + np.max(x)) / 2
    y -= (np.min(y) + np.max(y)) / 2
    # x and y are swapped to match the measurements' axes:
    return y, x, z, 'NOM surface'


class ToroidMirrorDistorted(roe.ToroidMirror):
    def __init__(self, *args, **kwargs):
        distorted_surface = kwargs.pop('distorted_surface')
        self.distorted_surface = distorted_surface
        roe.ToroidMirror.__init__(self, *args, **kwargs)
### here you specify the bump and its mesh ###
        self.warpX, self.warpY, self.warpZ, self.distortedSurfaceName =\
            distorted_surface()
        # print('xyz sizes:')
        # print(self.warpX.min(), self.warpX.max(), self.warpX.shape)
        # print(self.warpY.min(), self.warpY.max(), self.warpY.shape)
        # print(self.warpZ.min(), self.warpZ.max(), self.warpZ.shape)
        self.warpNX, self.warpNY = len(self.warpX), len(self.warpY)
        self.limPhysX = np.min(self.warpX), np.max(self.warpX)
        self.limPhysY = np.min(self.warpY), np.max(self.warpY)
        self.get_surface_limits()
        self.warpA, self.warpB = np.gradient(self.warpZ, self.warpX, self.warpY)
        self.warpA = np.arctan(self.warpA)
        self.warpB = np.arctan(self.warpB)

    def local_z_distorted(self, x, y):
        coords = np.array(
            [(x-self.limPhysX[0]) /
             (self.limPhysX[1]-self.limPhysX[0]) * (self.warpNX-1),
             (y-self.limPhysY[0]) /
             (self.limPhysY[1]-self.limPhysY[0]) * (self.warpNY-1)])
        # coords.shape = (2, self.nrays)
        z = ndimage.map_coordinates(self.warpZ, coords, order=1)
        return z

    def local_n_distorted(self, x, y):
        coords = np.array(
            [(x-self.limPhysX[0]) /
             (self.limPhysX[1]-self.limPhysX[0]) * (self.warpNX-1),
             (y-self.limPhysY[0]) /
             (self.limPhysY[1]-self.limPhysY[0]) * (self.warpNY-1)])
        a = ndimage.map_coordinates(self.warpA, coords, order=1)
        b = ndimage.map_coordinates(self.warpB, coords, order=1)
        return b, -a


def see_the_bump_old():
    if 'gaussian' in prefix:
        distorted_surface = gaussian_bump
    elif 'waviness' in prefix:
        distorted_surface = waviness
    elif 'NOM' in prefix:
        distorted_surface = read_NOM
    else:
        raise ValueError('unknown selector')

    beamLine = raycing.BeamLine()
    oe = ToroidMirrorDistorted(
        beamLine, 'warped', distorted_surface=distorted_surface)
    xi = oe.warpX
    yi = oe.warpY
    zi = oe.warpZ * 1e6
    print(xi.shape, yi.shape, zi.shape, zi.max()-zi.min())
    print(xi.min(), xi.max(), xi.shape)
    print(yi.min(), yi.max(), yi.shape)
    print(zi.min(), zi.max(), zi.shape)
    rmsA = ((oe.warpA**2).sum() / (oe.warpNX*oe.warpNY))**0.5
    rmsB = ((oe.warpB**2).sum() / (oe.warpNX*oe.warpNY))**0.5

    fig = plt.figure(figsize=(6, 8))
    fig.suptitle('{0}\n'.format(oe.distortedSurfaceName) +
                 u'rms slope errors:\ndz/dx = {0:.2f} µrad, '
                 u'dz/dy = {1:.2f} µrad'.format(rmsA*1e6, rmsB*1e6),
                 fontsize=14)
    rect_2D = [0.15, 0.08, 0.75, 0.8]
    ax = plt.axes(rect_2D)
    ax.contour(xi, yi, zi.T, levels=15, linewidths=0.5, colors='k')
    c = ax.contourf(xi, yi, zi.T, levels=15, cmap=plt.cm.jet)
    cbar = fig.colorbar(c)  # draw colorbar
    cbar.ax.set_ylabel(u'z (nm)')

    nsamples = 1000
    xmin, xmax = oe.limPhysX
    ymin, ymax = oe.limPhysY
    x = np.random.uniform(xmin, xmax, size=nsamples)
    y = np.random.uniform(ymin, ymax, size=nsamples)
    z = oe.local_z_distorted(x, y) * 1e6
    b, a = oe.local_n_distorted(x, y)
    ax.set_xlabel(u'x (mm)')
    ax.set_ylabel(u'y (mm)')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.scatter(x, y, c=z, marker='o', s=50, cmap=plt.cm.jet)
    ax.quiver(x, y, a, -b, edgecolor='w', color='w', headaxislength=7,
              headwidth=5, scale=6e-5, lw=1.)

    fig.savefig('surf_{0}_old.png'.format(oe.distortedSurfaceName))
    plt.show()


def see_the_bump_new():
    if 'gaussian' in prefix:
        bump = rfe.GaussianBump(bumpHeight=232.0, sigmaX=20., sigmaY=150.,
                                limPhysX=[-5, 5], limPhysY=[-125, 125])
    elif 'waviness' in prefix:
        bump = rfe.Waviness(amplitude=16.1, xWaveLength=20., yWaveLength=100.,
                            limPhysX=[-5, 5], limPhysY=[-125, 125])
    elif 'NOM' in prefix:
        bump = rfe.FigureErrorImported(
            'mock_surface.dat', orientation="YXZ", recenter=True)
    else:
        raise ValueError('unknown selector')

    beamLine = raycing.BeamLine()
    xi = bump.x1d
    yi = bump.y1d
    zi = bump.z2d
    print(xi.shape, yi.shape, zi.shape, zi.max()-zi.min())
    print(xi.min(), xi.max(), xi.shape)
    print(yi.min(), yi.max(), yi.shape)
    print(zi.min(), zi.max(), zi.shape)
    oe = roe.ToroidMirror(beamLine, 'warped', figureError=bump)
    oe.limPhysX = np.min(xi), np.max(xi)
    oe.limPhysY = np.min(yi), np.max(yi)
    oe.get_surface_limits()
    rmsy = ((bump.a2d**2).sum() / (bump.nx*bump.ny))**0.5
    rmsx = ((bump.b2d**2).sum() / (bump.nx*bump.ny))**0.5

    fig = plt.figure(figsize=(6, 8))
    fig.suptitle('{0}\n'.format(bump.name) +
                 u'rms slope errors:\ndz/dx = {0:.2f} µrad, '
                 u'dz/dy = {1:.2f} µrad'.format(rmsx*1e6, rmsy*1e6),
                 fontsize=14)
    rect_2D = [0.15, 0.08, 0.75, 0.8]
    ax = plt.axes(rect_2D)
    ax.contour(xi, yi, zi, levels=15, linewidths=0.5, colors='k')
    c = ax.contourf(xi, yi, zi, levels=15, cmap=plt.cm.jet)
    cbar = fig.colorbar(c)  # draw colorbar
    cbar.ax.set_ylabel(u'z (nm)')

    nsamples = 1000
    xmin, xmax = oe.limPhysX
    ymin, ymax = oe.limPhysY
    x = np.random.uniform(xmin, xmax, size=nsamples)
    y = np.random.uniform(ymin, ymax, size=nsamples)
    z = oe.local_z_distorted(x, y) * 1e6
    b, a = oe.local_n_distorted(x, y)
    ax.set_xlabel(u'x (mm)')
    ax.set_ylabel(u'y (mm)')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.scatter(x, y, c=z, marker='o', s=50, cmap=plt.cm.jet)
    ax.quiver(x, y, a, -b, edgecolor='w', color='w', headaxislength=7,
              headwidth=5, scale=6e-5, lw=1.)

    fig.savefig('surf_{0}_new.png'.format(bump.name))
    plt.show()


def build_beamline():
    beamLine = raycing.BeamLine()
    beamLine.oe = ToroidMirrorDistorted(
        beamLine, 'warped', center=[0, p, 0], pitch=pitch, R=Rnom, r=rdefocus)
    dx = beamLine.oe.limPhysX[1] - beamLine.oe.limPhysX[0]
    dy = beamLine.oe.limPhysY[1] - beamLine.oe.limPhysY[0]
#    beamLine.source = rs.GeometricSource(
#        beamLine, 'CollimatedSource', nrays=nrays, dx=source_dX, dz=source_dZ,
#        dxprime=dx/p/2, dzprime=dy/p*np.sin(pitch)/2)
    kwargs = dict(
        eE=3., eI=0.5, eEspread=0,
        eEpsilonX=eEpsilonX*1e9*emittanceFactor,
        eEpsilonZ=eEpsilonZ*1e9*emittanceFactor,
        betaX=betaX, betaZ=betaZ,
        period=18.5, n=108, K=K,
        filamentBeam=(what != 'rays'),
        targetOpenCL='CPU',
        xPrimeMax=dx/2/p*1e3, zPrimeMax=dy/2/p*np.sin(pitch)*1e3,
        xPrimeMaxAutoReduce=False,
        zPrimeMaxAutoReduce=False,
        eMin=E0-dE/2, eMax=E0+dE/2)
    beamLine.source = rs.Undulator(beamLine, nrays=nrays, **kwargs)
    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM0', [0, p+q, 0])
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', [0, p+q, q*np.tan(2*pitch)])
    return beamLine


def run_process_rays(beamLine):
    beamSource = beamLine.source.shine(fixedEnergy=E0)
    beamFSMsource = beamLine.fsm0.expose(beamSource)
    outDict = {'beamFSMsource': beamFSMsource}
    for iR, R in enumerate(Rs):
        beamLine.oe.R = R
        oeGlobal, oeLocal = beamLine.oe.reflect(beamSource)
        beamFSMrefl = beamLine.fsm1.expose(oeGlobal)
        outDict['beamFSMrefl{0:02d}'.format(iR)] = beamFSMrefl
    outDict['oeLocal'] = oeLocal
    if showIn3D:
        beamLine.prepare_flow()
    return outDict


def run_process_wave(beamLine):
#    waveOnOE = beamLine.oe.prepare_wave(beamLine.source, nrays)
#    beamSource = beamLine.source.shine(wave=waveOnOE, fixedEnergy=E0)
    beamLine.source.uniformRayDensity = True
    beamSource = beamLine.source.shine(fixedEnergy=E0)
    beamFSMsource = beamLine.fsm0.expose(beamSource)
    outDict = {'beamFSMsource': beamFSMsource}
    for iR, R in enumerate(Rs):
        beamLine.oe.R = R
        oeGlobal, oeLocal = beamLine.oe.reflect(beamSource)
        waveOnSample = beamLine.fsm1.prepare_wave(
            beamLine.oe, beamLine.fsmExpX, beamLine.fsmExpZ)
        rw.diffract(oeLocal, waveOnSample)
        outDict['beamFSMrefl{0:02d}'.format(iR)] = waveOnSample
    outDict['oeLocal'] = oeLocal
    return outDict

if what == 'rays':
    rr.run_process = run_process_rays
elif what == 'wave':
    rr.run_process = run_process_wave
else:
    raise ValueError('wrong job selector')


def define_plots(beamLine):
    plots = []

    xlimits = [-1, 1]
    plotSource = xrtp.XYCPlot(
        'beamFSMsource', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=xlimits),
        ePos=0, title='0-source')
    plots.append(plotSource)

    plotLocal = xrtp.XYCPlot(
        'oeLocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=beamLine.oe.limPhysY),
        caxis=xrtp.XYCAxis(r"theta", 'mrad'),
        ePos=1, title='1-oeLocal')
    plots.append(plotLocal)

    xlimits = [-500, 500]
    for iR, R in enumerate(Rs):
        plotRefl = xrtp.XYCPlot(
            'beamFSMrefl{0:02d}'.format(iR), (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', u'µm', limits=xlimits),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm', limits=xlimits),
            ePos=0, title='2-refl')
        Rstr = '{0:.1f}km'.format(R*1e-6) if R < 1e19 else 'inf'
        plotRefl.saveName = '{0}-{1}-R={2}.png'.format(prefix, iR, Rstr)
        Rstr = '{0:.1f} km'.format(R*1e-6) if R < 1e19 else r'$\infty$'
        plotRefl.textPanel = plotRefl.ax2dHist.text(
            0.01, 0.01, 'R = {0}'.format(Rstr),
            transform=plotRefl.ax2dHist.transAxes, size=12,
            color='w', ha='left', va='bottom')
        plots.append(plotRefl)
    ax = plotRefl.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    ax = plotRefl.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

    for plot in plots:
        plot.fluxFormatStr = '%.1p'
    return plots


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow(scale=[500, 10, 500], centerAt='warped')
        return
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=repeats, beamLine=beamLine)


if __name__ == '__main__':
    np.random.seed(1)
    # see_the_bump_old()  # with the ad hoc ToroidMirrorDistorted class
    see_the_bump_new()  # with the raycing.figure_error module
    # main()
