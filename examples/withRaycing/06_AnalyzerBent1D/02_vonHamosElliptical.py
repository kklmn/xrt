# -*- coding: utf-8 -*-
r"""
This example is a complete study of a crystal analyzer. The generator
``plot_generator()`` is rather complex and therefore needs some explanation.
The main loop changes the source type. After a flat-energy source has been
ray-traced (after ``yield``), the widths of *z* and energy distributions are
saved. Then a single line source is ray-traced and provides the width of *z*
distribution. From these 3 numbers we calculate energy resolution and, as a
check, ray-trace a third source with 7 energy lines with a spacing equal to the
previously calculated energy resolution. The source sizes, axis limits, number
of iterations etc. were determined experimentally and are given by lists in the
upper part of the script. The outputs are the plots and a text file with the
resulted energy resolutions."""
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr


crystalMaterial = 'Si'
if crystalMaterial == 'Si':
    d111 = 3.1354161
elif crystalMaterial == 'Ge':
    d111 = 3.2662725
else:
    raise

crystal = rm.CrystalDiamond((4, 4, 4), d111/4, elements=crystalMaterial)
#numiter = 16000
numiter = 60

Rm = 1e9  # meridional radius, mm
#Rs = 1000  # tmp sagittal radius, mm
Rs = 250  # tmp sagittal radius, mm
dphi = 0

beamV = 0.1/2.35  # vertical beam size
beamH = 0.1/2.35  # horizontal beam size

yAxesLim = 20

dxCrystal = 100.
dyCrystal = 100.
#dyCrystal = 50.
#dxCrystal = 300.
#dyCrystal = 70.

yAxisLim = 32  # Mythen length = 64 mm
yAxis1Line = -1.0, 0.2
yAxis7Lines = -1.0, 1.0
isDiced = True
isElliptical = True
elongation = 1.5

thetaDegree = 60

if thetaDegree == 40:
    if Rs > 800:
        eAxisFlat = 7.5e-3  # @ 40 deg, R=1000
    else:
        eAxisFlat = 3e-2  # @ 40 deg
elif thetaDegree == 60:
    if Rs > 800:
        eAxisFlat = 6.8e-3  # @ 60 deg, R=1000
    else:
        eAxisFlat = 2.0e-2  # @ 60 deg
elif thetaDegree == 80:
    if Rs > 800:
        eAxisFlat = 2.6e-3  # @ 80 deg, R=1000
    else:
        eAxisFlat = 9.0e-3  # @ 80 deg
else:
    raise


class EllipticalSagittalCylinderParam(roe.OE):
    def __init__(self, *args, **kwargs):
        kwargs = self.pop_kwargs(**kwargs)
        roe.OE.__init__(self, *args, **kwargs)
        self.isParametric = True
        self.reset_pqroll(self.p, self.q, self.roll)

    def reset_pqroll(self, p=None, q=None, roll=None):
        """This method allows re-assignment of *p*, *q* and *roll* from
        outside of the constructor.
        """
        if p is not None:
            self.p = p
        if q is not None:
            self.q = q
        if roll is not None:
            self.roll = roll
        gamma = np.arctan2((self.p + self.q)*np.sin(self.roll),
                           (self.q - self.p)*np.cos(self.roll))
        self.cosGamma = np.cos(gamma)
        self.sinGamma = np.sin(gamma)
        self.x0 = (self.q - self.p)/2. * np.sin(self.roll)
        self.z0 = (self.q + self.p)/2. * np.cos(self.roll)
        self.ellipseA = (self.q + self.p)/2.
        self.ellipseB = np.sqrt(self.q * self.p) * np.cos(self.roll)

    def pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p')  # source-to-mirror
        self.q = kwargs.pop('q')  # mirror-to-focus
        return kwargs

    def xyz_to_param(self, x, y, z):
        xNew, zNew = raycing.rotate_y(x - self.x0, z - self.z0, self.cosGamma,
                                      self.sinGamma)
        return xNew, np.arctan2(y, -zNew), np.sqrt(y**2 + zNew**2)  # s, phi, r

    def param_to_xyz(self, s, phi, r):
        x = s
        y = r * np.sin(phi)
        z = -r * np.cos(phi)
        xNew, zNew = raycing.rotate_y(x, z, self.cosGamma, -self.sinGamma)
        return xNew + self.x0, y, zNew + self.z0

    def local_r(self, s, phi):
        r = self.ellipseA * np.sqrt(1 - s**2/self.ellipseB**2)
        r /= abs(np.cos(phi))
        return np.where(abs(phi) < np.pi/2, r, np.ones_like(phi)*1e20)

    def local_n(self, s, phi):
        nr = -self.ellipseA / self.ellipseB * s /\
            np.sqrt(self.ellipseB**2 - s**2)
        norm = np.sqrt(nr**2 + 1)
        a = nr / norm
        b = np.zeros_like(s)
        c = 1. / norm
        aNew, cNew = raycing.rotate_y(a, c, self.cosGamma, -self.sinGamma)
        return [aNew, b, cNew]


class DicedOEParam(roe.DicedOE):
    def __init__(self, *args, **kwargs):
        """
        *dxFacet*, *dyFacet* size of the facets.

        *dxGap*, *dyGat* width of the gap between facets.
        """
        kwargs = self.__pop_kwargs(**kwargs)
        roe.OE.__init__(self, *args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        self.dxFacet = kwargs.pop('dxFacet', 2.1)
        self.dyFacet = kwargs.pop('dyFacet', 1.4)
        self.dxGap = kwargs.pop('dxGap', 0.05)
        self.dyGap = kwargs.pop('dyGap', 0.05)
        self.xStep = self.dxFacet + self.dxGap
        self.yStep = self.dyFacet + self.dyGap
        self.dsFacet = kwargs.pop('dsFacet', 2.1)
        self.dphiFacet = kwargs.pop('dphiFacet', 1.4)
        self.dsGap = kwargs.pop('dsGap', 0.05)
        self.dphiGap = kwargs.pop('dphiGap', 0.05)
        self.sStep = self.dsFacet + self.dsGap
        self.phiStep = self.dphiFacet + self.dphiGap
        return kwargs

    def facet_center_r(self, s, phi):
        """Z of the facet centers at (*s*, *phi*)."""
        return 0  # just flat

    def local_r(self, s, phi, skipReturnR=False):
        """Determines the surface of OE at (*x*, *y*) position."""
        r = self.facet_center_r(s, phi)
        x, y, z = self.param_to_xyz(s, phi, r)
        cs = (s / self.sStep).round() * self.sStep  # center of the facet
        cphi = (phi / self.phiStep).round() * self.phiStep
        cr = self.facet_center_r(cs, cphi)
        cx, cy, cz = self.param_to_xyz(cs, cphi, cr)
        cn = self.facet_center_n(cs, cphi)
        fx = x - cx  # local coordinate in the facet
        fy = y - cy
        if skipReturnR:
            return fx, fy, cn
        z = cz + (self.facet_delta_z(fx, fy) - cn[-3]*fx - cn[-2]*fy) / cn[-1]
        s, phi, r = self.xyz_to_param(x, y, z)
        return r

    def local_n(self, s, phi):
        fx, fy, cn = self.local_r(s, phi, skipReturnR=True)
        deltaNormals = self.facet_delta_n(fx, fy)
        if isinstance(deltaNormals[2], np.ndarray):
            useDeltaNormals = True
        else:
            useDeltaNormals = (deltaNormals[2] != 1)
        if useDeltaNormals:
            cn[-1] += deltaNormals[-1]
            cn[-2] += deltaNormals[-2]
            norm = (cn[-1]**2 + cn[-2]**2 + cn[-3]**2)**0.5
            cn[-1] /= norm
            cn[-2] /= norm
            cn[-3] /= norm
        if self.alpha:
            bAlpha, cAlpha = raycing.rotate_x(cn[1], cn[2],
                                              self.cosalpha, -self.sinalpha)
            return [cn[0], bAlpha, cAlpha, cn[-3], cn[-2], cn[-1]]
        else:
            return cn


class DicedEllipticalSagittalCylinderParam(
        DicedOEParam, EllipticalSagittalCylinderParam):
    def __init__(self, *args, **kwargs):
        kwargs = self.pop_kwargs(**kwargs)
        DicedOEParam.__init__(self, *args, **kwargs)
        self.reset_pqroll(self.p, self.q, self.roll)
        self.isParametric = True

    def pop_kwargs(self, **kwargs):
        return EllipticalSagittalCylinderParam.pop_kwargs(self, **kwargs)

    def facet_center_r(self, s, phi):
        return EllipticalSagittalCylinderParam.local_r(self, s, phi)

    def facet_center_n(self, s, phi):
        return EllipticalSagittalCylinderParam.local_n(self, s, phi)

if isDiced:
    xAxisLim = 2  # 2 * dxFacet
    facetKWargs = {'dxFacet': 1-0.05, 'dyFacet': dyCrystal+1,
                   'dxGap': 0.05, 'dyGap': 0.}
    if isElliptical:
        facetKWargs['dsFacet'] = 1-0.05
        facetKWargs['dphiFacet'] = np.pi
        facetKWargs['dsGap'] = 0.05
        facetKWargs['dphiGap'] = 0
        Toroid = DicedEllipticalSagittalCylinderParam
        analyzerName = crystalMaterial + 'vonHamosDicedElliptical'
    else:
        Toroid = roe.DicedJohannToroid
        analyzerName = crystalMaterial + 'vonHamosDicedCircular'
    dphi = np.arcsin((facetKWargs['dxFacet'] + facetKWargs['dxGap']) / Rs)
else:
    xAxisLim = 5
    facetKWargs = {}
    if isElliptical:
        Toroid = EllipticalSagittalCylinderParam
        analyzerName = crystalMaterial + 'vonHamosElliptical'
    else:
        Toroid = roe.JohannToroid
        analyzerName = crystalMaterial + 'vonHamosCircular'

if isElliptical:
    facetKWargs['p'] = Rs
    facetKWargs['q'] = Rs * elongation
#    eAxisFlat /= elongation
#    yAxisLine *= 2
else:
    facetKWargs['Rm'] = Rm
    facetKWargs['Rs'] = Rs


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine(azimuth=0, height=0)
    rs.GeometricSource(
        beamLine, 'GeometricSource', nrays=nrays, dx=beamH, dy=0,
        dz=beamV, distxprime='flat', distzprime='flat', polarization=None)
    beamLine.analyzer = Toroid(
        beamLine, analyzerName, surface=('',),
        limPhysX=(-dxCrystal/2, dxCrystal/2),
        limPhysY=(-dyCrystal/2, dyCrystal/2),
        shape='rect',
        **facetKWargs)
    beamLine.detector = rsc.Screen(beamLine, 'Detector', z=(0, 0, 1))
#    beamLine.s1h = ra.RectangularAperture(
#        beamLine, 'horizontal. slit', 0, Rs-10.,
#        ('left', 'right'), [-0.1, 0.1])
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
#    beamLine.s1h.propagate(beamSource)
    beamAnalyzerGlobal, beamAnalyzerLocal = \
        beamLine.analyzer.reflect(beamSource)
    beamDetector = beamLine.detector.expose(beamAnalyzerGlobal)
    outDict = {'beamSource': beamSource,
               'beamAnalyzerGlobal': beamAnalyzerGlobal,
               'beamAnalyzerLocal': beamAnalyzerLocal,
               'beamDetector': beamDetector
               }
    return outDict
rr.run_process = run_process


def align_spectrometer_Rs(beamLine, theta, Rs):
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sin2Theta = np.sin(2 * theta)
    cos2Theta = np.cos(2 * theta)
    p = Rs / sinTheta
    q = p * elongation if isElliptical else p
    yDet = p + q*cos2Theta
    zDet = q*sin2Theta

    beamLine.analyzer.center = 0, p, 0
    beamLine.analyzer.pitch = theta
    beamLine.detector.center = 0, yDet, zDet
    beamLine.detector.set_orientation(z=(0, cosTheta, sinTheta))

    beamLine.sources[0].dxprime = 1.1 * dxCrystal / p
    beamLine.sources[0].dzprime = dyCrystal * np.sin(theta) / p / 4
    print('theta={0}deg, Rs={1}mm: p={2}mm'.format(np.degrees(theta), Rs, p))


#def align_spectrometer_p(beamLine, theta, p):
#    sinTheta = np.sin(theta)
#    cosTheta = np.cos(theta)
#    sin2Theta = np.sin(2 * theta)
#    Rs = p * sinTheta
#    yDet = p * 2 * cosTheta**2
#    zDet = p * sin2Theta
#
#    beamLine.analyzer.center = 0, p, 0
#    beamLine.analyzer.Rs = Rs
#    beamLine.analyzer.pitch = theta
#    beamLine.detector.center = 0, yDet, zDet
#    beamLine.detector.set_orientation(z=(0, cosTheta, sinTheta))
#
#    beamLine.sources[0].dxprime = 1.1 * dxCrystal / p
#    beamLine.sources[0].dzprime = dyCrystal * np.sin(theta) / p
#    print('theta={0}deg, p={1}mm: Rs={2}mm'.format(np.degrees(theta), p, Rs))


def stripe_number(beam):
    phi = np.arcsin(beam.x / Rs)
    return np.round(phi / dphi)


def define_plots(beamLine):
    fwhmFormatStrE = '%.2f'
    plots = []
    plotsAnalyzer = []
    plotsDetector = []
    plotsE = []

    limits = [-dxCrystal/2 - 5, dxCrystal/2 + 5]
    plotAnE = xrtp.XYCPlot(
        'beamAnalyzerLocal', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=400, ppb=1, limits=limits),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=400, ppb=1, limits=limits),
        caxis=xrtp.XYCAxis('energy', 'eV', fwhmFormatStr='%.2f',
                           bins=200, ppb=2),
        title='xtal_E', oe=beamLine.analyzer, raycingParam=1000)
    plotAnE.caxis.fwhmFormatStr = fwhmFormatStrE
    plotAnE.textPanel = plotAnE.fig.text(
        0.88, 0.85, '', transform=plotAnE.fig.transFigure, size=14, color='r',
        ha='center')
    plotsAnalyzer.append(plotAnE)
    plotsE.append(plotAnE)

    if isDiced:
        plotAnS = xrtp.XYCPlot(
            'beamAnalyzerLocal', (1,),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=400, ppb=1, limits=limits),
            yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=400, ppb=1, limits=limits),
            caxis=xrtp.XYCAxis('stripe number', '', data=stripe_number),
            title='xtal_stripes', oe=beamLine.analyzer, raycingParam=1000)
        plotAnS.textPanel = plotAnS.fig.text(
            0.88, 0.85, '', transform=plotAnS.fig.transFigure, size=14,
            color='r', ha='center')
        plotsAnalyzer.append(plotAnS)

    if isDiced:
        xMax = (facetKWargs['dxFacet']+1) / 2
        limits = [-xMax, xMax]
    else:
        limits = [-3, 3]
    plot = xrtp.XYCPlot(
        'beamAnalyzerLocal', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limits),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=limits),
        caxis=xrtp.XYCAxis('energy', 'eV', fwhmFormatStr='%.2f'),
        title='xtal_E_zoom', oe=beamLine.analyzer, raycingParam=1000)
    plot.caxis.fwhmFormatStr = fwhmFormatStrE
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plotsAnalyzer.append(plot)
    plotsE.append(plot)

    if isDiced:
        plot = xrtp.XYCPlot(
            'beamAnalyzerLocal', (1,),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limits),
            yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=limits),
            caxis=xrtp.XYCAxis('stripe number', '', data=stripe_number),
            title='xtal_stripes_zoom', oe=beamLine.analyzer,
            raycingParam=1000)
        plot.textPanel = plot.fig.text(
            0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
            ha='center')
        plotsAnalyzer.append(plot)

    plotDetE = xrtp.XYCPlot(
        'beamDetector', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', fwhmFormatStr='%.3f'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', fwhmFormatStr='%.3f'),
        title='det_E')
    plotDetE.xaxis.limits = -xAxisLim, xAxisLim
    plotDetE.yaxis.limits = -yAxisLim, yAxisLim
    plotDetE.caxis.fwhmFormatStr = fwhmFormatStrE
    plotDetE.textPanel = plotDetE.fig.text(
        0.88, 0.8, '', transform=plotDetE.fig.transFigure, size=14, color='r',
        ha='center')
    plotsDetector.append(plotDetE)
    plotsE.append(plotDetE)

    if isDiced:
        plotDetS = xrtp.XYCPlot(
            'beamDetector', (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', 'mm', fwhmFormatStr='%.3f'),
            yaxis=xrtp.XYCAxis(r'$z$', 'mm', fwhmFormatStr='%.3f'),
            caxis=xrtp.XYCAxis('stripe number', '', data=stripe_number),
            beamC='beamAnalyzerLocal', title='det_stripes')
        plotDetS.xaxis.limits = -xAxisLim, xAxisLim
        plotDetS.yaxis.limits = -yAxisLim, yAxisLim
        plotDetS.textPanel = plotDetS.fig.text(
            0.88, 0.8, '', transform=plotDetS.fig.transFigure, size=14,
            color='r', ha='center')
        plotsDetector.append(plotDetS)

    for plot in plotsAnalyzer:
        plots.append(plot)
    for plot in plotsDetector:
        plots.append(plot)
    return plots, plotsAnalyzer, plotsDetector, plotsE, plotDetE


def plot_generator(plots, plotsAnalyzer, plotsDetector, plotsE, plotDetE,
                   beamLine):
    hklSeparator = ',' if np.any(np.array(crystal.hkl) > 10) else ''
    crystalLabel = '{0}{1[0]}{2}{1[1]}{2}{1[2]}'.format(
        crystalMaterial, crystal.hkl, hklSeparator)
    beamLine.analyzer.surface = crystalLabel,
    for plot in plotsAnalyzer:
        plot.draw_footprint_area()
    beamLine.analyzer.material = crystal
    theta = np.radians(thetaDegree)
    align_spectrometer_Rs(beamLine, theta, Rs)

    sinTheta = np.sin(theta)
    E0raw = rm.ch / (2 * crystal.d * sinTheta)
    dTheta = crystal.get_dtheta_symmetric_Bragg(E0raw)
    E0 = rm.ch / (2 * crystal.d * np.sin(theta + dTheta))
    offsetE = round(E0, 3)

    dELine = 0
    dzLine = 0
    for isource in np.arange(3):
#    for isource in [-1, ]:
        xrtr.runCardVals.repeats = numiter
        if isource == 0 or isource == -1:  # flat or norm
#            xrtr.runCardVals.repeats = 0
            eAxisMin = E0 * (1 - eAxisFlat)
            eAxisMax = E0 * (1 + eAxisFlat)
            for plot in plotsE:
                if plot is None:
                    continue
                plot.caxis.offset = int(round(offsetE, -2))
                plot.caxis.limits = [eAxisMin, eAxisMax]
            tt = r'{0}{1}$\theta = {2:.0f}^\circ$'.format(
                crystalLabel, '\n', thetaDegree)
            for plot in plots:
                try:
                    plot.textPanel.set_text(tt)
                except AttributeError:
                    pass
            if isource == -1:
                beamLine.sources[0].distE = 'normal'
                beamLine.sources[0].energies = E0, eAxisFlat/2.355
                sourcename = 'norm'
            else:
                beamLine.sources[0].distE = 'flat'
                beamLine.sources[0].energies = eAxisMin, eAxisMax
                sourcename = 'flat'
        elif isource == 1:  # line
#            xrtr.runCardVals.repeats *= 4
#            xrtr.runCardVals.repeats = 0
            beamLine.sources[0].distE = 'lines'
            beamLine.sources[0].energies = E0,
            sourcename = 'line'
            for plot in plotsDetector:
                plot.yaxis.limits = yAxis1Line
        else:
#            xrtr.runCardVals.repeats = 2560*16L
            tt = (r'{0}{1}$\theta = {2:.0f}^\circ${1}$' +
                  '\delta E = ${3:.3f} eV').format(
                crystalLabel, '\n', thetaDegree, dELine)
            for plot in plots:
                try:
                    plot.textPanel.set_text(tt)
                except AttributeError:
                    pass
            beamLine.sources[0].distE = 'lines'
            sourcename = '7lin'
            for plot in plotsDetector:
#                plot.yaxis.limits = [-dzLine*7, dzLine*7]
                plot.yaxis.limits = yAxis7Lines
            dEStep = dELine
            beamLine.sources[0].energies = \
                [E0 + dEStep * i for i in range(-3, 4)]
            eAxisMin = E0 - dEStep * 4
            eAxisMax = E0 + dEStep * 4
            for plot in plotsE:
                if plot is None:
                    continue
                plot.caxis.limits = [eAxisMin, eAxisMax]

        for plot in plots:
            filename = '{0}{1}-{2}-{3}'.format(
                analyzerName, thetaDegree, plot.title, sourcename)
            plot.saveName = filename + '.png'
#            plot.persistentName = filename + '.pickle'
        yield

        if isource == 0:
            dzFlat = plotDetE.dy
            dEFlat = plotDetE.dE
        elif isource == 1:
            dzLine = plotDetE.dy
            try:
                dELine = dzLine * dEFlat / dzFlat
            except:
                print('dzFlat={0}'.format(dzFlat))
                dELine = 0


def main():
    beamLine = build_beamline()
    plots, plotsAnalyzer, plotsDetector, plotsE, plotDetE = \
        define_plots(beamLine)
    args = [plots, plotsAnalyzer, plotsDetector, plotsE, plotDetE, beamLine]
    xrtr.run_ray_tracing(
        plots, generator=plot_generator, generatorArgs=args,
        beamLine=beamLine, processes='half')


def plotCylinders():
    import matplotlib.pyplot as plt

    beamLine = raycing.BeamLine(azimuth=0, height=0)
    beamLine.analyzer = DicedEllipticalSagittalCylinderParam(
        beamLine, analyzerName, surface=('',),
        limPhysX=(-dxCrystal/2, dxCrystal/2),
        limPhysY=(-dyCrystal/2, dyCrystal/2),
        shape='rect',
        **facetKWargs)

    fig = plt.figure(figsize=(15, 3), dpi=72)
    rect2d = [0.05, 0.1, 0.9, 0.8]
    ax1 = fig.add_axes(rect2d, aspect='equal')
    ax1.set_xlabel(r'$x$ (mm)')
    ax1.set_ylabel(r'$z$ (mm)')

    s = np.linspace(-dxCrystal/2, dxCrystal/2, 5000)

    beamLine.analyzer.reset_pqroll(Rs, Rs, 0)
    r = beamLine.analyzer.local_r(s, np.zeros_like(s))
    z = Rs - (r**2 - s**2)**0.5
    ax1.plot(s, z, 'b-', lw=1, label=r'circular ($q = p = $ 250 mm)')

    beamLine.analyzer.reset_pqroll(Rs, 1.5*Rs, 0)
    r = beamLine.analyzer.local_r(s, np.zeros_like(s))
    z = Rs - (r**2 - s**2)**0.5
    z -= np.min(z)
    ax1.plot(s, z, 'r-', lw=1, label=r'elliptical  ($q = $ 1.5 $p$)')

    ax1.set_xlim([-dxCrystal/2, dxCrystal/2])
    ax1.set_ylim([0, 10.5])
    ax1.legend(loc='upper center')

    fig.savefig('Cylinders.png')
    plt.show()


#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
#    plotCylinders()
