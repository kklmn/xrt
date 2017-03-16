# -*- coding: utf-8 -*-
r"""
.. !!! select one of the three functions to run at the very bottom !!!

.. _gratingDiffraction:

Diffraction from grating
------------------------

Various gratings described in [Boots]_ have been tested with xrt for
diffraction efficiency. The efficiency curves in [Boots]_ were calculated by
means of the code ``peg`` which provides almost identical results to those by
``REFLEC`` [REFLEC]_ but with reportedly better convergence. In order to have
comparison curves, we got the REFLEC results calculated by R. Sankari
[Sankari]_ which were basically equal to those in [Boots]_.

.. [Boots] M. Boots, D. Muir and A. Moewes, *Optimizing and characterizing
   grating efficiency for a soft X-ray emission spectrometer*,
   J. Synchrotron Rad. **20** (2013) 272–285.

.. [REFLEC] F. Schäfers and M. Krumrey, Technischer Bericht, BESSY TB 201
   (1996).

.. [Sankari] R. Sankari, private communication (2015).

The diffraction orders are shown below as transverse images and meridional
("polar exit angle") and sagittal ("azimuthal exit angle") cuts. Notice that
the diffraction orders are positioned on the screen "by themselves", i. e. with
only the use of the Kirchhoff diffraction integral. Also before it was
possible to work with grating diffraction orders in xrt within the geometrical
ray tracing approach. In that approach the rays were deflected according to the
grating equation. Here, in wave propagation, the grating equation was only used
to position the screen.

.. imagezoom:: _images/1-LEG_E-orders-tr1.*

\

.. imagezoom:: _images/1-LEG_E-orders-x.*

\

.. imagezoom:: _images/1-LEG_E-orders-z.*

Notice that in contrast to the conventional grating theories (also used in
``REFLEC``), the diffraction orders here have the sagittal dimension. And that
dimension has diffraction fringes and a variable width, too!

The resulting efficiency was obtained as ratio of the flux into the given order
over the incoming flux. The incoming radiation was considered as uniform,
parallel and fully coherent.

For the LEG (Low Energy Grating, see its properties in the figure below), the
efficiency curves are pretty similar to those by ``REFLEC``. The main
difference is the low-energy part. Our 1st order does not decrease so rapidly.
If we consider not the 2D exit angle but only the central azimuthal cut, the
resulted low-energy efficiencies are very similar to that of ``REFLEC`` (not
shown). Ref. [Boots]_ also provides experimental measurements which also have a
rapid low-energy decrease. It seems that the detector had a pinhole that might
cut the beam at low energies as the diffracted beam becomes wider there (see
the transverse pictures above), which may explain lower measured efficiency.

.. imagezoom:: _images/1-LEG_E-eff.*

For the IMP grating ("impurity", see its properties in the figure below), the
difference is bigger.

.. imagezoom:: _images/2-IMP_E-eff.*

We believe that ``REFLEC`` is essentially wrong at high energies. If we
mentally translate the working terraces of a blazed grating to form a
continuous plane, we get a mirror at the *pitch* + *blaze* angle. By energy
conservation, the overall grating efficiency (the sum into all orders) cannot
be higher than the reflectivity of such a mirror. The ``REFLEC`` curves can
violate this limit even for a single order, compare with the blue curve in the
figure above. The reason for such behavior seems to be the artificially
shadowless illumination by the incoming wave. Indeed, ``REFLEC`` assumes the
complete saw profile to work in the diffraction, whereas the back side and a
portion of the front side behind it stay in the shadow. We compare the two
gratings shown below, one is with 90 degree anti-blaze angle and the other is
with *pitch* as anti-blaze angle.

.. imagezoom:: _images/1-LEG_profile-adhoc.*
.. imagezoom:: _images/1-LEG_profile-anti.*

``REFLEC`` gives different efficiencies for these two cases (see above) whereas
xrt cannot distinguish them. We tried to artificially remove the shadows by
making the surface "emit" a coherent wave. The result was an increase in the
high-energy efficiency, similarly to the ``REFLEC``'s behaviour.

The factors which definitely will affect the efficiency are (1) restricted
coherence radius and (2) roughness. Both will be added into this example in a
later release of xrt.

We are open for further discussion on the above results with interested
scientists.
"""
#Set proper setting for the grating and comment/uncomment one of the three main
#invoked functions (at the very bottom).
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "16 Mar 2017"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import pickle
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.waves as rw

#coating = None
#coatingGr = rm.EmptyMaterial(kind='grating')

cwd = os.getcwd()
dE = 0.1
dx = 2.0
dz = 0.5
minOrder = 0  # even -1 is wrong because |beta| becomes > pi/2
maxOrder = 3
maxDisplayOrder = 3

#LEG:
pitch = np.radians(4.)
blaze = np.radians(1.85)
rho = 600.  # lines/mm
material = 'Au'
coating = rm.Material('Au', rho=19.3)
coatingGr = rm.Material('Au', rho=19.3, kind='grating')
#energies = np.linspace(60, 275, 44)
energies = np.linspace(60, 280, 23)
prefix = '1-LEG'

##IMP:
#pitch = np.radians(3.)
#blaze = np.radians(1.11)
#rho = 900.  # lines/mm
#material = 'Ni'
#coating = rm.Material('Ni', rho=8.902)
#coatingGr = rm.Material('Ni', rho=8.902, kind='grating')
##energies = np.linspace(75, 905, 84)
#energies = np.linspace(75, 915, 22)
#prefix = '2-IMP'

##MEG:
#pitch = np.radians(2.)
#blaze = np.radians(1.48)
#rho = 1200.  # lines/mm
#material = 'Ni'
#coating = rm.Material('Ni', rho=8.902)
#coatingGr = rm.Material('Ni', rho=8.902, kind='grating')
##energies = np.linspace(125, 925, 161)
#energies = np.linspace(125, 925, 41)
#prefix = '3-MEG'

##HEG:
#pitch = np.radians(2.)
#blaze = np.radians(1.52)
#rho = 2000.  # lines/mm
#material = 'Pt'
#coating = rm.Material('Pt', rho=21.45)
#coatingGr = rm.Material('Pt', rho=21.45, kind='grating')
#energies = np.linspace(240, 1160, 185)
#prefix = '4-HEG'

angles = np.linspace(0, 2e-3, 41)
#whatToScan = 'angle'
whatToScan = 'energy'
if whatToScan == 'energy':
    suffix = '_E'
else:
    suffix = '_pitch'

p = 50.
q = 5000.

nrays = 2e5
xBins, xppb = 128, 2
zBins, zppb = 128, 2
eBins, eppb = 1, 200
xName = 'yaw'
yName = 'pitch'
thetaOffset = pitch

xmaxW = 12.5 * dx / q
zmaxW = 2.5 * dz / q
screenThetas = np.linspace(-zmaxW, zmaxW, zBins)
orders = maxOrder + 1 - minOrder

if xBins == 1:
    screenPhis = 0.
else:
    screenPhis = np.linspace(-xmaxW, xmaxW, xBins)


class Grating(roe.OE):
    def local_g(self, x, y, rho=rho):
        return 0, -rho, 0  # constant line spacing along y


#class BlazedGratingAlt(roe.BlazedGrating):
#    def rays_good(self, x, y, is2ndXtal=False):
#        locState = roe.BlazedGrating.rays_good(self, x, y, is2ndXtal)
#        locState[(y % self.rho_1) > self.rho_1/2.] = self.lostNum
#        #print('{0} or {1} are lost'.format(
#            #(locState == self.lostNum).sum(), len(y)))
#        return locState


def order_2theta(order, E, pitch, rho=rho):
    l_d = rm.ch / E * 1e-7 * rho
    wrong = np.abs(np.asarray(order)*l_d - np.cos(pitch)) > 1
    if wrong.sum() > 0:
        raise ValueError('wrong orders:', np.asarray(order)[wrong])
    beta = np.arcsin(np.asarray(order)*l_d - np.cos(pitch))  # <0!
    return np.pi/2 + beta + pitch

sourceType = 'flat'
#sourceType = 'annulus'
#sourceType = 'divergent'
if sourceType == 'flat':
    kw = {'distx': 'flat', 'dx': dx, 'distz': 'flat', 'dz': dz,
          'distxprime': None, 'distzprime': None}
#    prefix += '-01' + sourceType + '-'
elif sourceType == 'annulus':
    kw = {'distx': 'annulus', 'dx': (0, dx/2),
          'distxprime': None, 'distzprime': None}
#    prefix += '-02' + sourceType + '-'
elif sourceType == 'divergent':
    dPrime = 2e-5
    kw = {'distx': None, 'distz': None,
          'distxprime': None, 'distzprime': 'flat',
          'dxprime': dPrime, 'dzprime': dPrime}
#    prefix += '-03' + sourceType + '-'

kw['distE'] = 'lines'
polarization = 'horizontal'
visualizeCrossSection = True

cmap = cm.get_cmap('jet')
fName = os.path.join(cwd, prefix + suffix)
pickleName = fName + '.pickle'

efficiencyFileName = 'efficiency' + prefix


def get_grating_area_fraction(rho, blaze, pitch):
    rho_1 = 1. / rho
    y1 = rho_1 * np.tan(blaze) / (np.tan(blaze) + np.tan(pitch))
    z1 = -y1 * np.tan(pitch)
    y2 = rho_1
    z2 = 0
    d = ((y2-y1)**2 + (z2-z1)**2)**0.5
    print('d*rho =', d*rho)
    print('d, 1/rho, rho =', d, rho_1, rho)
    return y1, y2, z1, z2, d*rho


def visualize_grating():
    beamLine = raycing.BeamLine()
#    bg = BlazedGratingAlt(
    bg = roe.BlazedGrating(
        beamLine, 'BlazedGrating', (0, p, 0), pitch=pitch, material=coating,
        blaze=blaze,
        #antiblaze=pitch,
        rho=rho)

    fig1 = plt.figure(figsize=(8, 6), dpi=72)
    rect2d = [0.1, 0.1, 0.8, 0.8]
    ax = fig1.add_axes(rect2d, aspect='auto')
    ax.set_xlabel(u'y (µm)')
    ax.set_ylabel(u'z (nm)')
    ax.set_title(
        u'Grating profile with {0:.0f} lines/mm and {1}° blaze angle\n'.format(
            rho, np.degrees(blaze)) +
        u'and beam footprint at {0}° pitch angle'.format(np.degrees(pitch)))

    maxY = 2.2 * bg.rho_1
    y = np.linspace(-2*maxY, 2*maxY, 280)
    z = bg.local_z(0, y)
    ax.plot(y*1e3, z*1e6, '-k', lw=2)

    beamSource = rs.Beam(nrays=len(y), forceState=1)
    beamSourceLoc = rs.Beam(copyFrom=beamSource)
    raycing.global_to_virgin_local(bg.bl, beamSource, beamSourceLoc,
                                   bg.center, slice(None))
    raycing.rotate_beam(beamSourceLoc, slice(None), pitch=-pitch)

    beamSource.z[:] = y * np.tan(pitch)
    beamGloSG, beamLoSG = bg.reflect(beamSource)

    ax.plot([beamSourceLoc.y*1e3, beamLoSG.y*1e3],
            [beamSourceLoc.z*1e6, beamLoSG.z*1e6], '-r', alpha=0.2, lw=0.5)
    ax.plot(beamLoSG.y*1e3, beamLoSG.z*1e6, 'or', alpha=0.5, lw=3)

    ax.set_xlim(-maxY*1e3, maxY*1e3)
    mz = bg.rho_1*np.tan(blaze)
    ax.set_ylim(-mz*1.1e6, mz*0.2e6)

    y1, y2, z1, z2, d = get_grating_area_fraction(rho, blaze, pitch)
    xs = np.array([y1, y2]) * 1e3
    ys = np.array([z1, z2]) * 1e6
    ax.plot(xs, ys, 'og', lw=3)

    fig1.savefig(prefix + '_profile.png')
    plt.show()


def build_beamline(nrays=nrays):
    beamLine = raycing.BeamLine()
    rs.GeometricSource(
        beamLine, 'source', nrays=nrays, polarization=polarization, **kw)

#    beamLine.bg = BlazedGratingAlt(
    beamLine.bg = roe.BlazedGrating(
        beamLine, 'BlazedGrating', (0, p, 0), pitch=pitch, material=coating,
        blaze=blaze, rho=rho)
    drho = beamLine.bg.get_grating_area_fraction()
    beamLine.bg.area = dx * dz / np.sin(pitch) * drho

    beamLine.gr = Grating(
        beamLine, 'Grating', (0, p, 0), pitch=pitch, material=coatingGr,
        order=range(minOrder, maxOrder+1))

    beamLine.fsm = rsc.HemisphericScreen(
        beamLine, 'FSM', (0, p, 0), R=q,
        x=(0, -np.sin(thetaOffset), np.cos(thetaOffset)),
        z=(0, np.cos(thetaOffset), np.sin(thetaOffset)))
    return beamLine


def run_process(beamLine):
    thetaNodes = (beamLine.orderThetas[:, np.newaxis] + screenThetas).flatten()
    waveFSM = beamLine.fsm.prepare_wave(beamLine.bg, screenPhis, thetaNodes)
    wrepeats = 1
    beamLine.fluxIn = 0
    for repeat in range(wrepeats):
        beamSource = beamLine.sources[0].shine(withAmplitudes=True)
        beamLine.fluxIn += (beamSource.Jss + beamSource.Jpp).sum()

        beamGglobal, beamGlocal = beamLine.gr.reflect(beamSource)
        beamFSMrays = beamLine.fsm.expose(beamGglobal)

        oeGlobal, oeLocal = beamLine.bg.reflect(beamSource)
        oeLocal.area = beamLine.bg.area
        rw.diffract(oeLocal, waveFSM)
        if wrepeats > 1:
            print('wave repeats: {0} of {1} done'.format(repeat+1, wrepeats))

    beamLine.fluxIn /= wrepeats
    beamLine.fluxOut = (oeLocal.Jss + oeLocal.Jpp)[oeLocal.state == 1].sum()
    outDict = {'beamSource': beamSource,
               'beamGglobal': beamGglobal, 'beamGlocal': beamGlocal,
               'beamFSMrays': beamFSMrays,
               'waveFSM': waveFSM}
    beamLine.waveFSM = waveFSM
    oeLocal.y = oeLocal.y % (4*beamLine.bg.rho_1) - 2*beamLine.bg.rho_1
    outDict['oeLocal'] = oeLocal
    saw = rs.Beam(copyFrom=oeLocal)
    saw.y = np.random.uniform(-1.1, 1.1, size=len(saw.x)) * beamLine.bg.rho_1
    saw.z = beamLine.bg.local_z(0, saw.y)
    outDict['saw'] = saw

    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []
    plotsR = []

    xmax = 1100 * beamLine.bg.rho_1
    plot = xrtp.XYCPlot(
        'oeLocal', aspect='auto',
        xaxis=xrtp.XYCAxis(r'$y$', u'µm', limits=[-xmax, xmax]),
        yaxis=xrtp.XYCAxis(r'$z$', 'nm', limits=[-1e3*np.tan(blaze)*xmax, 0]),
        caxis=xrtp.XYCAxis('energy', 'meV', bins=eBins, ppb=eppb))
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'saw', aspect='auto',
        xaxis=xrtp.XYCAxis(r'$y$', u'µm', limits=[-xmax, xmax]),
        yaxis=xrtp.XYCAxis(r'$z$', 'nm', limits=[-1e3*np.tan(blaze)*xmax, 0]),
        caxis=xrtp.XYCAxis('energy', 'meV', bins=eBins, ppb=eppb))
    plots.append(plot)

#    for order in range(minOrder, maxOrder+1):
    for order in [1]:
        plot = xrtp.XYCPlot(
            'beamFSMrays', aspect='auto',
            xaxis=xrtp.XYCAxis(r'$\phi$', u'µrad', bins=xBins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$\theta$', u'µrad', bins=zBins, ppb=zppb),
            caxis=xrtp.XYCAxis('energy', 'meV', bins=eBins, ppb=eppb))
        plot.title = 'beamFSMrays{0}'.format(order)
        plot.baseName = plot.title
        plot.order = order
        plots.append(plot)
        plotsR.append(plot)

        plot = xrtp.XYCPlot(
            'waveFSM', aspect='auto',
            xaxis=xrtp.XYCAxis(r'$\phi$', u'µrad', bins=xBins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$\theta$', u'µrad', bins=zBins, ppb=zppb),
            caxis=xrtp.XYCAxis('energy', 'meV', bins=eBins, ppb=eppb))
        plot.title = 'beamFSMwave{0}'.format(order)
        plot.baseName = plot.title
        plot.order = order
        plots.append(plot)
        plotsR.append(plot)

    for plot in plots:
#    plot.xaxis.limits = [-xmax, xmax]
        plot.xaxis.fwhmFormatStr = None
        plot.fluxFormatStr = '%.2p'
        if hasattr(plot, 'baseName'):
            plot.saveName = [plot.baseName + '.png']
#        plot.persistentName = plot.baseName + '.pickle'
    return plots, plotsR


def plot_generator(plots, plotsR, beamLine):
    xFactor = 1e6
    zFactor = 1e6

    orders = maxOrder + 1 - minOrder
    ilen = zBins * xBins * orders
    if whatToScan.startswith('angle'):
        scanAxis = angles
        lenAngles = len(angles)
        eff = np.zeros((lenAngles, orders))
        if visualizeCrossSection:
            extFlux = np.zeros((lenAngles, ilen))
        E0 = (energies[0] + energies[-1]) / 2
        beamLine.sources[0].energies = [E0]
    else:
        scanAxis = energies
        lenEnergies = len(energies)
        eff = np.zeros((lenEnergies, orders))
        if visualizeCrossSection:
            extFlux = np.zeros((lenEnergies, ilen))

    for isa, sa in enumerate(scanAxis):
        if whatToScan.startswith('angle'):
            print('angle scan:', sa, isa+1, 'of', lenAngles)
            pitchn = pitch + sa
            drho = beamLine.bg.get_grating_area_fraction()
            beamLine.bg.area = dx * dz / np.sin(pitchn) * drho
            thetaOffset = pitchn
            beamLine.fsm.set_orientation(
                x=(0, -np.sin(thetaOffset), np.cos(thetaOffset)),
                z=(0, np.cos(thetaOffset), np.sin(thetaOffset)))
            beamLine.bg.pitch = pitchn
        else:
            thetaOffset = pitch
            E0 = sa
            print('energy scan: {0}eV, {1} of {2}'.format(
                E0, isa+1, lenEnergies))
            for plot in plots:
                if plot.caxis.label == 'energy':
                    ef = plot.caxis.factor
                    plot.caxis.limits = [(E0-dE/2)*ef, (E0+dE/2)*ef]
                    plot.caxis.offset = E0 * ef
                    plot.caxis.offsetDisplayFactor = 1e-3
                    plot.caxis.offsetDisplayUnit = 'eV'
            beamLine.sources[0].energies = [E0]

        beamLine.orderThetas = np.pi/2 - order_2theta(
            range(minOrder, maxOrder+1), E0, thetaOffset) + thetaOffset

        for plot in plotsR:
            th = beamLine.orderThetas[plot.order-minOrder]
            plot.xaxis.limits = [-xmaxW*xFactor, xmaxW*xFactor]
            plot.yaxis.limits = [(th-zmaxW)*zFactor, (th+zmaxW)*zFactor]
            plot.yaxis.offset = round(th * zFactor, -1)
            plot.yaxis.offsetDisplayFactor = 1e-6
            plot.yaxis.offsetDisplayUnit = 'rad'
        yield

        flux = beamLine.waveFSM.Jss + beamLine.waveFSM.Jpp
        eff[isa, :] = flux.reshape(orders, zBins, xBins).sum(
            axis=(1, 2)) / beamLine.fluxIn
        print('efficiency sum over orders {0} through {1} = {2}'.format(
            minOrder, maxOrder, eff[isa, :].sum()))
        print('order efficiencies = {0}'.format(eff[isa, :]))

        if visualizeCrossSection:
            extFlux[isa, :] = flux
    dump = [minOrder, maxOrder, scanAxis, eff,
            xBins, zBins, screenPhis, screenThetas, visualizeCrossSection]

    if visualizeCrossSection:
        dump.append(extFlux)
    with open(pickleName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)

    if whatToScan == 'energy':
        dump = [energies, eff, minOrder, maxOrder]
        with open(efficiencyFileName+'.pickle', 'wb') as f:
            pickle.dump(dump, f, protocol=2)

        dump = [energies]
        dump += [eff[:, o] for o in range(orders)]
        np.savetxt(efficiencyFileName+'.txt', np.array(dump).T,
                   header='minOrder={0}, maxOrder={1}'.format(
                       minOrder, maxOrder))


def afterScript():
    print('Now run "visualize_efficiency()"')


def get_efficiency():
    beamLine = build_beamline()
    plots, plotsR = define_plots(beamLine)
    args = [plots, plotsR, beamLine]
    xrtr.run_ray_tracing(plots, repeats=2, beamLine=beamLine, processes=1,
                         generator=plot_generator, generatorArgs=args,
                         afterScript=afterScript)


def read_curves():
    with open(pickleName, 'rb') as f:
        res = pickle.load(f)
    return res


def create_fig(rect2d, cap, scanAxis, axisLabel, scanAxisFactor,
               maxOrder, withEbar=True, textcolor='k'):
    fig1 = plt.figure(figsize=(12, 6), dpi=72)
    rect2dX = rect2d[2] / (maxOrder+1)
    ax = []
    sharey = None
    for o in range(maxOrder+1):
        recti = [rect2d[0] + o*rect2dX, rect2d[1], rect2dX-0.004, rect2d[3]]
        axi = fig1.add_axes(recti, aspect='auto', sharey=sharey)
        sharey = axi
        axi.locator_params(axis='x', nbins=5)
        orderText = r'{0}$^{{\rm{1}}}$ order'.format(
            o, 'st' if o == 1 else 'nd' if o == 2 else 'rd' if o == 3
            else 'th')
        axi.text(0.01, 0.5, orderText, rotation='vertical',
                 transform=axi.transAxes, ha='left', va='center', fontsize=14,
                 color=textcolor)
        ax.append(axi)
    for axi in ax[1:]:
        plt.setp(axi.get_yticklabels(), visible=False)
    ax[maxOrder//2].set_xlabel(cap+u' exit angle (µrad)', fontsize=14)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 5))
    ax[0].set_ylabel('normalized intensity (a.u.)', fontsize=14)

    if withEbar:
        rect2d = [0.95, 0.1, 0.02, 0.8]
        ax1c = fig1.add_axes(rect2d, aspect='auto')
        ax1c.set_ylabel(axisLabel, fontsize=14)
        plt.setp(ax1c, xticks=())
        yLim = scanAxis[0] * scanAxisFactor, scanAxis[-1] * scanAxisFactor
        ax1c.set_ylim(yLim[0], yLim[1])
        a = np.outer(np.arange(0, 1, 0.01), np.ones(10))
        ax1c.imshow(a, aspect='auto', cmap=cmap, origin="lower",
                    extent=[0, 1, yLim[0], yLim[1]])

    return fig1, ax


def visualize_efficiency():
    if whatToScan == 'energy':
        axisLabel = 'energy (eV)'
        scanAxisFactor = 1
    else:
        axisLabel = 'pitch (mrad)'
        scanAxisFactor = 1e3

    fCalc1 = None
    fCalc2 = None
    fCalc3 = None
    if prefix.startswith('1-LEG'):
        eCalc1, fCalc1 = np.loadtxt('LEG_1st_1p85.dat', unpack=True)
        eCalc2, fCalc2 = np.loadtxt('LEG_2nd_1p85.dat', unpack=True)
    elif prefix.startswith('2-IMP'):
        eCalc1, fCalc1 = np.loadtxt('IMP_1st_1p11.dat', unpack=True)
        eCalc2, fCalc2 = np.loadtxt('IMP_2nd_1p11.dat', unpack=True)
        eCalc3, fCalc3 = np.loadtxt('apex175p9.dat', unpack=True)
    elif prefix.startswith('3-MEG'):
        eCalc1, fCalc1 = np.loadtxt('MEG_1st_1p48.dat', unpack=True)
        eCalc2, fCalc2 = np.loadtxt('MEG_2nd_1p48.dat', unpack=True)

    res = read_curves()
    minOrder, maxOrder, scanAxis, eff, xBins, zBins,\
        screenPhis, screenThetas, pickleCrossSection = res[0:9]
    maxPlotOrder = min(maxDisplayOrder, maxOrder)

    figEff = plt.figure(figsize=(6, 6), dpi=72)
    rect2d = [0.15, 0.1, 0.8, 0.8]
    axEff = figEff.add_axes(rect2d, aspect='auto', xlabel=axisLabel,
                            ylabel='absolute grating efficiency')
    axEff.set_title(
        u'{0} {1} at {2}° pitch and {3}° blaze angles'.format(
            material, prefix[2:5], np.degrees(pitch), np.degrees(blaze)))
    axEff.plot(scanAxis*scanAxisFactor, eff[:, 1-minOrder], '.r', lw=2,
               label='1 xrt')
    axEff.plot(scanAxis*scanAxisFactor, eff[:, 2-minOrder], '.g', lw=2,
               label='2 xrt')
    axEff.plot(scanAxis*scanAxisFactor, eff[:, 3-minOrder], '.b', lw=2,
               label='3 xrt')

#    axEff.plot(scanAxis*scanAxisFactor, eff[:, 4-minOrder], '.-', lw=2,
#               label='4 xrt')
#    axEff.plot(scanAxis*scanAxisFactor, eff[:, 5-minOrder], '.-', lw=2,
#               label='5 xrt')
#    axEff.plot(scanAxis*scanAxisFactor, eff[:, 6-minOrder], '.-', lw=2,
#               label='6 xrt')
#    axEff.plot(scanAxis*scanAxisFactor, eff[:, 7-minOrder], '.-', lw=2,
#               label='7 xrt')

    if whatToScan == 'energy':
        if fCalc1 is not None:
            axEff.plot(eCalc1, fCalc1, '-r', lw=1, label='1 reflec')
        if fCalc2 is not None:
            axEff.plot(eCalc2, fCalc2, '-g', lw=1, label='2 reflec')
        if fCalc3 is not None:
            axEff.plot(eCalc3, fCalc3, '-m', lw=1,
                       label=u'1 reflec, 3° anti-blaze')

        refl = coating.get_amplitude(
            scanAxis*scanAxisFactor, np.sin(pitch+blaze))
        rs = refl[0]
        axEff.plot(
            scanAxis*scanAxisFactor, abs(rs)**2, '-b', lw=2,
            label=material+u' mirror at\n{0}° pitch'.format(
                np.degrees(pitch+blaze)))

    lines = axEff.lines
    labels = [l.get_label() for l in lines]
    axEff.legend(lines, labels, title='orders', loc='upper right')
#    axEff.add_artist(hvLegend)
    axEff.set_xlim(scanAxis[0]*scanAxisFactor, scanAxis[-1]*scanAxisFactor)
    axEff.set_ylim(0, 1)
#    axEff.set_xlim(50, 275)

    figEff.savefig(fName + '-eff.png')

    if pickleCrossSection:
        extFlux = res[9]
        elen = extFlux.shape[0]
        orders = maxOrder + 1 - minOrder
        extFlux = extFlux.reshape(elen, orders, zBins, xBins)

        extFluxX = extFlux[:, :, zBins//2, :]
        extFluxZ = extFlux[:, :, :, xBins//2]

        xFactor = 1e6
        yFactor = 1e4
        rect2d = [0.1, 0.1, 0.75, 0.8]
        figx, axx = create_fig(rect2d, 'azimuthal', scanAxis, axisLabel,
                               scanAxisFactor, maxPlotOrder)
        figz, axz = create_fig(rect2d, 'polar', scanAxis, axisLabel,
                               scanAxisFactor, maxPlotOrder)

        for iE in range(elen):
            for o, iaxx, iaxz in zip(range(maxPlotOrder+1-minOrder), axx, axz):
                iaxx.plot(screenPhis*xFactor, extFluxX[iE, o, :]*yFactor,
                          '-', lw=0.5, color=cmap(float(iE)/(elen-1)))
                iaxz.plot(screenThetas*xFactor, extFluxZ[iE, o, :]*yFactor,
                          '-', lw=0.5, color=cmap(float(iE)/(elen-1)))
                iaxx.set_xlim(screenPhis[0]*xFactor*0.99,
                              screenPhis[-1]*xFactor*0.99)
                iaxz.set_xlim(screenThetas[0]*xFactor*0.99,
                              screenThetas[-1]*xFactor*0.99)

        maxY = max(extFluxX.max(), extFluxZ.max()) * yFactor
        iaxx.set_ylim(0, maxY)
        iaxz.set_ylim(0, maxY)
        figx.savefig(fName + '-orders-x.png')
        figz.savefig(fName + '-orders-z.png')

        for iE in [0, elen//2, elen-1]:
#        for iE in [0, elen-1]:
            rect2d = [0.1, 0.1, 0.85, 0.8]
            figc, axc = create_fig(rect2d, 'polar', scanAxis, axisLabel,
                                   scanAxisFactor, maxPlotOrder,
                                   withEbar=False, textcolor='w')
            titax = len(axc)//2
            axc[titax].set_title(
                'Images of grating orders on a spherical screen' +
                ' at {0:.0f} eV'.format(scanAxis[iE]))
            axc[0].set_ylabel(u'azimuthal exit angle (µrad)', fontsize=14)
            vmax = extFlux[iE, :, :, :].max()
            for o, iaxc in zip(range(maxPlotOrder+1-minOrder), axc):
                a = extFlux[iE, o, :, :].T
                iaxc.imshow(a, aspect='auto', cmap=cmap, origin="lower",
                            vmin=0, vmax=vmax,
                            interpolation='none',
                            extent=[screenThetas[0]*xFactor*0.99,
                                    screenThetas[-1]*xFactor*0.99,
                                    screenPhis[0]*xFactor*0.99,
                                    screenPhis[-1]*xFactor*0.99])
                iaxc.set_xlim(screenThetas[0]*xFactor*0.99,
                              screenThetas[-1]*xFactor*0.99)
                iaxc.set_ylim(screenPhis[0]*xFactor*0.99,
                              screenPhis[-1]*xFactor*0.99)

            figc.savefig(fName + '-xz-{0}.png'.format(scanAxis[iE]))
    plt.show()


if __name__ == '__main__':
#    visualize_grating()
    get_efficiency()
#    visualize_efficiency()
