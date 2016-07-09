# -*- coding: utf-8 -*-
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
#import matplotlib as mpl
import copy
import numpy as np
import matplotlib.pyplot as plt
import xlwt

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
from xrt.backends.raycing.physconsts import SIE0

withUndulator = True
withUrgentUndulator = False


def run(case):
    beamLine = raycing.BeamLine(azimuth=0, height=0)

    eMax = 200100.
    thetaMax, psiMax = 500e-6, 500e-6
    if case == 'Balder':
        Kmax = 8.446
        thetaMax, psiMax = 200e-6, 50e-6
        kwargs = dict(name='SoleilW50', eE=3.0, eI=0.5,
                      eEpsilonX=0.263, eEpsilonZ=0.008, betaX=9., betaZ=2.,
                      period=50., n=39, K=Kmax, eMax=eMax,
                      xPrimeMax=thetaMax*1e3, zPrimeMax=psiMax*1e3, distE='BW')
    elif case == 'BioMAX & NanoMAX':
        Kmax = 1.92
        thetaMax, psiMax = 100e-6, 50e-6
        kwargs = dict(name='IVU18.5', eE=3.0, eI=0.5,
                      eEpsilonX=0.263, eEpsilonZ=0.008, betaX=9., betaZ=2.,
                      period=18.5, n=108, K=Kmax, eMax=eMax,
                      xPrimeMax=thetaMax*1e3, zPrimeMax=psiMax*1e3, distE='BW')
    elif case == 'Veritas & Hippie':
        Kmax = 4.5
#        thetaMax, psiMax = 100e-6, 50e-6
#        thetaMax, psiMax = 500e-6, 500e-6
        thetaMax, psiMax = 100e-6, 200e-6
        kwargs = dict(name='IVU18.5', eE=3.0, eI=0.5,
                      eEpsilonX=0.263, eEpsilonZ=0.008, betaX=9., betaZ=2.,
                      period=48., n=81, K=Kmax, eMax=eMax,
                      xPrimeMax=thetaMax*1e3, zPrimeMax=psiMax*1e3, distE='BW')

    sourceW = rs.Wiggler(beamLine, **kwargs)
    energy = np.linspace(100., eMax, 201)
    theta = np.linspace(-1, 1, 101) * thetaMax
    psi = np.linspace(-1, 1, 101) * psiMax
#    theta = np.linspace(-1, 1, 15) * thetaMax
#    psi = np.linspace(-1, 1, 15) * psiMax
    dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
    I0W = sourceW.intensities_on_mesh(energy, theta, psi)[0]
    fluxW = I0W.sum(axis=(1, 2)) * dtheta * dpsi
    dE = energy[1] - energy[0]
    power = fluxW.sum()*dE*SIE0*1e3
    print('total power = {} W'.format(power))
    cumpower = np.cumsum(fluxW)*dE*SIE0*1e3 / power
    ind = np.argwhere(cumpower > 0.5)[0]
    y1, y2 = cumpower[ind-1], cumpower[ind]
    x1, x2 = energy[ind-1], energy[ind]
    Ec = (0.5*(x2-x1) - (y1*x2-y2*x1)) / (y2-y1)
    print('Ec = {0} eV'.format(Ec))

    if withUrgentUndulator:
        ukwargs = copy.copy(kwargs)
        del(ukwargs['distE'])
        del(ukwargs['betaX'])
        del(ukwargs['betaZ'])
        ukwargs['eSigmaX'] = (kwargs['eEpsilonX']*kwargs['betaX']*1e3)**0.5
        ukwargs['eSigmaZ'] = (kwargs['eEpsilonZ']*kwargs['betaZ']*1e3)**0.5
        ukwargs['eMin'] = energy[0]
        ukwargs['eMax'] = energy[-1]
        ukwargs['eN'] = len(energy)-1
        ukwargs['nx'] = len(theta)//2
        ukwargs['nz'] = len(psi)//2
        ukwargs['icalc'] = 3
        sourceU = rs.UndulatorUrgent(beamLine, **ukwargs)
        I0U = sourceU.intensities_on_mesh()[0]
        fluxUU = I0U.sum(axis=(1, 2)) * dtheta * dpsi * 4e6

    if withUndulator:
#        kwargs['targetOpenCL'] = None
#        kwargs['taper'] = 0, 4.2
#        kwargs['gIntervals'] = 2
        sourceU = rs.Undulator(beamLine, **kwargs)
        I0U = sourceU.intensities_on_mesh(energy, theta, psi)[0]
        fluxU = I0U.sum(axis=(1, 2)) * dtheta * dpsi

#    plot =plt.plot
    plot = plt.semilogy
#    plot = plt.loglog
    if not withUndulator:
        plot(energy/1000., fluxW, '-', lw=2, alpha=0.7)
    else:
        plot(energy/1000., fluxW, '-', lw=2, alpha=0.7,
             label='xrt, as wiggler')
        plot(energy/1000., fluxU, '-', lw=2, alpha=0.7,
             label='xrt, as undulator')
        if withUrgentUndulator:
            plot(energy/1000., fluxUU, '-', lw=2, alpha=0.7, label='Urgent')
        if case == 'BioMAX & NanoMAX':  # Spectra results
#            fnames = ['bionano2.dc0', 'bionano3.dc0']
#            labels = ['Spectra, accuracy {0}'.format(i) for i in [2, 3]]
            fnames = ['bionano3.dc0']
            labels = ['Spectra']
            for fname, label in zip(fnames, labels):
                e, f = np.loadtxt(fname, skiprows=10, usecols=(0, 1),
                                  unpack=True)
                plot(e/1000, f, lw=2, alpha=0.7, label=label)
    plt.gcf().suptitle(case, fontsize=14)
    ax = plt.gca()
    ax.set_xlabel(u'energy (keV)')
    ax.set_ylabel(u'flux through {0:.0f}×{1:.0f} µrad² (ph/s/0.1%BW)'.format(
                  theta[-1]*2e6, psi[-1]*2e6))
    ax.set_xlim(1, eMax/1e3)
    ax.set_ylim(1e3, None)
    if withUndulator:
        ax.legend(loc='lower left')

    plt.savefig(u'flux_{0}_{1:.0f}×{2:.0f}µrad².png'.format(
        case, theta[-1]*2e6, psi[-1]*2e6))

    wb = xlwt.Workbook()
    ws = wb.add_sheet('flux')
    ws.write(0, 0, u'energy (eV)')
    ws.write(0, 1, u'fluxW through {0:.0f}×{1:.0f} µrad² (ph/s/0.1%BW)'.format(
             theta[-1]*2e6, psi[-1]*2e6))
    ws.write(0, 2, u'fluxU through {0:.0f}×{1:.0f} µrad² (ph/s/0.1%BW)'.format(
             theta[-1]*2e6, psi[-1]*2e6))

    for i, e in enumerate(energy):
        ws.write(i+1, 0, e)
        ws.write(i+1, 1, fluxW[i])
        if withUndulator:
            ws.write(i+1, 2, fluxW[i])

    wb.save(u'flux_{0}_{1:.0f}×{2:.0f}µrad².xls'.format(
        case, theta[-1]*2e6, psi[-1]*2e6))

    plt.show()

if __name__ == '__main__':
#    run('Balder')
    run('BioMAX & NanoMAX')
#    run('Veritas & Hippie')
