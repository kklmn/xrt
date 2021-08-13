# -*- coding: utf-8 -*-
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
#import matplotlib as mpl
import copy
import numpy as np
import matplotlib.pyplot as plt
try:
    import xlwt
except ImportError:
    xlwt = None

import xrt.backends.raycing.sources as rs
from xrt.backends.raycing.physconsts import SIE0

withUndulator = True
#withUndulator = False
#withUrgentUndulator = True
withUrgentUndulator = False
#withSRWUndulator = True
withSRWUndulator = False


def run(case):
    eMax = 200100.
    eN = 201
    thetaMax, psiMax = 500e-6, 500e-6
    if case == 'Balder':
        Kmax = 8.446
#        Kmax = 3
        thetaMax, psiMax = 200e-6, 50e-6
#        thetaMax, psiMax = 1130e-6/2, 1180e-6/2  # asked by Magnus
        eMax, eN = 400100, 401
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
    elif case == 'Veritas' or case == 'Hippie':
        thetaMax, psiMax = 100e-6, 50e-6
#        thetaMax, psiMax = 100e-6, 200e-6  # asked by Magnus
#        thetaMax, psiMax = 500e-6, 500e-6  # asked by Magnus
        kwargs = dict(name='U48', eE=3.0, eI=0.5,
                      eEpsilonX=0.263, eEpsilonZ=0.008, betaX=9., betaZ=2.,
                      eMax=eMax,
                      xPrimeMax=thetaMax*1e3, zPrimeMax=psiMax*1e3, distE='BW')
        if case == 'Veritas':
            kwargs['period'] = 48.
            kwargs['n'] = 81
            kwargs['K'] = 4.51
        if case == 'Hippie':
            kwargs['period'] = 53.
            kwargs['n'] = 73
            kwargs['K'] = 5.28

    sourceW = rs.Wiggler(**kwargs)
    energy = np.linspace(100., eMax, eN)
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
        sourceU = rs.UndulatorUrgent(**ukwargs)
        I0U = sourceU.intensities_on_mesh()[0]
        fluxUU = I0U.sum(axis=(1, 2)) * dtheta * dpsi * 4e6
        fluxUU[fluxUU <= 0] = 1
        fluxUU[np.isnan(fluxUU)] = 1

    if withSRWUndulator:
        import pickle
        with open('c:\Ray-tracing\srw\SRWres.pickle', 'rb') as f:
            energySRW, thetaSRW, psiSRW, I0SRW = pickle.load(f)[0:4]
        dtheta = thetaSRW[1] - thetaSRW[0]
        dpsi = psiSRW[1] - psiSRW[0]
        fluxSRWU = I0SRW.sum(axis=(1, 2)) * dtheta * dpsi

    if withUndulator:
#        kwargs['targetOpenCL'] = None
#        kwargs['taper'] = 0, 4.2
#        kwargs['gp'] = 1e-4  # needed if does not converge
        sourceU = rs.Undulator(**kwargs)
        I0U = sourceU.intensities_on_mesh(energy, theta, psi)[0]
        fluxU = I0U.sum(axis=(1, 2)) * dtheta * dpsi

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(case, fontsize=14)
    rect2d1 = [0.12, 0.12, 0.85, 0.8]
    ax1 = fig.add_axes(rect2d1, aspect='auto')
    rect2d2 = [0.22, 0.19, 0.33, 0.5]
    ax2 = fig.add_axes(rect2d2, aspect='auto')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.patch.set_visible(False)
    for ax, lw in zip([ax1, ax2], [2, 2]):
    #    plot = ax.plot
        plot = ax.semilogy
    #    plot = ax.loglog
        plot(energy/1000., fluxW, '-', lw=lw, alpha=0.7,
             label='xrt, as wiggler')
        if withUndulator:
            plot(energy/1000., fluxU, '-', lw=lw, alpha=0.7,
                 label='xrt, as undulator')
            if withUrgentUndulator:
                plot(energy/1000., fluxUU, '-', lw=lw, alpha=0.7,
                     label='Urgent')
            if withSRWUndulator:
                plot(energySRW/1000., fluxSRWU, '-', lw=lw, alpha=0.7,
                     label='SRW (zero emittance)')
            if case == 'BioMAX & NanoMAX':  # Spectra results
    #            fnames = ['bionano2.dc0', 'bionano3.dc0']
    #            labels = ['Spectra, accuracy {0}'.format(i) for i in [2, 3]]
                fnames = ['bionano3.dc0']
                labels = ['Spectra']
                for fname, label in zip(fnames, labels):
                    e, f = np.loadtxt(fname, skiprows=10, usecols=(0, 1),
                                      unpack=True)
                    plot(e/1000, f, lw=lw, alpha=0.7, label=label)
    ax1.set_xlabel(u'energy (keV)')
    ax1.set_ylabel(u'flux through {0:.0f}×{1:.0f} µrad² (ph/s/0.1%BW)'.format(
                   theta[-1]*2e6, psi[-1]*2e6))
    ax1.set_xlim(1, eMax/1e3)
    if case == 'Veritas' or case == 'Hippie':
        ax1.set_ylim(1e1, None)
    else:
        ax1.set_ylim(1e3, None)
    ax2.set_xlim(1, 30)
    ax2.set_ylim(1e12, 2e15)
    if withUndulator:
        ax1.legend(loc='upper right')

    plt.savefig(u'flux_{0}_{1:.0f}×{2:.0f}µrad².png'.format(
        case, theta[-1]*2e6, psi[-1]*2e6))

    if xlwt is not None:
        wb = xlwt.Workbook()
        ws = wb.add_sheet('flux')
        ws.write(0, 0, u'energy (eV)')
        ws.write(
            0, 1, u'fluxW through {0:.0f}×{1:.0f} µrad² (ph/s/0.1%BW)'.format(
                theta[-1]*2e6, psi[-1]*2e6))
        if withUndulator:
            ws.write(
                0, 2,
                u'fluxU through {0:.0f}×{1:.0f} µrad² (ph/s/0.1%BW)'.format(
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
    # run('Balder')
    run('BioMAX & NanoMAX')
    # run('Veritas')
    # run('Hippie')
