# -*- coding: utf-8 -*-
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore

#import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import xlwt

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.apertures as ra
#import xrt.backends.raycing.oes as roe
#import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
#import xrt.backends.raycing.screens as rsc


#stripeSi = rm.Material('Si', rho=2.33)
#stripeSiO2 = rm.Material(('Si', 'O'), quantities=(1, 2), rho=2.2)
stripePt = rm.Material('Pt', rho=21.45)
#filterDiamond = rm.Material('C', rho=3.52, kind='plate')
#si = rm.CrystalSi(hkl=(1, 1, 1), tK=-171+273.15)

eMax = 200  # keV


def run(case):
    myBalder = raycing.BeamLine(azimuth=0, height=0)
    kwargs = dict(
        name='SoleilW50', center=(0, 0, 0),
        period=50., K=8.446, n=39, eE=3., eI=0.5,
        eSigmaX=48.66, eSigmaZ=6.197, eEpsilonX=0.263, eEpsilonZ=0.008,
        eMin=50, eMax=eMax*1e3+50, xPrimeMax=0.2, zPrimeMax=0.05,
        eN=2000, nx=20, nz=10)
    kwargs['distE'] = 'BW'
    source = rs.Wiggler(myBalder, **kwargs)
    E = np.mgrid[source.E_min:(source.E_max + 0.5*source.dE):source.dE]

    I0 = source.intensities_on_mesh()[0]
    flux = I0.sum(axis=(1, 2)) * source.dTheta * source.dPsi

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(u'Integrated {0}'.format(case) +
                 u' beam flux into 0.4$\\times$ 0.1 mrad$^2$',
                 fontsize=14)
#    fig.subplots_adjust(right=0.88)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'energy (keV)')
    if case == 'monochromatic':
        ax.set_ylabel(r'flux (ph/s/(Si111 DCM bw)')
    else:
        ax.set_ylabel(r'flux (ph/s/0.1%bw)')

    if case == 'monochromatic':
        refl = stripePt.get_amplitude(E, np.sin(1e-3))
        ras = refl[0]
        ax2 = ax.twinx()
        rI = (abs(ras)**2)**2
        ax2.plot(E*1e-3, rI, '-b', lw=2)
        ax2.set_ylabel('reflectivity of two Pt mirrors at 1 mrad', color='b')
        ax2.set_ylim(0, 1)
        fluxRes = flux*2e-4*1e3*rI
    else:
        fluxRes = flux
    ax.plot(E*1e-3, fluxRes, '-r', lw=2, label='wiggler')

    ax.set_ylim(0, None)
    ax.set_xlim(0, eMax)

    saveName = 'fluxBalder-{0}'.format(case)
    fig.savefig(saveName + '.png')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('flux')
    ws.write(0, 0, "energy (eV)")
    if case == 'monochromatic':
        ws.write(0, 1, "flux (ph/s/(Si111 DCM bw)")
    else:
        ws.write(0, 1, "flux (ph/s/0.1%bw)")

    for ie, (e, f) in enumerate(zip(E, fluxRes)):
        ws.write(ie+1, 0, e)
        ws.write(ie+1, 1, f)

    wb.save(saveName + '.xls')

    plt.show()

if __name__ == '__main__':
    run('white')
#    run('monochromatic')
    print('finished')
