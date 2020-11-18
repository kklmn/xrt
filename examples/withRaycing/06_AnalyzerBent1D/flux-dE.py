# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "14 October 2014"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
import xrt.backends.raycing.materials as rm

cases = []
band = 'narrow'
#band = '8e-4'

crystalMaterial = 'Si'
if crystalMaterial == 'Si':
    d111 = 3.1354161
elif crystalMaterial == 'Ge':
    d111 = 3.2662725
else:
    raise
crystal = rm.CrystalDiamond((4, 4, 4), d111/4, elements=crystalMaterial)
thetaDegrees = [40, 60, 80]


class Case:
    I0 = 1e13

    def __init__(self, name, style, thetaD, dxPrime, dzPrime, N, I, dE,
                 Nband8e_4=0, Iband8e_4=0):
        self.name = name
        self.style = style
        self.thetaD = thetaD
        self.dxPrime = dxPrime
        self.dzPrime = dzPrime
        self.N = N
        self.I = I  # 7 lines band
        self.dE = dE
        self.Nband8e_4 = Nband8e_4  # 8e-4 band
        self.Iband8e_4 = Iband8e_4
        if thetaD == 40:
            self.color = 'r'
            self.alpha = 1
        elif thetaD == 60:
            self.color = 'b'
            self.alpha = 0.5
        elif thetaD == 80:
            self.color = 'g'
            self.alpha = 0.25
        else:
            raise
        cases.append(self)
        self.get_flux()

    def get_flux(self):
        if band == 'narrow':
            I, N = self.I, self.N
        elif band == '8e-4':
            I, N = self.Iband8e_4, self.Nband8e_4
        else:
            raise
        self.flux = (self.I0 * I / N * self.dxPrime * self.dzPrime / (4*np.pi))

#  SivonHamosDiced40-det_E-7lin.png

thetaD = 40
dxP, dzP = 0.44, 0.257115043875
Case('von Hamos, diced 5 mm', 'v', thetaD, dxP, dzP, 16e8, 30337.3, 4.957,
     4e8, 7589.34)
Case('von Hamos, diced 1 mm', '^', thetaD, dxP, dzP, 16e8, 29142.5, 2.745,
     4e8, 7251.19)
Case('von Hamos, not diced', 'D', thetaD, dxP, dzP, 16e8, 30493.5, 1.815,
     4e8, 7645.88)
dxP, dzP = 0.162100707902, 0.104196326561
Case('Johansson', '*', thetaD, dxP, dzP, 256e6, 790259, 0.286,
     256e6, 408889)
Case('Johann', 'o', thetaD, dxP, dzP, 256e6, 261728, 5.722,
     256e6, 383036)
dxP, dzP = 0.0707066370655, 0.0413175911167
Case('Johann as von Hamos', 's', thetaD, dxP, dzP, 400e6, 30694.2, 0.473,
     4e8, 30707)

thetaD = 60
dxP, dzP = 0.44, 0.346410161514
Case('von Hamos, diced 5 mm', 'v', thetaD, dxP, dzP, 16e8, 43159.9, 2.011,
     4e8, 10701)
Case('von Hamos, diced 1 mm', '^', thetaD, dxP, dzP, 16e8, 41099.2, 1.128,
     4e8, 10428.9)
Case('von Hamos, not diced', 'D', thetaD, dxP, dzP, 16e8, 43410.3, 0.932,
     4e8, 10854.7)
dxP, dzP = 0.117802177587, 0.102019678411
Case('Johansson', '*', thetaD, dxP, dzP, 256e6, 1.74956e6, 0.196,
     256e6, 723768)
Case('Johann', 'o', thetaD, dxP, dzP, 256e6, 718938, 1.655,
     256e6, 798442)
dxP, dzP = 0.0952627944163, 0.075
Case('Johann as von Hamos', 's', thetaD, dxP, dzP, 400e6, 43530.6, 0.236,
     4e8, 43480.3)

thetaD = 80
dxP, dzP = 0.44, 0.393923101205
Case('von Hamos, diced 5 mm', 'v', thetaD, dxP, dzP, 16e8, 141777, 0.394,
     7.5e8, 66463.3)
Case('von Hamos, diced 1 mm', '^', thetaD, dxP, dzP, 16e8, 135873, 0.275,
     4e8, 34171.7)
Case('von Hamos, not diced', 'D', thetaD, dxP, dzP, 16e8, 143767, 0.257,
     4e8, 35448.9)
dxP, dzP = 0.102200009788, 0.100647361997
Case('Johansson', '*', thetaD, dxP, dzP, 256e6, 2.90378e6, 0.154,
     256e6, 1.10778e6)
Case('Johann', 'o', thetaD, dxP, dzP, 256e6, 2.05992e6, 0.365,
     256e6, 1.12375e6)
dxP, dzP = 0.108328852831, 0.0969846310393
Case('Johann as von Hamos', 's', thetaD, dxP, dzP, 400e6, 143235, 0.086,
     4e8, 143160)

block = len(cases) // 3


def plot_res_eff():
    fig = plt.figure(figsize=(8, 6), dpi=72)
    rect2d = [0.1, 0.1, 0.5, 0.74]
    ax1 = fig.add_axes(rect2d, aspect='auto')
#    ax2 = ax1.twinx()

    if band == 'narrow':
        bn = r'The source energy band equals $\pm 3\cdot\delta E$.'
    elif band == '8e-4':
        bn = r'The source energy band equals $8\cdot 10^{-4}$.'
    else:
        raise
    title = 'Resolution-efficiency chart of 1D bent Si444 crystal analysers\n'\
        + 'at 10$^{13}$ ph/s incoming flux'\
        + u' and 100×100 µm² source size.\n' + bn
    fig.text(0.5, 0.85, title, transform=fig.transFigure, size=14, color='k',
             ha='center')

    ax1.set_xlabel(r'resolution $\delta E$ (eV)', fontsize=14)
    ax1.set_ylabel(u'flux at detector (ph/s)', fontsize=14)
#    ax2.set_ylabel(u'vertical size FWHM (µm)', color='b')
#    fig.subplots_adjust(right=0.88, bottom=0.12)
    lines = []
    labels = []
    for case in cases:
        l, = ax1.loglog(case.dE, case.flux, case.color+case.style,
                        alpha=case.alpha, ms=10)
        lines.append(l)
        labels.append(case.name)

    for curve in range(block):
        x = [case.dE for case in cases[curve::block]]
        y = [case.flux for case in cases[curve::block]]
        ax1.loglog(x, y, 'gray', lw=0.5)

    ax1.set_xlim(0.07, None)
    ax1.set_ylim(1e5, 1.1e8)

    E0s = []
    for thetaDegree in thetaDegrees:
        theta = np.radians(thetaDegree)
        sinTheta = np.sin(theta)
        E0raw = rm.ch / (2 * crystal.d * sinTheta)
        dTheta = crystal.get_dtheta_symmetric_Bragg(E0raw)
        E0 = rm.ch / (2 * crystal.d * np.sin(theta + dTheta))
        E0s.append(E0)
    labelThetas = [r'{0}$^\circ,\ E\approx{1}$ eV'.format(
        t, int(round(e))) for t, e in zip(thetaDegrees, E0s)]
    legBragg = ax1.legend(lines[::block], labelThetas,
                          title='Bragg angle, Si444', numpoints=1, loc=(1, 0))
    for line in legBragg.get_lines():
        line._legmarker.set_marker([(4, 3), (-4, 3), (-4, -3), (4, -3)])
#        line._legmarker.set_marker('.')
#        raise

    leg = ax1.legend(lines[:block], labels[:block], title='crystal type',
                     numpoints=1, loc=(1, 0.5))
    for line in leg.get_lines():
        line._legmarker.set_markerfacecolor('gray')
        line._legmarker.set_markeredgecolor('gray')
    ax1.add_artist(legBragg)

    if band == 'narrow':
        fname = 'ResolutionEfficiency1D-narrowBand'
    elif band == '8e-4':
        fname = 'ResolutionEfficiency1D-8e-4Band'
    else:
        raise
    fig.savefig(fname+'.png')
    fig.savefig(fname+'.pdf')


if __name__ == '__main__':
    plot_res_eff()
    plt.show()
