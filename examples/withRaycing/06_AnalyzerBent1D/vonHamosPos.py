# -*- coding: utf-8 -*-
"""
"""
__author__ = "Konstantin Klementiev"
__date__ = "14 October 2014"
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

Rs = 250
thetaDegs = np.linspace(40, 80, 5)
colors = ['r', 'm', 'b', 'c', 'g']
xtalDx, xtalDy = 25, 12
detDx, detDy = 32, 8


def get_clasical_xtal_det(thetaDeg):
    theta = np.radians(thetaDeg)
    tanTheta = np.tan(theta)
    yDet = 2 * Rs / tanTheta
    return yDet


def get_xtal_det(thetaDeg):
    theta = np.radians(thetaDeg)
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sin2Theta = np.sin(2 * theta)
#    cos2Theta = np.cos(2 * theta)
    p = Rs / sinTheta
    yDet = p * 2 * cosTheta**2
    zDet = p * sin2Theta
    return p, yDet, zDet


def plot_classical_pos():
    fig = plt.figure(figsize=(6, 6), dpi=72)
    rect2d = [0.15, 0.1, 0.8, 0.8]
    ax1 = fig.add_axes(rect2d, aspect='auto')
#    ax2 = ax1.twinx()
    title = 'Crystal and detector positions for \n' +\
        'von Hamos spectrometer with parallel translations'
    fig.text(0.5, 0.91, title, transform=fig.transFigure, size=14, color='k',
             ha='center')

    ax1.set_xlabel(u'y (mm)')
    ax1.set_ylabel(u'z (mm)')
    ax1.set_xlim(-10, 650-10)
    ax1.set_ylim(-325, 650-325)

    source = plt.Circle((0, 0), xtalDy, fc='r', clip_on=True, lw=1, alpha=0.5)
    ax1.add_patch(source)

    detectors = []
    for thetaDeg, color in zip(thetaDegs, colors):
        ax1.plot([0, 650], [0, 0], lw=0.5)
        yDet = get_clasical_xtal_det(thetaDeg)

        ax1.plot([0, yDet/2, yDet], [0, -Rs, 0], ':', color=color, lw=1)

        bars = [(-xtalDx, -xtalDy), (xtalDx, -xtalDy),
                (xtalDx, 0), (-xtalDx, 0)]
#        xtal = plt.Circle((yDet/2, -Rs), xtalDy, color=color, clip_on=True)
#        ax1.add_patch(xtal)
        poly = [(yDet/2 + x, -Rs + y) for (x, y) in bars]
        xtal = mpl.patches.Polygon(poly, closed=True, fc=color,
                                   lw=0, alpha=0.5)
        ax1.add_patch(xtal)

        bars = [(-detDx, 0), (detDx, 0), (detDx, detDy), (-detDx, detDy)]
#        detector = plt.Circle((yDet, 0), xtalDy, color=color, clip_on=True)
#        ax1.add_patch(detector)
        poly = [(yDet + x, y) for (x, y) in bars]
        detector = mpl.patches.Polygon(poly, closed=True, fc=color,
                                       lw=0, alpha=0.5)
        ax1.add_patch(detector)
        detectors.append(detector)

    legBragg = ax1.legend(detectors,
                          [r'{0}$^\circ$'.format(a) for a in thetaDegs],
                          title='Bragg angle', numpoints=1, loc='upper left')
    for line in legBragg.get_lines():
        line._legmarker.set_marker([(3, 1), (-3, 1), (-3, -1), (3, -1)])
#        line._legmarker.set_marker('.')

    ax1.text(0.95, 0.95, '(a)', transform=ax1.transAxes, size=20,
             ha='right', va='top')
    fig.savefig('vonHamosPositionsClassic.png')


def plot_pos():
    fig = plt.figure(figsize=(6, 6), dpi=72)
    rect2d = [0.15, 0.1, 0.8, 0.8]
    ax1 = fig.add_axes(rect2d, aspect='auto')
#    ax2 = ax1.twinx()
    title = 'Crystal and detector positions for \n' +\
        'von Hamos spectrometer with fixed escape direction'
    fig.text(0.5, 0.91, title, transform=fig.transFigure, size=14, color='k',
             ha='center')

    ax1.set_xlabel(u'y (mm)')
    ax1.set_ylabel(u'z (mm)')
    ax1.set_xlim(-100, 650-100)
    ax1.set_ylim(-100, 650-100)

    source = plt.Circle((0, 0), xtalDy, fc='r', clip_on=True, lw=1, alpha=0.5)
    ax1.add_patch(source)

    detectors = []
    for thetaDeg, color in zip(thetaDegs, colors):
        ax1.plot([0, 650], [0, 0], lw=0.5)
        p, yDet, zDet = get_xtal_det(thetaDeg)

        ax1.plot([0, p, yDet], [0, 0, zDet], ':', color=color, lw=1)
        c = np.cos(np.radians(thetaDeg))
        s = np.sin(np.radians(thetaDeg))

        bars = [(-xtalDx, -xtalDy), (xtalDx, -xtalDy),
                (xtalDx, 0), (-xtalDx, 0)]
#        xtal = plt.Circle((p, 0), xtalDy, color=color, clip_on=True)
#        ax1.add_patch(xtal)
        poly = [(p + c*x - s*y, s*x + c*y) for (x, y) in bars]
        xtal = mpl.patches.Polygon(poly, closed=True, fc=color,
                                   lw=0, alpha=0.5)
        ax1.add_patch(xtal)

        bars = [(-detDx, 0), (detDx, 0), (detDx, detDy), (-detDx, detDy)]
#        detector = plt.Circle((yDet, zDet), xtalDy, color=color, clip_on=True)
#        ax1.add_patch(detector)
        poly = [(yDet + c*x - s*y, zDet + s*x + c*y) for (x, y) in bars]
        detector = mpl.patches.Polygon(poly, closed=True, fc=color,
                                       lw=0, alpha=0.5)
        ax1.add_patch(detector)
        detectors.append(detector)

    legBragg = ax1.legend(detectors,
                          [r'{0}$^\circ$'.format(a) for a in thetaDegs],
                          title='Bragg angle', numpoints=1, loc='upper left')
    for line in legBragg.get_lines():
        line._legmarker.set_marker([(3, 1), (-3, 1), (-3, -1), (3, -1)])
#        line._legmarker.set_marker('.')

    ax1.text(0.95, 0.95, '(b)', transform=ax1.transAxes, size=20,
             ha='right', va='top')
    fig.savefig('vonHamosPositionsFixedEscape.png')


if __name__ == '__main__':
    plot_classical_pos()
    plot_pos()
    plt.show()
