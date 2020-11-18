# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "08 Jul 2016"
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

crystal, E = rm.CrystalSi(hkl=(1, 1, 1)), 8040
dtheta = np.linspace(-30, 90, 601)
dt = dtheta[1] - dtheta[0]
theta = crystal.get_Bragg_angle(E) + dtheta*1e-6
refl = np.abs(crystal.get_amplitude(E, np.sin(theta))[0])**2  # s-polarization
spline = UnivariateSpline(dtheta, refl-refl.max()/2, s=0)
r11, r12 = spline.roots()  # find the roots

rc = np.convolve(refl, refl, 'same') / (refl.sum()*dt) * dt
spline = UnivariateSpline(dtheta, rc-rc.max()/2, s=0)
r21, r22 = spline.roots()  # find the roots

plt.plot(dtheta, refl, 'r', label=u'one crystal\nFWHM = {0:.1f} µrad'.format(
    crystal.get_Darwin_width(E)*1e6))
plt.axvspan(r11, r12, facecolor='r', alpha=0.05)
plt.plot(dtheta, rc, 'b', label=u'two crystal (conv)'
         u'\nFWHM = {0:.1f} µrad'.format(r22-r21))
plt.gca().set_xlabel(u'$\\theta - \\theta_{B}$ (µrad)')
plt.gca().set_ylabel(r'reflectivity')
plt.axvspan(r21, r22, facecolor='b', alpha=0.05)
plt.legend(loc='upper right', fontsize=12)
plt.gca().set_xlim(dtheta[0], dtheta[-1])

text = u'Rocking curve of {0}{1[0]}{1[1]}{1[2]} at E={2:.0f} eV'.format(
    crystal.name, crystal.hkl, E)
plt.text(0.5, 1.02, text, transform=plt.gca().transAxes, size=15, ha='center')
plt.show()
