# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

crystal = rm.CrystalSi(hkl=(1, 1, 1))

E = 9000
dtheta = np.linspace(-20, 80, 501)
theta = crystal.get_Bragg_angle(E) + dtheta*1e-6
curS, curP = crystal.get_amplitude(E, np.sin(theta))
print(crystal.get_a())
print(crystal.get_F_chi(E, 0.5/crystal.d))
print(u'Darwin width at E={0:.0f} eV is {1:.5f} µrad for s-polarization'.
      format(E, crystal.get_Darwin_width(E) * 1e6))

plt.plot(dtheta, abs(curS)**2, 'r', dtheta, abs(curP)**2, 'b')
plt.gca().set_xlabel(u'$\\theta - \\theta_{B}$ (µrad)')
plt.gca().set_ylabel(r'reflectivity')
plt.show()
