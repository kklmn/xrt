# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Feb 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

crystal = rm.CrystalSi(hkl=(1, 1, 1), tK=80)

E = 9000
alpha = np.linspace(-12, 90, 1001)  # degrees
thetaB = np.degrees(crystal.get_Bragg_angle(E))
dtheta0 = np.degrees(np.abs(crystal.get_dtheta_regular(E, np.radians(alpha))))
plt.semilogy(alpha+thetaB, dtheta0, '-r', label=r'without curvature')
dtheta1 = np.degrees(np.abs(crystal.get_dtheta(E, np.radians(alpha))))
plt.semilogy(alpha+thetaB, dtheta1, '-b', label=r'with curvature')
plt.gca().set_xlabel(r'$\theta_{B}-\alpha$ (deg)')
plt.gca().set_ylabel(r'$\delta\theta$ (deg)')
plt.gca().set_xlim(0, 90)
plt.gca().set_ylim(6e-5, 1)
plt.gcf().suptitle('Deviation from the Bragg angle, Si(111), {0:.0f}keV'.
                   format(E/1000), fontsize=16)
plt.legend()
plt.show()
