# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

substrate = rm.Material('Si', rho=2.33)
coating = rm.Material('Rh', rho=12.41, kind='mirror')
cMirror = rm.CoatedMirror(coating=coating, cThickness=300, 
                          substrate=substrate, surfaceRoughness=30,
                          substRoughness=30)

#E = np.logspace(1 + np.log10(3), 4 + np.log10(5), 501)
E = np.linspace(100, 29999, 1001)
theta = 4e-3
rs, rp = cMirror.get_amplitude(E, np.sin(theta))[0:2]
rs0, rp0 = coating.get_amplitude(E, np.sin(theta))[0:2]
#plt.semilogx(E, abs(rs)**2, 'r', E, abs(rp)**2, 'b')
refs, refp = plt.plot(E, abs(rs)**2, 'r', E, abs(rp)**2, 'b--')
refs0, refp0 = plt.plot(E, abs(rs0)**2, 'm', E, abs(rp0)**2, 'c--')
plt.legend([refs, refp, refs0, refp0],
           ['coated s-pol', 'coated p-pol', 'pure s-pol', 'pure p-pol'])
plt.gca().set_xlim(E[0], E[-1])
plt.show()
