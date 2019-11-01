# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "1 Nov 2019"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

#mSi = rm.Material('Si', rho=2.33)
#mW = rm.Material('W', rho=19.3)
#mL = rm.Multilayer(mSi, 27, mW, 18, 40, mSi)

mSi = rm.Material('Si', rho=2.33)
mW = rm.Material('W', rho=19.3)
mB4C = rm.Material(['B', 'C'], [4, 1], rho=2.52)
mL = rm.Multilayer(mB4C, 16, mW, 8, 150, mSi)

theta = 1.5

plt.subplot(121)
E = np.linspace(9.5, 10.5, 1001) * 1e3  # eV
rs, rp = mL.get_amplitude(E, np.sin(np.deg2rad(theta)))[0:2]
plt.plot(E*1e-3, abs(rs)**2, '-')

plt.subplot(122)
E *= 2
rs, rp = mL.get_amplitude(E, np.sin(np.deg2rad(theta)))[0:2]
plt.plot(E*1e-3, abs(rs)**2, '-')

plt.show()
