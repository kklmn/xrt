# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

mSi = rm.Material('Si', rho=2.33)
mW = rm.Material('W', rho=19.3)
mL = rm.Multilayer(mSi, 27, mW, 18, 40, mSi)

E = 10000
theta = np.linspace(0, 2.0, 1001)  # degrees
rs, rp = mL.get_amplitude(E, np.sin(np.deg2rad(theta)))[0:2]

plt.plot(theta, abs(rs)**2, 'r', theta, abs(rp)**2, 'b')
plt.show()
