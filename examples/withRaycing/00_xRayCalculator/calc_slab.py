# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

layerW = rm.Material('W', kind='thin mirror', rho=19.3, t=2.5e-6)

E = 10000
theta = np.linspace(0, 10, 501)  # degrees
rs, rp = layerW.get_amplitude(E, np.sin(np.deg2rad(theta)))[0:2]

plt.semilogy(theta, abs(rs)**2, 'r', theta, abs(rp)**2, 'b')
plt.show()
