# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

matDiamond = rm.Material('C', rho=3.52)

E = np.logspace(2 + np.log10(3), 4 + np.log10(3), 501)
thickness = 0.06  # mm
mu = matDiamond.get_absorption_coefficient(E)  # in cm^-1
transm = np.exp(-mu * thickness * 0.1)

plt.semilogx(E, transm)
plt.gca().set_xlim(E[0], E[-1])
plt.show()
