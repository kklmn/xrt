# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

stripe = rm.Material(('Si', 'O'), quantities=(1, 2), rho=2.2)

E = np.logspace(1 + np.log10(3), 4 + np.log10(5), 501)
theta = 7e-3
rs, rp = stripe.get_amplitude(E, np.sin(theta))[0:2]

plt.semilogx(E, abs(rs)**2, 'r', E, abs(rp)**2, 'b')
plt.gca().set_xlim(E[0], E[-1])
plt.show()
