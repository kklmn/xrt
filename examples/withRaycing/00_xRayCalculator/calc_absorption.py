# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "5 Jun 2018"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join(*['..']*3))  # analysis:ignore
import xrt.backends.raycing.materials as rm

E = np.logspace(1, 4 + np.log10(5), 501)
for table in ['Chantler', 'Chantler total']:
    mat = rm.Material('Be', rho=1.848, table=table)
    mu = mat.get_absorption_coefficient(E)
    plt.loglog(E, mu)

plt.gca().set_xlim(E[0], E[-1])
plt.show()
