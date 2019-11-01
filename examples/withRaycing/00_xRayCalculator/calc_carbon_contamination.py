# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "14 Mar 2019"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

mSi = rm.Material('Si', rho=2.33)
mAu = rm.Material('Au', rho=19.3)
mC = rm.Material('C', rho=2.26)
mAuCont = rm.Multilayer(tLayer=mC, tThickness=10,  # in Å
                        bLayer=mAu, bThickness=400, substrate=mSi,
                        nPairs=1, idThickness=2)

E = np.linspace(10, 120, 110)  # degrees
theta = np.radians(7) * np.ones_like(E)
rs, rp = mAu.get_amplitude(E, np.sin(theta))[0:2]
plt.plot(E, abs(rs)**2, label='s-pol Au')
plt.plot(E, abs(rp)**2, label='p-pol Au')

rs, rp = mAuCont.get_amplitude(E, np.sin(theta))[0:2]
plt.plot(E, abs(rs)**2, label='s-pol Au with 4 nm C')
plt.plot(E, abs(rp)**2, label='p-pol Au with 4 nm C')

plt.gca().set_title('Reflectivity of clean and contaminated gold')
plt.gca().legend()

plt.show()
