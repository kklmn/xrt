# -*- coding: utf-8 -*-
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import matplotlib.pyplot as plt
import xrt.backends.raycing.sources as rs

source = rs.Undulator(eE=3.0, eI=0.5, eEpsilonX=0.263, eEpsilonZ=0.008,
                      betaX=9., betaZ=2., period=19.3, n=101, K=0.52,
                      eEspread=0*1e-3, distE='BW')

energy = np.linspace(1000, 31000, 1501)
theta = np.linspace(-1, 1, 21) * 30e-6
psi = np.linspace(-1, 1, 21) * 30e-6
Ks = np.linspace(0.3, 2.2, 20)
Kmax, gmin = 2.2, 4.2
gaps = np.log(Kmax/Ks) / np.pi * source.L0 + gmin
harmonics = range(1, 17)

powers = source.power_vs_K(energy, theta, psi, harmonics, Ks)

plt.plot(gaps, powers, 'o-')
ax = plt.gca()
ax.set_xlabel(u'magnet gap (mm)')
ax.set_ylabel(u'total power through (60 µrad)² (W)')

maxPower = powers.max()
for i, (power, K, gap) in enumerate(zip(powers, Ks, gaps)):
    ax.text(gap, power + maxPower*0.03 * (i % 2 * 2 - 1),
            '{0}{1:.1f}'.format('K=' if i == 0 else '', K), fontsize=9,
            ha='center', va='center')

plt.savefig('undulator_power_CoSAXS.png')
plt.show()
