# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import xrt.backends.raycing.sources as rs

source = rs.Undulator(eE=3.0, eI=0.5, eEpsilonX=0.263, eEpsilonZ=0.008,
                      betaX=9.539, betaZ=1.982, period=18.5, n=108, K=0.52,
                      eEspread=0*8e-4, distE='BW', gIntervals=2)
print("please wait...")

energy = np.linspace(2000, 30000, 5601)
theta = np.linspace(-1, 1, 11) * 30e-6  # 60 µrad opening
psi = np.linspace(-1, 1, 11) * 30e-6  # 60 µrad opening
xlims, ylims = (3, 24), (1e14, 5e15)
Ks = np.linspace(0.3, 2.2, 20)
Kmax, gmin = 1.92, 4.2  # needed to calculate gap(K)
gaps = np.log(Kmax/Ks) / np.pi * source.L0 + gmin
harmonics = [1, 2, 3, 4, 5, 7, 9]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

tunesE, tunesF = source.tuning_curves(energy, theta, psi, harmonics, Ks)

# plot:
for tuneE, tuneF, harmonic, color in zip(tunesE, tunesF, harmonics, colors):
    plt.loglog(tuneE, tuneF, 'o-', label='{0}'.format(harmonic),
               color=color)
# format the graph:
ax = plt.gca()
ax.set_xlabel(u'energy (keV)')
ax.set_ylabel(u'flux through (60 µrad)² (ph/s/0.1% bw)')
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
plt.grid(which='both')
ax.legend(loc='upper right', title='harmonics')

# labels of K and gap:
for tuneE, tuneF, harmonic in zip(tunesE, tunesF, harmonics):
    if harmonic == 3:
        ax.text(tuneE[-1], tuneF[-1]*1.05, 'K=', fontsize=8,
                ha='right', va='bottom')
        ax.text(tuneE[-1]/1.05, tuneF[-1]/1.2, 'gap=\n(mm)', fontsize=8,
                color='b', ha='right', va='center')
    for x, y, K, gap in zip(tuneE, tuneF, Ks, gaps):
        if xlims[0] < x < xlims[1] and ylims[0] < y < ylims[1]:
            ax.text(x, y*1.05, '{0:.2f}'.format(K), fontsize=8)
            if harmonic == 3:
                ax.text(x/1.05, y/1.2, '{0:.2f}'.format(gap), fontsize=8,
                        color='b')
print("done!")

plt.savefig('calc_undulator_tune.png')
plt.show()
