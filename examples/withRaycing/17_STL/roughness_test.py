# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:46:14 2025

@author: Roman Chernikov
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import time
import os, sys; sys.path.append(os.path.join('..', '..', '..'))
from xrt.backends.raycing.roughness import RandomRoughness, GaussianBump, Waviness 

t0 = time.time()
rghR = RandomRoughness(limPhysX=[-20, 20], limPhysY=[-100, 100],
                      rms=1., corrLength=1, seed=20251201)
t1 = time.time()
rghG = GaussianBump(base=rghR, limPhysX=[-20, 20], limPhysY=[-100, 100],
                   bumpHeight=3., sigmaX=20, sigmaY=30)

rghW = Waviness(base=rghG, limPhysX=[-20, 20], limPhysY=[-100, 100],
               amplitude=2., xWaveLength=80, yWaveLength=20)
#print(rgh)
x = np.linspace(-20, 20, 200)
y = np.linspace(-100, 100, 1000)
xm, ym = np.meshgrid(x, y)
z = rghW.local_z(xm.flatten(), ym.flatten())
t2 = time.time()
print("Building spline takes", t1-t0, "s")
print("Evaluating spline for 200x1000 points takes", t2-t1, "s")

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xm, ym,
                       z.reshape(1000, 200),
                       cmap=cm.jet,
                   linewidth=0, antialiased=False)
ax.set_box_aspect((1, 10, 0.2))
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.figure()
#plt.imshow(z, origin='lower', aspect='equal')
plt.imshow(z.reshape((1000, 200)),
           origin='lower',
           aspect='equal',
           cmap='jet',
           extent=[-20, 20, -100, 100]
           )

plt.show()



