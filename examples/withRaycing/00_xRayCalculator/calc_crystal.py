# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm

try:
    import pyopencl as cl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    isOpenCL = True
except ImportError:
    isOpenCL = False

if isOpenCL:
    import xrt.backends.raycing.myopencl as mcl
    matCL = mcl.XRT_CL(r'materials.cl',
                           precisionOpenCL='float32',
#                       precisionOpenCL='float64',
                       # targetOpenCL='CPU'
                       )
else:
    matCL = None

crystal = rm.CrystalSi(hkl=(1, 1, 1), t=0.3)

E = 9728
dtheta = np.linspace(-120, 120, 1501)
print(np.degrees(crystal.get_Bragg_angle(E)))
theta = crystal.get_Bragg_angle(E) + dtheta*1e-6
curS, curP = crystal.get_amplitude_pytte(E, -np.sin(theta), ucl=matCL, Ry=1e5,
                                         alphaAsym=np.radians(5))
#curS, curP = crystal.get_amplitude(E, -np.sin(theta))
print(crystal.get_a())
print(crystal.get_F_chi(E, 0.5/crystal.d))
print(u'Darwin width at E={0:.0f} eV is {1:.5f} µrad for s-polarization'.
      format(E, crystal.get_Darwin_width(E) * 1e6))

plt.figure()
plt.plot(dtheta, abs(curS)**2, 'r', dtheta, abs(curP)**2, 'b')
plt.gca().set_xlabel(u'$\\theta - \\theta_{B}$ (µrad)')
plt.gca().set_ylabel(r'reflectivity')
plt.show()
