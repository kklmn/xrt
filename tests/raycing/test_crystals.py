# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:15:38 2014

@author: konkle
"""

import sys
sys.path.append(r"c:\Ray-tracing")
import xrt.backends.raycing.materials as rm

xtalSi = rm.CrystalSi(hkl=(1, 1, 1), tK=300, rho=2.3296)

xtalSiGeneral = rm.CrystalFromCell(
    'silicon', (1, 1, 1), a=5.4311946,
    atoms=['Si']*8,
    atomsXYZ=[[0.0, 0.0, 0.0],
              [0.0, 0.5, 0.5],
              [0.5, 0.0, 0.5],
              [0.5, 0.5, 0.0],
              [.25, .25, .25],
              [.25, .75, .75],
              [.75, .25, .75],
              [.75, .75, .25]])

E0 = 4000.

xtal = xtalSi
print(xtal.get_structure_factor(E0, 0.5/xtal.d))

xtal = xtalSiGeneral
print(xtal.get_structure_factor(E0, 0.5/xtal.d))
