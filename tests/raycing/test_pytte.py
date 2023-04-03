import numpy as np
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing as raycing
from xrt.backends.raycing.pyTTE_x import TTcrystal, TTscan, Quantity
from xrt.backends.raycing.pyTTE_x.pyTTE_rkpy import TakagiTaupin 
from matplotlib import pyplot as plt

thickness=0.3
#geometry = 'Laue reflected'  # for xrt. pytte will return transmitted as well
geometry = 'Bragg reflected'  # for xrt. pytte will return transmitted as well
#crystal = rm.CrystalSi(hkl=[1,1,1], geom=geometry)
crystal = rm.CrystalSi(hkl=[10,10,0], geom=geometry)
Rcurv = 2e4
alpha=np.radians(0.0)
E=[19728]
dtheta=np.linspace(-50, 10, 300)*1e-6

geotag = 0 if geometry.startswith('B') else np.pi*0.5

ttx = TTcrystal(crystal = 'Si', hkl=crystal.hkl, thickness = Quantity(thickness*1e3,'um'), 
                debye_waller = 1, xrt_crystal=crystal, Rx=Quantity(Rcurv, 'mm'),
                asymmetry = Quantity(alpha+geotag, 'rad'))
tts = TTscan(constant=Quantity(E[0],'eV'),
             scan=Quantity(dtheta, 'rad'),
             polarization='sigma')
#            ttp = TTscan(constant=Quantity(E[0],'eV'),
#                         scan=Quantity(theta-thetaCenter, 'rad'),
#                         polarization='pi')

scan_tt_s=TakagiTaupin(ttx, tts)
#            scan_tt_p=TakagiTaupin(ttx, ttp)        

if __name__ == '__main__':
    scan_vector, R, T, curSD = scan_tt_s.run()
#    print(curSD.shape)

#    dtheta = np.linspace(-5, 10, 501)
    theta = crystal.get_Bragg_angle(E[0]) + dtheta
    curS, curP = crystal.get_amplitude(E[0], np.sin(theta))
    
    plt.figure("reflectivity")
    plt.plot(np.degrees(dtheta)*3600, R, label='Reflectivity pytte')
    plt.plot(np.degrees(dtheta)*3600, abs(curS)**2, label='Reflectivity xrt plain')
    plt.legend()
    
    plt.figure("transmittivity")
    plt.plot(np.degrees(dtheta)*3600, T)
    plt.show()