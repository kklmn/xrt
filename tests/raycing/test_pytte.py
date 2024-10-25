import numpy as np
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.materials_crystals as rmc
import xrt.backends.raycing as raycing
from xrt.backends.raycing.pyTTE_x import TTcrystal as TTcrystalX
from xrt.backends.raycing.pyTTE_x import TTscan as TTscanX
from xrt.backends.raycing.pyTTE_x import Quantity as QuantityX
from xrt.backends.raycing.pyTTE_x.pyTTE_rkpy import TakagiTaupin as TakagiTaupinX
from pyTTE import TTcrystal, TTscan, Quantity, TakagiTaupin
from matplotlib import pyplot as plt

try:
    import pyopencl as cl
    import xrt.backends.raycing.myopencl as mcl
    targetOpenCL = 'auto'
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    xrt_cl = mcl.XRT_CL(r'materials.cl', precisionOpenCL='float32',
                        targetOpenCL=targetOpenCL)
    isOpenCL = hasattr(xrt_cl, 'cl_precisionF')
except ImportError:
    isOpenCL = False



case1 = {
    "name": "Bragg Flat",
    "crystal": "Si", # select from ('Si', 'Ge', 'Diamond', 'InSb', 'AlphaQuartz', 'Sapphire')
    "geometry": "Bragg reflected",  # Combination of Bragg/Laue reflected/transmitted
    "hkl": [1, 1, 1],
    "thickness": 1.,  # mm
    "asymmetry": 0.,  # degrees
    "in_plane_rotation": 0.,  # degrees
    "bending_Rm": np.inf,  # meters
    "bending_Rs": np.inf,  # meters
    "energy": 9000,  # eV
    "theta_array": np.linspace(-50, 150, 200),  # microrad
    "polarization": "sigma",  # or "pi"
    "precision": "float32"  # Only for opencl backend. "float64" for Laue
    }

case2 = {
    "name": "Bragg 1D bent",
    "crystal": "Si", # select from ('Si', 'Ge', 'Diamond', 'InSb', 'AlphaQuartz', 'Sapphire')
    "geometry": "Bragg reflected",  # Combination of Bragg/Laue reflected/transmitted
    "hkl": [1, 1, 1],
    "thickness": 1.,  # mm
    "asymmetry": 5.,  # degrees
    "in_plane_rotation": 0.,  # degrees
    "bending_Rm": 10.,  # meters
    "bending_Rs": np.inf,  # meters
    "energy": 9000,  # eV
    "theta_array": np.linspace(-50, 150, 200),  # microrad
    "polarization": "sigma",  # or "pi"
    "precision": "float32"  # Only for opencl backend. "float64" for Laue
    }

case3 = {
    "name": "Bragg 2D bent",
    "crystal": "Si", # select from ('Si', 'Ge', 'Diamond', 'InSb', 'AlphaQuartz', 'Sapphire')
    "geometry": "Bragg reflected",  # Combination of Bragg/Laue reflected/transmitted
    "hkl": [3, 3, 3],
    "thickness": 3.,  # mm
    "asymmetry": -2.,  # degrees
    "in_plane_rotation": 0.,  # degrees
    "bending_Rm": 10.,  # meters
    "bending_Rs": 10,  # meters
    "energy": 19000,  # eV
    "theta_array": np.linspace(-50, 150, 200),  # microrad
    "polarization": "sigma",  # or "pi"
    "precision": "float32"  # Only for opencl backend. "float64" for Laue
    }

def run_pytte_xrt_opencl(name, crystal, geometry, hkl, thickness, asymmetry,
                         in_plane_rotation, bending_Rm, bending_Rs, energy,
                         theta_array, polarization, precision):
    alpha = np.radians(asymmetry)
    crInstance = getattr(rmc, crystal)(hkl=hkl,
                                      geom=geometry,
                                      t=thickness)

    thetaCenter = np.arcsin(rm.ch / (2*crInstance.d*energy))
    E_in = np.ones_like(theta_array) * energy
    theta = theta_array*1e-6 + thetaCenter

    s0 = (np.zeros_like(theta), np.cos(theta+alpha), -np.sin(theta+alpha))
    sh = (np.zeros_like(theta), np.cos(theta-alpha), np.sin(theta-alpha))
    if geometry.startswith('Bragg'):
        precision = "float32"
        n = (0, 0, 1)  # outward surface normal
    else:
        n = (0, -1, 0)  # outward surface normal
    hn = (0, np.sin(alpha), np.cos(alpha))  # outward Bragg normal
    gamma0 = sum(i*j for i, j in zip(n, s0))
    gammah = sum(i*j for i, j in zip(n, sh))
    hns0 = sum(i*j for i, j in zip(hn, s0))

    matCL = mcl.XRT_CL(r'materials.cl',
                       precisionOpenCL=precision,
                       targetOpenCL=targetOpenCL)

    ampS, ampP = crInstance.get_amplitude_pytte(E_in, gamma0, gammah, hns0,
                                                ucl=matCL, alphaAsym=alpha,
                                                inPlaneRotation=in_plane_rotation,
                                                Ry=bending_Rm*1e3,
                                                Rx=bending_Rs*1e3)
    return np.abs(ampS)**2 if polarization == "sigma" else np.abs(ampP)**2

def run_pytte_xrt_cpu(name, crystal, geometry, hkl, thickness, asymmetry,
                      in_plane_rotation, bending_Rm, bending_Rs, energy,
                      theta_array, polarization, precision):

    alpha = np.radians(asymmetry)
    geotag = 0 if geometry.startswith('B') else np.pi*0.5
    crInstance = getattr(rmc, crystal)(hkl=hkl,
                                      geom=geometry,
                                      t=thickness)

    refl = geometry.endswith("lected")

    ttx = TTcrystalX(crystal=crystal,
                    hkl=hkl,
                    thickness=QuantityX(thickness*1e3,'um'),
                    debye_waller=1,
                    xrt_crystal=crInstance,
                    Rx=QuantityX(bending_Rm, 'm'),
                    Ry=QuantityX(bending_Rs, 'm'),
                    asymmetry=QuantityX(alpha+geotag, 'rad'),
                    in_plane_rotation=QuantityX(in_plane_rotation, 'deg'))

    tts = TTscanX(constant=QuantityX(energy,'eV'),
                  scan=QuantityX(theta_array, 'urad'),
                  polarization=polarization)

    scan_tt_s=TakagiTaupinX(ttx, tts)
    scan_vector, R, T, curSD = scan_tt_s.run()
    return R if refl else T

def run_pytte_original(name, crystal, geometry, hkl, thickness, asymmetry,
                           in_plane_rotation, bending_Rm, bending_Rs, energy,
                           theta_array, polarization, precision):
    alpha = np.radians(asymmetry)
    geotag = 0 if geometry.startswith('B') else np.pi*0.5
    refl = geometry.endswith("lected")

    ttx = TTcrystal(crystal=crystal,
                    hkl=hkl,
                    thickness=Quantity(thickness*1e3,'um'),
                    debye_waller=1,
#                    xrt_crystal=crInstance,
                    Rx=Quantity(bending_Rm, 'm'),
                    Ry=Quantity(bending_Rs, 'm'),
                    asymmetry=Quantity(alpha+geotag, 'rad'),
                    in_plane_rotation=Quantity(in_plane_rotation, 'deg'))

    tts = TTscan(constant=Quantity(energy,'eV'),
                 scan=Quantity(theta_array, 'urad'),
                 polarization=polarization)

    scan_tt_s=TakagiTaupin(ttx, tts)
    scan_vector, R, T = scan_tt_s.run()
    return R if refl else T


if __name__ == '__main__':
#    plt.ion()
    for calcParams in [case1, case2]: #, case3]:
        plt.figure(calcParams["name"])
        for func in [
                     run_pytte_xrt_opencl,
#                     run_pytte_xrt_cpu,
                     run_pytte_original
                     ]:
            data = func(**calcParams)
            plt.plot(calcParams['theta_array'], data,
                     label=str(func.__name__).split("_")[-1])
        plt.legend()
        plt.show(block=False)
    plt.show()
