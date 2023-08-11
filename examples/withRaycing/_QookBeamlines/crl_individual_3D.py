# -*- coding: utf-8 -*-
r"""
see crl_stack.py
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "08 Mar 2016"
import os, sys;
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
#sys.path.append(r"E:\xrt-1.3.5")  # analysis:ignore
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr

parabolaParam = 1.  # mm
zmax = 1.  # mm
dz = 5.  # mm
E0 = 9000.  # eV
p = 1000.  # source to 1st lens
q = 5000.  # 1st lens to focus

#Lens = roe.ParaboloidFlatLens
Lens = roe.DoubleParaboloidLens
#Lens = roe.ParabolicCylinderFlatLens

mBeryllium = rm.Material('Be', rho=1.848, kind='lens')
#mDiamond = rm.Material('C', rho=3.52, kind='lens')
#mAluminum = rm.Material('Al', rho=2.7, kind='lens')
#mSilicon = rm.Material('Si', rho=2.33, kind='lens')
#mNickel = rm.Material('Ni', rho=8.9, kind='lens')
#mLead = rm.Material('Pb', rho=11.35, kind='lens')
material = mBeryllium


def build_beamline(nrays=1e3):
    beamLine = raycing.BeamLine(height=0)
    rs.GeometricSource(
        beamLine, 'CollimatedSource', nrays=nrays,
        dx=0.5, dz=0.5, distxprime=None, distzprime=None, energies=(E0,))

    beamLine.lenses = []
    ilens = 0
    while True:
        roll = 0.
        if Lens == roe.ParabolicCylinderFlatLens:
            roll = -np.pi/4 if ilens % 2 == 0 else np.pi/4
        lens = Lens(
            beamLine, 'Lens{0:02d}'.format(ilens), center=[0, p + dz*ilens, 0],
            pitch=np.pi/2, roll=roll, t=0.1, material=material,
            limPhysX=[-2, 2], limPhysY=[-2, 2], shape='round',
            focus=parabolaParam, zmax=zmax, alarmLevel=0.1)
        beamLine.lenses.append(lens)
        if ilens == 0:
            nCRL = lens.get_nCRL(q, E0)
        ilens += 1
        if nCRL - ilens < 0.5:
            break

    beamLine.fsmF = rsc.Screen(beamLine, 'FSM-focus', [0, p+q, 0])
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
    outDict = {'beamSource': beamSource}

    beamIn = beamSource
    for ilens, lens in enumerate(beamLine.lenses):
        lglobal, llocal1, llocal2 = lens.double_refract(beamIn, needLocal=True)
        beamIn = lglobal
        strl = '_{0:02d}'.format(ilens)
        outDict['beamLensGlobal'+strl] = lglobal
        outDict['beamLensLocal1'+strl] = llocal1
        outDict['beamLensLocal2'+strl] = llocal2

    outDict['beamFSM2'] = beamLine.fsmF.expose(lglobal)

    beamLine.prepare_flow()
    return outDict
rr.run_process = run_process


def main():
    beamLine = build_beamline()
    beamLine.glow()  # centerAt='Lens{0:02d}_Exit'.format(len(beamLine.lenses)-1),
#                  colorAxis='xzprime')


if __name__ == '__main__':
    main()
