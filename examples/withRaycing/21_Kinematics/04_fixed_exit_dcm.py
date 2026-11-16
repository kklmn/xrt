# -*- coding: utf-8 -*-
"""Scan a fixed-exit Si(111) DCM assembled from individual raycing stages."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from xrt.backends import raycing
from xrt.backends.raycing.apertures import RectangularAperture
from xrt.backends.raycing.materials import CrystalSi
from xrt.backends.raycing.oes import OE
from xrt.backends.raycing.screens import Screen
from xrt.backends.raycing.sources import GeometricSource
from xrt.backends.raycing.stages import LinearStage, Pose, RotationStage


ENERGY_MIN = 5000.
ENERGY_MAX = 20000.
REFERENCE_ENERGY = (ENERGY_MIN + ENERGY_MAX) / 2
SCAN_POINTS = 7

MONO_Y = 10000.
SCREEN_DISTANCE = 2000.
FIXED_EXIT = 10.
AXIS_HEIGHT = 50.
SLIT_DISTANCE = 600.
SLIT_HALF_OPENING = 1.
CRYSTAL_X_LIMITS = (-10., 10.)
X1_Y_LIMITS = (-50., 50.)
X2_Y_LIMITS = (-50., 50.)


def bragg_pitch(material, energy):
    """Return the symmetric Bragg setting in radians, including refraction."""
    return material.get_Bragg_angle(energy) - material.get_dtheta(energy)


def x2_center_on_disk(pitch):
    """Return the nominal X2 hit point in the rotating disk frame, in mm."""
    # The two surfaces are parallel. For an exit rise h, their separation
    # normal to the surfaces is h/(2 cos pitch). Center X2 on the reflected
    # ray by placing it h/(2 sin pitch) along the disk's local Y axis.
    return np.array((0., FIXED_EXIT / (2 * np.sin(pitch)),
                     FIXED_EXIT / (2 * np.cos(pitch))))


def build_beamline():
    beamline = raycing.BeamLine(name='Fixed-exit Si(111) DCM')
    beamline.source = GeometricSource(
        beamline, name='Geometric source', nrays=1,
        distx=None, disty=None, distz=None, distxprime=None, distzprime=None,
        energies=(REFERENCE_ENERGY,))
    si111 = CrystalSi(hkl=(1, 1, 1), name='Si(111)')

    # At base travel zero, the X1 surface and entrance slit are at source Z=0.
    beamline.baseZ = LinearStage(
        name='Base height', axis='z', center=(0, MONO_Y, -AXIS_HEIGHT))
    beamline.disk = RotationStage(
        name='Bragg disk', axis='pitch', positionUnit='deg',
        mount=beamline.baseZ, mountOffset=Pose(center=(0, 0, AXIS_HEIGHT)))

    blades = dict(left=-SLIT_HALF_OPENING, right=SLIT_HALF_OPENING,
                  bottom=-SLIT_HALF_OPENING, top=SLIT_HALF_OPENING)
    beamline.slit1 = RectangularAperture(
        beamline, name='Slit 1', blades=blades, mount=beamline.baseZ,
        mountOffset=Pose(center=(0, -SLIT_DISTANCE, AXIS_HEIGHT)))
    beamline.x1 = OE(
        beamline, name='X1', material=si111, mount=beamline.disk,
        limPhysX=CRYSTAL_X_LIMITS, limPhysY=X1_Y_LIMITS)

    # Place the X2 optical origin at the reference hit point with both
    # translation stages at zero. Both travel axes rotate with the disk.
    reference_center = x2_center_on_disk(bragg_pitch(si111, REFERENCE_ENERGY))
    beamline.x2Long = LinearStage(
        name='X2 longitudinal', axis='y', mount=beamline.disk,
        mountOffset=Pose(center=reference_center))
    beamline.x2Perp = LinearStage(
        name='X2 perpendicular', axis='z', mount=beamline.x2Long)
    # The centered surface and the nominal beam hit share the cradle pivot.
    beamline.x2Theta = RotationStage(
        name='X2 cradle', axis='pitch', positionUnit='deg',
        pivotPoint=(0, 0, 0), mount=beamline.x2Perp)
    # Flip the reflecting face to look back at X1, retaining the local Y axis.
    beamline.x2 = OE(
        beamline, name='X2', material=si111, mount=beamline.x2Theta,
        mountOffset=Pose(roll='180 deg'), positionRoll='180 deg',
        rotationSequence='RyRzRx', limPhysX=CRYSTAL_X_LIMITS,
        limPhysY=X2_Y_LIMITS)

    beamline.slit2 = RectangularAperture(
        beamline, name='Slit 2', blades=blades, mount=beamline.baseZ,
        mountOffset=Pose(center=(0, SLIT_DISTANCE, AXIS_HEIGHT + FIXED_EXIT)))
    beamline.screen = Screen(
        beamline, name='Exit screen',
        center=(0, MONO_Y + SCREEN_DISTANCE, FIXED_EXIT))
    set_energy(beamline, REFERENCE_ENERGY)
    return beamline


def set_energy(beamline, energy):
    """Tune the disk and both X2 slides. The cradle remains a separate trim."""
    pitch = bragg_pitch(beamline.x1.material, energy)
    beamline.disk.position = np.degrees(pitch)
    travel = x2_center_on_disk(pitch) - beamline.x2Long.mountOffset.center
    beamline.x2Long.position = travel[1]
    beamline.x2Perp.position = travel[2]
    beamline.source.energies = (energy,)


def run_process(beamline):
    """Trace the beam through both slits and both crystal reflections."""
    source = beamline.source.shine()
    incident, slit1 = beamline.slit1.propagate(source, needNewGlobal=True)
    reflected, x1 = beamline.x1.reflect(incident)
    exit_beam, x2 = beamline.x2.reflect(reflected)
    transmitted, slit2 = beamline.slit2.propagate(exit_beam, needNewGlobal=True)
    screen = beamline.screen.expose(transmitted, onlyPositivePath=True)
    screen_global = beamline.screen.expose_global(
        transmitted, onlyPositivePath=True)
    return dict(beamSlit1=slit1, beamX1=x1, beamX2=x2, beamSlit2=slit2,
                beamScreen=screen, beamScreenGlobal=screen_global)


def scan(beamline, steps=SCAN_POINTS):
    for energy in np.linspace(ENERGY_MIN, ENERGY_MAX, steps):
        set_energy(beamline, energy)
        yield energy, run_process(beamline)


def main():
    beamline = build_beamline()
    for energy, beams in scan(beamline):
        screen = beams['beamScreenGlobal']
        good = screen.state == 1
        z = np.mean(screen.z[good]) if good.any() else np.nan
        flux = (screen.Jss[good] + screen.Jpp[good]).mean() if good.any() else 0.
        print(f'{energy:7.0f} eV | disk {beamline.disk.position:8.4f} deg | '
              f'X2 long {beamline.x2Long.position:+8.4f}, '
              f'perp {beamline.x2Perp.position:+7.4f} mm | '
              f'exit Z {z:7.3f} mm | relative intensity {flux:.3f}')


if __name__ == '__main__':
    main()
