# -*- coding: utf-8 -*-
"""Drive a side-mounted 1Crystal optic through the stage's yaw axis."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from xrt.backends import raycing
from xrt.backends.raycing.stages import Pose, RotationStage


def build_beamline():
    layout = Path(__file__).resolve().parents[1] / '_QookBeamlines' / '1crystal.xml'
    beamline = raycing.BeamLine(fileName=layout)
    source = beamline.sources[0]
    crystal = beamline.oes[0]
    crystal.limPhysX = [-10, 10]
    crystal.limPhysY = [-50, 50]
    positionMin, positionMax = sorted(
        np.degrees(crystal.material.get_Bragg_angle(energy))
        for energy in (source.eMin, source.eMax))

    # Place the stage at the midpoint of the crystal's negative X edge.
    # Base roll points stage Z along crystal X, so the offset to the crystal
    # center lies on the yaw axis. The opposite roll levels the optic at zero.
    x, y, z = crystal.center
    side_x = crystal.limPhysX[0]
    stage = RotationStage(
        name='Crystal side mount', axis='yaw', roll='90 deg',
        center=(x + side_x, y, z),
        position=(positionMin + positionMax) / 2,
        positionMin=positionMin, positionMax=positionMax, positionUnit='deg')
    crystal.mountOffset = Pose(center=(0, 0, -side_x), roll='-90 deg')
    crystal.mount = stage
    return beamline, stage, crystal


def scan(stage, crystal, steps=5):
    """Yield the stage yaw and the crystal pitch, both in degrees."""
    for yaw in np.linspace(stage.positionMin, stage.positionMax, steps):
        stage.position = yaw
        yield stage.position, np.degrees(crystal.pitch)


def main():
    _, stage, crystal = build_beamline()
    for position, _ in scan(stage, crystal):
        center = ', '.join(
            f'{round(value, 1) or 0.0:.1f}' for value in crystal.center)
        pitch, roll, yaw = np.degrees((crystal.pitch, crystal.roll, crystal.yaw))
        print(f'Stage {stage.axis} {position:.4f} {stage.positionUnit} -> '
              f'crystal center ({center}) mm | pitch {pitch:.4f}, '
              f'roll {roll:.4f}, yaw {yaw:.4f} deg')


if __name__ == '__main__':
    main()
