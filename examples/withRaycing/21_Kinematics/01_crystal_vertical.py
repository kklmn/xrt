# -*- coding: utf-8 -*-
"""Move the 1Crystal optic vertically from -2 to 2 mm in 10 steps."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from xrt.backends import raycing
from xrt.backends.raycing.stages import LinearStage, Pose


def build_beamline():
    layout = Path(__file__).resolve().parents[1] / '_QookBeamlines' / '1crystal.xml'
    beamline = raycing.BeamLine(fileName=layout)
    source = beamline.sources[0]
    crystal = beamline.oes[0]
    crystal.limPhysX = [-10, 10]
    crystal.limPhysY = [-50, 50]

    stage = LinearStage(
        name='Crystal height', axis='z', center=crystal.center,
        positionMin=-2, positionMax=2, positionUnit='mm')
    energy = (source.eMin + source.eMax) / 2
    pitch = crystal.material.get_Bragg_angle(energy)
    crystal.mountOffset = Pose(pitch=pitch)
    crystal.mount = stage
    return beamline, stage, crystal


def scan(stage, crystal, steps=10):
    """Yield stage travel and the crystal's world Z coordinate, both in mm."""
    for position in np.linspace(stage.positionMin, stage.positionMax, steps):
        stage.position = position
        yield stage.position, crystal.center[2]


def main():
    _, stage, crystal = build_beamline()
    for position, _ in scan(stage, crystal):
        center = ', '.join(f'{value:.1f}' for value in crystal.center)
        pitch, roll, yaw = np.degrees((crystal.pitch, crystal.roll, crystal.yaw))
        print(f'Stage {stage.axis} {position:.8f} {stage.positionUnit} -> '
              f'crystal center ({center}) mm | pitch {pitch:.8f}, '
              f'roll {roll:.8f}, yaw {yaw:.8f} deg')


if __name__ == '__main__':
    main()
