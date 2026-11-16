# -*- coding: utf-8 -*-
"""Drive the 1Crystal pitch stage in degrees or eV."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from xrt.backends import raycing
from xrt.backends.raycing.stages import RotationStage


IS_BRAGG = False  # True uses eV, False uses degrees.


def build_beamline(isBragg=IS_BRAGG):
    layout = Path(__file__).resolve().parents[1] / '_QookBeamlines' / '1crystal.xml'
    beamline = raycing.BeamLine(fileName=layout)
    source = beamline.sources[0]
    crystal = beamline.oes[0]
    crystalSi01 = crystal.material
    if isBragg:
        positionMin, positionMax = source.eMin, source.eMax
    else:
        positionMin, positionMax = sorted(
            np.degrees(crystalSi01.get_Bragg_angle(energy))
            for energy in (source.eMin, source.eMax))

    stage = RotationStage(
        name='Crystal pitch', axis='pitch', center=crystal.center,
        position=(positionMin + positionMax) / 2,
        positionMin=positionMin, positionMax=positionMax,
        positionUnit='eV' if isBragg else 'deg',
        crystal=crystalSi01 if isBragg else None)
    crystal.limPhysX = [-10, 10]
    crystal.limPhysY = [-50, 50]
    # The default mount offset aligns the crystal with the pitch stage.
    crystal.mount = stage
    return beamline, stage, crystal


def scan(stage, crystal, steps=5):
    """Yield the commanded stage position and the crystal pitch in degrees."""
    for position in np.linspace(stage.positionMin, stage.positionMax, steps):
        stage.position = position
        yield stage.position, np.degrees(crystal.pitch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bragg', action=argparse.BooleanOptionalAction,
                        default=IS_BRAGG,
                        help='use eV for stage position (default: %(default)s)')
    _, stage, crystal = build_beamline(isBragg=parser.parse_args().bragg)
    for position, _ in scan(stage, crystal):
        center = ', '.join(f'{value:.1f}' for value in crystal.center)
        pitch, roll, yaw = np.degrees((crystal.pitch, crystal.roll, crystal.yaw))
        print(f'Stage {stage.axis} {position:.4f} {stage.positionUnit} -> '
              f'crystal center ({center}) mm | pitch {pitch:.4f}, '
              f'roll {roll:.4g}, yaw {yaw:.4g} deg')


if __name__ == '__main__':
    main()
