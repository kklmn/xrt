# -*- coding: utf-8 -*-
"""Run the 1-crystal beamline in xrtGlow with EPICS controls."""

import os
import sys

exampleDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(
    os.path.join(exampleDir, '..', '..', '..')))  # analysis:ignore

import xrt.backends.raycing as raycing  # analysis:ignore


fileName = os.path.join(exampleDir, "1crystal.xml")


def main():
    beamLine = raycing.BeamLine(fileName=fileName)
    beamLine.glow(epicsPrefix="VBL")


if __name__ == "__main__":
    main()
