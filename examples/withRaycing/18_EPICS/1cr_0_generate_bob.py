# -*- coding: utf-8 -*-
"""Generate Phoebus BOB screens for the 1-crystal EPICS example."""

import os
import sys
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore

from xrt.backends.raycing.epics.generate_bob import main as generate_bob  # analysis:ignore


exampleDir = os.path.dirname(os.path.abspath(__file__))
fileName = os.path.join(exampleDir, "1crystal.xml")
outputDir = os.path.join(exampleDir, "bob")
epicsPrefix = "BL"


def main():
    return generate_bob([
        "--layout", fileName,
        "--output", outputDir,
        "--prefix", epicsPrefix,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
