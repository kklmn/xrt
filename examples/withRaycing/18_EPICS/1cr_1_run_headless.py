# -*- coding: utf-8 -*-
"""Run the 1-crystal beamline as a headless EPICS IOC."""

import os
import sys
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore

from xrt.backends.raycing.epics import DynamicBeamline  # analysis:ignore


exampleDir = os.path.dirname(os.path.abspath(__file__))
fileName = os.path.join(exampleDir, "1crystal.xml")
epicsPrefix = "BL"


def main():
    beamLine = DynamicBeamline(
        fileName=fileName,
        epicsPrefix=epicsPrefix,
        localEpics=True,
    )
    print("Serving {0} with EPICS prefix {1}:".format(
        os.path.basename(fileName), epicsPrefix))
    print("Use Ctrl+C to stop.")
    try:
        beamLine.run_forever()
    except KeyboardInterrupt:
        print("Stopping headless beamline.")


if __name__ == "__main__":
    main()
