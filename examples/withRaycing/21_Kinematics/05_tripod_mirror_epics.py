# -*- coding: utf-8 -*-
"""Generate Phoebus screens and run the tripod mirror in xrtGlow.

Glow serves the IOC on loopback. In the separate terminal used to start
Phoebus or another Channel Access client, set the following variables.

PowerShell::

    $env:EPICS_CA_ADDR_LIST = '127.0.0.1'
    $env:EPICS_CA_AUTO_ADDR_LIST = 'NO'

Bash or sh::

    export EPICS_CA_ADDR_LIST=127.0.0.1
    export EPICS_CA_AUTO_ADDR_LIST=NO

For Phoebus with the JCA backend, add the JVM option
``-Djca.use_env=true`` to enable environment lookup. Open
``tripod_mirror_bob/stages/Granite.bob`` after starting the example.
If the EPICS command-line tools are installed, check the connection with
``caget TRIPOD:Downstream_center:RBV``.
"""

from pathlib import Path
import sys

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR.parents[2]))

from xrt.backends.raycing import BeamLine
from xrt.backends.raycing.epics.generate_bob import BobWriter, generate_for_beamline


LAYOUT_FILE = EXAMPLE_DIR / 'tripod_mirror.xml'
BOB_DIR = EXAMPLE_DIR / 'tripod_mirror_bob'
EPICS_PREFIX = 'TRIPOD'


def generate_bob():
    beamline = BeamLine(fileName=LAYOUT_FILE)
    for file_name in generate_for_beamline(
            beamline, BOB_DIR, BobWriter(), prefix=EPICS_PREFIX):
        print(f'Generated {file_name}')


def run_glow():
    from xrt.gui.xrtGlow import qt, xrtGlow

    beamline = BeamLine(fileName=LAYOUT_FILE)
    app = qt.QApplication.instance() or qt.QApplication(sys.argv)
    viewer = xrtGlow(layout=beamline.export_to_json(), epicsPrefix=EPICS_PREFIX)
    viewer.loadTemplateScene(LAYOUT_FILE)
    viewer.show()
    app.exec_()


def main():
    generate_bob()
    run_glow()


if __name__ == '__main__':
    main()
