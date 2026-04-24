#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script launches xrtQook. It optionally defines custom classes of
optical elements and materials to be visible in xrtQook."""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "12 Sep 2024"

import argparse  # analysis:ignore
import sys  # analysis:ignore
import os  # analysis:ignore
sys.path.append(os.path.join('..', '..'))

# Fix opengl context mismatch on RHEL 9 / Wayland, should works on all linux OS
# System packages required to run xrtQook on Wayland (tested on RHEL 9 KDE)
# sudo dnf install \
#  mesa-libGL mesa-dri-drivers \
#  libglvnd-glx \
#  xorg-x11-server-Xwayland
if sys.platform == "linux":
    os.environ["PYOPENGL_PLATFORM"] = "glx"

import xrt.gui.xrtQook as xQ  # analysis:ignore


"""An example of adding custom classes of optical elements, here a class
CustomMirror (change to your class) from module customOEs (your module).
Copy this file to a writable folder and uncomment the following lines.
The module customOEs must be importable from that folder or just be there.
Then run this python file to start xrtQook. This works similarly with custom
material classes, see below."""

# import xrt.backends.raycing.oes as roe
# from customOEs import CustomMirror
# roe.CustomMirror = CustomMirror
# roe.__allSectioned__['My custom OEs'] = ('CustomMirror',)
# roe.allArguments.extend(['customMirrorArg1', 'customMirrorArg2'])
# # 'customMirrorArg1', 'customMirrorArg2' ... are parameters of
# # CustomMirror.__init__() that are described in init's docstrings.

# import xrt.backends.raycing.materials as rm
# from myMaterials import MyMultilayer
# rm.MyMultilayer = MyMultilayer
# rm.__allSectioned__['My custom materials'] = ('MyMultilayer',)


def main(argv=None):
    parser = argparse.ArgumentParser(description="starter of xrtQook")
    parser.add_argument(
        "-p", "--projectFile",
        metavar="NNN.xml",
        help="load an xml project file"
    )
    parser.add_argument(
        "-v", "--verbosity",
        type=int,
        default=0,
        help="verbosity level for diagnostic purpose, int 0 (default) to 50"
    )

    args = parser.parse_args(argv)

    # DEBUG_LEVEL = args.verbosity

    if any('spyder' in name.lower() for name in os.environ):
        pass
    else:
        if str(sys.executable).endswith('pythonw.exe'):
            sys.stdout = open("output.log", "w")

    # Optional Qt scaling:
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # os.environ["QT_SCALE_FACTOR"] = "1.5"

    app = xQ.qt.QApplication(sys.argv)

    ex = xQ.XrtQook(projectFile=args.projectFile)
    ex.setWindowTitle("xrtQook")
    ex.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
