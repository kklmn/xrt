# -*- coding: utf-8 -*-
"""This script launches xrtQook."""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Oct 2017"

import sys
import os
import platform
sys.path.append(os.path.join('..', '..'))
import xrt.gui.xrtQook2 as xQ


if __name__ == '__main__':
    if any('spyder' in name.lower() for name in os.environ):
        pass  # spyder is present
    else:
        if str(sys.executable).endswith('pythonw.exe'):
            sys.stdout = open("output.log", "w")

    # If xrtQook looks too small, one can play with scaling:
    # either with "auto" factor or with a manually set factor.
#    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1" # "1" is "yes", not factor
#    os.environ["QT_SCALE_FACTOR"] = "1.5"

    args = sys.argv
    if platform.system() in ['Darwin']:
        args.append("--disable-web-security")
    app = xQ.qt.QApplication(args)

    ex = xQ.XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.showMaximized()
    ex.show()
    sys.exit(app.exec_())
