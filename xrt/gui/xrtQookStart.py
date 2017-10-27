# -*- coding: utf-8 -*-
"""This script launches xrtQook."""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Oct 2017"

import sys
import os
sys.path.append(os.path.join('..', '..'))
import xrt.gui.xrtQook as xQ


if __name__ == '__main__':
    if any('spyder' in name.lower() for name in os.environ):
        pass  # spyder is present
    else:
        if str(sys.executable).endswith('pythonw.exe'):
            sys.stdout = open("output.log", "w")

    app = xQ.qt.QApplication(sys.argv)
    ex = xQ.XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.show()
    sys.exit(app.exec_())
