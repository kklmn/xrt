# -*- coding: utf-8 -*-
"""This script launches xrtQook. It optionally defines custom classes of
optical elements to be visible in xrtQook."""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "9 Jul 2023"

import sys
import os
sys.path.append(os.path.join('..', '..'))
import xrt.gui.xrtQook as xQ

"""An example of adding custom classes of optical elements, here a class
CustomMirror (change to your class) from module customOEs (your module).
Copy this file to a writable folder and uncomment the following lines.
The module customOEs must be importable from that folder or just be there.
Then run this python file to start xrtQook."""

# import xrt.backends.raycing.oes as roe
# from customOEs import CustomMirror
# roe.CustomMirror = CustomMirror
# roe.__allSectioned__['My custom OEs'] = ('CustomMirror',)
# roe.allArguments.extend(['customMirrorArg1', 'customMirrorArg2'])
# # 'customMirrorArg1', 'customMirrorArg2' ... are parameters of
# # CustomMirror.__init__() that are described in init's docstrings.


if __name__ == '__main__':
    if any('spyder' in name.lower() for name in os.environ):
        pass  # spyder is present
    else:
        if str(sys.executable).endswith('pythonw.exe'):
            sys.stdout = open("output.log", "w")

    # If xrtQook looks too small, one can play with scaling:
    # either with "auto" factor or with a manually set factor.
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1" # means "yes", not factor
    # os.environ["QT_SCALE_FACTOR"] = "1.5"

    args = sys.argv
    # args.append("--disable-web-security")
    app = xQ.qt.QApplication(args)

    ex = xQ.XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.show()
    sys.exit(app.exec_())
