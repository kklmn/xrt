# -*- coding: utf-8 -*-
u"""
The main interface to xrt is through a python script. Many examples of such
scripts can be found in the supplied folder ‘examples’. The script imports the
modules of xrt, instantiates beamline parts, such as synchrotron or geometric
sources, various optical elements, apertures and screens, specifies required
materials for reflection, refraction or diffraction, defines plots and sets job
parameters.

The Qt tool :mod:`xrtQook` takes these ingredients and prepares a ready to use
script that can be run within the tool itself or in an external Python context.
:mod:`xrtQook` features a parallelly updated help panel that, unlike the main
documentation, provides a complete list of parameters for the used classes,
also including those from the parental classes. :mod:`xrtQook` writes/reads the
recipes of beamlines into/from xml files.

In the present version, :mod:`xrtQook` does not provide automated generation of
*scans* and does not create *wave propagation* sequences. For these two tasks,
the corresponding script parts have to be written manually based on the
supplied examples and the present documentation.

See a short :ref:`tutorial for xrtQook <qook>`.

.. image:: _images/xrtQook.png
   :scale: 60 %

"""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "08 Mar 2016"

import sys
import os
sys.path.append(os.path.join('..', '..'))
import xrt.xrtQook as xQ


if __name__ == '__main__':
    app = xQ.QApplication(sys.argv)
    ex = xQ.XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.show()
    sys.exit(app.exec_())
