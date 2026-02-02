# -*- coding: utf-8 -*-
u"""
xrtQook -- a GUI for creating a beamline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. imagezoom:: _images/xrtQook.png
   :align: right
   :alt: &ensp;A view of xrtQook with an empty beamline tree on the left and a
       help panel on the right.

The main interface to xrt is through a python script. Many examples of such
scripts can be found in the supplied folders `examples` and `tests`. The script
imports the modules of xrt, instantiates beamline parts, such as synchrotron or
geometric sources, various optical elements, apertures and screens, specifies
required materials for reflection, refraction or diffraction, defines plots and
sets job parameters.

The Qt tool :mod:`xrtQook` takes these ingredients as GUI elements and prepares
a ready to use script that can be run within the tool itself or in an external
Python context. :mod:`xrtQook` has a parallelly updated help panel that
provides a complete list of parameters for the used objects. :mod:`xrtQook`
writes/reads the recipes of beamlines into/from xml files.

In the present version, :mod:`xrtQook` does not provide automated generation of
*scans* and does not create *wave propagation* sequences. For these two tasks,
the corresponding script parts have to be written manually based on the
supplied examples and the present documentation.

See a brief :ref:`tutorial for xrtQook <qook_tutorial>`.

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "16 Nov 2025"


import sys  # analysis:ignore
from .widgets import XrtQook  # analysis:ignore
from ..commons import qt  # analysis:ignore

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    ex = XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.show()
    sys.exit(app.exec_())
