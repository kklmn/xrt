# -*- coding: utf-8 -*-
u"""
xrtGlow -- an interactive 3D beamline viewer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The beamline created in xrtQook or in a python script can be interactively
viewed in an OpenGL based widget xrtGlow. It visualizes beams, footprints,
surfaces, apertures and screens. The brightness represents intensity and the
color represents an auxiliary user-selected distribution, typically energy.
A virtual screen can be put at any position and dragged by mouse with
simultaneous observation of the beam distribution on it. See two example
screenshots below (click to expand and read the captions).

The primary purpose of xrtGlow is to demonstrate the alignment correctness
given the fact that xrtQook can automatically calculate several positional and
angular parameters.

See aslo :ref:`Notes on using xrtGlow <glow_notes>`.

+-------------+-------------+-------------+
|   |glow1|   |   |glow2|   |   |glow3|   |
+-------------+-------------+-------------+

.. |glow1| imagezoom:: _images/xrtGlow1.png
   :alt: &ensp;A 3D view of a von Hamos spectrometer. The scaling on
       this image is isotropic. The coloring is by energy.

.. |glow2| imagezoom:: _images/xrtGlow2.png
   :align: center
   :alt: &ensp; A 3D view of three double-paraboloid lenses. The scaling on
       this image is isotropic. The source (on the left) is a parallel
       geometric source. The coloring is by axial divergence (red=0), showing
       the effect of refractive focusing.

.. |glow3| imagezoom:: _images/xrtGlow3.png
   :align: right
   :alt: &ensp; A 3D view of a beamline in the vicinity of its insertion device.
       The scaling in this image is anisotropic and strongly compressed along
       the beamline direction. The yellow lines represent the envelope of
       electron trajectories, visualizing the effect of the electron-beam
       emittance.

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

from .widgets import xrtGlow  # analysis:ignore
from .widgets.inspector import InstanceInspector, ConfigurablePlotWidget  # analysis:ignore
from ._utils import is_screen, is_aperture  # analysis:ignore
from ..commons import qt  # analysis:ignore
