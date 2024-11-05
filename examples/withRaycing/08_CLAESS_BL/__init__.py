# -*- coding: utf-8 -*-
r"""
ALBA CLÆSS beamline
-------------------

Files in ``\examples\withRaycing\08_CLAESS_BL``

This script produces images at various positions along the beamline, see image
captions in the enlarged figures.

+------------+------------+------------+
| |Claess01| | |Claess02| | |Claess03| |
+------------+------------+------------+
| |Claess04| | |Claess05| | |Claess06| |
+------------+------------+------------+
| |Claess07| | |Claess08| | |Claess09| |
+------------+------------+------------+
| |Claess10| | |Claess11| | |Claess12| |
+------------+------------+------------+
| |Claess13| |            |            |
+------------+------------+------------+

.. |Claess01| imagezoom:: _images/ClaessBL_N-Rh-01DiamondFSM1+FixedMask-wideE.*
   :alt: &ensp;at fixed FE mask
.. |Claess02| imagezoom:: _images/ClaessBL_N-Rh-02DiamondFSM1+FEMaskLT-wideE.*
   :alt: &ensp;at upstream half of FE movable mask
.. |Claess03| imagezoom:: _images/ClaessBL_N-Rh-03DiamondFSM1+FEmaskRB-wideE.*
   :alt: &ensp;at downstream half of FE movable mask
   :loc: upper-right-corner
.. |Claess04| imagezoom:: _images/ClaessBL_N-Rh-05VCM_footprintE-wideE.*
   :alt: &ensp;footprint on VCM
.. |Claess05| imagezoom:: _images/ClaessBL_N-Rh-08Xtal1_footprint-monoE.*
   :alt: &ensp;footprint on first DCM crystal
.. |Claess06| imagezoom:: _images/ClaessBL_N-Rh-11Xtal2_footprintE-monoE.*
   :alt: &ensp;footprint on second DCM crystal
   :loc: upper-right-corner
.. |Claess07| imagezoom:: _images/ClaessBL_N-Rh-12BSBlock-monoE.*
   :alt: &ensp;beam at the Bremsstrahlung block
.. |Claess08| imagezoom:: _images/ClaessBL_N-Rh-13XBPM4foils-monoE.*
   :alt: &ensp;image at foil holder of 4-diode XBPM
.. |Claess09| imagezoom:: _images/ClaessBL_N-Rh-16VFM_footprint-monoE.*
   :alt: &ensp;footprint on VFM
   :loc: upper-right-corner
.. |Claess10| imagezoom:: _images/ClaessBL_N-Rh-18OH-PS-FrontCollimator-monoE.*
   :alt: &ensp;front collimator of photon shutter
.. |Claess11| imagezoom:: _images/ClaessBL_N-Rh-19eh100To40Flange-monoE.*
   :alt: &ensp;image at reducer flange 100CF-to-40CF
.. |Claess12| imagezoom:: _images/ClaessBL_N-Rh-20slitEH-monoE.*
   :alt: &ensp;image at exp 4-blade slit
   :loc: upper-right-corner
.. |Claess13| imagezoom:: _images/ClaessBL_N-Rh-22FocusAtSampleE-monoE.*
   :alt: &ensp;image at focus (sample)

The script also exemplifies the usage of
:func:`~xrt.backends.raycing.apertures.RectangularAperture.touch_beam` for
finding the optimal size of slits.
"""
pass
