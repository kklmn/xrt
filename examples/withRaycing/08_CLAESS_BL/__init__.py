r"""
ALBA CLÆSS beamline
-------------------

Files in ``\examples\withRaycing\08_CLAESS_BL``

See the optical scheme of the beamline
`here <http://www.cells.es/Beamlines/CLAESS/optics_layout.html>`_.

This script produces images at various positions along the beamline.

The following 13 images are:

1) FSM image after the front end with the projected absorbed rays (red) at

  a) the fixed front end mask,
  b) upstream half and
  c) downstream half of the movable front end mask

2) footprint on VCM,
3) footprint on the 1st crystal of DCM,
4) footprint on the 2nd crystal of DCM,
5) beam at the Bremsstrahlung block,
6) image at the foil holder of 4-diode XBPM,
7) footprint on VFM,
8) front collimator of the photon shutter,
9) image at the reducer flange 100CF-to-40CF,
10) image at the EH 4-blade slit,
11) image at the focal (sample) point.

.. imagezoom:: _images/ClaessBL_N-Rh-01DiamondFSM1+FixedMask-wideE.*
.. imagezoom:: _images/ClaessBL_N-Rh-02DiamondFSM1+FEMaskLT-wideE.*
.. imagezoom:: _images/ClaessBL_N-Rh-03DiamondFSM1+FEmaskRB-wideE.*
.. imagezoom:: _images/ClaessBL_N-Rh-05VCM_footprintE-wideE.*
.. imagezoom:: _images/ClaessBL_N-Rh-08Xtal1_footprint-monoE.*
.. imagezoom:: _images/ClaessBL_N-Rh-11Xtal2_footprintE-monoE.*
.. imagezoom:: _images/ClaessBL_N-Rh-12BSBlock-monoE.*
.. imagezoom:: _images/ClaessBL_N-Rh-13XBPM4foils-monoE.*
.. imagezoom:: _images/ClaessBL_N-Rh-16VFM_footprint-monoE.*
.. imagezoom:: _images/ClaessBL_N-Rh-18OH-PS-FrontCollimator-monoE.*
.. imagezoom:: _images/ClaessBL_N-Rh-19eh100To40Flange-monoE.*
.. imagezoom:: _images/ClaessBL_N-Rh-20slitEH-monoE.*
.. imagezoom:: _images/ClaessBL_N-Rh-22FocusAtSampleE-monoE.*

The script also exemplifies the usage of
:func:`~xrt.backends.raycing.apertures.RectangularAperture.touch_beam` for
finding the optimal size of slits.
"""
pass
