# -*- coding: utf-8 -*-
r"""

Beamline optics
---------------

The images below are produced by the scripts in
``\examples\withRaycing\02_Balder_BL\``.
The examples show the scans of various optical elements at Balder@MaxIV
beamline. The source is a multipole conventional wiggler.

Diamond filter of varying thickness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shown are i) intensity downstream of the filter and its energy spectrum and
ii) absorbed power in the filter with its energy spectrum and power density
isolines at 85% and 95% of the maximum.

.. animation:: _images/filterThicknessI
.. animation:: _images/filterThicknessP

White-beam collimating mirror at varying pitch angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------+-----------------------------+-----------------------------+
|                 |           bare Si           |          Ir coating         |
+=================+=============================+=============================+
| flux downstream |                             |                             |
| of the mirror   |        |vcmSi-FSM|          |         |vcmIr-FSM|         |
| with its energy |                             |                             |
| spectrum        |                             |                             |
+-----------------+-----------------------------+-----------------------------+
| absorbed power  |                             |                             |
| in the mirror   |                             |                             |
| with its energy |                             |                             |
| spectrum and    |     |vcmSi-FootprintP|      |      |vcmIr-FootprintP|     |
| power density   |                             |                             |
| isolines at 90% |                             |                             |
| of the maximum  |                             |                             |
+-----------------+-----------------------------+-----------------------------+

.. |vcmSi-FSM| animation:: _images/vcmSi-FSM
.. |vcmSi-FootprintP| animation:: _images/vcmSi-FootprintP
.. |vcmIr-FSM| animation:: _images/vcmIr-FSM
   :loc: upper-right-corner
.. |vcmIr-FootprintP| animation:: _images/vcmIr-FootprintP
   :loc: upper-right-corner

Bending of collimating mirror
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shown are i) image downstream of the DCM and ii) image at the sample.

.. animation:: _images/vcmR-DCM
.. animation:: _images/vcmR-Sample

Bending of focusing mirror
~~~~~~~~~~~~~~~~~~~~~~~~~~

Shown is the image at the sample position.

.. animation:: _images/vfmR-Sample

Both mirrors (collimating + focusing) at varying pitch angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The sagittal radius of the focusing toroid mirror is optimal at 2 mrad pitch
angle.

.. animation:: _images/pitch-Sample

.. _dmm:

Scanning of Double Crystal Monochromator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------+-------------+
|   |Si111|   |   |Si311|   |
+-------------+-------------+

.. |Si111| animation:: _images/Si111
.. |Si311| animation:: _images/Si311

Scanning of Double Multilayer Monochromator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows a scan of a double multilayer monochromator – analog of a
double crystal monochromator. On the left is the footprint on the 1st
multilayer. On the right is a transversal image of the exit beam. The two
multilayers are equal and have 40 pairs of 27-Å-thick silicon on 18-Å-thick
tungsten layers on top of a silicon substrate.

+-------------+-------------+
|   |MLfp|    |  |MLexit|   |
+-------------+-------------+

.. |MLfp| animation:: _images/multilayer_1stML
.. |MLexit| animation:: _images/multilayer_afterDMM
"""
pass
