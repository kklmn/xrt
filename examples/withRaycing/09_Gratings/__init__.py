# -*- coding: utf-8 -*-
r"""
Gratings, FZPs, Bragg-Fresnel optics, cPGM beamline
---------------------------------------------------

Files in ``\examples\withRaycing\09_Gratings``

Simple gratings
~~~~~~~~~~~~~~~

The following pictures exemplify simple gratings with the dispersion vector a)
in the meridional plane and b) orthogonal to the meridional plane. Coloring is
done by energy and by diffraction order.

+--------+-----------------+-----------------+
|  |GM|  |      |ygE|      |      |ygO|      |
+--------+-----------------+-----------------+
|  |GS|  |      |xgE|      |      |xgO|      |
+--------+-----------------+-----------------+

.. |GM| imagezoom:: _images/GratingM.png
   :scale: 25 %
.. |ygE| imagezoom:: _images/y-gratingE.png
.. |ygO| imagezoom:: _images/y-gratingOrder.png
   :loc: upper-right-corner
.. |GS| imagezoom:: _images/GratingS.png
   :scale: 25 %
.. |xgE| imagezoom:: _images/x-gratingE.png
.. |xgO| imagezoom:: _images/x-gratingOrder.png
   :loc: upper-right-corner

Fresnel Zone Plate
~~~~~~~~~~~~~~~~~~

This example shows focusing of a quasi-monochromatic collimated beam by a
normal (orthogonal to the beam) FZP. The energy distribution is uniform within
400 ± 5 eV. The focal length is 2 mm. The phase shift in the zones is variable
and is relative to the central ray. As expected, the phase shift does not
influence the focusing properties and can be selected at will.

+---------------------------+------------------+
|  zoomed footprint on FZP  |   focal spot     |
+===========================+==================+
|          |FZPz|           |      |FZPf|      |
+---------------------------+------------------+

.. |FZPz| animation:: _images/FZPz
.. |FZPf| animation:: _images/FZPf

Bragg-Fresnel optics
~~~~~~~~~~~~~~~~~~~~

One can combine an arbitrarily curved crystal surface, also (variably)
asymmetrically cut, with a grating or zone structure on top of it. The
following example shows a Fresnel zone structure that focuses a collimated beam
at *q* = 20 m, whereas the Bragg crystal provides good energy resolution. One
can easily study how the band width affects the focusing properties (not shown
here).

.. imagezoom:: _images/BFZPlocalFull.png
.. imagezoom:: _images/BFZPlocal.png
.. animation:: _images/BraggFresnel

Generic cPGM beamline
~~~~~~~~~~~~~~~~~~~~~

.. imagezoom:: _images/FlexPES.png

This example shows a generic cPGM beamline aligned for a fixed focus regime.
The angles at the mirrors equal 2 degrees, *c*\ :sub:`ff` = 2.25, the line
density is 1221 mm\ :sup:`-1`\ .

An energy scan at a given vertical slit (here, 30 µm) between M3 and M4. Shown
are images at the slit and at the final focus 'Exp2':

.. animation:: _images/FlexPES-energyScanAtSlit
.. animation:: _images/FlexPES-energyScan

A vertical slit scan at a given energy (here, 40 eV) with a final dependency of
energy resolution and flux on the slit size:

.. animation:: _images/FlexPES-slitScan
.. imagezoom:: _images/FlexPES-dE.png

"""
pass
