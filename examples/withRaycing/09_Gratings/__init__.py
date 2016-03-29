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

+--------+---------+---------+
|  |GM|  |  |ygE|  |  |ygO|  |
+--------+---------+---------+
|  |GS|  |  |xgE|  |  |xgO|  |
+--------+---------+---------+

.. |GM| image:: _images/GratingM.png
   :scale: 25 %
.. |ygE| image:: _images/y-gratingE.png
   :scale: 50 %
.. |ygO| image:: _images/y-gratingOrder.png
   :scale: 50 %
.. |GS| image:: _images/GratingS.png
   :scale: 25 %
.. |xgE| image:: _images/x-gratingE.png
   :scale: 50 %
.. |xgO| image:: _images/x-gratingOrder.png
   :scale: 50 %

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
|     |zFZP| |zFZPZ|        |  |fFZP| |fFZPZ|  |
+---------------------------+------------------+

.. |zFZP| image:: _images/zFZP.swf
   :width: 360
   :height: 333
.. |zFZPZ| image:: _images/zoomIcon.png
   :width: 20
   :target: _images/zFZP.swf
.. |fFZP| image:: _images/fFZP.swf
   :width: 234
   :height: 205
.. |fFZPZ| image:: _images/zoomIcon.png
   :width: 20
   :target: _images/fFZP.swf

Bragg-Fresnel optics
~~~~~~~~~~~~~~~~~~~~

One can combine an arbitrarily curved crystal surface, also (variably)
asymmetrically cut, with a grating or zone structure on top of it. The
following example shows a Fresnel zone structure that focuses a collimated beam
at *q* = 20 m, whereas the Bragg crystal provides good energy resolution. One
can easily study how the band width affects the focusing properties (not shown
here).

.. image:: _images/BFZPlocalFull.png
   :scale: 50 %
.. image:: _images/BFZPlocal.png
   :scale: 50 %
.. image:: _images/fBraggFresnel.swf
   :width: 309
   :height: 205
.. image:: _images/zoomIcon.png
   :width: 20
   :target: _images/fBraggFresnel.swf

Generic cPGM beamline
~~~~~~~~~~~~~~~~~~~~~

.. image:: _images/FlexPES.png
   :scale: 50 %

This example shows a generic cPGM beamline aligned for a fixed focus regime.
The angles at the mirrors equal 2 degrees, *c*\ :sub:`ff` = 2.25, the line
density is 1221 mm\ :sup:`-1`\ .

An energy scan at a given vertical slit (here, 30 µm) between M3 and M4. Shown
are images at the slit and at the final focus 'Exp2':

.. image:: _images/FlexPES-energyScanAtSlit.swf
   :width: 310
   :height: 205
.. image:: _images/zoomIcon.png
   :width: 20
   :target: _images/FlexPES-energyScanAtSlit.swf
.. image:: _images/FlexPES-energyScan.swf
   :width: 320
   :height: 205
.. image:: _images/zoomIcon.png
   :width: 20
   :target: _images/FlexPES-energyScan.swf

A vertical slit scan at a given energy (here, 40 eV) with a final dependency of
energy resolution and flux on the slit size:

.. image:: _images/FlexPES-slitScan.swf
   :width: 320
   :height: 205
.. image:: _images/zoomIcon.png
   :width: 20
   :target: _images/FlexPES-slitScan.swf
.. image:: _images/FlexPES-dE.png
   :scale: 50 %

"""
pass
