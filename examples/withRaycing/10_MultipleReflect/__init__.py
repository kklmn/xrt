# -*- coding: utf-8 -*-
r"""
Multiple reflections
--------------------

Files in ``\examples\withRaycing\10_MultipleReflect``

.. _montel:

Montel mirror
~~~~~~~~~~~~~

Montel mirror consists of two orthogonal mirrors positioned side-by-side. It is
very similar to a KB mirror but more compact in the longitudinal direction. In
a Montel mirror a part of the incoming beam is first reflected by one side of
the mirror and then by the other side. Another part of the beam has the
opposite reflection sequence. There are also single reflections on either side
of the mirror. The non-sequential way of reflections makes such ray tracing
impossible in most of ray-tracing programs.

The images below show the result of reflection by a Montel mirror consisting of
a pair of parabolic mirrors. The mirrors can be selected by the user to have
another shape. The coloring is by categories and the number of reflections.
Notice a gap between the mirrors (here 0.2 mm) that transforms into a diagonal
gap in the final image.

.. image:: _images/Montel_par_exit_cat.png
   :scale: 50 %
.. image:: _images/Montel_par_exit_n.png
   :scale: 50 %

In the present example one can visualize the local footprints on either of the
mirrors. The footprints are colored by the number of reflections.

.. image:: _images/Montel_par_localHFM_n.png
   :scale: 50 %
.. image:: _images/Montel_par_localVFM_n.png
   :scale: 50 %

.. _polycapillary:

Polycapillary
~~~~~~~~~~~~~

This example also demonstrates the technique of propagating the beam through an
array of non-sequential optical elements. These are 397 glass capillaries,
close packed into a hexagonal bunch. The capillaries here serve for collimating
a divergent beam (fluorescence) born at the origin. Each capillary here is
parabolically (user-supplied shape) bent such that the left end tangents are
directed towards the source and the right ends are parallel to the *y* axis.
The capillary radius here is constant but can also be given by a user-function.
The images below show the geometry of the polycapillary: the screen at the
entrance that is colored by the categories of the exit beam (the green rays are
reflected by the capillaries, the orange ones are transmitted without any
reflection and the red ones are absorbed) and a longitudinal cross-section
(note very different scales of *x* and *y* axes).

.. image:: _images/NCapillaries-a-FSM1Cat.png
   :scale: 50 %
.. image:: _images/PolycapillaryZ0crosssection.png
   :scale: 50 %

Each capillary reflects the rays many times until they are out. The local
footprints over the capillary surface are vs. the polar angle and the
longitudinal coordinate. The coloring is histogrammed over incidence angle and
number of reflections. The longitudinal coordinate *s* has its zero at the exit
from the capillary.

+--------+-----------+-----------+------------+
|        |  layer 1  |  layer 6  |  layer 12  |
+========+===========+===========+============+
| n refl |   |cN0|   |   |cN5|   |   |cN11|   |
+--------+-----------+-----------+------------+
|   θ    | |cTheta0| | |cTheta5| | |cTheta11| |
+--------+-----------+-----------+------------+

.. |cN0| image:: _images/NCapillaries-b-Local0N.png
   :scale: 40 %
.. |cTheta0| image:: _images/NCapillaries-c-Local0Theta.png
   :scale: 40 %
.. |cN5| image:: _images/NCapillaries-b-Local5N.png
   :scale: 40 %
.. |cTheta5| image:: _images/NCapillaries-c-Local5Theta.png
   :scale: 40 %
.. |cN11| image:: _images/NCapillaries-b-Local11N.png
   :scale: 40 %
.. |cTheta11| image:: _images/NCapillaries-c-Local11Theta.png
   :scale: 40 %

At the exit from the polycapillary the beam is expanded and attenuated at the
periphery. The attenuation is due to the losses at each reflection; the number
of reflections increases at the periphery, as shown by the colored histogram
below. The phase space of the exit beam (shown is the horizontal one) shows the
quality of collimation, see below. The divergence of ~1 mrad is large and
cannot be efficiently used with a flat crystal analyzer.

.. image:: _images/NCapillaries-e-FSM2-xzN.png
   :scale: 50 %
.. image:: _images/NCapillaries-f-FSM2-xPhaseSpaceN.png
   :scale: 50 %

One may attempt to add a second-stage to collimate the beam with ~1 mrad
divergence. However, this would not work because the rays collimated by a
polycapillary have a very large distribution of the ray origins over the
longitudinal direction *y*, as shown below.

.. image:: _images/NCapillaries-g-CapillaryOut-depthX.png
   :scale: 50 %

"""
pass
