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

+-------------+-------------+
|  |montel1|  |  |montel2|  |
+-------------+-------------+

.. |montel1| imagezoom:: _images/Montel_par_exit_cat.png
.. |montel2| imagezoom:: _images/Montel_par_exit_n.png
   :loc: upper-right-corner

In the present example one can visualize the local footprints on either of the
mirrors. The footprints are colored by the number of reflections.

+-------------+-------------+
|  |montel3|  |  |montel4|  |
+-------------+-------------+

.. |montel3| imagezoom:: _images/Montel_par_localHFM_n.png
.. |montel4| imagezoom:: _images/Montel_par_localVFM_n.png
   :loc: upper-right-corner

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

+----------+----------+
|  |cap1|  |  |cap2|  |
+----------+----------+

.. |cap1| imagezoom:: _images/NCapillaries-a-FSM1Cat.png
.. |cap2| imagezoom:: _images/PolycapillaryZ0crosssection.png
   :loc: upper-right-corner

Each capillary reflects the rays many times until they are out. The local
footprints over the capillary surface are vs. the polar angle and the
longitudinal coordinate. The coloring is histogrammed over incidence angle and
number of reflections. The longitudinal coordinate *s* has its zero at the exit
from the capillary.

+--------+----------------------+----------------------+----------------------+
|        |       layer 1        |        layer 6       |       layer 12       |
+========+======================+======================+======================+
| n refl |         |cN0|        |         |cN5|        |        |cN11|        |
+--------+----------------------+----------------------+----------------------+
|   θ    |       |cTheta0|      |       |cTheta5|      |      |cTheta11|      |
+--------+----------------------+----------------------+----------------------+

.. |cN0| imagezoom:: _images/NCapillaries-b-Local0N.png
.. |cTheta0| imagezoom:: _images/NCapillaries-c-Local0Theta.png
.. |cN5| imagezoom:: _images/NCapillaries-b-Local5N.png
   :loc: upper-right-corner
.. |cTheta5| imagezoom:: _images/NCapillaries-c-Local5Theta.png
   :loc: upper-right-corner
.. |cN11| imagezoom:: _images/NCapillaries-b-Local11N.png
   :loc: upper-right-corner
.. |cTheta11| imagezoom:: _images/NCapillaries-c-Local11Theta.png
   :loc: upper-right-corner

At the exit from the polycapillary the beam is expanded and attenuated at the
periphery. The attenuation is due to the losses at each reflection; the number
of reflections increases at the periphery, as shown by the colored histogram
below. The phase space of the exit beam (shown is the horizontal one) shows the
quality of collimation, see below. The divergence of ~1 mrad is large and
cannot be efficiently used with a flat crystal analyzer.

+----------+----------+
|  |cap3|  |  |cap4|  |
+----------+----------+

.. |cap3| imagezoom:: _images/NCapillaries-e-FSM2-xzN.png
.. |cap4| imagezoom:: _images/NCapillaries-f-FSM2-xPhaseSpaceN.png
   :loc: upper-right-corner

One may attempt to add a second-stage to collimate the beam with ~1 mrad
divergence. However, this would not work because the rays collimated by a
polycapillary have a very large distribution of the ray origins over the
longitudinal direction *y*, as shown below.

.. imagezoom:: _images/NCapillaries-g-CapillaryOut-depthX.png
   :align: center

"""
pass
