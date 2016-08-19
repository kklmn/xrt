# -*- coding: utf-8 -*-
r"""
Quarter wave plates
-------------------

Files in ``\examples\withRaycing\05_QWP``

Collimated beam, Bragg transmission case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows the polarization properties of a collimated linearly
polarized beam passed through a diamond plate at various departure from the
nominal Bragg angle. The polarization plate is put at 45º to the diffraction
plane. Notice that the phase difference between the s- and p-polarized
components was calculated here not in the 1-field approximation as elsewhere
[Malgrange]_ that has a pole at :math:`\Delta\theta=0` but in the general
2-field approximation, see :mod:`~xrt.backends.raycing.materials`.

.. [Malgrange] C. Giles, C. Malgrange, J. Goulon, F. de Bergevin, C. Vettier,
    E. Dartyge, A. Fontaine, C. Giorgetti and S. Pizzini, J. Appl. Cryst.
    **27** (1994) 232;
    C. Giles, C. Vettier, F. de Bergevin, C. Malgrange, G. Grübel, and
    F. Grossi, Rev. Sci. Instrum. **66** (1995) 1518;
    J. Goulon, C. Malgrange, C. Giles, C. Neumann, A. Rogalev, E. Moguiline,
    F. De Bergevin and C. Vettier, J. Synchrotron Rad. **3** (1996) 272.

Beam images after the QWP with color axis as 1) energy, 2) circular
polarization rate, 3) phase shift between s- and p-components and 4) ratio of
axes of the polarization ellipse. Watch how the circular polarization rate
becomes close to 1 or -1 at certain departure angles; here, between 16 and 32
arcsec (plus or minus). At the same time also the ratio of axes of the
polarization ellipse becomes close to 1 or -1 and with narrow distribution.

*E* ~ 9 keV, crystal thickness = 200 µm.

+------------+------------+------------+------------+
|  |QWPBTE|  |  |QWPBTC|  |  |QWPBTP|  |  |QWPBTA|  |
+------------+------------+------------+------------+

.. |QWPBTE| animation:: _images/QWP-1BT-E
.. |QWPBTC| animation:: _images/QWP-1BT-CircPolRate
.. |QWPBTP| animation:: _images/QWP-1BT-PhaseShift
   :loc: upper-right-corner
.. |QWPBTA| animation:: _images/QWP-1BT-PolAxesRatio
   :loc: top: -208px; left: -450px;

Collimated beam, Laue transmission case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same as in the revious subsection but for the Laue case. The thickness of the
crystal was selected as to give the path length similar to that in the Bragg
case.

*E* ~ 9 keV, crystal thickness = 500 µm.

+------------+------------+------------+------------+
|  |QWPLTE|  |  |QWPLTC|  |  |QWPLTP|  |  |QWPLTA|  |
+------------+------------+------------+------------+

.. |QWPLTE| animation:: _images/QWP-2LT-E
.. |QWPLTC| animation:: _images/QWP-2LT-CircPolRate
.. |QWPLTP| animation:: _images/QWP-2LT-PhaseShift
   :loc: upper-right-corner
.. |QWPLTA| animation:: _images/QWP-2LT-PolAxesRatio
   :loc: top: -208px; left: -450px;

Convergent beam, Laue transmission case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example the beam is focused by a toroidal mirror. Due to the large
angular variation in the beam (3 mrad at the position of the QWP), the
resulting circular polarization rate is low. The initial polarization is
horizontal and the diffraction plane of the QWP is turned by 45º from vertical.

*E* ~ 9 keV, crystal thickness = 500 µm.

.. animation:: _images/QWP-3LT-conv-CircPolRate

Convergent beam, bent-Laue transmission case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One way to improve the low circular polarization rate in a divergent or
convergent beam is to aperture it, with the obvious disadvantage of lowering
the beam flux. Another, less obvious, way is to bend the crystal with the
radius equal to the distance from the sample (focal point) to the QWP. In this
particular example the Laue case is considered. The bending is done
cylindrically in the diffraction plane which is at 45º to the initial
polarization plane (horizontal). Watch the circular polarization rate at
:math:`\pm` 32 arcsec departure angle.

*E* ~ 9 keV, crystal thickness = 500 µm.

.. animation:: _images/QWP-4LT-conv-bent-CircPolRate
"""
pass
