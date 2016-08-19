# -*- coding: utf-8 -*-
r"""
Comparison of 1D-bent crystal analyzers
---------------------------------------

Files in ``\examples\withRaycing\06_AnalyzerBent1D``

Rowland circle based analyzers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This study compares simply bent and ground-bent spectrometers utilizing Bragg
and Laue crystals. The bending is cylindrical (one-dimensional).

.. imagezoom:: _images/BraggLaue.*

:Conditions: Rowland circle diameter = 1 m, 70v × 200h µm² unpolarized
   fluorescence source, crystal size = 100meridional × 20saggittal mm².

The energy resolution was calculated as described in `the CDR of a diced
Johansson-like spectrometer at Alba/CLÆSS beamline
<http://www.cells.es/Beamlines/CLAESS/EXD-BL22-FA-0001v4.0.pdf>`_. This
requires two images: 1) of a flat energy distribution source and 2) of a
monochromatic source. The image is energy dispersive in the diffraction plane,
which can be used in practice with a position sensitive detector or with a slit
scan in front of a bulk detector. From these two images the energy resolution
*δE* was calculated and then 3) a verifying image was ray-traced for a source
of 7 energy lines evenly spaced with the found step *δE*. Such images are shown
for the four crystal geometries at a particular Bragg angle:

+----------+---------------------+---------------------+---------------------+
| geometry |     flat source     |     line source     |       7 lines       |
+==========+=====================+=====================+=====================+
| Bragg    |                     |                     |                     |
| simply   |      |bb_flat|      |      |bb_line|      |      |bb_7lin|      |
| bent     |                     |                     |                     |
+----------+---------------------+---------------------+---------------------+
| Bragg    |                     |                     |                     |
| ground   |      |bg_flat|      |      |bg_line|      |      |bg_7lin|      |
| bent     |                     |                     |                     |
+----------+---------------------+---------------------+---------------------+
| Laue     |                     |                     |                     |
| simply   |      |lb_flat|      |      |lb_line|      |      |lb_7lin|      |
| bent     |                     |                     |                     |
+----------+---------------------+---------------------+---------------------+
| Laue     |                     |                     |                     |
| ground   |      |lg_flat|      |      |lg_line|      |      |lg_7lin|      |
| bent     |                     |                     |                     |
+----------+---------------------+---------------------+---------------------+

.. |bb_flat| imagezoom:: _images/1D-01b-Si444-60-det_E-flat.*
.. |bb_line| imagezoom:: _images/1D-01b-Si444-60-det_E-line.*
.. |bb_7lin| imagezoom:: _images/1D-01b-Si444-60-det_E-7lin.*
   :loc: upper-right-corner
.. |bg_flat| imagezoom:: _images/1D-02gb-Si444-60-det_E-flat.*
.. |bg_line| imagezoom:: _images/1D-02gb-Si444-60-det_E-line.*
.. |bg_7lin| imagezoom:: _images/1D-02gb-Si444-60-det_E-7lin.*
   :loc: upper-right-corner
.. |lb_flat| imagezoom:: _images/1D-03lb-Si444-60-det_E-flat.*
.. |lb_line| imagezoom:: _images/1D-03lb-Si444-60-det_E-line.*
.. |lb_7lin| imagezoom:: _images/1D-03lb-Si444-60-det_E-7lin.*
   :loc: upper-right-corner
.. |lg_flat| imagezoom:: _images/1D-04lgb-Si444-60-det_E-flat.*
.. |lg_line| imagezoom:: _images/1D-04lgb-Si444-60-det_E-line.*
.. |lg_7lin| imagezoom:: _images/1D-04lgb-Si444-60-det_E-7lin.*
   :loc: upper-right-corner

The energy distribution over the crystal surface is hyperbolic for Bragg and
ellipsoidal for Laue crystals. Therefore, Laue crystals have limited acceptance
in the sagittal direction whereas Bragg crystals have the hyperbola branches
even for large sagittal sizes. Notice the full crystal coverage in the
meridional direction for the two ground-bent cases.

+----------+---------------------+---------------------+---------------------+
| geometry |     flat source     |     line source     |       7 lines       |
+==========+=====================+=====================+=====================+
| Bragg    |                     |                     |                     |
| simply   |      |xbb_flat|     |      |xbb_line|     |      |xbb_7lin|     |
| bent     |                     |                     |                     |
+----------+---------------------+---------------------+---------------------+
| Bragg    |                     |                     |                     |
| ground   |      |xbg_flat|     |      |xbg_line|     |      |xbg_7lin|     |
| bent     |                     |                     |                     |
+----------+---------------------+---------------------+---------------------+
| Laue     |                     |                     |                     |
| simply   |      |xlb_flat|     |      |xlb_line|     |      |xlb_7lin|     |
| bent     |                     |                     |                     |
+----------+---------------------+---------------------+---------------------+
| Laue     |                     |                     |                     |
| ground   |      |xlg_flat|     |      |xlg_line|     |      |xlg_7lin|     |
| bent     |                     |                     |                     |
+----------+---------------------+---------------------+---------------------+

.. |xbb_flat| imagezoom:: _images/1D-01b-Si444-60-xtal_E-flat.*
.. |xbb_line| imagezoom:: _images/1D-01b-Si444-60-xtal_E-line.*
.. |xbb_7lin| imagezoom:: _images/1D-01b-Si444-60-xtal_E-7lin.*
   :loc: upper-right-corner
.. |xbg_flat| imagezoom:: _images/1D-02gb-Si444-60-xtal_E-flat.*
.. |xbg_line| imagezoom:: _images/1D-02gb-Si444-60-xtal_E-line.*
.. |xbg_7lin| imagezoom:: _images/1D-02gb-Si444-60-xtal_E-7lin.*
   :loc: upper-right-corner
.. |xlb_flat| imagezoom:: _images/1D-03lb-Si444-60-xtal_E-flat.*
.. |xlb_line| imagezoom:: _images/1D-03lb-Si444-60-xtal_E-line.*
.. |xlb_7lin| imagezoom:: _images/1D-03lb-Si444-60-xtal_E-7lin.*
   :loc: upper-right-corner
.. |xlg_flat| imagezoom:: _images/1D-04lgb-Si444-60-xtal_E-flat.*
.. |xlg_line| imagezoom:: _images/1D-04lgb-Si444-60-xtal_E-line.*
.. |xlg_7lin| imagezoom:: _images/1D-04lgb-Si444-60-xtal_E-7lin.*
   :loc: upper-right-corner

As a matter of principles checking, let us consider how the initially
unpolarized beam becomes partially polarized after being diffracted by the
crystal analyzer. As expected, the beam is fully polarized at 45° Bragg angle
(Brewster angle in x-ray regime). CAxis here is degree of polarization:

+-------------+-------------+
|    Bragg    |    Laue     |
+=============+=============+
|  |DPBragg|  |  |DPLaue|   |
+-------------+-------------+

.. |DPBragg| animation:: _images/1D-DegOfPol_Bragg
.. |DPLaue| animation:: _images/1D-DegOfPol_Laue

.. rubric:: Comments

1) The ground-bent crystals are more efficient as the whole their surface works
   for a single energy, as opposed to simply bent crystals which have different
   parts reflecting the rays of different energies.
2) When the crystal is close to the source (small θ for Bragg and large θ for
   Laue), the images are distorted, even for the ground-bent crystals.
3) The Bragg case requires small pixel size in the meridional direction (~10 µm
   for 1-m-diameter Rowland circle) for a good spatial resolution but can
   profit from its compactness. The Laue case requires a big detector of a size
   comparable to that of the crystal but the pixel size is not required to be
   small.
4) The comparison of energy resolution in Bragg and Laue cases is not strictly
   correct here. While the former case can use the small beam size at the
   detector for utilizing energy dispersive property of the spectrometer, the
   latter one has a big image at the detector which is restricted by the size
   of the crystal. The size of the 'white' beam image is therefore correct only
   for the crystal size selected here. The Laue case can still be used in
   energy dispersive regime if 2D image analysis is utilized. At the present
   conditions, the energy resolution of Bragg crystals is better than that of
   Laue crystals except at small Bragg angles and low diffraction orders.
5) The energy resolution in ground-bent cases is not always better than that
   in simply bent cases because of strongly curved images. If the sagittal size
   of the crystal is smaller or :ref:`sagittal bending is used
   <dicedBentAnalyzers>`, the advantage of ground-bent crystals is clearly
   visible not only in terms of efficiency but also in terms of energy
   resolution.

.. _VonHamos:

Von Hamos analyzer
~~~~~~~~~~~~~~~~~~

A von Hamos spectrometer has axial symmetry around the axis connecting the
source and the detector. The analyzing crystal is cylindrically bent with the
radius equal to the crystal-to-axis distance. In this scheme, the emission
escape direction depends on the Bragg angle (energy). In practice, the
spectrometer axis is adapted such that the escape direction is appropriate for
a given sample setup. In particular, the escape direction can be kept in back
scattering (relatively to the sample), see the figure below. In the latter case
the mechanical model is more complex and includes three translations and two
rotations. In the figure below, the crystal is sagittally curved around the
source–detector line. The detector plane is perpendicular to the sketch.
Left: the classical setup [vH]_ with 2 translations.
Right: the setup with an invariant escape direction.

.. imagezoom:: _images/vonHamosPositionsClassic.*
.. imagezoom:: _images/vonHamosPositionsFixedEscape.*

The geometrical parameters for the von Hamos spectrometer were taken from
[vH_SLS]_: a diced 100 (sagittal) × 50 (meridional) mm² Si(444) crystal is
curved with Rs = 250 mm. The width of segmented facets was taken equal to 5 mm
(as in [vH_SLS]_) and 1 mm together with a continuously bent case.

.. [vH] L. von Hámos, *Röntgenspektroskopie und Abbildung mittels gekrümmter
   Kristallreflektoren II. Beschreibung eines fokussierenden Spektrographen mit
   punktgetreuer Spaltabbildung*, Annalen der Physik **411** (1934) 252–260

.. [vH_SLS] J. Szlachetko, M. Nachtegaal, E. de Boni, M. Willimann,
   O. Safonova, J. Sa, G. Smolentsev, M. Szlachetko, J. A. van Bokhoven,
   J.-Cl. Dousse, J. Hoszowska, Y. Kayser, P. Jagodzinski, A. Bergamaschi,
   B. Schmitt, C. David, and A. Lücke, *A von Hamos x-ray spectrometer based on
   a segmented-type diffraction crystal for single-shot x-ray emission
   spectroscopy and time-resolved resonant inelastic x-ray scattering studies*,
   Rev. Sci. Instrum. **83** (2012) 103105.

The calculation of energy resolution requires two detector images: 1) of a flat
energy distribution source and 2) of a monochromatic source. From these two
images the energy resolution *δE* was calculated and then 3) a verifying image
was ray-traced for a source of 7 energy lines evenly spaced with the found step
*δE*. Such images are shown for different dicing sizes at a particular Bragg
angle.

+---------+--------------------+--------------------+--------------------+
| crystal |     flat source    |     line source    |       7 lines      |
+=========+====================+====================+====================+
| diced   |                    |                    |                    |
| 5 mm    |     |vH5_flat|     |     |vH5_line|     |     |vH5_7lin|     |
+---------+--------------------+--------------------+--------------------+
| diced   |                    |                    |                    |
| 1 mm    |     |vH1_flat|     |     |vH1_line|     |     |vH1_7lin|     |
+---------+--------------------+--------------------+--------------------+
| not     |                    |                    |                    |
| diced   |     |vHc_flat|     |     |vHc_line|     |     |vHc_7lin|     |
+---------+--------------------+--------------------+--------------------+

.. |vH5_flat| imagezoom:: _images/SivonHamos-5mmDiced60-det_E-flat.*
.. |vH5_line| imagezoom:: _images/SivonHamos-5mmDiced60-det_E-line.*
.. |vH5_7lin| imagezoom:: _images/SivonHamos-5mmDiced60-det_E-7lin.*
   :loc: upper-right-corner

.. |vH1_flat| imagezoom:: _images/SivonHamos-1mmDiced60-det_E-flat.*
.. |vH1_line| imagezoom:: _images/SivonHamos-1mmDiced60-det_E-line.*
.. |vH1_7lin| imagezoom:: _images/SivonHamos-1mmDiced60-det_E-7lin.*
   :loc: upper-right-corner

.. |vHc_flat| imagezoom:: _images/SivonHamos-notDiced60-det_E-flat.*
.. |vHc_line| imagezoom:: _images/SivonHamos-notDiced60-det_E-line.*
.. |vHc_7lin| imagezoom:: _images/SivonHamos-notDiced60-det_E-7lin.*
   :loc: upper-right-corner

With the coloring by stripe (crystal facet) number, the image below explains
why energy resolution is worse when stripes are wider and the crystal is
sagittally larger. The peripheral stripes contribute to aberrations which
increase the detector image.

+------------+--------------------------------------+
|  crystal   | line source colored by stripe number |
+============+======================================+
| diced 5 mm |      |vH5_line_stripes|              |
+------------+--------------------------------------+
| diced 1 mm |      |vH1_line_stripes|              |
+------------+--------------------------------------+

.. |vH5_line_stripes| imagezoom:: _images/SivonHamos-5mmDiced60-det_stripes-line.*
.. |vH1_line_stripes| imagezoom:: _images/SivonHamos-1mmDiced60-det_stripes-line.*


The efficiency of a von Hamos spectrometer is significantly lower as compared
to Johann and Johansson crystals. The reason for the lower efficiency can be
understood from the figure below, where the magnified footprint on the crystal
is shown: only a narrow part of the crystal surface contributes to a given
energy band. Here, in the 5-mm-stripe case a bandwidth of ~12 eV uses less than
1 mm of the crystal!

.. imagezoom:: _images/SivonHamos-5mmDiced60-xtal_E_zoom-7lin.*

Comparison of Rowland circle based and von Hamos analyzers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An additional case was also included here: when a Johann crystal is rotated by
90⁰ around the sample-to-crystal line, it becomes a von Hamos crystal that has
to be put at a correct distance corresponding to the 1 m sagittal radius. This
case is labelled as “Johann as von Hamos”.

In comparing with a von Hamos spectrometer, one should realize its strongest
advantage – inherent energy dispersive operation without a need for energy
scan. This advantage is especially important for broad emission lines. Below,
the comparison is made for two cases: (1) a narrow energy band (left figure),
which is more interesting for valence band RIXS and which assumes a high
\resolution monochromator in the primary beam and (2) a wide energy band (right
figure), which is more interesting for core state RIXS and normal fluorescence
detection. The desired position on the charts is in the upper left corner. As
seen in the figures, the efficiency of the von Hamos crystals (i) is
independent of the energy band (equal for the left and right charts), which
demonstrates truly energy-dispersive behavior of the crystals but (ii) is
significantly lower as compared to the Johann and Johansson crystals. A way to
increase efficiency is to place the crystal closer to the source, which
obviously worsens energy resolution because of the increased angular source
size. Inversely, if the crystal is put at a further distance, the energy
resolution is improved (square symbols) but the efficiency is low because of a
smaller solid angle collected. The left figure is with a narrow energy band
equal to the 6-fold energy resolution. The right figure is with a wide energy
band equal to 8·10 :sup:`-4`·E (approximate width of K β lines [Henke]_).

.. imagezoom:: _images/ResolutionEfficiency1D-narrowBand.*
.. imagezoom:: _images/ResolutionEfficiency1D-8e-4Band.*

Finally, among the compared 1D-bent spectrometers the Johansson type is the
best in the combination of good energy resolution and high efficiency. It is
the only one that can function both as a high resolution spectrometer and a
fluorescence detector. One should bear in mind, however, two very strong
advantages of von Hamos spectrometers: (1) they do not need alignment – a
crystal and a detector positioned approximately will most probably immediately
work and (2) the image is inherently energy dispersive with a flat (energy
independent) detector response. The low efficiency and mediocre energy
resolution are a price for the commissioning-free energy dispersive operation.
Rowland circle based spectrometers will always require good alignment, and
among them only the Johansson-type spectrometer can be made energy dispersive
with a flat detector response.

.. _elliptical_VonHamos:

Circular and elliptical von Hamos analyzers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The axial symmetry of the classical von Hamos spectrometer [vH]_ results in a
close detector-to-sample position at large Bragg angles. A single detector may
find enough space there but when the spectrometer has several branches, the
corresponding detectors come close to each other, which restricts both the
space around the sample and the accessible Bragg angle range. A solution to
this problem could be an increased magnification of the crystal from the
classical 1:1. The axis of the circular cilynder is then split into *two* axes
representing the two foci of an ellipsoid, see the scheme below. The lower axis
holds the source (sample) and the upper one holds the detector. The crystal in
the figure has the magnification 1:1 for the circular crystal (left part) and
1.5:1 for the elliptical one (right part).

.. imagezoom:: _images/CircularAndElliptical_vonHamos_s.*

The crystal is diced along the cylinder axis with 1 mm pitch. The difference in
the circular and elliptical figures is shown below.

.. imagezoom:: _images/Cylinders.*

The elliptical figure results in some aberrations, as seen by the monochromatic
images below, which worsens energy resolution.

+------------+--------------------+--------------------+--------------------+
| crystal    |    flat source     |     line source    |      7 lines       |
+============+====================+====================+====================+
| bent as    |                    |                    |                    |
| circular   |     |circ_flat|    |     |circ_line|    |     |circ_7lin|    |
| cylinder   |                    |                    |                    |
+------------+--------------------+--------------------+--------------------+
| bent as    |                    |                    |                    |
| elliptical |     |ell_flat|     |     |ell_line|     |     |ell_7lin|     |
| cylinder   |                    |                    |                    |
+------------+--------------------+--------------------+--------------------+

.. |circ_flat| imagezoom:: _images/SivonHamosDicedCircular60-det_E-flat.*
.. |circ_line| imagezoom:: _images/SivonHamosDicedCircular60-det_E-line.*
.. |circ_7lin| imagezoom:: _images/SivonHamosDicedCircular60-det_E-7lin.*
   :loc: upper-right-corner
.. |ell_flat| imagezoom:: _images/SivonHamosDicedElliptical60-det_E-flat.*
.. |ell_line| imagezoom:: _images/SivonHamosDicedElliptical60-det_E-line.*
.. |ell_7lin| imagezoom:: _images/SivonHamosDicedElliptical60-det_E-7lin.*
   :loc: upper-right-corner

"""
pass
