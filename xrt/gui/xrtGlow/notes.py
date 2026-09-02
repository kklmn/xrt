# -*- coding: utf-8 -*-
u"""
.. _glow_notes:

Notes on using xrtGlow
----------------------

.. imagezoom:: _images/xrtGlow1.png
   :align: right

- Examine a few examples in `.../examples/withRaycing/_QookBeamlines`.

- Export to image is available under the context menu -> File. You can save and
  load the scene settings (camera position, model orientation, rays opacity and
  so on) as well.

- From xrtGlow, press F1 to see the available keyboard shortcuts. Also observe
  the available pop-up menu by right mouse click.

- Movements of the model are possible with Shift-MouseLeft. Centering the scene
  can be done by (a) right click on the element name in Selection and then
  "Center here" or (b) right click on the element itself in the scene and then
  "Center view".

- The color histogram without Virtual Screen shows the color map -- the
  correspondence between the selected physical parameter (e.g. energy) and the
  colors. With Virtual Screen active (by F3), the plot shows a histogram of the
  selected parameter as distributed on Virtual Screen. In both cases the user
  may select a sub-band on the color plot by the mouse. The vertical extent in
  that selection is irrelevant.

- Examine dynamic properties of an optical elements or a screen by
  right-clicking it. The plot in the inspectror panel shows a local footprint or
  a screen view.

- Virtual Screen is instantiated by F3 nearly at the view center. It can be
  moved along the beamline by Ctrl-MouseLeft drag.

- Rays or footprints visualisation can be enabled/disabled either by setting
  corresponding checkboxes in the Navigation Panel for individual elements or
  globally by changing the opacity of the lines and points in the Color Panel.
  The same applies for the Projections.

- Intensity cut-off allows to omit the visualisation of the darkest/weakest
  rays. It is especially important if Intensity defines the Value key in HSV
  color space when dark rays can shadow the whole beam.

.. imagezoom:: _images/xrtGlow3.png
   :align: right

- A convenient way to inspect a detailed beam footprint on the coordinate grid
  is to use Projections: disable the Perspective, select only the footprint of
  interest on the Navigation Panel (or disable all and just leave the Virtual
  Screen on), enable the projection, set to zero the Projection Line Opacity
  (or Line Width, it will do the job too), increase the Projection Point
  Opacity to improve the visibility, enable the Fine Grid. Increase the number
  of rays in the source if necessary.

.. imagezoom:: _images/xrtGlow4.png
   :align: right

- If you have any doubts regarding the orientation of the optical element or
  trying to identify the directions, you can plot local coordinate axes by
  checking the corresponding option on the Scene panel or in the context menu.
  Make sure that the surface rendering is enabled for this element in the
  Navigation panel. Orientation of the diffraction planes will be represented
  by the yellow arrow in case of crystals with asymmetric cut.

- Depth test is disabled by default for Points. Enable it if you do not want the
  footprints to shine through solid surfaces of optical elements. Be aware that
  Points may be obscured by Lines (rays) in this case.

- Antialiasing can improve the visual quality of the scene, but it seriously
  affects the performance (depending on the number of rays / elements in the
  model), only enable it after all modifications to the scene are applied,
  prior the Export to file. Nevertheless antialiasing is always enabled for the
  coordinate grid.

- Default Zoom does not involve the coordinate grid, if you want to Zoom In/Out
  the whole scene, use Ctrl-MouseWheel.


"""
