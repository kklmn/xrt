# -*- coding: utf-8 -*-
u"""
.. _glow_notes:

Notes on using xrtGlow
----------------------

.. imagezoom:: _images/xrtGlow1.png

- 3D glasses button is a two-state button. When it is pressed, xrtGlow will
  update its view whenever changes are made to the beamline in xrtQook. If you
  close the window of xrtGlow and the button is pressed, xrtGlow will pop up
  again after any change in xrtQook. To really close xrtGlow, deactivate the
  button.

- The Navigation panel of xrtGlow has several columns. The last columns may be
  hidden in the initial view. You can access them by enlarging the window.

- Load the example `.../xrtQook/savedBeamlines/lens3.xml` and follow the
  instructions in Description tab in order to understand the visualization
  precision vs the swiftness of the 3D manipulations.

- From xrtGlow, press F1 to see the available keyboard shortcuts. Also observe
  the available pop-up menu by right mouse click.

- The color histogram without Virtual Screen shows the color map -- the
  correspondence between the selected physical parameter (e.g. energy) and the
  colors. With Virtual Screen active (by F3), the plot shows a histogram of the
  selected parameter as distributed on Virtual Screen. In both cases the user
  may select a sub-band on the color plot by the mouse. The vertical extent in
  that selection is irrelevant.

"""
