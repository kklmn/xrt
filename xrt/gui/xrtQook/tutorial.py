# -*- coding: utf-8 -*-
u"""
.. _qook_tutorial:

Using xrtQook for script generation
-----------------------------------

- Start xrtQook: type ``python xrtQook.pyw`` from xrt/xrtQook or, if you have
  installed xrt by running setup.py, type ``xrtQook.pyw`` from any location.
- Rename beamLine to myTestBeamline by double clicking on it (you do not have
  to, only for demonstration).
- Right-click on myTestBeamline and Add Source → BendingMagnet.

  .. imagezoom:: _images/qookTutor1.png
     :scale: 60 %

- In its properties change eMin to 10000-1 and eMax to 10000+1. The middle of
  this range will be used to automatically align crystals (one crystal in this
  example). Blue color indicates non-default values. These will be necessarily
  included into the generated script. All the default-valued parameters do not
  propagate into the script.

  .. imagezoom:: _images/qookTutor2.png
     :scale: 60 %

- In Materials tab create a crystalline material:
  right click -> Add Material -> CrystalSi. This will create a Si111 crystal at
  room temperature.

  .. imagezoom:: _images/qookTutor3.png
     :scale: 60 %

- In Beamline tab right click -> Add OE -> OE. This will add an optical element
  with a flat surface.

  .. note::
     The sequence of the inserted optical elements does matter! This sequence
     determines the order of beam propagation.

  .. imagezoom:: _images/qookTutor4.png
     :scale: 60 %

- In its properties select the created crystal as 'material', put [0, 20000, 0]
  as 'center' (i.e. 20 m from source) and "auto" (with or without quotes) as
  'pitch'.
- Add a screen to the beamline. Give it [0, 21000, "auto"] as 'center'. Its
  height -- the last coordinate -- will be automatically calculated from the
  previous elements.

  .. imagezoom:: _images/qookTutor5.png
     :scale: 60 %

- Add methods to the beamline elements (with right click):

  a) shine() to the source,
  b) reflect() to the optical element and select a proper beam for it – the
     global beam from the source, it has a default name but you may rename it
     in the shine();
  c) expose() to the screen and select a proper beam for it – the global beam
     from the OE, it has a default name but you may rename it in the reflect();

  Red colored words indicate None as a selected beam. If you continue with the
  script generation, the script will result in a run time error.

  .. imagezoom:: _images/qookTutor6.png
     :scale: 60 %

- Add a plot in Plots tab and select the local screen beam.
- Save the beamline layout as xml.
- Generate python script (the button with a code page and the python logo),
  save the script and run it.
- In the console output you can read the actual pitch (Bragg angle) for the
  crystal and the screen position.

  .. imagezoom:: _images/qookTutor7.png
     :scale: 60 %

"""
