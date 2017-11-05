Raycing backend
---------------

.. module:: xrt.backends.raycing

.. automodule:: xrt.backends.raycing.__init__

.. autoclass:: xrt.backends.raycing.BeamLine
   :members: __init__

.. module:: xrt.backends.raycing.run

.. autofunction:: xrt.backends.raycing.run.run_process

.. automodule:: xrt.backends.raycing.sources

.. automodule:: xrt.backends.raycing.oes

.. automodule:: xrt.backends.raycing.materials

.. automodule:: test_materials

.. .. automodule:: xrt.backends.raycing.stages
.. .. autoclass:: xrt.backends.raycing.stages.Tripod()
..    :members: __init__, set_jacks, get_orientation
.. .. autoclass:: xrt.backends.raycing.stages.OneXStage()
..    :members: __init__, select_surface
.. .. autoclass:: xrt.backends.raycing.stages.TwoXStages(OneXStage)
..    :members: __init__, set_x_stages, get_orientation

.. automodule:: xrt.backends.raycing.apertures
.. autoclass:: xrt.backends.raycing.apertures.RectangularAperture()
   :members: __init__, get_divergence, set_divergence, propagate, touch_beam
.. autoclass:: xrt.backends.raycing.apertures.SetOfRectangularAperturesOnZActuator(RectangularAperture)
   :members: __init__, select_aperture
.. autoclass:: xrt.backends.raycing.apertures.RoundAperture()
   :members: __init__, get_divergence, propagate

.. automodule:: xrt.backends.raycing.screens
.. automodule:: xrt.backends.raycing.waves

.. automodule:: test_waves
