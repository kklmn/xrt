# -*- coding: utf-8 -*-
r"""
Detectors
---------

Module :mod:`~xrt.backends.raycing.detectors` is intended for simulating
realistic detectors of different kinds - area detectors, photon counting
detectors, photodiodes, ionization chambers.

.. autosummary::
   :signatures: none

   DetectorScreen

.. autoclass:: xrt.backends.raycing.detectors.DetectorScreen()
   :members: __init__, expose

"""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__all__ = ('DetectorScreen',)

import numpy as np

from .. import raycing
from .screens import Screen, allArguments as screenArguments

allArguments = screenArguments + ('cLimits', 'histShape')


class DetectorScreen(Screen):
    """Flat detector for beam histogramming. Similar to Screen but with
    physical sizes and mesh sizes. The histogramming is done within `expose()`
    and saved in `self.image`."""

    def __init__(self, bl=None, name='', center=[0, 0, 0], x='auto', z='auto',
                 **kwargs):
        """
        *limPhysX* and *limPhysY*: [*min*, *max*] where *min*, *max* are floats
            Physical dimensions.

        *histShape* [*rows*, *columns*]
            Mesh size.


        """
        super().__init__(bl, name, center, x, z, **kwargs)
        self.__pop_kwargs(**kwargs)
        if np.sum(self.limPhysX) > 0:
            self.image = np.zeros(np.int32(self.histShape))

    def __pop_kwargs(self, **kwargs):
        self.cLimits = kwargs.pop('cLimits', None)  # for spectral limits
        self.histShape = kwargs.pop('histShape', [256, 256])

    @property
    def cLimits(self):
        return self._cLimits

    @cLimits.setter
    def cLimits(self, cLimits):
        if cLimits is None:
            self._cLimits = raycing.Limits([0, 0])
        else:
            self._cLimits = raycing.Limits(cLimits)

    @property
    def histShape(self):
        return self._histShape

    @histShape.setter
    def histShape(self, histShape):
        if histShape is None:
            self._histShape = raycing.Image2D([256, 256])
        else:
            self._histShape = raycing.Image2D(histShape)

    @raycing.append_to_flow_decorator
    def expose(self, beam=None, onlyPositivePath=False, withHistogram=False):
        """Expose a global beam and return it in detector-local coordinates.

        *withHistogram*: bool
            If True, calculate the histogram over the physical detector
            surface and store it in ``self.image``.

        .. Returned values: beamLocal
        """
        blo = super().expose(beam, onlyPositivePath)

        if withHistogram:
            if any([np.sum(np.abs(x)) == 0 for x in
                    [self.limPhysX, self.limPhysY]]):
                print("Using auto limits for histogramming")
                self.limPhysX = raycing.Limits(
                    self.footprint[-1][:, 0].tolist())
                self.limPhysY = raycing.Limits(
                    self.footprint[-1][:, 2].tolist())

            limitsIn = [self.limPhysX if isinstance(self.limPhysX, list) else
                        self.limPhysX.tolist(),
                        self.limPhysY if isinstance(self.limPhysY, list) else
                        self.limPhysY.tolist()]
            # print(limitsIn, self.histShape)
            hist2d, hist2dRGB, limitsOut = raycing.build_hist(
                    blo, limits=limitsIn, isScreen=True, shape=self.histShape,
                    cDataFunc=None, cLimits=None)
            self.image = hist2d

        return blo
