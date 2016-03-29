# -*- coding: utf-8 -*-
"""
Module :mod:`dummy` represents the simplest backend with the minimum output
required for building histograms. You can use this backend to generate your
simple examples, as exemplified in ``xrt_logo.py``::

    def local_output():
        # define your (x, y, intensity, cData, locNrays) here
        return x, y, intensity, energy, locNrays
    dummy.run_process = local_output #is invoked by xrt to get the rays
"""
import numpy as np

nrays = 25000


def run_process(nrays=nrays):
    x = np.random.normal(size=nrays)
    y = np.random.normal(size=nrays)
    intensity = np.ones_like(x)
    energy = x + y * 2.0 + 5000
#    energy = np.random.uniform(4998, 5002, size=locNrays)
    return x, y, intensity, energy, nrays
