# -*- coding: utf-8 -*-
"""The simplest example of using shadow with xrt."""
__author__ = "Konstantin Klementiev"
__date__ = "1 Mar 2012"
import sys
sys.path.append(r"c:\Alba\Ray-tracing\with Python")
import xrt.runner as xrtr
import xrt.plotter as xrtp
plot1 = xrtp.XYCPlot('star.01')
plot1.caxis.fwhmFormatStr = None
xrtr.run_ray_tracing(plot1, repeats=40, updateEvery=2, backend='shadow')
