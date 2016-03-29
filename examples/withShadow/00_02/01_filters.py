# -*- coding: utf-8 -*-
r"""
Diamond filters of different thickness
--------------------------------------

Uses :mod:`shadow` backend.
File: `\\examples\\withShadow\\00_02\\01_filters.py`

The source is MPW-80 wiggler of ALBA-CLÆSS beamline.

:ref:`type 1 of global normalization<globalNorm>`

.. image:: _images/filt0000mum_norm1.*
   :scale: 45 %
.. image:: _images/filt0060mum_norm1.*
   :scale: 45 %
.. image:: _images/filt0400mum_norm1.*
   :scale: 45 %
.. image:: _images/filt1000mum_norm1.*
   :scale: 45 %
.. image:: _images/filt1500mum_norm1.*
   :scale: 45 %
"""
__author__ = "Konstantin Klementiev"
__date__ = "1 Mar 2012"
import sys
sys.path.append(r"c:\Alba\Ray-tracing\with Python")
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.shadow as shadow


def main():
    plot1 = xrtp.XYCPlot('star.01')
    plot1.caxis.fwhmFormatStr = None
    plot1.xaxis.limits = [-15, 15]
    plot1.yaxis.limits = [-15, 15]
    textPanel = plot1.fig.text(0.88, 0.8, '', transform=plot1.fig.transFigure,
                               size=14, color='r', ha='center')
    #==========================================================================
    processes = 4
    #==========================================================================
    start01 = shadow.files_in_tmp_subdirs('start.01', processes)

    def plot_generator():
        for i, thick in enumerate([0, 60, 400, 1000, 1500]):
#        for i, thick in enumerate([0, ]):
            shadow.modify_input(start01, ('THICK(1)', str(thick * 1e-4)))
            filename = 'filt%04imum' % thick
            plot1.title = filename
            plot1.saveName = [filename + '.pdf', filename + '.png']
            textPanel.set_text('filter\nthickness\n%s $\mu$m' % thick)
            yield 0

    xrtr.run_ray_tracing(
        [plot1, ], repeats=160, updateEvery=2, backend='shadow',
        energyRange=[2400, 22400], generator=plot_generator,
        processes=processes, globalNorm=True)

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
