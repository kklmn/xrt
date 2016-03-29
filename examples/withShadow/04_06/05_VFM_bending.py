# -*- coding: utf-8 -*-
r"""
Bending of focusing mirror
--------------------------

Uses :mod:`shadow` backend.
File: `\\examples\\withShadow\\04_06\\05_VFM_bending.py`

Pictures at the sample position,
:ref:`type 1 of global normalization<globalNorm>`

+---------+---------+---------+---------+
| |VFMR1| | |VFMR2| | |VFMR3| |         |
+---------+---------+---------+ |VFMR4| |
| |VFMR7| | |VFMR6| | |VFMR5| |         |
+---------+---------+---------+---------+

.. |VFMR1| image:: _images/VFM_R0317731_norm1.*
   :scale: 35 %
.. |VFMR2| image:: _images/VFM_R0363711_norm1.*
   :scale: 35 %
.. |VFMR3| image:: _images/VFM_R0416345_norm1.*
   :scale: 35 %
.. |VFMR4| image:: _images/VFM_R0476597_norm1.*
   :scale: 35 %
   :align: middle
.. |VFMR5| image:: _images/VFM_R0545567_norm1.*
   :scale: 35 %
.. |VFMR6| image:: _images/VFM_R0624518_norm1.*
   :scale: 35 %
.. |VFMR7| image:: _images/VFM_R0714895_norm1.*
   :scale: 35 %
"""
__author__ = "Konstantin Klementiev"
__date__ = "1 Mar 2012"
import sys
sys.path.append(r"c:\Alba\Ray-tracing\with Python")
import numpy as np
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.shadow as shadow


def main():
    plot1 = xrtp.XYCPlot('star.04')
    plot1.caxis.offset = 6000
    plot1.xaxis.limits = [-1, 1]
    plot1.yaxis.limits = [-1, 1]
    plot1.yaxis.factor *= -1
    textPanel1 = plot1.fig.text(0.86, 0.8, '', transform=plot1.fig.transFigure,
                                size=12, color='r', ha='center')
    #==========================================================================
    threads = 4
    #==========================================================================
    start01 = shadow.files_in_tmp_subdirs('start.01', threads)
    start04 = shadow.files_in_tmp_subdirs('start.04', threads)
    rmirr0 = 744680.
    shadow.modify_input(start01, ('RMIRR', str(rmirr0)))
    angle = 4.7e-3
    tIncidence = 90 - angle * 180 / np.pi
    shadow.modify_input(
        start01, ('T_INCIDENCE', str(tIncidence)),
        ('T_REFLECTION', str(tIncidence)))
    shadow.modify_input(
        start04, ('T_INCIDENCE', str(tIncidence)),
        ('T_REFLECTION', str(tIncidence)))
    rmaj0 = 476597.0

    def plot_generator():
        for rmaj in np.logspace(-1, 1, 7, base=1.5) * rmaj0:
            shadow.modify_input(start04, ('R_MAJ', str(rmaj)))
            filename = 'VFM_R%07i' % rmaj
            plot1.title = filename
            plot1.saveName = [filename + '.pdf', filename + '.png']
            textToSet = 'focusing mirror\nmeridional radius\n$R =$ %.1f km'\
                % (rmaj * 1e-5)
            textPanel1.set_text(textToSet)
            yield

    xrtr.run_ray_tracing(
        plot1, repeats=640, updateEvery=2, threads=threads,
        energyRange=[5998, 6002], generator=plot_generator, globalNorm=True,
        backend='shadow')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
