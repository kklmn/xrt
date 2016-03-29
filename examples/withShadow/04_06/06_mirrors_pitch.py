# -*- coding: utf-8 -*-
r"""
Variable pitch angle of collimating and focusing mirrors
--------------------------------------------------------

Uses :mod:`shadow` backend.
File: `\\examples\\withShadow\\04_06\\06_mirrors_pitch.py`

Pictures at the sample position, :ref:`type 1 of
global normalization<globalNorm>`.
The nominal pitch angle for the toroid mirror is 4.7 mrad.

+-------+-------+-------+-------+
| |MP1| | |MP2| | |MP3| |       |
+-------+-------+-------+ |MP4| |
| |MP7| | |MP6| | |MP5| |       |
+-------+-------+-------+-------+

.. |MP1| image:: _images/04Rh_pitch_41_norm1.*
   :scale: 35 %
.. |MP2| image:: _images/04Rh_pitch_43_norm1.*
   :scale: 35 %
.. |MP3| image:: _images/04Rh_pitch_45_norm1.*
   :scale: 35 %
.. |MP4| image:: _images/04Rh_pitch_47_norm1.*
   :scale: 35 %
   :align: middle
.. |MP5| image:: _images/04Rh_pitch_49_norm1.*
   :scale: 35 %
.. |MP6| image:: _images/04Rh_pitch_51_norm1.*
   :scale: 35 %
.. |MP7| image:: _images/04Rh_pitch_53_norm1.*
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
    plot2 = xrtp.XYCPlot('star.04')
    plot2.caxis.offset = 6000
    plot2.xaxis.limits = [-4, 4]
    plot2.yaxis.limits = [-4, 4]
    plot2.yaxis.factor *= -1
    textPanel2 = plot2.fig.text(0.88, 0.8, '', transform=plot2.fig.transFigure,
                                size=12, color='r', ha='center')
    #==========================================================================
    threads = 4
    #==========================================================================
    start01 = shadow.files_in_tmp_subdirs('start.01', threads)
    start04 = shadow.files_in_tmp_subdirs('start.04', threads)
    shadow.modify_input(start01, ('THICK(1)', str(60 * 1e-4)))
    stripe = 'Rh'
    shadow.modify_input(start01, ('FILE_REFL', stripe+'_refl.dat'))
    shadow.modify_input(start04, ('FILE_REFL', stripe+'_refl.dat'))

    def plot_generator():
        shadow.modify_input(start01, ('F_REFLEC', '1'))
        for angle in np.linspace(4.1e-3, 5.3e-3, 7):
            rmirr = 744680 * 4.7e-3 / angle
            shadow.modify_input(start01, ('RMIRR', str(rmirr)))
            r_maj = 476597 * 4.7e-3 / angle
            shadow.modify_input(start04, ('R_MAJ', str(r_maj)))
            tIncidence = 90 - angle * 180 / np.pi
            shadow.modify_input(
                start01, ('T_INCIDENCE', str(tIncidence)),
                ('T_REFLECTION', str(tIncidence)))
            shadow.modify_input(
                start04, ('T_INCIDENCE', str(tIncidence)),
                ('T_REFLECTION', str(tIncidence)))
            filename04 = '04' + stripe + '_pitch_%.1f' % (angle*1e3)
            plot2.title = filename04
            plot2.saveName = \
                [filename04 + '.pdf', filename04 + '.png', filename04 + '.svg']
#            plot2.persistentName = filename04 + '.pickle'
            textPanel2.set_text(
                stripe + ' coating,\nangle\n%.1f mrad' % (angle*1e3))
            yield 0

    xrtr.run_ray_tracing(
        plot2, repeats=640, updateEvery=2, threads=threads,
        energyRange=[5998, 6002], generator=plot_generator, globalNorm=True,
        backend='shadow')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
