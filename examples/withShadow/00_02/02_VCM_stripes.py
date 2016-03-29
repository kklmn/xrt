# -*- coding: utf-8 -*-
r"""
Collimating mirror
------------------

Uses :mod:`shadow` backend.
File: `\\examples\\withShadow\\00_02\\02_VCM_stripes.py`

The source is MPW-80 wiggler of ALBA-CLÆSS beamline.

Si stripe, pitch angles 4.5--1.5 mrad
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`type 2 of global normalization<globalNorm>`

.. image:: _images/VCM_45_Si_norm2.*
   :scale: 45 %
.. image:: _images/VCM_40_Si_norm2.*
   :scale: 45 %
.. image:: _images/VCM_35_Si_norm2.*
   :scale: 45 %
.. image:: _images/VCM_30_Si_norm2.*
   :scale: 45 %
.. image:: _images/VCM_25_Si_norm2.*
   :scale: 45 %
.. image:: _images/VCM_20_Si_norm2.*
   :scale: 45 %
.. image:: _images/VCM_15_Si_norm2.*
   :scale: 45 %

Pitch angle 4.5 mrad, stripes: Si, Rh, Pt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`type 2 of global normalization<globalNorm>`

.. image:: _images/VCM_45_Si_norm2.*
   :scale: 45 %
.. image:: _images/VCM_45_Rh_norm2.*
   :scale: 45 %
.. image:: _images/VCM_45_Pt_norm2.*
   :scale: 45 %
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
    shadow.modify_input(start01, ('THICK(1)', str(60 * 1e-4)))
    stripes = ['Si', 'Rh', 'Pt']
    angles = np.linspace(4.5e-3, 1.5e-3, 7)

    def plot_generator():
        shadow.modify_input(start01, ('F_REFLEC', '0'))
        tIncidence = 90 - angles[0] * 180 / np.pi
        shadow.modify_input(
            start01, ('T_INCIDENCE', str(tIncidence)),
            ('T_REFLECTION', str(tIncidence)))
        filename = 'VCM0_nomirror'
        plot1.title = filename
        plot1.saveName = [filename + '.pdf', filename + '.png']
        textPanel.set_text('no mirror')
        yield 0

        shadow.modify_input(start01, ('F_REFLEC', '1'))
        for angle in angles:
            tIncidence = 90 - angle * 180 / np.pi
            shadow.modify_input(
                start01, ('T_INCIDENCE', str(tIncidence)),
                ('T_REFLECTION', str(tIncidence)))
            for stripe in [stripes[0], ]:
                shadow.modify_input(start01, ('FILE_REFL', stripe+'_refl.dat'))
                filename = 'VCM_' + str(angle * 1e4) + '_' + stripe
                plot1.title = filename
                plot1.saveName = [filename + '.pdf', filename + '.png']
                textPanel.set_text(
                    stripe + ' coating,\npitch angle\n%.1f mrad'
                    % (angle * 1e3))
                yield 0
        for angle in [angles[0], ]:
            tIncidence = 90 - angle * 180 / np.pi
            shadow.modify_input(
                start01, ('T_INCIDENCE', str(tIncidence)),
                ('T_REFLECTION', str(tIncidence)))
            for stripe in [stripes[1], stripes[2]]:
                shadow.modify_input(start01, ('FILE_REFL', stripe+'_refl.dat'))
                filename = 'VCM_' + str(angle * 1e4) + '_' + stripe
                plot1.title = filename
                plot1.saveName = [filename + '.pdf', filename + '.png']
                textPanel.set_text(
                    stripe + ' coating,\npitch angle\n%.1f mrad'
                    % (angle * 1e3))
                yield 0

    xrtr.run_ray_tracing(
        plot1, repeats=160, updateEvery=4, processes=processes,
        energyRange=[2400, 22400], generator=plot_generator, globalNorm=True,
        backend='shadow')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
