# -*- coding: utf-8 -*-
r"""
Scanning of Double Crystal Monochromator
----------------------------------------

Uses :mod:`shadow` backend. The energy range is equal for Si111 and Si311
cases.

Si111
~~~~~

:ref:`type 1 of global normalization<globalNorm>`

.. image:: _images/Si111_192340_norm1.*
   :scale: 21 %
.. image:: _images/Si111_192360_norm1.*
   :scale: 21 %
.. image:: _images/Si111_192380_norm1.*
   :scale: 21 %
.. image:: _images/Si111_192400_norm1.*
   :scale: 21 %
.. image:: _images/Si111_192420_norm1.*
   :scale: 21 %
.. image:: _images/Si111_192440_norm1.*
   :scale: 21 %
.. image:: _images/Si111_192460_norm1.*
   :scale: 21 %

Si311
~~~~~

:ref:`type 1 of global normalization<globalNorm>`

.. image:: _images/Si311_391095_norm1.*
   :scale: 21 %
.. image:: _images/Si311_391141_norm1.*
   :scale: 21 %
.. image:: _images/Si311_391188_norm1.*
   :scale: 21 %
.. image:: _images/Si311_391234_norm1.*
   :scale: 21 %
.. image:: _images/Si311_391281_norm1.*
   :scale: 21 %
.. image:: _images/Si311_391328_norm1.*
   :scale: 21 %
.. image:: _images/Si311_391374_norm1.*
   :scale: 21 %
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
    plot1 = xrtp.XYCPlot('star.03')
    plot1.xaxis.limits = [-15, 15]
    plot1.yaxis.limits = [-15, 15]
    plot1.yaxis.factor *= -1
    textPanel = plot1.fig.text(0.88, 0.8, '', transform=plot1.fig.transFigure,
                               size=14, color='r', ha='center')
    #==========================================================================
    threads = 4
    #==========================================================================
    start01 = shadow.files_in_tmp_subdirs('start.01', threads)
    start02 = shadow.files_in_tmp_subdirs('start.02', threads)
    start03 = shadow.files_in_tmp_subdirs('start.03', threads)
    shadow.modify_input(start01, ('THICK(1)', str(60 * 1e-4)))
    shadow.modify_input(start01, ('FILE_REFL', 'Rh_refl.dat'))
    shadow.modify_input(start02, ('F_CENTRAL', '0'))
    shadow.modify_input(start03, ('F_CENTRAL', '0'))

    def plot_generator():
        for crystal in ['Si111', 'Si311']:
            theta0 = 19.234
            theta1 = 19.246
            if crystal == 'Si311':
                theta0 = np.arcsin(np.sin(theta0*np.pi/180) *
                                   np.sqrt(11.0/3)) * 180 / np.pi
                theta1 = np.arcsin(np.sin(theta1*np.pi/180) *
                                   np.sqrt(11.0/3)) * 180 / np.pi
            shadow.modify_input(start02, ('FILE_REFL', crystal + '.rc'))
            shadow.modify_input(start03, ('FILE_REFL', crystal + '.rc'))
            for theta in np.linspace(theta0, theta1, 7):
                shadow.modify_input(start02, ('T_INCIDENCE', str(90 - theta)),
                                    ('T_REFLECTION', str(90 - theta)))
                shadow.modify_input(start03, ('T_INCIDENCE', str(90 - theta)),
                                    ('T_REFLECTION', str(90 - theta)))
                filename = crystal + '_%5i' % (theta * 1e4)
                plot1.title = filename
                plot1.saveName = [filename + '.pdf', filename + '.png']
#                plot1.persistentName = filename + '.pickle'
                textPanel.set_text(
                    crystal + '\nBragg angle\n$\\theta =$ %.3f$^o$' % theta)
                yield 0

    xrtr.run_ray_tracing(
        plot1, repeats=640, updateEvery=2, energyRange=[5998, 6003],
        generator=plot_generator, threads=threads, globalNorm=True,
        backend='shadow')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
