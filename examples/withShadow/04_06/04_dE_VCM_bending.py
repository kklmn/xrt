# -*- coding: utf-8 -*-
r"""
Bending of collimating mirror
-----------------------------

Uses :mod:`shadow` backend.
File: `\\examples\\withShadow\\03\\03_DCM_energy.py`

Influence onto energy resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pictures after monochromator,
:ref:`type 2 of global normalization<globalNorm>`. The nominal radius is 7.4
km. Watch the energy distribution when the bending radius is smaller or greater
than the nominal one.

+---------+---------+---------+---------+
| |VCMR1| | |VCMR2| | |VCMR3| |         |
+---------+---------+---------+ |VCMR4| |
| |VCMR7| | |VCMR6| | |VCMR5| |         |
+---------+---------+---------+---------+

.. |VCMR1| image:: _images/03VCM_R0496453_norm2.*
   :scale: 35 %
.. |VCMR2| image:: _images/03VCM_R0568297_norm2.*
   :scale: 35 %
.. |VCMR3| image:: _images/03VCM_R0650537_norm2.*
   :scale: 35 %
.. |VCMR4| image:: _images/03VCM_R0744680_norm2.*
   :scale: 35 %
   :align: middle
.. |VCMR5| image:: _images/03VCM_R0852445_norm2.*
   :scale: 35 %
.. |VCMR6| image:: _images/03VCM_R0975806_norm2.*
   :scale: 35 %
.. |VCMR7| image:: _images/03VCM_R1117020_norm2.*
   :scale: 35 %

Influence onto focusing
~~~~~~~~~~~~~~~~~~~~~~~

Pictures at the sample position,
:ref:`type 1 of global normalization<globalNorm>`

+----------+----------+----------+----------+
| |VCMRF1| | |VCMRF2| | |VCMRF3| |          |
+----------+----------+----------+ |VCMRF4| |
| |VCMRF7| | |VCMRF6| | |VCMRF5| |          |
+----------+----------+----------+----------+

.. |VCMRF1| image:: _images/04VCM_R0496453_norm1.*
   :scale: 35 %
.. |VCMRF2| image:: _images/04VCM_R0568297_norm1.*
   :scale: 35 %
.. |VCMRF3| image:: _images/04VCM_R0650537_norm1.*
   :scale: 35 %
.. |VCMRF4| image:: _images/04VCM_R0744680_norm1.*
   :scale: 35 %
   :align: middle
.. |VCMRF5| image:: _images/04VCM_R0852445_norm1.*
   :scale: 35 %
.. |VCMRF6| image:: _images/04VCM_R0975806_norm1.*
   :scale: 35 %
.. |VCMRF7| image:: _images/04VCM_R1117020_norm1.*
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
    plot1 = xrtp.XYCPlot('star.03')
    plot1.caxis.offset = 6000
    plot2 = xrtp.XYCPlot('star.04')
    plot2.caxis.offset = 6000
    plot1.xaxis.limits = [-15, 15]
    plot1.yaxis.limits = [-15, 15]
    plot1.yaxis.factor *= -1
    plot2.xaxis.limits = [-1, 1]
    plot2.yaxis.limits = [-1, 1]
    plot2.yaxis.factor *= -1
    textPanel1 = plot1.fig.text(
        0.89, 0.82, '', transform=plot1.fig.transFigure,
        size=14, color='r', ha='center')
    textPanel2 = plot2.fig.text(
        0.89, 0.82, '', transform=plot2.fig.transFigure,
        size=14, color='r', ha='center')
    #==========================================================================
    threads = 4
    #==========================================================================
    start01 = shadow.files_in_tmp_subdirs('start.01', threads)
    start04 = shadow.files_in_tmp_subdirs('start.04', threads)
    rmaj0 = 476597.0
    shadow.modify_input(start04, ('R_MAJ', str(rmaj0)))
    angle = 4.7e-3
    tIncidence = 90 - angle * 180 / np.pi
    shadow.modify_input(
        start01, ('T_INCIDENCE', str(tIncidence)),
        ('T_REFLECTION', str(tIncidence)))
    shadow.modify_input(
        start04, ('T_INCIDENCE', str(tIncidence)),
        ('T_REFLECTION', str(tIncidence)))
    rmirr0 = 744680.

    def plot_generator():
        for rmirr in np.logspace(-1., 1., 7, base=1.5) * rmirr0:
            shadow.modify_input(start01, ('RMIRR', str(rmirr)))
            filename = 'VCM_R%07i' % rmirr
            filename03 = '03' + filename
            filename04 = '04' + filename
            plot1.title = filename03
            plot2.title = filename04
            plot1.saveName = [filename03 + '.pdf', filename03 + '.png']
            plot2.saveName = [filename04 + '.pdf', filename04 + '.png']
#            plot1.persistentName = filename03 + '.pickle'
#            plot2.persistentName = filename04 + '.pickle'
            textToSet = 'collimating\nmirror\n$R =$ %.1f km' % (rmirr * 1e-5)
            textPanel1.set_text(textToSet)
            textPanel2.set_text(textToSet)
            yield

    def after():
    #    import subprocess
    #    subprocess.call(["python", "05-VFM-bending.py"],
    #      cwd='/home/kklementiev/Alba/Ray-tracing/with Python/05-VFM-bending')
        pass

    xrtr.run_ray_tracing(
        [plot1, plot2], repeats=640, updateEvery=2,
        energyRange=[5998, 6002], generator=plot_generator, threads=threads,
        globalNorm=True, afterScript=after, backend='shadow')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
