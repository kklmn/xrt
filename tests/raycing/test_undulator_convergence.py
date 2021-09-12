# -*- coding: utf-8 -*-
"""
Tests of undulator convergence
------------------------------

Find the test module `test_undulator_convergence.py`, as well as several other
tests for raycing backend, in `/tests/raycing`.

This is a study of a tapered undulator created from
:class:`xrt.backends.raycing.sources.SourceFromField`. Its magnetic field is
shown below.

.. imagezoom:: _images/B_vs_Z.png

The output of `IntegratedSource.test_convergence()` shows convergence of the
calculated intensity signaled by various indicators:

.. imagezoom:: _images/test_convergence.png
    
The transverse intensity distribution vs number of nodes:

.. video:: _videos/Imap_frames.mp4
   :controls:
   :loop:

The flux distribution vs number of nodes:

.. imagezoom:: _images/Total_flux.png
       
"""

__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "12 Sep 2021"
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rs
from xrt.backends.raycing.physconsts import K2B

saveFrameImages = False
saveVideo = True


def flux_through_aperture(energy, theta, psi, I0):
    dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
    flux = I0.sum(axis=(1, 2)) * dtheta * dpsi
    plt.plot(energy, flux)


def intensity_in_transverse_plane(energy, theta, psi, I0):
    plt.imshow(I0[len(energy)//2, :, :],
               extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])


def colored_intensity_in_transverse_plane(energy, theta, psi, I0):
    import matplotlib as mpl
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_axes([0.15*6/7, 0.15, 0.7*6/7, 0.7])
    ax.set_xlabel(u'$\\theta$ (µrad)')
    ax.set_ylabel(u'$\\psi$ (µrad)')
    axC = fig.add_axes([0.95*6/7, 0.15, 0.05, 0.7])
    plt.setp(axC.get_xticklabels(), visible=False)
    axC.yaxis.set_label_position("right")
    axC.yaxis.tick_right()
    axC.set_xlim([0, 1])
    axC.set_ylim([energy[0], energy[-1]])
    axC.set_ylabel('energy (eV)')

    hue, tmp, tmp = np.meshgrid(energy, theta, psi, indexing='ij')
    hue -= hue.min()
    hue /= hue.max()
    sat = np.ones_like(hue)
    val = I0 / I0.max()
    # hsvI0 = np.stack((hue, sat, val), axis=3) # numpy1.10
    hsvI0 = np.zeros([d for d in hue.shape] + [3])
    hsvI0[:, :, :, 0] = hue
    hsvI0[:, :, :, 1] = sat
    hsvI0[:, :, :, 2] = val

    rgbI0 = mpl.colors.hsv_to_rgb(hsvI0)
    rgbI0sum = rgbI0.sum(axis=0)
    ax.imshow(rgbI0sum / rgbI0sum.max(),
              extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])

    hue = np.array(energy)[:, np.newaxis]
    hue -= hue.min()
    hue /= hue.max()
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    # hsvC = np.stack((hue, sat, val), axis=2) # numpy1.10
    hsvC = np.zeros([d for d in hue.shape] + [3])
    hsvC[:, :, 0] = hue
    hsvC[:, :, 1] = sat
    hsvC[:, :, 2] = val
    rgbC = mpl.colors.hsv_to_rgb(hsvC)
    axC.imshow(rgbC / rgbC.max(), aspect='auto', origin='lower',
               extent=[0, 1, energy[0], energy[-1]])


def main():
    energy = np.linspace(9150, 9150, 1)
    # energy = np.linspace(3850, 24150, 3)
    theta = np.linspace(-1, 1, 256) * 120e-6
    psi = np.linspace(-1, 1, 256) * 120e-6

    Np = 40
    L0 = 30
    Kmax = 1.45
    alphaZ = 0.2
    zMin, zMax = -0.5*Np*L0, 0.5*Np*L0
    fieldZ = np.linspace(zMin, zMax, int(Np*L0)+1)
    
    #tapered
    fieldK = Kmax*(1.-alphaZ*(0.5-(fieldZ-zMin)/(Np*L0)))*np.sin(2*np.pi*fieldZ/L0)
    fieldB = K2B * fieldK / L0
    plt.figure("B")
    plt.plot(fieldZ, fieldB)
    plt.xlabel('z (mm)', fontsize=12)
    plt.ylabel('B (T)', fontsize=12)
    plt.savefig("B_vs_Z.png")
    #plt.show()
    fieldArray = np.vstack((fieldZ, fieldB)).T

    kwargsCF = dict(
        eE=3., eI=0.5,  # Parameters of the synchrotron ring [GeV], [Ampere]
        betaX=9.539, betaZ=1.982,
        # eEspread=eEspread,  # Energy spread of the electrons in the ring
        #     period=30., n=40,  # Parameters of the undulator, period in [mm]
        #     K=1.45,  # Deflection parameter (ignored if targetE is not None)
        # eSigmaX=eSigmaX, eSigmaZ=eSigmaZ,  # Size of the electron beam [mkm]
        xPrimeMax=theta[-1]*1e3, zPrimeMax=psi[-1]*1e3,
        eMax=np.max(energy),
        gp=1e-4,
        # gNodes=2000,
        # uniformRayDensity=uniformRayDensity,
        # filamentBeam=filamentBeam,  # Single wavefront
        customField = fieldArray,
        # targetOpenCL="GPU",
        # R0=R0,   # Near Field.
        eEpsilonX=0.263, eEpsilonZ=0.008,)  # Emittance [nmrad]

    source = rs.SourceFromField(**kwargsCF)
    fig = source.test_convergence()[-1]
    fig.savefig("test_convergence.png")
    gnMax = source.gNodes
    print('gnMax = {}'.format(gnMax))

    I0, l1, l2, l3 = source.intensities_on_mesh(energy, theta, psi)
    nodeArr = []
    fluxArr = []

    dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
    
    fig = plt.figure(figsize=(4.08, 3.84))
    ax = fig.add_axes([0.17, 0.12, 0.82, 0.87])
    ax.set_xlabel("x' (µrad)")
    ax.set_ylabel("y' (µrad)")
    im = ax.imshow(np.random.uniform(size=(256, 256)),
                   extent=[theta[0]*1e6, theta[-1]*1e6,
                           psi[0]*1e6, psi[-1]*1e6],
                   cmap='viridis')
    nPanel = fig.text(
        0.01, 0.99, '', transform=ax.transAxes, size=10, color='w',
        ha='left', va='top')
    fPanel = fig.text(
        0.99, 0.99, '', transform=ax.transAxes, size=10, color='w',
        ha='right', va='top')

    ns = np.linspace(8, gnMax, gnMax-7, dtype=int)
    if saveVideo:
        frames = []
        import cv2  # pip install opencv-python
    for nord, n in enumerate(ns):
        if n % 2 != 0:
            continue
        print('{0} of {1}'.format(n, gnMax))
        source.gNodes = n
        I0, l1, l2, l3 = source.intensities_on_mesh(energy, theta, psi)
        flux = I0.sum(axis=(1, 2)) * dtheta * dpsi
        imap = I0[-1, :, :]
        im.set_data(imap.T/np.max(imap))
        nPanel.set_text(r'{0} nodes'.format(n*source.gIntervals))
        fPanel.set_text(r'{0:.5e}  ph/s/0.1%bw'.format(flux[0]))
        fig.canvas.draw()
        if saveFrameImages:
            fig.savefig("Imap_frame_{}.png".format(nord))
        if saveVideo:
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
        nodeArr.append(source.gNodes*source.gIntervals)
        fluxArr.append(flux)

    if saveVideo:
        out = cv2.VideoWriter(
            'Imap_frames.mp4',
            -1, # cv2.VideoWriter_fourcc(*'MP4V'),
            15, fig.canvas.get_width_height())
        for frame in frames:
            out.write(frame[:, :, ::-1])  # bgr -> rgb
        out.release()
        
    plt.figure("Total flux")
    plt.gca().set_xlabel("Number of nodes")
    plt.gca().set_ylabel(r"Flux (ph/s/0.1%bw)")
    plt.semilogy(np.array(nodeArr), np.array(fluxArr))
    plt.savefig("Total_flux.png")
    # plt.show

    # flux_through_aperture(energy, theta, psi, I0)
    # intensity_in_transverse_plane(energy, theta, psi, I0)
    # colored_intensity_in_transverse_plane(energy, theta, psi, I0)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("Done in {0} s".format(time.time()-t0))
    plt.show()
