# -*- coding: utf-8 -*-
r"""
Created with xrtQook (xrt 2.0.0)

Scan a fixed-exit Si(111) DCM assembled from individual raycing stages.
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2026-11-17"

import numpy as np
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials.elemental as rmatsel
import xrt.backends.raycing.materials.compounds as rmatsco
import xrt.backends.raycing.materials.crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.figure_error as rfe
import xrt.backends.raycing.stages as rstages
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

Si_111_ = rmats.crystals_basic.CrystalSi(
    d=3.1354575567115175,
    elements=['Si'],
    quantities=[1.0],
    name='Si_111_')


def build_beamline():
    bl = raycing.BeamLine(
        name='Fixed_exit_Si_111__DCM',
        description=None)

    Base_height = rstages.LinearStage(
        bl=bl,
        name='Base_height',
        axis='z',
        center=[0.0, 10000.0, -250.0],
        mountOffset={'center': [0.0, 0.0, 0.0],
                     'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                     'rotationSequence': 'RzRyRx'},
        positionMin=-20.0,
        positionMax=20.0,
        positionUnit='mm',
        renderSize={'width': 200.0, 'topLength': 500.0, 'bottomLength': 500.0,
                    'topHeight': 20.0, 'bottomHeight': 100.0, 'grooves': None,
                    'grooveWidth': 6.0, 'grooveDepth': 5.0, 'gap': 0.5})

    Bragg_disk = rstages.RotationStage(
        bl=bl,
        name='Bragg_disk',
        axis='yaw',
        position=9.101863673348971,
        pivotPoint=[0.0, 0.0, 0.0],
        center=[0.0, 0.0, 0.0],
        mount=Base_height,
        mountOffset={'center': [-10.0, 0.0, 250.0],
                     'pitch': -0.0, 'roll': '90 deg', 'yaw': 0.0,
                     'rotationSequence': 'RzRyRx'},
        positionMin=5.0,
        positionMax=35.0,
        positionUnit='deg',
        renderSize={'width': None, 'length': None, 'shape': None,
                    'topHeight': 2.0, 'bottomHeight': 2.0, 'radius': 150.0})

    X2_longitudinal = rstages.LinearStage(
        bl=bl,
        name='X2_longitudinal',
        axis='y',
        center=[0.0, 0.0, 0.0],
        mount=Bragg_disk,
        mountOffset={'center': [-30.0, 31.6075, 10.0],
                     'pitch': '180 deg', 'roll': '90 deg', 'yaw': '180 deg',
                     'rotationSequence': 'RzRyRx'},
        positionMin=-20.0,
        positionMax=20.0,
        positionUnit='mm')

    X2_perpendicular = rstages.LinearStage(
        bl=bl,
        name='X2_perpendicular',
        axis='z',
        position=5.0,
        center=[0.0, 0.0, 0.0],
        mount=X2_longitudinal,
        mountOffset={'center': [0.0, 0.0, 10.0],
                     'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                     'rotationSequence': 'RzRyRx'},
        positionMin=-5.0,
        positionMax=8.0,
        positionUnit='mm')

    X2_cradle = rstages.RotationStage(
        bl=bl,
        name='X2_cradle',
        axis='pitch',
        pivotPoint=[0.0, 0.0, 10.0],
        center=[0.0, 0.0, 0.0],
        mount=X2_perpendicular,
        mountOffset={'center': [0.0, 0.0, 0.0],
                     'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                     'rotationSequence': 'RzRyRx'},
        positionMin=-45.0,
        positionMax=45.0,
        positionUnit='deg',
        renderSize={'width': None, 'length': None, 'shape': None,
                    'topHeight': 2.0, 'bottomHeight': 2.0, 'radius': 10.0})

    bl.Geometric_source = rsources.geoms.GeometricSource(
        bl=bl,
        name='Geometric_source',
        center=[0.0, 0.0, 0.0],
        nrays=50000,
        dx=2,
        dz=2,
        distxprime=None,
        distzprime=None,
        energies=[12500.0])

    bl.Slit_1 = rapts.RectangularAperture(
        bl=bl,
        name='Slit_1',
        center=[0.0, 0.0, 0.0],
        blades={'left': -1.0, 'right': 1.0, 'bottom': -1.0, 'top': 1.0},
        mount='Base_height',
        mountOffset={'center': [0.0, -200.0, 250.0],
                     'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                     'rotationSequence': 'RzRyRx'})

    bl.X1 = roes.base.OE(
        bl=bl,
        name='X1',
        center=[0.0, 0.0, 0.0],
        mount='Bragg_disk',
        mountOffset={'center': [0.0, 0.0, 10.0],
                     'pitch': 0.0, 'roll': '-90 deg', 'yaw': 0.0,
                     'rotationSequence': 'RzRyRx'},
        material=Si_111_,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-50.0, 50.0],
        order=1)

    bl.X2 = roes.base.OE(
        bl=bl,
        name='X2',
        center=[0.0, 0.0, 0.0],
        mount='X2_cradle',
        mountOffset={'center': [0.0, 0.0, 10.0],
                     'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                     'rotationSequence': 'RzRyRx'},
        rotationSequence='RyRzRx',
        material=Si_111_,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-50.0, 50.0],
        order=1)

    bl.Slit_2 = rapts.RectangularAperture(
        bl=bl,
        name='Slit_2',
        center=[0.0, 0.0, 0.0],
        blades={'left': -1.0, 'right': 1.0, 'bottom': -1.0, 'top': 1.0},
        mount='Base_height',
        mountOffset={'center': [0.0, 200.0, 260.0],
                     'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                     'rotationSequence': 'RzRyRx'})

    bl.Exit_screen = rscreens.Screen(
        bl=bl,
        name='Exit_screen',
        center=[0.0, 12000.0, 10.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0])

    return bl


def run_process(bl):
    Geometric_source_global = bl.Geometric_source.shine()

    Slit_1_local = bl.Slit_1.propagate(
        beam=Geometric_source_global)

    X1_global, X1_local = bl.X1.reflect(
        beam=Geometric_source_global)

    X2_global, X2_local = bl.X2.reflect(
        beam=X1_global)

    Slit_2_local = bl.Slit_2.propagate(
        beam=X2_global)

    Exit_screen_local = bl.Exit_screen.expose(
        beam=X2_global)

    outDict = {
        'Geometric_source_global': Geometric_source_global,
        'Slit_1_local': Slit_1_local,
        'X1_global': X1_global,
        'X1_local': X1_local,
        'X2_global': X2_global,
        'X2_local': X2_local,
        'Slit_2_local': Slit_2_local,
        'Exit_screen_local': Exit_screen_local}
    return outDict


rrun.run_process = run_process


def define_plots(bl):
    plots = []
    return plots


def main():
    Fixed_exit_Si_111__DCM = build_beamline()
    Fixed_exit_Si_111__DCM.glow(scale=[1e2, 1e2, 1e2], centerAt='X1')
#    E0 = list(Fixed_exit_Si_111__DCM.Geometric_source.energies)[0]
#    Fixed_exit_Si_111__DCM.alignE = E0
#    plots = define_plots(Fixed_exit_Si_111__DCM)
#    xrtrun.run_ray_tracing(
#        plots=plots,
#        backend='raycing',
#        beamLine=Fixed_exit_Si_111__DCM)


if __name__ == '__main__':
    main()
