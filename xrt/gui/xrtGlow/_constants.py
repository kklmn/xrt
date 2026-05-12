# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:56:58 2026

"""

from ..commons import qt

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

msg_start = {"command": "start"}
msg_stop = {"command": "stop"}
msg_exit = {"command": "exit"}

itemTypes = {'beam': 0, 'footprint': 1, 'surface': 2, 'label': 3}

ambient = {}
diffuse = {}
specular = {}
shininess = {}

ambient['Cu'] = qt.QVector4D(0.8, 0.4, 0., 1.)
diffuse['Cu'] = qt.QVector4D(0.50, 0.25, 0., 1.)
specular['Cu'] = qt.QVector4D(1., 0.9, 0.8, 1.)
shininess['Cu'] = 100.

ambient['Si'] = qt.QVector4D(2*0.29225, 2*0.29225, 2*0.29225, 1.)
diffuse['Si'] = qt.QVector4D(0.50754, 0.50754, 0.50754, 1.)
specular['Si'] = qt.QVector4D(1., 0.9, 0.8, 1.)
shininess['Si'] = 100.

ambient['Quartz'] = qt.QVector4D(0.84, 0.85, 0.87, 1.)
diffuse['Quartz'] = qt.QVector4D(0.28*2, 0.31*2, 0.35*2, 1.)
specular['Quartz'] = qt.QVector4D(0.92, 0.96, 1.00, 1.)
shininess['Quartz'] = 180.

ambient['Screen'] = qt.QVector4D(2*0.29225, 2.2*0.29225, 1.8*0.29225, 1.)
# ambient['Screen'] = qt.QVector4D(2*0.29225, 2*0.29225, 2*0.29225, 1.)
diffuse['Screen'] = qt.QVector4D(0.50754, 1.1*0.50754, 0.9*0.50754, 1.)
specular['Screen'] = qt.QVector4D(0.9, 1., 0.8, 1.)
shininess['Screen'] = 80.

ambient['selected'] = qt.QVector4D(0.89225, 0.89225, 0.49225, 1.)

scr_m = qt.QMatrix4x4(1, 0, 0, 0,  0, 0, -1, 0,  0, 1, 0, 0,  0, 0, 0, 1)

_DEBUG_ = True  # If False, exceptions inside the module are ignored
MAXRAYS = 500000

DEFAULT_SCENE_SETTINGS = {
    'aspect': 1,
    'viewPortGL': [0, 0, 500, 500],
    'cameraAngle': 60.,
    'cameraDistance': 3.5,

    'sceneColors': {
        'normalColors': {
                'bgColor': [0.0, 0.0, 0.0],
                'lineColor': [1.0, 1.0, 1.0],
                'textColor': [1.0, 1.0, 0.0]
                },
        'invertedColors': {
                'bgColor': [1.0, 1.0, 1.0],
                'lineColor': [0.0, 0.0, 0.0],
                'textColor': [0.0, 0.0, 1.0]
                }
        },

    'scaleVec': [1e3, 1e1, 1e3],
    'rotations': [0., 0.],

    'coordOffset': [0., 0., 0.],
    'tmpOffset': [0., 0., 0.],
    'tVec': [0., 0., 0.],
    'mModLocal': qt.QMatrix4x4(),

    'colorAxis': 'energy',
    'colorMin': 1e20,
    'colorMax': -1e20,
    'selColorMin': 1e20,
    'selColorMax': -1e20,
    'cutoffI': 0.01,
    'globalNorm': True,
    'iHSV': False,
    'lineOpacity': 0.1,
    'lineWidth': 1,
    'pointOpacity': 0.1,
    'pointSize': 1,

    'drawGrid': True,
    'perspectiveEnabled': True,
    'fineGridEnabled': False,
    'aPos': [0.9, 0.9, 0.9],
    'projectionsVisibility': [0, 0, 0],

    'lineProjectionOpacity': 0.1,
    'lineProjectionWidth': 1,
    'pointProjectionOpacity': 0.1,
    'pointProjectionSize': 1,

    'enableAA': False,
    'globalColors': True,
    'linesDepthTest': False,
    'pointsDepthTest': False,
    'invertColors': False,
    'autoSizeOe': True,
    'rayFlag': {1, 2, 3},
    'showLostRays': False,
    'showLocalAxes': False,
    'showInternalBeam': True,
    'showElectronTrajectory': False,
    'trajectoryWithEmittance': False,

    'oeThickness': 5.,
    'oeThicknessForce': None,
    'slitThicknessFraction': 50.,
    'apertureBladeWidth': 5.,
    'apertureDefaultSpan': 10.,
    'apertureThickness': 1.,
    'electronEnvelopeStep': 10.,
    'electronEnvelopeSize': 3.,  # in sigma's
    'maxLen': 1.,

    'fontSize': 5,
    'labelCoordPrec': 1,
    'tiles': [25, 25],

    'geomSrcParam': {
#                     'shape': 'sddh',  # or 'sphere'
                     'shape': 'sphere',
                     'radius': 2,
                     'stacks': 8,
                     'slices': 12,
                     'spikeScale': 5,
                     'faceColor': [0.1, 0.9, 0.9, 1],
                     'edgeColor': [1, 0, 1, 1]
                     },

    'magnetShape': {'gap': 10,
                    'period': 40,
                    'dx': 40,
                    'dy': 40,
                    'dz': 10
                    }
    }

COLOR_CONTROL_LABELS = {
    'globalNorm': 'Global Normalization',
    'iHSV': 'Intensity as HSV Value'
    }

SCENE_CONTROL_LABELS = {
    # 'drawGrid': 'checkDrawGrid',
    # 'perspectiveEnabled': 'checkPerspect',
    'enableAA': 'Enable antialiasing',
    'globalColors': 'Use global colors',
    'linesDepthTest': 'Depth test for Lines',
    'pointsDepthTest': 'Depth test for Points',
    'invertColors': 'Invert scene color',
    'showLocalAxes': 'Show local axes',
    'showInternalBeam': 'Show internal beams in multi-surface OEs',
    }

RENDERING_CONTROL_LABELS = {
    'autoSizeOe': 'OE size match beam',
    }

SCENE_TEXTEDITS = {
    'maxLen': {
            'label': 'Scene limit, mm',
            'tooltip': ''},
    }

RENDERING_TEXTEDITS = {
    'oeThickness': {
            'label': 'Default OE thickness, mm',
            'tooltip': 'For OEs that do not have thickness'},
    'oeThicknessForce': {
            'label': 'Force OE thickness, mm',
            'tooltip': 'For OEs that have thickness, e.g. plates or lenses'},
    }

APERTURE_RENDERING_TEXTEDITS = {
    'slitThicknessFraction': {
            'label': 'Aperture frame size, %',
            'tooltip': ''},
    'apertureBladeWidth': {
            'label': 'Aperture blade width, mm',
            'tooltip': ''},
    'apertureDefaultSpan': {
            'label': 'Aperture default span, mm',
            'tooltip': ''},
    'apertureThickness': {
            'label': 'Aperture thickness, mm',
            'tooltip': ''},
    }

SOURCE_RENDERING_CONTROL_LABELS = {
    'showElectronTrajectory': 'Show electron trajectory',
    'trajectoryWithEmittance': 'With emittance',
    }

SOURCE_RENDERING_TEXTEDITS = {
    'electronEnvelopeStep': {
            'label': 'Electron envelope step, mm',
            'tooltip': 'Spacing between transverse emittance ellipses'},
    'electronEnvelopeSize': {
            'label': 'Electron envelope size, sigma',
            'tooltip': 'Multiplier for electron beam sigma in trajectory '
                       'envelope rendering'},
    }

SOURCE_MAGNET_TEXTEDITS = {
    'gap': {
            'label': 'Magnet gap, mm',
            'tooltip': ''},
    'dx': {
            'label': 'Magnet width dx, mm',
            'tooltip': ''},
    'dz': {
            'label': 'Magnet height dz, mm',
            'tooltip': ''},
    }
