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

ambient['Screen'] = qt.QVector4D(2*0.29225, 2.2*0.29225, 1.8*0.29225, 1.)
# ambient['Screen'] = qt.QVector4D(2*0.29225, 2*0.29225, 2*0.29225, 1.)
diffuse['Screen'] = qt.QVector4D(0.50754, 1.1*0.50754, 0.9*0.50754, 1.)
specular['Screen'] = qt.QVector4D(0.9, 1., 0.8, 1.)
shininess['Screen'] = 80.

ambient['selected'] = qt.QVector4D(0.89225, 0.89225, 0.49225, 1.)

scr_m = qt.QMatrix4x4(1, 0, 0, 0,  0, 0, -1, 0,  0, 1, 0, 0,  0, 0, 0, 1)
#scr_m = qt.QMatrix4x4(0, 0, 1, 0,  1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 0, 1)
# scr_m = qt.QMatrix4x4(1, 0, 0, 0,  0, 1, 0, 0,  0, 0, -1, 0,  0, 0, 0, 1)

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
    # 'fineGridEnabled': False,
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
    'showLostRays': False,
    'showLocalAxes': False,
    'showInternalBeam': True,

    'oeThickness': 5.,
    'oeThicknessForce': None,
    'slitThicknessFraction': 50.,
    'apertureBladeThickness': 5.,
    'apertureDefaultSpan': 10.,
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
    'autoSizeOe': 'OE size match beam',
    'showLostRays': 'Show lost rays',
    'showLocalAxes': 'Show local axes',
    'showInternalBeam': 'Show internal beams in multi-surface OEs',
    }

SCENE_TEXTEDITS = {
    'oeThickness': {
            'label': 'Default OE thickness, mm',
            'tooltip': 'For OEs that do not have thickness'},
    'oeThicknessForce': {
            'label': 'Force OE thickness, mm',
            'tooltip': 'For OEs that have thickness, e.g. plates or lenses'},
    'slitThicknessFraction': {
            'label': 'Aperture frame size, %',
            'tooltip': ''},
    'maxLen': {
            'label': 'Scene limit, mm',
            'tooltip': ''},
   }
