# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:56:58 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

from ..commons import qt

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
# scr_m = qt.QMatrix4x4(1, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0,  0, 0, 0, 1)
# scr_m = qt.QMatrix4x4(1, 0, 0, 0,  0, 1, 0, 0,  0, 0, -1, 0,  0, 0, 0, 1)

_DEBUG_ = True  # If False, exceptions inside the module are ignored
MAXRAYS = 500000