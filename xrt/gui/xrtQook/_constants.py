# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 18:04:26 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import os  # analysis:ignore

path_to_xrt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
myTab = 4*" "

useSlidersInTree = False
withSlidersInTree = ['pitch', 'roll', 'yaw', 'bragg']
slidersInTreeScale = {'pitch': 0.1, 'roll': 0.1, 'yaw': 0.1, 'bragg': 1e-3}
redStr = ':red:`{0}`'

# os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-logging --log-level=3"
isUnitsEnabled = False  # TODO:

_DEBUG_ = False  # If False, exceptions inside the module are ignored
