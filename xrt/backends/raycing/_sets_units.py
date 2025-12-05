# -*- coding: utf-8 -*-
import numpy as np

allBeamFields = ('energy', 'x', 'xprime', 'y', 'z', 'zprime', 'xzprime',
                 'a', 'b', 'path', 'phase_shift', 'reflection_number', 'order',
                 'circular_polarization_rate', 'polarization_degree',
                 'polarization_psi',  'ratio_ellipse_axes', 's', 'r',
                 'theta', 'phi', 'incidence_angle',
                 'elevation_d', 'elevation_x', 'elevation_y', 'elevation_z',
                 'Ep_amp', 'Ep_phase', 'Es_amp', 'Es_phase')

orientationArgSet = {'center', 'pitch', 'roll', 'yaw', 'bragg',
                     'braggOffset', 'rotationSequence', 'positionRoll',
                     'x', 'z'}

shapeArgSet = {'limPhysX', 'limPhysY', 'limPhysX2', 'limPhysY2', 'opening',
               'R', 'r', 'Rm', 'Rs', 'p', 'q', 'f1', 'f2', 'pAxis',
               'parabolaAxis', 'shape', 'renderStyle',
               'n', 'period', 'fileName', 'orientation'}  # TODO: sources

derivedArgSet = {'center', 'pitch', 'bragg', 'R', 'r', 'Rm', 'Rs'}

renderOnlyArgSet = {'renderStyle', 'name'}

compoundArgs = {'center': ['x', 'y', 'z'],
                'lim': ['lmin', 'lmax'],
                'opening': ['left', 'right', 'bottom', 'top'],
                'image': ['width', 'height']}

dependentArgs = {'eSigmaX', 'eSigmaZ', 'betaX', 'betaZ',
                 'K', 'B0', 'rho', 'Kx', 'Ky', 'B0x', 'B0y'}

diagnosticArgs = ('gamma', 'E1', 'eSigmaXprime', 'eSigmaZprime',
                  'ellipseA', 'ellipseB', 'hyperbolaA', 'hyperbolaB')

allUnitsAng = {'rad': 1.,
               'mrad': 1e-3,
               'urad': 1e-6,
               'deg': np.pi/180.,
               'mdeg': 1e-3*np.pi/180.,
               'arcsec': np.pi/180./3600.}

allUnitsAngStr = {'rad': u'rad',
                  'mrad': u'mrad',
                  'urad': u'µrad',
                  'deg': u'°',
                  'mdeg': u'm°',
                  'arcsec': r'arcsec'}

allUnitsLen = {'angstroem': 1e-7,
               'nm': 1e-6,
               'um': 1e-3,
               'mm': 1.,
               'm': 1e3,
               'km': 1e6}

allUnitsLenStr = {'angstroem': u'Å',
                  'nm': r'nm',
                  'um': u'µm',
                  'mm': r'mm',
                  'm': r'm',
                  'km': r'km'}

allUnitsEnergy = {'meV': 1e-3,
                  'eV': 1,
                  'keV': 1e3,
                  'MeV': 1e6,
                  'GeV': 1e9}

allUnitsEnergyStr = {'meV': 'meV',
                     'eV': 'eV',
                     'keV': 'keV',
                     'MeV': 'MeV',
                     'GeV': 'GeV'}

allUnitsEmittance = {'pmrad': 1e-3,
                     'nmrad': 1}

allUnitsEmittanceStr = {'pmrad': 'pm⋅rad',
                        'nmrad': 'nm⋅rad'}

allUnitsCurrent = {'mA': 1e-3,
                   'A': 1}

allUnitsCurrentStr = {'mA': 'mA',
                      'A': 'A'}

lengthUnitParams = {'center': 'mm',
                    'R': 'mm',
                    'r': 'mm',
                    'Rm': 'mm',
                    'Rs': 'mm',
                    'dx': 'mm',
                    'dy': 'mm',
                    'dz': 'mm',
                    'beta': 'm'}  # WIP
