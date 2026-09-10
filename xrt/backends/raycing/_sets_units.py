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
               'blades', 'vertices',
               'shadeFraction', 'dx', 'dz', 'px', 'pz', 'nx', 'nz',
               'R', 'r', 'Rm', 'Rs', 'p', 'q', 'f1', 'f2', 'pAxis',
               'parabolaAxis', 'shape', 'renderStyle',
               'n', 'period', 'fileName', 'orientation'}  # TODO: sources

derivedArgSet = {'center', 'pitch', 'bragg', 'R', 'r', 'Rm', 'Rs'}

renderOnlyArgSet = {'renderStyle', 'name'}

compoundArgs = {'center': ['x', 'y', 'z'],
                'x': ['x', 'y', 'z'],
                'z': ['x', 'y', 'z'],
                'lim': ['lmin', 'lmax'],
                'limPhysX': ['lmin', 'lmax'],
                'limPhysY': ['lmin', 'lmax'],
                'limPhysX2': ['lmin', 'lmax'],
                'limPhysY2': ['lmin', 'lmax'],
                'limOptX': ['lmin', 'lmax'],
                'limOptY': ['lmin', 'lmax'],
                'limOptX2': ['lmin', 'lmax'],
                'limOptY2': ['lmin', 'lmax'],
                'opening': ['left', 'right', 'bottom', 'top'],
                'blades': ['left', 'right', 'bottom', 'top'],
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
                    'px': 'mm',
                    'pz': 'mm',
                    'beta': 'm'}  # WIP


def auto_unit(lbl, unit):
    uRet = unit
    fRet = None
    if lbl in ['x', 'y', 'z']:
        if unit not in (allUnitsLenStr.keys() |
                        allUnitsLenStr.values()):
            uRet = 'mm'
            fRet = 1
    elif lbl in ["x'", "y'", "z'"]:
        if unit not in (allUnitsAngStr.keys() |
                        allUnitsAngStr.values()):
            uRet = 'mrad'
            fRet = 1e3
    elif lbl in ['energy', 'e']:
        if unit not in (allUnitsEnergyStr.keys() |
                        allUnitsEnergyStr.values()):
            uRet = 'eV'
            fRet = 1
    else:
        uRet = ''

    return uRet, fRet


# Input grammars shared by the GUI argument delegates. Tuple keys describe
# accepted alternatives. Arguments not found here are strings by default. For
# names present in compoundArgs, the grammar is applied to every component and
# compoundArgs supplies the required length.
argumentInputGroups = {
    ('scalar', 'None'): {
        'alarmLevel', 'compressX', 'compressZ', 'eSigmaX', 'eSigmaZ',
        'fixedOffset', 'limOptX', 'limOptX2', 'limOptY', 'limOptY2',
        'limPhysX', 'limPhysX2', 'limPhysY', 'limPhysY2', 'p', 'q', 'R0',
        'Rm', 'RmBragg', 'rho', 'Rs', 'RsBragg', 'seed', 't', 'thinnestZone',
        'totalFlux', 'zmax', 'pickleEvery', 'repeats', 'updateEvery',
        'factor', 'a', 'V'},
    ('scalar', 'inf'): {'substThickness'},
    ('scalar', 'string'): {'processes', 'threads'},
    ('scalar', 'auto'): {'center', 'nrays', 'x', 'z'},
    ('scalar', 'sequence'): {'dx', 'dy', 'dz', 'focus', 'nCRL', 'rms', 'r',
                             'R'},
    ('scalar', 'sequence', 'None'): {'order', 'taper'},
    'angle': {
        'antiblaze', 'blaze', 'braggOffset', 'cryst1roll',
        'cryst2finePitch', 'cryst2pitch', 'cryst2roll', 'extraPitch',
        'extraRoll', 'extraYaw', 'grazingAngle', 'maxxprime', 'maxzprime',
        'minxprime', 'minzprime', 'orientationAngle', 'phaseDeg',
        'positionRoll', 'roll', 'slopeAngle', 'theta', 'wedgeAngle', 'yaw'},
    ('angle', 'None'): {'alpha'},
    ('angle', 'energy', 'auto'): {'bragg', 'pitch'},
    ('angle', 'sequence'): {
        'dxprime', 'dzprime', 'xPrimeMax', 'zPrimeMax'},
    'energy': {'E', 'eE', 'eMax', 'eMin'},
    'string': {
        'afterScript', 'crossSection', 'extraRotationSequence',
        'name', 'orientation', 'rotationSequence', 'title',
        'bl', 'customField', 'efficiencyFile', 'fileName',
        'persistentName', 'saveName', 'beam',
        },
    'format': {'fwhmFormatStr', 'contourFmt', 'fluxFormatStr'},
    'sequence': {
        'atoms', 'atomsXYZ', 'coeffs', 'columnFactors', 'energies',
        'histShape', 'hkl', 'plots', 'vertices'},
    ('sequence', 'None'): {
        'afterScriptArgs', 'atomsFraction', 'cLimits', 'contourColors',
        'contourLevels', 'efficiency', 'energyRange', 'generatorArgs',
        'gratingDensity', 'limits', 'pAxis', 'parabolaAxis', 'quantities',
        'surface', 'targetE'},
    ('string', 'sequence'): {'elements'},
    ('string', 'sequence', 'None'): {'refractiveIndex'},
    'dict': {'afterScriptKWargs', 'blades', 'generatorKWargs'},
    ('sequence', 'inf', 'None'): {'f1', 'f2'},
    'scalar': {
        'amplitude', 'B0', 'B0x', 'B0y', 'betaX', 'betaZ', 'bumpHeight',
        'b', 'bThickness', 'bThicknessLow', 'c',
        'corrLength', 'cryst2longTransl', 'cryst2perpTransl', 'cX', 'cY',
        'd', 'depth', 'dxFacet', 'dxGap', 'dyFacet', 'dyGap', 'eEpsilonX',
        'eEpsilonZ', 'eEspread', 'eI', 'eN', 'ellipseA', 'ellipseB', 'f',
        'factDW', 'gIntervals', 'gp', 'gridStep', 'idThickness', 'K', 'Kx',
        'Ky', 'L0', 'materialsIndex', 'n', 'N', 'nPairs', 'nRK', 'nSpokes',
        'nx', 'nz', 'period', 'phaseShift', 'phi0',
        'contourFactor', 'ePos', 'offset', 'phiOffset', 'ppb', 'px', 'pz',
        'r0', 'raycingParam', 'rx', 'rz',
        'nu', 'power', 'shadeFraction', 'sigmaX', 'sigmaY', 'substRoughness',
        'tK', 'tThickness', 'tThicknessLow', 'thetaOffset', 'vortex',
        'vortexNradial', 'vorticity', 'workingDistance', 'xPos', 'yPos',
        'xWaveLength', 'yWaveLength', 'bins', 'outline', 'beta', 'gamma'},
    }


# No unresolved argument input types at present.
# unknownType = set()


# Arguments for which DynamicArgumentDelegate creates a populated QComboBox.
# comboBoxType = {
#    'aspect', 'autoAppendToBL', 'baseFE', 'distE', 'distx', 'distxprime',
#    'disty', 'distz', 'distzprime', 'figureError', 'filamentBeam',
#    'isCentralZoneBlack', 'isClosed', 'isCylindrical', 'isParametric',
#    'material', 'material2', 'polarization', 'precisionOpenCL', 'recenter',
#    'renderStyle', 'rmsKind', 'shape', 'shouldCheckCenter', 'surfaceHint',
#    'targetOpenCL', 'uniformRayDensity', 'withCentralRay',
#    'xPrimeMaxAutoReduce', 'zPrimeMaxAutoReduce', 'generator', 'beamAbsorb',
#    'beamC', 'beamState', 'bLayer', 'geom', 'kind', 'substrate', 'tLayer',
#    'table',
#    # Plot and axis delegates provide populated selectors for these fields.
#    'beam', 'data', 'density', 'fluxKind', 'fluxUnit', 'label',
#    'rayFlag', 'unit'}
