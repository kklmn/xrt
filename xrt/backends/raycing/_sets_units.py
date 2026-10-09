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

shapeArgSet = {'limPhysX', 'limPhysY', 'limPhysX2', 'limPhysY2',
               'limOptX', 'limOptY', 'limOptX2', 'limOptY2', 'opening',
               'blades', 'vertices',
               'shadeFraction', 'dx', 'dz', 'px', 'pz', 'nx', 'nz',
               'R', 'r', 'Rm', 'Rs', 'p', 'q', 'f1', 'f2', 'pAxis',
               'parabolaAxis', 'shape', 'renderStyle', 'renderSize',
               'n', 'period', 'fileName', 'orientation',
               'surfaceHintX', 'surfaceHintY',
               'focus', 'zmax', 't', 'nCRL'}  # TODO: sources

calculatedArgSet = {'R', 'r', 'Rm', 'Rs', 'focus', 'nCRL'}
derivedArgSet = {'center', 'pitch', 'bragg'} | calculatedArgSet

renderOnlyArgSet = {'renderStyle', 'renderSize', 'name'}

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

dependentArgGroups = (
    ('eSigmaX', 'betaX', 'eEpsilonX'),
    ('eSigmaZ', 'betaZ', 'eEpsilonZ'),
    ('K', 'B0', 'rho', 'period', 'eE'),
    ('Kx', 'B0x', 'period', 'eE', 'targetE'),
    ('Ky', 'B0y', 'K', 'period', 'eE', 'targetE'),
    ('rho', 'coeffs', 'coefficientConvention'),
    ('p', 'f1'),
    ('q', 'f2'),
    ('p', 'q'),
    ('cryst2perpTransl', 'fixedOffset', 't'),
    ('surfaceHintX', 'surfaceHintY'),
    ('focus', 'nCRL', 'material'),
)

diagnosticArgs = ('gamma', 'E1', 'eSigmaXprime', 'eSigmaZprime',
                  'ellipseA', 'ellipseB', 'hyperbolaA', 'hyperbolaB', 'cff',
                  'diffractionAngle', 'includedAngle',
                  'RsagFit', 'RmerFit', 'conicXFit', 'conicYFit', 'fitRmsError')

allUnitsAng = {'rad': 1.,
               'mrad': 1e-3,
               'urad': 1e-6,
               'nrad': 1e-9,
               'deg': np.pi/180.,
               'mdeg': 1e-3*np.pi/180.,
               'arcsec': np.pi/180./3600.}

allUnitsAngStr = {'rad': u'rad',
                  'mrad': u'mrad',
                  'urad': u'µrad',
                  'nrad': u'nrad',
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

# GUI unit hints for arguments that depart from the mm/rad/eV defaults.
# Values describe bare numeric input (or diagnostic output), not conversions.
argumentUnitExceptions = {
    'eE': 'GeV',
    'eI': 'A',
    'eSigmaX': 'µm',
    'eSigmaZ': 'µm',
    'eEpsilonX': 'nm·rad',
    'eEpsilonZ': 'nm·rad',
    'betaX': 'm',
    'betaZ': 'm',
    'xPrimeMax': 'mrad',
    'zPrimeMax': 'mrad',
    'B0': 'T',
    'B0x': 'T',
    'B0y': 'T',
    'phaseDeg': '°',
    'polarization': '° (numeric linear polarization angle)',
    'd': 'Å (crystal interatomic spacing)',
    'V': 'Å³',
    'tK': 'K',
    'tThickness': 'Å',
    'bThickness': 'Å',
    'tThicknessLow': 'Å',
    'bThicknessLow': 'Å',
    'idThickness': 'Å',
    'substThickness': 'Å',
    'substRoughness': 'Å',
    'cThickness': 'Å',
    'surfaceRoughness': 'Å',
    'bumpHeight': 'nm',
    'diffractionAngle': '°',
    'includedAngle': '°',
    'fitRmsError': 'µm',
    'cameraAngle': '°',
    'rotations': '° (scene view)',
}


# Reused argument names mapped to (class path, unit) alternatives.
# Class paths are resolved by the GUI after the backend has initialized.
argumentUnitContextExceptions = {
    'a': (('xrt.backends.raycing.materials.Crystal', 'Å'),),
    'b': (('xrt.backends.raycing.materials.Crystal', 'Å'),),
    'c': (('xrt.backends.raycing.materials.Crystal', 'Å'),),
    'alpha': (('xrt.backends.raycing.materials.CrystalFromCell', '°'),),
    'beta': (('xrt.backends.raycing.materials.CrystalFromCell', '°'),),
    'gamma': (('xrt.backends.raycing.materials.CrystalFromCell', '°'),),
    'rho': (
        ('xrt.backends.raycing.sources.BendingMagnet', 'm'),
        ('xrt.backends.raycing.materials.Material', 'g/cm³'),
        ('xrt.backends.raycing.oes.gratings.ProfiledGrating', 'mm⁻¹')),
    'amplitude': (('xrt.backends.raycing.figure_error.Waviness', 'nm'),),
    'rms': (('xrt.backends.raycing.figure_error.RandomRoughness',
             'nm (rmsKind=height); µrad (rmsKind=slope)'),),
}


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
    from ._flow_utils import normalize_mu

    unit = normalize_mu(unit)
    uRet = unit
    fRet = None
    if lbl in ['x', 'y', 'z', 'r', 's']:
        if unit not in (allUnitsLenStr.keys() |
                        allUnitsLenStr.values()):
            uRet = 'mm'
            fRet = 1
    elif lbl in ["x'", "y'", "z'", "theta", "phi"]:
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
    ('float', 'None'): {
        'alarmLevel', 'compressX', 'compressZ', 'eSigmaX', 'eSigmaZ',
        'fixedOffset', 'limOptX', 'limOptX2', 'limOptY', 'limOptY2',
        'limPhysX', 'limPhysX2', 'limPhysY', 'limPhysY2', 'p', 'q', 'R0',
        'RmBragg', 'rho', 'RsBragg', 't', 'thinnestZone',
        'totalFlux', 'zmax',
        'factor', 'a', 'V', 'fixedEnergy'},
    ('integer', 'None'): {'seed', 'pickleEvery', 'repeats', 'updateEvery'},
    ('float', 'inf'): {'substThickness'},
    ('integer', 'half', 'all'): {'processes', 'threads'},
    ('float', 'auto'): {'center', 'x', 'z'},
    ('float', 'sequence'): {'dx', 'dy', 'dz', 'focus', 'nCRL', 'rms', 'r',
                            'R', 'w0'},
    ('float', 'sequence', 'None'): {'taper', 'Rm', 'Rs'},
    ('integer', 'sequence', 'None'): {'order'},
    'angle': {
        'antiblaze', 'blaze', 'braggOffset', 'cryst1roll', 'mosaicity',
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
        'customField', 'efficiencyFile', 'fileName',
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
        'surface', 'targetE', 'jack1', 'jack2', 'jack3', 'tx1', 'tx2'},
    ('string', 'sequence'): {'elements'},
    ('string', 'sequence', 'None'): {'refractiveIndex'},
    'dict': {'afterScriptKWargs', 'blades', 'generatorKWargs', 'renderSize'},
    ('sequence', 'inf', 'None'): {'f1', 'f2'},
    'integer': {
        'eN', 'gIntervals', 'N', 'nPairs', 'nRK', 'nSpokes',
        'nx', 'nz', 'nrays', 'ppb', 'bins', 'vortexNradial'},
    'float': {
        'amplitude', 'B0', 'B0x', 'B0y', 'betaX', 'betaZ', 'bumpHeight',
        'b', 'bThickness', 'bThicknessLow', 'c', 'cameraAngle',
        'cameraDistance', 'coordOffset', 'azimuth', 'height',
        'corrLength', 'cryst2longTransl', 'cryst2perpTransl', 'cX', 'cY',
        'd', 'depth', 'dxFacet', 'dxGap', 'dyFacet', 'dyGap', 'eEpsilonX',
        'eEpsilonZ', 'eEspread', 'eI', 'ellipseA', 'ellipseB', 'f',
        'factDW', 'gp', 'gridStep', 'idThickness', 'K', 'Kx',
        'Ky', 'L0', 'materialsIndex', 'n', 'period', 'phaseShift', 'phi0',
        'contourFactor', 'ePos', 'offset', 'phiOffset', 'px', 'pz',
        'r0', 'raycingParam', 'rx', 'rz',
        'nu', 'power', 'rotations', 'scaleVec', 'shadeFraction', 'sigmaX',
        'sigmaY', 'substRoughness',
        'tK', 'tThickness', 'tThicknessLow', 'thetaOffset', 'vortex',
        'vorticity', 'workingDistance', 'xPos', 'yPos',
        'xWaveLength', 'yWaveLength', 'tVec', 'outline', 'beta',
        'gamma', 'xCylinder1', 'hCylinder1', 'xCylinder2', 'hCylinder2'},
    }


# Optional instructions for input types. auto and None have no extra help.
argumentInputTooltips = {
    'angle': (
        'Enter an angle with an optional unit, e.g. <nobr><code>3 mrad</code></nobr> '
        'or <nobr><code>0.2 deg</code></nobr>.\n'
        'Bare numbers use {unit}.\n'
        '<b>Units:</b> {angleUnits}.'),
    'integer': (
        'Enter a whole number.\n'
        'Scientific notation and arithmetic are accepted '
        'when the result is a whole number.'),
}

# Parameter-specific instructions override generic input-type instructions.
argumentTooltips = {
    'order': 'Diffraction order, or a sequence of integer diffraction orders.',
    'nCRL': (
        '<b>Number:</b> rounded to the nearest integer; minimum 1.\n'
        '<b>Tuple:</b> <nobr><code>(focalDistance, E)</code></nobr> in mm and eV '
        'calculates the count.\n\n'
        'Keep <code>focus</code> numeric.'),
    'focus': (
        'Parabola focal parameter in mm.\n'
        'For a target lens focal distance, enter '
        '<nobr><code>(focalDistance, E)</code></nobr> in mm and eV.\n\n'
        'Keep <code>nCRL</code> numeric.'),
    'R': (
        'Meridional radius in mm, or <nobr><code>(p, q)</code></nobr>\n'
        'with object and image distances in mm.\n'
        'Elements supporting an explicit angle also accept '
        '<nobr><code>(p, q, pitch)</code></nobr>, with an optional angular unit '
        'for <code>pitch</code>.'),
    'r': (
        'Sagittal radius in mm, or <nobr><code>(p, q)</code></nobr>\n'
        'with object and image distances in mm.\n'
        'Elements supporting an explicit angle also accept '
        '<nobr><code>(p, q, pitch)</code></nobr>, with an optional angular unit '
        'for <code>pitch</code>.'),
    'Rm': (
        'Meridional radius in mm.\n'
        'On elements supporting automatic focusing, '
        'enter <nobr><code>(p, q)</code></nobr> with object and image distances in mm.'),
    'Rs': (
        'Sagittal radius in mm.\n'
        'On elements supporting automatic focusing, '
        'enter <nobr><code>(p, q)</code></nobr> with object and image distances in mm.'),
    'pitch': (
        'Enter an angle, e.g. <nobr><code>3 mrad</code></nobr> or <nobr><code>0.2 deg</code></nobr>,\n'
        'or an alignment energy, e.g. <nobr><code>9 keV</code></nobr>.\n\n'
        'Bare numbers use {unit}.'),
    'bragg': (
        'Enter an angle, e.g. <nobr><code>3 mrad</code></nobr> or <nobr><code>0.2 deg</code></nobr>,\n'
        'or an alignment energy, e.g. <nobr><code>9 keV</code></nobr>.\n\n'
        'Bare numbers use {unit}.'),
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
