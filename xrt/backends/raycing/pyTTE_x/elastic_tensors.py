from __future__ import division, print_function
import numpy as np

# Elastic constants for single crystals in units 10^11 Pa
# Source: CRC Handbook of Chemistry and Physics, 82nd edition
#
# Contains only (some) crystals available in xraylib

CRYSTALS = {
    'AlphaQuartz': {
        'system': 'trigonal',
        'C11': 0.8670, 'C12': 0.0704, 'C13': 0.1191, 'C14': -0.1804,
        'C33': 1.0575, 'C44': 0.5820},
    'Be': {
        'system': 'hexagonal',
        'C11': 2.923, 'C12': 0.267, 'C13':  0.140, 'C33': 3.364, 'C55': 1.625},
    'Beryl': {
        'system': 'hexagonal',
        'C11':  2.800, 'C12':  0.990, 'C13':  0.670,
        'C33': 2.480, 'C55':  0.658},
    'Copper': {
        'system': 'cubic', 'C11': 1.683, 'C12':  1.221, 'C44': 0.757},
    # From H.J. McSkimin and P. Andreatch Jr. https://doi.org/10.1063/1.1661636
    'Diamond': {'system': 'cubic', 'C11':  10.79, 'C12':  1.24, 'C44':  5.78},

    # at 300K, from L. J. Slutsky and C. W. Garland
    # https://doi.org/10.1103/PhysRev.113.167
    'InSb': {'system': 'cubic', 'C11': 0.6669, 'C12': 0.3645, 'C44': 0.3020},
    'GaAs': {'system': 'cubic', 'C11': 1.1877, 'C12': 0.5372, 'C44': 0.5944},
    'Ge': {'system': 'cubic', 'C11': 1.2835, 'C12': 0.4823, 'C44': 0.6666},
    'LiF': {'system': 'cubic', 'C11': 1.1397, 'C12': 0.4767, 'C44': 0.6364},
    'Sapphire': {
        'system': 'trigonal',
        'C11': 4.9735, 'C12': 1.6397, 'C13': 1.1220, 'C14': -0.2358,
        'C33': 4.9911, 'C44': 1.4739},
    'Si': {'system': 'cubic', 'C11': 1.6578, 'C12': 0.6394, 'C44': 0.7962},

    'prototype_cubic': {'system': 'cubic', 'C11': 11, 'C12': 12, 'C44': 44},
    'prototype_tetragonal': {
        'system': 'tetragonal', 'C11': 11, 'C12': 12, 'C13': 13, 'C16': 16,
        'C33': 33, 'C44': 44, 'C66': 66},
    'prototype_orthorhombic': {
        'system': 'orthorhombic', 'C11': 11, 'C12': 12, 'C13': 13,
        'C22': 22, 'C23': 23, 'C33': 33, 'C44': 44, 'C55': 55, 'C66': 66},
    'prototype_monoclinic': {
        'system': 'monoclinic', 'C11': 11, 'C12': 12, 'C13': 13, 'C15': 15,
        'C22': 22, 'C23': 23, 'C25': 25,
        'C33': 33, 'C35': 35, 'C44': 44, 'C46': 46, 'C55': 55, 'C66': 66},
    'prototype_hexagonal': {
        'system': 'hexagonal', 'C11': 11, 'C12': 12, 'C13': 13,
        'C33': 33, 'C55': 55},
    'prototype_trigonal': {
        'system': 'trigonal', 'C11': 11, 'C12': 12, 'C13': 13, 'C14': 14,
        'C33': 33, 'C44': 44},
    'prototype_triclinic': {
        'system': 'triclinic', 'C11': 11, 'C12': 12, 'C13': 13, 'C14': 14,
        'C15': 15, 'C16': 16,
        'C22': 22, 'C23': 23, 'C24': 24, 'C25': 25, 'C26': 26,
        'C33': 33, 'C34': 34, 'C35': 35, 'C36': 36,
        'C44': 44, 'C45': 45, 'C46': 46,
        'C55': 55, 'C56': 56,
        'C66': 66}
}

CRYSTALS['Si2'] = CRYSTALS['Si']
CRYSTALS['Si_NIST'] = CRYSTALS['Si']


def list_crystals(remove_prototypes=True):
    '''
    Returns the list of crystals with elastic data available.

    Input:
        remove_prototypes = boolean, whether omit prototypes for crystal data
        entries from the  list (default: True)
    Output:
        xtal_str_list = list of available crystal strings to be used with
        crystal_vectors() or elastic_matrices(). When the crystallographic data
        is available  in xraylib, the strings are congruent.
    '''

    xtal_str_list = list(CRYSTALS.keys())

    if remove_prototypes:
        for i in range(len(xtal_str_list)-1, -1, -1):
            if xtal_str_list[i][:10] == 'prototype_':
                xtal_str_list.pop(i)

    return xtal_str_list


def matrix2tensor(matrix, mtype):
    '''
    Converts the elastic matrices using Voigt notation to elastic tensors.

    Input:
        matrix = 6x6 matrix in Voigt notation
        mtype = 'C' or 'S' for stiffness or compliance matrix, respectively

    Output:
        T = 3x3x3x3 stiffness or compliance tensor
    '''

    T = np.zeros((3, 3, 3, 3))

    if mtype == 'C':
        # Stiffness matrix
        T11 = matrix[0, 0]
        T12 = matrix[0, 1]
        T13 = matrix[0, 2]
        T14 = matrix[0, 3]
        T15 = matrix[0, 4]
        T16 = matrix[0, 5]

        T22 = matrix[1, 1]
        T23 = matrix[1, 2]
        T24 = matrix[1, 3]
        T25 = matrix[1, 4]
        T26 = matrix[1, 5]

        T33 = matrix[2, 2]
        T34 = matrix[2, 3]
        T35 = matrix[2, 4]
        T36 = matrix[2, 5]

        T44 = matrix[3, 3]
        T45 = matrix[3, 4]
        T46 = matrix[3, 5]

        T55 = matrix[4, 4]
        T56 = matrix[4, 5]

        T66 = matrix[5, 5]
    elif mtype == 'S':
        # compliance matrix
        T11 = matrix[0, 0]
        T12 = matrix[0, 1]
        T13 = matrix[0, 2]
        T14 = matrix[0, 3]/2
        T15 = matrix[0, 4]/2
        T16 = matrix[0, 5]/2

        T22 = matrix[1, 1]
        T23 = matrix[1, 2]
        T24 = matrix[1, 3]/2
        T25 = matrix[1, 4]/2
        T26 = matrix[1, 5]/2

        T33 = matrix[2, 2]
        T34 = matrix[2, 3]/2
        T35 = matrix[2, 4]/2
        T36 = matrix[2, 5]/2

        T44 = matrix[3, 3]/4
        T45 = matrix[3, 4]/4
        T46 = matrix[3, 5]/4

        T55 = matrix[4, 4]/4
        T56 = matrix[4, 5]/4

        T66 = matrix[5, 5]/4
    else:
        raise Exception('Invalid elastic matrix type!')

    T[0, 0, 0, 0] = T11
    T[0, 0, 1, 1], T[1, 1, 0, 0] = T12, T12
    T[0, 0, 2, 2], T[2, 2, 0, 0] = T13, T13
    T[0, 0, 1, 2], T[0, 0, 2, 1], T[1, 2, 0, 0], T[2, 1, 0, 0] = \
        T14, T14, T14, T14
    T[0, 0, 2, 0], T[0, 0, 0, 2], T[0, 2, 0, 0], T[2, 0, 0, 0] = \
        T15, T15, T15, T15
    T[0, 0, 0, 1], T[0, 0, 1, 0], T[0, 1, 0, 0], T[1, 0, 0, 0] = \
        T16, T16, T16, T16

    T[1, 1, 1, 1] = T22
    T[1, 1, 2, 2], T[2, 2, 1, 1] = T23, T23
    T[1, 1, 1, 2], T[1, 1, 2, 1], T[1, 2, 1, 1], T[2, 1, 1, 1] = \
        T24, T24, T24, T24
    T[1, 1, 2, 0], T[1, 1, 0, 2], T[0, 2, 1, 1], T[2, 0, 1, 1] = \
        T25, T25, T25, T25
    T[1, 1, 0, 1], T[1, 1, 1, 0], T[0, 1, 1, 1], T[1, 0, 1, 1] = \
        T26, T26, T26, T26

    T[2, 2, 2, 2] = T33
    T[2, 2, 1, 2], T[2, 2, 2, 1], T[1, 2, 2, 2], T[2, 1, 2, 2] = \
        T34, T34, T34, T34
    T[2, 2, 2, 0], T[2, 2, 0, 2], T[0, 2, 2, 2], T[2, 0, 2, 2] = \
        T35, T35, T35, T35
    T[2, 2, 0, 1], T[2, 2, 1, 0], T[0, 1, 2, 2], T[1, 0, 2, 2] = \
        T36, T36, T36, T36

    T[1, 2, 1, 2], T[1, 2, 2, 1], T[2, 1, 1, 2], T[2, 1, 2, 1] = \
        T44, T44, T44, T44
    T[1, 2, 2, 0], T[1, 2, 0, 2], T[2, 1, 2, 0], T[2, 1, 0, 2] = \
        T45, T45, T45, T45
    T[2, 0, 1, 2], T[0, 2, 1, 2], T[2, 0, 2, 1], T[0, 2, 2, 1] = \
        T45, T45, T45, T45
    T[1, 2, 0, 1], T[1, 2, 1, 0], T[2, 1, 0, 1], T[2, 1, 1, 0] = \
        T46, T46, T46, T46
    T[0, 1, 1, 2], T[1, 0, 1, 2], T[0, 1, 2, 1], T[1, 0, 2, 1] = \
        T46, T46, T46, T46

    T[2, 0, 2, 0], T[2, 0, 0, 2], T[0, 2, 2, 0], T[0, 2, 0, 2] = \
        T55, T55, T55, T55
    T[2, 0, 0, 1], T[2, 0, 1, 0], T[0, 2, 0, 1], T[0, 2, 1, 0] = \
        T56, T56, T56, T56
    T[0, 1, 2, 0], T[1, 0, 2, 0], T[0, 1, 0, 2], T[1, 0, 0, 2] = \
        T56, T56, T56, T56

    T[0, 1, 0, 1], T[0, 1, 1, 0], T[1, 0, 0, 1], T[1, 0, 1, 0] = \
        T66, T66, T66, T66

    return T


def tensor2matrix(tensor, ttype):
    '''
    Converts the elastic tensors to matrices using Voigt notation.

    Input:
        tensor = 3x3x3x3 elastic tensor
        mtype = 'C' or 'S' for stiffness or compliance tensor, respectively

    Output:
        matrix = 6x6 stiffness or compliance matrix
    '''

    T = tensor

    if ttype == 'C':
        # stiffness matrix
        matrix = np.array([
            [T[0, 0, 0, 0], T[0, 0, 1, 1], T[0, 0, 2, 2],
                T[0, 0, 1, 2], T[0, 0, 0, 2], T[0, 0, 0, 1]],
            [T[1, 1, 0, 0], T[1, 1, 1, 1], T[1, 1, 2, 2],
                T[1, 1, 1, 2], T[1, 1, 0, 2], T[1, 1, 0, 1]],
            [T[2, 2, 0, 0], T[2, 2, 1, 1], T[2, 2, 2, 2],
                T[2, 2, 1, 2], T[2, 2, 0, 2], T[2, 2, 0, 1]],
            [T[2, 1, 0, 0], T[2, 1, 1, 1], T[2, 1, 2, 2],
                T[1, 2, 1, 2], T[1, 2, 0, 2], T[1, 2, 0, 1]],
            [T[2, 0, 0, 0], T[2, 0, 1, 1], T[2, 0, 2, 2],
                T[2, 0, 1, 2], T[0, 2, 0, 2], T[2, 0, 0, 1]],
            [T[1, 0, 0, 0], T[1, 0, 1, 1], T[1, 0, 2, 2], T[1, 0, 1, 2],
             T[1, 0, 0, 2], T[0, 1, 0, 1]]])

    elif ttype == 'S':
        # compliance matrix
        matrix = np.array([
            [T[0, 0, 0, 0],   T[0, 0, 1, 1],   T[0, 0, 2, 2], 2 *
                T[0, 0, 1, 2], 2*T[0, 0, 0, 2], 2*T[0, 0, 0, 1]],
            [T[1, 1, 0, 0],   T[1, 1, 1, 1],   T[1, 1, 2, 2], 2 *
                T[1, 1, 1, 2], 2*T[1, 1, 0, 2], 2*T[1, 1, 0, 1]],
            [T[2, 2, 0, 0],   T[2, 2, 1, 1],   T[2, 2, 2, 2], 2 *
                T[2, 2, 1, 2], 2*T[2, 2, 0, 2], 2*T[2, 2, 0, 1]],
            [2*T[2, 1, 0, 0], 2*T[2, 1, 1, 1], 2*T[2, 1, 2, 2],
                4*T[1, 2, 1, 2], 4*T[1, 2, 0, 2], 4*T[1, 2, 0, 1]],
            [2*T[2, 0, 0, 0], 2*T[2, 0, 1, 1], 2*T[2, 0, 2, 2],
                4*T[2, 0, 1, 2], 4*T[0, 2, 0, 2], 4*T[2, 0, 0, 1]],
            [2*T[1, 0, 0, 0], 2*T[1, 0, 1, 1], 2*T[1, 0, 2, 2],
             4*T[1, 0, 1, 2], 4*T[1, 0, 0, 2], 4*T[0, 1, 0, 1]]])
    else:
        raise Exception('Invalid elastic tensor type!')

    return matrix


def elastic_matrices(xtal_str):
    '''
    Returns the stiffness and compliance matrices for a given crystal.

    Input:
        xtal_str = crystal string e.g. 'Si', 'Ge', 'AlphaQuartz'
    Output:
        C_matrix = stiffness matrix in units 10^{11} Pa
        S_matrix = compliance matrix in units 10^{-11} Pa^-1
    '''

    try:
        xtal_data = CRYSTALS[xtal_str]
    except KeyError:
        raise KeyError("Elastic parameters for '"+str(xtal_str)+"' not found!")

    C11, C12, C13, C14, C15, C16 = 0, 0, 0, 0, 0, 0
    C22, C23, C24, C25, C26 = 0, 0, 0, 0, 0
    C33, C34, C35, C36 = 0, 0, 0, 0
    C44, C45, C46 = 0, 0, 0
    C55, C56 = 0, 0
    C66 = 0

    if xtal_data['system'] == 'cubic':
        C11, C12, C44 = xtal_data['C11'], xtal_data['C12'], xtal_data['C44']
        C22, C13, C23, C33, C55, C66 = C11, C12, C12, C11, C44, C44
    elif xtal_data['system'] == 'tetragonal':
        C11, C12, C13, C16 = xtal_data['C11'], xtal_data['C12'], \
            xtal_data['C13'], xtal_data['C16']
        C33, C44, C66 = xtal_data['C33'], xtal_data['C44'], xtal_data['C66']
        C22, C23, C26, C55 = C11, C13, -C16, C44
    elif xtal_data['system'] == 'orthorhombic':
        C11, C12, C13, C22, C23 = xtal_data['C11'], xtal_data[
            'C12'], xtal_data['C13'], xtal_data['C22'], xtal_data['C23']
        C33, C44, C55, C66 = xtal_data['C33'], xtal_data['C44'], \
            xtal_data['C55'], xtal_data['C66']
    elif xtal_data['system'] == 'monoclinic':
        C11, C12, C13, C15 = xtal_data['C11'], xtal_data['C12'], \
            xtal_data['C13'], xtal_data['C15']
        C22, C23, C25, C33, C35 = xtal_data['C22'], xtal_data[
            'C23'], xtal_data['C25'], xtal_data['C33'], xtal_data['C35']
        C44, C46, C55, C66 = xtal_data['C44'], xtal_data['C46'], \
            xtal_data['C55'], xtal_data['C66']
    elif xtal_data['system'] == 'hexagonal':
        C11, C12, C13 = xtal_data['C11'], xtal_data['C12'], xtal_data['C13']
        C33, C55 = xtal_data['C33'], xtal_data['C55']
        C22, C23, C44 = C11, C13, C55
        C66 = (C11-C12)/2
    elif xtal_data['system'] == 'trigonal':
        C11, C12, C13, C14 = xtal_data['C11'], xtal_data['C12'], \
            xtal_data['C13'], xtal_data['C14']
        C33, C44 = xtal_data['C33'], xtal_data['C44']
        C22, C23, C24, C55, C56 = C11, C13, -C14, C44, C14
        C66 = (C11-C12)/2
    elif xtal_data['system'] == 'triclinic':
        C11, C12, C13, C14, C15, C16 = xtal_data['C11'], xtal_data['C12'], \
            xtal_data['C13'], xtal_data['C14'], xtal_data['C15'], \
            xtal_data['C16']
        C22, C23, C24, C25, C26 = xtal_data['C22'], xtal_data['C23'], \
            xtal_data['C24'], xtal_data['C25'], xtal_data['C26']
        C33, C34, C35, C36 = xtal_data['C33'], xtal_data['C34'], \
            xtal_data['C35'], xtal_data['C36']
        C44, C45, C46 = xtal_data['C44'], xtal_data['C45'], xtal_data['C46']
        C55, C56 = xtal_data['C55'], xtal_data['C56']
        C66 = xtal_data['C66']
    else:
        ValueError('Not a valid crystal system!')

    # Elastic matrices of the non-rotated coordinate system
    C_matrix = np.array([[C11, C12, C13, C14, C15, C16],
                         [C12, C22, C23, C24, C25, C26],
                         [C13, C23, C33, C34, C35, C36],
                         [C14, C24, C34, C44, C45, C46],
                         [C15, C25, C35, C45, C55, C56],
                         [C16, C26, C36, C46, C56, C66]])

    S_matrix = np.linalg.inv(C_matrix)

    return C_matrix, S_matrix


def rotate_elastic_tensor(tensor, rotation_matrix):
    '''
    Performs the rotation described by rotation_matrix to the given elastic
    tensor.

    Input:
        tensor = 3x3x3x3 elastic tensor
        rotation_matrix = 3x3 rotation matrix
    Output:
        tensor_rot = rotated tensor
    '''
    tensor_rot = tensor
    for i in range(4):
        tensor_rot = np.tensordot(
            rotation_matrix, tensor_rot, axes=((1,), (i,)))

    return tensor_rot


def rotate_elastic_matrix(matrix, mtype, rotation_matrix):
    '''
    Performs the rotation described by rotation_matrix to the given elastic
    matrix.

    Input:
        matrix = 6x6 elastic matrix
        mtype = 'C' if the matrix to be rotated is the stiffness matrix
                or 'S' if the compliance matrix
        rotation_matrix = 3x3 rotation matrix
    Output:
        matrix_rot = rotated matrix
    '''
    tensor = matrix2tensor(matrix, mtype)
    tensor_rot = rotate_elastic_tensor(tensor, rotation_matrix)
    matrix_rot = tensor2matrix(tensor_rot, mtype)

    return matrix_rot
