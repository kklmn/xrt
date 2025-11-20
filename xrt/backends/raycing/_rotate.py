# -*- coding: utf-8 -*-
import numpy as np


def rotate_x(y, z, cosangle, sinangle):
    """3D rotaion around *x* (pitch). *y* and *z* are values or arrays.
    Positive rotation is for positive *sinangle*. Returns *yNew, zNew*."""
    return cosangle*y - sinangle*z, sinangle*y + cosangle*z


def rotate_y(x, z, cosangle, sinangle):
    """3D rotaion around *y* (roll). *x* and *z* are values or arrays.
    Positive rotation is for positive *sinangle*. Returns *xNew, zNew*."""
    return cosangle*x + sinangle*z, -sinangle*x + cosangle*z


def rotate_z(x, y, cosangle, sinangle):
    """3D rotaion around *z*. *x* and *y* are values or arrays.
    Positive rotation is for positive *sinangle*. Returns *xNew, yNew*."""
    return cosangle*x - sinangle*y, sinangle*x + cosangle*y


def rotate_beam(beam, indarr=None, rotationSequence='RzRyRx',
                pitch=0, roll=0, yaw=0, skip_xyz=False, skip_abc=False,
                is2ndXtal=False):
    """Rotates the *beam* indexed by *indarr* by the angles *yaw, roll, pitch*
    in the sequence given by *rotationSequence*. A leading '-' symbol of
    *rotationSequence* reverses the sequences.
    """
    angles = {'z': yaw, 'y': roll, 'x': pitch}
    rotates = {'z': rotate_z, 'y': rotate_y, 'x': rotate_x}
    if not skip_xyz:
        coords1 = {'z': beam.x, 'y': beam.x, 'x': beam.y}
        coords2 = {'z': beam.y, 'y': beam.z, 'x': beam.z}
    if not skip_abc:
        vcomps1 = {'z': beam.a, 'y': beam.a, 'x': beam.b}
        vcomps2 = {'z': beam.b, 'y': beam.c, 'x': beam.c}

    if rotationSequence[0] == '-':
        seq = rotationSequence[6] + rotationSequence[4] + rotationSequence[2]
    else:
        seq = rotationSequence[1] + rotationSequence[3] + rotationSequence[5]
    for s in seq:
        angle, rotate = angles[s], rotates[s]
        if not skip_xyz:
            c1, c2 = coords1[s], coords2[s]
        if not skip_abc:
            v1, v2 = vcomps1[s], vcomps2[s]
        if angle != 0:
            cA = np.cos(angle)
            sA = np.sin(angle)
            if indarr is None:
                indarr = slice(None)
            if not skip_xyz:
                c1[indarr], c2[indarr] = rotate(c1[indarr], c2[indarr], cA, sA)
            if not skip_abc:
                v1[indarr], v2[indarr] = rotate(v1[indarr], v2[indarr], cA, sA)


def rotate_xyz(x, y, z, indarr=None, rotationSequence='RzRyRx',
               pitch=0, roll=0, yaw=0):
    """Rotates the arrays *x*, *y* and *z* indexed by *indarr* by the angles
    *yaw, roll, pitch* in the sequence given by *rotationSequence*. A leading
    '-' symbol of *rotationSequence* reverses the sequences.
    """
    angles = {'z': yaw, 'y': roll, 'x': pitch}
    rotates = {'z': rotate_z, 'y': rotate_y, 'x': rotate_x}
    coords1 = {'z': x, 'y': x, 'x': y}
    coords2 = {'z': y, 'y': z, 'x': z}

    if rotationSequence[0] == '-':
        seq = rotationSequence[6] + rotationSequence[4] + rotationSequence[2]
    else:
        seq = rotationSequence[1] + rotationSequence[3] + rotationSequence[5]
    for s in seq:
        angle, rotate = angles[s], rotates[s]
        c1, c2 = coords1[s], coords2[s]
        if angle != 0:
            cA = np.cos(angle)
            sA = np.sin(angle)
            if indarr is None:
                indarr = slice(None)
            c1[indarr], c2[indarr] = rotate(c1[indarr], c2[indarr], cA, sA)
    return x, y, z


def rotate_point(point, rotationSequence='RzRyRx', pitch=0, roll=0, yaw=0):
    """Rotates the *point* (3-sequence) by the angles *yaw, roll, pitch*
    in the sequence given by *rotationSequence*. A leading '-' symbol of
    *rotationSequence* reverses the sequences.
    """
    angles = {'z': yaw, 'y': roll, 'x': pitch}
    rotates = {'z': rotate_z, 'y': rotate_y, 'x': rotate_x}
    ind1 = {'z': 0, 'y': 0, 'x': 1}
    ind2 = {'z': 1, 'y': 2, 'x': 2}
    newp = [coord for coord in point]
    if rotationSequence[0] == '-':
        seq = rotationSequence[6] + rotationSequence[4] + rotationSequence[2]
    else:
        seq = rotationSequence[1] + rotationSequence[3] + rotationSequence[5]
    for s in seq:
        angle, rotate = angles[s], rotates[s]
        if angle != 0:
            cA = np.cos(angle)
            sA = np.sin(angle)
            newp[ind1[s]], newp[ind2[s]] = rotate(
                newp[ind1[s]], newp[ind2[s]], cA, sA)
    return newp
