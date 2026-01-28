# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:17:55 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import numpy as np  # analysis:ignore
from matplotlib.colors import hsv_to_rgb  # analysis:ignore

from ..commons import qt  # analysis:ignore

from ...backends.raycing import sources as rsources  # analysis:ignore
from ...backends.raycing import screens as rscreens  # analysis:ignore
from ...backends.raycing import oes as roes  # analysis:ignore
from ...backends.raycing import apertures as rapertures  # analysis:ignore


def create_qt_buffer(data, isIndex=False,
                     usage_hint=qt.QOpenGLBuffer.DynamicDraw):
    """Create and populate a QOpenGLBuffer."""
    bufferType = qt.QOpenGLBuffer.IndexBuffer if isIndex else\
        qt.QOpenGLBuffer.VertexBuffer
    buffer = qt.QOpenGLBuffer(bufferType)
    buffer.create()
    buffer.setUsagePattern(usage_hint)
    buffer.bind()
    data = np.array(data, np.uint32 if isIndex else np.float32)
    buffer.allocate(data.tobytes(), data.nbytes)
    buffer.release()

    return buffer


def update_qt_buffer(buffer, data, isIndex=False):
    buffer.bind()
    data = np.array(data, np.uint32 if isIndex else np.float32)
    buffer.write(0, data, data.nbytes)
    buffer.release()


def generate_hsv_texture(width, s, v):
    h = np.linspace(0., 1., width, endpoint=False)
    hsv_data = hsv_to_rgb(np.vstack(
            (h, s*np.ones_like(h), v*np.ones_like(h))).T)
    return (hsv_data * 255).astype(np.uint8)


def basis_rotation_q(xyz_start, xyz_end):
    """This function calculates quaternion that transfroms a basis set to
    a new one.
    xyz_start: nested list or 3x3 numpy array representing 3 vectors defining
    the initial orthogonal basis set
    xyz_start: nested list or 3x3 numpy array representing the target
    orthogonal basis set

    """
    U = np.array(xyz_start, dtype=float)
    V = np.array(xyz_end, dtype=float)

    R = np.matmul(V, U.T)

    tr = np.trace(R)

    Q0 = lambda M, G: 0.25*G
    Q1 = lambda M, G: (M[2, 1]-M[1, 2])/G
    Q2 = lambda M, G: (M[0, 2]-M[2, 0])/G
    Q3 = lambda M, G: (M[1, 0]-M[0, 1])/G
    Q4 = lambda M, G: (M[1, 0]+M[0, 1])/G
    Q5 = lambda M, G: (M[2, 0]+M[0, 2])/G
    Q6 = lambda M, G: (M[1, 2]+M[2, 1])/G

    if tr > 0:
        S = 2*np.sqrt(tr + 1.)
        q = [Q0(R, S), Q1(R, S), Q2(R, S), Q3(R, S)]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2*np.sqrt(1. + R[0, 0] - R[1, 1] - R[2, 2])
        q = [Q1(R, S), Q0(R, S), Q4(R, S), Q5(R, S)]
    elif R[1, 1] > R[2, 2]:
        S = 2*np.sqrt(1. + R[1, 1] - R[0, 0] - R[2, 2])
        q = [Q2(R, S), Q4(R, S), Q0(R, S), Q6(R, S)]
    else:
        S = 2*np.sqrt(1. + R[2, 2] - R[0, 0] - R[1, 1])
        q = [Q3(R, S), Q5(R, S), Q6(R, S), Q0(R, S)]
    qnp = np.array(q)

    return qnp/np.linalg.norm(qnp)


def is_oe(oe):
    return isinstance(oe, roes.OE)


def is_dcm(oe):
    return isinstance(oe, roes.DCM)


def is_plate(oe):
    return isinstance(oe, roes.Plate)


def is_screen(oe):
    return isinstance(oe, rscreens.Screen)


def is_aperture(oe):
    res = isinstance(oe, (rapertures.RectangularAperture,
                          rapertures.RoundAperture,
                          rapertures.PolygonalAperture))
    return res


def is_source(oe):
    res = isinstance(oe, (rsources.SourceBase, rsources.GeometricSource,
                          rsources.GaussianBeam, rsources.MeshSource,
                          rsources.CollimatedMeshSource,
                          rsources.BeamFromFile))
    return res


def snsc(angleDeg, phDeg):
    angleRad = np.radians(angleDeg-phDeg)
    return 0.5*(np.sign(np.cos(angleRad))+np.sign(np.sin(angleRad)))
