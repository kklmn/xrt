# -*- coding: utf-8 -*-
import numpy as np


def get_energy(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.E


def get_x(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.x


def get_y(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.y


def get_z(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.z


def get_s(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.s if hasattr(beam, 's') else np.zeros_like(beam.x)


def get_phi(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.phi if hasattr(beam, 'phi') else np.zeros_like(beam.x)


def get_r(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.r if hasattr(beam, 'r') else np.zeros_like(beam.x)


def get_a(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.a


def get_b(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.b


def get_xprime(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.a / beam.b


def get_zprime(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.c / beam.b


def get_xzprime(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return (beam.a**2 + beam.c**2)**0.5 / beam.b


def get_path(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.path


def get_order(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.order if hasattr(beam, 'order') else beam.state


def get_reflection_number(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.nRefl if hasattr(beam, 'nRefl') else beam.state


def get_elevation_d(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.elevationD if hasattr(beam, 'elevationD') else \
        np.zeros_like(beam.x)
# if hasattr(beam, 'elevationD') else np.zeros_like(beam.x)


def get_elevation_x(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.elevationX if hasattr(beam, 'elevationX') else beam.x


def get_elevation_y(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.elevationY if hasattr(beam, 'elevationY') else beam.y


def get_elevation_z(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.elevationZ if hasattr(beam, 'elevationZ') else beam.z


def get_Es_amp(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.abs(beam.Es) if hasattr(beam, 'Es') else np.zeros_like(beam.x)


def get_Ep_amp(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.abs(beam.Ep) if hasattr(beam, 'Ep') else np.zeros_like(beam.x)


def get_Es_phase(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.angle(beam.Es) if hasattr(beam, 'Es') else np.zeros_like(beam.x)
#    return np.arctan2(beam.Es.imag, beam.Es.real)


def get_Ep_phase(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.angle(beam.Ep) if hasattr(beam, 'Ep') else np.zeros_like(beam.x)
#    return np.arctan2(beam.Ep.imag, beam.Ep.real)


def get_polarization_degree(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    II = (beam.Jss + beam.Jpp)
    II[II <= 0] = 1.
    pd = np.sqrt((beam.Jss-beam.Jpp)**2 + 4.*abs(beam.Jsp)**2) / II
    pd[II <= 0] = 0.
    return pd


def get_ratio_ellipse_axes(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    dI2 = (beam.Jss - beam.Jpp)**2
    return 2. * beam.Jsp.imag /\
        (np.sqrt(dI2 + 4*abs(beam.Jsp)**2) + np.sqrt(dI2 + 4*beam.Jsp.real**2))


def get_circular_polarization_rate(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    II = (beam.Jss + beam.Jpp)
    II[II <= 0] = 1.
    cpr = 2. * beam.Jsp.imag / II
    cpr[II <= 0] = 0.
    return cpr


def get_polarization_psi(beam):
    """Angle between the semimajor axis of the polarization ellipse relative to
    the s polarization. Used for retrieving data for x-, y- or c-axis of a
    plot."""
#    return 0.5 * np.arctan2(2.*beam.Jsp.real, beam.Jss-beam.Jpp) * 180 / np.pi
    return 0.5 * np.arctan2(2.*beam.Jsp.real, beam.Jss-beam.Jpp)


def get_phase_shift(beam):  # in units of pi!
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return np.angle(beam.Jsp) / np.pi


def get_incidence_angle(beam):
    """Used for retrieving data for x-, y- or c-axis of a plot."""
    return beam.theta if hasattr(beam, 'theta') else np.zeros_like(beam.x)


get_theta = get_incidence_angle
