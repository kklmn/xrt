from __future__ import division, print_function
import numpy as np


def crystal_vectors(xtal):
    '''
    Calculates the direct and reciprocal unit vectors for a given
    crystal data obtained from xraylib.Crystal_GetCrystal(). Assumes the direct
    vector a1 to be perpendicular to x-axis and a2 to lie in xy-plane.

    Parameters
    ----------

    xtal : dict
        Representation of a crystal returned by xraylib.Crystal_GetCrystal()


    Returns
    -------

        a_matrix : 3x3 Numpy array
            Matrix containing the direct unit vectors as columns in
            angstroms.

        b_matrix : 3x3 Numpy array
            Matrix containing the reciprocal unit vectors as columns in
            inverse angstroms.

    '''

    a = xtal['a']
    b = xtal['b']
    c = xtal['c']
    alpha = np.radians(xtal['alpha'])
    beta = np.radians(xtal['beta'])
    gamma = np.radians(xtal['gamma'])

    # calculate the direct vectors
    a1 = a*np.array([[1, 0, 0]])
    a2 = b*np.array([[np.cos(gamma), np.sin(gamma), 0]])

    aux1 = np.cos(beta) * np.sin(gamma)
    aux2 = np.cos(alpha) - np.cos(beta) * np.cos(gamma)
    aux3 = np.sqrt(np.sin(gamma)**2 - np.cos(alpha)**2 - np.cos(beta)**2
                   - 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))

    a3 = c/np.sin(gamma)*np.array([[aux1, aux2, aux3]])

    # calculate the reciprocal vectors
    volume = np.dot(np.cross(a1, a2), a3.T)
    b1 = 2*np.pi*np.cross(a2, a3)/volume
    b2 = 2*np.pi*np.cross(a3, a1)/volume
    b3 = 2*np.pi*np.cross(a1, a2)/volume

    a_matrix = np.concatenate((a1, a2, a3)).T
    b_matrix = np.concatenate((b1, b2, b3)).T

    return a_matrix, b_matrix
