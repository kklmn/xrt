from __future__ import division, print_function
import numpy as np


def axis_angle(u,theta):
    '''
    Computes a matrix which performs a rotation of theta degrees counterclockwise about axis u.

    Input:
        u = 3-element vector giving the axis of rotation (does not need to be normalized)
        theta = angle of counterclockwise rotation (in deg) 
    Output:
        R = 3x3 rotation matrix
    '''    

    #normalize
    u = np.array(u)
    u = u/np.sqrt(u[0]**2+u[1]**2+u[2]**2)
    #rotation angle
    th = np.radians(theta)

    #rotation matrix
    R=np.array([[        np.cos(th) + u[0]**2*(1-np.cos(th)), u[0]*u[1]*(1-np.cos(th)) - u[2]*np.sin(th), u[0]*u[2]*(1-np.cos(th)) + u[1]*np.sin(th)],
                [ u[0]*u[1]*(1-np.cos(th)) + u[2]*np.sin(th),        np.cos(th) + u[1]**2*(1-np.cos(th)), u[1]*u[2]*(1-np.cos(th)) - u[0]*np.sin(th)],
                [ u[0]*u[2]*(1-np.cos(th)) - u[1]*np.sin(th), u[1]*u[2]*(1-np.cos(th)) + u[0]*np.sin(th),        np.cos(th) + u[2]**2*(1-np.cos(th))]])

    return R


def align_vector_with_z_axis(h):
    '''
    Computes the rotation matrix which aligns the given vector along z-axis.
    For example, for reflection (hkl), h = h*b1 + k*b2 + l*b3, where bi
    are the primitive reciprocal vectors.

    Input:
        h = 3-element vector to be aligned
    Output:
        R = 3x3 rotation matrix
    '''

    if h[0] or h[1]:
        #rotation axis
        u = np.array([h[1],-h[0]])/np.sqrt(h[0]**2+h[1]**2)
        #rotation angle
        th = np.arccos(h[2]/np.sqrt(h[0]**2+h[1]**2+h[2]**2))
    else:
        if h[2] > 0:
            #zero deg rotation about -y
            u = np.array([0,-1])
            th = 0
        else:
            #180 deg rotation about -y
            u = np.array([0,-1])
            th = np.pi

    #rotation matrix
    R=np.array([[ np.cos(th) + u[0]**2*(1-np.cos(th)),          u[0]*u[1]*(1-np.cos(th)),  u[1]*np.sin(th)],
                [            u[0]*u[1]*(1-np.cos(th)), np.cos(th)+u[1]**2*(1-np.cos(th)), -u[0]*np.sin(th)],
                [                    -u[1]*np.sin(th),                   u[0]*np.sin(th),       np.cos(th)]])

    return R


def inplane_rotation(alpha):
    '''
    Rotates the given tensor around the z-axis by alpha degrees counterclockwise.

    Input:
        alpha = angle of counterclockwise rotation (in deg) 
    Output:
        R = 3x3 rotation matrix
    '''

    R = axis_angle([0,0,1], alpha)

    return R

def rotate_asymmetry(phi):
    '''
    Rotates the given tensor around the y-axis by phi degrees counterclockwise.
    This corresponds to the definition of clockwise-positive asymmetry angle in
    xz-plane as defined in the documentation.

    Input:
        phi = asymmetry angle (in deg) 
    Output:
        R = 3x3 rotation matrix
    '''

    R = axis_angle([0,1,0], phi)

    return R

