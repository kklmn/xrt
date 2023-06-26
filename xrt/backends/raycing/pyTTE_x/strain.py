# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:12:28 2022

@author: chernir
"""

import numpy as np

xtal_data = {'system' : 'cubic', 'C11' : 1.6578, 'C12' : 0.6394, 'C44' : 0.7962}

C11, C12, C13, C14, C15, C16  = 0, 0, 0, 0, 0, 0
C22, C23, C24, C25, C26 = 0, 0, 0, 0, 0
C33, C34, C35, C36 = 0, 0, 0, 0
C44, C45, C46 = 0, 0, 0
C55, C56 = 0, 0
C66 = 0
       
if xtal_data['system'] == 'cubic':
    C11, C12, C44 = xtal_data['C11'], xtal_data['C12'], xtal_data['C44']
    C22, C13, C23, C33, C55, C66 = C11, C12, C12, C11, C44, C44


C_matrix = np.array([[C11, C12, C13, C14, C15, C16],
                     [C12, C22, C23, C24, C25, C26],
                     [C13, C23, C33, C34, C35, C36],
                     [C14, C24, C34, C44, C45, C46],
                     [C15, C25, C35, C45, C55, C56],
                     [C16, C26, C36, C46, C56, C66]])

S_matrix = np.linalg.inv(C_matrix)

R1, R2 = 5e3, np.inf

if np.isinf(float(R1)):
    invR1 = 0
    R1 = 'inf'
else:
    invR1 = 1.0/R1

if np.isinf(float(R2)):
    invR2 = 0
    R2 = 'inf'
else:
    invR2 = 1.0/R2

# Calculate the rotation angle alpha
S = np.array(S_matrix)
#meps = finfo(type(S[0][0])).eps #machine epsilon
meps = 1e-16
    
if abs(S[5,0]) < meps and abs(S[5,1]) < meps and abs(S[1,1] - S[0,0]) < meps\
and abs(S[0,0] + S[1,1] - 2*S[0,1] - S[5,5]) < meps:
    alpha = 0  
else:
    Aa = S[5,5]*(S[0,0] + S[1,1] + 2*S[0,1]) - (S[5,0] + S[5,1])**2
    Ba = 2*(S[5,1]*(S[0,1] + S[0,0]) - S[5,0]*(S[0,1] + S[1,1])) 
    Ca = S[5,5]*(S[1,1]-S[0,0]) + S[5,0]**2 - S[5,1]**2
    Da = 2*(S[5,1]*(S[0,1] - S[0,0]) + S[5,0]*(S[0,1] - S[1,1]))

    alpha = 0.5*np.arctan2(Da*(invR2+invR1) - Ba*(invR2-invR1), 
                           Aa*(invR2-invR1) - Ca*(invR2+invR1))

def rotate_elastic_tensor(tensor, rotation_matrix):
    '''
    Performs the rotation described by rotation_matrix to the given elastic tensor.

    Input:
        tensor = 3x3x3x3 elastic tensor
        rotation_matrix = 3x3 rotation matrix
    Output:
        tensor_rot = rotated tensor
    '''
    tensor_rot = tensor
    for i in range(4):
        tensor_rot = np.tensordot(rotation_matrix,tensor_rot,axes=((1,),(i,)))

    return tensor_rot

def rotate_elastic_matrix(matrix, mtype, rotation_matrix):
    '''
    Performs the rotation described by rotation_matrix to the given elastic matrix.

    Input:
        matrix = 6x6 elastic matrix
        mtype = 'C' if the matrix to be rotated is the stiffness matrix 
                or 'S' if the compliance matrix
        rotation_matrix = 3x3 rotation matrix
    Output:
        matrix_rot = rotated matrix
    '''
    tensor = matrix2tensor(matrix,mtype)
    tensor_rot = rotate_elastic_tensor(tensor, rotation_matrix)
    matrix_rot = tensor2matrix(tensor_rot,mtype)

    return matrix_rot

# rotate S by alpha
Sp = rotate_elastic_matrix(S, 'S', inplane_rotation(alpha))

# Compute torques
m_divider = 2*(Sp[0, 0]*Sp[1, 1] - Sp[0, 1]*Sp[0, 1])

mx = ((Sp[0, 1] - Sp[1, 1])*(invR2 + invR1)
     +(Sp[0, 1] + Sp[1, 1])*(invR2 - invR1)*np.cos(2*alpha))
mx = mx / m_divider

my = ((Sp[0, 1] - Sp[0, 0])*(invR2 + invR1)
     -(Sp[0, 1] + Sp[0, 0])*(invR2 - invR1)*np.cos(2*alpha))
my = my / m_divider  

# Coefficients for the Jacobian
coef1 = Sp[2, 0]*mx + Sp[2, 1]*my
coef2 = ((Sp[4, 0]*mx + Sp[4, 1]*my)*np.cos(alpha) 
        -(Sp[3, 0]*mx + Sp[3, 1]*my)*np.sin(alpha))

def jacobian(x,z):
#        print("shape")
    ux_x = -invR1*(z+0.5*thickness)
    ux_z = -invR1*x + coef2*(z+0.5*thickness)

    uz_x = invR1*x
    uz_z = coef1*(z+0.5*thickness)

    return [[ux_x,ux_z],[uz_x,uz_z]]

def strain_term(z):
    x = -z*cot_alpha0
    u_jac = jacobian(x,z)
    duh_dsh = h*(sin_phi*cos_alphah*u_jac[0][0]
                +sin_phi*sin_alphah*u_jac[0][1]
                +cos_phi*cos_alphah*u_jac[1][0] 
                +cos_phi*sin_alphah*u_jac[1][1]
                )
    return gammah_step*duh_dsh
    