from __future__ import division, print_function
from numpy import array, finfo, cos, sin, arctan2, isinf
from .elastic_tensors import rotate_elastic_matrix
from .rotation_matrix import inplane_rotation


def isotropic_plate(R1, R2, nu, thickness):
    '''
    Creates a function for computing the Jacobian of the displacement field
    for an isotropic plate.

    Parameters
    ----------

    R1 : float, 'inf' or None
        The meridional bending radius due to torques (in the same units as R1
        and thickness). Use 'inf' if the direction is not curved. None sets the
        corresponding torque to zero and R1 is calculated as an anticlastic
        reaction to the other torque.

    R2 : float, 'inf' or None
        The sagittal bending radius due to torques (in the same units as R1
        and thickness). Use 'inf' if the direction is not curved. None sets the
        corresponding torque to zero and R1 is calculated as an anticlastic
        reaction to the other torque.

    nu : float
        Poisson's ratio of the material.

    thickness : float
        The thickness of the crystal (in the same units as R1 and R2)


    Returns
    -------

    jacobian : function
        Returns the partial derivatives of the displacement vector u as a
        function of coordinates (x,z). The length scale is determined by the
        units of R1, R2 and thickness.

    R1 : float
        The meridional bending radius due to torques (in the units defined by
        the input). Calculated by the function if due to anticlastic bending.

    R2 : float
        The sagittal bending radius due to torques (in the units defined by the
        input). Calculated by the function if due to anticlastic bending.
    '''

    if R1 is None:
        if R2 is None or isinf(float(R2)):
            invR1 = 0
            invR2 = 0

            R1 = 'inf'
            R2 = 'inf'
        else:
            R1 = -R2/nu  # anticlastic bending
            invR1 = 1.0/R1

    elif R2 is None:
        if isinf(float(R1)):
            invR1 = 0
            invR2 = 0

            R1 = 'inf'
            R2 = 'inf'
        else:
            R2 = -R1/nu  # anticlastic bending
            invR2 = 1.0/R2

    else:
        if isinf(float(R1)):
            invR1 = 0
            R1 = 'inf'
        else:
            invR1 = 1/R1

        if isinf(float(R2)):
            invR2 = 0
            R2 = 'inf'
        else:
            invR2 = 1/R2

    def jacobian(x, z):
        ux_x = -(z+0.5*thickness)*invR1
        ux_z = -x*invR1

        uz_x = x*invR1
        uz_z = nu/(1-nu)*(invR1+invR2)*(z+0.5*thickness)

        return [[ux_x, ux_z], [uz_x, uz_z]]

    return jacobian, R1, R2, [nu/(1-nu)*(invR1+invR2), 0, invR1, 0, invR2]


def anisotropic_plate_fixed_torques(R1, R2, S, thickness):
    '''
    Creates a function for computing the Jacobian of the displacement field
    for an anisotropic plate with fixed torques.

    Parameters
    ----------

    R1 : float, 'inf' or None
        The meridional bending radius due to torques (in the same units as R1
        and thickness). Use 'inf' if the direction is not curved. None sets the
        corresponding torque to zero and R1 is calculated as an anticlastic
        reaction to the other torque.

    R2 : float, 'inf' or None
        The sagittal bending radius due to torques (in the same units as R1
        and thickness). Use 'inf' if the direction is not curved. None sets the
        corresponding torque to zero and R1 is calculated as an anticlastic
        reaction to the other torque.

    S : 6x6 Numpy array of floats
        The compliance matrix in the Voigt notation. Units are not important.

    thickness : float
        The thickness of the crystal (in the same units as R1 and R2)


    Returns
    -------

    jacobian : function
        Returns the partial derivatives of the displacement vector u as a
        function of coordinates (x,z). The length scale is determined by the
        units of R1, R2 and thickness.

    R1 : float
        The meridional bending radius due to torques (in the units defined by
        the input). Calculated by the function if due to anticlastic bending.

    R2 : float
        The sagittal bending radius due to torques (in the units defined by the
        input). Calculated by the function if due to anticlastic bending.
    '''

    S = array(S)

    if R1 is None:
        m1 = 0  # no torque about y-axis

        if R2 is None or isinf(float(R2)):
            # If both bending radii are set to None, or R2 = inf when m1 = 0,
            # it implies that there no torques acting on the wafer
            m2 = 0
            R1 = 'inf'
            R2 = 'inf'
        else:
            m2 = -1.0/(S[1, 1] * R2)
            R1 = -1.0/(S[0, 1] * m2)  # anticlastic reaction

    elif R2 is None:
        m2 = 0  # no torque about x-axis
        if isinf(float(R1)):
            m1 = 0
            R1 = 'inf'
            R2 = 'inf'
        else:
            m1 = -1.0/(S[0, 0] * R1)
            R2 = -1.0/(S[1, 0] * m1)  # anticlastic reaction
    else:
        if isinf(float(R1)):
            R1 = 'inf'  # for output
            invR1 = 0
        else:
            invR1 = 1/R1

        if isinf(float(R2)):
            R2 = 'inf'  # for output
            invR2 = 0
        else:
            invR2 = 1/R2

        m_divider = S[1, 1]*S[0, 0] - S[0, 1]*S[0, 1]
        m1 = (S[0, 1]*invR2 - S[1, 1]*invR1)/m_divider
        m2 = (S[0, 1]*invR1 - S[0, 0]*invR2)/m_divider

    # Coefficients for the Jacobian
    coef1 = S[0, 0]*m1 + S[0, 1]*m2
    coef2 = S[4, 0]*m1 + S[4, 1]*m2
    coef3 = S[2, 0]*m1 + S[2, 1]*m2

    def jacobian(x, z):
        ux_x = coef1*(z+0.5*thickness)
        ux_z = coef1*x + coef2*(z+0.5*thickness)

        uz_x = -coef1*x
        uz_z = coef3*(z+0.5*thickness)

        return [[ux_x, ux_z], [uz_x, uz_z]]

    return jacobian, R1, R2, [coef3, coef2, invR1, coef1, invR2]  # TODO coef1


def anisotropic_plate_fixed_shape(R1, R2, S, thickness):
    '''
    Creates a function for computing the Jacobian of the displacement field
    for an anisotropic plate with fixed shape.

    Parameters
    ----------

    R1 : float or 'inf'
        The meridional bending radius (in the same units as R2 and thickness).
        Use 'inf' if the direction is not curved.

    R2 : float or 'inf'
        The sagittal bending radius (in the same units as R1 and thickness)
        Use 'inf' if the direction is not curved.

    S : 6x6 Numpy array of floats
        The compliance matrix in the Voigt notation. Units are not important.

    thickness : float
        The thickness of the crystal (in the same units as R1 and R2)


    Returns
    -------

    jacobian : function
        Returns the partial derivatives of the displacement vector u as a
        function of coordinates (x,z). The length scale is determined by the
        units of R1, R2 and thickness.

    R1 : float
        The meridional bending radius due to torques (in the units defined by
        the input). Returned for the output compatibility.

    R2 : float
        The sagittal bending radius due to torques (in the units defined by the
        input). Returned for the output compatibility.
    '''

    # Convert the bending radii to their inverses:
    if isinf(float(R1)):
        invR1 = 0
        R1 = 'inf'
    else:
        invR1 = 1.0/R1

    if isinf(float(R2)):
        invR2 = 0
        R2 = 'inf'
    else:
        invR2 = 1.0/R2

    # Calculate the rotation angle alpha
    S = array(S)
    meps = finfo(type(S[0][0])).eps  # machine epsilon

    if abs(S[5, 0]) < meps and abs(S[5, 1]) < meps and\
            abs(S[1, 1] - S[0, 0]) < meps and\
            abs(S[0, 0] + S[1, 1] - 2*S[0, 1] - S[5, 5]) < meps:
        alpha = 0
    else:
        Aa = S[5, 5]*(S[0, 0] + S[1, 1] + 2*S[0, 1]) - (S[5, 0] + S[5, 1])**2
        Ba = 2*(S[5, 1]*(S[0, 1] + S[0, 0]) - S[5, 0]*(S[0, 1] + S[1, 1]))
        Ca = S[5, 5]*(S[1, 1] - S[0, 0]) + S[5, 0]**2 - S[5, 1]**2
        Da = 2*(S[5, 1]*(S[0, 1] - S[0, 0]) + S[5, 0]*(S[0, 1] - S[1, 1]))

        alpha = 0.5*arctan2(Da*(invR2+invR1) - Ba*(invR2-invR1),
                            Aa*(invR2-invR1) - Ca*(invR2+invR1))

    # rotate S by alpha
    Sp = rotate_elastic_matrix(S, 'S', inplane_rotation(alpha))

    # Compute torques
    m_divider = 2*(Sp[0, 0]*Sp[1, 1] - Sp[0, 1]*Sp[0, 1])

    mx = ((Sp[0, 1] - Sp[1, 1])*(invR2 + invR1) +
          (Sp[0, 1] + Sp[1, 1])*(invR2 - invR1)*cos(2*alpha))
    mx = mx / m_divider

    my = ((Sp[0, 1] - Sp[0, 0])*(invR2 + invR1) -
          (Sp[0, 1] + Sp[0, 0])*(invR2 - invR1)*cos(2*alpha))
    my = my / m_divider

    # Coefficients for the Jacobian
    coef1 = Sp[2, 0]*mx + Sp[2, 1]*my
    coef2 = ((Sp[4, 0]*mx + Sp[4, 1]*my)*cos(alpha) -
             (Sp[3, 0]*mx + Sp[3, 1]*my)*sin(alpha))
    coef3 = ((Sp[4, 0]*mx + Sp[4, 1]*my)*sin(alpha) +
             (Sp[3, 0]*mx + Sp[3, 1]*my)*cos(alpha))

    def jacobian(x, z):
        ux_x = -invR1*(z+0.5*thickness)
        ux_z = -invR1*x + coef2*(z+0.5*thickness)

        uz_x = invR1*x
        uz_z = coef1*(z+0.5*thickness)

        return [[ux_x, ux_z], [uz_x, uz_z]]

    return jacobian, R1, R2, [coef1, coef2, invR1, coef3, invR2]
