import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian
from scipy.linalg import null_space

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))

    dq = IK_velocity(q_in, v_in, omega_in)

    J = calcJacobian(q_in)

    v_omega = np.vstack((v_in, omega_in))
    nans = np.isnan(v_omega[:,0])
    J = J[~nans]
    null = np.eye(7) - np.linalg.pinv(J) @ J
    null = null @ b

    return np.reshape(dq + null, (7,))
