import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    v_omega = np.vstack((v_in, omega_in))

    J = calcJacobian(q_in)

    nans = np.isnan(v_omega[:,0])
    J = J[~nans]
    v_omega = v_omega[~nans]

    dq, _, _, _ = np.linalg.lstsq(J, v_omega, rcond=None)
    
    return dq


if __name__ == '__main__':
    q_in = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    v_in = np.array([0, 0, -0.1])
    omega_in = np.array([0, 0, 0])
    np.set_printoptions(precision=4, suppress=True)
    print(IK_velocity(q_in, v_in, omega_in))