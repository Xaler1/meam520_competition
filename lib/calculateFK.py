import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout


        self.offsets = np.array([
            [0, 0, 0.141, 1],
            [0, 0, 0, 1],
            [0, 0, 0.195, 1],
            [0, 0, 0, 1],
            [0, 0, 0.125, 1],
            [0, 0, -0.015, 1],
            [0, 0, 0.051, 1],
            [0, 0, 0, 1]
        ])

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """


        # Your Lab 1 code starts here
        jointPositions = np.zeros((8,3))

        transforms = self.compute_Ai(q)

        jointPositions[0] = [0, 0, 0.141]
        for i in range(1, 8, 1):
            jointPositions[i] = (transforms[i] @ self.offsets[i])[:3]


        # Your code ends here

        return jointPositions, transforms[-1]

    # feel free to define additional helper methods to modularize your solution for lab 1


    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        transforms = self.compute_Ai(q)
        positions, _ = self.forward(q)
        axis_of_rotation = np.zeros((3, 7))
        up_vec = np.array([0, 0, 1, 0])
        for i in range(7):
            local_up_vec = up_vec + self.offsets[i]
            world_up_vec = (transforms[i] @ local_up_vec)[:3]
            axis_of_rotation[:, i] = world_up_vec - positions[i]
            axis_of_rotation[:, i] /= np.linalg.norm(axis_of_rotation[:, i])

        return axis_of_rotation

    def get_transitions(self, q):
        T01 = np.array([
            [np.cos(q[0]), 0, -np.sin(q[0]), 0],
            [np.sin(q[0]), 0, np.cos(q[0]), 0],
            [0, -1, 0, 0.333],
            [0, 0, 0, 1]
        ])

        T12 = np.array([
            [np.cos(q[1]), 0, np.sin(q[1]), 0],
            [np.sin(q[1]), 0, -np.cos(q[1]), 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        T23 = np.array([
            [np.cos(q[2]), 0, np.sin(q[2]), 0.0825 * np.cos(q[2])],
            [np.sin(q[2]), 0, -np.cos(q[2]), 0.0825 * np.sin(q[2])],
            [0, 1, 0, 0.316],
            [0, 0, 0, 1]
        ])

        T34 = np.array([
            [np.cos(q[3]), 0, -np.sin(q[3]), -0.0825 * np.cos(q[3])],
            [np.sin(q[3]), 0, np.cos(q[3]), -0.0825 * np.sin(q[3])],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        T45 = np.array([
            [np.cos(q[4]), 0, np.sin(q[4]), 0],
            [np.sin(q[4]), 0, -np.cos(q[4]), 0],
            [0, 1, 0, 0.384],
            [0, 0, 0, 1]
        ])

        T56 = np.array([
            [np.cos(q[5]), 0, np.sin(q[5]), 0.088 * np.cos(q[5])],
            [np.sin(q[5]), 0, -np.cos(q[5]), 0.088 * np.sin(q[5])],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        T6e = np.array([
            [np.cos(q[6] - pi / 4), -np.sin(q[6] - pi / 4), 0, 0],
            [np.sin(q[6] - pi / 4), np.cos(q[6] - pi / 4), 0, 0],
            [0, 0, 1, 0.21],
            [0, 0, 0, 1]
        ])

        transitions = [T01, T12, T23, T34, T45, T56, T6e]
        return transitions


    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """

        transitions = self.get_transitions(q)
        transforms = []
        T0e = np.eye(4)
        transforms.append(np.copy(T0e))
        for i in range(7):
            T0e = T0e @ transitions[i]
            transforms.append(np.copy(T0e))

        return transforms

