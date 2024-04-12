import numpy as np

import rospy
from core.interfaces import ArmController
from lib.calculateFK import FK
from lib.IK_velocity_null import IK_velocity_null
from lib.IK_velocity import IK_velocity
from lib.IK_position_null import IK
from lib.calcJacobian import calcJacobian
from core.utils import time_in_seconds
from lib.calcAngDiff import calcAngDiff

def rotvec_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3)

    # Normalize to get rotation axis.
    k = rotvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R

class Follower:

    active = False
    fk = FK()
    ik = IK()
    target = None
    complete = False
    last_iteration_time = None
    start = None
    start_time = None

    def reach_target(self, state):
        if self.active and not self.complete:
            try:
                t = time_in_seconds() - self.start_time

                # get desired trajectory position and velocity
                xdes = np.array([0.307 + 0.2 * np.sin(t), 0, 0.487])
                x_vel = 0.2 * np.cos(t)
                vdes = np.array([x_vel, 0, 0])

                ang = -np.pi + (np.pi / 4.0) * np.sin(1.0 * t)
                r = ang * np.array([1.0, 0.0, 0.0])
                Rdes = rotvec_to_matrix(r)

                ang_v = (np.pi / 4.0) * 1.0 * np.cos(1.0 * t)
                ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])

                # get current end effector position
                q = state['position']

                joints, T0e = self.fk.forward(q)

                R = (T0e[:3, :3])
                x = (T0e[0:3, 3])
                curr_x = np.copy(x.flatten())

                # First Order Integrator, Proportional Control with Feed Forward
                kp = 5
                v = vdes + kp * (xdes - curr_x)

                # Rotation
                kr = 5
                omega = ang_vdes + kr * calcAngDiff(Rdes, R).flatten()

                dq = IK_velocity(q, v, omega).flatten()

                # Get the correct timing to update with the robot
                if self.last_iteration_time == None:
                    self.last_iteration_time = time_in_seconds()

                self.dt = time_in_seconds() - self.last_iteration_time
                self.last_iteration_time = time_in_seconds()

                new_q = q + self.dt * dq

                self.arm.safe_set_joint_positions_velocities(new_q, dq)

            except rospy.exceptions.ROSException:
                pass

        return

    def set_arm(self, arm):
        self.arm = arm

    def set_target(self, start, target):
        self.last_iteration_time = time_in_seconds()
        self.start_time = time_in_seconds()
        self.start = start
        self.target = target
        self.complete = False
        self.active = True