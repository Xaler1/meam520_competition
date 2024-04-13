import numpy as np

import rospy
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from lib.calculateFK import FK
from lib.IK_velocity_null import IK_velocity_null
from lib.IK_velocity import IK_velocity
from lib.IK_position_null import IK
from lib.calcJacobian import calcJacobian
from core.utils import time_in_seconds
from lib.calcAngDiff import calcAngDiff
from lib.utils import euler_to_se3

from threading import Thread
from trajectories import Trajectory, Waypoints, Spline, HermitSpline, BetterSpline
from time import sleep


class Controller:
    active = False
    last_iteration_time = None
    start_time = 0
    trajectory: Trajectory = None
    prev_v = None

    def __init__(self):
        rospy.init_node("team_script")
        callback = lambda state: self.callback(state)
        self.arm = ArmController(on_state_callback=callback)
        self.arm.set_arm_speed(0.5)
        self.detector = ObjectDetector()
        self.fk = FK()
        self.ik = IK()


    def start_position(self):
        start_position = np.array(
            [-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353 + np.pi / 2, 0.75344866])
        self.arm.safe_move_to_position(start_position)  # on your mark!
        self.arm.close_gripper()

    def callback(self, state):
        if self.active:
            q = state['position']
            _, T0e = self.fk.forward(q)
            t = time_in_seconds() - self.start_time

            v, omega, xdes, Rdes, done = self.trajectory(T0e, t)
            if done:
                self.active = False
                return

            R = (T0e[:3, :3])
            x = (T0e[0:3, 3])
            kp = 1
            curr_x = np.copy(x.flatten())
            if v is None:
                v = np.zeros(3)
                kp = 4
            if xdes is not None:
                # First Order Integrator, Proportional Control with Feed Forward
                v = v + kp * (xdes - curr_x)
            if Rdes is not None:
                # Rotation
                kr = 1
                omega = omega + kr * calcAngDiff(Rdes, R).flatten()

            dq = IK_velocity(q, v, omega).flatten()

            if self.last_iteration_time == None:
                self.last_iteration_time = time_in_seconds()

            self.dt = time_in_seconds() - self.last_iteration_time
            self.last_iteration_time = time_in_seconds()

            new_q = q + self.dt * dq

            self.arm.safe_set_joint_positions_velocities(new_q, dq)

    def open_gripper_async(self):
        t = Thread(target=self.arm.open_gripper)
        t.start()
        return t

    def close_gripper_async(self):
        t = Thread(target=self.arm.close_gripper)
        t.start()
        return t

    def start(self):
        H_ee_camera = self.detector.get_H_ee_camera()
        print("Camera pose:\n", H_ee_camera)

        q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])
        t1 = self.open_gripper_async()
        self.arm.safe_move_to_position(q)
        t1.join()

        detections = self.detector.get_detections()
        for (name, pose) in detections:
            print(name, '\n', pose)

        transforms = self.fk.compute_Ai(q)
        H0c = transforms[-1] @ H_ee_camera
        block_locations = []
        block_rotations = []
        for (name, pose) in detections:
            world_pose = H0c @ pose
            block_locations.append(world_pose[:3, 3])
            block_rotations.append(world_pose[:3, :3])
            print(block_locations[-1])

        for i in range(len(block_locations)):
            block_locations[i][2] = 0.225

        target_id = int(input("Select block to pick up: "))
        rot = block_rotations[target_id]
        rotation_z = np.arctan2(rot[1, 0], rot[0, 0]) % (np.pi / 2)
        if rotation_z > np.pi / 4:
            rotation_z -= np.pi / 2
        elif rotation_z < -np.pi / 4:
            rotation_z += np.pi / 2

        target_loc = block_locations[target_id] + np.array([0, 0, 0.1])
        target1 = euler_to_se3(-np.pi, 0, rotation_z, target_loc)
        target2 = euler_to_se3(-np.pi, 0, rotation_z, block_locations[target_id])

        self.trajectory = HermitSpline([transforms[-1], target1, target2])
        self.start_time = time_in_seconds()
        self.active = True
        # while self.active:
        #     sleep(0.1)

        input("Press enter to continue...")


