from multiprocessing import Queue
from computer import Task, TaskTypes
from executor import Command, CommandTypes
from lib.utils import euler_to_se3
import numpy as np
from time import sleep


def stack_static(to_computer: Queue, from_executor: Queue, stack_positions: list):
    static_observation = euler_to_se3(-np.pi, 0, 0, np.array([0.5, -0.15, 0.5]))
    observation_poses = [
        euler_to_se3(-np.pi, -np.pi / 8, 0, np.array([0.45, -0.18, 0.5])),
        euler_to_se3(-np.pi, np.pi / 8, 0, np.array([0.5, -0.18, 0.5])),
        euler_to_se3(-np.pi - np.pi / 8, 0, 0, np.array([0.5, -0.13, 0.5])),
        euler_to_se3(-np.pi + np.pi / 8, 0, 0, np.array([0.5, -0.22, 0.5])),
    ]

    blocks = {}
    for pose in observation_poses:
        task = Task("static", TaskTypes.MOVE_TO, pose)
        to_computer.put(task)
        command = Command("observe", CommandTypes.GET_OBSERVED_BLOCKS)
        task = Task("observe", TaskTypes.BYPASS, command=command)
        to_computer.put(task)
        while True:
            if not from_executor.empty():
                done_id = from_executor.get()
                if done_id == "observe":
                    sleep(0.1)
                    break
        observed_blocks = from_executor.get()
        print("Saw", len(observed_blocks), "blocks")
        for name in observed_blocks:
            block = observed_blocks[name]
            if name not in blocks:
                blocks[name] = []
            blocks[name].append(block)

    static_block_poses = []
    for name in blocks:
        poses = blocks[name]
        if len(poses) <= 2:
            static_block_poses.append(poses[0])
        else:
            distances = np.zeros(len(poses))
            for i in range(len(poses)):
                for j in range(len(poses)):
                    distances[i] += np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3])
            static_block_poses.append(poses[np.argmin(distances)])

    for i in range(len(static_block_poses)):
        rot = static_block_poses[i][:3, :3]
        loc = static_block_poses[i][:3, 3]
        forward = np.array([1, 0, 0])
        up = np.array([0, 0, 1])
        left = np.array([0, 1, 0])
        forward_rotated = rot @ forward.T
        up_rotated = rot @ up.T
        left_rotated = rot @ left.T

        vectors = [forward_rotated, up_rotated, left_rotated]
        yaw = None
        for vector in vectors:
            angle = np.arccos(np.dot(vector, up))
            if np.abs(np.abs(angle) - np.pi / 2) < 0.1:
                yaw = np.arctan2(vector[1], vector[0])
                break

        if yaw is None:
            print("Catastrophic error, failed to determine yaw")
            return

        while yaw > np.pi / 2:
            yaw -= np.pi / 2
        while yaw < -np.pi / 2:
            yaw += np.pi / 2
        if yaw > np.pi / 4:
            yaw -= np.pi / 2
        if yaw < -np.pi / 4:
            yaw += np.pi / 2

        print("----------------Yaw", np.rad2deg(yaw), "-----------------")

        loc[2] = 0.225
        static_block_poses[i] = euler_to_se3(-np.pi, 0, yaw, loc)
        task = Task(str(i) + "-grab", TaskTypes.GRAB_BLOCK, static_block_poses[i])
        to_computer.put(task)
        task = Task(str(i) + "-stack", TaskTypes.PLACE_BLOCK, stack_positions[i])
        to_computer.put(task)
