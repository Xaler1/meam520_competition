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
        euler_to_se3(-np.pi, np.pi / 8, 0, np.array([0.55, -0.18, 0.5])),
        euler_to_se3(-np.pi - np.pi/8, 0, 0, np.array([0.5, -0.13, 0.5])),
        euler_to_se3(-np.pi + np.pi/8, 0, 0, np.array([0.5, -0.22, 0.5])),
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
        print(observed_blocks)
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


    print(static_block_poses)

    for i in range(len(static_block_poses)):
        print(static_block_poses[i].shape)
        rot = static_block_poses[i][:3, :3]
        loc = static_block_poses[i][:3, 3]
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        # find closest 90 degree rotation
        while yaw < -np.pi / 2:
            yaw += np.pi / 2
        while yaw > np.pi / 2:
            yaw -= np.pi / 2

        print("----------------------Rotation Z", yaw)

        loc[2] = 0.225
        static_block_poses[i] = euler_to_se3(-np.pi, 0, yaw, loc)
        task = Task(str(i) + "-grab", TaskTypes.GRAB_BLOCK, static_block_poses[i])
        to_computer.put(task)
        task = Task(str(i) + "-stack", TaskTypes.PLACE_BLOCK, stack_positions[i])
        to_computer.put(task)