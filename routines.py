from multiprocessing import Queue
from computer import Task, TaskTypes
from executor import Command, CommandTypes
from lib.utils import euler_to_se3
import numpy as np
import time
from time import sleep

def get_blocks(to_computer: Queue, from_executor: Queue):
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
    return observed_blocks

def get_goal_pose(pose, dynamic_mode=False):
    rot = pose[:3, :3]
    loc = pose[:3, 3]
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

    if dynamic_mode:
        while yaw < np.pi / 4:
            yaw += np.pi / 2


    loc[2] = 0.225
    pose = euler_to_se3(-np.pi, 0, yaw, loc)
    return pose


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
        observed_blocks = get_blocks(to_computer, from_executor)
        for name in observed_blocks:
            block = observed_blocks[name]
            if name not in blocks:
                blocks[name] = []
            blocks[name].append(block)

    static_block_poses = []
    for name in blocks:
        poses = blocks[name]
        if len(poses) == 1:
            continue
        if len(poses) <= 2:
            static_block_poses.append(poses[0])
        else:
            distances = np.zeros(len(poses))
            for i in range(len(poses)):
                for j in range(len(poses)):
                    distances[i] += np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3])
            static_block_poses.append(poses[np.argmin(distances)])

    for i in range(len(static_block_poses)):
        pose = get_goal_pose(static_block_poses[i])
        task = Task(str(i) + "-grab", TaskTypes.GRAB_BLOCK, pose)
        to_computer.put(task)
        task = Task(str(i) + "-stack", TaskTypes.PLACE_BLOCK, stack_positions[i])
        to_computer.put(task)


def stack_dynamic(to_computer: Queue, from_executor: Queue, stack_positions: list):

    observation_pose = euler_to_se3(-np.pi, 0, np.pi/2, np.array([0.0, 0.7, 0.4]))
    rot_axis = np.array([0, 0, 1])
    w = np.pi * 2 * 0.52 / 60

    for i in range(3):
        task = Task("dynamic", TaskTypes.MOVE_TO, observation_pose)
        to_computer.put(task)
        observed_blocks = []
        found = False
        while not found:
            sleep(0.1)
            while len(observed_blocks) < 1:
                observed_blocks = get_blocks(to_computer, from_executor)
            for name in observed_blocks:
                block = observed_blocks[name]
                pose = get_goal_pose(block, dynamic_mode=True)
                print("found block with x", pose[0, 3])
                if abs(pose[0, 3]) < 0.1:
                    found = True
                    break
            observed_blocks = []

        delay = 2
        theta = delay * w

        loc = pose[:3, 3]
        rot = pose[:3, :3]
        offset = np.array([0, 0.99, 0])
        loc = loc - offset

        # rotate by theta around z axis
        transform = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        loc = transform @ loc
        rot = transform @ rot
        loc = loc + offset
        pose[:3, 3] = loc
        pose[:3, :3] = rot
        np.set_printoptions(precision=3, suppress=True)

        task = Task("dynamic", TaskTypes.GRAB_BLOCK, pose, hover_gap=0.05)
        to_computer.put(task)
        task = Task("dynamic", TaskTypes.PLACE_BLOCK, stack_positions[i])
        to_computer.put(task)

    return 3


def shuffle_blocks(to_computer: Queue, from_positions, to_positions):
    to_i = 0
    for i in range(len(from_positions) - 1, -1, -1):
        task = Task("shuffle", TaskTypes.GRAB_BLOCK, from_positions[i], hover_gap=0.1)
        to_computer.put(task)
        task = Task("shuffle", TaskTypes.PLACE_BLOCK, to_positions[to_i], hover_gap=0.1)
        to_computer.put(task)
        to_i += 1
