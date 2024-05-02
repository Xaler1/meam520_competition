from multiprocessing import Queue
from computer import Task, TaskTypes
from executor import Command, CommandTypes
from lib.utils import euler_to_se3
import numpy as np
import time
from time import sleep
import random

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

def get_goal_pose(pose, dynamic_mode=False, blue=False):
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

    while yaw > 0:
        yaw -= np.pi / 2
    while yaw < -np.pi:
        yaw += np.pi / 2
    if yaw < -np.pi / 4:
        yaw += np.pi / 2

    if dynamic_mode:
        if blue:
            while yaw > - np.pi/4:
                yaw -= np.pi/2
        else:
            while yaw < np.pi / 4:
                yaw += np.pi / 2


    loc[2] = 0.225
    pose = euler_to_se3(-np.pi, 0, yaw, loc)
    return pose

def observe_statics(to_computer: Queue, from_executor: Queue, observation_poses):
    blocks = {}
    for i, pose in enumerate(observation_poses):
        task = Task(f"observe_static-{i}", TaskTypes.MOVE_TO, pose)
        to_computer.put(task)
        
        observed_blocks = get_blocks(to_computer, from_executor)
        print("observed blocks:", len(observed_blocks))
        
        for name in observed_blocks:
            block = observed_blocks[name]
            if name not in blocks:
                blocks[name] = []
            blocks[name].append(block)

    static_block_poses = []
    for name in blocks:
        poses = blocks[name]
        if len(poses) <= 1:
            continue
        if len(poses) <= 2:
            static_block_poses.append(poses[0])
        else:
            distances = np.zeros(len(poses))
            for i in range(len(poses)):
                for j in range(len(poses)):
                    distances[i] += np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3])
            static_block_poses.append(poses[np.argmin(distances)])

    filtered_block_poses = []
    print("\n\n")
    for pose in static_block_poses:
        print(pose)
        if get_goal_pose(pose) is None:
            continue
        valid = True
        for other in filtered_block_poses:
            dist = np.linalg.norm(pose[:3, 3] - other[:3, 3])
            if dist < 0.05:
                valid = False
                break
        if valid:
            filtered_block_poses.append(pose)

    return filtered_block_poses


def stack_static(to_computer: Queue, from_executor: Queue, stack_positions: list, config: dict):
    obs = config["static_observations"]
    observation_poses = [euler_to_se3(pose["roll"], pose["pitch"], pose["yaw"], np.array([pose["x"], pose["y"], pose["z"]])) for pose in obs]

    pos_offsets = config["offset_static"]

    static_block_poses = observe_statics(to_computer, from_executor, observation_poses)

    print("\n Final Blocks observed:", len(static_block_poses))

    midpoint = euler_to_se3(-np.pi, 0, -np.pi/2, np.array([0.2, 0, 0.45]))
    for i in range(len(static_block_poses)):
        pose = get_goal_pose(static_block_poses[i])
        pose[0, 3] += pos_offsets["x"]
        pose[1, 3] += pos_offsets["y"]
        pose[2, 3] += pos_offsets["z"]

        task = Task(str(i) + "-grab", TaskTypes.GRAB_BLOCK, pose, hover_gap=0.1)
        to_computer.put(task)

        task = Task(str(i) + "-stack", TaskTypes.PLACE_BLOCK, stack_positions[i], hover_gap=0.11)
        to_computer.put(task)

    return len(static_block_poses)


def stack_dynamic(to_computer: Queue, from_executor: Queue, from_computer: Queue, stack_positions: list, config: dict):

    pos_offsets = config["offset_dynamic"]
    obs = config["dynamic_observation"]

    observation_pose = euler_to_se3(obs["roll"], obs["pitch"], obs["yaw"], np.array([obs["x"], obs["y"], obs["z"]]))
    w = np.pi * 2 * 0.52 / 60

    successful = 0
    last_pose = None
    for i in range(len(stack_positions)):
        task = Task("observe_dynamic", TaskTypes.MOVE_TO, observation_pose)
        to_computer.put(task)
        observed_blocks = []
        found = False
        while not found:
            sleep(0.1)
            while len(observed_blocks) < 1:
                observed_blocks = get_blocks(to_computer, from_executor)
            for name in observed_blocks:
                block = observed_blocks[name]
                pose = get_goal_pose(block, dynamic_mode=True, blue=config["blue"])
                pose[0, 3] += pos_offsets["x"]
                pose[1, 3] += pos_offsets["y"]
                pose[2, 3] += pos_offsets["z"]
                if abs(pose[0, 3]) < 0.05:
                    if last_pose is None:
                        last_pose = pose
                    else:
                        loc = pose[:2, 3]
                        last_loc = last_pose[:2, 3]
                        diff = loc - last_loc
                        print(diff)
                        if -0.01 < diff[0] < 0.01 and -0.02 < diff[1] < 0.02:
                            found = True
                            break
                        last_pose = pose
            observed_blocks = []

        delay = 2
        theta = delay * w

        loc = pose[:3, 3]
        rot = pose[:3, :3]
        offset = np.array([0, config["world_center"], 0])
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

        while not from_computer.empty():
            from_computer.get()
        task = Task("dynamic", TaskTypes.GRAB_BLOCK, pose, hover_gap=0.05)
        to_computer.put(task)

        # Verify that the arm was able to grab the dynamic block
        while from_computer.empty():
            sleep(0.1)
        result = from_computer.get()
        if not result:
            continue

        task = Task("dynamic", TaskTypes.PLACE_BLOCK, stack_positions[successful], hover_gap=0.15)
        to_computer.put(task)
        successful += 1

    return successful


def shuffle_blocks(to_computer: Queue, from_positions, to_positions):
    to_i = 0
    for i in range(len(from_positions) - 1, -1, -1):
        midpoint = to_positions[max(to_i - 1, 0)].copy()
        midpoint[2, 3] = 0.6
        midpoint[0, 3] = 0.45
        midpoint[1, 3] = from_positions[i][1, 3]
        task = Task(f"shuffle-{to_i}", TaskTypes.MOVE_TO, midpoint)
        to_computer.put(task)
        task = Task(f"{i}-shuffle-1", TaskTypes.GRAB_BLOCK, from_positions[i], hover_gap=0.1)
        to_computer.put(task)

        midpoint = to_positions[to_i].copy()
        midpoint[2, 3] = 0.6
        midpoint[0, 3] = 0.45
        midpoint[1, 3] = from_positions[i][1, 3]
        task = Task(f"shuffle-{10*(to_i+1)}", TaskTypes.MOVE_TO, midpoint)
        to_computer.put(task)
        task = Task(f"{i}-shuffle-2", TaskTypes.PLACE_BLOCK, to_positions[to_i], hover_gap=0.05)
        to_computer.put(task)
        to_i += 1

def calibration_print(text):
    print("--calibration--", text)

def calibration(to_computer: Queue, from_executor, config: dict):
    calibration_print("Starting calibration")
    obs = config["static_observations"]
    observation_poses = [euler_to_se3(pose["roll"], pose["pitch"], pose["yaw"], np.array([pose["x"], pose["y"], pose["z"]])) for pose in obs]

    pos_offsets = config["offset_static"]

    calibration_print("Static Observation positions")
    counter = 0
    for i, pose in enumerate(observation_poses):
        calibration_print(f"Observation {i}")
        satisfied = False
        while not satisfied:
            calibration_print(f"Current x,y,z: {pose[0, 3]}, {pose[1, 3]}, {pose[2, 3]}")
            task = Task(f"observe_static-{counter}", TaskTypes.MOVE_TO, pose)
            to_computer.put(task)
            calibration_print("Input new x, y, z or 'y' if satisfied")
            response = input()
            if response == 'y':
                satisfied = True
            else:
                x, y, z = map(float, response.split())
                pose[0, 3] = x
                pose[1, 3] = y
                pose[2, 3] = z

            obs[i]["x"] = pose[0, 3]
            obs[i]["y"] = pose[1, 3]
            obs[i]["z"] = pose[2, 3]
            counter += 1

    calibration_print("Completed Static Observation positions:")
    print(obs)
    print("\n\n")

    calibration_print("Static block horizontal offsets")

    static_block_poses = observe_statics(to_computer, from_executor, observation_poses)
    sample_pose = static_block_poses[0]

    satisfied = False
    while not satisfied:
        calibration_print(f"Current x,y: {pos_offsets['x']}, {pos_offsets['y']}")
        pose = get_goal_pose(sample_pose)
        pose[0, 3] += pos_offsets["x"]
        pose[1, 3] += pos_offsets["y"]
        pose[2, 3] += pos_offsets["z"]
        hover_pose = pose.copy()
        hover_pose[2, 3] += 0.05
        task = Task(f"calibration-{counter}", TaskTypes.MOVE_TO, hover_pose)
        to_computer.put(task)
        calibration_print("Input new x, y or 'y' if satisfied")
        response = input()
        if response == 'y':
            satisfied = True
        else:
            x, y = map(float, response.split())
            pos_offsets["x"] = x
            pos_offsets["y"] = y
        counter += 1

    calibration_print("Static block vertical offset")
    satisfied = False
    while not satisfied:
        calibration_print(f"Current z: {pos_offsets['z']}")
        pose = get_goal_pose(sample_pose)
        pose[0, 3] += pos_offsets["x"]
        pose[1, 3] += pos_offsets["y"]
        pose[2, 3] += pos_offsets["z"]
        task = Task(f"calibration-{counter}", TaskTypes.MOVE_TO, pose)
        to_computer.put(task)
        calibration_print("Input new z or 'y' if satisfied")
        response = input()
        if response == 'y':
            satisfied = True
        else:
            z = float(response)
            pos_offsets["z"] = z
        counter += 1

    calibration_print("Completed Static block offsets:")
    print(pos_offsets)



