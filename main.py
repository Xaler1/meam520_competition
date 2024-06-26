import sys
import numpy as np
from copy import deepcopy
from math import pi
import threading
import signal

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
from lib.calculateFK import FK
from lib.IK_position_null import IK
from lib.utils import euler_to_se3
from enum import Enum
from multiprocessing import Process, Queue
from computer import Computer, Task, TaskTypes
from executor import Executor, Command, CommandTypes
from manipulator import Manipulator, Action, ActionType
from observer import Observer
from time import sleep
from routines import stack_static, stack_dynamic, shuffle_blocks, calibration
import json
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument("config", type=str, help="Path to config file", default="red-sim.json")
args = args.parse_args()
with open(args.config, "r") as f:
    config = json.load(f)

class KnownPoses(Enum):
    STATIC_OBSERVATION = euler_to_se3(-np.pi, 0, 0, np.array([0.5, -0.15, 0.5]))


st0 = config["stack_0"]
STATIC_STACK = [
    euler_to_se3(-np.pi, 0, 0, np.array([st0["x"], st0["y"], st0["z"] + i * 0.05])) for i in range(7)
]

for i in range(5):
    STATIC_STACK.append(euler_to_se3(-np.pi, -np.pi / 2, 0, np.array([st0["x"], st0["y"], st0["z"] + 0.35 + i * 0.05])))

st1 = config["stack_1"]
DYNAMIC_STACK = [
    euler_to_se3(-np.pi, 0, 0, np.array([st1["x"], st1["y"], st1["z"] + i * 0.05])) for i in range(8)
]

class KnownConfigs(Enum):
    START = np.array(
        [-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353 + np.pi / 2, 0.75344866])


if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    main_to_computer = Queue()
    computer_to_executor = Queue()
    executor_to_manipulator = Queue()
    manipulator_to_executor = Queue()
    executor_to_main = Queue()
    static_observations = Queue()
    dynamic_observations = Queue()
    dynamic_locations = Queue()
    computer_to_main = Queue()

    computer = Computer(main_to_computer, computer_to_executor, computer_to_main)
    executor = Executor(computer_to_executor, executor_to_main, executor_to_manipulator, manipulator_to_executor,
                        static_observations)
    manipulator = Manipulator(executor_to_manipulator,
                              static_observations,
                              dynamic_observations,
                              manipulator_to_executor)
    observer = Observer(dynamic_locations, dynamic_observations, team)


    computer_process = Process(target=computer.run)
    computer_process.start()
    executor_process = Process(target=executor.run)
    executor_process.start()
    manipulator_process = Process(target=manipulator.run)
    manipulator_process.start()
    observer_process = Process(target=observer.run)
    observer_process.start()

    children = [computer_process, executor_process, manipulator_process, observer_process]
    master_pid = os.getpid()
    print("[INFO] Master pid:", master_pid)
    children = [child.pid for child in children]
    os.system(f"python3 cleaner.py {master_pid} {' '.join(map(str, children))} &")

    command = Command("start", CommandTypes.MOVE_TO, KnownConfigs.START.value)
    task = Task("start", TaskTypes.BYPASS, command=command)
    main_to_computer.put(task)
    while executor_to_main.empty():
        sleep(0.1)
    print("[INFO] Moved to start pose")
    executor_to_main.get()

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")  # get set!
    print("Go!\n")  # go!

    command = Command("open", CommandTypes.OPEN_GRIPPER, do_async=True)
    task = Task("open", TaskTypes.BYPASS, command=command)
    main_to_computer.put(task)

    dyn_h = stack_dynamic(main_to_computer, executor_to_main, DYNAMIC_STACK[:3], config)
    stat_h, dyn_h = stack_static(main_to_computer, executor_to_main, STATIC_STACK[:4], config, dynamic_locations, DYNAMIC_STACK[dyn_h:], dyn_h)
    shuffle_blocks(main_to_computer, DYNAMIC_STACK[:dyn_h], STATIC_STACK[stat_h:stat_h + dyn_h])
    dynamic_grabbed = stack_dynamic(main_to_computer, executor_to_main, STATIC_STACK[stat_h+dyn_h:], config)
    # calibration(main_to_computer, executor_to_main, config)
