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
from time import sleep
from routines import stack_static, stack_dynamic, shuffle_blocks
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
STACK_0 = [
    euler_to_se3(-np.pi, 0, 0, np.array([st0["x"], st0["y"], st0["z"] + i * 0.05])) for i in range(8)
]

st1 = config["stack_1"]
STACK_1 = [
    euler_to_se3(-np.pi, 0, 0, np.array([st1["x"], st1["y"], st1["z"] + i * 0.05])) for i in range(8)
]

class KnownConfigs(Enum):
    START = np.array(
        [-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353 + np.pi / 2, 0.75344866])


def cleaner(master_pid, children: list):
    while True:
        if not os.path.exists(f"/proc/{master_pid}"):
            print("Cleaning up children")
            for child in children:
                os.kill(child.pid, signal.SIGKILL)
            break
        sleep(1)


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
    observation_queue = Queue()
    computer_to_main = Queue()

    computer = Computer(main_to_computer, computer_to_executor, computer_to_main)
    executor = Executor(computer_to_executor, executor_to_main, executor_to_manipulator, manipulator_to_executor,
                        observation_queue)
    manipulator = Manipulator(executor_to_manipulator, observation_queue, manipulator_to_executor)

    computer_process = Process(target=computer.run)
    computer_process.start()
    executor_process = Process(target=executor.run)
    executor_process.start()
    manipulator_process = Process(target=manipulator.run)
    manipulator_process.start()

    children = [computer_process, executor_process, manipulator_process]
    master_pid = os.getpid()
    print("Master pid:", master_pid)
    children = [child.pid for child in children]
    os.system(f"python3 cleaner.py {master_pid} {' '.join(map(str, children))} &")

    command = Command("start", CommandTypes.MOVE_TO, KnownConfigs.START.value)
    task = Task("start", TaskTypes.BYPASS, command=command)
    main_to_computer.put(task)
    while executor_to_main.empty():
        sleep(0.1)
    print("Moved to start pose")
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

    #stack_static(main_to_computer, executor_to_main, STACK_0[:4])
    #dynamic_grabbed = stack_dynamic(main_to_computer, executor_to_main, STACK_1[:3], config)
    #stack_static(main_to_computer, executor_to_main, STACK_0[:4], config)
    #shuffle_blocks(main_to_computer, STACK_1[:dynamic_grabbed], STACK_0[4:4+dynamic_grabbed])
    #dynamic_grabbed = stack_dynamic(main_to_computer, executor_to_main, STACK_1[:5], config)
