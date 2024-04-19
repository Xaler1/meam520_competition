import sys
import numpy as np
from copy import deepcopy
from math import pi
import threading

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

class KnownPoses(Enum):
    STATIC_OBSERVATION = euler_to_se3(-np.pi, 0, 0, np.array([0.5, -0.15, 0.5]))

STACK_0 = [
        euler_to_se3(-np.pi, 0, 0, np.array([0.55, 0.15, 0.225])),
        euler_to_se3(-np.pi, 0, 0, np.array([0.55, 0.15, 0.275])),
        euler_to_se3(-np.pi, 0, 0, np.array([0.55, 0.15, 0.325])),
        euler_to_se3(-np.pi, 0, 0, np.array([0.55, 0.15, 0.375])),
        euler_to_se3(-np.pi, 0, 0, np.array([0.55, 0.15, 0.425])),
        euler_to_se3(-np.pi, 0, 0, np.array([0.55, 0.15, 0.475])),
    ]




class KnownConfigs(Enum):
    START = np.array([-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353 + np.pi / 2, 0.75344866])

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    main_to_computer = Queue()
    computer_to_executor = Queue()
    executor_to_manipulator = Queue()
    manipulator_to_executor = Queue()
    executor_to_main = Queue()
    observation_queue = Queue()

    computer = Computer(main_to_computer, computer_to_executor)
    executor = Executor(computer_to_executor, executor_to_main, executor_to_manipulator, manipulator_to_executor, observation_queue)
    manipulator = Manipulator(executor_to_manipulator, observation_queue, manipulator_to_executor)

    computer_process = Process(target=computer.run)
    computer_process.start()
    executor_process = Process(target=executor.run)
    executor_process.start()
    manipulator_process = Process(target=manipulator.run)
    manipulator_process.start()

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
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    command = Command("open", CommandTypes.OPEN_GRIPPER, do_async=True)
    task = Task("open", TaskTypes.BYPASS, command=command)
    main_to_computer.put(task)

    task = Task("static", TaskTypes.MOVE_TO, KnownPoses.STATIC_OBSERVATION.value)
    main_to_computer.put(task)
    while True:
        if not executor_to_main.empty():
            done_id = executor_to_main.get()
            if done_id == "static":
                break



    command = Command("observe", CommandTypes.GET_OBSERVED_BLOCKS)
    task = Task("observe", TaskTypes.BYPASS, command=command)
    main_to_computer.put(task)
    while True:
        if not executor_to_main.empty():
            done_id = executor_to_main.get()
            if done_id == "observe":
                sleep(0.1)
                break

    print("Getting observations")
    static_block_poses = executor_to_main.get()
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
        main_to_computer.put(task)
        task = Task(str(i) + "-stack", TaskTypes.PLACE_BLOCK, STACK_0[i])
        main_to_computer.put(task)


    input("Press Enter to kill all")
    print("Terminating")
    computer_process.kill()
    executor_process.kill()
    manipulator_process.kill()


    # Uncomment to get middle camera depth/rgb images
    # mid_depth = detector.get_mid_depth()
    # mid_rgb = detector.get_mid_rgb()

    # Move around...

    # END STUDENT CODE