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
from controller import Controller
from enum import Enum
from multiprocessing import Process, Queue
from computer import Computer, Task, TaskTypes
from executer import Executer, Command, CommandTypes
from time import sleep

class Positions(Enum):
    STATIC_OBSERVATION = euler_to_se3(-np.pi, 0, 0, np.array([0.5, -0.15, 0.5]))

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    to_computer = Queue()
    computer_to_executer = Queue()
    from_executer = Queue()
    computer = Computer(to_computer, computer_to_executer)
    executer = Executer(computer_to_executer, from_executer)
    computer_process = Process(target=computer.run)
    computer_process.start()
    executer_process = Process(target=executer.run)
    executer_process.start()

    command = Command(CommandTypes.MOVE_TO, np.array([-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353 + np.pi / 2, 0.75344866]))
    computer_to_executer.put(command)
    while from_executer.empty():
        sleep(0.1)
    from_executer.get()


    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    command = Command(CommandTypes.OPEN_GRIPPER, do_async=True)
    computer_to_executer.put(command)
    while from_executer.empty():
        sleep(0.1)
    from_executer.get()

    task = Task(TaskTypes.MOVE_TO, Positions.STATIC_OBSERVATION.value)
    to_computer.put(task)
    while from_executer.empty():
        sleep(0.1)
    from_executer.get()

    command = Command(CommandTypes.GET_OBSERVED_BLOCKS)
    computer_to_executer.put(command)
    while from_executer.empty():
        sleep(0.1)
        pass
    static_block_poses = from_executer.get()

    for i in range(len(static_block_poses)):
        rot = static_block_poses[i][:3, :3]
        loc = static_block_poses[i][:3, 3]
        rotation_z = np.arctan2(rot[1, 0], rot[0, 0]) % (np.pi / 2)
        if rotation_z > np.pi / 4:
            rotation_z -= np.pi / 2
        elif rotation_z < -np.pi / 4:
            rotation_z += np.pi / 2

        loc[2] = 0.225
        static_block_poses[i] = euler_to_se3(-np.pi, 0, rotation_z, loc)
        task = Task(TaskTypes.GRAB_BLOCK, static_block_poses[i])
        to_computer.put(task)
        task = Task(TaskTypes.MOVE_TO, Positions.STATIC_OBSERVATION.value)
        to_computer.put(task)

    computer.join()
    executer.join()




    # Uncomment to get middle camera depth/rgb images
    # mid_depth = detector.get_mid_depth()
    # mid_rgb = detector.get_mid_rgb()

    # Move around...

    # END STUDENT CODE