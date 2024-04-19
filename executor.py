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

from multiprocessing import Queue
from enum import Enum
from dataclasses import dataclass
import numpy as np
from threading import Thread
from time import sleep
from manipulator import Manipulator, Action, ActionType


class CommandTypes(Enum):
    MOVE_TO = 1
    CLOSE_GRIPPER = 2
    OPEN_GRIPPER = 3
    GRAB_BLOCK = 4
    GET_OBSERVED_BLOCKS = 6


@dataclass
class Command:
    id: str
    command_type: CommandTypes
    target_q: np.ndarray = None
    do_async: bool = False
    extra_fast: bool = False
    order: int = 0


class Executor:
    def __init__(self,
                 from_computer: Queue,
                 to_main: Queue,
                 to_manipulator: Queue,
                 from_manipulator: Queue,
                 observations: Queue
                 ):
        self.fk = FK()
        self.ik = IK()
        self.from_computer = from_computer
        self.to_main = to_main
        self.to_manipulator = to_manipulator
        self.from_manipulator = from_manipulator
        self.observations = observations

    def get_observed_blocks(self, current_q, H_ee_camera, detections):
        for (name, pose) in detections:
            print(name, '\n', pose)

        transforms = self.fk.compute_Ai(current_q)
        H0c = transforms[-1] @ H_ee_camera
        block_poses = {}
        for (name, pose) in detections:
            world_pose = H0c @ pose
            block_poses[name] = world_pose

        return block_poses

    def wait_for_action(self, id: str):
        while True:
            if not self.from_manipulator.empty():
                completed = self.from_manipulator.get()
                if completed == id:
                    return

    def run_command(self, command: Command, current_config):
        id = command.id
        # Moving to config q
        print("Executor got command:", id, "| Async:", command.do_async)
        if command.command_type == CommandTypes.MOVE_TO:
            if command.do_async and current_config is not None:
                diff = np.abs(current_config - command.target_q)
                _, current_pose = self.fk.forward(current_config)
                _, target_pose = self.fk.forward(command.target_q)
                current_loc = current_pose[:3, 3]
                target_loc = target_pose[:3, 3]
                dist = np.linalg.norm(target_loc - current_loc)
                timing = 1 * dist + 0.8
                if command.extra_fast:
                    timing -= 0.8
                print("timing", timing)
                sleep(timing)
                action = Action(id, ActionType.MOVE_TO, target_q=command.target_q)
                self.to_manipulator.put(action)
                sleep(timing)
            else:
                action = Action(id, ActionType.MOVE_TO, target_q=command.target_q)
                self.to_manipulator.put(action)
                self.wait_for_action(id)
            current_config = command.target_q

        # Closing the gripper
        elif command.command_type == CommandTypes.CLOSE_GRIPPER:
            print("Executer got command to close gripper")
            action = Action(id, ActionType.CLOSE_GRIPPER)
            self.to_manipulator.put(action)
            if not command.do_async:
                self.wait_for_action(id)

        # Grabbing a block
        elif command.command_type == CommandTypes.GRAB_BLOCK:
            print("Executor got command to grab block")
            action = Action(id, ActionType.SET_GRIPPER, target_width=0.045, target_force=50)
            self.to_manipulator.put(action)
            if not command.do_async:
                self.wait_for_action(id)
            else:
                sleep(0.3)

        # Opening gripper
        elif command.command_type == CommandTypes.OPEN_GRIPPER:
            print("Executor got command to open gripper")
            action = Action(id, ActionType.OPEN_GRIPPER)
            self.to_manipulator.put(action)
            if not command.do_async:
                self.wait_for_action(id)

        # Getting a list of observed blocks
        elif command.command_type == CommandTypes.GET_OBSERVED_BLOCKS:
            print("Executor got command to get observed blocks")
            if not self.observations.empty():
                self.observations.get()
            while True:
                sleep(0.1)
                if not self.observations.empty():
                    observation = self.observations.get()
                    camera_transform = observation.camera_transform
                    detections = observation.camera_detections
                    break
            poses = self.get_observed_blocks(current_config, camera_transform, detections)
            self.to_main.put(id)
            self.to_main.put(poses)

        print("Executor finished", id)
        if command.command_type != CommandTypes.GET_OBSERVED_BLOCKS:
            self.to_main.put(id)
        return current_config

    def run(self):
        current_config = None
        order = 0
        holdback = None
        while True:
            if holdback is not None and holdback.order == order:
                print("Executor executing held-back command, pray to god this is the only one")
                current_config = self.run_command(holdback, current_config)
                order += 1
            if not self.from_computer.empty():
                command = self.from_computer.get()
                if order < command.order:
                    print("Executor got out of order command. Expected:", order, " Actual:", command.order)
                    holdback = command
                    continue
                order += 1
                current_config = self.run_command(command, current_config)


