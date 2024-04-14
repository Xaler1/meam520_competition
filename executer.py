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

class CommandTypes(Enum):
    MOVE_TO = 1
    CLOSE_GRIPPER = 2
    OPEN_GRIPPER = 3
    GRAB_BLOCK = 4
    GET_OBSERVED_BLOCKS = 6

@dataclass
class Command:
    command_type: CommandTypes
    target_q: np.ndarray = None
    do_async: bool = False
    extra_fast: bool = False


class Executer:
    def __init__(self, from_computer: Queue, to_main: Queue):
        self.fk = FK()
        self.ik = IK()
        self.from_computer = from_computer
        self.to_main = to_main


    def get_observed_blocks(self, current_q, detector: ObjectDetector):
        H_ee_camera = detector.get_H_ee_camera()
        detections = detector.get_detections()
        for (name, pose) in detections:
            print(name, '\n', pose)

        transforms = self.fk.compute_Ai(current_q)
        H0c = transforms[-1] @ H_ee_camera
        block_poses = []
        for (name, pose) in detections:
            world_pose = H0c @ pose
            block_poses.append(world_pose)

        return block_poses

    def open_gripper_async(self, arm: ArmController):
        t = Thread(target=arm.open_gripper)
        t.start()
        sleep(0.2)
        return t

    def close_gripper_async(self, arm: ArmController):
        t = Thread(target=arm.close_gripper)
        t.start()
        return t

    def grab_block_async(self, arm: ArmController):
        t = Thread(target=arm.exec_gripper_cmd, args=(0.045, 50))
        t.start()
        sleep(0.3)


    def move_arm_async(self, arm: ArmController, q) -> Thread:
        t = Thread(target=arm.safe_move_to_position, args=(q,))
        t.start()
        return t


    def run(self):
        rospy.init_node("team_script")
        arm = ArmController()
        detector = ObjectDetector()
        arm.set_arm_speed(0.5)
        current_config = None
        while True:
            if not self.from_computer.empty():
                command = self.from_computer.get()
                if command.command_type == CommandTypes.MOVE_TO:
                    print("Executer got command to move")
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
                        thread = self.move_arm_async(arm, command.target_q)
                        sleep(timing)
                    else:
                        arm.safe_move_to_position(command.target_q)
                    current_config = command.target_q
                elif command.command_type == CommandTypes.CLOSE_GRIPPER:
                    print("Executer got command to close gripper")
                    if command.do_async:
                        self.close_gripper_async(arm)
                    else:
                        arm.close_gripper()
                    pass
                elif command.command_type == CommandTypes.GRAB_BLOCK:
                    print("Executer got command to grab block")
                    if command.do_async:
                        self.grab_block_async(arm)
                    else:
                        arm.exec_gripper_cmd(0.045, 50)
                elif command.command_type == CommandTypes.OPEN_GRIPPER:
                    print("Executer got command to open gripper")
                    if command.do_async:
                        self.open_gripper_async(arm)
                    else:
                        arm.open_gripper()
                elif command.command_type == CommandTypes.GET_OBSERVED_BLOCKS:
                    print("Executer got command to get observed blocks")
                    poses = self.get_observed_blocks(current_config, detector)
                    self.to_main.put(poses)

                self.to_main.put("done")
