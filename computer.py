from lib.IK_position_null import IK

import numpy as np
from multiprocessing import Process, Queue
from dataclasses import dataclass
from enum import Enum
from executer import Command, CommandTypes


class TaskTypes(Enum):
    GRAB_BLOCK = 1
    PLACE_BLOCK = 2
    MOVE_TO = 3
    BYPASS = 4


@dataclass
class Task:
    task_type: TaskTypes
    target_pose: np.ndarray = None
    command: Command = None


class Computer:
    def __init__(self, from_main: Queue, to_executer: Queue):
        self.from_main = from_main
        self.to_executer = to_executer
        self.ik = IK()
        self.default_pose = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])


    def move_command(self, target, start=None, do_async=False, extra_fast=False):
        if start is None:
            start = self.default_pose
        q, _, _, _ = self.ik.inverse(target, start, alpha=0.86)
        command = Command(CommandTypes.MOVE_TO, q, do_async=do_async, extra_fast=extra_fast)
        self.to_executer.put(command)
        return q



    def run(self):
        while True:
            if not self.from_main.empty():
                task = self.from_main.get()
                target = task.target_pose
                if task.task_type == TaskTypes.MOVE_TO:
                    print("Computer got command to move")
                    self.move_command(target, do_async=True)
                elif task.task_type == TaskTypes.GRAB_BLOCK:
                    print("Computer got command to grab block")
                    hover_pose = task.target_pose.copy()
                    hover_pose[2, 3] += 0.1
                    q = self.move_command(hover_pose, do_async=True, extra_fast=True)
                    self.move_command(target, start=q, do_async=True)
                    command = Command(CommandTypes.GRAB_BLOCK, do_async=True)
                    self.to_executer.put(command)
                    command = Command(CommandTypes.MOVE_TO, q, do_async=True)
                    self.to_executer.put(command)
                elif task.task_type == TaskTypes.PLACE_BLOCK:
                    print("Computer got command to place block")
                    hover_pose = task.target_pose.copy()
                    hover_pose[2, 3] += 0.1
                    q = self.move_command(hover_pose, do_async=True, extra_fast=True)
                    self.move_command(target, start=q, do_async=True)
                    command = Command(CommandTypes.OPEN_GRIPPER, do_async=True)
                    self.to_executer.put(command)
                    command = Command(CommandTypes.MOVE_TO, q, do_async=True)
                    self.to_executer.put(command)
                elif task.task_type == TaskTypes.BYPASS:
                    command = task.command
                    self.to_executer.put(command)


