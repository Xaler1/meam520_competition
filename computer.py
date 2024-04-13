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


@dataclass
class Task:
    task_type: TaskTypes
    target_pose: np.ndarray = None


class Computer:
    def __init__(self, from_main: Queue, to_executer: Queue):
        self.from_main = from_main
        self.to_executer = to_executer
        self.ik = IK()
        self.default_pose = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])


    def run(self):
        while True:
            if not self.from_main.empty():
                task = self.from_main.get()
                target = task.target_pose
                if task.task_type == TaskTypes.MOVE_TO:
                    print("Computer got command to move")
                    q, _, _, _ = self.ik.inverse(target, self.default_pose, alpha=0.86)
                    command = Command(CommandTypes.MOVE_TO, q)
                    self.to_executer.put(command)
                if task.task_type == TaskTypes.GRAB_BLOCK:
                    print("Computer got command to grab block")
                    target1 = task.target_pose.copy()
                    target1[2, 3] += 0.15
                    q, _, _, _ = self.ik.inverse(target1, self.default_pose, alpha=0.86)
                    command = Command(CommandTypes.MOVE_TO, q, do_async=True)
                    self.to_executer.put(command)
                    q, _, _, _ = self.ik.inverse(task.target_pose, q, alpha=0.86)
                    command = Command(CommandTypes.MOVE_TO, q, do_async=True)
                    self.to_executer.put(command)
                    command = Command(CommandTypes.CLOSE_GRIPPER)
                    self.to_executer.put(command)


