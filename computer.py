from lib.IK_position_null import IK

import numpy as np
from multiprocessing import Process, Queue
from dataclasses import dataclass
from enum import Enum
from executor import Command, CommandTypes
from time import time


class TaskTypes(Enum):
    GRAB_BLOCK = 1
    PLACE_BLOCK = 2
    MOVE_TO = 3
    BYPASS = 4
    MOVE_BLOCK = 5


@dataclass
class Task:
    id: str
    task_type: TaskTypes
    target_pose: np.ndarray = None
    target_pose2: np.ndarray = None
    command: Command = None
    hover_gap: float = 0.1


class Computer:
    def __init__(self, from_main: Queue, to_executor: Queue, to_main: Queue):
        self.from_main = from_main
        self.to_executor = to_executor
        self.to_main = to_main
        self.ik = IK()
        self.default_pose = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])

    def move_command(self, id: str, order: int, target, start=None, do_async=False, extra_fast=False, known_q=None):
        if start is None:
            start = self.default_pose
        if known_q is None:
            q, _, success, _ = self.ik.inverse(target, start, alpha=0.86)
        else:
            q = known_q
        command = Command(id, CommandTypes.MOVE_TO, q, do_async=do_async, extra_fast=extra_fast, order=order)
        self.to_executor.put(command)
        return q

    def run(self):
        order = 0
        last_q = None
        cache_q = {}
        while True:
            if not self.from_main.empty():
                task = self.from_main.get()
                target = task.target_pose
                id = task.id

                # Move to a location
                if task.task_type == TaskTypes.MOVE_TO:
                    print("Computer got command to move")
                    if id in cache_q:
                        last_q = self.move_command(id, order, target, start=last_q, do_async=False, known_q=cache_q[id])
                    else:
                        last_q = self.move_command(id, order, target, start=last_q, do_async=False)
                    cache_q[id] = last_q
                    order += 1

                # Move to and grab a block
                elif task.task_type == TaskTypes.GRAB_BLOCK:
                    print("Computer got command to grab block")
                    hover_pose = task.target_pose.copy()
                    hover_pose[2, 3] += task.hover_gap
                    q = self.move_command(id + "-0", order, hover_pose, start=last_q, do_async=True, extra_fast=True)
                    order += 1
                    self.move_command(id + "-1", order, target, start=q, do_async=True)
                    order += 1
                    command = Command(id + "-2", CommandTypes.GRAB_BLOCK, do_async=True, order=order)
                    order += 1
                    self.to_executor.put(command)
                    command = Command(id + "-3", CommandTypes.MOVE_TO, q, do_async=True, order=order)
                    self.to_executor.put(command)
                    last_q = q

                # Move to a location and place a block
                elif task.task_type == TaskTypes.PLACE_BLOCK:
                    print("Computer got command to place block")
                    hover_pose = task.target_pose.copy()
                    hover_pose[2, 3] += task.hover_gap
                    q = self.move_command(id + "-0", order, hover_pose, do_async=True, extra_fast=True)
                    order += 1
                    self.move_command(id + "-1", order, target, start=q, do_async=True)
                    order += 1
                    command = Command(id + "-2", CommandTypes.OPEN_GRIPPER, do_async=True, order=order)
                    order += 1
                    self.to_executor.put(command)
                    command = Command(id + "-3", CommandTypes.MOVE_TO, q, do_async=True, order=order)
                    self.to_executor.put(command)
                    last_q = q

                elif task.task_type == TaskTypes.MOVE_BLOCK:
                    print("Computer got command to move block")
                    target1 = task.target_pose
                    target2 = task.target_pose2
                    max_height = max(target1[2, 3], target2[2, 3])
                    hover_pose = target1.copy()
                    hover_pose[2, 3] = max_height + task.hover_gap
                    q = self.move_command(id + "-0", order, hover_pose, start=last_q, do_async=True)
                    order += 1
                    q = self.move_command(id + "-1", order, target1, start=q, do_async=True)
                    order += 1
                    command = Command(id + "-2", CommandTypes.GRAB_BLOCK, do_async=True, order=order)
                    order += 1
                    self.to_executor.put(command)
                    q = self.move_command(id + "-3", order, hover_pose, start=q, do_async=True)
                    order += 1

                    hover_pose = target2.copy()
                    hover_pose[2, 3] = max_height + task.hover_gap
                    q = self.move_command(id + "-4", order, hover_pose, start=q, do_async=True)
                    order += 1
                    q = self.move_command(id + "-5", order, target2, start=q, do_async=True)
                    order += 1
                    command = Command(id + "-6", CommandTypes.OPEN_GRIPPER, do_async=True, order=order)
                    order += 1
                    self.to_executor.put(command)
                    q = self.move_command(id + "-7", order, hover_pose, start=q, do_async=True)
                    order += 1



                # Send command straight to executor
                elif task.task_type == TaskTypes.BYPASS:
                    command = task.command
                    command.order = order
                    order += 1
                    self.to_executor.put(command)
