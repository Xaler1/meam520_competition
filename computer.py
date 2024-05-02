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


@dataclass
class Task:
    id: str
    task_type: TaskTypes
    target_pose: np.ndarray = None
    target_pose2: np.ndarray = None
    command: Command = None
    hover_gap: float = 0.1
    extra_hover: bool = True


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
            if not success:
                print(f"--------------------------------- FAILED TO FIND Q for {id} ---------------------------------")
                return None, False
        else:
            q = known_q
        command = Command(id, CommandTypes.MOVE_TO, q, do_async=do_async, extra_fast=extra_fast, order=order)
        self.to_executor.put(command)
        return q, True

    def run(self):
        order = 0
        last_q = None
        cache_q = {}
        while True:
            if not self.from_main.empty():
                task = self.from_main.get()
                target = task.target_pose
                id = task.id
                print("Computer got command", task.task_type, "for", id)

                # Move to a location
                if task.task_type == TaskTypes.MOVE_TO:
                    if id in cache_q:
                        print("Using cached position for", id)
                        last_q, _ = self.move_command(id, order, target, start=last_q, do_async=False, known_q=cache_q[id])
                    else:
                        last_q, _ = self.move_command(id, order, target, start=last_q, do_async=False)
                    cache_q[id] = last_q
                    order += 1

                # Move to and grab a block
                elif task.task_type == TaskTypes.GRAB_BLOCK:
                    hover_pose = task.target_pose.copy()
                    hover_pose[2, 3] += task.hover_gap
                    q, success = self.move_command(id + "-0", order, hover_pose, start=last_q, do_async=True, extra_fast=False)
                    if not success:
                        self.to_main.put(False)
                        continue
                    order += 1
                    _, success =  self.move_command(id + "-1", order, target, start=q, do_async=True)
                    if not success:
                        last_q = q
                        self.to_main.put(False)
                        continue
                    order += 1
                    command = Command(id + "-2", CommandTypes.GRAB_BLOCK, do_async=True, order=order)
                    order += 1
                    self.to_executor.put(command)
                    command = Command(id + "-3", CommandTypes.MOVE_TO, q, do_async=True, order=order)
                    self.to_executor.put(command)
                    last_q = q

                    self.to_main.put(True)


                # Move to a location and place a block
                elif task.task_type == TaskTypes.PLACE_BLOCK:
                    hover_pose = task.target_pose.copy()
                    hover_pose[2, 3] += task.hover_gap
                    q, _ = self.move_command(id + "-0", order, hover_pose, do_async=True, extra_fast=False)
                    order += 1
                    q, _ = self.move_command(id + "-1", order, target, start=q, do_async=True)
                    order += 1
                    command = Command(id + "-2", CommandTypes.OPEN_GRIPPER, do_async=True, order=order)
                    order += 1
                    self.to_executor.put(command)
                    if task.extra_hover:
                        hover_pose[2, 3] += 0.05
                    q, _ = self.move_command(id + "-3", order, hover_pose, start=q, do_async=True)
                    #command = Command(id + "-3", CommandTypes.MOVE_TO, q, do_async=True, order=order)
                    #self.to_executor.put(command)
                    last_q = q



                # Send command straight to executor
                elif task.task_type == TaskTypes.BYPASS:
                    command = task.command
                    command.order = order
                    order += 1
                    self.to_executor.put(command)
