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

class ActionType(Enum):
    MOVE_TO = 1
    CLOSE_GRIPPER = 2
    OPEN_GRIPPER = 3
    SET_GRIPPER = 4

@dataclass
class Action:
    id: str
    action_type: ActionType
    target_q: np.ndarray = None
    target_width: float = None
    target_force: float = None

@dataclass
class Observation:
    camera_transform: np.ndarray
    camera_detections: np.ndarray
    mid_rgb: np.ndarray
    mid_depth: np.ndarray



class Manipulator:
    def __init__(self, from_executor: Queue, observations: Queue, completions: Queue):
        self.from_executor = from_executor
        self.observations = observations
        self.completions = completions
        self.camera_transform = None

    def run(self):
        rospy.init_node("team_script")
        arm = ArmController()
        detector = ObjectDetector()
        self.camera_transform = detector.get_H_ee_camera()
        arm.set_arm_speed(0.5)
        threads = []
        while True:
            # Create and post observation
            if self.observations.empty():
                camera_transform = self.camera_transform,
                detections = detector.get_detections()
                mid_depth = detector.get_mid_depth()
                mid_rgb = detector.get_mid_rgb()
                observation = Observation(
                    camera_transform,
                    detections,
                        mid_rgb,
                    mid_depth
                )
                self.observations.put(observation)

            # Perform action
            if not self.from_executor.empty():
                action = self.from_executor.get()
                t = None
                # moving to config
                if action.action_type == ActionType.MOVE_TO:
                    q = action.target_q
                    t = Thread(target=arm.safe_move_to_position, args=(q,), name=action.id)


                # opening gripper
                elif action.action_type == ActionType.OPEN_GRIPPER:
                    t = Thread(target=arm.open_gripper, name=action.id)

                # closing gripper
                elif action.action_type == ActionType.CLOSE_GRIPPER:
                    t = Thread(target=arm.close_gripper, name=action.id)

                # grabbing with gripper
                elif action.action_type == ActionType.SET_GRIPPER:
                    width = action.target_width
                    force = action.target_force
                    t = Thread(target=arm.exec_gripper_cmd, args=(width, force))

                if t is not None:
                    t.start()
                    threads.append(t)


            # Check dead threads
            to_delete = []
            for thread in threads:
                if not thread.is_alive():
                    to_delete.append(thread)
            for thread in to_delete:
                print("Manipulator finished", thread.name)
                thread.join()
                threads.remove(thread)
                self.completions.put(thread.name)
