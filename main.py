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
from follower import Follower



if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    follower = Follower()
    callback = lambda state: follower.reach_target(state)
    arm = ArmController(on_state_callback=callback)
    follower.set_arm(arm)
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!
    arm.close_gripper()

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    # get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()
    print("Camera pose:\n", H_ee_camera)

    q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])
    # open gripper in new process
    t1 = threading.Thread(target=arm.open_gripper)
    t1.start()

    arm.safe_move_to_position(q)

    # Detect some blocks...
    detections = detector.get_detections()
    for (name, pose) in detections:
         print(name,'\n',pose)

    t1.join()

    fk = FK()
    ik = IK()
    transforms = fk.compute_Ai(q)
    H0c = transforms[-1] @ H_ee_camera
    block_locations = []
    block_rotations = []
    for (name, pose) in detections:
        world_pose = H0c @ pose
        block_locations.append(world_pose[:3, 3])
        block_rotations.append(world_pose[:3, :3])
        print(block_locations[-1])

    arm.safe_move_to_position(arm.neutral_position())

    target_id = int(input("Select block to pick up: "))
    rot = block_rotations[target_id]
    rotation_z = np.arctan2(rot[1, 0], rot[0, 0]) % (np.pi/2)

    target_loc = block_locations[target_id] + np.array([0, 0, 0.1])
    target = euler_to_se3(-np.pi, 0, rotation_z, target_loc)

    # _, T0e = fk.forward(q)
    # follower.set_target(T0e, target)
    # while not follower.complete:
    #     pass
    q, _, _, _ = ik.inverse(target, q, alpha=.86)
    arm.safe_move_to_position(q)

    target_loc = block_locations[target_id]
    target = euler_to_se3(-np.pi, 0, rotation_z, target_loc)
    q, _, _, _ = ik.inverse(target, q, alpha=.86)
    arm.safe_move_to_position(q)




    # Uncomment to get middle camera depth/rgb images
    # mid_depth = detector.get_mid_depth()
    # mid_rgb = detector.get_mid_rgb()

    # Move around...

    # END STUDENT CODE