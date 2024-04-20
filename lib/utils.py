import numpy as np
from math import sin, cos, pi


def euler_to_so3(r, p, y):
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    so3 = rz @ ry @ rx
    return so3


def euler_to_se3(r, p, y, v):

    so3 = euler_to_so3(r, p, y)
    se3 = np.vstack((np.hstack((so3, v.reshape(-1, 1))), np.array([0, 0, 0, 1])))
    return se3



def trans(d):
    """
    Compute pure translation homogenous transformation
    """
    return np.array([

        [1, 0, 0, d[0]],
        [0, 1, 0, d[1]],
        [0, 0, 1, d[2]],
        [0, 0, 0, 1],

    ])

def roll(a):
    """
    Compute homogenous transformation for rotation around x axis by angle a
    """
    return np.array([

        [1, 0, 0, 0],
        [0, np.cos(a), -np.sin(a), 0],
        [0, np.sin(a), np.cos(a), 0],
        [0, 0, 0, 1],

    ])

def pitch(a):
    """
    Compute homogenous transformation for rotation around y axis by angle a
    """
    return np.array([

        [np.cos(a), 0, -np.sin(a), 0],
        [0, 1, 0, 0],
        [np.sin(a), 0, np.cos(a), 0],
        [0, 0, 0, 1],

    ])

def yaw(a):
    """
    Compute homogenous transformation for rotation around z axis by angle a
    """
    return np.array([

        [ cos(a), -sin(a), 0, 0 ],
        [ sin(a),  cos(a), 0, 0 ],
        [      0,       0, 1, 0 ],
        [      0,       0, 0, 1 ],
    ])

def transform(d,rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotation corresponding to roll-pitch-yaw euler angles
    """
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])
