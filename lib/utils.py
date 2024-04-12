import numpy as np

def euler_to_so3(r,p,y):
    rx = np.array([[1,0,0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p),0,np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y),-np.sin(y),0], [np.sin(y), np.cos(y),0],[0, 0, 1]])
    so3 = rz @ ry @ rx
    return so3

def euler_to_se3(r, p, y, v):
    so3 = euler_to_so3(r,p,y)
    se3 = np.vstack((np.hstack((so3, v.reshape(-1,1))), np.array([0,0,0,1])))
    return se3