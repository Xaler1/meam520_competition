import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    fk = FK()
    matrices = fk.compute_Ai(q_in)

    Jv = np.zeros((3, 7))
    Jw = np.zeros((3, 7))
    for i, mat in enumerate(matrices[:-1]):
        Jv[:, i] = np.cross(mat[:3, 2], matrices[-1][:3, 3] - mat[:3, 3])
        Jw[:, i] = mat[:3, 2]

    J = np.vstack((Jv, Jw))

    J = np.array(J).astype(np.float64)

    return J

