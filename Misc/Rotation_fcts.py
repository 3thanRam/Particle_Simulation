import numpy as np
from Particles.Global_Variables import Global_variables

DIM_Numb = Global_variables.DIM_Numb


def Rotate_vector_xyzAxes(Vector, angle):
    Cos = np.cos(angle)
    Sin = np.sin(angle)
    if DIM_Numb == 3:
        Yaw = [[[Cos, -Sin, 0], [Sin, Cos, 0], [0, 0, 1]]]
        Pitch = [[[Cos, 0, Sin], [0, 1, 0], [-Sin, 0, Cos]]]
        Rotmatrix = np.matmul(Yaw, Pitch)
    elif DIM_Numb == 2:
        Rotmatrix = np.array([[Cos, -Sin], [Sin, Cos]])
    return np.matmul(Rotmatrix, Vector)


def ROT2D(Angle):
    ROT = np.array([[np.cos(Angle), -np.sin(Angle)], [np.sin(Angle), np.cos(Angle)]])
    return ROT


def angle_axis_quat(theta, axis):
    """
    Given an angle and an axis, it returns a quaternion.
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    return np.append([np.cos(theta / 2)], np.sin(theta / 2) * axis)


def mult_quat(q1, q2):
    """
    Quaternion multiplication.
    """
    q3 = np.copy(q1)
    q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return q3


def rotate_quat(quat, vect):
    """
    Rotate a vector with the rotation defined by a quaternion.
    """
    # Transfrom vect into an quaternion
    vect = np.append([0], vect)
    # Normalize it
    norm_vect = np.linalg.norm(vect)
    vect = vect / norm_vect
    # Computes the conjugate of quat
    quat_ = np.append(quat[0], -quat[1:])
    # The result is given by: quat * vect * quat_
    res = mult_quat(quat, mult_quat(vect, quat_)) * norm_vect
    return res[1:]
