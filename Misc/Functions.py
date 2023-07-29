import numpy as np
from Particles.Global_Variables import Global_variables
import math

ROUNDDIGIT = Global_variables.ROUNDDIGIT
V0 = Global_variables.V0
DIM_Numb = Global_variables.DIM_Numb
L_FCT = Global_variables.L_FCT

rng = np.random.default_rng()


def has_any_nonzero(List):
    """Returns True if there is a zero anywhere in the list"""
    return any(value != 0 for value in List)


def has_all_nonzero(List):
    """Returns True if all the elements the list are zero"""
    return all(value != 0 for value in List)


def ROUND(x):
    """
    Rounds the given number 'x' to the number of digits specified by 'ROUNDDIGIT'.
    :param x: the number to be rounded
    :return: the rounded number
    """
    return round(x, ROUNDDIGIT)


def GenNewList(Nlist):
    """Returns a list containing empty lists of lengths given by each number in Nlist

    Args:
        Nlist (_type_): _description_
    """
    return [[[] for n in range(ni)] for ni in Nlist]


def COUNTFCT(LIST, elem, mode=None):
    """
    Counts the number of times 'elem' appears in the  in 'LIST'.
    :param LIST: a list of sublists
    :param elem: a list to test
    :return: 1 if 'elem' appears in  in 'LIST' otherwise 0
    """

    if len(elem) == 2:  # id,type search
        if not isinstance(LIST, np.ndarray):
            LIST = np.array(LIST)
        if not isinstance(elem, np.ndarray):
            elem = np.array(elem)

        if LIST.size == 0:
            return 0
        elif LIST.shape[0] == 1:
            Testequal = np.array_equal(LIST[0], elem)
            if isinstance(Testequal, bool):
                return Testequal
            else:
                return (Testequal).all()
        SEARCH = np.where((LIST[:] == elem).all(axis=1))[0]
        cnt = len(SEARCH)
        return cnt
    else:
        if isinstance(LIST, np.ndarray):
            LIST = list(LIST)
        if isinstance(elem, np.ndarray):
            elem = list(elem)

        if isinstance(elem[1], np.ndarray):
            elem[1] = list(elem[1])
        for sublist in LIST:
            if isinstance(sublist, np.ndarray):
                sublist = list(sublist)
            if isinstance(sublist[1], np.ndarray):
                sublist[1] = list(sublist[1])
            if sublist == elem:
                return 1
        return 0


def numpy_norm(Vector, axis=None):
    return np.linalg.norm(Vector, axis)


def ONE_Dim_norm(Vector, axis=None):
    if axis is None:
        return Vector[0] * Vector[0]
        # return sum(v*v for v in Vector) ** 0.5
    else:
        return np.linalg.norm(Vector, axis)


def TWO_Dim_norm(Vector, axis=None):
    if axis is None:
        return math.hypot(Vector[0], Vector[1])
        # return sum(v*v for v in Vector) ** 0.5
    else:
        return np.linalg.norm(Vector, axis)


def THREE_Dim_norm(Vector, axis=None):
    if axis is None:
        return math.hypot(Vector[0], Vector[1], Vector[2])
        # return sum(v*v for v in Vector) ** 0.5
    else:
        return np.linalg.norm(Vector, axis)


if DIM_Numb == 3:
    NORM = THREE_Dim_norm
elif DIM_Numb == 2:
    NORM = TWO_Dim_norm
else:
    NORM = ONE_Dim_norm
