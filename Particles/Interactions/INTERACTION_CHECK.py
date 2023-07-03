import numpy as np


from Particles.Interactions.INTERACTION_DEF import COLTYPE
from Particles.Global_Variables import Global_variables
from Particles.Dictionary import PARTICLE_DICT

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

ROUNDDIGIT = Global_variables.ROUNDDIGIT
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
L_FCT = Global_variables.L_FCT


# Global_variables.L=Global_variables.Global_variables.L
# Global_variables.Linf=Global_variables.Global_variables.Linf


def has_any_nonzero(List):
    """Returns True if there is a zero anywhere in the list"""
    for value in List:
        if value != 0:
            return True
    return False


def has_all_nonzero(List):
    """Returns True if all the elements the list are zero"""
    for value in List:
        if value != 0:
            return False
    return True


def in_all_bounds(List, t=None):
    if t == None:
        Lmaxi, Lmini = Global_variables.L, Global_variables.Linf
    else:
        Lmaxi, Lmini = L_FCT[0](t), L_FCT[1](t)
    for indV, value in enumerate(List):
        if value > Lmaxi[indV] or value < Lmini[indV]:
            return False
    return True


def ROUND(x):
    """
    Rounds the given number 'x' to the number of digits specified by 'ROUNDDIGIT'.
    :param x: the number to be rounded
    :return: the rounded number
    """
    return round(x, ROUNDDIGIT)


def MINIMISE(difA, difB, t, Sa, Sb, Dist_min):
    """
    This function takes in five arguments, `difA`, `difB`, `t`, `Sa`, and `Sb`, and returns a float value.

    Args:
    - difA (numpy array): an array containing the differences between the x-component velocities of two particles.
    - difB (numpy array): an array containing the differences between the x-component positions of two particles.
    - t (float): a time value.
    - Sa (numpy array): an array containing the x-component of the acceleration of particle a.
    - Sb (numpy array): an array containing the x-component of the acceleration of particle b.

    Returns:
    - A float value representing the time at which two particles meet, or -1 if they do not meet.
    """

    # Find the possible time values where the two particles meet.

    mask = difA != 0
    Tsol = np.divide(-difB, difA, where=mask)

    # Filter the time values that are within the acceptable range.
    Tsol = Tsol[
        (0.5 * (Sa * Tsol + Sb) > L_FCT[1](Tsol))
        & (0.5 * (Sa * Tsol + Sb) < L_FCT[0](Tsol))
        & (Tsol < t)
        & (Tsol > t - dt)
    ]

    # If there are possible exact meeting times, return the first chronologicaly
    if Tsol.size > 0:
        Tmini = []
        for tsol in Tsol:
            dist = np.sqrt(np.sum((difA * tsol + difB) ** 2))
            if dist < Dist_min:
                Tmini.append(tsol)
        if len(Tmini) > 0:
            if len(Tmini) > 1:
                return ROUND(min(Tmini))
            else:
                return ROUND(Tmini[0])

    if (
        DIM_Numb == 1
    ):  # If there are no possible meeting times and there is only one dimension, return -1.
        return -1
    else:  # If there are no possible meeting times and there are multiple dimensions,find the earliest possible meeting time among the dimensions.
        if has_all_nonzero(difA):
            return t - 0.5 * dt
        else:
            tmini = t
            for dim in range(DIM_Numb):
                if difA[dim] != 0:
                    td = -difB[dim] / difA[dim]
                    if td < t - dt:
                        tminid = t - 0.99 * dt
                    else:
                        tminid = t * 0.99
                    if tminid < tmini:
                        tmini = tminid
            return ROUND(tmini)


def INTERCHECK_ND(a1, b1, p1, a2, b2, p2, t, z1, z2, Tstart, Tend):
    """
    Check if the line segments defined by points a1, b1 and a2, b2 intersect
    during the time interval [Tstart, Tend] and return information about the
    intersection.

    Args:
    - a1, b1: numpy arrays with the coordinates of the endpoints of the first segment.
    - p1: an integer representing the index of the first segment.
    - a2, b2: numpy arrays with the coordinates of the endpoints of the second segment.
    - p2: an integer representing the index of the second segment.
    - t: current time.
    - z1, z2: numpy arrays with the velocities of the first and second segment, respectively.
    - Tstart, Tend: start and end times of the time interval.

    Returns:
    - If the segments intersect during the time interval, returns a list with the following elements:
      - 1 if the intersection point is inside the first segment, 2 if it is inside the second segment.
      - tmini: time of intersection.
      - xo: coordinates of intersection point.
      - z1, z2: velocities of the first and second segment, respectively.
    - If the segments do not intersect during the time interval, returns [0].
    """
    # Calculate the difference of the slopes and intercepts of the two lines
    difA = np.array(a1 - a2, dtype=float)
    difB = np.array(b1 - b2, dtype=float)
    # Calculate the sum of the slopes and intercepts of the two lines
    SA = np.array(a1 + a2, dtype=float)
    SB = np.array(b1 + b2, dtype=float)
    # Check if the lines are parallel
    if has_any_nonzero(difA):
        # Find the time t for the intersection point with minimum distance between the lines
        Dist_min = (
            PARTICLE_DICT[PARTICLE_NAMES[p1[1]]]["size"]
            + PARTICLE_DICT[PARTICLE_NAMES[p2[1]]]["size"]
        )
        tmini = MINIMISE(difA, difB, t, SA, SB, Dist_min)

        # Calculate the minimum distance D between the lines at the intersection point
        D = np.sqrt(np.sum((difA * tmini + difB) ** 2))

        # Calculate the coordinates of the intersection point
        xo = np.array((tmini * SA + SB) / 2, dtype=np.float64)
        xo = np.round(xo, decimals=ROUNDDIGIT)

        # Check if the intersection point is valid

        if 0 < Tstart <= tmini <= Tend and D <= Dist_min and in_all_bounds(xo, tmini):
            return [COLTYPE(p1, p2), tmini, xo, z1, z2]

    # If the lines are parallel or do not intersect during the time interval, return [0]
    return [0]


def INTERCHECK_1D(a1, b1, p1, a2, b2, p2, t, z1, z2, Tstart, Tend):
    """Faster INTERCHECK function for 1D"""

    a1, a2 = float(a1), float(a2)
    b1, b2 = b1[0], b2[0]
    difA = a1 - a2
    difB = b2 - b1
    if difA != 0:
        ti = ROUND(difB / difA)
        xo = np.array([round(ti * a2 + b2, ROUNDDIGIT - 2)])
        if (
            Tstart <= ti <= Tend
            and t - dt <= ti <= t
            and ti > 0
            and Global_variables.Linf < xo < Global_variables.L
        ):
            return [COLTYPE(p1, p2), ti, xo, z1, z2]
    return [0]
