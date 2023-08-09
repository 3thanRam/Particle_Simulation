import numpy as np
from Interactions.Interaction_definition import (
    COLTYPE,
)
from Particles.Global_Variables import Global_variables
from Particles.Dictionary import PARTICLE_DICT
from Misc.Functions import ROUND, NORM

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

ROUNDDIGIT = Global_variables.ROUNDDIGIT
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
L_FCT = Global_variables.L_FCT


def quad_equ(a, b, c, bound_start, bound_end):
    Delta2 = b**2 - 4 * a * c

    if Delta2 == 0:
        to = round(-b / (2 * a), ROUNDDIGIT)
        if bound_start <= to <= bound_end:
            return [to]
    elif Delta2 > 0:
        sols = []
        Delta = Delta2**0.5
        for i in range(2):
            to = round((-b + ((-1) ** i) * Delta) / (2 * a), ROUNDDIGIT)
            if bound_start <= to <= bound_end:
                sols.append(to)
        return sols
    return []


def GetABC(acoef_1, acoef_2, bcoef_1, bcoef_2, TotSize):
    Delta_b = bcoef_2 - bcoef_1
    Delta_a = acoef_2 - acoef_1
    A = np.dot(Delta_a, Delta_a)
    B = 2 * np.dot(Delta_a, Delta_b)
    C = np.dot(Delta_b, Delta_b) - TotSize**2
    return A, B, C


def MINIMISE(a1, b1, a2, b2, Dist1, Dist2, Tstart, Tend):
    """
    Search for possible intersections between particle trajectories and if there are multiple then get first occuring

    Args:
    - a1/a2 (numpy array): an array containing the velocties of the two particles.
    - b1/b2 (numpy array): an array containing the b coefs of the two particles.
    - Dist1/Dist2 (float): sizes of the two particles.
    - Tstart/Tend (float): start and end time of possible interactions between the particles.

    Returns:
    - Tmini(float): time of first possible collision
    - Xo(ndarray): location of first possible collision (empty if none )
    """

    # Find the possible time values where the two particles meet.

    # Calculate the difference of the slopes and intercepts of the two lines

    # Dini = NORM(Tstart * (a1 - a2) + (b1 - b2))
    # if Dini < (Dist1 + Dist2):
    #    return (Tstart, [])
    Dfin = NORM(Tend * (a1 - a2) + (b1 - b2))  # NORM(X1_fin - X2_fin)

    if Dfin <= (Dist1 + Dist2):
        # quad formula A*u^2 +B*u+C
        A, B, C = GetABC(a1, a2, b1, b2, Dist1 + Dist2)
        Tsols = quad_equ(A, B, C, Tstart, Tend)

        for i in range(len(Tsols)):
            Tmini = min(Tsols)
            x1 = Tmini * a1 + b1
            x2 = Tmini * a2 + b2
            R = Dist1 / (Dist1 + Dist2)
            Xo = x1 + (x2 - x1) * R
            Xo = np.round(Xo, decimals=ROUNDDIGIT)
            Dt = ROUND(NORM(x1 - x2))
            if Dt <= (Dist1 + Dist2):
                return (Tmini, Xo)
            Tsols.remove(Tmini)
    return (0, [])


def INTERCHECK(a1, b1, p1, a2, b2, p2, t, z1, z2, Tstart, Tend):
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
      - type of collision defined in INTERACTION_DEF
      - tmini: time of intersection.
      - xo: coordinates of intersection point.
      - z1, z2: velocities of the first and second segment, respectively.
    - If the segments do not intersect during the time interval, returns [0].
    """

    Dist1, Dist2 = (
        PARTICLE_DICT[PARTICLE_NAMES[p1[1]]]["size"] / 2,
        PARTICLE_DICT[PARTICLE_NAMES[p2[1]]]["size"] / 2,
    )
    tmini, xo = MINIMISE(a1, b1, a2, b2, Dist1, Dist2, Tstart, Tend)
    if len(xo) > 0:
        return [COLTYPE(p1, p2), tmini, xo, z1, z2]
    else:
        # If the lines are parallel or do not intersect during the time interval, return [0]
        return [0]
