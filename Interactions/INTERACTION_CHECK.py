import numpy as np


from Interactions.INTERACTION_DEF import COLTYPE
from Particles.Global_Variables import Global_variables
from Particles.Dictionary import PARTICLE_DICT
from Misc.Functions import ROUND
from Misc.Position_Fcts import in_all_bounds
from itertools import product

from System.SystemClass import SYSTEM


Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

ROUNDDIGIT = Global_variables.ROUNDDIGIT
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
L_FCT = Global_variables.L_FCT


def quad_equ(a, b, c, bound_start, bound_end):
    Delta2 = b**2 - 4 * a * c

    def U_to_T_transf(u_val):
        return bound_start + u_val * (bound_end - bound_start)

    if Delta2 == 0:
        Uo = -b / (2 * a)
        to = U_to_T_transf(Uo)
        if bound_start <= to <= bound_end:
            return [to]
    elif Delta2 > 0:
        sols = []
        Delta = Delta2**0.5
        for i in range(2):
            Uo = (-b + ((-1) ** i) * Delta) / (2 * a)
            to = U_to_T_transf(Uo)
            if bound_start <= to <= bound_end:
                sols.append(to)
        return sols
    else:
        return []


def MINIMISE(t, a1, b1, a2, b2, Dist1, Dist2, Tstart, Tend, p1, id1, p2, id2):
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

    # Calculate the difference of the slopes and intercepts of the two lines

    Tmini = np.inf
    Xo = np.array([])
    # Tstart, Tend = t - dt, t
    X1_ini, X1_fin = a1 * Tstart + b1, a1 * Tend + b1
    X2_ini, X2_fin = a2 * Tstart + b2, a2 * Tend + b2

    Dini = np.linalg.norm(X1_ini - X2_ini)
    if Dini < (Dist1 + Dist2):
        R = Dist1 / (Dist1 + Dist2)
        Xoini = X1_ini + (X2_ini - X1_ini) * R
        return (Tstart, Xoini)
    Dfin = np.linalg.norm(X1_fin - X2_fin)

    if Dfin <= (Dist1 + Dist2):
        AB = X2_ini - X1_ini
        vab = a2 - a1
        rab = Dist1 + Dist2

        # quad formula A*u^2 +B*u+C
        A = np.dot(vab, vab)  # np.linalg.norm(a2 - a1) ** 2
        if A == 0 and Dfin < (Dist1 + Dist2):
            R = Dist1 / (Dist1 + Dist2)
            Xo = X1_ini + (X2_ini - X1_ini) * R
            return (Tstart, Xo)
        B = 2 * np.dot(vab, AB)  # (a2 - a1), (X2_ini - X1_ini))
        C = np.dot(AB, AB) - rab**2

        qfct = lambda x: A * x**2 + B * x + C
        Tsols = quad_equ(A, B, C, Tstart, Tend)

        for i in range(len(Tsols)):
            Tmini = min(Tsols)
            # Xo = (Tmini * a1 + b1 + Tmini * a2 + b2) / 2
            x1 = Tmini * a1 + b1
            x2 = Tmini * a2 + b2
            R = Dist1 / (Dist1 + Dist2)
            Xoi = x1 + (x2 - x1) * R
            Xoi = np.array(Xo, dtype=float)
            Xoi = np.round(Xo, decimals=ROUNDDIGIT)
            # Global_variables.COLPTS.append([Tmini, xo, 3])
            Dt = np.linalg.norm(x1 - x2)
            if Dt < (Dist1 + Dist2):
                return (Tmini, Xoi)
            # print("dmiss", Dt, Dfin, (Dist1 + Dist2))
            Tsols.remove(Tmini)
        # else:
        # particle1 = SYSTEM.Get_Particle(p1[1], p1[0], id1)
        # particle2 = SYSTEM.Get_Particle(p2[1], p2[0], id2)
        # print(X1_ini, particle1.X, particle1.X + dt * particle1.V, X1_fin)
        # print(X2_ini, particle2.X, particle2.X + dt * particle2.V, X2_fin)
        # print("miss", Tstart, Tend, len(Tsols), Dini <= (Dist1 + Dist2), "\n")

    return (Tmini, Xo)


def INTERCHECK_ND(a1, b1, p1, a2, b2, p2, t, z1, z2, Tstart, Tend, id1, id2):
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

    Dist1, Dist2 = (
        PARTICLE_DICT[PARTICLE_NAMES[p1[1]]]["size"] / 2,
        PARTICLE_DICT[PARTICLE_NAMES[p2[1]]]["size"] / 2,
    )
    tmini, xo = MINIMISE(
        t, a1, b1, a2, b2, Dist1, Dist2, Tstart, Tend, p1, id1, p2, id2
    )
    if xo.size > 0:
        return [COLTYPE(p1, p2), tmini, xo, z1, z2]
    else:
        # If the lines are parallel or do not intersect during the time interval, return [0]
        return [0]


def INTERCHECK_1D(a1, b1, p1, a2, b2, p2, t, z1, z2, Tstart, Tend, id1, id2):
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
