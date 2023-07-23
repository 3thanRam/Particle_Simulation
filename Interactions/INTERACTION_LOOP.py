import numpy as np
import System.SystemClass
from itertools import combinations

from Particles.Global_Variables import Global_variables

if Global_variables.DIM_Numb == 1:
    from Interactions.INTERACTION_CHECK import INTERCHECK_1D

    INTERCHECK = INTERCHECK_1D
else:
    from Interactions.INTERACTION_CHECK import INTERCHECK_ND

    INTERCHECK = INTERCHECK_ND

BOUNDARY_COND = Global_variables.BOUNDARY_COND
if BOUNDARY_COND == 0:
    from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_PER

    BOUNDARY_FCT = BOUNDARY_FCT_PER
else:
    from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_HARD

    BOUNDARY_FCT = BOUNDARY_FCT_HARD

from Interactions.INTERACTION_DEF import COLTYPE
from Interactions.TYPES.ANNIHILATION import ANNIHILATE
from Interactions.TYPES.COLLISION import COLLIDE

# from Interactions.TYPES.ABSORPTION import ABSORBE
from Misc.Functions import COUNTFCT


dt = Global_variables.dt


def timestat_end(B1, B2, t, z1, z2, t1PARA, t2PARA):
    """
    Computes the start and end times for a given time t and its corresponding
    indices z1 and z2 within the time arrays t1PARA and t2PARA, respectively.
    These indices are used to obtain the start and end times tstart and tend for
    the corresponding segments of the signal.

    Args:
    - B1: numpy.ndarray. First signal array.
    - B2: numpy.ndarray. Second signal array.
    - t: float. Time value.
    - z1: int. Index of t in the first signal time array t1PARA.
    - z2: int. Index of t in the second signal time array t2PARA.
    - t1PARA: numpy.ndarray. First signal time array.
    - t2PARA: numpy.ndarray. Second signal time array.

    Returns:
    A tuple of floats (tstart, tend) corresponding to the start and end times for
    the segments of the signals B1 and B2 that correspond to the time t.
    """

    # Determine the start and end times for the segments of the signal B1
    # that correspond to the time t.
    if z1 != 0:
        tstart1 = t1PARA[z1]
    if z1 != len(B1) - 1:
        tend1 = t1PARA[z1 + 1]

    # Determine the start and end times for the segments of the signal B2
    # that correspond to the time t.
    if z2 != 0:
        tstart2 = t2PARA[z2]
    if z2 != len(B2) - 1:
        tend2 = t2PARA[z2 + 1]

    # Determine the start time tstart for the corresponding segments of B1 and B2.
    if z1 == 0:
        if z2 == 0:
            tstart = t - dt
        else:
            tstart = tstart2
    else:
        if z2 == 0:
            tstart = tstart1
        else:
            tstart = max(tstart1, tstart2)

    # Determine the end time tend for the corresponding segments of B1 and B2.
    if z1 == len(B1) - 1:
        if z2 == len(B2) - 1:
            tend = t
        else:
            tend = tend2
    else:
        if z2 == len(B2) - 1:
            tend = tend1
        else:
            tend = min(tend1, tend2)

    return (tstart, tend)


def Interaction_Loop_Check(Xf, t, GroupList):
    """
    This function simulates the collisions between particles in the system between time instants t-dt and t

    Parameters:
        Xf (list): List of particle positions and id parameters that compose the system.
        t (float): The current time.
        GroupList(list): list of particles grouped together based on close positions to each other or boundaries

    Returns:
        updated state of the system (Xf).
    """
    kill_list = []
    INTERACT_HIST = []
    NumbDone, NumbCols = 0, -1
    while NumbCols != 0:
        NumbCols = 0
        INTERACT_HIST = []  # gather history of Collisions
        INTERACT_SEARCH = []  # list with less info for faster search

        mintime = np.inf
        for GroupID, Group in enumerate(GroupList):
            for I1, I2 in combinations(range(len(Group)), 2):
                particle1, particle2 = Group[I1], Group[I2]
                p1, id1 = particle1
                p2, id2 = particle2
                P1 = System.SystemClass.SYSTEM.Get_Particle(p1[1], p1[0], id1)
                P2 = System.SystemClass.SYSTEM.Get_Particle(p2[1], p2[0], id2)
                A1, B1, t1params, p1, id1 = P1.Coef_param_list[:5]
                A2, B2, t2params, p2, id2 = P2.Coef_param_list[:5]

                if COLTYPE(p1, p2) == 0 or (id1 == id2 and p1 == p2):
                    continue
                for z1, b1 in enumerate(B1):
                    if isinstance(b1[0], str):
                        continue
                    a1 = BOUNDARY_FCT(A1, p1, id1, z1)
                    for z2, b2 in enumerate(B2):
                        if isinstance(b2[0], str):
                            continue
                        a2 = BOUNDARY_FCT(A2, p2, id2, z2)
                        tstart, tend = timestat_end(
                            B1, B2, t, z1, z2, t1params, t2params
                        )
                        INTER = INTERCHECK(
                            a1, b1, p1, a2, b2, p2, t, z1, z2, tstart, tend
                        )
                        Collisiontype = INTER[0]
                        if Collisiontype != 0:
                            if Collisiontype == 2 or Collisiontype == 1:
                                if INTER[1] < mintime:
                                    mintime = INTER[1]
                            if INTER[1] > mintime:
                                continue
                            colinfo = [
                                INTER[1],
                                INTER[2],
                                Collisiontype,
                                z1,
                                z2,
                                p1,
                                id1,
                                p2,
                                id2,
                            ]
                            searchinfo = [INTER[1], INTER[2], p1, id1, p2, id2]

                            Clptinfo = [INTER[1], INTER[2], Collisiontype]
                            if (
                                COUNTFCT(INTERACT_SEARCH, searchinfo) == 0
                                and COUNTFCT(Global_variables.COLPTS, Clptinfo) == 0
                            ):
                                NumbCols += 1
                                INTERACT_HIST.append(colinfo)
                                INTERACT_SEARCH.append(searchinfo)

        INTERACT_HIST = sorted(INTERACT_HIST, key=lambda x: x[0])

        for Interaction in INTERACT_HIST:
            if Interaction[2] == 3:
                if COUNTFCT(Global_variables.COLPTS, Interaction[:3]) == 0:
                    Xf = COLLIDE(Interaction, Xf, GroupList, t)
                    NumbDone += 1
                    break
            elif Interaction[2] == 2:
                if COUNTFCT(kill_list, Interaction[:3]) == 0:
                    kill_list.append(Interaction[:3])
                    Xf, GroupList = ANNIHILATE(Interaction, Xf, GroupList, t)
                    NumbDone += 1
                    break
    return Xf
