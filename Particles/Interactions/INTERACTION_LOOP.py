import numpy as np
from itertools import combinations

from Particles.Global_Variables import Global_variables

# INTERCHECK = Global_variables.INTERCHECK

# BOUNDARY_FCT = Global_variables.BOUNDARY_FCT
BOUNDARY_COND = Global_variables.BOUNDARY_COND

if Global_variables.DIM_Numb == 1:
    from Particles.Interactions.INTERACTION_CHECK import INTERCHECK_1D

    INTERCHECK = INTERCHECK_1D
    # Global_variables.INTERCHECK = INTERCHECK_1D
else:
    from Particles.Interactions.INTERACTION_CHECK import INTERCHECK_ND

    INTERCHECK = INTERCHECK_ND
    # Global_variables.INTERCHECK = INTERCHECK_ND
if BOUNDARY_COND == 0:
    from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_PER

    BOUNDARY_FCT = BOUNDARY_FCT_PER
    # Global_variables.BOUNDARY_FCT = BOUNDARY_FCT_PER
else:
    from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_HARD

    BOUNDARY_FCT = BOUNDARY_FCT_HARD
    # Global_variables.BOUNDARY_FCT = BOUNDARY_FCT_HARD

from Particles.Interactions.INTERACTION_DEF import COLTYPE
from Particles.Interactions.TYPES.ANNIHILATION import ANNIHILATE
from Particles.Interactions.TYPES.COLLISION import COLLIDE
from Particles.Interactions.TYPES.ABSORPTION import ABSORBE
from Misc.Functions import COUNTFCT


dt = Global_variables.dt


def GetCoefs(F, t):
    """Calculate coefficients of the interpolation polynomials for each particle.

    Args:
    - F (list): List of Fock coefficients for particles and antiparticles.

    Returns:
    - COEFS (list): List of coefficients of the interpolation polynomials for each particle.
                    Each element in the list is a list containing the following elements:
                    [a, b, ts, id, ends]

    The interpolation polynomials for each particle are defined as follows:
        f(t) = a * t + b[0]         if ends==0
        f(t) = a * t + b[0] + ... + b[ends] * (t - ts[1]) * ... * (t - ts[ends]) if ends>0
    """
    from Particles.SystemClass import SYSTEM

    COEFLIST = []
    (
        Param_INTERPOS,
        Param_POS,
        Param_Velocity,
        Param_Time,
        Param_ID_TYPE,
        Param_endtype,
    ) = F

    for coefgroup in range(len(Param_INTERPOS)):
        COEFS = []

        # Extract required values for each group
        xinters = np.array(
            [Interpos for Interpos in Param_INTERPOS[coefgroup]], dtype=object
        )
        xfs = np.array([posi for posi in Param_POS[coefgroup]], dtype=object)
        Vs = [vel for vel in Param_Velocity[coefgroup]]
        ts = np.array([tparams for tparams in Param_Time[coefgroup]], dtype=object)
        id_type = np.array([id_type for id_type in Param_ID_TYPE[coefgroup]])
        ends = np.array([endtype for endtype in Param_endtype[coefgroup]])

        for ind0 in range(len(id_type)):
            a = Vs[ind0]

            if ends[ind0] == 0:
                b = [xfs[ind0] - a * float(*ts[ind0])]
            else:
                b = []
                if BOUNDARY_COND == 0:
                    for r in range(ends[ind0]):
                        b.append(xinters[ind0][r][0] - a * float(ts[ind0][1 + r]))
                    b.append(xfs[ind0] - a * float(ts[ind0][0]))
                else:
                    A = np.copy(a)
                    for r in range(ends[ind0]):
                        b.append(xinters[ind0][r][0] - A * float(ts[ind0][1 + r]))
                        flipindex, flipvalue = SYSTEM.Vflipinfo[id_type[ind0][2]][
                            id_type[ind0][1]
                        ][id_type[ind0][0]][r]
                        A[flipindex] = flipvalue
                    b.append(xfs[ind0] - A * float(ts[ind0][0]))
            COEFS.append(
                [
                    a,
                    b,
                    ts[ind0],
                    [id_type[ind0][1], id_type[ind0][2]],
                    int(id_type[ind0][0]),
                    xinters[ind0],
                    ends[ind0],
                ]
            )
        COEFLIST.append(COEFS)
    return COEFLIST


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


def Interaction_Loop_Check(F, t, CHG_particle_Params):
    """
    This function simulates the collisions between particles in the system.

    Parameters:
        F (list): List of coefficients that describe the state of the system.
        t (float): The current time.

    Returns:
        updated state of the system (F).
    """
    GroupList = GetCoefs(CHG_particle_Params, t)

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
                A1, B1, t1params, p1, id1 = particle1[:5]
                A2, B2, t2params, p2, id2 = particle2[:5]
                if COLTYPE(p1, p2) == 0 or (id1 == id2 and p1 == p2):
                    continue
                for z1, b1 in enumerate(B1):
                    if isinstance(b1, str):
                        continue
                    a1 = BOUNDARY_FCT(A1, p1, id1, z1)
                    for z2, b2 in enumerate(B2):
                        if isinstance(b2, str):
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
                    F, GroupList = COLLIDE(Interaction, F, GroupList, t)
                    NumbDone += 1
            elif Interaction[2] == 2:
                if COUNTFCT(kill_list, Interaction[:3]) == 0:
                    kill_list.append(Interaction[:3])
                    F, GroupList = ANNIHILATE(Interaction, F, GroupList, t)
                    NumbDone += 1
                    break
            elif Interaction[2] == 1:
                if COUNTFCT(Global_variables.COLPTS, Interaction[:3]) == 0:
                    F, GroupList = ABSORBE(Interaction, F, GroupList, t)
                    NumbDone += 1
                    break

    return F
