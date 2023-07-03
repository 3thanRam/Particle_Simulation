import numpy as np
import time
from itertools import combinations

from operator import itemgetter
import matplotlib.pyplot as plt

from Display import Density

DENS_FCT = Density.Denfct


####Natural Units#####
######################
E_cst = -1 / (4 * np.pi)
Grav_cst = E_cst * 10**-45
C_speed = 1
g = 1
Strong_cst = g**2 / (4 * np.pi * C_speed**2)
Fine_Struc_Cst = 0.0072973525628
charge_e = np.sqrt(4 * np.pi * Fine_Struc_Cst)


ROUNDDIGIT = 0


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

    Returns:
        _type_: _description_
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


def CHECK_time(elem, t, start_stop):
    """
    Check if a given element is within the time range specified by a current time t and a time window of length dt.
    If the element is an array, the function returns the minimum or maximum value of the elements that are within the time
    window depending on the value of the start_stop parameter.

    Args:
    elem: An array of elements or a float.
    t: A float representing the current time.
    start_stop: A string indicating whether to return the minimum or maximum value of elem within the time window.
        Should be 'start' or 'end'.

    Returns:
    A float representing the value of elem that is within the time window specified by t and dt.
    If elem is an array, returns either the minimum or maximum value of the elements that are within the time window
    depending on the value of the start_stop parameter. If there are no elements within the time window, returns the time
    at the start or end of the window depending on the value of the start_stop parameter.
    """
    if start_stop == "start":
        compare_fct = [max, np.max]
        t_extr = t - dt
    elif start_stop == "end":
        compare_fct = [min, np.min]
        t_extr = t

    if isinstance(elem, np.ndarray):
        elem = elem[(elem >= t - dt) & (elem <= t)]
        if elem.size > 1:
            elem = compare_fct[1](elem)
        elif elem.size == 1:
            elem = elem[0]
        else:
            elem = t_extr
    elif not isinstance(elem, float):
        elem = t_extr

    return elem


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
                        flipindex, flipvalue = Global_variables.Vflipinfo[
                            id_type[ind0][2]
                        ][id_type[ind0][1]][id_type[ind0][0]][r]
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


def COLLISIONS(F, t, CHG_particle_Params):
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


rng = np.random.default_rng()


sd = 0


def Gen_Field(Xarray, SystemList):
    Field_DICT = {}
    for i in range(len(Xarray[0])):
        xpos, i_type0, i_type1, i_id = Xarray[0][i]
        Field_DICT.update({(i_type0, i_type1, i_id): i})
    Global_variables.Field_DICT = Field_DICT

    loc_arr = np.array(
        [[Xiarray["Pos"][i] for Xiarray in Xarray] for i in range(len(Xarray[0]))]
    )
    charges = ((-1) ** (Xarray[0]["TypeID0"])) * np.array(
        [
            PARTICLE_DICT[PARTICLE_NAMES[index]]["charge"]
            for index in Xarray[0]["TypeID1"]
        ]
    )

    ELEC_matrix = E_cst * charge_e * np.outer(charges, charges)

    mass_matrix = np.array([s.M for s in SystemList])

    # GRAV_matrix=Grav_cst*np.outer(mass_matrix, mass_matrix)

    Tot_matrix = ELEC_matrix  # +GRAV_matrix

    DELTA = (loc_arr.T[..., np.newaxis] - loc_arr.T[:, np.newaxis]).T
    distances = np.linalg.norm(DELTA, axis=-1)

    Non_zero_mask = distances != 0  # to avoid divergences

    # Get direction of force
    unit_vector = np.zeros_like(DELTA.T)
    np.divide(DELTA.T, distances, out=unit_vector, where=Non_zero_mask)
    unit_vector = unit_vector.T

    EG_force = np.zeros_like(Tot_matrix)
    np.divide(Tot_matrix, distances**2, out=EG_force, where=Non_zero_mask)

    Quark_ind_LIST = [6, 7, 8, 9, 10, 11]

    SystemList = []
    SYST = Global_variables.SYSTEM
    TotnumbAllpart, Quark_Numb = 0, 0
    for S_ind in range(Numb_of_TYPES):
        SystemList += SYST[S_ind][0] + SYST[S_ind][1]
        TotnumbAllpart += len(SYST[S_ind][0]) + len(SYST[S_ind][1])
        if S_ind in Quark_ind_LIST:
            Quark_Numb += len(SYST[S_ind][0]) + len(SYST[S_ind][1])

    if Quark_Numb != 0:
        STRONG_FORCE, BIG_vel_matrix = STRONG_FORCE_GROUP(SystemList, TotnumbAllpart)
    else:
        STRONG_FORCE = np.zeros((TotnumbAllpart, TotnumbAllpart))
    force = EG_force + STRONG_FORCE

    acc_i = np.zeros((1, 100, 100))  # unit_vector* force
    acc_i = np.zeros_like(unit_vector.T * force) / np.ones_like(mass_matrix)
    maskM = mass_matrix != 0
    np.divide(unit_vector.T * force, mass_matrix, out=acc_i, where=maskM)

    acc = acc_i.T.sum(axis=1)

    return acc


def main(T, n1, n2, vo, l, Numb_Dimensions, BoundsCond, D, File_path_name=None):
    """
    Simulates the behavior of n1 particles and n2 antiparticles in a D dimensional box

    Args:
    T (float): The time range to simulate.
    D (int): The type of representation: draw just densities/0 or also draw trajectories/1
    n1 (int): The initial number of particles.
    n2 (int): The initial number of antiparticles.
    vo (float): The order of the velocities of the particles.
    l (float): The length of the box.
    Numb_Dimensions (int): The number of dimensions to simulate
    BoundsCond(int): The type of boundaries periodic/0 or hard/1
    File_path_name: Where to save the video if the simulation is 3D and D=1

    Returns:
    None: The function does not return anything, but it prints the total time of the simulation and
    draws the points if D=1.

    """

    global ROUNDDIGIT, Numb_of_TYPES, DIM_Numb, distmax, PARTICLE_DICT, PARTICLE_NAMES, Dist_min, INTERCHECK, BOUNDARY_FCT, BOUNDARY_COND, COLTYPE, ANNIHILATE, COLLIDE, ABSORBE, SpontaneousEvents, STRONG_FORCE_GROUP, L_FCT, Vmax, dt, Global_variables

    # import  as Global_var
    from Particles.Dictionary import PARTICLE_DICT

    Numb_of_TYPES = len(PARTICLE_DICT)
    PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

    BOUNDARY_COND = BoundsCond
    LO, LOinf = np.array([l[d] for d in range(Numb_Dimensions)]), np.array(
        [0 for d in range(Numb_Dimensions)]
    )
    L_FCT = [lambda x: LO + 0 * np.cos(x), lambda x: LOinf + 0 * np.sin(x + 1)]

    from Particles.Global_Variables import init

    init(Numb_Dimensions, BoundsCond, L_FCT)
    from Particles.Global_Variables import Global_variables

    from Particles.Interactions.TYPES.ANNIHILATION import ANNIHILATE
    from Particles.Interactions.TYPES.COLLISION import COLLIDE
    from Particles.Interactions.TYPES.ABSORPTION import ABSORBE
    from Particles.Interactions.TYPES.SPONTANEOUS import SpontaneousEvents
    from Particles.Interactions.TYPES.SPONTANEOUS import STRONG_FORCE_GROUP
    from Particles.Interactions.INTERACTION_DEF import COLTYPE

    ROUNDDIGIT = Global_variables.ROUNDDIGIT
    DIM_Numb = Global_variables.DIM_Numb
    Vmax = Global_variables.Vmax
    dt = Global_variables.dt
    distmax = 1.5 * Vmax * dt
    Dist_min = Global_variables.Dist_min
    INTERCHECK = Global_variables.INTERCHECK
    BOUNDARY_FCT = Global_variables.BOUNDARY_FCT
    L_FCT = Global_variables.L_FCT

    T = int(10 * T)
    time0 = time.time()  # Starting time for the simulation

    Vol = 1
    Dens = [[[], []] for num in range(Numb_of_TYPES)]
    L, Linf = L_FCT[0](0), L_FCT[1](0)
    for d in range(DIM_Numb):
        Vol *= L[d] - Linf[d]  # Volume of box
    for p in range(Numb_of_TYPES):
        Dens[p][0].append(Global_variables.Ntot[p][0] / Vol)
        Dens[p][1].append(Global_variables.Ntot[p][1] / Vol)
        # Initializing density values

    Dper = 0
    if D == 0 or D == 1:
        print("Generating Points")
        print(Dper, "%", end="\r")

    # Simulation loop

    ################
    dtype = [("Pos", float), ("TypeID0", int), ("TypeID1", int), ("index", int)]
    get = itemgetter(1, 2)
    get2 = itemgetter(3, 4, 5, 6)

    SpontaneousEvents(0)

    for ti in range(1, T):
        perc = int(100 * ti / T)
        if perc > Dper and D == 0 or D == 1:  # and perc%5==0
            Dper = perc
            print(Dper, "%", end="\r")

        t = ROUND(ti * dt)
        L, Linf, Lo, Loinf = (
            L_FCT[0](t),
            L_FCT[1](t),
            L_FCT[0](t - dt),
            L_FCT[1](t - dt),
        )

        Vsup, Vinf = (L - Lo) / dt, (Linf - Loinf) / dt
        Global_variables.Bound_Params = [Vsup, L - Vsup * t, Vinf, Linf - Vinf * t]

        Global_variables.Vflipinfo = [
            [
                [[] for ni in range(maxnumbpart + 1)],
                [[] for ni in range(maxnumbanti + 1)],
            ]
            for maxnumbpart, maxnumbanti in Global_variables.MaxIDperPtype
        ]

        SYST = []

        for PartTypegroup in Global_variables.SYSTEM:
            for PartOrAnti_group in PartTypegroup:
                if PartOrAnti_group:
                    SYST += PartOrAnti_group

        # ESYST_ini=[s.Energy for s in SYST]
        # print('Esyst_i',t,sum(ESYST_ini),'\n')
        Xi = np.array(
            [
                [(s.X[d], s.parity[0], s.parity[1], s.ID) for s in SYST]
                for d in range(DIM_Numb)
            ],
            dtype=dtype,
        )

        Global_variables.FIELD = Gen_Field(
            Xi, SYST
        )  # update electric field according to positions and charges of particles

        Xi.sort(order="Pos")
        # Initializing list to track particle positions at time t

        # Updating positions for all particles with info about id,position,type
        DOINFOLIST = np.array([s.DO(t) for s in SYST], dtype="object")

        Xf = np.array(
            [
                [
                    (
                        DOINFOLIST[s][0][d],
                        DOINFOLIST[s][1][0],
                        DOINFOLIST[s][1][1],
                        DOINFOLIST[s][2],
                    )
                    for s in range(len(SYST))
                ]
                for d in range(DIM_Numb)
            ],
            dtype=dtype,
        )
        Xf.sort(order="Pos")

        # [Xfin,p,id,Xinter,V,t_list,NZ]

        DO_TYPE_PARTorANTI = np.array([elem[1][0] for elem in DOINFOLIST])
        DO_TYPE_CHARGE = np.array([elem[1][1] for elem in DOINFOLIST])
        DO_INDEX = np.array([elem[2] for elem in DOINFOLIST])

        INI_TYPE_PARTorANTI = [
            np.array([elem[1] for elem in Xi[d]]) for d in range(DIM_Numb)
        ]
        INI_TYPE_CHARGE = [
            np.array([elem[2] for elem in Xi[d]]) for d in range(DIM_Numb)
        ]

        END_TYPE_PARTorANTI = [
            np.array([elem[1] for elem in Xf[d]]) for d in range(DIM_Numb)
        ]
        END_TYPE_CHARGE = [
            np.array([elem[2] for elem in Xf[d]]) for d in range(DIM_Numb)
        ]

        CHANGESdim = [
            np.where(
                (Xi[d]["index"] != Xf[d]["index"])
                | (INI_TYPE_PARTorANTI[d] != END_TYPE_PARTorANTI[d])
                | (INI_TYPE_CHARGE[d] != END_TYPE_CHARGE[d])
            )
            for d in range(DIM_Numb)
        ]
        if DIM_Numb == 1:
            CHANGES = CHANGESdim[0][0]
        elif DIM_Numb == 2:
            CHANGES = np.intersect1d(*CHANGESdim)
        elif DIM_Numb == 3:
            CHANGES = np.intersect1d(
                np.intersect1d(CHANGESdim[0], CHANGESdim[1]), CHANGESdim[2]
            )

        CHGind = []  # index in Xi/Xf

        # particle involved in interactions parameters
        Param_INTERPOS = []
        Param_POS = []
        Param_Velocity = []
        Param_Time = []
        Param_ID_TYPE = []
        Param_endtype = []

        for (
            chg
        ) in (
            CHANGES
        ):  #  particles of index between ini and end could interact with the particle
            CHGind.append([])
            Param_INTERPOS.append([])
            Param_POS.append([])
            Param_Velocity.append([])
            Param_Time.append([])
            Param_ID_TYPE.append([])
            Param_endtype.append([])
            for d in range(DIM_Numb):
                matchval = np.where(
                    (Xi[d]["index"][chg] == Xf[d]["index"])
                    & (INI_TYPE_PARTorANTI[d][chg] == END_TYPE_PARTorANTI[d])
                    & (INI_TYPE_CHARGE[d][chg] == END_TYPE_CHARGE[d])
                )[0][0]

                Xa, Xb = Xf[d]["Pos"][matchval], Xi[d]["Pos"][chg]
                if abs(Xa - Xb) <= distmax[d]:
                    mini = min(chg, matchval)
                    maxi = max(chg, matchval) + 1
                else:
                    # fast particles could "skip" the intermediate zone and not be seen
                    # we need to extend the range to account for this
                    Xinf = np.min((Xa, Xb))
                    Xsup = np.max((Xa, Xb))

                    Inflist = np.where(
                        (Xf[d]["Pos"] <= Xinf) & (Xf[d]["Pos"] >= Xsup - distmax[d])
                    )[0]
                    Suplist = np.where(
                        (Xf[d]["Pos"] >= Xsup) & (Xf[d]["Pos"] <= Xinf + distmax[d])
                    )[0]
                    if Suplist.size == 0:
                        maxi = max(chg, matchval) + 1
                    else:
                        maxi = Suplist[Xf[d]["Pos"][Suplist].argmax()] + 2
                    if Inflist.size == 0:
                        mini = min(chg, matchval)
                    else:
                        mini = Inflist[Xf[d]["Pos"][Inflist].argmin()] - 1

                for elem_ind in range(mini, maxi):
                    doindex = np.where(
                        (DO_INDEX == Xf[d]["index"][elem_ind])
                        & (DO_TYPE_PARTorANTI == Xf[d]["TypeID0"][elem_ind])
                        & (DO_TYPE_CHARGE == Xf[d]["TypeID1"][elem_ind])
                    )[0][0]

                    InterPos, Velocity, TimeParams, Endtype = get2(DOINFOLIST[doindex])
                    POS, TYPE0, TYPE1, ID = (
                        Xf[d]["Pos"][elem_ind],
                        Xf[d]["TypeID0"][elem_ind],
                        Xf[d]["TypeID1"][elem_ind],
                        Xf[d]["index"][elem_ind],
                    )
                    POSLIST = [[] for d in range(DIM_Numb)]
                    POSLIST[d] = POS
                    for d2 in range(1, DIM_Numb):
                        d2 += d
                        if d2 >= DIM_Numb:
                            d2 -= DIM_Numb
                        posind2 = np.where(
                            (Xf[d2]["index"] == ID)
                            & (Xf[d2]["TypeID0"] == TYPE0)
                            & (Xf[d2]["TypeID1"] == TYPE1)
                        )[0][0]
                        POSLIST[d2] = Xf[d2]["Pos"][posind2]
                    if (
                        CHGind[-1] == []
                        or COUNTFCT(Param_ID_TYPE[-1], [ID, TYPE0, TYPE1], 1) == 0
                    ):
                        CHGind[-1].append(elem_ind)
                        Param_INTERPOS[-1].append(InterPos)
                        Param_POS[-1].append(POSLIST)
                        Param_Velocity[-1].append(Velocity)
                        Param_Time[-1].append(TimeParams)
                        Param_ID_TYPE[-1].append(np.array([ID, TYPE0, TYPE1]))
                        Param_endtype[-1].append(Endtype)

            if len(Param_INTERPOS[-1]) == 0:
                CHGind.remove([])
                Param_INTERPOS.remove([])
                Param_POS.remove([])
                Param_Velocity.remove([])
                Param_Time.remove([])
                Param_ID_TYPE.remove([])
                Param_endtype.remove([])

        # if part interact with bounds then they interact with particles differently than those above
        if BOUNDARY_COND == 1:
            BOUNDARYCHECKS = [
                np.where(
                    (
                        Xi[:]["Pos"] < (Linf[:, np.newaxis] + Vmax[:, np.newaxis] * dt)
                    ).any(axis=0)
                )[0],
                np.where(
                    (Xi[:]["Pos"] > (L[:, np.newaxis] - Vmax[:, np.newaxis] * dt)).any(
                        axis=0
                    )
                )[0],
            ]
        else:
            BOUNDARYCHECKS = [
                np.where(
                    (Xi["Pos"] < (Linf[:, np.newaxis] + Vmax[:, np.newaxis] * dt)).any(
                        axis=0
                    )
                    | (Xi["Pos"] > (L[:, np.newaxis] - Vmax[:, np.newaxis] * dt)).any(
                        axis=0
                    )
                )[0]
            ]

        for Bcheck in BOUNDARYCHECKS:
            if len(Bcheck) > 0:
                CHGind.append([])
                Param_INTERPOS.append([])
                Param_POS.append([])
                Param_Velocity.append([])
                Param_Time.append([])
                Param_ID_TYPE.append([])
                Param_endtype.append([])
                for PART_B_inf_index in Bcheck:
                    for d in range(DIM_Numb):
                        elem_ind = np.where(
                            (Xf[d]["index"] == Xi[d]["index"][PART_B_inf_index])
                            & (
                                Xi[d]["TypeID0"][PART_B_inf_index]
                                == END_TYPE_PARTorANTI[d]
                            )
                            & (Xi[d]["TypeID1"][PART_B_inf_index] == END_TYPE_CHARGE[d])
                        )[0][0]
                        doindex = np.where(
                            (DO_INDEX == Xf[d]["index"][elem_ind])
                            & (DO_TYPE_PARTorANTI == Xf[d]["TypeID0"][elem_ind])
                            & (DO_TYPE_CHARGE == Xf[d]["TypeID1"][elem_ind])
                        )[0][0]
                        InterPos, Velocity, TimeParams, Endtype = get2(
                            DOINFOLIST[doindex]
                        )
                        POS, TYPE0, TYPE1, ID = (
                            Xf[d]["Pos"][elem_ind],
                            Xf[d]["TypeID0"][elem_ind],
                            Xf[d]["TypeID1"][elem_ind],
                            Xf[d]["index"][elem_ind],
                        )

                        POSLIST = [[] for d in range(DIM_Numb)]
                        POSLIST[d] = POS

                        for d2 in range(1, DIM_Numb):
                            d2 += d
                            if d2 >= DIM_Numb:
                                d2 -= DIM_Numb

                            posind2 = np.where(
                                (Xf[d2]["index"] == ID)
                                & (END_TYPE_PARTorANTI[d2] == TYPE0)
                                & (END_TYPE_CHARGE[d2] == TYPE1)
                            )[0][0]
                            POSLIST[d2] = Xf[d2]["Pos"][posind2]
                        CHGind[-1].append(elem_ind)
                        Param_INTERPOS[-1].append(InterPos)
                        Param_POS[-1].append(POSLIST)
                        Param_Velocity[-1].append(Velocity)
                        Param_Time[-1].append(TimeParams)
                        Param_ID_TYPE[-1].append(np.array([ID, TYPE0, TYPE1]))
                        Param_endtype[-1].append(Endtype)
        REMOVELIST = []

        PARAMS = [
            Param_INTERPOS,
            Param_POS,
            Param_Velocity,
            Param_Time,
            Param_ID_TYPE,
            Param_endtype,
        ]

        for I1, I2 in combinations(range(len(Param_ID_TYPE)), 2):
            if I1 in REMOVELIST or I2 in REMOVELIST:
                continue
            indtypeGroup1, indtypeGroup2 = Param_ID_TYPE[I1], Param_ID_TYPE[I2]
            A = np.array(indtypeGroup1)
            B = np.array(indtypeGroup2)
            nrows, ncols = A.shape
            dtypeO = {
                "names": ["f{}".format(i) for i in range(ncols)],
                "formats": ncols * [A.dtype],
            }

            Overlap, inda, indb = np.intersect1d(
                A.view(dtypeO), B.view(dtypeO), return_indices=True
            )
            Osize = Overlap.shape[0]
            if Osize != 0:
                A_s = A.shape[0]
                B_s = B.shape[0]

                if Osize == B_s:
                    REMOVELIST.append(I2)
                elif Osize == A_s:
                    REMOVELIST.append(I1)
                elif Osize >= 0.9 * B_s:
                    RANGE = list(np.arange(B_s))
                    indNotb = [indN for indN in RANGE if indN not in indb]
                    getB = itemgetter(*indNotb)
                    for parametr in PARAMS:
                        if len(indNotb) > 1:
                            parametr[I1].extend(getB(parametr[I2]))
                        else:
                            parametr[I1].append(getB(parametr[I2]))
                    REMOVELIST.append(I2)
                elif Osize >= 0.3 * A_s:
                    RANGE = list(np.arange(A_s))
                    indNota = [indN for indN in RANGE if indN not in inda]
                    getA = itemgetter(*indNota)
                    for parametr in PARAMS:
                        if len(indNota) > 1:
                            parametr[I2].extend(getA(parametr[I1]))
                        else:
                            parametr[I2].append(getA(parametr[I1]))
                    REMOVELIST.append(I1)

        REMOVELIST.sort(
            reverse=True
        )  # removing top to bottom to avoid index changes after each removal

        for removeind in REMOVELIST:
            for parametr in PARAMS:
                parametr.pop(removeind)
        Global_variables.DOINFOLIST = DOINFOLIST

        if len(Param_endtype) != 0:
            Xf = COLLISIONS(Xf, t, PARAMS)
            DOINFOLIST = Global_variables.DOINFOLIST
            DO_TYPE_PARTorANTI = np.array([elem[1][0] for elem in DOINFOLIST])
            DO_TYPE_CHARGE = np.array([elem[1][1] for elem in DOINFOLIST])
            DO_INDEX = np.array([elem[2] for elem in DOINFOLIST])

        # Update particle positions and track their movement
        for indUpdate in range(len(Xf[0])):  # dimension 0
            if not Xf[0][indUpdate]:
                continue
            pos_search = [Xf[0][indUpdate][0]]
            type_search = [Xf[0][indUpdate][1], Xf[0][indUpdate][2]]
            id_search = Xf[0][indUpdate][3]
            for d in range(1, DIM_Numb):
                Pos_d_index = np.where(
                    (Xf[d]["index"] == id_search)
                    & (Xf[d]["TypeID0"] == type_search[0])
                    & (Xf[d]["TypeID1"] == type_search[1])
                )[0][0]
                pos_search.append(Xf[d]["Pos"][Pos_d_index])

            Conv_search_Index = type_search[1]
            for s in Global_variables.SYSTEM[Conv_search_Index][type_search[0]]:
                if s.ID == id_search:
                    partORanti = s.parity[0]
                    s.X = pos_search
                    doindex = np.where(
                        (DO_INDEX == id_search)
                        & (DO_TYPE_PARTorANTI == type_search[0])
                        & (DO_TYPE_CHARGE == type_search[1])
                    )[0][0]
                    xinterargs, Velocity, targs, Endtype = get2(DOINFOLIST[doindex])
                    if Endtype > 0:
                        for nz in range(len(targs) - 1):
                            Global_variables.TRACKING[Conv_search_Index][partORanti][
                                id_search
                            ].extend(
                                [
                                    [targs[nz + 1], xinterargs[nz][0]],
                                    ["T", "X"],
                                    [targs[nz + 1], xinterargs[nz][1]],
                                ]
                            )
                        Global_variables.ALL_TIME.extend(targs[1:])
                    Global_variables.TRACKING[Conv_search_Index][partORanti][
                        id_search
                    ].append([t, pos_search])
                    break

        # Update the densities of particles
        if DIM_Numb == 1:
            Vol = L[0] - Linf[0]
        else:
            Vol = np.prod(L - Linf)
        for p in range(len(Global_variables.SYSTEM)):
            Dens[p][0].append(Global_variables.Ntot[p][0] / Vol)
            Dens[p][1].append(Global_variables.Ntot[p][1] / Vol)

        # SYST=[]
        # for PartTypegroup in Global_variables.SYSTEM:
        #    for PartOrAnti_group in PartTypegroup:
        #        if PartOrAnti_group:
        #            SYST+=PartOrAnti_group

        # ESYST_fin=[s.Energy for s in SYST]
        # print('Esyst_f',t,sum(ESYST_fin),'\n')
        if Global_variables.Ntot == [[0, 0] for p in range(Numb_of_TYPES)]:
            T = ti
            break
        if ti != T - 1:
            SpontaneousEvents(t)

    Dt = round(time.time() - time0, 2)

    # post-simulation operations depending on the input parameter D, which determines the type of output the function produces.

    if (
        D == 1
    ):  # if D is equal to 1, then the function produces a graphical output of the simulation results using the DRAW function.
        print("Generating time:", Dt, "s")
        import Display.DisplayDRAW as DRAW_TRAJ

        Global_variables.ALL_TIME.extend([ROUND(i * dt) for i in range(T)])
        Global_variables.ALL_TIME = [*set(Global_variables.ALL_TIME)]
        Global_variables.ALL_TIME.sort()
        DRAW_PARAMS = [
            T,
            dt,
            [Linf, L],
            DIM_Numb,
            Global_variables.COLPTS,
            Global_variables.TRACKING,
            Dens,
            L_FCT,
            BOUNDARY_COND,
            Global_variables.ALL_TIME,
            PARTICLE_DICT,
        ]
        if File_path_name != None:
            DRAW_PARAMS.append(File_path_name)
        DRAW_TRAJ.DRAW(*DRAW_PARAMS)
    elif (
        D == 0
    ):  # the function produces a density plot of the simulation results using the DENS_FCT function
        print("Total time:", Dt, "s")
        Trange = np.linspace(0, (T - 1) * dt, len(Global_variables.ALL_TIME))
        fig, ax = plt.subplots()
        DENS_FCT(DIM_Numb, Dens, Trange, ax, PARTICLE_DICT)
        plt.show()
    else:
        return Dt


if __name__ == "__main__":
    from Dialog.CHOICES import SET_PARAMETERS

    main(*SET_PARAMETERS())
