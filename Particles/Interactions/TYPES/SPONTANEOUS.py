import numpy as np

rng = np.random.default_rng()
from itertools import product, permutations
from collections import Counter

from Particles.Dictionary import PARTICLE_DICT
from Particles.ParticleClass import Particle
from Particles.Global_Variables import Global_variables

Numb_of_TYPES = len(PARTICLE_DICT)

C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt


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


def COLOR_ERROR(Colour_group):
    raise ValueError("Invalid Quark grouping", len(Colour_group), Colour_group)


# STRONG FORCE PARAMS#
LO_strg = PARTICLE_DICT["up_Quark"]["size"] * 6  # L_strong_cutoff/2

L_strong_cutoff = LO_strg * 2

k_strong = 5 / ((L_strong_cutoff - LO_strg))
# k_strong = 2.3 / (dt * 0.05 * (L_strong_cutoff - LO_strg))


def STRONG_FORCE_GROUP(TotnumbAllpart):
    from Particles.SystemClass import SYSTEM

    dtype = [("Pos", float), ("TypeID0", int), ("TypeID1", int), ("index", int)]
    Xarray = np.array(
        [
            [(s.X[d], s.parity[0], s.parity[1], s.ID) for s in SYSTEM.Particles_List]
            for d in range(DIM_Numb)
        ],
        dtype=dtype,
    )
    loc_arr = np.array(
        [[Xiarray["Pos"][i] for Xiarray in Xarray] for i in range(len(Xarray[0]))]
    )
    DELTA = (loc_arr.T[..., np.newaxis] - loc_arr.T[:, np.newaxis]).T
    distances = np.linalg.norm(DELTA, axis=-1)
    strong_charge_matrix = np.array([s.Strong_Charge for s in SYSTEM.Particles_List])
    vel_matrix = np.array([np.linalg.norm(s.V) for s in SYSTEM.Particles_List])
    BIG_vel_matrix = np.outer(vel_matrix, vel_matrix)

    Strong_DIST = np.where(
        (0 != distances) & (np.abs(distances) < L_strong_cutoff), distances, 0
    ) * np.outer(strong_charge_matrix, strong_charge_matrix)

    def SUBLIST_SEARCH(LIST, element):
        for sb_ind, SUBLIST in enumerate(LIST):
            if element in SUBLIST:
                return sb_ind
        return -1

    COLOR_GROUP = []
    for I_ind, distI in enumerate(Strong_DIST):
        for J_ind, distJ in enumerate(distI):
            if I_ind == J_ind or distJ == 0:
                continue

            SubgroupIndi = SUBLIST_SEARCH(COLOR_GROUP, I_ind)
            SubgroupIndj = SUBLIST_SEARCH(COLOR_GROUP, J_ind)

            if SubgroupIndi != -1:
                if SubgroupIndj == -1:
                    COLOR_GROUP[SubgroupIndi].append(J_ind)
            elif SubgroupIndj != -1:
                COLOR_GROUP[SubgroupIndj].append(I_ind)
            else:
                COLOR_GROUP.append([I_ind, J_ind])

    STRONG_FORCE = np.zeros((TotnumbAllpart, TotnumbAllpart))

    NEW_COLOR_GROUP = []

    def QUARK_SEPERATOR(Quark_Group, Quark_Numb, RANGE_ARG):
        start = 0
        for c in range(RANGE_ARG):
            end = Quark_Numb * (c + 1)
            QUARK_ids = Quark_Group[start:end]
            NEW_COLOR_GROUP.append(QUARK_ids)
            start = end

    def QUARK_REGROUP(Numb_Meson, Numb_3Baryon, Quark_Group):
        # COL_LIST=[SystemList[quarkId].Colour_Charge for quarkId in Quark_Group]
        # ADD_GLUONS=[]#gluons added to maintain colour neutrality
        # id_done_list=[]
        NEW_Quark_Group = []

        if Numb_Meson != 0:
            NEW_Quark_Group = Quark_Group[0 : 2 * Numb_Meson]
            # NEW_Quark_Group.extend(HADRON_SPLIT(COL_LIST,Quark_Group,2,Numb_Meson))
            if Numb_3Baryon != 0:
                NEW_Quark_Group.extend(
                    Quark_Group[2 * Numb_Meson : 2 * Numb_Meson + 3 * Numb_3Baryon]
                )
                # NEW_Quark_Group.extend(HADRON_SPLIT(COL_LIST,Quark_Group,3,Numb_3Baryon))
        else:
            NEW_Quark_Group = Quark_Group
            # NEW_Quark_Group.extend(HADRON_SPLIT(COL_LIST,Quark_Group,3,Numb_3Baryon))
        if len(NEW_Quark_Group) != len(Quark_Group):
            COLOR_ERROR(Quark_Group)
        # split quark group into given numb of mesons and baryons according to colour neutrality and gluon colour rules
        return NEW_Quark_Group

    for Col_group in COLOR_GROUP:
        QUARK_SEP_PARMS = []
        if len(Col_group) <= 3:
            if len(Col_group) > 1:
                QUARK_SEP_PARMS.append([Col_group, len(Col_group), 1])
            else:
                COLOR_ERROR(Col_group)
        elif len(Col_group) % 3 == 0:
            Num3baryons = int(len(Col_group) / 3)
            NumMesons = 0
            Neutral_Group = QUARK_REGROUP(NumMesons, Num3baryons, Col_group)

            QUARK_SEP_PARMS.append([Neutral_Group, 3, Num3baryons])
        elif len(Col_group) % 3 == 2:
            Num3baryons = int(len(Col_group) / 3)
            NumMesons = 1
            Neutral_Group = QUARK_REGROUP(NumMesons, Num3baryons, Col_group)

            QUARK_SEP_PARMS.append([Neutral_Group[:2], 2, NumMesons])
            QUARK_SEP_PARMS.append([Neutral_Group[2:], 3, Num3baryons])
        elif len(Col_group) % 2 == 1:
            Num3baryons = 1
            NumMesons = int(len(Col_group) / 2) - 1
            Neutral_Group = QUARK_REGROUP(NumMesons, Num3baryons, Col_group)

            QUARK_SEP_PARMS.append([Neutral_Group[:-3], 2, NumMesons])
            QUARK_SEP_PARMS.append([Neutral_Group[-3:], 3, Num3baryons])
        elif len(Col_group) % 2 == 0:
            Num3baryons = 0
            NumMesons = int(len(Col_group) / 2)
            Neutral_Group = QUARK_REGROUP(NumMesons, Num3baryons, Col_group)

            QUARK_SEP_PARMS.append([Neutral_Group, 2, NumMesons])
        else:
            COLOR_ERROR(Col_group)

        for Group_Info, k, rangeParameters in QUARK_SEP_PARMS:
            QUARK_SEPERATOR(Group_Info, k, rangeParameters)

    COLOR_GROUP = NEW_COLOR_GROUP
    for Col_group in COLOR_GROUP:
        for quark_ind_A in Col_group:
            for quark_ind_B in Col_group:
                if quark_ind_A == quark_ind_B:
                    continue
                DIST = distances[quark_ind_A, quark_ind_B]
                STRONG_FORCE[quark_ind_A][quark_ind_B] = -k_strong * (DIST - LO_strg)
    return (STRONG_FORCE, BIG_vel_matrix)


def SpontaneousEvents(t):
    Emax = 10**4
    ##########
    # ELM######
    ##########
    # PARTICLE_SPONT_NAMES=['electron','muon','tau']
    PARTICLE_SPONT_NAMES = [
        "electron",
        "muon",
        "tau",
        "up_Quark",
        "down_Quark",
        "strange_Quark",
        "charm_Quark",
        "bottom_Quark",
        "top_Quark",
    ]
    EcreationMinList = [
        2 * PARTICLE_DICT[partName]["mass"] * C_speed**2
        for partName in PARTICLE_SPONT_NAMES
    ]

    RemoveTypeInd = PARTICLE_DICT["photon"]["index"]

    from Particles.SystemClass import SYSTEM

    PHOTON_LIST = [part for part in SYSTEM.Particles_List if part.name == "photon"]

    PHOTON_ID_LIST = [photon.ID for photon in PHOTON_LIST]
    Energy_List = np.array([photon.Energy for photon in PHOTON_LIST])

    Energy_List_CUT = np.where(Energy_List > Emax, Emax, Energy_List)
    GEN_INFO = []
    GEN_INFO_search = []
    # EcreationMin=2*C_speed**2

    for Eminindex, Ecreatmini in enumerate(EcreationMinList):
        P_list = np.where(
            Energy_List_CUT > Ecreatmini,
            np.exp(Energy_List_CUT / (Ecreatmini * Emax)) - 1,
            0,
        )
        # P_list=[pi[0] for pi in Prob_Array]
        P_list = np.where(P_list > 0.9, 0.9, np.where(P_list < 0, 0, P_list))
        if np.trim_zeros(P_list).size > 0:
            Gen = np.array([rng.choice([0, 1], p=[1 - pi, pi]) for pi in P_list])
            GEN = list(np.where(Gen == 1)[0])
            for indGen in GEN:
                if indGen not in GEN_INFO_search:
                    GEN_INFO_search.append(indGen)
                    GEN_INFO.append([Eminindex, indGen])

    GEN_INFO.sort(key=lambda x: x[1], reverse=True)
    for nameind, killind in GEN_INFO:
        NewpartName = PARTICLE_SPONT_NAMES[nameind]  #'electron'

        # PosCenter = Global_variables.SYSTEM[RemoveTypeInd][0][killind].X
        Particle_Kill = SYSTEM.Get_Particle(RemoveTypeInd, 0, PHOTON_ID_LIST[killind])
        PosCenter = Particle_Kill.X
        Global_variables.COLPTS.append([t, PosCenter, 4])

        Posdeviation = min(
            np.min(np.abs(Global_variables.L - PosCenter)),
            np.min(np.abs(Global_variables.Linf - PosCenter)),
        ) / (10 * max(abs(Global_variables.L - Global_variables.Linf)))
        PosParam1 = PosCenter - Posdeviation
        PosParam2 = PosCenter + Posdeviation

        Energyval = Energy_List[killind] / 2
        Vparam = Particle_Kill.V

        photDirection = Vparam / np.linalg.norm(Vparam)
        NEWMASS = PARTICLE_DICT[NewpartName]["mass"]
        VParam = (photDirection / (C_speed * NEWMASS)) * (
            (Energyval**2 - NEWMASS**2 * C_speed**4) / (C_speed**2 * NEWMASS)
        ) ** 0.5  # V=p/m=pc/mc=direction*sqrt(Etot**2-Emass**2)/mc
        if DIM_Numb == 3:
            theta = 45 * np.pi / 180
            VParam1 = np.round(
                rotate_quat(angle_axis_quat(theta, photDirection), VParam), decimals=10
            )
            VParam2 = np.round(
                rotate_quat(angle_axis_quat(-theta, photDirection), VParam), decimals=10
            )
        elif DIM_Numb == 2:
            theta = 45 * np.pi / 180
            VParam1 = np.matmul(ROT2D(theta), VParam)
            VParam2 = np.matmul(ROT2D(-theta), VParam)
        else:
            VParam1, VParam2 = VParam, -VParam
        CREATEparam1 = "Spont_Create", PosParam1, VParam1, t, Energyval
        CREATEparam2 = "Spont_Create", PosParam2, VParam2, t, Energyval
        CREATEPARAMS = [CREATEparam1, CREATEparam2]
        # remove photon
        SYSTEM.Remove_particle(RemoveTypeInd, 0, PHOTON_ID_LIST[killind])
        # Global_variables.SYSTEM[RemoveTypeInd][0].pop(killind)
        Global_variables.Ntot[RemoveTypeInd][0] -= 1

        Typeindex = PARTICLE_DICT[NewpartName]["index"]

        if PARTICLE_DICT[NewpartName]["Strong_Charge"] != 0:
            COLOUR = [[-1, 0, 0], [1, 0, 0]]
        else:
            COLOUR = [None, None]

        for i in range(2):
            SYSTEM.Add_Particle(Typeindex, i, CREATEPARAMS[i])
            """Global_variables.SYSTEM[Typeindex][i].append(
                Particle(
                    name=NewpartName,
                    parity=(i, Typeindex),
                    ID=idi,
                    ExtraParams=CREATEPARAMS[i],
                    Colour_Charge=COLOUR[i],
                )
            )"""
            # SYSTEM.TRACKING[Typeindex][i][idi].insert(0, [t, PosCenter])
    ##############
    # STRONG FORCE#
    ##############

    Quark_ind_LIST = [6, 7, 8, 9, 10, 11]
    QUARK_SPONT_NAMES = [
        "up_Quark",
        "down_Quark",
        "strange_Quark",
        "charm_Quark",
        "bottom_Quark",
        "top_Quark",
    ]
    EcreateLIST = [
        2 * PARTICLE_DICT[partName]["mass"] * C_speed**2
        for partName in QUARK_SPONT_NAMES
    ]

    Quark_Numb = SYSTEM.Quarks_Numb
    TotnumbAllpart = SYSTEM.Tot_Numb
    if Quark_Numb == 0:
        Global_variables.S_Force = np.zeros((TotnumbAllpart, TotnumbAllpart))
        return 0

    STRONG_FORCE, BIG_vel_matrix = STRONG_FORCE_GROUP(TotnumbAllpart)

    SPRING_ENERGY = np.abs(STRONG_FORCE * BIG_vel_matrix * dt)

    SPRING_ENERGY_CUT = np.where(SPRING_ENERGY > Emax, Emax, SPRING_ENERGY)

    GEN_INFO = []
    GEN_INFO_search = []

    for Eminindex, Ecreatei in enumerate(EcreateLIST):
        P_list = np.where(
            SPRING_ENERGY_CUT > Ecreatei,
            np.exp(SPRING_ENERGY_CUT / (Ecreatei * Emax)) - 1,
            0,
        )
        # if np.isinf(P_list).any():
        P_list = np.where(P_list > 1, 0.99, np.where(P_list < 0, 0, P_list))

        if np.trim_zeros(P_list.flatten()).size > 0:
            Nonzero_inds = np.where(P_list != 0)
            for NonzeroNumb in range(len(Nonzero_inds[0])):
                quark1_ind, quark2_ind = (
                    Nonzero_inds[0][NonzeroNumb],
                    Nonzero_inds[1][NonzeroNumb],
                )
                if (quark1_ind, quark2_ind) in GEN_INFO_search:
                    continue
                pi = P_list[quark1_ind, quark2_ind]
                Rand = rng.choice([0, 1], p=[1 - pi, pi])

                if Rand == 1:
                    GEN_INFO_search.append((quark1_ind, quark2_ind))
                    GEN_INFO.append([Eminindex, quark1_ind, quark2_ind])

    # GEN_INFO.sort(key=lambda x:x[1],reverse=True)
    for GENnumb, (nameind, ParentSystindex1, ParentSystindex2) in enumerate(GEN_INFO):
        NewpartName = QUARK_SPONT_NAMES[nameind]  # eg 'electron'

        parity1, id1, X1 = (
            SYSTEM.Particles_List[ParentSystindex1].parity,
            SYSTEM.Particles_List[ParentSystindex1].ID,
            np.array(SYSTEM.Particles_List[ParentSystindex1].X),
        )
        PartOrAnti1, ind1 = parity1

        parity2, id2, X2 = (
            SYSTEM.Particles_List[ParentSystindex2].parity,
            SYSTEM.Particles_List[ParentSystindex2].ID,
            np.array(SYSTEM.Particles_List[ParentSystindex2].X),
        )
        PartOrAnti2, ind2 = parity2

        PosCenter = 0.5 * (X1 + X2)

        Global_variables.COLPTS.append([t, PosCenter, 4])

        Posdeviation = min(
            np.min(np.abs(Global_variables.L - PosCenter)),
            np.min(np.abs(Global_variables.Linf - PosCenter)),
        ) / (10 * max(abs(Global_variables.L - Global_variables.Linf)))
        PosParam1 = PosCenter - Posdeviation
        PosParam2 = PosCenter + Posdeviation

        Energyval = SPRING_ENERGY[ParentSystindex1][ParentSystindex2] / 2

        dX = X1 - X2
        ParentDirection = dX / np.linalg.norm(dX)

        NEWMASS = PARTICLE_DICT[NewpartName]["mass"]

        VParam = (ParentDirection / (C_speed * NEWMASS)) * (
            (Energyval**2 - NEWMASS**2 * C_speed**4) / (C_speed**2 * NEWMASS)
        ) ** 0.5
        VParam1, VParam2 = VParam, -VParam
        CREATEparam1 = "Spont_Create", PosParam1, VParam1, t, Energyval
        CREATEparam2 = "Spont_Create", PosParam2, VParam2, t, Energyval
        CREATEPARAMS = [CREATEparam1, CREATEparam2]

        print("Strong Energy added", Energyval * 2)

        Typeindex = PARTICLE_DICT[NewpartName]["index"]
        for i in range(2):
            SYSTEM.Add_Particle(Typeindex, i, CREATEPARAMS[i])
