import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from Particles.Global_Variables import Global_variables
import System.SystemClass as System_module

Numb_of_TYPES = len(PARTICLE_DICT)
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt

# STRONG FORCE PARAMS#
LO_strg = PARTICLE_DICT["up_Quark"]["size"] * 6
L_strong_cutoff = LO_strg * 2
k_strong = 5 / ((L_strong_cutoff - LO_strg))
rng = np.random.default_rng()
Emax = 10**4


def COLOR_ERROR(Colour_group):
    raise ValueError("Invalid Quark grouping", len(Colour_group), Colour_group)


def STRONG_FORCE_GROUP(TotnumbAllpart):
    dtype = [("Pos", float), ("TypeID0", int), ("TypeID1", int), ("index", int)]
    Xarray = np.array(
        [
            [
                (s.X[d], s.parity[0], s.parity[1], s.ID)
                for s in System_module.SYSTEM.Particles_List
            ]
            for d in range(DIM_Numb)
        ],
        dtype=dtype,
    )
    loc_arr = np.array(
        [[Xiarray["Pos"][i] for Xiarray in Xarray] for i in range(len(Xarray[0]))]
    )
    DELTA = (loc_arr.T[..., np.newaxis] - loc_arr.T[:, np.newaxis]).T
    distances = np.linalg.norm(DELTA, axis=-1)
    strong_charge_matrix = np.array(
        [s.Strong_Charge for s in System_module.SYSTEM.Particles_List]
    )
    vel_matrix = np.array(
        [np.linalg.norm(s.V) for s in System_module.SYSTEM.Particles_List]
    )
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
    """
    Creates particles from strong force based on probability weighted by their energies

    Args:
        t (float): current time of simulation

    """
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

    Quark_Numb = System_module.SYSTEM.Quarks_Numb
    TotnumbAllpart = System_module.SYSTEM.Tot_Numb
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
        NewpartName = QUARK_SPONT_NAMES[nameind]

        parity1, id1, X1 = (
            System_module.SYSTEM.Particles_List[ParentSystindex1].parity,
            System_module.SYSTEM.Particles_List[ParentSystindex1].ID,
            np.array(System_module.SYSTEM.Particles_List[ParentSystindex1].X),
        )
        PartOrAnti1, ind1 = parity1

        parity2, id2, X2 = (
            System_module.SYSTEM.Particles_List[ParentSystindex2].parity,
            System_module.SYSTEM.Particles_List[ParentSystindex2].ID,
            np.array(System_module.SYSTEM.Particles_List[ParentSystindex2].X),
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
            System_module.SYSTEM.Add_Particle(Typeindex, i, CREATEPARAMS[i])
