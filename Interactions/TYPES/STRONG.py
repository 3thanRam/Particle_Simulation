import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from Particles.Global_Variables import Global_variables
import System.SystemClass

Numb_of_TYPES = len(PARTICLE_DICT)
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt

# STRONG FORCE PARAMS#
LO_strg = PARTICLE_DICT["up_Quark"]["size"] * 6
L_strong_cutoff = LO_strg * 2
k_strong = 5 / ((L_strong_cutoff - LO_strg))


def COLOR_ERROR(Colour_group):
    raise ValueError("Invalid Quark grouping", len(Colour_group), Colour_group)


def STRONG_FORCE_GROUP(TotnumbAllpart):
    SYSTEM = System.SystemClass.SYSTEM
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
