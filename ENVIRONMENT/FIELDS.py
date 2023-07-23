import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from Particles.Global_Variables import Global_variables
from Interactions.TYPES.SPONTANEOUS import STRONG_FORCE_GROUP
from Misc.Relativistic_functions import gamma_factor
from itertools import permutations

DIM_Numb = Global_variables.DIM_Numb
Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
C_speed = Global_variables.C_speed
dt = Global_variables.dt
####Natural Units#####
######################
# c, me, ħ, ε0,=1
E_cst = -1 / (4 * np.pi)  # *ε0
M_cst = 1 / (4 * np.pi)
# Grav_cst = E_cst * 10**-45
Fine_Struc_Cst = 0.0072973525628
charge_e = np.sqrt(4 * np.pi * Fine_Struc_Cst)


def gamma_factor(u):
    return 1 / np.sqrt(1 - (np.dot(u, u) / C_speed) ** 2)


def Gen_Field(Xarray, SystemList, Quark_Numb, TotnumbAllpart):
    Field_DICT = {}
    for i in range(len(Xarray[0])):
        xpos, i_type0, i_type1, i_id = Xarray[0][i]
        Field_DICT.update({(i_type0, i_type1, i_id): i})
    Global_variables.Field_DICT = Field_DICT

    loc_arr = np.array(
        [
            [Xarray[d]["Pos"][i] for d in range(DIM_Numb)] + [0] * (3 - DIM_Numb)
            for i in range(len(Xarray[0]))
        ]
    )
    charges = np.array([s.Elec_Charge for s in SystemList])
    # GRAV_matrix=Grav_cst*np.outer(mass_matrix, mass_matrix)
    velocity_matrix = np.array([s.V for s in SystemList])
    mass_matrix = np.array([s.M for s in SystemList])

    N = len(SystemList)
    E = np.zeros((N, 3))
    B = np.zeros((N, 3))

    # Non_zero_mass =
    for i, j in permutations((n for n in range(N) if mass_matrix[n] != 0), 2):
        dist_i = PARTICLE_DICT[PARTICLE_NAMES[Xarray[0]["TypeID1"][i]]]["size"] / 2
        dist_j = PARTICLE_DICT[PARTICLE_NAMES[Xarray[0]["TypeID1"][j]]]["size"] / 2
        rmin = dist_i + dist_j
        r = loc_arr[j] - loc_arr[i]
        r_norm = np.linalg.norm(r)
        if r_norm > rmin:
            v = velocity_matrix[i]
            q = charges[j]
            E[i] += q * r / r_norm**3
            if DIM_Numb == 2:
                gamma = gamma_factor(v)
                B[i] += np.array([-r[1], r[0], 0]) * q * gamma / r_norm**3
            elif DIM_Numb == 3:
                B[i] += np.cross(v, E[i]) / C_speed**2
    """for i in (n for n in range(N) if mass_matrix[n] != 0):
        dist_i = PARTICLE_DICT[PARTICLE_NAMES[Xarray[0]["TypeID1"][i]]]["size"] / 2
        for j in (n for n in range(N) if mass_matrix[n] != 0):
            dist_j = PARTICLE_DICT[PARTICLE_NAMES[Xarray[0]["TypeID1"][j]]]["size"] / 2
            rmin = dist_i + dist_j
            r = loc_arr[j] - loc_arr[i]
            r_norm = np.linalg.norm(r)
            if i != j and r_norm > rmin:
                v = velocity_matrix[i]
                q = charges[j]

                E[i] += q * r / r_norm**3
                if DIM_Numb == 2:
                    gamma = gamma_factor(v)
                    B[i] += np.array([-r[1], r[0], 0]) * q * gamma / r_norm**3
                elif DIM_Numb == 3:
                    B[i] += np.cross(v, E[i]) / C_speed**2"""

    # Construct the electromagnetic field tensor
    F = np.zeros((4, 4, len(SystemList)))

    for i in range(DIM_Numb):
        F[0, i + 1, :] = E[:, i]
        F[i + 1, 0, :] = -E[:, i]

    F[3, 2, :] = B[:, 0]  # Bx
    F[2, 3, :] = -B[:, 0]

    F[3, 1, :] = B[:, 1]  # By
    F[1, 3, :] = -B[:, 1]

    F[2, 1, :] = B[:, 2]  # Bz
    F[1, 2, :] = -B[:, 2]

    return F
