import numpy as np

from Particles.Dictionary import PARTICLE_DICT

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

from Particles.Global_Variables import Global_variables
from Particles.Interactions.TYPES.SPONTANEOUS import STRONG_FORCE_GROUP


####Natural Units#####
######################
E_cst = -1 / (4 * np.pi)
Grav_cst = E_cst * 10**-45
Fine_Struc_Cst = 0.0072973525628
charge_e = np.sqrt(4 * np.pi * Fine_Struc_Cst)


def Gen_Field(Xarray, SystemList, Quark_Numb, TotnumbAllpart):
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

    """SystemList = []
    SYST = Global_variables.SYSTEM
    TotnumbAllpart, Quark_Numb = 0, 0
    for S_ind in range(Numb_of_TYPES):
        SystemList += SYST[S_ind][0] + SYST[S_ind][1]
        TotnumbAllpart += len(SYST[S_ind][0]) + len(SYST[S_ind][1])
        if S_ind in Quark_ind_LIST:
            Quark_Numb += len(SYST[S_ind][0]) + len(SYST[S_ind][1])"""

    if Quark_Numb != 0:
        STRONG_FORCE, BIG_vel_matrix = STRONG_FORCE_GROUP(TotnumbAllpart)
    else:
        STRONG_FORCE = np.zeros((TotnumbAllpart, TotnumbAllpart))
    force = EG_force + STRONG_FORCE

    acc_i = np.zeros((1, 100, 100))  # unit_vector* force
    acc_i = np.zeros_like(unit_vector.T * force) / np.ones_like(mass_matrix)
    maskM = mass_matrix != 0
    np.divide(unit_vector.T * force, mass_matrix, out=acc_i, where=maskM)

    acc = acc_i.T.sum(axis=1)

    return acc
