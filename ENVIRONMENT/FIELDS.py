import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from Particles.Global_Variables import Global_variables
from Interactions.TYPES.SPONTANEOUS import STRONG_FORCE_GROUP

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
C_speed = Global_variables.C_speed
dt = Global_variables.dt
####Natural Units#####
######################
# c, me, ħ, ε0,=1
E_cst = -1  # / (4 * np.pi)
Grav_cst = E_cst * 10**-45
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

    if Quark_Numb != 0:
        STRONG_FORCE = STRONG_FORCE_GROUP(TotnumbAllpart)[0]
    else:
        STRONG_FORCE = np.zeros((TotnumbAllpart, TotnumbAllpart))
    force = EG_force + STRONG_FORCE

    """Big_mass_matrix = np.outer(mass_matrix, mass_matrix)

    velocity_matrix = np.array([s.V for s in SystemList])
    velocities = np.outer(velocity_matrix, velocity_matrix)
    gamma_matrix = gamma_factor(velocities)
    Big_gamma_matrix = np.outer(gamma_matrix, gamma_matrix)
    print(velocity_matrix.shape)
    print(force.shape, velocities.shape, Big_mass_matrix.shape, Big_gamma_matrix.shape)
    a = (force - np.dot(force, velocities) * velocities / C_speed**2) / (
        Big_mass_matrix * Big_gamma_matrix
    )
    print(a.shape)
    print(5 / 0)"""
    acc_i = np.zeros_like(unit_vector.T * force) / np.ones_like(mass_matrix)
    maskM = mass_matrix != 0
    np.divide(unit_vector.T * force, mass_matrix, out=acc_i, where=maskM)
    acc = acc_i.T.sum(axis=1)
    Vel = acc * dt
    Vel = np.where(
        np.linalg.norm(Vel) >= 0.99 * C_speed, 0.9 * Vel / np.linalg.norm(Vel), Vel
    )
    return Vel
