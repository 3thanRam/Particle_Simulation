import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from Particles.Global_Variables import Global_variables
from Misc.Relativistic_functions import gamma_factor

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

Gamma_array = np.vectorize(gamma_factor)


def Gen_Field(Xarray, SystemList, Quark_Numb, TotnumbAllpart):
    """Generates electromagnetic field tensor for each elm interacting particle

    Args:
        Xarray (ndarray): array of particle positions and their id information
        SystemList (SYSTEM_CLASS): SYSTEM_CLASS object
        Quark_Numb (int): current number of quarks
        TotnumbAllpart (int): current total number of particles

    Returns:
        F (ndarray) array of ELM tensor of size (4,4,total number of particles)
    """
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

    def Radius_Sum(index_a, index_b):
        return (
            PARTICLE_DICT[PARTICLE_NAMES[Xarray[0]["TypeID1"][index_a]]]["size"]
            + PARTICLE_DICT[PARTICLE_NAMES[Xarray[0]["TypeID1"][index_b]]]["size"]
        ) / 2

    def Array_genE_B():
        DELTA = (loc_arr.T[..., np.newaxis] - loc_arr.T[:, np.newaxis]).T

        d_min = np.array([[Radius_Sum(i, j) for j in range(N)] for i in range(N)])
        distances = np.linalg.norm(DELTA, axis=-1)
        Non_zero_mask = distances > d_min  # to avoid divergences
        # Get directions of forces
        unit_vector = np.zeros_like(DELTA.T)
        np.divide(DELTA.T, distances, out=unit_vector, where=Non_zero_mask)
        unit_vector = unit_vector.T

        Charge_matrix = np.outer(charges, np.ones_like(charges))
        ELEC_matrix = E_cst * Charge_matrix
        E_field = np.zeros_like(ELEC_matrix)
        np.divide(ELEC_matrix, distances**2, out=E_field, where=Non_zero_mask)

        E = (unit_vector.T * E_field).T.sum(axis=1)

        def OneD_B():
            return np.zeros((N, 3))

        def TwoD_B():
            Gamma = np.outer(
                np.ones(N),
                np.fromiter(
                    (
                        gamma_factor(velocity_matrix[part_i])
                        if mass_matrix[part_i] != 0
                        else 0
                        for part_i in range(N)
                    ),
                    "float",
                    N,
                ),
            )
            DELTA2 = np.array(
                [
                    [[-DELTA[i, j][1], DELTA[i, j][0], 0] for j in range(N)]
                    for i in range(N)
                ]
            )
            unit_vector2 = np.zeros_like(DELTA2.T)
            np.divide(DELTA2.T, distances, out=unit_vector2, where=Non_zero_mask)
            unit_vector2 = unit_vector2.T
            MAGN_matrix = M_cst * distances * Gamma * Charge_matrix
            B_field = np.zeros_like(MAGN_matrix)
            np.divide(MAGN_matrix, distances**3, out=B_field, where=Non_zero_mask)
            return (unit_vector2.T * B_field).T.sum(axis=1)

        def ThreeD_B():
            # Not sure if this is correct
            return np.cross(velocity_matrix, E) / C_speed**2

        return E, [OneD_B, TwoD_B, ThreeD_B][DIM_Numb - 1]()

    E, B = Array_genE_B()

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
