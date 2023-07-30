import numpy as np
from Particles.Global_Variables import Global_variables
from Misc.Functions import NORM

DIM_Numb = Global_variables.DIM_Numb
L_FCT = Global_variables.L_FCT
rng = np.random.default_rng()
Numb_Rand_create = 50
time, Radius = 0, 0

B_up, B_down = np.empty(DIM_Numb), np.empty(DIM_Numb)


def Set_Bounds():
    global B_up, B_down
    Lup, Ldown = L_FCT[0](time), L_FCT[1](time)
    B_up = Lup - Radius
    B_down = Ldown + Radius


def In_1D_bounds(array):
    return B_down[0] < array[0] < B_up[0]


def In_2D_bounds(array):
    return (B_down[0] < array[0] < B_up[0]) & (B_down[1] < array[1] < B_up[1])


def In_3D_bounds(array):
    return (
        (B_down[0] < array[0] < B_up[0])
        & (B_down[1] < array[1] < B_up[1])
        & (B_down[2] < array[2] < B_up[2])
    )


if DIM_Numb == 3:
    Quick_in_Bounds = In_3D_bounds
elif DIM_Numb == 2:
    Quick_in_Bounds = In_2D_bounds
elif DIM_Numb == 1:
    Quick_in_Bounds = In_1D_bounds


def Both_inBounds(Pos1, Pos2):
    return Quick_in_Bounds(Pos1) & Quick_in_Bounds(Pos2)


def Find_space_for_particles(PosCenter, Particle_radius, V1, V2, t):
    """Find room to place 2 particles to avoid creating particles outside of boundaries

    Args:
        PosCenter (ndarray): point to look around
        Particle_radius (float): radius of particle to place
        V1 (ndarray): velocity of particle 1
        V2 (ndarray): velocity of particle 2
        t (float): current time of the simulation

    Returns:
        PosParam1,PosParam2(ndarray): positions to place particles
    """
    global time, Radius
    time, Radius = t, Particle_radius
    Set_Bounds()
    top_face = np.min(((PosCenter + Particle_radius), Global_variables.L), axis=0)
    bottom_face = np.max(
        ((PosCenter - Particle_radius), Global_variables.Linf),
        axis=0,
    )

    if all(
        (top_face[d] - bottom_face[d]) > 4 * Particle_radius for d in range(DIM_Numb)
    ):
        Pos_shift = Particle_radius * V1 / NORM(V1)
        PosParam1 = PosCenter + Pos_shift
        PosParam2 = PosCenter - Pos_shift
    Count = 0
    while True:
        if Count % Numb_Rand_create == 0:
            RandPOS = rng.uniform(
                bottom_face, top_face, size=(Numb_Rand_create, DIM_Numb)
            )
        Pos_shift = RandPOS[Count % Numb_Rand_create]
        PosParam1 = PosCenter + Pos_shift
        PosParam2 = PosCenter - Pos_shift
        dist = NORM(PosParam1 - PosParam2)
        if Both_inBounds(PosParam1, PosParam2) and dist > 2 * Particle_radius:
            return PosParam1, PosParam2

        Count += 1
        if Count > 10**4:
            return None, None
