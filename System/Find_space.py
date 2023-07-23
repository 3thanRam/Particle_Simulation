import numpy as np
from Particles.Global_Variables import Global_variables
from Misc.Position_Fcts import in_all_bounds
import System.SystemClass

DIM_Numb = Global_variables.DIM_Numb
rng = np.random.default_rng()


def Find_space_for_particles(PosCenter, Particle_radius, V1, V2, t):
    """Find room to place 2 particles to avoid creating particles outside of boundaries

    Args:
        PosCenter (ndarray): point to look around
        Particle_radius (float): radius of particle to place
        V1 (ndarray): velocity of particle 1
        V2 (ndarray): velocity of particle 2
        t (float): current time of the simulation

    Returns:
        PosParam1/PosParam2(ndarray): positions to place particles
    """
    top_face = np.min(((PosCenter + Particle_radius), Global_variables.L), axis=0)
    bottom_face = np.max(
        ((PosCenter - Particle_radius), Global_variables.Linf),
        axis=0,
    )

    if np.all((top_face - bottom_face) > 4 * Particle_radius):
        Pos_shift = Particle_radius * V1 / np.linalg.norm(V1)
        PosParam1 = PosCenter + Pos_shift
        PosParam2 = PosCenter - Pos_shift
    Count = 0
    while True:
        if Count % 10 == 0:
            RandPOS = rng.uniform(bottom_face, top_face, size=(10, DIM_Numb))
        Pos_shift = RandPOS[Count % 10]
        PosParam1 = PosCenter + Pos_shift
        PosParam2 = PosCenter - Pos_shift
        dist = np.linalg.norm(PosParam1 - PosParam2)
        if (
            in_all_bounds(PosParam1, t=t, ParticleSize=2 * Particle_radius)
            and in_all_bounds(PosParam2, t=t, ParticleSize=2 * Particle_radius)
            and dist > 2 * Particle_radius
        ):
            return PosParam1, PosParam2

        Count += 1
        if Count > 10**4:
            return None, None
