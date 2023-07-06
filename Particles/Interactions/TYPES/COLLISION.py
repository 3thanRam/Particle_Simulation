import numpy as np
import vg

rng = np.random.default_rng()


from Particles.Dictionary import PARTICLE_DICT
from Particles.ParticleClass import Particle
from Particles.Global_Variables import Global_variables


BOUNDARY_COND = Global_variables.BOUNDARY_COND
if BOUNDARY_COND == 0:
    from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_PER

    BOUNDARY_FCT = BOUNDARY_FCT_PER
else:
    from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_HARD

    BOUNDARY_FCT = BOUNDARY_FCT_HARD
ROUNDDIGIT = Global_variables.ROUNDDIGIT
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
Vmax = Global_variables.Vmax


def COLLIDE(FirstAnn, F, COEFSlist, t):
    """update tracking information after collision point.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - F (list): List of dictionaries containing information about the particles in the simulation, indexed by particle type and ID.

    Returns:
    - F (list): Updated list of dictionaries containing information about the particles in the simulation.
    """

    # Extract information about the collision
    ti, xo, coltype, z1, z2, p1, id1, p2, id2 = FirstAnn

    Global_variables.ALL_TIME.append(ti)
    Global_variables.COLPTS.append([ti, xo, coltype])  # FirstAnn)

    return (F, COEFSlist)
