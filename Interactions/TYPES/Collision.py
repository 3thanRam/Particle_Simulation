import numpy as np

rng = np.random.default_rng()
from Particles.Global_Variables import Global_variables
import System.SystemClass as System_module
from Interactions.TYPES.Compton import (
    Compton_scattering,
)
from Interactions.TYPES.Relativistic import (
    Relativistic_Collision,
)
from Misc.Position_Fcts import in_all_bounds

BOUNDARY_COND = Global_variables.BOUNDARY_COND
if BOUNDARY_COND == 0:
    from Environment.Boundary_types import (
        BOUNDARY_FCT_PER,
    )

    BOUNDARY_FCT = BOUNDARY_FCT_PER
else:
    from Environment.Boundary_types import (
        BOUNDARY_FCT_HARD,
    )

    BOUNDARY_FCT = BOUNDARY_FCT_HARD
ROUNDDIGIT = Global_variables.ROUNDDIGIT
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
Vmax = Global_variables.Vmax


def COLLIDE(FirstAnn, Xf, COEFSlist, t):
    """update the particles 1,2 involved in a collision and update tracking information and collision points.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - Xf (list): List of about the particles in the simulation, indexed by particle type and ID.
    - COEFSlist (list): List containing groups of possibly interacting particles
    -t (float): current time of simulation
    Returns:
    - Xf (list): Updated Xf list
    """

    # Extract information about the collision
    ti, xo, coltype, z1, z2, p1, id1, p2, id2 = FirstAnn
    Global_variables.ALL_TIME.append(ti)
    Global_variables.COLPTS.append([ti, xo, coltype])

    partORAnti1, partORAnti2 = p1[0], p2[0]
    p1index, p2index = p1[1], p2[1]

    particle1 = System_module.SYSTEM.Get_Particle(p1index, partORAnti1, id1)
    particle2 = System_module.SYSTEM.Get_Particle(p2index, partORAnti2, id2)

    V1, b1, Targs1, p1c, id1c, Xinter1, ends1 = particle1.Coef_param_list
    V2, b2, Targs2, p2c, id2c, Xinter2, ends2 = particle2.Coef_param_list

    # get the particles involved in the collision

    E1, P1, M1 = particle1.Energy, particle1.P, particle1.M
    E2, P2, M2 = particle2.Energy, particle2.P, particle2.M

    # create photons
    if z1 != 0:
        V1 = BOUNDARY_FCT(V1, p1, id1, z1)
    if z2 != 0:
        V2 = BOUNDARY_FCT(V2, p2, id2, z2)

    Pos1, Pos2 = V1 * ti + b1[z1], V2 * ti + b2[z2]

    Pos1 = np.array([*Pos1], dtype=float)
    Pos2 = np.array([*Pos2], dtype=float)

    if (
        p1index == 13 or p2index == 13
    ):  # if one particle is photon then Compton_scattering
        VParam1, NewP1, NewE1, VParam2, NewP2, NewE2 = Compton_scattering(
            p1index, V1, M1, P1, E1, V2, M2, P2, E2
        )
    else:
        VParam1, NewP1, NewE1, VParam2, NewP2, NewE2 = Relativistic_Collision(
            V1, M1, P1, E1, V2, M2, P2, E2
        )

    particle1.V = VParam1
    particle1.Energy = NewE1
    particle1.P = NewP1

    particle2.V = VParam2
    particle2.Energy = NewE2
    particle2.P = NewP2

    Sizes = [particle1.Size / 2, particle2.Size / 2]
    Targs1, Xinter1 = list(Targs1), list(Xinter1)
    Targs2, Xinter2 = list(Targs2), list(Xinter2)
    Targs = [Targs1, Targs2]
    ARGS = [
        [
            Targs1,
            b1,
            Xinter1,
            System_module.SYSTEM.Vflipinfo[p1index][partORAnti1][id1],
        ],
        [
            Targs2,
            b2,
            Xinter2,
            System_module.SYSTEM.Vflipinfo[p2index][partORAnti2][id2],
        ],
    ]
    for part_ind in range(2):
        for tind in range(len(Targs[part_ind]) - 1):
            if Targs[part_ind][-1 - tind] > ti:
                for arg in ARGS[part_ind]:
                    arg.pop(-1)
            else:
                break

    def Set_F_data(Xf, particle_index, partORanti, id, NewXdata):
        for d in range(DIM_Numb):
            for subind, subF in enumerate(Xf[d]):
                if (
                    subF[1] == partORanti
                    and subF[2] == particle_index
                    and subF[3] == id
                ):
                    Xf[d][subind][0] = NewXdata[d]
        return Xf

    DT = t - ti
    Xinter = [Xinter1, Xinter2]
    b = [b1, b2]
    Pos = [Pos1, Pos2]
    Vparam = [VParam1, VParam2]
    pindex = [p1index, p2index]
    partORAnti = [partORAnti1, partORAnti2]
    id_list = [id1, id2]
    for part_ind in range(2):
        p, partorAnti, id = pindex[part_ind], partORAnti[part_ind], id_list[part_ind]
        position, velocity = Pos[part_ind], Vparam[part_ind]
        Targs[part_ind].append(ti)
        Xinter[part_ind].append([position, position])
        b[part_ind].append(position - velocity * ti)
        System_module.SYSTEM.Vflipinfo[p][partorAnti][id].append(velocity)
        Xinter[part_ind] = np.array(Xinter[part_ind])
        Xend = position + DT * velocity
        size = Sizes[part_ind]
        if not in_all_bounds(Xend, t, size):
            print("error bounced out of bounds")
            print(Xend, t, "  ", position, ti, "\n")
        Xf = Set_F_data(Xf, p, partorAnti, id, Xend)

        NewCoef = [
            velocity,
            b[part_ind],
            Targs[part_ind],
            (partorAnti, p),
            id,
            Xinter[part_ind],
            Xend,
        ]
        Particle = System_module.SYSTEM.Get_Particle(p, partorAnti, id)
        Particle.Coef_param_list = NewCoef
        Particle.V = velocity
    return Xf
