import numpy as np
import vg

rng = np.random.default_rng()


from Particles.Dictionary import PARTICLE_DICT
from Particles.ParticleClass import Particle
from Particles.Global_Variables import Global_variables
from System.SystemClass import SYSTEM


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
    """update the particles 1,2 involved in a collision and update tracking information and collision points.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - F (list): List of dictionaries containing information about the particles in the simulation, indexed by particle type and ID.

    Returns:
    - F (list): Updated list of dictionaries containing information about the particles in the simulation.
    """

    # Extract information about the collision
    ti, xo, coltype, z1, z2, p1, id1, p2, id2 = FirstAnn

    Global_variables.ALL_TIME.append(ti)
    Global_variables.COLPTS.append([ti, xo, coltype])
    # return F

    partORAnti1, partORAnti2 = p1[0], p2[0]
    p1index, p2index = p1[1], p2[1]

    particle1 = SYSTEM.Get_Particle(p1index, partORAnti1, id1)
    particle2 = SYSTEM.Get_Particle(p2index, partORAnti2, id2)

    V1, b1, Targs1, p1c, id1c, Xinter1, ends1 = particle1.Coef_param_list
    V2, b2, Targs2, p2c, id2c, Xinter2, ends2 = particle2.Coef_param_list

    # get the particles involved in the collision

    E1, P1, M1 = particle1.Energy, particle1.P, particle1.M
    E2, P2, M2 = particle2.Energy, particle2.P, particle2.M
    Etot = E1 + E2
    Ptot = M1 * V1 + M2 * V2

    # create photons
    if z1 != 0:
        V1 = BOUNDARY_FCT(V1, p1, id1, z1)
    if z2 != 0:
        V2 = BOUNDARY_FCT(V2, p2, id2, z2)

    Pos1, Pos2 = V1 * ti + b1[z1], V2 * ti + b2[z2]

    Pos1 = np.array([*Pos1], dtype=float)
    Pos2 = np.array([*Pos2], dtype=float)

    Mtot = M1 + M2
    Mdif = M1 - M2

    VParam1 = (Mdif / Mtot) * V1 + (2 * M2 / Mtot) * V2

    VParam2 = (Ptot - VParam1 * M1) / M2
    Targs1, b1, Xinter1 = list(Targs1), list(b1), list(Xinter1)
    Targs2, b2, Xinter2 = list(Targs2), list(b2), list(Xinter2)
    Targs = [Targs1, Targs2]
    Rem_ind = [[], []]
    for targind, Targ in enumerate(Targs):
        for tind, tval in enumerate(Targ[1:]):
            if tval > ti:
                Rem_ind[targind].append(tind - 1)
    Rem_ind[0].sort(reverse=True)
    Rem_ind[1].sort(reverse=True)
    for remind in Rem_ind[0]:
        Targs1.pop(remind + 1)
        b1.pop(remind)
        Xinter1.pop(remind)
        ends1 -= 1
        SYSTEM.Vflipinfo[p1index][partORAnti1][id1].pop(remind)
    for remind in Rem_ind[1]:
        Targs2.pop(remind + 1)
        b2.pop(remind)
        Xinter2.pop(remind)
        ends2 -= 1
        SYSTEM.Vflipinfo[p2index][partORAnti2][id2].pop(remind)
    DT = t - ti
    Xend1, Xend2 = Pos1 + DT * VParam1, Pos2 + DT * VParam2

    Targs1.append(ti)
    Targs2.append(ti)
    x1o = Pos1
    x2o = Pos2

    Xinter1.append([x1o, x1o])
    Xinter2.append([x2o, x2o])
    newb1 = x1o - VParam1 * ti
    newb2 = x2o - VParam2 * ti
    b1.append(newb1)
    b2.append(newb2)
    ends1 += 1
    ends2 += 1

    SYSTEM.Vflipinfo[p1index][partORAnti1][id1].append([0, VParam1[0]])
    SYSTEM.Vflipinfo[p2index][partORAnti2][id2].append([0, VParam2[0]])

    b1 = np.array(b1)
    b2 = np.array(b2)
    Xinter1 = np.array(Xinter1)
    Xinter2 = np.array(Xinter2)

    NewCoefs = [
        [VParam1, b1, Targs1, p1, id1, Xinter1, ends1],
        [VParam2, b2, Targs2, p2, id2, Xinter2, ends2],
    ]

    def Set_F_data(F, particle_index, partORanti, id, NewXdata):
        for d in range(DIM_Numb):
            for subind, subF in enumerate(F[d]):
                if (
                    subF[1] == partORanti
                    and subF[2] == particle_index
                    and subF[3] == id
                ):
                    F[d][subind][0] = NewXdata[d]
        return F

    F = Set_F_data(F, p1[1], p1[0], id1, Xend1)
    F = Set_F_data(F, p2[1], p2[0], id2, Xend2)

    SYSTEM.Particle_set_coefs(p1index, partORAnti1, id1, NewCoefs[0])
    SYSTEM.Particle_set_coefs(p2index, partORAnti2, id2, NewCoefs[1])
    return F
