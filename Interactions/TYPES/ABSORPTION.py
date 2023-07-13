import numpy as np
import vg

rng = np.random.default_rng()

from Particles.Global_Variables import Global_variables
import System.SystemClass
from operator import itemgetter

item_get = itemgetter(2, 5)

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


def ABSORBE(FirstAnn, F, COEFSlist, t):
    """update tracking information after collision point.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - F (list): List of dictionaries containing information about the particles in the simulation, indexed by particle type and ID.

    Returns:
    - F (list): Updated list of dictionaries containing information about the particles in the simulation.
    """

    # Extract information about the collision
    ti, xo, coltype, zA, zB, pA, idA, pB, idB = FirstAnn
    SYSTEM = System.SystemClass.SYSTEM

    if pA[1] == 13:  # p1 is absorbed by p2
        p1, id1, z1 = pA, idA, zA
        p2, id2, z2 = pB, idB, zB
    else:
        p1, id1, z1 = pB, idB, zB
        p2, id2, z2 = pA, idA, zA
    partORAnti1, partORAnti2 = p1[0], p2[0]
    p1index, p2index = p1[1], p2[1]

    Global_variables.ALL_TIME.append(ti)
    Global_variables.COLPTS.append([ti, xo, coltype])
    Grnumblist, Targs1 = [], []

    Particle1 = SYSTEM.Get_Particle(p1index, partORAnti1, id1)
    Particle2 = SYSTEM.Get_Particle(p2index, partORAnti2, id2)

    Targs1, Xinter1 = item_get(Particle1.Coef_param_list)
    E1, V1 = Particle1.Energy, Particle1.V
    V_add = V1 / np.linalg.norm(V1)
    for groupnumb, coefgroup in enumerate(COEFSlist):
        for coefinfo in coefgroup:
            typetest, idtest = coefinfo
            if typetest == p1 and idtest == id1:
                COEFSlist[groupnumb].remove(coefinfo)

    V2 = Particle2.V
    Particle2.Energy += E1
    Vboost = V_add * E1 / (np.linalg.norm(Vmax) * Particle2.M)
    Particle2.V += Vboost

    SYSTEM.Remove_particle(p1index, partORAnti1, id1)
    if z1 > 0:
        for zi in range(z1):
            SYSTEM.TRACKING[p1index][partORAnti1][id1].extend(
                [
                    [Targs1[zi + 1], Xinter1[zi][0]],
                    ["T", "X"],
                    [Targs1[zi + 1], Xinter1[zi][1]],
                ]
            )
            Global_variables.ALL_TIME.extend(Targs1[1:])

    Global_variables.ALL_TIME.append(ti)
    V2, b2, Targs2, p2c, id2c, Xinter2, ends2 = Particle2.Coef_param_list
    Targs2, b2, Xinter2 = list(Targs2), list(b2), list(Xinter2)

    Rem_ind = []
    for tind, tval in enumerate(Targs2[1:]):
        if tval > ti:
            Rem_ind.append(tind - 1)
    Rem_ind.sort(reverse=True)
    for remind in Rem_ind:
        Targs2.pop(remind + 1)
        b2.pop(remind)
        Xinter2.pop(remind)
        ends2 -= 1
        SYSTEM.Vflipinfo[p2index][partORAnti2][id2].pop(remind)

    if z2 != 0:
        V2 = BOUNDARY_FCT(V2, p2, id2, z2)

    Pos2 = V2 * ti + b2[z2]
    Pos2 = np.array([*Pos2], dtype=float)

    VParam2 = Particle2.V
    DT = t - ti
    Xend2 = Pos2 + DT * VParam2
    Targs2.append(ti)
    x2o = Pos2
    Xinter2.append([x2o, x2o])
    newb2 = x2o - VParam2 * ti
    b2.append(newb2)
    SYSTEM.Vflipinfo[p2index][partORAnti2][id2].append([0, VParam2[0]])
    b2 = np.array(b2)
    Xinter2 = np.array(Xinter2)

    Particle2.Coef_param_list = [VParam2, b2, Targs2, p2, id2, Xinter2, Xend2]

    NewFO = [list(F[d]) for d in range(DIM_Numb)]
    for d in range(DIM_Numb):
        for subind, subF in enumerate(NewFO[d]):
            # [('Pos', float), ('TypeID0', int),('TypeID1', int),('index', int)]
            if subF[1] == p1[0] and subF[2] == p1[1] and subF[3] == id1:
                NewFO[d].remove(subF)
            if subF[1] == p2[0] and subF[2] == p2[1] and subF[3] == id2:
                NewFO[d][subind][0] = Xend2[d]

    dtype = [
        ("Pos", float),
        ("TypeID0", int),
        ("TypeID1", int),
        ("index", int),
    ]
    F = np.array(NewFO, dtype=dtype)
    # F.sort(order='Pos')

    Global_variables.COLPTS.append([ti, xo, coltype])

    return (F, COEFSlist)
