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

    Particle1 = System.SystemClass.SYSTEM.Get_Particle(p1index, partORAnti1, id1)
    Targs1, Xinter1 = item_get(Particle1.Coef_param_list)
    Etot, Vpart = Particle1.Energy, Particle1.V
    vect_direct = Vpart / np.linalg.norm(Vpart)

    for groupnumb, coefgroup in enumerate(COEFSlist):
        killcoef = []
        for indtokill, coefinfo in enumerate(coefgroup):
            typetest, idtest = coefinfo
            if typetest == p1 and idtest == id1:
                killcoef.append(indtokill)
        if killcoef:
            Grnumblist.append(groupnumb)
        killcoef.sort(reverse=True)
        for coefkillindex in killcoef:
            COEFSlist[groupnumb].pop(coefkillindex)

    # Remove the particles involved in the collision from their respective particle lists

    System.SystemClass.SYSTEM.Change_Particle_Energy_velocity(
        p2index, partORAnti2, id2, Etot, vect_direct
    )
    System.SystemClass.SYSTEM.Remove_particle(p1index, partORAnti1, id1)
    # If the particles have a history of collisions, update the tracking information
    if z1 > 0:
        for zi in range(z1):
            System.SystemClass.SYSTEM.TRACKING[p1index][partORAnti1][id1].extend(
                [
                    [Targs1[zi + 1], Xinter1[zi][0]],
                    ["T", "X"],
                    [Targs1[zi + 1], Xinter1[zi][1]],
                ]
            )
            Global_variables.ALL_TIME.extend(Targs1[1:])

    # System.SystemClass.SYSTEM.TRACKING[p1index][partORAnti1][id1].append([ti, [*xo]])
    # System.SystemClass.SYSTEM.TRACKING[p2index][partORAnti2][id2].append([ti, [*xo]])

    Global_variables.ALL_TIME.append(ti)

    NewFO = [list(F[d]) for d in range(DIM_Numb)]
    TOKILL = [[] for d in range(DIM_Numb)]
    for d in range(DIM_Numb):
        for subind, subF in enumerate(NewFO[d]):
            # [('Pos', float), ('TypeID0', int),('TypeID1', int),('index', int)]
            if subF[1] == p1[0] and subF[2] == p1[1] and subF[3] == id1:
                TOKILL[d].append(subF)
    # Remove the information about the particles involved in the collision from the collision point list and decrement the total number of particles
    for d in range(DIM_Numb):
        for killparticl in TOKILL[d]:
            NewFO[d].remove(killparticl)
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
