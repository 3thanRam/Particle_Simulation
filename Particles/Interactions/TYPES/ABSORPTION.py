import numpy as np
import vg

rng = np.random.default_rng()

from Particles.Dictionary import PARTICLE_DICT
from Particles.ParticleClass import Particle
from Particles.Global_Variables import Global_variables


BOUNDARY_FCT = Global_variables.BOUNDARY_FCT
BOUNDARY_COND = Global_variables.BOUNDARY_COND
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
    partORAnti1 = p1[0]
    partORAnti2 = p2[0]
    Global_variables.ALL_TIME.append(ti)
    Global_variables.COLPTS.append([ti, xo, coltype])  # FirstAnn)

    Grnumblist, Targs1 = [], []
    for groupnumb, coefgroup in enumerate(COEFSlist):
        killcoef = []
        for indtokill, coefinfo in enumerate(coefgroup):
            typetest = coefinfo[-4]
            idtest = coefinfo[-3]
            if typetest == p1 and idtest == id1:
                killcoef.append(indtokill)
                Targs1 = coefinfo[2]
                Xinter1 = coefinfo[-2]
        if killcoef:
            Grnumblist.append(groupnumb)
        killcoef.sort(reverse=True)
        for coefkillindex in killcoef:
            coefgroup.pop(coefkillindex)

    p1index = p1[1]
    p2index = p2[1]
    # p1index=TYPE_to_index_Dict[p1[1]]
    # p2index=TYPE_to_index_Dict[p2[1]]

    # Remove the particles involved in the collision from their respective particle lists

    Etot = 0
    for s in range(len(Global_variables.SYSTEM[p1index][partORAnti1])):
        if Global_variables.SYSTEM[p1index][partORAnti1][s].ID == id1:
            Etot += Global_variables.SYSTEM[p1index][partORAnti1][s].Energy
            Vpart = Global_variables.SYSTEM[p1index][partORAnti1][s].V
            Global_variables.SYSTEM[p1index][partORAnti1].remove(
                Global_variables.SYSTEM[p1index][partORAnti1][s]
            )
            vect_direct = Vpart / np.linalg.norm(Vpart)
            break

    for s in range(len(Global_variables.SYSTEM[p2index][partORAnti2])):
        if Global_variables.SYSTEM[p2index][partORAnti2][s].ID == id2:
            Global_variables.SYSTEM[p2index][partORAnti2][s].Energy += Etot
            Vboost = (
                vect_direct
                * Etot
                / (
                    np.linalg.norm(Vmax)
                    * Global_variables.SYSTEM[p2index][partORAnti2][s].M
                )
            )
            Global_variables.SYSTEM[p2index][partORAnti2][s].V += Vboost
            break

    NewFO = [list(F[d]) for d in range(DIM_Numb)]
    TOKILL = [[] for d in range(DIM_Numb)]

    for d in range(DIM_Numb):
        for subind, subF in enumerate(NewFO[d]):
            # [('Pos', float), ('TypeID0', int),('TypeID1', int),('index', int)]
            if subF[1] == p1[0] and subF[2] == p1[1] and subF[3] == id1:
                TOKILL[d].append(subF)

    # If the particles have a history of collisions, update the tracking information
    if z1 > 0:
        for zi in range(z1):
            Global_variables.TRACKING[p1index][partORAnti1][id1].extend(
                [
                    [Targs1[zi + 1], Xinter1[zi][0]],
                    ["T", "X"],
                    [Targs1[zi + 1], Xinter1[zi][1]],
                ]
            )
            Global_variables.ALL_TIME.extend(Targs1[1:])

    Global_variables.TRACKING[p1index][partORAnti1][id1].append([ti, [*xo]])
    Global_variables.TRACKING[p2index][partORAnti2][id2].append([ti, [*xo]])
    Global_variables.ALL_TIME.append(ti)

    # Remove the information about the particles involved in the collision from the collision point list and decrement the total number of particles
    for d in range(DIM_Numb):
        for killparticl in TOKILL[d]:
            NewFO[d].remove(killparticl)
    dtype = [
        ("Pos", float),
        ("TypeID0", int),
        ("TypeID1", int),
        ("index", int),
    ]  # [('Pos', float), ('Type', np.ndarray),('index', int)]#[('Pos', float), ('Type', int),('index', int)]
    F = np.array(NewFO, dtype=dtype)
    # F.sort(order='Pos')

    Global_variables.COLPTS.append([ti, xo, coltype])
    Global_variables.Ntot[p1index][partORAnti1] -= 1
    return (F, COEFSlist)
