import numpy as np
import vg

rng = np.random.default_rng()


from Particles.Dictionary import PARTICLE_DICT

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

from Particles.Global_Variables import Global_variables
import System.SystemClass
from operator import itemgetter

item_get = itemgetter(
    0,
    2,
    5,
)

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


def ANNIHILATE(FirstAnn, Xf, COEFSlist, t):
    """Remove the particles 1,2 involved in a collision, create resulting photons and update tracking information and collision points.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - Xf (list): List of about the particles in the simulation, indexed by particle type and ID.
    - COEFSlist (list): List containing groups of possibly interacting particles
    -t (float): current time of simulation

    Returns:
    - Xf (list): Updated Xf list
    - COEFSlist (list): Updated COEFSlist list
    """

    # Extract information about the collision
    ti, xo, coltype, z1, z2, p1, id1, p2, id2 = FirstAnn
    SYSTEM = System.SystemClass.SYSTEM

    partORAnti1, partORAnti2 = p1[0], p2[0]
    p1index = p1[1]
    p2index = p2[1]

    Particle1 = SYSTEM.Get_Particle(p1index, partORAnti1, id1)
    Particle2 = SYSTEM.Get_Particle(p2index, partORAnti2, id2)
    E1 = Particle1.Energy
    V1, Targs1, Xinter1 = item_get(Particle1.Coef_param_list)
    E2 = Particle2.Energy
    V2, Targs2, Xinter2 = item_get(Particle2.Coef_param_list)

    Grnumblist = []
    for groupnumb, coefgroup in enumerate(COEFSlist):
        killcoef = []
        for indtokill, coefinfo in enumerate(coefgroup):
            typetest, idtest = coefinfo  # [-4]
            # idtest = coefinfo[-3]
            if (typetest == p1 and idtest == id1) or (typetest == p2 and idtest == id2):
                killcoef.append(indtokill)
        if killcoef:
            Grnumblist.append(groupnumb)
        killcoef.sort(reverse=True)
        for coefkillindex in killcoef:
            COEFSlist[groupnumb].pop(coefkillindex)

    # Remove the particles involved in the collision from their respective particle lists

    Etot = E1 + E2

    SYSTEM.Remove_particle(p1index, partORAnti1, id1)
    SYSTEM.Remove_particle(p2index, partORAnti2, id2)

    # create photons
    if z1 != 0:
        V1 = BOUNDARY_FCT(V1, p1, id1, z1)
    if z2 != 0:
        V2 = BOUNDARY_FCT(V2, p2, id2, z2)

    if DIM_Numb == 3:
        alpha, beta = vg.angle(V1, V2, look=vg.basis.z), vg.angle(
            V1, V2, look=vg.basis.y
        )
        VParam1 = Vmax * np.array(
            [np.cos(alpha) * np.cos(beta), np.sin(alpha) * np.cos(beta), -np.sin(beta)]
        )  # [1,0,0] rotated by alpha,beta around y,z
        VParam2 = Vmax * np.array(
            [
                np.cos(-alpha) * np.cos(-beta),
                np.sin(-alpha) * np.cos(-beta),
                -np.sin(-beta),
            ]
        )  # [1,0,0] rotated by alpha,beta around y,z
    elif DIM_Numb == 2:
        SD = 0.1745329251994329  # 10 degrees
        v1_u, v2_u = V1 / np.linalg.norm(V1), V2 / np.linalg.norm(V2)
        Angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        theta = Angle + np.round(rng.normal(Angle, SD, 1), ROUNDDIGIT)[0]
        VParam1 = Vmax * np.array([np.cos(theta), np.sin(theta)])
        VParam2 = Vmax * np.array([np.cos(-theta), np.sin(-theta)])
    else:
        VParam1 = Vmax
        VParam2 = -Vmax
    NewE1, NewE2 = Etot / 2, Etot / 2
    Createparams1 = "Post_Interaction", xo, VParam1, ti, NewE1
    Createparams2 = (
        "Post_Interaction",
        xo,
        VParam2,
        ti,
        NewE2,
    )

    def CREATE_PARTICLE_fromAnnil(ParticleType, Create_particle_param):
        """Remove the particles 1,2 involved in a collision and get their parameters

        Args:
            ParticleType (tuple): information about the type of particle
            Create_particle_param (list): list of parameters about the particle to create

        Returns:
            Ncoef(list): parameters detailling the trajectory of the created particle
            New_Xend(Ndarray): endpoint of the particle
        """
        partoranti, typeindex = ParticleType
        Crindex = typeindex

        SYSTEM.Add_Particle(Crindex, partoranti, Create_particle_param)

        SYSTEM.Particles_List[-1].MOVE(t, (t - ti))
        (
            New_V,
            b,
            New_Tpara,
            New_p,
            New_id,
            New_Xinter,
            New_Xend,
        ) = SYSTEM.Particles_List[-1].Coef_param_list

        if len(New_Xinter) == 0:
            b = [New_Xend - New_V * float(*New_Tpara)]
        else:
            b = []
            if BOUNDARY_COND == 0:
                for r in range(len(New_Xinter)):
                    b.append(New_Xinter[r][0] - New_V * float(New_Tpara[1 + r]))
                b.append(New_Xend - New_V * float(New_Tpara[0]))
            else:
                A = np.copy(New_V)
                for r in range(len(New_Xinter)):
                    b.append(New_Xinter[r][0] - A * float(New_Tpara[1 + r]))
                    A = SYSTEM.Vflipinfo[Crindex][partoranti][New_id][r]
                b.append(New_Xend - A * float(New_Tpara[0]))

        if len(New_Xinter) == 0:
            b.insert(0, "PreCreation")
            New_Tpara.append(ti)
            SYSTEM.Vflipinfo[Crindex][partoranti][New_id].append(New_V)
            Xinter_LIST = [[xo, xo]]  # [[New_Xend, New_Xend]]  # [[xo, xo]]
        else:
            xinterkilllist, vflipkill_list, b_kill_list, Tpara_kill_list = (
                [],
                [],
                [],
                [],
            )
            Xinter_LIST = list(New_Xinter)
            for it, newTpar in enumerate(New_Tpara):
                if newTpar < ti:
                    Tpara_kill_list.append(newTpar)
                    b_kill_list.append(b[it - 1])
                    xinterkilllist.append(it - 1)
                    vflipkill_list.append(
                        SYSTEM.Vflipinfo[Crindex][partoranti][New_id][it - 1]
                    )
            SYSTEM.Vflipinfo[Crindex][partoranti][New_id].insert(0, New_V)
            for Tpara_kill in Tpara_kill_list:
                New_Tpara.remove(Tpara_kill)
            New_Tpara.insert(1, ti)
            for b_kill in b_kill_list:
                b.remove(b_kill)
            xinterkilllist.sort(reverse=True)
            Newelem = []
            for killxinter_index in xinterkilllist:
                if len(Xinter_LIST) == 1:
                    Newelem = Xinter_LIST[killxinter_index]
                Xinter_LIST.pop(killxinter_index)
            for vflipkill in vflipkill_list:
                SYSTEM.Vflipinfo[Crindex][partoranti][New_id].remove(vflipkill)
            if len(Newelem) == 0:
                Newelem = Xinter_LIST[-1].copy()
            Newelem[0] = xo
            Newelem[1] = xo
            Xinter_LIST.append(Newelem)
            b.insert(0, "PreCreation")

        New_Xinter = np.array(Xinter_LIST)
        Ncoef = [New_V, b, New_Tpara, New_p, New_id, New_Xinter, New_Xend]
        return (Ncoef, New_Xend)

    Coef_info1, Xend1 = CREATE_PARTICLE_fromAnnil((0, 13), Createparams1)
    Coef_info2, Xend2 = CREATE_PARTICLE_fromAnnil((0, 13), Createparams2)
    Xendlist = [Xend1, Xend2]
    NewCoefs = [Coef_info1, Coef_info2]

    SYSTEM.Particle_set_coefs(
        Coef_info1[3][1], Coef_info1[3][0], Coef_info1[4], Coef_info1
    )
    SYSTEM.Particle_set_coefs(
        Coef_info2[3][1], Coef_info2[3][0], Coef_info2[4], Coef_info2
    )
    NewFO = [list(Xf[d]) for d in range(DIM_Numb)]
    TOKILL = [[] for d in range(DIM_Numb)]
    for d in range(DIM_Numb):
        for subind, subF in enumerate(NewFO[d]):
            # [('Pos', float), ('TypeID0', int),('TypeID1', int),('index', int)]
            if (subF[1] == p1[0] and subF[2] == p1[1] and subF[3] == id1) or (
                subF[1] == p2[0] and subF[2] == p2[1] and subF[3] == id2
            ):  # (subF[2]==p2 and subF[0]==id2)
                TOKILL[d].append(subF)

    # If the particles have a history of collisions, update the tracking information
    if z1 > 0:
        for zi in range(z1):
            SYSTEM.TRACKING[p1index][partORAnti1][id1].extend(
                [
                    [Targs1[zi + 1], Xinter1[zi][0]],
                    ["T", "X"],
                    [Targs1[zi + 1], Xinter1[zi][1]],
                ]
            )

    if z2 > 0:
        for zi in range(z2):
            SYSTEM.TRACKING[p2index][partORAnti2][id2].extend(
                [
                    [Targs2[zi + 1], Xinter2[zi][0]],
                    ["T", "X"],
                    [Targs2[zi + 1], Xinter2[zi][1]],
                ]
            )
    Global_variables.ALL_TIME.extend(Targs1[1:])
    Global_variables.ALL_TIME.extend(Targs2[1:])

    Global_variables.ALL_TIME.append(ti)

    # Remove the information about the particles involved in the collision from the collision point list and decrement the total number of particles and add product of annihilation
    for d in range(DIM_Numb):
        NewFelem = []
        for killparticl in TOKILL[d]:
            if len(NewFO[d]) == 1:
                NewFelem = killparticl.copy()
            NewFO[d].remove(killparticl)
        for Nco, Ncoef in enumerate(NewCoefs):
            N_Xd, N_type, N_id = Xendlist[Nco][d], Ncoef[-4], Ncoef[-3]
            if len(NewFelem) == 0:
                NewFelem = (NewFO[d][-1]).copy()
            NewFelem[0] = N_Xd
            NewFelem[1] = N_type[0]
            NewFelem[2] = N_type[1]
            NewFelem[3] = N_id
            NewFO[d].append(NewFelem)

    dtype = [
        ("Pos", float),
        ("TypeID0", int),
        ("TypeID1", int),
        ("index", int),
    ]
    Xf = np.array(NewFO, dtype=dtype)
    # Xf.sort(order='Pos')

    for gnum in Grnumblist:
        for Ncoef in NewCoefs:
            Newp, Newid = Ncoef[3], Ncoef[4]
            COEFSlist[gnum].append([Newp, Newid])

    Global_variables.COLPTS.append([ti, xo, coltype])

    return (Xf, COEFSlist)
