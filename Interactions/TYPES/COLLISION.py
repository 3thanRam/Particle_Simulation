import numpy as np

rng = np.random.default_rng()
from Particles.Global_Variables import Global_variables
import System.SystemClass


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


def gamma_factor(v):
    return 1 / (1 - (v / C_speed) ** 2) ** 0.5


def lorentz_boost(Energy, momentum, Velocity, INV=1):
    V_norm = np.linalg.norm(Velocity)
    V_direction = Velocity / V_norm
    Gamma = gamma_factor(V_norm)
    PdotV = np.dot(momentum, V_direction)
    New_E = Gamma * (Energy - INV * V_norm * PdotV / C_speed)
    New_P = (
        momentum
        + (Gamma - 1) * PdotV * V_direction
        - INV * Gamma * Energy * Velocity / C_speed
    )
    return New_E, New_P


def INV_lorentz_boost(Energy, momentum, Velocity):
    V_norm = np.linalg.norm(Velocity)
    V_direction = Velocity / V_norm
    Gamma = gamma_factor(V_norm)
    PdotV = np.dot(momentum, V_direction)
    New_E = Gamma * (Energy - V_norm * PdotV / C_speed)
    New_P = (
        momentum
        + (Gamma - 1) * PdotV * V_direction
        - Gamma * Energy * Velocity / C_speed
    )
    return New_E, New_P


def Get_V_from_P(momentum, mass):
    P_norm = np.linalg.norm(momentum)
    direction = P_norm / momentum
    V_norm = P_norm * (1 / (1 + (P_norm / (mass * C_speed)) ** 2)) ** 0.5 / mass
    return V_norm * direction


def COLLIDE(FirstAnn, Xf, COEFSlist, t):
    """update the particles 1,2 involved in a collision and update tracking information and collision points.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - Xf (list): List of dictionaries containing information about the particles in the simulation, indexed by particle type and ID.

    Returns:
    - Xf (list): Updated list of dictionaries containing information about the particles in the simulation.
    """

    # Extract information about the collision
    ti, xo, coltype, z1, z2, p1, id1, p2, id2 = FirstAnn
    SYSTEM = System.SystemClass.SYSTEM

    Global_variables.ALL_TIME.append(ti)
    Global_variables.COLPTS.append([ti, xo, coltype])

    partORAnti1, partORAnti2 = p1[0], p2[0]
    p1index, p2index = p1[1], p2[1]

    particle1 = SYSTEM.Get_Particle(p1index, partORAnti1, id1)
    particle2 = SYSTEM.Get_Particle(p2index, partORAnti2, id2)

    V1, b1, Targs1, p1c, id1c, Xinter1, ends1 = particle1.Coef_param_list
    V2, b2, Targs2, p2c, id2c, Xinter2, ends2 = particle2.Coef_param_list

    # get the particles involved in the collision

    E1, P1, M1 = particle1.Energy, particle1.P, particle1.M
    E2, P2, M2 = particle2.Energy, particle2.P, particle2.M

    Ptot = M1 * V1 + M2 * V2

    # create photons
    if z1 != 0:
        V1 = BOUNDARY_FCT(V1, p1, id1, z1)
    if z2 != 0:
        V2 = BOUNDARY_FCT(V2, p2, id2, z2)

    Pos1, Pos2 = V1 * ti + b1[z1], V2 * ti + b2[z2]

    Pos1 = np.array([*Pos1], dtype=float)
    Pos2 = np.array([*Pos2], dtype=float)

    if p1index == 13 or p2index == 13:  # Compton scattering
        if p1index == 13:
            pi_photon, Ei_photon = P1, E1
            P2 = gamma_factor(np.linalg.norm(V2)) * M2 * V2
            E2 = ((M2 * C_speed**2) ** 2 + np.dot(P2, P2) * C_speed**2) ** 0.5
            Vboost = P2 / E2
            pi_particle, Ei_particle, M_particle = P2, E2, M2
        else:
            pi_photon, Ei_photon = P2, E2
            P1 = gamma_factor(np.linalg.norm(V1)) * M1 * V1
            E1 = ((M1 * C_speed**2) ** 2 + np.dot(P1, P1) * C_speed**2) ** 0.5
            Vboost = P1 / E1
            pi_particle, Ei_particle, M_particle = P1, E1, M1
        Ei_boosted_photon, Pi_boosted_photon = lorentz_boost(
            Ei_photon, pi_photon, Vboost
        )
        Ei_boosted_particle, Pi_boosted_particle = lorentz_boost(
            Ei_particle, pi_particle, Vboost
        )

        def Cross_section(alpha, Re):
            alpha *= np.pi / 180
            R_Ef_Ei = 1 / (1 + Ei_boosted_photon * (1 - np.cos(alpha)))
            DCS = (
                0.5 * (Re / R_Ef_Ei) ** 2 * (1 / R_Ef_Ei + R_Ef_Ei - np.sin(alpha) ** 2)
            )
            return DCS

        # cross_section=lambda O: 0.5*Re*(1/)
        # import matplotlib.pyplot as plt
        # X = np.linspace(0, 180, 10**3)
        # RE = [1, 10]
        # for Re in RE:
        #    Y = [Cross_section(x, Re) for x in X]
        #    plt.plot(X, Y, label=str(Re))
        # plt.legend()
        # plt.show()

        theta = 10 * np.pi / 180
        cotan_phi = (1 + Ei_boosted_photon / Ei_boosted_particle) * np.tan(theta / 2)
        phi = np.arctan(cotan_phi)
        Ef_boosted_photon = Ei_boosted_photon / (
            1 + Ei_boosted_photon * (1 - np.cos(theta))
        )
        pf_boosted_particle_norm = (
            (Ei_boosted_photon - Ef_boosted_photon + Ei_boosted_particle) ** 2
            - Ei_boosted_particle**2
        ) ** 0.5 / C_speed
        pf_boosted_particle_direction = (
            Pi_boosted_photon / np.linalg.norm(Pi_boosted_photon)
        ) / np.cos(phi)
        pf_boosted_particle = pf_boosted_particle_norm * pf_boosted_particle_direction

        pf_boosted_photon_norm = Ef_boosted_photon / C_speed
        pf_boosted_photon_direction = -(
            Pi_boosted_photon / np.linalg.norm(Pi_boosted_photon)
        ) / np.cos(theta)
        pf_boosted_photon = pf_boosted_photon_norm * pf_boosted_photon_direction

        Ef_boosted_particle = (
            Ei_boosted_particle**2 + (pf_boosted_particle_norm * C_speed) ** 2
        ) ** 0.5
        if p1index == 13:
            NewE1, NewP1 = lorentz_boost(
                Ef_boosted_photon, pf_boosted_photon, Vboost, INV=-1
            )
            VParam1 = C_speed * NewP1 / np.linalg.norm(NewP1)
            NewE2, NewP2 = lorentz_boost(
                Ef_boosted_particle, pf_boosted_particle, Vboost, INV=-1
            )
            VParam2 = Get_V_from_P(NewP2, M2)
            Vn2 = np.linalg.norm(VParam2)
            if Vn2 >= C_speed:
                VParam2 *= 0.99 * C_speed / Vn2
                print("collision velocity error2")
        else:
            NewE2, NewP2 = lorentz_boost(
                Ef_boosted_photon, pf_boosted_photon, Vboost, INV=-1
            )
            VParam2 = C_speed * NewP2 / np.linalg.norm(NewP2)
            NewE1, NewP1 = lorentz_boost(
                Ef_boosted_particle, pf_boosted_particle, Vboost, INV=-1
            )
            VParam1 = Get_V_from_P(NewP1, M1)
            Vn1 = np.linalg.norm(VParam1)
            if Vn1 >= C_speed:
                VParam1 *= 0.99 * C_speed / Vn1
                print("collision velocity error1")
    else:
        NewE1, NewE2 = E1, E2
        Mtot = M1 + M2
        Mdif = M1 - M2
        VParam1 = (Mdif / Mtot) * V1 + (2 * M2 / Mtot) * V2
        VParam2 = (Ptot - VParam1 * M1) / M2

        Vn1, Vn2 = np.linalg.norm(VParam1), np.linalg.norm(VParam2)
        if Vn1 >= C_speed:
            VParam1 *= 0.99 * C_speed / Vn1
        if Vn2 >= C_speed:
            VParam2 *= 0.99 * C_speed / Vn2

        NewP1, NewP2 = M1 * VParam1, M2 * VParam2

    particle1.V = VParam1
    particle1.Energy = NewE1
    particle1.P = NewP1

    particle2.V = VParam2
    particle2.Energy = NewE2
    particle2.P = NewP2

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

    SYSTEM.Vflipinfo[p1index][partORAnti1][id1].append([0, VParam1[0]])
    SYSTEM.Vflipinfo[p2index][partORAnti2][id2].append([0, VParam2[0]])

    b1 = np.array(b1)
    b2 = np.array(b2)
    Xinter1 = np.array(Xinter1)
    Xinter2 = np.array(Xinter2)

    NewCoefs = [
        [VParam1, b1, Targs1, p1, id1, Xinter1, Xend1],
        [VParam2, b2, Targs2, p2, id2, Xinter2, Xend2],
    ]

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

    Xf = Set_F_data(Xf, p1[1], p1[0], id1, Xend1)
    Xf = Set_F_data(Xf, p2[1], p2[0], id2, Xend2)

    SYSTEM.Particle_set_coefs(p1index, partORAnti1, id1, NewCoefs[0])
    SYSTEM.Particle_set_coefs(p2index, partORAnti2, id2, NewCoefs[1])
    return Xf
