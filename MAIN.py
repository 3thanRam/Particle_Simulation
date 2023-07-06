import numpy as np
import time
from operator import itemgetter
import matplotlib.pyplot as plt

from Particles.Dictionary import PARTICLE_DICT

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

from Display.Density import DENS_FCT


def Update_Density(Dens, L, Linf, Numb_Per_TYPE):
    Vol = 1
    if DIM_Numb == 1:
        Vol = L[0] - Linf[0]
    else:
        Vol = np.prod(L - Linf)
    for p in range(len(Numb_Per_TYPE)):
        Dens[p][0].append(Numb_Per_TYPE[p][0] / Vol)
        Dens[p][1].append(Numb_Per_TYPE[p][1] / Vol)
    return Dens


def Update_L_Lmin(t):
    return (L_FCT[0](t), L_FCT[1](t))


def main(T, Repr_type, File_path_name=None):
    """
    Simulates the behavior of n1 particles and n2 antiparticles in a Repr_type dimensional box

    Args:
    T (float): The time range to simulate.
    Repr_type (int): The type of representation: draw just densities/0 or also draw trajectories/1
    File_path_name: Where to save the video if the simulation is 3D and Repr_type=1

    Returns:
    None: The function does not return anything, but it prints the total time of the simulation and
    draws the points if Repr_type=1.

    """

    from Misc.Functions import ROUND
    from Particles.SystemClass import init

    init()
    from Particles.SystemClass import SYSTEM
    from Particles.Interactions.TYPES.SPONTANEOUS import SpontaneousEvents
    from Particles.Interactions.INTERACTION_LOOP import Interaction_Loop_Check

    T = int(10 * T)
    time0 = time.time()  # Starting time for the simulation

    Dens = [[[], []] for num in range(Numb_of_TYPES)]
    L, Linf = Update_L_Lmin(0)

    Dper = 0
    if Repr_type == 0 or Repr_type == 1:
        print("Generating Points")
        print(Dper, "%", end="\r")

    Nset = [[0, 0] for Ntype in range(Numb_of_TYPES)]
    Nset[13] = [5, 0]  # photons
    Nset[6] = [2, 0]  # upquark
    Nset[7] = [1, 0]  # downquark
    ## elec 0
    ## muon 2
    ## quarks: 6-11

    # Simulation loop
    ################

    for (
        index,
        (Npart, Nantipart),
    ) in enumerate(Nset):
        for Np in range(Npart):
            SYSTEM.Add_Particle(index, 0)
        for Na in range(Nantipart):
            SYSTEM.Add_Particle(index, 1)

    SpontaneousEvents(0)
    Dens = Update_Density(Dens, L, Linf, SYSTEM.Numb_Per_TYPE)
    for ti in range(1, T):
        perc = int(100 * ti / T)
        if perc > Dper and Repr_type == 0 or Repr_type == 1:  # and perc%5==0
            Dper = perc
            print(Dper, "%", end="\r")

        t = ROUND(ti * dt)
        L, Linf = Update_L_Lmin(t)
        Global_variables.Update_Bound_Params(L, Linf, t)
        SYSTEM.UPDATE(t)
        PARAMS = SYSTEM.PREPARE_PARAMS()

        if len(PARAMS[-1]) != 0:
            Xf = Interaction_Loop_Check(SYSTEM.Xf, t, PARAMS)
        else:
            Xf = SYSTEM.Xf

        # Update particle positions and tracking history
        for indUpdate in range(len(Xf[0])):
            if not Xf[0][indUpdate]:
                continue
            pos_search = [Xf[0][indUpdate][0]]
            PartOrAnti_search, typeindex_search, id_search = [*Xf[0][indUpdate]][1:4]

            for d in range(1, DIM_Numb):
                Pos_d_index = np.where(
                    (Xf[d]["index"] == id_search)
                    & (Xf[d]["TypeID0"] == PartOrAnti_search)
                    & (Xf[d]["TypeID1"] == typeindex_search)
                )[0][0]
                pos_search.append(Xf[d]["Pos"][Pos_d_index])

            SYSTEM.UPDATE_TRACKING(
                typeindex_search, PartOrAnti_search, id_search, t, pos_search
            )

        # Update the densities of particles
        Dens = Update_Density(Dens, L, Linf, SYSTEM.Numb_Per_TYPE)

        if SYSTEM.Tot_Numb == 0:
            T = ti
            break
        if ti != T - 1:
            SpontaneousEvents(t)

    Dt = round(time.time() - time0, 2)

    # post-simulation operations depending on the input parameter Repr_type, which determines the type of output the function produces.

    if (
        Repr_type == 1
    ):  # if Repr_type is equal to 1, then the function produces a graphical output of the simulation results using the DRAW function.
        print("Generating time:", Dt, "s")
        import Display.DisplayDRAW as DRAW_TRAJ

        ALL_TIME = Global_variables.ALL_TIME
        ALL_TIME.extend([ROUND(i * dt) for i in range(T)])
        ALL_TIME = [*set(ALL_TIME)]
        ALL_TIME.sort()
        DRAW_PARAMS = [
            T,
            dt,
            [Linf, L],
            DIM_Numb,
            Global_variables.COLPTS,
            SYSTEM.TRACKING,
            Dens,
            L_FCT,
            BOUNDARY_COND,
            ALL_TIME,
            PARTICLE_DICT,
        ]
        if File_path_name != None:
            DRAW_PARAMS.append(File_path_name)
        DRAW_TRAJ.DRAW(*DRAW_PARAMS)
    elif (
        Repr_type == 0
    ):  # the function produces a density plot of the simulation results using the DENS_FCT function
        print("Total time:", Dt, "s")
        Trange = np.linspace(0, (T - 1) * dt, len(Global_variables.ALL_TIME))
        fig, ax = plt.subplots()
        DENS_FCT(DIM_Numb, Dens, Trange, ax, PARTICLE_DICT)
        plt.show()
    else:
        return Dt


if __name__ == "__main__":
    from Dialog.CHOICES import SET_PARAMETERS

    SET_Param = SET_PARAMETERS()

    T_param, Repr_mode, filename = SET_Param[0], SET_Param[7], SET_Param[8]
    l, Numb_Dimensions, BOUNDARY_COND = SET_Param[4], SET_Param[5], SET_Param[6]

    LO, LOinf = np.array(l), np.array([0 for d in range(Numb_Dimensions)])
    L_FCT = [lambda x: LO + 0 * np.cos(x), lambda x: LOinf + 0 * np.sin(x + 1)]

    from Particles.Global_Variables import init

    init(Numb_Dimensions, BOUNDARY_COND, L_FCT)
    from Particles.Global_Variables import Global_variables

    DIM_Numb = Global_variables.DIM_Numb
    Vmax = Global_variables.Vmax
    dt = Global_variables.dt
    L_FCT = Global_variables.L_FCT

    main(T_param, Repr_mode, filename)
