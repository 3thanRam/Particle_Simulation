import numpy as np
import time
import matplotlib.pyplot as plt

from Particles.Dictionary import PARTICLE_DICT
from Display.Density import DENS_FCT

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]


def Update_Density(Dens, L, Linf, Numb_Per_TYPE):
    """Update Density list, a list containing the number of particles of each type divided by the volume of the box at each time instant dt

    Args:
        Dens (list):Old Density list, list of shape (number of particles,2,number of spatial dimensions,number of time steps) where the second number represents particle/antiparticle
        L (ndarray): Top faces of the box
        Linf (ndarray): Lower faces of the box
        Numb_Per_TYPE (list): number of particles of each type

    Returns:
        _type_: Update Density list
    """
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
    """Get upper and lower boundaries of the box at a time t

    Args:
        t (float): time at which to get the values

    Returns:
        up_L,lowL(ndarray,ndarray):upper and lower boundaries
    """
    return (L_FCT[0](t), L_FCT[1](t))


def main(T, Repr_type, Nset):
    """
    Main loop of the simulation updates the particles in a DIM_Numb dimensional box, for a number T of dt intervals, during dt the trajectories are considered to be straight lines
    After the T intervals or if there are no particles the simulation will either only display the densities of each particle type as a function of time or also display the trajectories depending on Repr_type

    Args:
    T (int): Number of time intervals to simulate.
    Repr_type (int): The type of representation: draw just densities/0 or also draw trajectories/1
    File_path_name: Where to save the video if the simulation possible

    """

    from System.SystemClass import init

    init()
    from System.SystemClass import SYSTEM
    from Interactions.TYPES.SPONTANEOUS import SpontaneousEvents

    T = int(10 * T)
    time0 = time.time()  # Starting time for the simulation

    Dens = [[[], []] for num in range(Numb_of_TYPES)]
    L, Linf = Update_L_Lmin(0)

    Dper = 0
    if Repr_type == 0 or Repr_type == 1:
        print("Generating Points")
        print(Dper, "%", end="\r")

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
        if perc > Dper and (Repr_type == 0 or Repr_type == 1):  # and perc%5==0
            Dper = perc
            print(Dper, "%", end="\r")

        t = ROUND(ti * dt)
        L, Linf = Update_L_Lmin(t)
        Global_variables.Update_Bound_Params(L, Linf, t)
        SYSTEM.UPDATE(t)
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
        # print("Energy", SYSTEM.TOTAL_ENERGY(), "\n")
        if ti != T - 1:
            SpontaneousEvents(t)

    Dt = round(time.time() - time0, 2)

    # post-simulation operations depending on the input parameter Repr_type, which determines the type of output the function produces.

    ALL_TIME = Global_variables.ALL_TIME
    ALL_TIME.extend([ROUND(i * dt) for i in range(T)])
    ALL_TIME = [*set(ALL_TIME)]
    ALL_TIME.sort()

    if (
        Repr_type == 1
    ):  # if Repr_type is equal to 1, then the function produces a graphical output of the simulation results using the DRAW function.
        print("Generating time:", Dt, "s")
        import Display.DisplayDRAW as DRAW_TRAJ

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
    from Settings.INPUT import SET_PARAMETERS

    (
        DIM_Numb,
        Time_steps,
        box_size,
        BOUNDARY_COND,
        Repr_mode,
        iniNparticles_set,
    ) = SET_PARAMETERS()

    LO, LOinf = np.array(box_size), np.array([0 for d in range(DIM_Numb)])
    # L_FCT = [lambda x: LO + np.cos(x), lambda x: LOinf + np.sin(x + 1)]
    L_FCT = [lambda x: LO, lambda x: LOinf]

    from Particles.Global_Variables import init

    init(DIM_Numb, BOUNDARY_COND, L_FCT)
    from Particles.Global_Variables import Global_variables
    from Misc.Functions import ROUND

    Vmax = Global_variables.Vmax
    dt = Global_variables.dt
    L_FCT = Global_variables.L_FCT

    main(Time_steps, Repr_mode, iniNparticles_set)
