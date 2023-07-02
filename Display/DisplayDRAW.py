import matplotlib.pyplot as plt


import numpy as np
from Display import Density

DENS_FCT = Density.Denfct


PARTICLE_DICT = {}

DIM_Numb = 0
dt = 0


def DRAW(
    Tt,
    dt0,
    Lparam,
    DIM_Numb0,
    COLPTS,
    TRACKING,
    Density,
    LFcts,
    BOUNDARY_COND,
    ALLtimes,
    PARTICLE_DICT0,
    File_path_name=None,
):
    """
    Draw trajectories and particle densities of a simulation.

    Parameters:
    -----------
    Tt: int
        Number of time steps in the simulation.
    dt: float
        Size of time step in the simulation.
    Lparam: float
        Size of the simulation box.
    DIM_Numb0: int
        Number of dimensions of the simulation.
    COLPTS: list
        List of colored points to plot. Each element is a tuple of two lists: the first one contains the x coordinates of
        the points, and the second one contains the y coordinates of the points. The third element is an integer that
        specifies the color of the points (0 for red, 1 for blue, 2 for black, and 3 for yellow).
    TRACKING: list
        List of particle trajectories to plot. Each element is a list of tuples, where each tuple contains two elements:
        the first one is the time at which the particle was tracked, and the second one is a tuple that contains the
        particle's position coordinates in each dimension.
    Density: list
        List of particle densities to plot. Each element is a tuple of two lists: the first one contains the time steps,
        and the second one contains the densities. The densities are represented as integers, and the time steps are
        represented as floats.
    DIM_Numb: int, optional (default=0)
        Number of dimensions of the simulation.
    ALL_TIME: list
        collection of all the time values (only needed in 2D case)

    Returns:
    --------
    None
    """
    global DIM_Numb, L, Linf, dt, PARTICLE_DICT
    dt = dt0
    Linf, L = Lparam
    DIM_Numb = DIM_Numb0
    PARTICLE_DICT = PARTICLE_DICT0
    # PART_Colors,Event_Colors,PART_Style,Event_Style,PART_LStyle,Event_LStyle

    if DIM_Numb != 3:
        Trange = np.linspace(0, (Tt - 1) * dt, len(Density[0]))
        # Generate the figure for plotting
        fig = plt.figure(0)
        # Plot densities as a function of time in one subplot
        ax2 = fig.add_subplot(1, 2, 2)
        DENS_FCT(DIM_Numb0, Density, Trange, ax2, PARTICLE_DICT)
        # Plot Trajectories as a function of time in another subplot
        ax = fig.add_subplot(1, 2, 1)
        if DIM_Numb == 1:
            from Display.DRAW1D import TRAJ_1D

            ax.set_xlabel("Time(s)")
            ax.set_ylabel("Position X")
            TRAJ_1D(ax, TRACKING, Trange, COLPTS, LFcts)
        else:
            from Display.DRAW2D import TRAJ_2D

            plt.subplots_adjust(bottom=0.225)
            ax.set_xlabel("Position X")
            ax.set_ylabel("Position Y")
            TRAJ_2D(fig, ax, TRACKING, ALLtimes, Density, COLPTS, LFcts, dt)
    else:
        from Display.DRAW3D import TRAJ_3D

        TRAJ_3D(
            dt,
            BOUNDARY_COND,
            Density,
            ALLtimes,
            TRACKING,
            COLPTS,
            LFcts,
            File_path_name,
        )
