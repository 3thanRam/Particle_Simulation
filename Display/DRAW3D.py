import numpy as np
from matplotlib import colors

from Particles.Dictionary import PARTICLE_DICT


Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]


def TRAJ_3D(
    dt, BOUNDARY_COND, Density, ALL_TIME, TRACKING, COLPTS, L_HIST, File_path_name
):
    """
    Save a 3D video Plot of a set of particles

    Parameters:
    -----------
    dt: float
        Size of time step in the simulation.
    BOUNDARY_COND: int
        Represents if boundaries are periodic(0) or hard(1)
    ALL_TIME: list
        collection of all the time values
    TRACKING: list
        List of particle trajectories to plot. Each element is a list of tuples, where each tuple contains two elements:
        the first one is the time at which the particle was tracked, and the second one is a tuple that contains the
        particle's position coordinates in each dimension.
    COLPTS: list
        List of colored points to plot. Each element is a tuple of two lists: the first one contains the x coordinates of
        the points, and the second one contains the y coordinates of the points. The third element is an integer that
        specifies the color of the points (0 for red, 1 for blue, 2 for black, and 3 for yellow).
    File_path_name: str
        Path to where to save the video

    """
    from mayavi import mlab
    import imageio
    import time
    from scipy.special import sph_harm
    from Display.COMPLETE_DATA import COMPLETE_DATAPTS

    Time = ALL_TIME

    # Particles gain extra time values during collisions which causes some particles to have more saved points than others and at different times
    # this loop adds the necesary points by interpolating the position before/after the time value to be added
    TRACKING = COMPLETE_DATAPTS(TRACKING, Time, 3)

    ACTIVE_PARTS = []
    for typeindex, densType in enumerate(Density):
        for antiorpart, antiorpartDens in enumerate(densType):
            if antiorpartDens != [0 for i in range(len(antiorpartDens))]:
                ACTIVE_PARTS.append([antiorpart, typeindex])

    Part_AntiPart_Data = [
        [
            [[[], [], []] for tnumb in range(len(ALL_TIME))],
            [[[], [], []] for tnumb in range(len(ALL_TIME))],
        ]
        for p in range(len(ACTIVE_PARTS))
    ]
    for ActPart_index, (antiorpartindex, particleindex) in enumerate(ACTIVE_PARTS):
        Part_AntiPart_Data.append([[]])
    for time_ind, tval in enumerate(ALL_TIME):
        for ActPart_index, (antiorpartindex, particleindex) in enumerate(ACTIVE_PARTS):
            for partdata in TRACKING[particleindex][antiorpartindex]:
                if not isinstance(partdata[0][time_ind], str):
                    for d in range(3):
                        Part_AntiPart_Data[ActPart_index][antiorpartindex][time_ind][
                            d
                        ].append(partdata[d][time_ind])

    COLPTS = [[colpts[0], *colpts[1], colpts[2]] for colpts in COLPTS]
    COLPTS = np.array(COLPTS)

    from Particles.Global_Variables import Global_variables

    PartSize = Global_variables.Dist_min / 2

    ExtTparam_min = 3 * dt
    ExtTparam_max = 3 * dt

    # create list of values which will be used to draw collisions/annihilations
    # Each added "pen" slows the program so it's faster to use the minimum possible and reuse them
    # Hence only initiliase a number equal to the max number of Annihilations/Collisions that will be drawn at the same time
    # (Note: Annihilations/Collisions are seperated even though the only difference is color because changing the color of a "pen" requires resetting it so it's just as fast and easier to describe by seperating them )
    if COLPTS.ndim > 1:
        freqmaxCol = 0
        freqmaxAnn = 0
        freqmaxAbs = 0
        freqmaxCre = 0
        OnlyCreat = COLPTS[np.where(COLPTS[:, -1] == 4)]
        Onlycol = COLPTS[np.where(COLPTS[:, -1] == 3)]
        OnlyAnn = COLPTS[np.where(COLPTS[:, -1] == 2)]
        OnlyAbso = COLPTS[np.where(COLPTS[:, -1] == 1)]
        for timef in Time:
            freqc = Onlycol[
                (Onlycol[:, 0] <= timef + ExtTparam_max)
                & (Onlycol[:, 0] >= max(timef - ExtTparam_min, 0))
            ].shape[0]
            freqa = OnlyAnn[
                (OnlyAnn[:, 0] <= timef + ExtTparam_max)
                & (OnlyAnn[:, 0] >= max(timef - ExtTparam_min, 0))
            ].shape[0]
            freqabso = OnlyAbso[
                (OnlyAbso[:, 0] <= timef + ExtTparam_max)
                & (OnlyAbso[:, 0] >= max(timef - ExtTparam_min, 0))
            ].shape[0]
            freqCreate = OnlyCreat[
                (OnlyCreat[:, 0] <= timef + ExtTparam_max)
                & (OnlyCreat[:, 0] >= max(timef - ExtTparam_min, 0))
            ].shape[0]

            freqmaxCol = max(freqc, freqmaxCol)
            freqmaxAnn = max(freqa, freqmaxAnn)
            freqmaxAbs = max(freqabso, freqmaxAbs)
            freqmaxCre = max(freqCreate, freqmaxCre)
    else:
        freqmaxCol = 0
        freqmaxAnn = 0
        freqmaxAbs = 0
        freqmaxCre = 0

    # I couldn't find a simple way to draw a star for Annihilations/Collisions so I decided to draw a harmonic function instead
    m, n = 3, 4
    r = 0.3
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:21j, 0 : 2 * pi : 21j]
    s = sph_harm(m, n, theta, phi).real
    s[s < 0] *= 0.97
    s /= s.max()
    x_v = PartSize * s * r * sin(phi) * cos(theta)
    y_v = PartSize * s * r * sin(phi) * sin(theta)
    z_v = PartSize * s * r * cos(phi)

    def GEN_CUBOID(L_up, L_down):
        """
        Generate points to draw 3D cuboid

        Args:
            L_up (List): list of end points of box
            L_down (_type_): list of start points of box
        """
        radius = 5  # particle radius to make bouncing of wall look more realistic
        R0 = [
            np.linspace(-radius + L_down[i], L_up[i] + radius, 10**3)
            for i in range(3)
        ]
        RL = [(L_up[i] + radius) * np.ones_like(R0[0]) for i in range(3)]
        R_zero = [(-radius + L_down[i]) * np.ones_like(R0[0]) for i in range(3)]
        RA1 = [R_zero[1], RL[1]]
        RA2 = [R_zero[2], RL[2]]
        RB1 = [R_zero[0], RL[0]]
        RB2 = [R_zero[2], RL[2]]
        RC1 = [R_zero[0], RL[0]]
        RC2 = [R_zero[1], RL[1]]
        PTLIST = []
        for i in range(2):
            for ii in range(2):
                X, Y, Z = (
                    np.concatenate((R0[0], RB1[i], RC1[i])),
                    np.concatenate((RA1[i], R0[1], RC2[ii])),
                    np.concatenate((RA2[ii], RB2[ii], R0[2])),
                )
                PTLIST.append([X, Y, Z])
        return PTLIST

    # I couldn't find a clear way to display the box using the grid so I create visible boundaries by drawing the box
    Lsup, Lmin = L_HIST[0](Time[0]), L_HIST[1](Time[0])
    CUBEDATA = GEN_CUBOID(Lsup, Lmin)
    Xlist, Ylist, Zlist = np.array([]), np.array([]), np.array([])
    for Xc, Yc, Zc in CUBEDATA:
        Xlist = np.concatenate((Xlist, Xc))
        Ylist = np.concatenate((Ylist, Yc))
        Zlist = np.concatenate((Zlist, Zc))
    # print(Time[0],Xlist.shape,Ylist.shape,Zlist.shape)

    # BOUND_DRAW=[mlab.plot3d(Xc,Yc,Zc,line_width=0.01,color=(0,0,0),tube_radius=0.1).mlab_source for Xc,Yc,Zc in CUBEDATA]

    DENSITY = [[0, 0] for p in range(len(ACTIVE_PARTS))]

    for ActPart_index, (antiorpartindex, particleindex) in enumerate(ACTIVE_PARTS):
        DENSITY[ActPart_index][antiorpartindex] = [
            len(Part_AntiPart_Data[ActPart_index][antiorpartindex][tvalind][0])
            for tvalind in range(len(Time))
        ]

    # for tvalind in range(len(Time)):
    #    DENSITY[0].append(len(Part_AntiPart_Data[0][tvalind][0]))
    #    DENSITY[1].append(len(Part_AntiPart_Data[1][tvalind][0]))
    #    DENSITY[2].append(len(Part_AntiPart_Data[2][tvalind][0]))
    SAVE_MODE = 0

    if SAVE_MODE == 1:
        mlab.options.offscreen = True  # Stops the view window popping up and makes sure you get the correct size screenshots.
    # Set up the figure
    width, height = 320, 320
    fig = mlab.figure(size=(width, height))
    # Initialise the "pens" that will draw the particles/antiparticles
    PART_OR_ANTI_MARKER = ["sphere", "cube"]

    PARTICLE_PENS = []
    for ActPart_index, (antiorpartindex, particleindex) in enumerate(ACTIVE_PARTS):
        PARTICLE_PENS.append(
            mlab.points3d(
                [0.5],
                [0.5],
                [0.5],
                color=colors.to_rgb(
                    PARTICLE_DICT[PARTICLE_NAMES[particleindex]]["color"]
                ),
                mode=PART_OR_ANTI_MARKER[antiorpartindex],
                scale_mode="scalar",
                scale_factor=PartSize,
            ).mlab_source
        )

    def Disp(tparam, tparam_index):
        Npartparam = [
            "\n   "
            + "Anti" * antiorpartindex
            + PARTICLE_NAMES[particleindex]
            + ":"
            + str(DENSITY[ActPart_index][antiorpartindex][tparam_index])
            for ActPart_index, (antiorpartindex, particleindex) in enumerate(
                ACTIVE_PARTS
            )
        ]
        Npartparam = "".join(Npartparam)
        return "Time:" + str(round(tparam, 2)) + "s \n Numb Particles:" + Npartparam

    # Initialise the "pens" that will draw the Annihilations/Collisions
    COLLISION = [
        mlab.mesh(0 * x_v, 0 * y_v, 0 * z_v, scalars=s, color=(1, 1, 0)).mlab_source
        for i in range(freqmaxCol)
    ]
    ANNIHILATION = [
        mlab.mesh(0 * x_v, 0 * y_v, 0 * z_v, scalars=s, color=(0, 0, 0)).mlab_source
        for i in range(freqmaxAnn)
    ]
    ABSO = [
        mlab.mesh(0 * x_v, 0 * y_v, 0 * z_v, scalars=s, color=(0, 1, 0)).mlab_source
        for i in range(freqmaxAbs)
    ]
    CREAT = [
        mlab.mesh(0 * x_v, 0 * y_v, 0 * z_v, scalars=s, color=(1, 0, 1)).mlab_source
        for i in range(freqmaxCre)
    ]
    BOUND_DRAW0 = mlab.plot3d(
        Xlist, Ylist, Zlist, line_width=0.01, color=(0, 0, 0), tube_radius=0.1
    ).mlab_source
    TIME_DISP = mlab.text(
        0, 0, str(0), width=0.6
    )  # initiliase time and particle number display

    def SingularScene(i):
        f.scene.disable_render = True
        update_scene(i)
        f.scene.disable_render = False

    def update_scene(i):
        theta = (
            0.5 * i * 360 / len(Time)
        )  # angle to spin camera to avoid blocking certain parts of the box
        Lsup, Lmin = L_HIST[0](Time[i]), L_HIST[1](Time[i])
        mlab.view(
            azimuth=theta,
            elevation=theta / 2,
            distance=3.5 * max(Lsup - Lmin),
            focalpoint=(
                (Lsup[0] + Lmin[0]) / 2,
                (Lsup[1] + Lmin[1]) / 2,
                (Lsup[2] + Lmin[2]) / 2,
            ),
        )  # ,roll=theta
        TIME_DISP.set(text=Disp(Time[i], i))  # Display time and particle number
        # TIME_DISP.set(text='Time={0:.2f},Npart:{1},Nanti:{2}'.format(round(Time[i], 2),DENSITY[0][i],DENSITY[1][i]))#Display time and particle number

        CUBEDATA = GEN_CUBOID(Lsup, Lmin)
        Xlist, Ylist, Zlist = np.array([]), np.array([]), np.array([])
        for Xc, Yc, Zc in CUBEDATA:
            Xlist = np.concatenate((Xlist, Xc))
            Ylist = np.concatenate((Ylist, Yc))
            Zlist = np.concatenate((Zlist, Zc))
        BOUND_DRAW0.set(x=Xlist, y=Ylist, z=Zlist)
        if COLPTS.ndim > 1:
            COL = COLPTS[
                np.where(
                    (COLPTS[:, 0] < Time[i] + ExtTparam_max)
                    & (COLPTS[:, 0] > max(Time[i] - ExtTparam_min, 0))
                )
            ]
        else:
            COL = []
        c, a, ab, cr = 0, 0, 0, 0
        # Plot Collisions/Annihilations
        for col in COL:
            if col[-1] == 3:
                if (COLLISION[c].x != 0 * x_v).all():
                    COLLISION[c].set(x=x_v + col[1], y=y_v + col[2], z=z_v + col[3])
                else:
                    COLLISION[c].reset(x=x_v + col[1], y=y_v + col[2], z=z_v + col[3])
                c += 1
            elif col[-1] == 2:
                if (ANNIHILATION[a].x != 0 * x_v).all():
                    ANNIHILATION[a].set(x=x_v + col[1], y=y_v + col[2], z=z_v + col[3])
                else:
                    ANNIHILATION[a].reset(
                        x=x_v + col[1], y=y_v + col[2], z=z_v + col[3]
                    )
                a += 1
            elif col[-1] == 1:
                if (ABSO[ab].x != 0 * x_v).all():
                    ABSO[ab].set(x=x_v + col[1], y=y_v + col[2], z=z_v + col[3])
                else:
                    ABSO[ab].reset(x=x_v + col[1], y=y_v + col[2], z=z_v + col[3])
                ab += 1
            else:
                if (CREAT[cr].x != 0 * x_v).all():
                    CREAT[cr].set(x=x_v + col[1], y=y_v + col[2], z=z_v + col[3])
                else:
                    CREAT[cr].reset(x=x_v + col[1], y=y_v + col[2], z=z_v + col[3])
                cr += 1
        # Reset unused Collisions/Annihilations "pens" to make them disappear
        for cempty in range(len(COLLISION) - c):
            if (COLLISION[-1 - cempty].x != 0 * x_v).all():
                COLLISION[-1 - cempty].reset(
                    x=0 * x_v, y=0 * y_v, z=0 * z_v, color=(0, 1, 0)
                )
        for aempty in range(len(ANNIHILATION) - a):
            if (ANNIHILATION[-1 - aempty].x != 0 * x_v).all():
                ANNIHILATION[-1 - aempty].reset(
                    x=0 * x_v, y=0 * y_v, z=0 * z_v, color=(0, 1, 0)
                )
        for abempty in range(len(ABSO) - ab):
            if (ABSO[-1 - abempty].x != 0 * x_v).all():
                ABSO[-1 - abempty].reset(
                    x=0 * x_v, y=0 * y_v, z=0 * z_v, color=(0, 1, 0)
                )
        for Creatempty in range(len(CREAT) - cr):
            if (CREAT[-1 - Creatempty].x != 0 * x_v).all():
                CREAT[-1 - Creatempty].reset(
                    x=0 * x_v, y=0 * y_v, z=0 * z_v, color=(0, 1, 0)
                )
        # Plot Particles

        # for ActPart_index,(antiorpartindex,particleindex) in enumerate(ACTIVE_PARTS):
        for pen_ind, pen in enumerate(
            PARTICLE_PENS
        ):  # Draw each particle and interaction point
            antiorpartindex, particleindex = ACTIVE_PARTS[pen_ind]
            PartDensity = DENSITY[pen_ind][antiorpartindex]
            if PartDensity[i] != 0:
                X, Y, Z = Part_AntiPart_Data[pen_ind][antiorpartindex][i]
                if PartDensity[i] != PartDensity[i - 1] or i == 0:
                    pen.reset(x=X, y=Y, z=Z)
                else:
                    pen.set(x=X, y=Y, z=Z)
            elif PartDensity[i - 1] != 0 and i != 0:
                pen.reset(x=[], y=[], z=[])

    if SAVE_MODE == 0:

        @mlab.animate(delay=10)  # ,ui=False)
        def anim():
            f = mlab.gcf()
            for i in range(len(Time)):
                f.scene.disable_render = True
                update_scene(i)
                f.scene.disable_render = False
                # if i<len(Time)-1:
                #    Ani.delay=int(10+(Time[i+1]-Time[i])*1000)
                yield

    if SAVE_MODE == 0:
        Ani = anim()
        mlab.show()
    else:
        f = mlab.gcf()
        f.scene.disable_render = True  # turning off rendering when not necesary greatly reduces computation time
        BO = [
            "_Periodic",
            "_BOUNCE",
        ]  # text to add to name of file to distinguish boundary types
        writer = imageio.get_writer(
            File_path_name + BO[BOUNDARY_COND] + ".mp4", fps=15
        )  # Writer object which can be used to write data and meta data to the specified file

        Dper = 0
        # loop to get screenshot at each time value
        for i in range(len(Time)):
            perc = int(100 * i / len(Time))
            if perc > Dper and perc % 5 == 0:
                Dper = perc
                print(Dper, "%", end="\r")
            SingularScene(i)
            writer.append_data(
                mlab.screenshot(mode="rgb", antialiased=True)
            )  # add each screenshot to writer for it to convert to video
        writer.close()
        print("Done")
