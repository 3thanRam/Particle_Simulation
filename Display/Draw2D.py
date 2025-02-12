from Display.Complete_data import (
    COMPLETE_DATAPTS,
)
from Particles.Dictionary import PARTICLE_DICT
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation
import matplotlib.collections

from Particles.Global_Variables import Global_variables

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
DIM_Numb = 2


def TRAJ_2D(fig, ax, TRACKING, ALLtimes, Density, COLPTS, LFct, dt):
    """
    Plot trajectories of particles in 2D case
    """
    Extrtime = dt  # prolong display time to help visualisation
    anim_running = True
    IntervalVAR = 0
    ACTIVE_PARTS = []
    for typeindex, densType in enumerate(Density):
        for antiorpart, antiorpartDens in enumerate(densType):
            if antiorpartDens != [0 for i in range(len(antiorpartDens))]:
                ACTIVE_PARTS.append([antiorpart, typeindex])

    TRACKING = COMPLETE_DATAPTS(TRACKING, ALLtimes, 2)

    class Index:
        ind = 0

        def PlayPause(self, event):
            """
            Pause or Play animation
            """
            nonlocal anim_running
            if anim_running:
                ani.pause()
                anim_running = False
            else:
                ani.resume()
                anim_running = True

        def SpeedDown(self, event):
            """
            Decrease fps of animation
            """
            nonlocal ani
            ani.event_source.interval *= 1.25
            print("ms between frames", ani.event_source.interval, end="\r")

            ani.event_source.stop()
            # ani.frame_seq = ani.new_frame_seq()
            ani = animation.FuncAnimation(
                fig,
                run,
                data_gen,
                blit=False,
                interval=ani.event_source.interval,
                repeat=True,
                save_count=len(ALLtimes) - 2,
            )
            ani.event_source.start()
            event.canvas.draw()
            fig.canvas.draw()

        def SpeedUp(self, event):
            """
            Increase fps of animation
            """
            nonlocal ani
            ani.event_source.interval *= 0.75
            print("ms between frames", ani.event_source.interval, end="\r")

            ani.event_source.stop()
            ani = animation.FuncAnimation(
                fig,
                run,
                data_gen,
                blit=False,
                interval=ani.event_source.interval,
                repeat=True,
                save_count=len(ALLtimes) - 2,
            )
            ani.event_source.start()
            event.canvas.draw()
            fig.canvas.draw()

    callback = Index()

    SpeeddownAx = fig.add_axes([0.1, 0.05, 0.13, 0.075])
    Downbutton = Button(SpeeddownAx, "Slow Down")
    Downbutton.on_clicked(callback.SpeedDown)

    PlaypauseAx = fig.add_axes([0.25, 0.05, 0.13, 0.075])
    Pbutton = Button(PlaypauseAx, "Play/Pause")
    Pbutton.on_clicked(callback.PlayPause)

    SpeedupAx = fig.add_axes([0.40, 0.05, 0.13, 0.075])
    Upbutton = Button(SpeedupAx, "Speed Up")
    Upbutton.on_clicked(callback.SpeedUp)

    def data_gen():
        """
        Generates data about the points to be plotted at each instant
        """
        t = data_gen.t
        time_ind = 0
        while t < ALLtimes[-2]:
            P_TYPE = [[[], []] for i in range(len(ACTIVE_PARTS))]
            time_ind += 1
            t = ALLtimes[time_ind]

            for ActPart_index, (antiorpartindex, particleindex) in enumerate(
                ACTIVE_PARTS
            ):
                for partdata in TRACKING[particleindex][antiorpartindex]:
                    Part_data = partdata[time_ind]
                    Part_Time = Part_data[0]
                    if not isinstance(Part_Time, str):
                        Part_Xpos, Part_Ypos = Part_data[1]
                        P_TYPE[ActPart_index][0].append(Part_Xpos)
                        P_TYPE[ActPart_index][1].append(Part_Ypos)

            COLS, ANNILS, ABSO, CREAT = [[], []], [[], []], [[], []], [[], []]
            for colpts in COLPTS:
                interT = colpts[0]
                if abs(ALLtimes[time_ind] - interT) <= Extrtime:
                    if colpts[-1] == 3:
                        COLS[0].append(colpts[1][0])
                        COLS[1].append(colpts[1][1])
                    elif colpts[-1] == 2:
                        ANNILS[0].append(colpts[1][0])
                        ANNILS[1].append(colpts[1][1])
                    elif colpts[-1] == 1:
                        ABSO[0].append(colpts[1][0])
                        ABSO[1].append(colpts[1][1])
                    else:
                        CREAT[0].append(colpts[1][0])
                        CREAT[1].append(colpts[1][1])

            PTS_data = [COLS, ANNILS, ABSO, CREAT] + P_TYPE
            yield time_ind, t, PTS_data

    data_gen.t = 0
    ax.set_title(
        "Animation of Particle-Antiparticle\n Trajectories in a "
        + str(DIM_Numb)
        + "D box over Time"
    )
    ax.grid()

    # initialise pens
    PARTICLE_PENS = []

    for antiOrpartind, Pindex in ACTIVE_PARTS:
        PART_SIZE = PARTICLE_DICT[PARTICLE_NAMES[Pindex]]["size"]
        PartColour = [PARTICLE_DICT[PARTICLE_NAMES[Pindex]]["color"], "white"]
        PARTICLE_PENS.append(
            matplotlib.collections.EllipseCollection(
                PART_SIZE,
                PART_SIZE,
                np.zeros_like(PART_SIZE),
                offsets=[],
                units="xy",
                transOffset=ax.transData,
                edgecolors=[PartColour[0], PartColour[0]],
                facecolors=PartColour[antiOrpartind],
            )
        )

    Interaction_Size = 100 * Global_variables.Dist_min**2

    COLLISION = matplotlib.collections.StarPolygonCollection(
        numsides=7,
        offsets=[],
        transOffset=ax.transData,
        sizes=(Interaction_Size,),
        zorder=2.5,
        facecolors="black",
    )
    ANNIHILATION = matplotlib.collections.StarPolygonCollection(
        numsides=7,
        offsets=[],
        transOffset=ax.transData,
        sizes=(Interaction_Size,),
        zorder=2.5,
        facecolors="yellow",
    )
    ABSORPTION = matplotlib.collections.StarPolygonCollection(
        numsides=7,
        offsets=[],
        transOffset=ax.transData,
        sizes=(Interaction_Size,),
        zorder=2.5,
        facecolors="green",
    )
    CREATIONS = matplotlib.collections.StarPolygonCollection(
        numsides=7,
        offsets=[],
        transOffset=ax.transData,
        sizes=(Interaction_Size,),
        zorder=2.5,
        facecolors="purple",
    )

    def DRAWBOX(time):
        """
        Get Upper and lower boundaries of the box in order to draw it on the figure
        """
        Xmax, Ymax = LFct[0](time)
        Xmin, Ymin = LFct[1](time)

        Xval = [Xmax, Xmax, Xmin, Xmin, Xmax]
        Yval = [Ymax, Ymin, Ymin, Ymax, Ymax]
        return (Xval, Yval)

    Xbox, Ybox = DRAWBOX(0)
    (BOXpen,) = ax.plot(Xbox, Ybox, color="black")

    def Disp(tparam, t_ind, PARTLIST):
        """
        Get information about current time and number of particles to display at a given time tparam
        """
        Npartparam = [len(partdata[0]) for partdata in PARTLIST]
        return "Time:" + str(round(tparam, 2)) + "s \n Numb:" + str(Npartparam)

    time_text = ax.text(
        LFct[1](0)[0] + 0.5, LFct[1](0)[1] + 0.5, Disp(0, 0, ["0"]), fontsize=7
    )
    PT_PENS = [COLLISION, ANNIHILATION, ABSORPTION, CREATIONS] + PARTICLE_PENS
    for ptCollection in PT_PENS:
        ax.add_collection(ptCollection)

    labels = []
    lines = []

    # PARTICLE LEGEND
    PART_LStyle = "None"
    Particle_Colors = []
    Particle_EdgeColors = []
    for antiOrpartind, Pindex in ACTIVE_PARTS:
        labels.append("anti" * antiOrpartind + PARTICLE_NAMES[Pindex])
        Particle_EdgeColors.append(PARTICLE_DICT[PARTICLE_NAMES[Pindex]]["color"])
        if antiOrpartind == 0:
            Particle_Colors.append(PARTICLE_DICT[PARTICLE_NAMES[Pindex]]["color"])
        else:
            Particle_Colors.append("white")

    lines.extend(
        [
            Line2D(
                [],
                [],
                color=Particle_Colors[c],
                markeredgecolor=Particle_EdgeColors[c],
                linewidth=3,
                marker="o",
                linestyle=PART_LStyle,
            )
            for c in range(len(ACTIVE_PARTS))
        ]
    )

    # EVENT LEGEND
    Event_Style = "x"
    Event_LStyle = "None"
    labels.extend(["Absorption", "Annihilation", "Collision", "Creation"])
    Event_Colors = ["green", "yellow", "black", "purple"]
    lines.extend(
        [
            Line2D(
                [],
                [],
                color=Event_Colors[c],
                linewidth=3,
                marker=Event_Style,
                linestyle=Event_LStyle,
            )
            for c in range(4)
        ]
    )

    ax.legend(lines, labels, prop=dict(size=8), loc="upper right")

    def run(data):
        """
        Called by the FuncAnimation function and Draws the points given by the data_gen function
        """
        # global legend
        ti, t, PTS_data = data

        for pen_ind, pen in enumerate(
            PT_PENS
        ):  # Draw each particle and interaction point
            pen.set_offsets(np.c_[PTS_data[pen_ind][0], PTS_data[pen_ind][1]])
        time_text.set_text(
            Disp(t, ti, PTS_data[4:])
        )  # Display time & numb of particles
        Xbox, Ybox = DRAWBOX(t)
        BOXpen.set_data(Xbox, Ybox)  # Display Box lengths

        Xzoom = 0.15 * abs(Xbox[0] - Xbox[2])
        Yzoom = 0.15 * abs(Ybox[0] - Ybox[1])

        ax.set_xlim(Xbox[2] - Xzoom, Xbox[0] + Xzoom)
        ax.set_ylim(Ybox[1] - Yzoom, Ybox[0] + Yzoom)

        return PT_PENS, time_text, BOXpen

    # Create animation from series of images of matplotlib figures updated at each time instant
    IntervalVAR = int(750 * ALLtimes[-1] / (len(ALLtimes) - 2))
    ani = animation.FuncAnimation(
        fig,
        run,
        data_gen,
        blit=False,
        interval=IntervalVAR,
        repeat=True,
        save_count=len(ALLtimes) - 2,
    )

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('Traj2D.gif', writer='imagemagick',fps=60)
    # ani.save('Traj2D.gif',writer=writer)

    plt.show()
