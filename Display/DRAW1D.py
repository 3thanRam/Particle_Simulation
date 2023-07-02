import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Particles.Dictionary import PARTICLE_DICT


DIM_Numb = 1

Event_Colors = ["green", "yellow", "black", "purple"]

PART_Style = "_"
Event_Style = "o"
PART_LStyle = "-"
Event_LStyle = "None"


def TRAJ_1D(ax, TRACKING, Trange, COLPTS, L_HIST):
    """
    Plot trajectories of particles in for 1D case
    """
    ax.set_title(
        "Trajectory of particles \n as a function of Time\n  in "
        + str(DIM_Numb)
        + " Dimensions"
    )
    # Iterate over the particles and their trajectories to plot them
    Numb_of_TYPES = len(PARTICLE_DICT)
    PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
    PART_LStyle = ["-", "--"]
    lines = []
    labels = []
    for p1 in range(len(TRACKING)):
        for p0 in range(len(TRACKING[p1])):
            if len(TRACKING[p1][p0]) != 0:
                lines.append(
                    Line2D(
                        [],
                        [],
                        color=PARTICLE_DICT[PARTICLE_NAMES[p1]]["color"],
                        linewidth=3,
                        marker=PART_Style,
                        linestyle=PART_LStyle[p0],
                    )
                )
                labels.append(p0 * "ANTI-" + PARTICLE_NAMES[p1])
            for tr in range(len(TRACKING[p1][p0])):
                T = []
                X = [[] for i in range(DIM_Numb)]
                for elem in TRACKING[p1][p0][tr]:
                    Ts = elem[0]
                    if (
                        type(Ts) == float
                        or type(Ts) == np.float64
                        or type(Ts) == int
                        or type(Ts) == str
                    ):
                        T.append(Ts)
                    else:
                        T.append(Ts[0])
                    Xs = elem[1]
                    if type(Xs) == str:
                        for xd in range(DIM_Numb):
                            X[xd].append("X")
                    elif (
                        type(Xs[0]) == float
                        or type(Xs[0]) == np.ndarray
                        or type(Xs[0]) == np.float64
                    ):
                        for xd in range(DIM_Numb):
                            X[xd].append(Xs[xd])
                st = 0
                for i in range(T.count("T") + 1):
                    if T.count("T") > 0:
                        end = T.index("T")

                        if PARTICLE_NAMES[p1] == "photon":
                            with mpl.rc_context({"path.sketch": (5, 15, 1)}):
                                ax.plot(
                                    T[st:end],
                                    [Xi[st:end] for Xi in X][0],
                                    color=PARTICLE_DICT[PARTICLE_NAMES[p1]]["color"],
                                    linestyle=PART_LStyle[p0],
                                )
                        else:
                            ax.plot(
                                T[st:end],
                                [Xi[st:end] for Xi in X][0],
                                color=PARTICLE_DICT[PARTICLE_NAMES[p1]]["color"],
                                linestyle=PART_LStyle[p0],
                            )
                        T.remove("T")
                        for xd in range(DIM_Numb):
                            X[xd].remove("X")
                        st = end
                    else:
                        if PARTICLE_NAMES[p1] in [
                            "photon",
                            "w-boson",
                            "w+boson",
                            "zboson",
                        ]:
                            with mpl.rc_context({"path.sketch": (5, 15, 1)}):
                                ax.plot(
                                    T[st:],
                                    [Xi[st:] for Xi in X][0],
                                    color=PARTICLE_DICT[PARTICLE_NAMES[p1]]["color"],
                                    linestyle=PART_LStyle[p0],
                                )
                        else:
                            ax.plot(
                                T[st:],
                                [Xi[st:] for Xi in X][0],
                                color=PARTICLE_DICT[PARTICLE_NAMES[p1]]["color"],
                                linestyle=PART_LStyle[p0],
                            )

    # for tl in Trange:
    #    ax.vlines(tl,-2,32,'black','dotted')
    #   #ax.vlines(tl,Linf[0],L[0],'black','dotted')
    for col in COLPTS:
        ax.plot(
            col[0], col[1], ms=4, marker=Event_Style, color=Event_Colors[col[2] - 1]
        )

    ax.plot(Trange, L_HIST[0](Trange), color="black")
    ax.plot(Trange, L_HIST[1](Trange), color="black")

    # lines = [Line2D([], [], color=PART_Colors[c], linewidth=3, marker=PART_Style,linestyle=PART_LStyle) for c in range(3)]+[Line2D([], [], color=Event_Colors[c], linewidth=3, marker=Event_Style,linestyle=Event_LStyle) for c in range(4)]
    labels.extend(["Absorption", "Annihilation", "Collision", "Creation"])
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
    ax.legend(lines, labels)
    plt.tight_layout()
    # plt.savefig('Traj1D')

    plt.show()
