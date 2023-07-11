import numpy as np
from Particles.Global_Variables import Global_variables

DIM_Numb = Global_variables.DIM_Numb

DIM_Numb = Global_variables.DIM_Numb
Vmax = Global_variables.Vmax
dt = Global_variables.dt
distmax = 1.5 * Vmax * dt
from Misc.Functions import COUNTFCT

from System.Group_boundary import Group_close_bounds
from System.Reduce_Overlap import REMOVE_OVERLAP


def Group_particles(Xi, Xf, BOUNDARYCHECKS):
    # dtype = [("Pos", float), ("TypeID0", int), ("TypeID1", int), ("index", int)]
    CHANGESdim = [
        np.where(
            (Xi[d]["index"] != Xf[d]["index"])
            | (Xi[d]["TypeID0"] != Xf[d]["TypeID0"])
            | (Xi[d]["TypeID1"] != Xf[d]["TypeID1"])
        )
        for d in range(DIM_Numb)
    ]
    CHANGESdim2 = [
        np.where((Xi[d]["index"] == Xi[d]["index"])) for d in range(DIM_Numb)
    ]
    if DIM_Numb == 1:
        CHANGES = CHANGESdim[0][0]
    elif DIM_Numb == 2:
        CHANGES = np.intersect1d(*CHANGESdim2)
    elif DIM_Numb == 3:
        CHANGES = np.intersect1d(
            np.intersect1d(CHANGESdim[0], CHANGESdim[1]), CHANGESdim[2]
        )
    CHGind = []  # index in Xi/Xf
    # particle involved in interactions parameters
    PARAMS = []  # each elem is list of [type0,type1,ID]
    for (
        chg
    ) in (
        CHANGES
    ):  #  particles of index between ini and end could interact with the particle
        CHGind.append([])
        # for parametr in PARAMS:
        PARAMS.append([])
        for d in range(DIM_Numb):
            matchval = np.where(
                (Xi[d]["index"][chg] == Xf[d]["index"])
                & (Xi[d]["TypeID0"][chg] == Xf[d]["TypeID0"])
                & (Xi[d]["TypeID1"][chg] == Xf[d]["TypeID1"])
            )[0][0]
            Xa, Xb = Xf[d]["Pos"][matchval], Xi[d]["Pos"][chg]
            Xinf = np.min((Xa, Xb))
            Xsup = np.max((Xa, Xb))
            Inflist = np.where(
                (Xf[d]["Pos"] <= Xinf) & (Xf[d]["Pos"] >= Xsup - distmax[d])
            )[0]
            Suplist = np.where(
                (Xf[d]["Pos"] >= Xsup) & (Xf[d]["Pos"] <= Xinf + distmax[d])
            )[0]
            if Suplist.size == 0:
                maxi = max(chg, matchval) + 1
            else:
                maxi = Suplist[Xf[d]["Pos"][Suplist].argmax()] + 2
            if Inflist.size == 0:
                mini = min(chg, matchval)
            else:
                mini = Inflist[Xf[d]["Pos"][Inflist].argmin()] - 1
            mini = max(0, mini)
            maxi = min(maxi, len(Xf[d]))
            for elem_ind in range(mini, maxi):
                POS, TYPE0, TYPE1, ID = (
                    Xf[d]["Pos"][elem_ind],
                    Xf[d]["TypeID0"][elem_ind],
                    Xf[d]["TypeID1"][elem_ind],
                    Xf[d]["index"][elem_ind],
                )
                POSLIST = [[] for d in range(DIM_Numb)]
                POSLIST[d] = POS
                for d2 in range(1, DIM_Numb):
                    d2 += d
                    if d2 >= DIM_Numb:
                        d2 -= DIM_Numb
                    posind2 = np.where(
                        (Xf[d2]["index"] == ID)
                        & (Xf[d2]["TypeID0"] == TYPE0)
                        & (Xf[d2]["TypeID1"] == TYPE1)
                    )[0][0]
                    POSLIST[d2] = Xf[d2]["Pos"][posind2]
                if CHGind[-1] == [] or COUNTFCT(PARAMS[-1], [ID, TYPE0, TYPE1], 1) == 0:
                    CHGind[-1].append(elem_ind)
                    PARAMS[-1].append([ID, TYPE0, TYPE1])
        if len(PARAMS[-1]) == 0:
            PARAMS.remove([])
            CHGind.remove([])
    # if particles interact with bounds then they can interact with particles differently (e.g particles at top and bottom of box can interact if periodic bounds) than those accounted for above

    PARAMS = Group_close_bounds(PARAMS, CHGind, Xi, Xf, BOUNDARYCHECKS)

    return REMOVE_OVERLAP(PARAMS)
