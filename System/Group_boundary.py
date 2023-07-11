import numpy as np
from Particles.Global_Variables import Global_variables

DIM_Numb = Global_variables.DIM_Numb

DIM_Numb = Global_variables.DIM_Numb
Vmax = Global_variables.Vmax
dt = Global_variables.dt
distmax = 1.5 * Vmax * dt
from Misc.Functions import COUNTFCT


def Group_close_bounds(PARAMS, CHGind, Xi, Xf, BOUNDARYCHECKS):
    for Bcheck in BOUNDARYCHECKS:
        if len(Bcheck) > 0:
            PARAMS.append([])
            CHGind.append([])
            for PART_B_inf_index in Bcheck:
                for d in range(DIM_Numb):
                    elem_ind = np.where(
                        (Xf[d]["index"] == Xi[d]["index"][PART_B_inf_index])
                        & (Xi[d]["TypeID0"][PART_B_inf_index] == Xf[d]["TypeID0"])
                        & (Xi[d]["TypeID1"][PART_B_inf_index] == Xf[d]["TypeID1"])
                    )[0][0]
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
                    PARAMS[-1].append([ID, TYPE0, TYPE1])
                    CHGind[-1].append(elem_ind)
    return PARAMS
