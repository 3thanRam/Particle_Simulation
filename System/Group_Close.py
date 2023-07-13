import numpy as np
from Particles.Global_Variables import Global_variables
from Misc.Functions import COUNTFCT
from System.Group_boundary import Group_close_bounds
from System.Reduce_Overlap import REMOVE_OVERLAP
from operator import itemgetter

DIM_Numb = Global_variables.DIM_Numb
DIM_Numb = Global_variables.DIM_Numb
Vmax = Global_variables.Vmax
dt = Global_variables.dt
distmax = Global_variables.distmax
get_fct = itemgetter(1, 2, 3)


def Group_particles(Xi, Xf):
    # particle involved in interactions parameters
    PARAMS = []  # each elem will be a list [type0,type1,ID]
    for ind_Xi in range(
        len(Xi[0])
    ):  #  particles of index between ini and end and within distmax+particle_size of those points could interact with the particle
        PARAMS.append([])
        for d in range(DIM_Numb):
            for xnumb, X_list in enumerate([Xi[d], Xf[d]]):
                if xnumb == 0:
                    ind_X = ind_Xi
                else:
                    ind_X = np.where(
                        (Xi[d]["index"][ind_Xi] == Xf[d]["index"])
                        & (Xi[d]["TypeID0"][ind_Xi] == Xf[d]["TypeID0"])
                        & (Xi[d]["TypeID1"][ind_Xi] == Xf[d]["TypeID1"])
                    )[0][0]

                X_inf = X_list["Pos"][ind_X] - distmax
                X_sup = X_list["Pos"][ind_X] + distmax
                Close_X_inds = np.where(
                    (X_list["Pos"] >= X_inf) & (X_list["Pos"] <= X_sup)
                )[0]
                for ind_close in Close_X_inds:
                    TYPE0, TYPE1, ID = get_fct(X_list[ind_close])
                    if COUNTFCT(PARAMS[-1], [ID, TYPE0, TYPE1], 1) == 0:
                        PARAMS[-1].append([ID, TYPE0, TYPE1])
        if len(PARAMS[-1]) == 0:
            PARAMS.remove([])

    # if particles interact with bounds then they can interact with particles differently (e.g particles at top and bottom of box can interact if periodic bounds) than those accounted for above
    PARAMS = Group_close_bounds(PARAMS, Xi, Xf)

    return REMOVE_OVERLAP(PARAMS)
