import numpy as np
from Particles.Global_Variables import Global_variables
from operator import itemgetter
from Misc.Functions import COUNTFCT

DIM_Numb = Global_variables.DIM_Numb
Vmax = Global_variables.Vmax
dt = Global_variables.dt
distmax = Global_variables.distmax
get_fct = itemgetter(1, 2, 3)


def BOUNDARY_CHECK(X_list, d):
    return np.where(
        (X_list["Pos"] < (Global_variables.Linf[d] + distmax))
        | (X_list["Pos"] > (Global_variables.L[d] - distmax))
    )[0]


def Group_close_bounds(PARAMS, Xi, Xf):
    """Group particles that are close to boundaries at start, end  because they could interact with each other
    Args:
        PARAMS (list): list of grouped particle information (id,particle or antiparticle,type of particle)
        Xi (_type_): Initial positions of particles (at time t-dt)
        Xf (_type_): Final positions of particles (at time t)

    Returns:
        updated list of grouped particle information
    """
    PARAMS.append([])
    for Xlist in [Xi, Xf]:
        for d, xd in enumerate(Xlist):
            for closebound_ind in BOUNDARY_CHECK(xd, d):
                TYPE0, TYPE1, ID = get_fct(xd[closebound_ind])
                if COUNTFCT(PARAMS[-1], [ID, TYPE0, TYPE1], 1) == 0:
                    PARAMS[-1].append([ID, TYPE0, TYPE1])
    if len(PARAMS[-1]) == 0:
        PARAMS.remove([])
    return PARAMS
