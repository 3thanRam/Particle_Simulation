import numpy as np
from itertools import combinations
from operator import itemgetter

Overlap_parameter = 0.5


def REMOVE_OVERLAP(PARAMS):
    # The same particles can be present in many groups so we must reduce overlap
    REMOVELIST = []
    for I1, I2 in combinations(range(len(PARAMS)), 2):
        if I1 in REMOVELIST or I2 in REMOVELIST:
            continue
        indtypeGroup1, indtypeGroup2 = PARAMS[I1], PARAMS[I2]
        A = np.array(indtypeGroup1)
        B = np.array(indtypeGroup2)
        nrows, ncols = A.shape
        dtypeO = {
            "names": ["f{}".format(i) for i in range(ncols)],
            "formats": ncols * [A.dtype],
        }
        Overlap, inda, indb = np.intersect1d(
            A.view(dtypeO), B.view(dtypeO), return_indices=True
        )
        Osize = Overlap.shape[0]
        if Osize != 0:
            A_s, B_s = A.shape[0], B.shape[0]
            if Osize == B_s:
                REMOVELIST.append(I2)
            elif Osize == A_s:
                REMOVELIST.append(I1)
            elif Osize >= Overlap_parameter * B_s or Osize >= Overlap_parameter * A_s:
                if Osize >= Overlap_parameter * B_s:
                    AB_s, indab, I21, I12 = B_s, indb, I1, I2
                else:
                    AB_s, indab, I21, I12 = A_s, inda, I2, I1
                RANGE = list(np.arange(AB_s))
                indNotab = [indN for indN in RANGE if indN not in indab]
                getAB = itemgetter(*indNotab)
                if len(indNotab) > 1:
                    PARAMS[I21].extend(getAB(PARAMS[I12]))
                else:
                    PARAMS[I21].append(getAB(PARAMS[I12]))
                REMOVELIST.append(I12)
    REMOVELIST.sort(
        reverse=True
    )  # removing top to bottom to avoid index changes after each removal
    for removeind in REMOVELIST:
        PARAMS.pop(removeind)
    GroupList = []
    for paramgroup in PARAMS:
        GroupList.append([])
        for params in paramgroup:
            id, p_0, p_1 = params
            GroupList[-1].append([(p_0, p_1), id])
    return GroupList
