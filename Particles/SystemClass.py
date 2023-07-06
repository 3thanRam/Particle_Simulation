# from dataclasses import dataclass, field
# @dataclass(slots=True)
import numpy as np
from Particles.Dictionary import PARTICLE_DICT

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
dtype = [("Pos", float), ("TypeID0", int), ("TypeID1", int), ("index", int)]
from Particles.Global_Variables import Global_variables
from Particles.ParticleClass import Particle
from Misc.Functions import COUNTFCT

DIM_Numb = Global_variables.DIM_Numb
Vmax = Global_variables.Vmax
BOUNDARY_COND = Global_variables.BOUNDARY_COND
dt = Global_variables.dt

from itertools import combinations

from operator import itemgetter

distmax = 1.5 * Vmax * dt

get2 = itemgetter(3, 4, 5, 6)

POSCENTER = np.array([1 for d in range(DIM_Numb)])


SYSTEM = []


class SYSTEM_CLASS:
    def __init__(self):
        self.Particles_List = []
        self.Numb_Per_TYPE = [[0, 0] for i in range(Numb_of_TYPES)]
        self.MAX_ID_PER_TYPE = [[0, 0] for i in range(Numb_of_TYPES)]
        self.Tot_Numb = 0
        self.Quarks_Numb = 0
        self.TRACKING = [[[], []] for i in range(Numb_of_TYPES)]
        self.Vflipinfo = [[[], []] for i in range(Numb_of_TYPES)]
        if BOUNDARY_COND == 1:
            self.SYSTEM_BOUNDARY_CHECK_FCT = self.SYSTEM_BOUNDARY_CHECK_HARD
        else:
            self.SYSTEM_BOUNDARY_CHECK_FCT = self.SYSTEM_BOUNDARY_CHECK_PERIODIC

    def SYSTEM_BOUNDARY_CHECK_PERIODIC(self):
        return [
            np.where(
                (
                    self.Xi["Pos"]
                    < (Global_variables.Linf[:, np.newaxis] + Vmax[:, np.newaxis] * dt)
                ).any(axis=0)
                | (
                    self.Xi["Pos"]
                    > (Global_variables.L[:, np.newaxis] - Vmax[:, np.newaxis] * dt)
                ).any(axis=0)
            )[0]
        ]

    def SYSTEM_BOUNDARY_CHECK_HARD(self):
        return [
            np.where(
                (
                    self.Xi[:]["Pos"]
                    < (Global_variables.Linf[:, np.newaxis] + Vmax[:, np.newaxis] * dt)
                ).any(axis=0)
            )[0],
            np.where(
                (
                    self.Xi[:]["Pos"]
                    > (Global_variables.L[:, np.newaxis] - Vmax[:, np.newaxis] * dt)
                ).any(axis=0)
            )[0],
        ]

    def Add_Particle(self, index, PartOrAnti, ExtraParams=None):
        if PARTICLE_DICT[PARTICLE_NAMES[index]]["Strong_Charge"] != 0:
            COLOUR = [0, 0, 0]
            if ExtraParams == None:
                P_ExtraParams = ["INIT_Quark_CREATION", POSCENTER]
            else:
                P_ExtraParams = ExtraParams
            self.Quarks_Numb += 1
        elif ExtraParams == None:
            COLOUR = None
            P_ExtraParams = None
        else:
            COLOUR = None
            P_ExtraParams = ExtraParams

        self.Numb_Per_TYPE[index][PartOrAnti] += 1
        self.Tot_Numb += 1
        P_name = PARTICLE_NAMES[index]

        P_parity = (PartOrAnti, PARTICLE_DICT[PARTICLE_NAMES[index]]["index"])
        P_ID = self.MAX_ID_PER_TYPE[index][PartOrAnti]
        self.MAX_ID_PER_TYPE[index][PartOrAnti] += 1
        P_Colour_Charge = COLOUR
        self.Particles_List.append(
            Particle(
                name=P_name,
                parity=P_parity,
                ID=P_ID,
                Colour_Charge=P_Colour_Charge,
                ExtraParams=P_ExtraParams,
            )
        )

    def FIND_particle(self, index, PartOrAnti, ID):
        for p_numb, particle in enumerate(self.Particles_List):
            if (
                (particle.parity[1] == index)
                and (particle.parity[0] == PartOrAnti)
                and (particle.ID == ID)
            ):
                return p_numb

    def UPDATE_particle_position(self, index, PartOrAnti, ID, NewPos):
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)
        self.Particles_List[SEARCH_id].X = NewPos

    def Remove_particle(self, index, PartOrAnti, ID):
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)
        self.Particles_List.pop(SEARCH_id)
        self.Numb_Per_TYPE[index][PartOrAnti] -= 1
        self.Tot_Numb -= 1
        if PARTICLE_DICT[PARTICLE_NAMES[index]]["Strong_Charge"] != 0:
            self.Quarks_Numb -= 1

    def RESET_Vflipinfo(self):
        self.Vflipinfo = [
            [
                [[] for ni in range(maxnumbpart + 1)],
                [[] for ni in range(maxnumbanti + 1)],
            ]
            for maxnumbpart, maxnumbanti in self.MAX_ID_PER_TYPE
        ]

    def Get_Energy_velocity_Remove_particle(self, index, PartOrAnti, ID):
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)
        particle = self.Particles_List[SEARCH_id]
        E, V = particle.Energy, particle.V
        self.Particles_List.pop(SEARCH_id)
        self.Numb_Per_TYPE[index][PartOrAnti] -= 1
        self.Tot_Numb -= 1
        if PARTICLE_DICT[PARTICLE_NAMES[index]]["Strong_Charge"] != 0:
            self.Quarks_Numb -= 1
        return [E, V]

    def Change_Particle_Energy_velocity(self, index, PartOrAnti, ID, E_add, V_add):
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)

        self.Particles_List[SEARCH_id].Energy += E_add
        Vboost = (
            V_add * E_add / (np.linalg.norm(Vmax) * self.Particles_List[SEARCH_id].M)
        )
        self.Particles_List[SEARCH_id].V += Vboost

    def Get_Particle(self, index, PartOrAnti, ID):
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)
        return self.Particles_List[SEARCH_id]

    def Get_Mass_Matrix(self):
        return np.array([particle.M for particle in self.Particles_List])

    def Get_XI(self):
        from ENVIRONMENT.FIELDS import Gen_Field

        Xi = np.array(
            [
                [
                    (particle.X[d], particle.parity[0], particle.parity[1], particle.ID)
                    for particle in self.Particles_List
                ]
                for d in range(DIM_Numb)
            ],
            dtype=dtype,
        )
        Global_variables.FIELD = Gen_Field(
            Xi, self.Particles_List, self.Quarks_Numb, self.Tot_Numb
        )  # update electric field according to positions and charges of particles

        Xi.sort(order="Pos")
        self.Xi = Xi

    def Get_DO(self, t):
        self.DOINFOLIST = np.array(
            [particle.DO(t) for particle in self.Particles_List], dtype="object"
        )
        self.Update_DO_SubLists()

    def Update_DO_SubLists(self):
        self.DO_TYPE_PARTorANTI = np.array([elem[1][0] for elem in self.DOINFOLIST])
        self.DO_TYPE_CHARGE = np.array([elem[1][1] for elem in self.DOINFOLIST])
        self.DO_INDEX = np.array([elem[2] for elem in self.DOINFOLIST])

    def TOTAL_ENERGY(self):
        return sum(particle.Energy for particle in self.Particles_List)

    def Get_XF(self):
        Xf = np.array(
            [
                [
                    (
                        self.DOINFOLIST[s][0][d],
                        self.DOINFOLIST[s][1][0],
                        self.DOINFOLIST[s][1][1],
                        self.DOINFOLIST[s][2],
                    )
                    for s in range(len(self.Particles_List))
                ]
                for d in range(DIM_Numb)
            ],
            dtype=dtype,
        )
        Xf.sort(order="Pos")
        self.Xf = Xf

    def UPDATE(self, t):
        self.RESET_Vflipinfo()
        self.Get_XI()
        self.Get_DO(t)
        self.Get_XF()

    def PREPARE_PARAMS(self):
        DOINFOLIST = self.DOINFOLIST
        DO_TYPE_PARTorANTI = self.DO_TYPE_PARTorANTI
        DO_TYPE_CHARGE = self.DO_TYPE_CHARGE
        DO_INDEX = self.DO_INDEX
        Xi = self.Xi
        Xf = self.Xf

        INI_TYPE_PARTorANTI = [
            np.array([elem[1] for elem in Xi[d]]) for d in range(DIM_Numb)
        ]
        INI_TYPE_CHARGE = [
            np.array([elem[2] for elem in Xi[d]]) for d in range(DIM_Numb)
        ]

        END_TYPE_PARTorANTI = [
            np.array([elem[1] for elem in Xf[d]]) for d in range(DIM_Numb)
        ]
        END_TYPE_CHARGE = [
            np.array([elem[2] for elem in Xf[d]]) for d in range(DIM_Numb)
        ]

        CHANGESdim = [
            np.where(
                (Xi[d]["index"] != Xf[d]["index"])
                | (INI_TYPE_PARTorANTI[d] != END_TYPE_PARTorANTI[d])
                | (INI_TYPE_CHARGE[d] != END_TYPE_CHARGE[d])
            )
            for d in range(DIM_Numb)
        ]
        if DIM_Numb == 1:
            CHANGES = CHANGESdim[0][0]
        elif DIM_Numb == 2:
            CHANGES = np.intersect1d(*CHANGESdim)
        elif DIM_Numb == 3:
            CHANGES = np.intersect1d(
                np.intersect1d(CHANGESdim[0], CHANGESdim[1]), CHANGESdim[2]
            )

        CHGind = []  # index in Xi/Xf

        # particle involved in interactions parameters
        (
            Param_INTERPOS,
            Param_POS,
            Param_Velocity,
            Param_Time,
            Param_ID_TYPE,
            Param_endtype,
        ) = ([], [], [], [], [], [])
        PARAMS = [
            Param_INTERPOS,
            Param_POS,
            Param_Velocity,
            Param_Time,
            Param_ID_TYPE,
            Param_endtype,
        ]
        for (
            chg
        ) in (
            CHANGES
        ):  #  particles of index between ini and end could interact with the particle
            CHGind.append([])
            for parametr in PARAMS:
                parametr.append([])

            for d in range(DIM_Numb):
                matchval = np.where(
                    (Xi[d]["index"][chg] == Xf[d]["index"])
                    & (INI_TYPE_PARTorANTI[d][chg] == END_TYPE_PARTorANTI[d])
                    & (INI_TYPE_CHARGE[d][chg] == END_TYPE_CHARGE[d])
                )[0][0]

                Xa, Xb = Xf[d]["Pos"][matchval], Xi[d]["Pos"][chg]
                if abs(Xa - Xb) <= distmax[d]:
                    mini = min(chg, matchval)
                    maxi = max(chg, matchval) + 1
                else:
                    # fast particles could "skip" the intermediate zone and not be seen
                    # we need to extend the range to account for this
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

                for elem_ind in range(mini, maxi):
                    doindex = np.where(
                        (DO_INDEX == Xf[d]["index"][elem_ind])
                        & (DO_TYPE_PARTorANTI == Xf[d]["TypeID0"][elem_ind])
                        & (DO_TYPE_CHARGE == Xf[d]["TypeID1"][elem_ind])
                    )[0][0]

                    InterPos, Velocity, TimeParams, Endtype = get2(DOINFOLIST[doindex])
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
                    if (
                        CHGind[-1] == []
                        or COUNTFCT(Param_ID_TYPE[-1], [ID, TYPE0, TYPE1], 1) == 0
                    ):
                        CHGind[-1].append(elem_ind)
                        ADD_params = [
                            InterPos,
                            POSLIST,
                            Velocity,
                            TimeParams,
                            np.array([ID, TYPE0, TYPE1]),
                            Endtype,
                        ]
                        for param_numb, parametr in enumerate(PARAMS):
                            parametr[-1].append(ADD_params[param_numb])
            if len(Param_INTERPOS[-1]) == 0:
                for parametr in PARAMS:
                    parametr.remove([])
                CHGind.remove([])

        # if particles interact with bounds then they can interact with particles differently (e.g particles at top and bottom of box can interact if periodic bounds) than those accounted for above
        BOUNDARYCHECKS = self.SYSTEM_BOUNDARY_CHECK_FCT()

        for Bcheck in BOUNDARYCHECKS:
            if len(Bcheck) > 0:
                for parametr in PARAMS:
                    parametr.append([])
                CHGind.append([])
                for PART_B_inf_index in Bcheck:
                    for d in range(DIM_Numb):
                        elem_ind = np.where(
                            (Xf[d]["index"] == Xi[d]["index"][PART_B_inf_index])
                            & (
                                Xi[d]["TypeID0"][PART_B_inf_index]
                                == END_TYPE_PARTorANTI[d]
                            )
                            & (Xi[d]["TypeID1"][PART_B_inf_index] == END_TYPE_CHARGE[d])
                        )[0][0]
                        doindex = np.where(
                            (DO_INDEX == Xf[d]["index"][elem_ind])
                            & (DO_TYPE_PARTorANTI == Xf[d]["TypeID0"][elem_ind])
                            & (DO_TYPE_CHARGE == Xf[d]["TypeID1"][elem_ind])
                        )[0][0]
                        InterPos, Velocity, TimeParams, Endtype = get2(
                            DOINFOLIST[doindex]
                        )
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
                                & (END_TYPE_PARTorANTI[d2] == TYPE0)
                                & (END_TYPE_CHARGE[d2] == TYPE1)
                            )[0][0]
                            POSLIST[d2] = Xf[d2]["Pos"][posind2]
                        ADD_params = [
                            InterPos,
                            POSLIST,
                            Velocity,
                            TimeParams,
                            np.array([ID, TYPE0, TYPE1]),
                            Endtype,
                        ]
                        for param_numb, parametr in enumerate(PARAMS):
                            parametr[-1].append(ADD_params[param_numb])
                        CHGind[-1].append(elem_ind)

        return self.REMOVE_OVERLAP(PARAMS)

    def REMOVE_OVERLAP(self, PARAMS):
        # The same particles can be present in many groups so we must reduce overlap
        REMOVELIST = []
        Param_ID_TYPE = PARAMS[-2]
        for I1, I2 in combinations(range(len(Param_ID_TYPE)), 2):
            if I1 in REMOVELIST or I2 in REMOVELIST:
                continue
            indtypeGroup1, indtypeGroup2 = Param_ID_TYPE[I1], Param_ID_TYPE[I2]
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
                elif Osize >= 0.9 * B_s or Osize >= 0.3 * A_s:
                    if Osize >= 0.9 * B_s:
                        AB_s, indab, I21, I12 = B_s, indb, I1, I2
                    else:
                        AB_s, indab, I21, I12 = A_s, inda, I2, I1
                    RANGE = list(np.arange(AB_s))
                    indNotab = [indN for indN in RANGE if indN not in indab]
                    getAB = itemgetter(*indNotab)
                    for parametr in PARAMS:
                        if len(indNotab) > 1:
                            parametr[I21].extend(getAB(parametr[I12]))
                        else:
                            parametr[I21].append(getAB(parametr[I12]))
                    REMOVELIST.append(I12)

        REMOVELIST.sort(
            reverse=True
        )  # removing top to bottom to avoid index changes after each removal
        for removeind in REMOVELIST:
            for parametr in PARAMS:
                parametr.pop(removeind)
        return PARAMS

    def UPDATE_TRACKING(self, index, PartOrAnti, ID, t, NewPos):
        self.UPDATE_particle_position(index, PartOrAnti, ID, NewPos)

        doindex = np.where(
            (self.DO_INDEX == ID)
            & (self.DO_TYPE_PARTorANTI == PartOrAnti)
            & (self.DO_TYPE_CHARGE == index)
        )[0][0]

        get = itemgetter(3, 5, 6)
        xinterargs, targs, Endtype = get(self.DOINFOLIST[doindex])
        if Endtype > 0:
            for nz in range(len(targs) - 1):
                self.TRACKING[index][PartOrAnti][ID].extend(
                    [
                        [targs[nz + 1], xinterargs[nz][0]],
                        ["T", "X"],
                        [targs[nz + 1], xinterargs[nz][1]],
                    ]
                )
            Global_variables.ALL_TIME.extend(targs[1:])
        self.TRACKING[index][PartOrAnti][ID].append([t, NewPos])


def init():
    global SYSTEM
    SYSTEM = SYSTEM_CLASS()
