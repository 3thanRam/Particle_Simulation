import numpy as np
import Interactions.INTERACTION_LOOP
import ENVIRONMENT.FIELDS
from Particles.Dictionary import PARTICLE_DICT
from Particles.Global_Variables import Global_variables
from System.Group_Close import Group_particles
from Particles.ParticleClass import Particle
from operator import itemgetter

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
dtype = [("Pos", float), ("TypeID0", int), ("TypeID1", int), ("index", int)]
DIM_Numb = Global_variables.DIM_Numb
Vmax = Global_variables.Vmax
BOUNDARY_COND = Global_variables.BOUNDARY_COND
dt = Global_variables.dt


distmax = 1.5 * Vmax * dt

get_T_Xinter = itemgetter(2, 5)

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

        P_ID = self.MAX_ID_PER_TYPE[index][PartOrAnti]
        self.MAX_ID_PER_TYPE[index][PartOrAnti] += 1

        P_name = PARTICLE_NAMES[index]
        P_parity = (PartOrAnti, PARTICLE_DICT[PARTICLE_NAMES[index]]["index"])

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

    def Get_Particle(self, index, PartOrAnti, ID):
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)

        return self.Particles_List[SEARCH_id]

    def Particle_set_coefs(self, index, PartOrAnti, ID, New_Coef_info):
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)
        self.Particles_List[SEARCH_id].Coef_param_list = New_Coef_info
        self.Particles_List[SEARCH_id].V = New_Coef_info[0]

    def Get_XI(self):
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
        Global_variables.FIELD = ENVIRONMENT.FIELDS.Gen_Field(
            Xi, self.Particles_List, self.Quarks_Numb, self.Tot_Numb
        )  # update electric field according to positions and charges of particles

        Xi.sort(order="Pos")
        self.Xi = Xi

    def UPDATE_DO(self, t):
        for particle in self.Particles_List:
            particle.MOVE(t)

    def TOTAL_ENERGY(self):
        return sum(particle.Energy for particle in self.Particles_List)

    def Get_XF(self):
        Xf = np.array(
            [
                [
                    (
                        p.Coef_param_list[-1][d],
                        p.Coef_param_list[3][0],
                        p.Coef_param_list[3][1],
                        p.Coef_param_list[4],
                    )
                    for p in self.Particles_List
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
        self.UPDATE_DO(t)
        self.Get_XF()
        self.UPDATE_XF(t)

    def UPDATE_XF(self, t):
        PARAMS = Group_particles(self.Xi, self.Xf)
        if len(PARAMS) != 0:
            self.Xf = Interactions.INTERACTION_LOOP.Interaction_Loop_Check(
                self.Xf, t, PARAMS
            )
        else:
            print("NO CHG PARAMS for t=", t)

    def UPDATE_TRACKING(self, index, PartOrAnti, ID, t, NewPos):
        particle = self.Get_Particle(index, PartOrAnti, ID)
        targs, xinterargs = get_T_Xinter(particle.Coef_param_list)
        for nz in range(len(targs) - 1):
            self.TRACKING[index][PartOrAnti][ID].extend(
                [
                    [targs[nz + 1], xinterargs[nz][0]],
                    ["T", "X"],
                    [targs[nz + 1], xinterargs[nz][1]],
                ]
            )
        particle.X = NewPos
        Global_variables.ALL_TIME.extend(targs[1:])
        self.TRACKING[index][PartOrAnti][ID].append([t, NewPos])


def init():
    global SYSTEM
    SYSTEM = SYSTEM_CLASS()
