# from dataclasses import dataclass, field
# @dataclass(slots=True)
import numpy as np
from Particles.Dictionary import PARTICLE_DICT

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
dtype = [("Pos", float), ("TypeID0", int), ("TypeID1", int), ("index", int)]
from Particles.Global_Variables import Global_variables
from Particles.ParticleClass import Particle

DIM_Numb = Global_variables.DIM_Numb
Vmax = Global_variables.Vmax
POSCENTER = np.array([1 for d in range(DIM_Numb)])
from ENVIRONMENT.FIELDS import Gen_Field

SYSTEM = []


class SYSTEM_CLASS:
    def __init__(self):
        self.Particles_List = []
        # self.Particles_List_Type_Sorted = [[[], []] for i in range(Numb_of_TYPES)]
        self.Numb_Per_TYPE = [[0, 0] for i in range(Numb_of_TYPES)]
        self.MAX_ID_PER_TYPE = [[0, 0] for i in range(Numb_of_TYPES)]
        self.Tot_Numb = 0
        self.Quarks_Numb = 0
        self.TRACKING = [[[], []] for i in range(Numb_of_TYPES)]
        self.Vflipinfo = [[[], []] for i in range(Numb_of_TYPES)]

    def Add_Particle(self, index, PartOrAnti, ExtraParams=None):
        if PARTICLE_DICT[PARTICLE_NAMES[index]]["Strong_Charge"] != 0:
            COLOUR = [0, 0, 0]
            P_ExtraParams = ["INIT_Quark_CREATION", POSCENTER]
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

    def Remove_particle(self, index, PartOrAnti, ID):
        for p_numb, particle in enumerate(self.Particles_List):
            if (
                (particle.parity[1] == index)
                and (particle.parity[0] == PartOrAnti)
                and (particle.ID == ID)
            ):
                self.Particles_List.pop(p_numb)
                self.Numb_Per_TYPE[index][PartOrAnti] -= 1
                self.Tot_Numb -= 1
                if PARTICLE_DICT[PARTICLE_NAMES[index]]["Strong_Charge"] != 0:
                    self.Quarks_Numb -= 1
                break

    def Get_Energy_velocity_Remove_particle(self, index, PartOrAnti, ID):
        for p_numb, particle in enumerate(self.Particles_List):
            if (
                (particle.parity[1] == index)
                and (particle.parity[0] == PartOrAnti)
                and (particle.ID == ID)
            ):
                E, V = particle.Energy, particle.V
                self.Particles_List.pop(p_numb)
                self.Numb_Per_TYPE[index][PartOrAnti] -= 1
                self.Tot_Numb -= 1
                if PARTICLE_DICT[PARTICLE_NAMES[index]]["Strong_Charge"] != 0:
                    self.Quarks_Numb -= 1
                return [E, V]

    def Change_Particle_Energy_velocity(self, index, PartOrAnti, ID, E_add, V_add):
        for p_numb, particle in enumerate(self.Particles_List):
            if (
                (particle.parity[1] == index)
                and (particle.parity[0] == PartOrAnti)
                and (particle.ID == ID)
            ):
                self.Particles_List[p_numb].Energy += E_add
                Vboost = V_add * E_add / (np.linalg.norm(Vmax) * particle.M)
                self.Particles_List[p_numb].V += Vboost
                # E, V = particle.Energy, particle.V
                # self.Particles_List.pop(p_numb)
                # return [E, V]

    # Global_variables.SYSTEM[p2index][partORAnti2][s].Energy += Etot
    # Vboost = (vect_direct* Etot/ (np.linalg.norm(Vmax)* Global_variables.SYSTEM[p2index][partORAnti2][s].M))
    # Global_variables.SYSTEM[p2index][partORAnti2][s].V += Vboost
    # def __getitem__(self, index, PartOrAnti):
    #    return self.Particles_List[index][PartOrAnti]
    def Get_Particle(self, index, PartOrAnti, ID):
        for p_numb, particle in enumerate(self.Particles_List):
            if (
                (particle.parity[1] == index)
                and (particle.parity[0] == PartOrAnti)
                and (particle.ID == ID)
            ):
                return particle

    def Get_Mass_Matrix(self):
        return np.array([particle.M for particle in self.Particles_List])

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
        Global_variables.FIELD = Gen_Field(
            Xi, self.Particles_List, self.Quarks_Numb, self.Tot_Numb
        )  # update electric field according to positions and charges of particles

        Xi.sort(order="Pos")
        self.Xi = Xi

    def Get_DO(self, t):
        self.DOINFOLIST = np.array(
            [particle.DO(t) for particle in self.Particles_List], dtype="object"
        )
        self.DO_TYPE_PARTorANTI = np.array([elem[1][0] for elem in self.DOINFOLIST])
        self.DO_TYPE_CHARGE = np.array([elem[1][1] for elem in self.DOINFOLIST])
        self.DO_INDEX = np.array([elem[2] for elem in self.DOINFOLIST])

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


def init():
    global SYSTEM
    SYSTEM = SYSTEM_CLASS()


""" 
Basically fancy list of ParticleClass objects

features:
-add/remove particles
-search for particle using different methods of identification
-generate ordered list of id based on position before and after dt
-Update TRACKING info  

"""
