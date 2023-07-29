import numpy as np
import Interactions.INTERACTION_LOOP
import System.Position_class
from Particles.Dictionary import PARTICLE_DICT
from Particles.Global_Variables import Global_variables
from System.Group_Close import Group_particles
from Particles.ParticleClass import Particle
from operator import itemgetter
from Misc.Relativistic_functions import Momentum_Calc
from Misc.Functions import NORM

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

DIM_Numb = Global_variables.DIM_Numb
Vmax = Global_variables.Vmax
BOUNDARY_COND = Global_variables.BOUNDARY_COND
dt = Global_variables.dt


distmax = 1.5 * Vmax * dt

get_T_Xinter = itemgetter(2, 5)

POSCENTER = np.array([1 for d in range(DIM_Numb)])


SYSTEM = []


class SYSTEM_CLASS:
    """
    Class containing particles and methods needed to do various changes and updates
    """

    def __init__(self):
        self.Particles_List = []
        self.Numb_Per_TYPE = [[0, 0] for i in range(Numb_of_TYPES)]
        self.MAX_ID_PER_TYPE = [[0, 0] for i in range(Numb_of_TYPES)]
        self.Tot_Numb = 0
        self.Quarks_Numb = 0
        self.TRACKING = [[[], []] for i in range(Numb_of_TYPES)]
        self.Vflipinfo = [[[], []] for i in range(Numb_of_TYPES)]

    def Add_Particle(self, index, PartOrAnti, ExtraParams=None):
        """Add a particle of given type to the system

        Args:
            index (int): represents the type of subatomic particle
            PartOrAnti (int): represents if particle or antiparticle
            ExtraParams (list, optional): extra parameters needed for some types of creations
        """
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
        """
        Find a particle in the system using its  index, PartOrAnti and ID
        """
        for p_numb, particle in enumerate(self.Particles_List):
            if (
                (particle.parity[1] == index)
                and (particle.parity[0] == PartOrAnti)
                and (particle.ID == ID)
            ):
                return p_numb

    def Remove_particle(self, index, PartOrAnti, ID):
        """
        Remove a particle identified using index, PartOrAnti, ID
        """
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)
        self.Particles_List.pop(SEARCH_id)
        self.Numb_Per_TYPE[index][PartOrAnti] -= 1
        self.Tot_Numb -= 1
        if PARTICLE_DICT[PARTICLE_NAMES[index]]["Strong_Charge"] != 0:
            self.Quarks_Numb -= 1

    def RESET_Vflipinfo(self):
        """
        Set all particles vflipinfo to empty list
        """
        self.Vflipinfo = [
            [
                [[] for ni in range(maxnumbpart + 1)],
                [[] for ni in range(maxnumbanti + 1)],
            ]
            for maxnumbpart, maxnumbanti in self.MAX_ID_PER_TYPE
        ]

    def Get_Particle(self, index, PartOrAnti, ID):
        """Get a particle identified using index, PartOrAnti, ID"""
        SEARCH_id = self.FIND_particle(index, PartOrAnti, ID)
        return self.Particles_List[SEARCH_id]

    def UPDATE_DO(self, t):
        """
        perform MOVE(t) method on each particle in the system
        """
        for particle in self.Particles_List:
            particle.MOVE(t)

    def TOTAL_ENERGY(self):
        """
        Return the total energy of the system at the time at which called
        """
        return sum(particle.Energy for particle in self.Particles_List)

    def UPDATE(self, t):
        """
        Update system variables to current values
        """
        self.RESET_Vflipinfo()
        Position_LISTS = System.Position_class.Position_LISTS
        Position_LISTS.Get_X_list(self.Particles_List, self.Quarks_Numb, self.Tot_Numb)
        self.UPDATE_DO(t)
        Position_LISTS.Get_X_list(self.Particles_List)
        self.UPDATE_XF(t)

    def UPDATE_XF(self, t):
        """
        Check for interactions and deal with them between t-dt and t
        """
        Position_LISTS = System.Position_class.Position_LISTS
        PARAMS = Group_particles(Position_LISTS.Xi, Position_LISTS.Xf)
        if len(PARAMS) != 0:
            Position_LISTS.Xf = Interactions.INTERACTION_LOOP.Interaction_Loop_Check(
                Position_LISTS.Xf, t, PARAMS
            )
        else:
            print("NO CHG PARAMS for t=", t)

    def UPDATE_TRACKING(self, index, PartOrAnti, ID, t, NewPos):
        """
        Update trajectory history of particle, identified by index, PartOrAnti, ID at time t with new position NewPos
        """
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
        Vlist = self.Vflipinfo[index][PartOrAnti][ID]
        if Vlist:
            newV = Vlist[-1]
            particle.V = newV
            if particle.M != 0:
                particle.P = Momentum_Calc(newV, particle.M)
            else:
                particle.P = NORM(particle.P) * newV / NORM(newV)
        Global_variables.ALL_TIME.extend(targs[1:])
        self.TRACKING[index][PartOrAnti][ID].append([t, NewPos])


def init():
    """
    initialise SYSTEM as a SYSTEM_CLASS object
    """
    global SYSTEM
    SYSTEM = SYSTEM_CLASS()
