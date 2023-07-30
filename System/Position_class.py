from dataclasses import dataclass, field
import numpy as np

import ENVIRONMENT.FIELDS
from Particles.Global_Variables import Global_variables
from System.Group_Close import Group_particles
import Interactions.INTERACTION_LOOP

DIM_Numb = Global_variables.DIM_Numb

dtype = [("Pos", float), ("TypeID0", int), ("TypeID1", int), ("index", int)]
Position_LISTS = []


@dataclass(slots=True)
class Position_list_class:
    Xi: list = field(default_factory=list)
    Xf: list = field(default_factory=list)

    def Get_X_list(self, Particles_List, Quarks_Numb=None, Tot_Numb=None):
        """
        Generate array of identifying information and position before or after dt of each particle
        if Quarks_Numb and Tot_Numb are given then it's before dt
        """
        X_array = np.array(
            [
                [
                    particle.Position(d, Tot_Numb is not None)
                    for particle in Particles_List
                ]
                for d in range(DIM_Numb)
            ],
            dtype=dtype,
        )

        if Tot_Numb:
            Global_variables.FIELD = ENVIRONMENT.FIELDS.Gen_Field(
                X_array, Particles_List, Quarks_Numb, Tot_Numb
            )  # update electric field according to positions and charges of particles before dt

        X_array.sort(order="Pos")

        if Tot_Numb:
            self.Xi = X_array
        else:
            self.Xf = X_array
        return X_array

    def UPDATE_XF(self, t):
        """
        Check for interactions and deal with them between t-dt and t
        """
        PARAMS = Group_particles(self.Xi, self.Xf)
        if len(PARAMS) != 0:
            self.Xf = Interactions.INTERACTION_LOOP.Interaction_Loop_Check(
                self.Xf, t, PARAMS
            )
        else:
            print("NO CHG PARAMS for t=", t)


def init_pos():
    """
    initialise SYSTEM as a SYSTEM_CLASS object
    """
    global Position_LISTS
    Position_LISTS = Position_list_class()
