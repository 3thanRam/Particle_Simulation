from dataclasses import dataclass, field
import numpy as np

import ENVIRONMENT.FIELDS
from Particles.Global_Variables import Global_variables
import System.SystemClass

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

    def Update_tracking_info(self, t):
        """
        Update particle positions and tracking history
        """
        SYSTEM = System.SystemClass.SYSTEM
        Xf = self.Xf

        for indUpdate in range(len(Xf[0])):
            if not Xf[0][indUpdate]:
                continue
            pos_search = [Xf[0][indUpdate][0]]
            PartOrAnti_search, typeindex_search, id_search = [*Xf[0][indUpdate]][1:4]
            for d in range(1, DIM_Numb):
                Pos_d_index = np.where(
                    (Xf[d]["index"] == id_search)
                    & (Xf[d]["TypeID0"] == PartOrAnti_search)
                    & (Xf[d]["TypeID1"] == typeindex_search)
                )[0][0]
                pos_search.append(Xf[d]["Pos"][Pos_d_index])
            SYSTEM.UPDATE_TRACKING(
                typeindex_search, PartOrAnti_search, id_search, t, pos_search
            )


def init_pos():
    """
    initialise SYSTEM as a SYSTEM_CLASS object
    """
    global Position_LISTS
    Position_LISTS = Position_list_class()
