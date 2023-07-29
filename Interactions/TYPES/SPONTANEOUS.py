import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from Interactions.TYPES.STRONG import STRONG_FORCE_GROUP
from Particles.Global_Variables import Global_variables
from Misc.Relativistic_functions import Get_V_from_P
from Misc.Rotation_fcts import rotate_quat, ROT2D, angle_axis_quat
from System.Find_space import Find_space_for_particles
import System.SystemClass as System_module
from Misc.Functions import NORM

rng = np.random.default_rng()
Numb_of_TYPES = len(PARTICLE_DICT)
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
Emax = 10**4


def SpontaneousEvents(t):
    """
    Creates massive particles from photons based on probability weighted by their energies

    Args:
        t (float): current time of simulation

    """
    ##########
    # ELM######
    ##########
    # PARTICLE_SPONT_NAMES=['electron','muon','tau']
    PARTICLE_SPONT_NAMES = [
        "electron",
        "muon",
        "tau",
        "up_Quark",
        "down_Quark",
        "strange_Quark",
        "charm_Quark",
        "bottom_Quark",
        "top_Quark",
    ]
    EcreationMinList = [
        2 * PARTICLE_DICT[partName]["mass"] * C_speed**2
        for partName in PARTICLE_SPONT_NAMES
    ]

    RemoveTypeInd = PARTICLE_DICT["photon"]["index"]

    PHOTON_LIST = [
        part for part in System_module.SYSTEM.Particles_List if part.name == "photon"
    ]

    PHOTON_ID_LIST = [photon.ID for photon in PHOTON_LIST]
    Energy_List = np.array([photon.Energy for photon in PHOTON_LIST])

    Energy_List_CUT = np.where(Energy_List > Emax, Emax, Energy_List)
    GEN_INFO = []
    GEN_INFO_search = []

    for Eminindex, Ecreatmini in enumerate(EcreationMinList):
        P_list = np.where(
            Energy_List_CUT > Ecreatmini,
            np.exp(Energy_List_CUT / (Ecreatmini * Emax)) - 1,
            0,
        )
        P_list = np.where(P_list > 0.9, 0.9, np.where(P_list < 0, 0, P_list))
        if np.trim_zeros(P_list).size > 0:
            Gen = np.array([rng.choice([0, 1], p=[1 - pi, pi]) for pi in P_list])
            GEN = list(np.where(Gen == 1)[0])
            for indGen in GEN:
                if indGen not in GEN_INFO_search:
                    GEN_INFO_search.append(indGen)
                    GEN_INFO.append([Eminindex, indGen])

    GEN_INFO.sort(key=lambda x: x[1], reverse=True)
    for nameind, killind in GEN_INFO:
        NewpartName = PARTICLE_SPONT_NAMES[nameind]
        Particle_size = PARTICLE_DICT[NewpartName]["size"]

        Particle_Kill = System_module.SYSTEM.Get_Particle(
            RemoveTypeInd, 0, PHOTON_ID_LIST[killind]
        )
        PosCenter = Particle_Kill.X
        Global_variables.COLPTS.append([t, PosCenter, 4])
        Energyval = Energy_List[killind] / 2
        NEWMASS = PARTICLE_DICT[NewpartName]["mass"]
        photmomentum = Particle_Kill.P
        photDirection = photmomentum / NORM(photmomentum)

        New_P_norm = (Energyval**2 - NEWMASS**2 * C_speed**4) ** 0.5 / C_speed
        VParam = Get_V_from_P(New_P_norm * photDirection, NEWMASS)

        if DIM_Numb == 3:
            theta = 45 * np.pi / 180
            VParam1 = np.round(
                rotate_quat(angle_axis_quat(theta, photDirection), VParam), decimals=10
            )
            VParam2 = np.round(
                rotate_quat(angle_axis_quat(-theta, photDirection), VParam), decimals=10
            )
        elif DIM_Numb == 2:
            theta = 45 * np.pi / 180
            VParam1 = np.matmul(ROT2D(theta), VParam)
            VParam2 = np.matmul(ROT2D(-theta), VParam)
        else:
            VParam1, VParam2 = VParam, -VParam

        PosParam1, PosParam2 = Find_space_for_particles(
            np.array(PosCenter), Particle_size / 2, VParam1, VParam2, t
        )
        if PosParam1 is not None:
            CREATEparam1 = "Spont_Create", PosParam1, VParam1, t, Energyval
            CREATEparam2 = "Spont_Create", PosParam2, VParam2, t, Energyval
            CREATEPARAMS = [CREATEparam1, CREATEparam2]
            # remove photon
            System_module.SYSTEM.Remove_particle(
                RemoveTypeInd, 0, PHOTON_ID_LIST[killind]
            )

            Typeindex = PARTICLE_DICT[NewpartName]["index"]
            print("sp")
            for i in range(2):
                System_module.SYSTEM.Add_Particle(Typeindex, i, CREATEPARAMS[i])
