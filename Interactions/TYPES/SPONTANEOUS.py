import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from Interactions.TYPES.STRONG import STRONG_FORCE_GROUP
from Particles.Global_Variables import Global_variables
from Misc.Relativistic_functions import Get_V_from_P
from Misc.Rotation_fcts import rotate_quat, ROT2D, angle_axis_quat
from System.Find_space import Find_space_for_particles
import System.SystemClass

rng = np.random.default_rng()
Numb_of_TYPES = len(PARTICLE_DICT)
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt

Emax = 10**4


def SpontaneousEvents(t):
    SYSTEM = System.SystemClass.SYSTEM
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

    PHOTON_LIST = [part for part in SYSTEM.Particles_List if part.name == "photon"]

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

        Particle_Kill = SYSTEM.Get_Particle(RemoveTypeInd, 0, PHOTON_ID_LIST[killind])
        PosCenter = Particle_Kill.X
        Global_variables.COLPTS.append([t, PosCenter, 4])
        Energyval = Energy_List[killind] / 2
        NEWMASS = PARTICLE_DICT[NewpartName]["mass"]
        photmomentum = Particle_Kill.P
        photDirection = photmomentum / np.linalg.norm(photmomentum)

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
            SYSTEM.Remove_particle(RemoveTypeInd, 0, PHOTON_ID_LIST[killind])

            Typeindex = PARTICLE_DICT[NewpartName]["index"]
            print("sp")
            for i in range(2):
                SYSTEM.Add_Particle(Typeindex, i, CREATEPARAMS[i])

    ##############
    # STRONG FORCE#
    ##############

    Quark_ind_LIST = [6, 7, 8, 9, 10, 11]
    QUARK_SPONT_NAMES = [
        "up_Quark",
        "down_Quark",
        "strange_Quark",
        "charm_Quark",
        "bottom_Quark",
        "top_Quark",
    ]
    EcreateLIST = [
        2 * PARTICLE_DICT[partName]["mass"] * C_speed**2
        for partName in QUARK_SPONT_NAMES
    ]

    Quark_Numb = SYSTEM.Quarks_Numb
    TotnumbAllpart = SYSTEM.Tot_Numb
    if Quark_Numb == 0:
        Global_variables.S_Force = np.zeros((TotnumbAllpart, TotnumbAllpart))
        return 0

    STRONG_FORCE, BIG_vel_matrix = STRONG_FORCE_GROUP(TotnumbAllpart)

    SPRING_ENERGY = np.abs(STRONG_FORCE * BIG_vel_matrix * dt)

    SPRING_ENERGY_CUT = np.where(SPRING_ENERGY > Emax, Emax, SPRING_ENERGY)

    GEN_INFO = []
    GEN_INFO_search = []

    for Eminindex, Ecreatei in enumerate(EcreateLIST):
        P_list = np.where(
            SPRING_ENERGY_CUT > Ecreatei,
            np.exp(SPRING_ENERGY_CUT / (Ecreatei * Emax)) - 1,
            0,
        )
        P_list = np.where(P_list > 1, 0.99, np.where(P_list < 0, 0, P_list))

        if np.trim_zeros(P_list.flatten()).size > 0:
            Nonzero_inds = np.where(P_list != 0)
            for NonzeroNumb in range(len(Nonzero_inds[0])):
                quark1_ind, quark2_ind = (
                    Nonzero_inds[0][NonzeroNumb],
                    Nonzero_inds[1][NonzeroNumb],
                )
                if (quark1_ind, quark2_ind) in GEN_INFO_search:
                    continue
                pi = P_list[quark1_ind, quark2_ind]
                Rand = rng.choice([0, 1], p=[1 - pi, pi])

                if Rand == 1:
                    GEN_INFO_search.append((quark1_ind, quark2_ind))
                    GEN_INFO.append([Eminindex, quark1_ind, quark2_ind])

    # GEN_INFO.sort(key=lambda x:x[1],reverse=True)
    for GENnumb, (nameind, ParentSystindex1, ParentSystindex2) in enumerate(GEN_INFO):
        NewpartName = QUARK_SPONT_NAMES[nameind]

        parity1, id1, X1 = (
            SYSTEM.Particles_List[ParentSystindex1].parity,
            SYSTEM.Particles_List[ParentSystindex1].ID,
            np.array(SYSTEM.Particles_List[ParentSystindex1].X),
        )
        PartOrAnti1, ind1 = parity1

        parity2, id2, X2 = (
            SYSTEM.Particles_List[ParentSystindex2].parity,
            SYSTEM.Particles_List[ParentSystindex2].ID,
            np.array(SYSTEM.Particles_List[ParentSystindex2].X),
        )
        PartOrAnti2, ind2 = parity2

        PosCenter = 0.5 * (X1 + X2)

        Global_variables.COLPTS.append([t, PosCenter, 4])

        Posdeviation = min(
            np.min(np.abs(Global_variables.L - PosCenter)),
            np.min(np.abs(Global_variables.Linf - PosCenter)),
        ) / (10 * max(abs(Global_variables.L - Global_variables.Linf)))
        PosParam1 = PosCenter - Posdeviation
        PosParam2 = PosCenter + Posdeviation

        Energyval = SPRING_ENERGY[ParentSystindex1][ParentSystindex2] / 2

        dX = X1 - X2
        ParentDirection = dX / np.linalg.norm(dX)

        NEWMASS = PARTICLE_DICT[NewpartName]["mass"]

        VParam = (ParentDirection / (C_speed * NEWMASS)) * (
            (Energyval**2 - NEWMASS**2 * C_speed**4) / (C_speed**2 * NEWMASS)
        ) ** 0.5
        VParam1, VParam2 = VParam, -VParam
        CREATEparam1 = "Spont_Create", PosParam1, VParam1, t, Energyval
        CREATEparam2 = "Spont_Create", PosParam2, VParam2, t, Energyval
        CREATEPARAMS = [CREATEparam1, CREATEparam2]

        print("Strong Energy added", Energyval * 2)

        Typeindex = PARTICLE_DICT[NewpartName]["index"]
        for i in range(2):
            SYSTEM.Add_Particle(Typeindex, i, CREATEPARAMS[i])
