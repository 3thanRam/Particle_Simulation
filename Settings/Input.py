import numpy as np
from Particles.Dictionary import PARTICLE_DICT

Numb_of_TYPES = len(PARTICLE_DICT)
PARTICLE_NAMES = [*PARTICLE_DICT.keys()]


recommend_Dim = 3
Parameter_dict = {
    "Numb_Dimensions": {"type": int, "Rec": recommend_Dim, "bounds": [1, 3]},
    "time_steps": {"type": int, "Rec": [20, 10, 20], "bounds": [0, np.inf]},
    "box_size": {
        "type": float,
        "Rec": [[30], [15, 15], [20, 20, 20]],
        "bounds": [0, np.inf],
    },
    "BOUNDARY_COND": {"type": int, "Rec": 1, "bounds": [0, 1]},
    "Repr_mode": {"type": int, "Rec": 1, "bounds": [0, 2]},
}
Params = [*Parameter_dict.keys()]


Rec = [
    "r",
    "recommended",
    "rec",
]


def SET_PARAMETERS():
    """
    set initial parameters needed to start the simulation
    """
    MODE = input("Recommended or Custom settings?(r/c) ")

    Input_parameters = []
    if MODE in Rec:
        INI_Nset = [[0, 0] for Ntype in range(Numb_of_TYPES)]
        INI_Nset[2] = [10, 10]  # electron/positron
        # Nset[13] = [5, 0]  # photons
        INI_Nset[6] = [2, 0]  # upquark
        INI_Nset[7] = [1, 0]  # downquark
        ## elec 0
        ## muon 2
        ## quarks: 6-11
        for p_numb, param in enumerate(Params):
            if p_numb in [1, 2]:
                Rec_param = Parameter_dict[param]["Rec"][Input_parameters[0] - 1]
            else:
                Rec_param = Parameter_dict[param]["Rec"]
            Input_parameters.append(Rec_param)
        Input_parameters.append(INI_Nset)
        return Input_parameters

    for p_numb, param in enumerate(Params):
        if p_numb in [1, 2]:
            Rec_param = Parameter_dict[param]["Rec"][Input_parameters[0] - 1]
        else:
            Rec_param = Parameter_dict[param]["Rec"]

        User_inp = input(f"Please choose {param}: (rec {Rec_param})  ")  #
        while True:
            if isinstance(User_inp, str) and User_inp.lower() in Rec:
                User_inp = Rec_param
                Input_parameters.append(User_inp)
                break
            try:
                User_inp = Parameter_dict[param]["type"](User_inp)
                if Parameter_dict[param]["bounds"]:
                    if (
                        Parameter_dict[param]["bounds"][0]
                        <= User_inp
                        <= Parameter_dict[param]["bounds"][1]
                    ):
                        if p_numb != 2:
                            Input_parameters.append(User_inp)
                            break
                        elif len(Input_parameters) < 3:
                            Input_parameters.append([User_inp])
                            if len(Input_parameters[-1]) == Input_parameters[0]:
                                break
                        else:
                            Input_parameters[-1].append(User_inp)
                            if len(Input_parameters[-1]) == Input_parameters[0]:
                                break
                    else:
                        print(
                            f"Must be between{Parameter_dict[param]['bounds'][0]} and {Parameter_dict[param]['bounds'][1]}"
                        )
                else:
                    Input_parameters.append(User_inp)
                    break
                User_inp = input(f"Please choose {param}: (rec {Rec_param})  ")
            except ValueError:
                print(
                    f"Must be a {Parameter_dict[param]['type']} and not {type(User_inp)}"
                )
                User_inp = input(f"Please choose {param}: (rec {Rec_param})  ")

    INI_Nset = [[0, 0] for n in range(Numb_of_TYPES)]
    for n in range(Numb_of_TYPES):
        name = PARTICLE_NAMES[n]

        if n < 12:
            Part_Anti = 2
        else:
            Part_Anti = 1
        for i in range(Part_Anti):
            name = "anti" * i + PARTICLE_NAMES[n]
            User_inp = input(f"Please choose Number of {name}  ")
            while True:
                try:
                    User_inp = int(User_inp)
                    if User_inp >= 0:
                        break
                except:
                    print("Must be int and postive (or zero)")
                    User_inp = input(f"Please choose Number of {name}  ")
            INI_Nset[n][i] = User_inp
    Input_parameters.append(INI_Nset)
    return Input_parameters
