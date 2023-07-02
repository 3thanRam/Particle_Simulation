import numpy as np


def SET_PARAMETERS():
    """
    Specifies paramteres , called by main function .
    Args:
    T (float): The time range to simulate.
    D (int): The type of representation: draw just densities/0 or also draw trajectories/1
    n1 (int): The initial number of particles.
    n2 (int): The initial number of antiparticles.
    vo (float): The order of the velocities of the particles.
    l (float): The length of the box.
    DIM (int): The number of dimensions to simulate
    Bounds(int): The type of boundaries periodic/0 or hard/1
    File_path_name: Where to save the video if the simulation is 3D and D=1
    """
    Timeparam, N_part, N_antipart, V_param, Box_size = 0, 0, 0, 0, 0
    REPR, DIM, Bounds, Vdistr, KeepV = 2, 0, 2, 2, -1

    Answer = ""
    while Answer not in ["r", "c", "q"]:
        Answer = input("Would you like Recommended or custom settings?(r/c) ").lower()

    if Answer == "c":
        print(
            "You have chosen custom settings, depending on the configuration the computation time might be long!"
        )
    elif Answer == "q":  # quicklaunch
        print("Quick Launch")
        DIM = 2
        Vrec = [[5], [5, 10], [10, 12, 15]][DIM - 1]
        REPR = 1
        Bounds = 1
        # GEN_V=RANDCHOICE
        filename = ""
        T, n1, n2, v, L = [
            [20, 50, 50, Vrec, [30]],
            [10, 40, 40, Vrec, [5, 5]],
            [30, 15, 15, Vrec, [20, 20, 20]],
        ][DIM - 1]
        setParams = [T, n1, n2, v, L, DIM, Bounds, REPR, filename]
        return setParams

    while REPR not in [0, 1]:
        try:
            REPR = int(
                input(
                    "Would you like to only see densities or also plot trajectories?(0/1) "
                )
            )
        except:
            print("Please choose between: ", [0, 1])
    while DIM not in [1, 2, 3]:
        try:
            DIM = int(input("Please choose number of dimensions (1,2,3) "))
        except:
            print("Please choose between: ", [1, 2, 3])
    while Bounds not in [0, 1]:
        try:
            Bounds = int(
                input("Please choose either periodic or hard boundaries (0/1) ")
            )
        except:
            print("Please choose between: ", [0, 1])
    while KeepV not in [0, 1]:
        try:
            KeepV = int(
                input(
                    "Would you like to abruptly change speed or keep adding random speed to initial value (0/1) "
                )
            )
        except:
            print("Please choose between: ", [0, 1])
    while Vdistr not in ["u", "n", "w"]:
        try:
            Vdistr = str(
                input(
                    "Please choose either uniform , normal or random walk velocity distribution (U/N/W) "
                )
            ).lower()
        except:
            print("Please choose between: ", [0, 1])

    Distparam = Vdistr
    if Distparam == "n":
        GEN_V = "GAUSS"
        sd = []
        for d in range(DIM):
            Z = 0
            while Z == 0:
                try:
                    Paramtest = float(
                        input("Please Choose Standard deviation " + str(d + 1) + " : ")
                    )
                    if 0 < Paramtest < np.inf:
                        sd.append(Paramtest)
                        Z = 1
                    else:
                        print("Outside of parameter bounds", (0, np.inf), "Try again")
                except:
                    print(str(Paramtest) + " is not a float, try again")
    elif Distparam == "w":
        GEN_V = "RANDCHOICE"
    PARAMS = (
        ["Time parameter ", "number of particles ", "number of antiparticles "]
        + ["velocity parameter " + str(i + 1) + str(" : ") for i in range(DIM)]
        + ["Box lenght " + str(i + 1) + str(" : ") for i in range(DIM)]
    )  # each parameter
    Paramtype = [float, int, int] + 2 * DIM * [float]  # type of each parameter
    PARAMBounds = (
        [(0, np.inf), (0, np.inf), (0, np.inf)]
        + DIM * [(-np.inf, np.inf)]
        + DIM * [(0, np.inf)]
    )  # Boundaries for valid values for each parameter
    if Answer == "r":
        if Distparam != "n":
            Vrec = [[10], [5, 10], [10, 12, 15]][DIM - 1]
        else:
            Vrec = [0 for d in range(DIM)]
        RECOMMENDED = [
            [5, 50, 50, Vrec, [100]],
            [10, 15, 15, Vrec, [30, 30]],
            [20, 15, 15, Vrec, [25, 25, 25]],
        ]
        PARAMS = RECOMMENDED[DIM - 1]
    else:
        for para in range(len(PARAMS)):
            Z = 0
            while Z == 0:
                try:
                    Paramtest = Paramtype[para](
                        input("Please Choose " + str(PARAMS[para]))
                    )
                    if PARAMBounds[para][0] < Paramtest < PARAMBounds[para][1]:
                        PARAMS[para] = Paramtest
                        Z = 1
                    else:
                        print(
                            "Outside of parameter bounds",
                            PARAMBounds[para],
                            "Try again",
                        )
                except:
                    print(
                        str(Paramtest)
                        + " is not a "
                        + str(Paramtype[para])
                        + " try again"
                    )
        PARAMS = PARAMS[:3] + [PARAMS[3 : 3 + DIM]] + [PARAMS[3 + DIM :]]
    if DIM == 3:
        filename = ""
        if filename == "":
            print(
                'Enter location to store video in "filename" variable (near end of Proj_MAIN.py) then restart the program'
            )
            import sys

            sys.exit()
        return (*PARAMS, DIM, Bounds, REPR, filename)
    else:
        return (*PARAMS, DIM, Bounds, REPR)
