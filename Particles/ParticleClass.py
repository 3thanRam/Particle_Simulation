import numpy as np
from dataclasses import dataclass, field


from Particles.Dictionary import PARTICLE_DICT

PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
# from MAIN import GEN_V

from Particles.Global_Variables import Global_variables


ROUNDDIGIT = Global_variables.ROUNDDIGIT

C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb


dt = Global_variables.dt
Vmax = Global_variables.Vmax
L_FCT = Global_variables.L_FCT
Dist_min = Global_variables.Dist_min
BOUNDARY_COND = Global_variables.BOUNDARY_COND
V0 = Global_variables.V0

rng = np.random.default_rng()
Xini = []


# Variables that will need to be reset at each use
# Global_variables.TRACKING=Global_variables.Global_variables.TRACKING
# Global_variables.Vflipinfo=Global_variables.Global_variables.Vflipinfo

# Global_variables.L=Global_variables.Global_variables.L
# Global_variables.Linf=Global_variables.Global_variables.Linf


def GAUSS():
    """Generates a random velocity vector based on gaussian distribution of mean V0 and Standard deviation sd.

    Global Args Used:
    - V0 (float): Mean (“centre”) of the distribution.
    - sd (float): Standard deviation of the distribution (Non-negative).
    - DIM_Numb (int): number of dimensions.
    - ROUNDDIGIT (int): number of decimal places to round velocity components to.

    Returns:
    - np.ndarray: a random velocity vector with values rounded to ROUNDDIGIT decimal places.
    """
    return np.round(rng.normal(0, V0 / 3, DIM_Numb), ROUNDDIGIT)


def RANDCHOICE():
    """Pick between -V0 and V0 at random.

    Global Args Used:
    - V0 (float): absolute velocity value for each component.
    Returns:
    - np.ndarray: a random velocity vector
    """
    return rng.choice([-V0, V0], p=[0.5, 0.5])


def UNIFORM():
    """Generates a random velocity vector with components between -V0 and V0.

    Global Args Used:
    - V0 (float): maximum absolute velocity value for each component.
    - DIM_Numb (int): number of dimensions.
    - ROUNDDIGIT (int): number of decimal places to round velocity components to.

    Returns:
    - np.ndarray: a random velocity vector with values rounded to ROUNDDIGIT decimal places.
    """
    return np.round(rng.uniform(-V0, V0, DIM_Numb), ROUNDDIGIT)


GEN_V = UNIFORM


def ROUND(x):
    """
    Rounds the given number 'x' to the number of digits specified by 'ROUNDDIGIT'.
    :param x: the number to be rounded
    :return: the rounded number
    """
    return round(x, ROUNDDIGIT)


def BOUNDS(xi, xf, V, t, id, p, Mass):
    """
    Computes the boundaries for a particle based on its position and velocity.

    Args:
    - xi (array): The initial position of the particle.
    - xf (array): The final position of the particle.
    - V (array): The velocity of the particle.
    - t (float): The current time.
    - id (int): The ID of the particle.
    - p (int): The ID of the process that is executing the function.

    Returns:
    - A list with the following elements:
        - Xini (array): The initial position of the particle.
        - Xinter (array): The intermediate positions of the particle (if any).
        - Xfin (array): The final position of the particle.
        - V (array): The velocity of the particle.
        - t_list (list): A list with the time values at which the particle reaches an intermediate position.
        - id (int): The ID of the particle.
        - NZ (int): The number of intermediate positions.
    """
    # PART_SIZE = PARTICLE_DICT[PARTICLE_NAMES[p[1]]]["size"]

    Xini = xi
    Xfin = xf
    Xinter = []
    t_list = [t]

    # Check if the final position is inside the boundaries
    if in_all_bounds(Xfin, t):
        return [Xfin, p, id, Xinter, V, t_list, 0]

    # Initialize parameters
    t_params = np.zeros(DIM_Numb)
    x_params = np.zeros((DIM_Numb, 2))

    # Check if the particle hits any boundary
    Vchgsign = V.copy()
    for d in range(DIM_Numb):
        b = Xfin[d] - V[d] * t

        CONDsup, CONDinf = False, False
        if Global_variables.BoundSup_V[d] != V[d]:
            SOL_sup = ROUND(
                (b - Global_variables.BoundSup_b[d])
                / (Global_variables.BoundSup_V[d] - V[d])
            )
            CONDsup = t - dt <= SOL_sup <= t
        if Global_variables.BoundInf_V[d] != V[d]:
            SOL_inf = ROUND(
                (b - Global_variables.BoundInf_b[d])
                / (Global_variables.BoundInf_V[d] - V[d])
            )
            CONDinf = t - dt <= SOL_inf <= t

        if CONDinf:
            t0 = ROUND(SOL_inf)
        elif CONDsup:
            t0 = ROUND(SOL_sup)
        else:
            continue

        L_t0, Linf_t0 = L_FCT[0](t0)[d], L_FCT[1](t0)[d]
        Xinterd = b + V * t0

        if CONDinf:
            if BOUNDARY_COND == 0:
                Li_0, Li_1 = Linf_t0, L_t0
                bi = L_t0 - V[d] * t0
                # bi=b+(L_t0[d]-Linf_t0[d])
            else:
                if (
                    -V[d] + (Global_variables.BoundInf_V[d])
                    < Global_variables.BoundInf_V[d]
                ):
                    Vchgsign[d] = np.sign(Global_variables.BoundInf_V[d]) * abs(
                        V[d]
                    ) + (Global_variables.BoundInf_V[d])
                else:
                    Vchgsign[d] = -V[d] + (Global_variables.BoundInf_V[d])

                bi = Xinterd[d] - Vchgsign[d] * t0
                Li_0, Li_1 = Linf_t0, Linf_t0
        else:
            if BOUNDARY_COND == 0:
                Li_0, Li_1 = L_t0, Linf_t0
                bi = Linf_t0 - V[d] * t0
                # bi=b-(L_t0[d]-Linf_t0[d])
            else:
                if (
                    -V[d] + (Global_variables.BoundSup_V[d])
                    > Global_variables.BoundSup_V[d]
                ):
                    Vchgsign[d] = np.sign(Global_variables.BoundSup_V[d]) * abs(
                        V[d]
                    ) + (Global_variables.BoundSup_V[d])
                else:
                    Vchgsign[d] = -V[d] + (Global_variables.BoundSup_V[d])

                bi = Xinterd[d] - Vchgsign[d] * t0
                Li_0, Li_1 = L_t0, L_t0

        t_params[d] = t0
        x_params[d][0], x_params[d][1] = Li_0, Li_1
        Xfin[d] = Vchgsign[d] * t + bi

    # Check if the particle hits any boundary and get the number of hits
    NZ = np.count_nonzero(t_params)

    # Compute intermediate positions if the particle hits any boundary
    if NZ > 0:
        Xinter = np.zeros((NZ, 2, DIM_Numb))
        d_prev = 0
        # Vinter=V.copy()
        Vinter = np.zeros_like(V)
        for nz in range(NZ):
            mask = np.where((t_params > 0) & (t_params != np.inf))
            d_arg = mask[0][t_params[mask].argmin()]
            t_val = t_params[d_arg]
            t_list.append(t_val)
            L_tv_d = L_FCT[0](t_val)[d_arg]
            Linf_tv_d = L_FCT[1](t_val)[d_arg]

            if nz == 0:
                Xinter[nz][0] = np.where(
                    (Xini + V * dt > Linf_tv_d) & (Xini + V * dt < L_tv_d),
                    Xini + V * dt,
                    Xini,
                )
                Xinter[nz][1] = np.where(
                    (Xini + V * dt > Linf_tv_d) & (Xini + V * dt < L_tv_d),
                    Xini + V * dt,
                    Xini,
                )
                Vinter[d_arg] = Vchgsign[d_arg]
            else:
                val0 = Xinter[nz - 1][0] + Vinter * (t_val - (t - dt))  # -t_list[-2])
                Xinter[nz][0] = np.where(
                    (val0 > Linf_tv_d) & (val0 < L_tv_d), val0, Xinter[nz - 1][0]
                )
                val1 = Xinter[nz - 1][1] + Vinter * (t_val - (t - dt))
                Xinter[nz][1] = np.where(
                    (val1 > Linf_tv_d) & (val1 < L_tv_d), val1, Xinter[nz - 1][1]
                )
                Xinter[nz][0][d_prev] = Xinter[nz][1][d_prev]
                Vinter[d_arg] = Vchgsign[d_arg]

            Xinter[nz][0][d_arg] = x_params[d_arg][0]
            Xinter[nz][1][d_arg] = x_params[d_arg][1]
            t_params[d_arg] = np.inf
            d_prev = d_arg
            Global_variables.Vflipinfo[p[1]][p[0]][id].append([d_arg, Vchgsign[d_arg]])
            # Global_variables.Vflipinfo[TYPE_to_index_Dict[p[1]]][p[0]][id].append([d_arg,Vchgsign[d_arg]])
    return [Xfin, p, id, Xinter, V, t_list, NZ]


def in_all_bounds(List, t=None):
    if t == None:
        Lmaxi, Lmini = Global_variables.L, Global_variables.Linf
    else:
        Lmaxi, Lmini = L_FCT[0](t), L_FCT[1](t)
    for indV, value in enumerate(List):
        if value > Lmaxi[indV] or value < Lmini[indV]:
            return False
    return True


def Pos_fct(min_value, max_value, EXtraparams=None):
    """Generate a random position vector with DIM_Numb dimensions and values between min_value and max_value (both inclusive)

    Args:
        min_value (float or int): The minimum value for each dimension of the position vector
        max_value (float or int): The maximum value for each dimension of the position vector

    Returns:
        A list of length DIM_Numb representing the position vector
    """
    # Create a list of DIM_Numb random values between min_value and max_value (both excluded)
    if EXtraparams == None:
        pos = rng.uniform(min_value + 1e-12, max_value)
    else:
        Center, Gaussparam = EXtraparams
        pos = [np.inf, np.inf, np.inf]
        while not in_all_bounds(pos):
            pos = np.round(rng.normal(Center, Gaussparam, DIM_Numb), ROUNDDIGIT)
    return pos


def CHECKstartPos(testPOS):
    """
    Checks if the initial position of a particle is far enough from the others.

    Args:
    testPOS (List[float]): The initial position of the particle.

    Returns:
    True if the position is not valid, False otherwise.
    """
    if in_all_bounds(testPOS, 0):
        return True
    if len(Xini) > 0:
        XiArray = np.array(Xini)
        argmini = (np.sum((testPOS - XiArray) ** 2)).argmin()
        dmin = np.sqrt(np.sum((testPOS - Xini[argmini]) ** 2))
        return dmin > Dist_min
    else:
        return True


def GEN_X(XtraParam=None):
    """Generates a random position within a given distance of minimum separation from other particles.

    Args:
    - Epsilon (float): minimum distance between particles.
    - Global_variables.L (int): maximum position value for each coordinate.
    - Dist_min (float): minimum distance allowed between generated position and existing positions.
    - Xini (List[List[int]]): list of existing particle positions.

    Returns:
    - List[int]: a new particle position that meets the minimum distance requirement from other particles.
    """
    # Generate a new position randomly within the bounds of Global_variables.L.
    if XtraParam == None:
        POS = Pos_fct(L_FCT[1](0), L_FCT[0](0))
        # While the generated position is not far enough from existing positions,
        # generate a new position.
        while not CHECKstartPos(np.array(POS)):
            POS = Pos_fct(L_FCT[1](0), L_FCT[0](0))
        # Add the new position to the list of existing positions.
        Xini.append(POS)
    else:
        POS = Pos_fct(L_FCT[1](0), L_FCT[0](0), XtraParam)
    return POS


def FIELDS(charge, id):
    """retrieve effect of fields on a given particle

    Args:
        charge (float): Charge of the particle
        id (int): id of the particle
        mass (float): mass of the particle
        position (list/array): position of the particle
    Returns:
    Speed to be added to particle due to the fields
    """
    Field_DICT = Global_variables.Field_DICT
    FIELD = Global_variables.FIELD
    field_Acceleration = FIELD[Field_DICT[(*charge, id)]]
    field_V = field_Acceleration * dt
    return field_V


@dataclass(slots=True)
class Particle:
    """A class representing a particle in a simulation.

    Attributes:
    -----------
    parity : int
        The parity of the particle, which determines its behavior when interacting with the boundary.
    ID : int
        A unique identifier for the particle.
    X : list of floats
        The position of the particle in the simulation space.
    V_av : list of floats
        The average velocity of the particle.

    Methods:
    --------
    DO(t):
        Performs a single step of the simulation for the particle.

    """

    name: str
    parity: int
    ID: int
    ExtraParams: list = field(default_factory=list)
    M: float = field(init=False)
    Strong_Charge: float = field(init=False)
    Colour_Charge: list = field(default_factory=list)

    X: np.ndarray = field(init=False)
    V: np.ndarray = field(init=False)

    Energy: float = field(init=False)
    P: float = field(init=False)

    def __post_init__(self):
        self.M = PARTICLE_DICT[self.name]["mass"]
        self.Strong_Charge = PARTICLE_DICT[self.name]["Strong_Charge"]

        if self.Strong_Charge != 0:
            if not self.Colour_Charge or self.Colour_Charge == 0:
                print("Error:Colour_Charge of quark ill defined at creation ")
                print(5 / 0)

        partORanti, typeIndex = self.parity
        id = self.ID

        if not self.ExtraParams:
            # creation of particles at t=0
            X = GEN_X()
            self.X = X
            Global_variables.TRACKING[typeIndex][partORanti][id].append([0.0, X])
            if self.M != 0:
                V = GEN_V()
                while np.linalg.norm(V) > C_speed:
                    V = GEN_V()
                self.V = V
                gamma = 1 / (np.sqrt(1 - (np.linalg.norm(V) / C_speed) ** 2))
                self.P = self.V * self.M * gamma
                self.Energy = (
                    (self.M * C_speed**2) ** 2
                    + np.linalg.norm((self.P * C_speed) ** 2)
                ) ** 0.5
            else:
                v = GEN_V()
                self.V = C_speed * v / np.linalg.norm(v)
                self.P = 150 * v
                self.Energy = np.linalg.norm(self.P * C_speed)
        elif self.ExtraParams[0] == "INIT_Quark_CREATION":
            PosParam = self.ExtraParams[1]
            self.X = PosParam * (1 + 0.01 * rng.uniform(-1, 1))
            V = GEN_V()
            while np.linalg.norm(V) > C_speed:
                V = GEN_V()
            self.V = V
            gamma = 1 / (np.sqrt(1 - (np.linalg.norm(V) / C_speed) ** 2))
            self.P = self.V * self.M * gamma
            self.Energy = (
                (self.M * C_speed**2) ** 2 + np.linalg.norm((self.P * C_speed) ** 2)
            ) ** 0.5
        elif self.ExtraParams[0] == "Post_Interaction":
            PosParam, VParam, TvalueParam, Energyval = self.ExtraParams[
                1:
            ]  # PosParam,VParam,TStepsNumbParam=ExtraParams[1:]
            X = PosParam
            self.X = X
            self.V = VParam
            self.Energy = Energyval
            self.P = Energyval / C_speed
            Global_variables.TRACKING[typeIndex][partORanti].append([[TvalueParam, X]])
            Global_variables.Vflipinfo[typeIndex][partORanti].append([])
        elif self.ExtraParams[0] == "Spont_Create":
            PosParam, VParam, TvalueParam, Energyval = self.ExtraParams[1:]
            X = PosParam  # GEN_X(PosParam)
            self.X = X
            self.V = VParam
            self.Energy = Energyval
            self.P = Energyval / C_speed
            Global_variables.TRACKING[typeIndex][partORanti].append([[TvalueParam, X]])
            Global_variables.Vflipinfo[typeIndex][partORanti].append([])

    def DO(self, t):
        xi = self.X.copy()
        Vt = self.V.copy()
        if self.name != "photon":
            Vt += FIELDS(self.parity, self.ID)
        Vt = np.where(abs(Vt) > Vmax, Vmax * np.sign(Vt), Vt)
        xf = np.round(xi + dt * Vt, ROUNDDIGIT)
        self.V = np.where(
            (xf > Global_variables.L),
            -np.abs(Vt),
            np.where((xf < Global_variables.Linf), np.abs(Vt), Vt),
        )

        return BOUNDS(xi, xf, Vt, t, self.ID, self.parity, self.M)
