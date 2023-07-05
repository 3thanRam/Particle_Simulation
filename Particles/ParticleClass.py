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

from Misc.Velocity_Fcts import UNIFORM  # RANDCHOICE,GAUSS
from Misc.Position_Fcts import GEN_X
from ENVIRONMENT.BOUNDARY_CHECK import BOUNDS_Collision_Check

GEN_V = UNIFORM


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
    Size: float = field(init=False)
    Colour_Charge: list = field(default_factory=list)

    X: np.ndarray = field(init=False)
    V: np.ndarray = field(init=False)

    Energy: float = field(init=False)
    P: float = field(init=False)

    def __post_init__(self):
        self.M = PARTICLE_DICT[self.name]["mass"]
        self.Strong_Charge = PARTICLE_DICT[self.name]["Strong_Charge"]
        self.Size = PARTICLE_DICT[self.name]["size"] / 2
        if self.Strong_Charge != 0:
            if not self.Colour_Charge or self.Colour_Charge == 0:
                print("Error:Colour_Charge of quark ill defined at creation ")
                print(5 / 0)

        partORanti, typeIndex = self.parity
        id = self.ID

        if not self.ExtraParams:
            # creation of particles at t=0
            X = GEN_X(self.Size)
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
            Vt += (
                Global_variables.FIELD[
                    Global_variables.Field_DICT[(*self.parity, self.ID)]
                ]
                * dt
            )
        Vt = np.where(abs(Vt) > Vmax, Vmax * np.sign(Vt), Vt)
        xf = np.round(xi + dt * Vt, ROUNDDIGIT)
        self.V = np.where(
            (xf > Global_variables.L - self.Size),
            -np.abs(Vt),
            np.where((xf < Global_variables.Linf + self.Size), np.abs(Vt), Vt),
        )

        return BOUNDS_Collision_Check(xi, xf, Vt, t, self.ID, self.parity, self.M)
