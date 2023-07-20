import numpy as np
import System.SystemClass
from dataclasses import dataclass, field
from Particles.Dictionary import PARTICLE_DICT
from Particles.Global_Variables import Global_variables
from Misc.Velocity_Fcts import UNIFORM  # RANDCHOICE,GAUSS
from Misc.Relativistic_functions import (
    gamma_factor,
    Energy_Calc,
    Velocity_add,
    Get_V_from_P,
    Momentum_Calc,
)

from Misc.Position_Fcts import GEN_X, in_all_bounds, Pos_point_around
from ENVIRONMENT.BOUNDARY_CHECK import BOUNDS_Collision_Check
from operator import itemgetter

PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
ROUNDDIGIT = Global_variables.ROUNDDIGIT
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb

dt = Global_variables.dt
Vmax = Global_variables.Vmax
BOUNDARY_COND = Global_variables.BOUNDARY_COND
rng = np.random.default_rng()
Xini = []
item_get = itemgetter(0, -2, 3, -1)
V_fct = UNIFORM

Fine_Struc_Cst = 0.0072973525628
charge_e = np.sqrt(4 * np.pi * Fine_Struc_Cst)


def GEN_V():
    v = V_fct()
    while np.linalg.norm(v) >= C_speed:
        v = V_fct()
    return v


def Velocity_Momentum(mass):
    velocity = GEN_V()
    if mass != 0:
        p = Momentum_Calc(velocity, mass)
        v = velocity
    else:
        p = velocity
        v = C_speed * velocity / np.linalg.norm(velocity)

    return (v, p)


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
    MOVE(t):
        Performs a single step of the simulation for the particle.

    """

    name: str
    parity: tuple
    ID: int
    ExtraParams: list = field(default_factory=list)
    M: float = field(init=False)
    Strong_Charge: float = field(init=False)
    Elec_Charge: float = field(init=False)
    Size: float = field(init=False)
    Colour_Charge: list = field(default_factory=list)

    X: np.ndarray = field(init=False)
    V: np.ndarray = field(init=False)

    Energy: float = field(init=False)
    P: float = field(init=False)

    Coef_param_list: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.M = PARTICLE_DICT[self.name]["mass"]
        self.Strong_Charge = PARTICLE_DICT[self.name]["Strong_Charge"]
        self.Size = PARTICLE_DICT[self.name]["size"] / 2
        if self.Strong_Charge != 0 and (not self.Colour_Charge):
            raise ValueError("Colour_Charge of quark ill defined at creation")

        partORanti, typeIndex = self.parity
        self.Elec_Charge = (
            (-1) ** partORanti * PARTICLE_DICT[self.name]["charge"] * charge_e
        )
        id = self.ID

        if (not self.ExtraParams) or self.ExtraParams[0] == "INIT_Quark_CREATION":
            # creation of particles at t=0
            if not self.ExtraParams:
                X = GEN_X(self.Size)
            else:
                X = Pos_point_around(self.ExtraParams[1], self.Size)

            V, P = Velocity_Momentum(self.M)
            E = Energy_Calc(P, self.M)
            System.SystemClass.SYSTEM.TRACKING[typeIndex][partORanti].append([[0.0, X]])
        elif (
            self.ExtraParams[0] == "Post_Interaction"
            or self.ExtraParams[0] == "Spont_Create"
        ):
            PosParam, VParam, TvalueParam, Energyval = self.ExtraParams[1:]
            X, V, E = PosParam, VParam, Energyval
            if typeIndex == 13:
                vdirection = V / np.linalg.norm(V)
                P = Energyval * vdirection / C_speed
                V = C_speed * vdirection
                E = Energy_Calc(P, self.M)
            else:
                P = Momentum_Calc(V, self.M)
            System.SystemClass.SYSTEM.TRACKING[typeIndex][partORanti].append(
                [[TvalueParam, X]]
            )
            System.SystemClass.SYSTEM.Vflipinfo[typeIndex][partORanti].append([])
        self.X, self.V, self.Energy, self.P = X, V, E, P

    def MOVE(self, t, DT=dt):
        xi = self.X
        Vt = self.V.copy()
        DE = 0
        if self.name != "photon":
            FORCE = Global_variables.FIELD
            field_index = Global_variables.Field_DICT[(*self.parity, self.ID)]
            # Get the position and velocity of the i-th particle
            gamma = gamma_factor(Vt)
            U = gamma * np.array([C_speed] + [vt for vt in Vt] + [0] * (3 - DIM_Numb))
            K = self.Elec_Charge * np.einsum(
                "ijk,j->ik", FORCE, U
            )  # self.Elec_Charge * FORCE[:, :, field_index].dot(U)

            F = K[1 : DIM_Numb + 1, field_index]  # / self.M
            dP = F * DT
            NewP = self.P + dP
            Vt = Get_V_from_P(NewP, self.M)

            NewE = Energy_Calc(NewP, self.M)
            Eloss_theo = np.dot(F, self.V) * DT
            DE = (NewE - self.Energy) - Eloss_theo
            self.P = NewP
            self.V = Vt
            self.Energy = NewE

        # if self.Strong_Charg!=0:
        #    Vstrong=Global_variables.Strong_FIELD[...]
        #    Vt=Velocity_add(Vt,Vstrong)

        xf = np.round(xi + DT * Vt, ROUNDDIGIT)
        self.V = np.where(
            (xf > Global_variables.L - self.Size),
            -np.abs(Vt),
            np.where((xf < Global_variables.Linf + self.Size), np.abs(Vt), Vt),
        )

        if in_all_bounds(xf, t, self.Size):
            do_info = [xf, [], [t]]
        else:
            do_info = BOUNDS_Collision_Check(
                xi, xf, Vt, t, self.ID, self.parity, self.M
            )
        a = Vt
        b = []
        xfin, xinter, Tparam = do_info

        if BOUNDARY_COND == 0:
            for r in range(len(xinter)):
                b.append(xinter[r][0] - a * float(Tparam[1 + r]))
            b.append(xfin - a * float(Tparam[0]))
        else:
            A = np.copy(a)
            for r in range(len(xinter)):
                b.append(xinter[r][0] - A * float(Tparam[1 + r]))
                flipindex, flipvalue = System.SystemClass.SYSTEM.Vflipinfo[
                    self.parity[1]
                ][self.parity[0]][self.ID][r]
                A[flipindex] = flipvalue
            b.append(xfin - A * float(Tparam[0]))

        self.Coef_param_list = [a, b, Tparam, self.parity, self.ID, xinter, xfin]
        return DE
