import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from dataclasses import dataclass, field
from typing import Callable

Numb_of_TYPES = len(PARTICLE_DICT)

PARTICLE_NAMES = [*PARTICLE_DICT.keys()]


DT = [0.1, 0.1, 0.05]  # time step (depends on numb of dimensions)
DistList = [1e-5, 1, 4]  # capture distance (depends on numb of dimensions)
Speed_light = 1
Global_variables = []


@dataclass(slots=True)
class Global_var:
    # Where Global variables are stored in order to commnicate them between different scripts
    DIM_Numb: int
    L_FCT: list
    BOUNDARY_COND: int

    L: np.ndarray = field(init=False)
    Linf: np.ndarray = field(init=False)

    V0: np.ndarray = field(init=False)
    Vmax: np.ndarray = field(init=False)

    Bound_Params: list = field(init=False)

    C_speed: float = Speed_light
    distmax: float = 0
    Dist_min: float = field(init=False)
    dt: float = field(init=False)

    Field_DICT: dict = field(default_factory=dict)

    ALL_TIME: list = field(default_factory=list)
    FIELD: list = field(default_factory=list)
    COLPTS: list = field(default_factory=list)

    File_path_name: str = ""
    ROUNDDIGIT: int = 15

    INTERCHECK: Callable = field(init=False)
    BOUNDARY_FCT: Callable = field(init=False)
    # SYSTEM: list = field(init=False)
    S_Force: np.ndarray = field(init=False)

    def __post_init__(self):
        self.L, self.Linf = self.L_FCT[0](0), self.L_FCT[1](0)
        self.V0 = Speed_light * np.ones(self.DIM_Numb)
        self.Vmax = self.V0 / 3

        self.Bound_Params = [0, 0, 0, 0]
        self.Dist_min = float(DistList[self.DIM_Numb - 1])
        self.dt = DT[self.DIM_Numb - 1]
        if self.DIM_Numb == 0:
            max_part_size = 0
        else:
            max_part_size = 1
        self.distmax = np.linalg.norm(self.Vmax * self.dt + max_part_size)

    def Update_Bound_Params(self, L, Linf, t):
        dt = self.dt
        L_prev, Linf_prev = self.L_FCT[0](t - dt), self.L_FCT[1](t - dt)
        Vsup, Vinf = (L - L_prev) / dt, (Linf - Linf_prev) / dt
        self.Bound_Params = [Vsup, L - Vsup * t, Vinf, Linf - Vinf * t]


def init(DIM_Numb, BOUNDARY_COND, L_FCT):
    global Global_variables
    Global_variables = Global_var(
        DIM_Numb=DIM_Numb, BOUNDARY_COND=BOUNDARY_COND, L_FCT=L_FCT
    )
