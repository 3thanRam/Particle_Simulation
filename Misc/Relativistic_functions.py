import numpy as np
from Particles.Global_Variables import Global_variables

C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb

import warnings

warnings.filterwarnings("error")


def gamma_factor(v):
    return 1 / (1 - (v / C_speed) ** 2) ** 0.5


def lorentz_boost(Energy, momentum, Velocity, INV=1):
    V_norm = np.linalg.norm(Velocity)
    V_direction = Velocity / V_norm
    Gamma = gamma_factor(V_norm)
    PdotV = np.dot(momentum, V_direction)
    New_E = Gamma * (Energy - INV * V_norm * PdotV / C_speed)
    New_P = (
        momentum
        + (Gamma - 1) * PdotV * V_direction
        - INV * Gamma * Energy * Velocity / C_speed
    )
    return New_E, New_P


def Get_V_from_P(momentum, mass):
    P_norm = np.linalg.norm(momentum)
    direction = momentum / P_norm
    V_norm = P_norm * (1 / (1 + (P_norm / (mass * C_speed)) ** 2)) ** 0.5 / mass
    return V_norm * direction


def Energy_Calc(momentum, mass):
    return (
        (mass * C_speed**2) ** 2 + np.linalg.norm((momentum * C_speed) ** 2)
    ) ** 0.5


def Velocity_add(v1, v2):
    Dotprod = np.dot(v1, v2)
    prefact = 1 / (1 + Dotprod * C_speed**2)
    reci_gamma = 1 / gamma_factor(np.linalg.norm(v2))

    Vfin = prefact * (
        reci_gamma * v1 + v2 + (1 - reci_gamma) * Dotprod * v2 / np.dot(v2, v2)
    )

    return Vfin
