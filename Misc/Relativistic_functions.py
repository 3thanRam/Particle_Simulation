import numpy as np
from Particles.Global_Variables import Global_variables
from Misc.Functions import NORM

C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb

# import warnings

# warnings.filterwarnings("error")


def RELAT_ERROR(velocity):
    raise ValueError("Error Particle velocity >= c", velocity)


def gamma_factor(v):
    """
    Calculate lorentz factor of velocity v
    """
    if isinstance(v, float):
        V = v
    else:
        V = NORM(v)
    if V >= C_speed:
        RELAT_ERROR(V)
    return 1 / (1 - (V / C_speed) ** 2) ** 0.5


def lorentz_boost(Energy, momentum, Velocity, INV=1):
    """
    Perform lorentz boost of Energy, momentum by into reference with velocity
    """
    V_norm = NORM(Velocity)
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
    """
    Calculate velocity of particle of given mass and momentum
    """
    P_norm = NORM(momentum)
    direction = momentum / P_norm
    V_norm = P_norm * (1 / (1 + (P_norm / (mass * C_speed)) ** 2)) ** 0.5 / mass
    return V_norm * direction


def Momentum_Calc(velocity, mass):
    """
    Calculate momentum of particle of given mass and velocity
    """
    return mass * velocity * gamma_factor(velocity)


def Energy_Calc(momentum, mass):
    """
    Calculate Energy of particle of given mass and momentum
    """
    return (
        (mass * C_speed**2) ** 2 + C_speed**2 * np.dot(momentum, momentum)
    ) ** 0.5


def Velocity_add(v1, v2):
    """
    Perform relativistic velocity addition
    """
    Dotprod = np.dot(v1, v2)
    prefact = 1 / (1 + Dotprod * C_speed**2)
    reci_gamma = 1 / gamma_factor(NORM(v2))

    Vfin = prefact * (
        reci_gamma * v1 + v2 + (1 - reci_gamma) * Dotprod * v2 / np.dot(v2, v2)
    )

    return Vfin
