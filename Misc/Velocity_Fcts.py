import numpy as np
from Particles.Global_Variables import Global_variables

ROUNDDIGIT = Global_variables.ROUNDDIGIT
V0 = Global_variables.V0
DIM_Numb = Global_variables.DIM_Numb
C_speed = Global_variables.C_speed

rng = np.random.default_rng()


def GAUSS():
    """Generates a random velocity vector based on gaussian distribution of mean 0 and Standard deviation V0/3.

    Global Args Used:
    - V0 (float): Mean (“centre”) of the distribution.
    - sd (float): Standard deviation of the distribution (Non-negative).
    - DIM_Numb (int): number of dimensions.
    - ROUNDDIGIT (int): number of decimal places to round velocity components to.

    Returns:
    - np.ndarray: a random velocity vector with values rounded to ROUNDDIGIT decimal places.
    """
    V = np.round(rng.normal(0, V0 / 3, DIM_Numb), ROUNDDIGIT)
    while np.linalg.norm(V) > C_speed:
        V = np.round(rng.normal(0, V0 / 3, DIM_Numb), ROUNDDIGIT)
    return V


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
