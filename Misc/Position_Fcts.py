import numpy as np

from Particles.Global_Variables import Global_variables
from Misc.Functions import NORM

ROUNDDIGIT = Global_variables.ROUNDDIGIT
V0 = Global_variables.V0
DIM_Numb = Global_variables.DIM_Numb
L_FCT = Global_variables.L_FCT
Dist_min = Global_variables.Dist_min

rng = np.random.default_rng()
Xini = []
Ini_size = []


def In_1D_bounds(array, L_up, L_down):
    return L_down[0] < array[0] < L_up[0]


def In_2D_bounds(array, L_up, L_down):
    return (L_down[0] < array[0] < L_up[0]) & (L_down[1] < array[1] < L_up[1])


def In_3D_bounds(array, L_up, L_down):
    return (
        (L_down[0] < array[0] < L_up[0])
        & (L_down[1] < array[1] < L_up[1])
        & (L_down[2] < array[2] < L_up[2])
    )


if DIM_Numb == 3:
    In_Bounds = In_3D_bounds
elif DIM_Numb == 2:
    In_Bounds = In_2D_bounds
elif DIM_Numb == 1:
    In_Bounds = In_1D_bounds

Array1, Array2 = np.empty(DIM_Numb), np.empty(DIM_Numb)


def GetLmaxmin(time, ParticleSize):
    """Get upper and lower limits for the center of the particle, annoyingly fast method of creating an array"""
    Lup, Ldown = L_FCT[0](time), L_FCT[1](time)
    Array1 = Lup - ParticleSize
    Array2 = Ldown + ParticleSize
    return Array1, Array2


def in_all_bounds(POSarray, time, ParticleSize=0):
    """Returns True if all elements of POSarray are inside the box"""
    Lmaxi, Lmini = GetLmaxmin(time, ParticleSize)

    return In_Bounds(POSarray, Lmaxi, Lmini)


def Pos_fct(min_value, max_value):
    """Generate a random position vector with DIM_Numb dimensions and values between min_value and max_value

    Args:
        min_value (float or int): The minimum value for each dimension of the position vector
        max_value (float or int): The maximum value for each dimension of the position vector

    Returns:
        A list of length DIM_Numb representing the position vector
    """
    # Create a list of DIM_Numb random values between min_value and max_value (both excluded)
    pos = rng.uniform(min_value + 1e-12, max_value)
    return pos


def CHECKstartPos(Particle_POS, Particle_size):
    """
    Checks if the initial position of a particle is far enough from the others.

    Args:
    testPOS (List[float]): The initial position of the particle.

    Returns:
    True if the position is not valid, False otherwise.
    """
    if not in_all_bounds(Particle_POS, 0):
        return False
    if len(Xini) > 0:
        XiArray = np.array(Xini)
        Size_array = np.array(Ini_size) + Particle_size
        dist = NORM(XiArray - Particle_POS, axis=1)
        return (dist > Size_array).all()
    else:
        return True


def GEN_X(Part_size):
    """Generates a random position within a given distance of minimum separation from other particles.

    Args:
    - Part_size (float): size of particle, so minimum distance allowed between generated position and existing positions.

    Returns:
    - List[int]: a new particle position that meets the minimum distance requirement from other particles.
    """
    # Generate a new position randomly within the bounds of Global_variables.L.
    SizeArray = Part_size * np.ones(DIM_Numb)

    Lmin = L_FCT[1](0) + SizeArray
    Lmax = L_FCT[0](0) - SizeArray
    POS = Pos_fct(Lmin, Lmax)
    # While the generated position is not far enough from existing positions,
    # generate a new position.
    while not CHECKstartPos(np.array(POS), Part_size):
        POS = Pos_fct(Lmin, Lmax)
    # Add the new position to the list of existing positions.
    Xini.append(POS)
    Ini_size.append(Part_size)
    return POS


def Pos_fct2(Point):
    """
    Generate random location around Point
    """
    pos = Point * (1 + 0.01 * rng.uniform(-2, 2, Point.shape))
    return pos


def Pos_point_around(Point, Part_size):
    """Generate the position of particle of size Part_size around a point Point

    Args:
        Point (ndarray): location of point
        Part_size (float): size of particle

    Returns:
        POS (ndarray): location to place particle
    """
    POS = Pos_fct2(Point)
    Count = 0
    while not CHECKstartPos(POS, Part_size):
        POS = Pos_fct2(Point)
        Count += 1
        if Count > 10**4:
            raise RuntimeError("Couldn't find room for particle, exiting program")
    return POS
