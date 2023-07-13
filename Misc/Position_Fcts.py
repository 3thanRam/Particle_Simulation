import numpy as np
from Particles.Global_Variables import Global_variables

ROUNDDIGIT = Global_variables.ROUNDDIGIT
V0 = Global_variables.V0
DIM_Numb = Global_variables.DIM_Numb
L_FCT = Global_variables.L_FCT
Dist_min = Global_variables.Dist_min
rng = np.random.default_rng()
Xini = []
Ini_size = []


def in_all_bounds(List, t=None, ParticleSize=0):
    if t is None:  # t == None:
        Lmaxi, Lmini = Global_variables.L, Global_variables.Linf
    else:
        Lmaxi, Lmini = L_FCT[0](t), L_FCT[1](t)
    Lmaxi -= ParticleSize
    Lmini += ParticleSize
    for indV, value in enumerate(List):
        if value > Lmaxi[indV] or value < Lmini[indV]:
            return False
    return True


def Pos_fct(min_value, max_value):
    """Generate a random position vector with DIM_Numb dimensions and values between min_value and max_value (both inclusive)

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
        dist = np.linalg.norm(XiArray - Particle_POS, axis=1)
        return (dist > Size_array).all()
    else:
        return True


def GEN_X(Part_size):
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


# while not CHECKstartPos(np.array(POS), Part_size):
#   POS = Pos_fct(Lmin, Lmax)
def Pos_fct2(Point):
    pos = Point * (1 + 0.01 * rng.uniform(-2, 2, Point.shape))
    return pos


def Pos_point_around(Point, Part_size):
    POS = Pos_fct2(Point)
    while not CHECKstartPos(POS, Part_size):
        POS = Pos_fct2(Point)
    return POS
