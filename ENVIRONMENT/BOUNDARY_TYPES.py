import numpy as np


from Particles.Global_Variables import Global_variables


def SIGNFLIP(a, p, id, z):
    """
    Flip the signs of selected elements of the input array a, based on the indices provided in Vflipinfo[p][id][:z].

    Args:
    a (array-like): The input array to modify.
    p (int): Index of the first-level list in Vflipinfo that contains the indices to be flipped.
    id (int): Index of the second-level list in Vflipinfo that contains the indices to be flipped.
    z (int): Index in the second-level list of the last index to be flipped.

    Returns:
    array-like: The modified input array, with the signs of the selected elements flipped.

    """
    for find in range(z - 1 + 1):
        flipindex, flipvalue = Global_variables.Vflipinfo[p[1]][p[0]][id][find]
        # flipindex,flipvalue=Vflipinfo[TYPE_to_index_Dict[p[1]]][p[0]][id][find]
        a[flipindex] = flipvalue
        # a[Vflipinfo[TYPE_to_index_Dict[p]][id][find]] *= -1
    return a


def BOUNDARY_FCT_HARD(A, p, id, z):
    """If boundaries are not periodic, then the particles need to be reflected. This function reverses the velocity of a particles in the direction that encounters the boundary.

    Args:
        A (_type_): Velocities
        p (_type_): parity of the particle
        id (_type_): id parity of the particle
        z (_type_): identifies which part of the trajectory of the particle we are looking at.
    """
    if z == 0:
        return A
    else:
        NewA = SIGNFLIP(np.copy(A), p, id, z)
        return NewA


def BOUNDARY_FCT_PER(A, p, id, z):
    """If boundaries are periodic then no need to reverse speeds at boundaries. Allthough this function looks useless it avoids checking boundaries for each loop"""
    return A
