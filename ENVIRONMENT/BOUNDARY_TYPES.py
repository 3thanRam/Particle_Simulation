import System.SystemClass as System_module


def BOUNDARY_FCT_HARD(A, p, id, z):
    """If boundaries are not periodic, then the particles need to be reflected. This function gets the new velocity if there's a boundary collision

    Args:
        A (ndarray): Velocities
        p (tuple): parity of the particle
        id (int): id parity of the particle
        z (int): identifies which part of the trajectory of the particle we are looking at.
    """
    if z == 0:
        return A
    else:
        NewA = System_module.SYSTEM.Vflipinfo[p[1]][p[0]][id][z - 1]
        return NewA


def BOUNDARY_FCT_PER(A, p, id, z):
    """If boundaries are periodic then no need to reverse speeds at boundaries. Allthough this function looks useless it avoids checking boundaries for each loop"""
    return A
