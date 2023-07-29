import numpy as np
import System.SystemClass as System_module
from Particles.Dictionary import PARTICLE_DICT
from Misc.Functions import ROUND
from Misc.Position_Fcts import in_all_bounds
from Particles.Global_Variables import Global_variables

PARTICLE_NAMES = [*PARTICLE_DICT.keys()]
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
L_FCT = Global_variables.L_FCT
BOUNDARY_COND = Global_variables.BOUNDARY_COND


def BOUNDS_Collision_Check(Xfin, Velocity, t, id, p):
    """
    Computes possible boundary collisions for a particle based on its position and velocity.

    Args:
    - xf (array): The final position of the particle.
    - V (array): The velocity of the particle.
    - t (float): The current time.
    - id (int): The ID of the particle.
    - p (int): The ID of the process that is executing the function.

    Returns:
    - A list with the following elements:
        - Xinter (array): The intermediate positions of the particle (if any).
        - Xfin (array): The final position of the particle.
        - t_list (list): A list with the time values at which the particle reaches an intermediate position.
        - b_list (list): list of b coeficient of affine trajectories for each intermediate position
    """
    PART_SIZE = PARTICLE_DICT[PARTICLE_NAMES[p[1]]]["size"] / 2
    Xinter = []
    t_list = [t]

    BoundSup_V, BoundSup_b, BoundInf_V, BoundInf_b = Global_variables.Bound_Params

    b_list = []
    t_prev = t - dt
    X_inter = Xfin

    def Bounds():
        nonlocal t_prev, X_inter
        tmin = np.inf

        b = X_inter - Velocity * t
        b_list.append(b)

        for d in range(DIM_Numb):
            SOL_sup = ROUND(
                (b[d] - (BoundSup_b[d] - PART_SIZE)) / (BoundSup_V[d] - Velocity[d])
            )
            CONDsup = t_prev < SOL_sup < t
            SOL_inf = ROUND(
                (b[d] - (BoundInf_b[d] + PART_SIZE)) / (BoundInf_V[d] - Velocity[d])
            )
            CONDinf = t_prev < SOL_inf < t
            if CONDinf:
                if CONDsup:
                    t_test = min(SOL_sup, SOL_inf)
                    if t_test < tmin:
                        tmin = t_test
                        d_chg = d
                        chg_param = 1 * (SOL_sup > SOL_inf)
                elif SOL_inf < tmin:
                    tmin = SOL_inf
                    d_chg = d
                    chg_param = 1
            elif CONDsup and SOL_sup < tmin:
                tmin = SOL_sup
                d_chg = d
                chg_param = -1

        if tmin != np.inf:
            t_list.append(tmin)
            if BOUNDARY_COND == 0:
                Xint1, Xint2 = b + Velocity * tmin, b + Velocity * tmin
            else:
                Xint1, Xint2 = b + Velocity * tmin, b + Velocity * tmin
            Xinter.append([Xint1, Xint2])
            t_prev = tmin
            System_module.SYSTEM.Vflipinfo[p[1]][p[0]][id].append(Velocity)
            Velocity[d_chg] = chg_param * abs(Velocity[d_chg])
            X_inter = Xint2 + Velocity * (t - tmin)
            return not in_all_bounds(X_inter, t, PART_SIZE)
        else:
            return False

    Count = 0
    while True:
        if not Bounds():
            break
        Count += 1
        if Count > 10:
            raise ValueError("Can't advance particle inside of boundaries")

    return X_inter, np.array(Xinter), t_list, b_list
