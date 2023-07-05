import numpy as np

from Particles.Dictionary import PARTICLE_DICT

PARTICLE_NAMES = [*PARTICLE_DICT.keys()]

from Misc.Functions import ROUND
from Misc.Position_Fcts import in_all_bounds

from Particles.Global_Variables import Global_variables

DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
L_FCT = Global_variables.L_FCT
BOUNDARY_COND = Global_variables.BOUNDARY_COND


def BOUNDS_Collision_Check(xi, xf, V, t, id, p, Mass):
    """
    Computes the boundaries for a particle based on its position and velocity.

    Args:
    - xi (array): The initial position of the particle.
    - xf (array): The final position of the particle.
    - V (array): The velocity of the particle.
    - t (float): The current time.
    - id (int): The ID of the particle.
    - p (int): The ID of the process that is executing the function.

    Returns:
    - A list with the following elements:
        - Xini (array): The initial position of the particle.
        - Xinter (array): The intermediate positions of the particle (if any).
        - Xfin (array): The final position of the particle.
        - V (array): The velocity of the particle.
        - t_list (list): A list with the time values at which the particle reaches an intermediate position.
        - id (int): The ID of the particle.
        - NZ (int): The number of intermediate positions.
    """
    PART_SIZE = PARTICLE_DICT[PARTICLE_NAMES[p[1]]]["size"] / 2

    Xini = xi
    Xfin = xf
    Xinter = []
    t_list = [t]

    # Check if the final position is inside the boundaries
    if in_all_bounds(Xfin, t, PART_SIZE):
        return [Xfin, p, id, Xinter, V, t_list, 0]

    # Initialize parameters
    t_params = np.zeros(DIM_Numb)
    x_params = np.zeros((DIM_Numb, 2))

    # Check if the particle hits any boundary
    Vchgsign = V.copy()
    BoundSup_V, BoundSup_b, BoundInf_V, BoundInf_b = Global_variables.Bound_Params
    for d in range(DIM_Numb):
        b = Xfin[d] - V[d] * t

        CONDsup, CONDinf = False, False
        if BoundSup_V[d] != V[d]:
            SOL_sup = ROUND((b - (BoundSup_b[d] - PART_SIZE)) / (BoundSup_V[d] - V[d]))
            CONDsup = t - dt <= SOL_sup <= t
        if BoundInf_V[d] != V[d]:
            SOL_inf = ROUND((b - (BoundInf_b[d] + PART_SIZE)) / (BoundInf_V[d] - V[d]))
            CONDinf = t - dt <= SOL_inf <= t

        if CONDinf:
            t0 = ROUND(SOL_inf)
        elif CONDsup:
            t0 = ROUND(SOL_sup)
        else:
            continue
        L_t0, Linf_t0 = L_FCT[0](t0)[d] - PART_SIZE, L_FCT[1](t0)[d] + PART_SIZE
        Xinterd = b + V * t0

        if CONDinf:
            if BOUNDARY_COND == 0:
                Li_0, Li_1 = Linf_t0, L_t0
                bi = L_t0 - V[d] * t0
                # bi=b+(L_t0[d]-Linf_t0[d])
            else:
                if -V[d] + (BoundInf_V[d]) < BoundInf_V[d]:
                    Vchgsign[d] = np.sign(BoundInf_V[d]) * abs(V[d]) + (BoundInf_V[d])
                else:
                    Vchgsign[d] = -V[d] + (BoundInf_V[d])

                bi = Xinterd[d] - Vchgsign[d] * t0
                Li_0, Li_1 = Linf_t0, Linf_t0
        else:
            if BOUNDARY_COND == 0:
                Li_0, Li_1 = L_t0, Linf_t0
                bi = Linf_t0 - V[d] * t0
                # bi=b-(L_t0[d]-Linf_t0[d])
            else:
                if -V[d] + (BoundSup_V[d]) > BoundSup_V[d]:
                    Vchgsign[d] = np.sign(BoundSup_V[d]) * abs(V[d]) + (BoundSup_V[d])
                else:
                    Vchgsign[d] = -V[d] + (BoundSup_V[d])

                bi = Xinterd[d] - Vchgsign[d] * t0
                Li_0, Li_1 = L_t0, L_t0

        t_params[d] = t0
        x_params[d][0], x_params[d][1] = Li_0, Li_1
        Xfin[d] = Vchgsign[d] * t + bi

    # Check if the particle hits any boundary and get the number of hits
    NZ = np.count_nonzero(t_params)

    # Compute intermediate positions if the particle hits any boundary
    if NZ > 0:
        Xinter = np.zeros((NZ, 2, DIM_Numb))
        d_prev = 0
        # Vinter=V.copy()
        Vinter = np.zeros_like(V)
        for nz in range(NZ):
            mask = np.where((t_params > 0) & (t_params != np.inf))
            d_arg = mask[0][t_params[mask].argmin()]
            t_val = t_params[d_arg]
            t_list.append(t_val)
            L_tv_d = L_FCT[0](t_val)[d_arg] - PART_SIZE
            Linf_tv_d = L_FCT[1](t_val)[d_arg] + PART_SIZE

            if nz == 0:
                Xinter[nz][0] = np.where(
                    (Xini + V * dt > Linf_tv_d) & (Xini + V * dt < L_tv_d),
                    Xini + V * dt,
                    Xini,
                )
                Xinter[nz][1] = np.where(
                    (Xini + V * dt > Linf_tv_d) & (Xini + V * dt < L_tv_d),
                    Xini + V * dt,
                    Xini,
                )
                Vinter[d_arg] = Vchgsign[d_arg]
            else:
                val0 = Xinter[nz - 1][0] + Vinter * (t_val - (t - dt))  # -t_list[-2])
                Xinter[nz][0] = np.where(
                    (val0 > Linf_tv_d) & (val0 < L_tv_d), val0, Xinter[nz - 1][0]
                )
                val1 = Xinter[nz - 1][1] + Vinter * (t_val - (t - dt))
                Xinter[nz][1] = np.where(
                    (val1 > Linf_tv_d) & (val1 < L_tv_d), val1, Xinter[nz - 1][1]
                )
                Xinter[nz][0][d_prev] = Xinter[nz][1][d_prev]
                Vinter[d_arg] = Vchgsign[d_arg]

            Xinter[nz][0][d_arg] = x_params[d_arg][0]
            Xinter[nz][1][d_arg] = x_params[d_arg][1]
            t_params[d_arg] = np.inf
            d_prev = d_arg
            Global_variables.Vflipinfo[p[1]][p[0]][id].append([d_arg, Vchgsign[d_arg]])
    return [Xfin, p, id, Xinter, V, t_list, NZ]
