import numpy as np
from Particles.Global_Variables import Global_variables
from Misc.Relativistic_functions import (
    Get_V_from_P,
    lorentz_boost,
)
from Misc.Rotation_fcts import Rotate_vector_xyzAxes
from scipy.stats import rv_continuous

angle_range = np.linspace(0, np.pi, 10**3)
Fine_Struc_Cst = 0.0072973525628
charge_e = np.sqrt(4 * np.pi * Fine_Struc_Cst)
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb


class Probability_Distribution(rv_continuous):
    "Custom distribution"

    def _pdf(self, angle, m1, E1, m2, E2):
        """Normalised cross section of the interaction

        Args:
            angle (float): angle
            m1 (float): mass of particle 1
            E1 (float): initial Energy of particle 1
            m2 (float): mass of particle 2
            E2 (float): initial Energy of particle 2

        Returns:
            float representing probability of interaction for given parameters angle, m1, E1, m2, E2
        """
        s = (E1 + E2) ** 2
        E = (s / 4) ** 0.5
        Norm_cst = np.pi * (Fine_Struc_Cst / E) ** 2 / 3
        return self.Cross_section(angle, m1, E1, m2, E2) / Norm_cst

    def Cross_section(self, angle, m1, E1, m2, E2):
        """Calcute cross section of the interaction

        Args:
            angle (float): angle
            m1 (float): mass of particle 1
            E1 (float): initial Energy of particle 1
            m2 (float): mass of particle 2
            E2 (float): initial Energy of particle 2

        Returns:
            DCS(float): cross section of interaction for given parameters  angle, m1, E1, m2, E2
        """
        s = (E1 + E2) ** 2
        E = (s / 4) ** 0.5
        Cos2 = np.cos(angle) ** 2
        Sin2 = 1 - Cos2
        if E < 10**10:
            m1_2, m2_2 = m1**2, m2**2
            E2 = E**2
            R_s = (m1_2 + m2_2) / E2
            R_c = m1_2 * m2_2 / E2**2
        else:
            R_s, R_c = 0, 0
        M2 = (4 * np.pi * Fine_Struc_Cst) ** 2 * (1 + Cos2 + R_s * Sin2 + R_c * Cos2)
        DCS = M2 / (64 * np.pi**2 * s)
        return DCS


Distr_fct = Probability_Distribution(
    name="Custom distribution",
    a=0,
    b=np.pi,
)


def Relativistic_Collision(V1, M1, P1, E1, V2, M2, P2, E2):
    """Compute Relativistic ellastic collision between 2 massive particles by first boosting to center of momentum reference frame then performing the collision before returning to unboosted reference frame

    Args:
        p1index (tuple): particle 1 index
        V1 (ndarray): particle 1 velocity
        M1 (float): particle 1 mass
        P1 (ndarray): particle 1 momentum
        E1 (float): particle 1 Energy
        V2 (ndarray): particle 2 velocity
        M2 (float): particle 2 mass
        P2 (ndarray): particle 2 momentum
        E2 (float): particle 2 Energy

    Returns:
        VParam1 (ndarray): New particle 1 velocity
        M1 (float): particle 1 mass
        NewP1 (ndarray):New particle 1 momentum
        NewE1 (float): New particle 1 Energy
        VParam2 (ndarray): New particle 2 velocity
        M2 (float): particle 2 mass
        NewP2 (ndarray): New particle 2 momentum
        NewE2 (float): New particle 2 Energy
    """
    if DIM_Numb == 1:
        return (V1, P1, E1, V2, P2, E2)

    # Vboost = P1 / E1
    COMboost = (P1 + P2) / (E1 + E2)
    Vboost = COMboost
    Boost_E1_ini, Boost_P1_ini = lorentz_boost(E1, P1, Vboost)
    Boost_E2_ini, Boost_P2_ini = lorentz_boost(E2, P2, Vboost)

    theta = Distr_fct.rvs(M1, E1, M2, E2)
    # theta_lab = M2 * np.sin(theta) / (E1 + E2 * np.cos(theta))
    Boost_P2_fin = Rotate_vector_xyzAxes(Boost_P2_ini, theta)

    Boost_P1_fin = Boost_P1_ini + Boost_P2_ini - Boost_P2_fin

    Boost_E1_fin = (
        Boost_E1_ini**2
        + C_speed**2
        * (np.dot(Boost_P1_fin, Boost_P1_fin) - np.dot(Boost_P1_ini, Boost_P1_ini))
    ) ** 0.5

    # Boost_E2_fin=Boost_E1_ini+Boost_E2_ini-Boost_E1_fin
    Boost_E2_fin = (
        Boost_E2_ini**2
        + C_speed**2
        * (np.dot(Boost_P2_fin, Boost_P2_fin) - np.dot(Boost_P2_ini, Boost_P2_ini))
    ) ** 0.5

    NewE1, NewP1 = lorentz_boost(Boost_E1_fin, Boost_P1_fin, Vboost, INV=-1)
    NewE2, NewP2 = lorentz_boost(Boost_E2_fin, Boost_P2_fin, Vboost, INV=-1)
    VParam1 = Get_V_from_P(NewP1, M1)
    VParam2 = Get_V_from_P(NewP2, M2)

    return (VParam1, NewP1, NewE1, VParam2, NewP2, NewE2)
