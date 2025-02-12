import numpy as np
from scipy.stats import rv_continuous
from scipy import integrate, interpolate
from Particles.Global_Variables import Global_variables
from Misc.Relativistic_functions import (
    lorentz_boost,
    Get_V_from_P,
)
from Misc.Functions import NORM
from Misc.Rotation_fcts import Rotate_vector_xyzAxes


C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb

angle_range = np.linspace(0, np.pi, 10**3)
Fine_Struc_Cst = 0.0072973525628
charge_e = np.sqrt(4 * np.pi * Fine_Struc_Cst)
Re = charge_e**2 / (4 * np.pi * C_speed**2)


class Probability_Distribution(rv_continuous):
    "Custom distribution"

    def _pdf(self, alpha, Ei_boosted_photon):
        """Normalised cross section of the interaction

        Args:
            alpha (float): angle
            Ei_boosted_photon (float): Energy of incident photon

        Returns:
            float representing probability of interaction for given parameters alpha, Ei_boosted_photon
        """
        Norm_cst = integrate.quad(
            self.Cross_section, args=(Ei_boosted_photon), a=0, b=np.pi
        )[0]
        return self.Cross_section(alpha, Ei_boosted_photon) / Norm_cst

    # def _cdf(self, alpha, Ei_boosted_photon):
    #    return integrate.quad(
    #        self.Cross_section, args=(Ei_boosted_photon), a=0, b=alpha
    #    )[0]
    # def Interpolate(self, X, Y):
    #    return interpolate.interp1d(X, Y)
    # def _ppf(self, x, Ei_boosted_photon):
    #    Angle_values = np.linspace(0, np.pi, 10**3)
    #    N_cst = self._cdf(np.pi, Ei_boosted_photon)
    #    vectorized_function = np.vectorize(self._cdf)
    #    Cumul_energy_values = (
    #        vectorized_function(Angle_values, Ei_boosted_photon) / N_cst
    #    )
    #    # Cumul_energy_values = [
    #    #    self.CDF(angle, Ei_boosted_photon) / N_cst for angle in Angle_values
    #    # ]
    #    # Cumul_energy_values = [
    #    #    self._cdf(angle, Ei_boosted_photon) for angle in Angle_values
    #    # ]
    #    ppf = self.Interpolate(Cumul_energy_values, Angle_values)
    #    return ppf(x)

    def Cross_section(self, alpha, Ei_boosted_photon):
        """Calcute cross section of the interaction

        Args:
            alpha (float): angle
            Ei_boosted_photon (float): Energy of incident photon

        Returns:
            DCS(float): cross section of interaction for given parameters alpha, Ei_boosted_photon
        """
        R_Ef_Ei = 1 / (1 + Ei_boosted_photon * (1 - np.cos(alpha)))
        DCS = 0.5 * (Re / R_Ef_Ei) ** 2 * (1 / R_Ef_Ei + R_Ef_Ei - np.sin(alpha) ** 2)
        return DCS


Distr_fct = Probability_Distribution(
    name="Custom distribution",
    a=0,
    b=np.pi,
)


def Compton_scattering(p1index, V1, M1, P1, E1, V2, M2, P2, E2):
    """Compute Compton_scattering between photon and massive particle by first boosting to particle velocity reference frame then performing the collision before returning to unboosted reference frame

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
    if p1index == 13:
        pi_photon, Ei_photon = P1, E1
        pi_particle, Ei_particle, M_particle = P2, E2, M2
    else:
        pi_photon, Ei_photon = P2, E2
        pi_particle, Ei_particle, M_particle = P1, E1, M1
    Vboost = pi_particle / Ei_particle
    Ei_boosted_photon, Pi_boosted_photon = lorentz_boost(Ei_photon, pi_photon, Vboost)
    Ei_boosted_particle, Pi_boosted_particle = lorentz_boost(
        Ei_particle, pi_particle, Vboost
    )
    theta = Distr_fct.rvs(Ei_boosted_photon)
    ini_direction = Pi_boosted_photon / NORM(Pi_boosted_photon)
    Ef_boosted_photon = Ei_boosted_photon / (
        1 + (Ei_boosted_photon * (1 - np.cos(theta)) / (M_particle * C_speed**2))
    )
    pf_boosted_photon_norm = Ef_boosted_photon / C_speed
    if DIM_Numb != 1:
        pf_boosted_photon_direction = Rotate_vector_xyzAxes(ini_direction, theta)
    else:
        pf_boosted_photon_direction = -ini_direction

    pf_boosted_photon = pf_boosted_photon_norm * pf_boosted_photon_direction
    pf_boosted_particle = Pi_boosted_photon - pf_boosted_photon
    Ef_boosted_particle = Ei_boosted_particle + Ei_boosted_photon - Ef_boosted_photon

    if p1index == 13:
        NewE1, NewP1 = lorentz_boost(
            Ef_boosted_photon, pf_boosted_photon, Vboost, INV=-1
        )
        VParam1 = C_speed * NewP1 / NORM(NewP1)
        NewE2, NewP2 = lorentz_boost(
            Ef_boosted_particle, pf_boosted_particle, Vboost, INV=-1
        )
        VParam2 = Get_V_from_P(NewP2, M2)
    else:
        NewE2, NewP2 = lorentz_boost(
            Ef_boosted_photon, pf_boosted_photon, Vboost, INV=-1
        )
        VParam2 = C_speed * NewP2 / NORM(NewP2)
        NewE1, NewP1 = lorentz_boost(
            Ef_boosted_particle, pf_boosted_particle, Vboost, INV=-1
        )
        VParam1 = Get_V_from_P(NewP1, M1)
    return (VParam1, NewP1, NewE1, VParam2, NewP2, NewE2)
