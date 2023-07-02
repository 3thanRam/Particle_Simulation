# Parity violation
# Helicity = spin projection along momentum axis
# One defines also Chirality which tends to be ~ the helicity in
# the ultrarelativistic limit

# The chirality of a particle is more abstract: It is determined by whether the
# particle transforms in a right- or left-handed representation of the Poincaré
# group. Chirality can be left handed or right handed.

# Only left handed particles and right handed antiparticles are
# involved in weak interaction


def W_plus():  # Does modify quark flavor
    # charge +1
    # quark:
    # (u,c,t)->(d,s,b)
    # (d,s,b)->(u,c,t)

    # lepton - neutrinos: W+ ->e+ & ve

    return ()


def W_minus():  # Does modify quark flavor
    # charge -1
    # quark:
    # (u,c,t)->(d,s,b)
    # (d,s,b)->(u,c,t)

    # lepton - neutrinos: W+ ->e- & anti_ve
    return ()


def Z():  # Neutral
    # basically phton but with mass and diff coupling constant
    return ()


"""
transition amplitude is related to the CKM matrix:

[[Vud,Vus,Vub],
[Vcd,Vcs,Vcb],
[Vtd,Vts,Vtb]]

[[0.974,0.226,0.004],
[0.226,0.973,0.042],
[0.009,0.041,0.999]]
"""
