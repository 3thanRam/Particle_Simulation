LEPTON_INDEXES = [0, 2, 4]
NEUTRINO_INDEXES = [1, 3, 5]
QUARK_INDEXES = [6, 7, 8, 9, 10, 11]

BOSON_INDEXES = [12, 13, 14, 15, 16]

FERMION_INDEXES = [i for i in range(12)]


def COLTYPE(pA, pB):
    """
    Gets type of interaction between 2 particles based on the type of particles involved

    Args:
        pA (tuple): information about the type of first particle
        pB (tuple): information about the type of a second particle

    Returns:
        int representing the type of interaction
    """
    partORantiA, IndexA = pA
    partORantiB, IndexB = pB

    if IndexA == IndexB and IndexA != 13:  # photons dont interact with each other
        if partORantiA == partORantiB:
            if 5 < IndexA < 12:
                return 0  # quark-quark Collision should depend on colour (annihilation also)
            else:
                return 3  # Collision
        else:
            return 2  # Annihilation
    elif IndexA != IndexB:
        if IndexA < 12:
            if IndexB < 12:
                if 5 < IndexA < 12 and 5 < IndexB < 12:
                    return 0  # Collision
                else:
                    return 3  # Collision
            elif IndexB == 13:
                return 3  # compton scattering
        elif IndexB < 12 and IndexA == 13:
            return 3  # compton scattering
    return 0  # no interaction


# Conservation laws:
#  Energy, momentum
#  Angular momentum
#  Electric charge
#  Lepton flavour
#  Baryon number
#  Quark flavour (depending on the interaction)
