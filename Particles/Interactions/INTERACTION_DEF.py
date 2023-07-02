def COLTYPE(pA, pB):
    partORantiA, IndexA = pA
    partORantiB, IndexB = pB

    if IndexA == IndexB and IndexA != 13:  # photons dont interact with each other
        if partORantiA == partORantiB:
            return 3  # Collision
        else:
            return 2  # Annihilation
    elif IndexA != IndexB:
        if (IndexA == 13 and IndexB != 13) or (IndexB == 13 and IndexA != 13):
            return 1  # absorption of photon
        else:
            return 3  # Collision
    return 0  # no interaction


# Conservation laws:
#  Energy, momentum
#  Angular momentum
#  Electric charge
#  Lepton flavour
#  Baryon number
#  Quark flavour (depending on the interaction)
