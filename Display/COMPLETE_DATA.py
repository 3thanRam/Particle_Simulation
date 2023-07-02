def COMPLETE_DATAPTS(DATApts, TIMEpts, DIM_Numb):
    """adds interpolation of points at interaction times so that all particles have a position at each time (where they're alive)

    Args:
        DATApts (_type_): Tracking data containing existing positions of each particle at each time step or event that the particle was involved in
        TIMEpts (_type_): list of times of all events
    """
    if DIM_Numb == 2:
        DEADpart = ["T", ["X", "Y"]]
    else:
        DEADpart = ["T", ["X", "Y", "Z"]]

        def EMPTY_DOUBLE_LIST(DIM_Numb, dataN, DATA):
            Particle_list = [
                [[] for di in range(DIM_Numb)] for n in range(len(DATA[dataN][0]))
            ]
            ANTIPart_list = [
                [[] for di in range(DIM_Numb)] for n in range(len(DATA[dataN][1]))
            ]
            return [Particle_list, ANTIPart_list]

        PART = [EMPTY_DOUBLE_LIST(3, p, DATApts) for p in range(len(DATApts))]

        # PART=[[[[] for di in range(DIM_Numb)] for n in range(Nu[i])] for i in range(len(DATApts))]
        # Nu=[len(DATApts[0]),len(DATApts[1]),len(DATApts[2])]
        # PART=[[[[] for di in range(DIM_Numb)] for n in range(Nu[i])] for i in range(len(DATApts))]

    for p in range(len(DATApts)):
        for partORanti, partdata in enumerate(DATApts[p]):
            if not partdata:
                continue
            for dataindex, particledata in enumerate(partdata):
                ParticleTimes = [particlevalues[0] for particlevalues in particledata]
                Tend = ParticleTimes[-1]
                Tstart = Tend
                for partTime in ParticleTimes:
                    if not isinstance(partTime, str):
                        if partTime < Tstart:
                            Tstart = partTime

                Newparticledata = [[] for ti in range(len(TIMEpts))]

                for timeindex, timevalue in enumerate(TIMEpts):
                    if timevalue == Tend:
                        Newparticledata[timeindex] = particledata[-1]
                        Newparticledata[timeindex + 1 :] = [DEADpart] * (
                            len(TIMEpts) - timeindex - 1
                        )

                        if DIM_Numb == 3:
                            POS = particledata[-1][1]
                            for dim in range(DIM_Numb):
                                PART[p][partORanti][dataindex][dim].append(POS[dim])
                                PART[p][partORanti][dataindex][dim].extend(
                                    ["D"] * (len(TIMEpts) - timeindex - 1)
                                )
                        break
                    elif timevalue in ParticleTimes:
                        part_timeindex = ParticleTimes.index(timevalue)
                        Newparticledata[timeindex] = particledata[part_timeindex]
                        if DIM_Numb == 3:
                            POS = particledata[part_timeindex][1]
                            for dim in range(DIM_Numb):
                                PART[p][partORanti][dataindex][dim].append(POS[dim])
                    elif timevalue < Tstart:
                        if DIM_Numb == 3:
                            POS = ["X", "Y", "Z"]
                        else:
                            POS = ["X", "Y"]
                        Newparticledata[timeindex] = ["PreCreation", POS]
                        if DIM_Numb == 3:
                            for dim in range(DIM_Numb):
                                PART[p][partORanti][dataindex][dim].append(POS[dim])
                    else:
                        PrevPart = Newparticledata[timeindex - 1]
                        t0 = PrevPart[0]
                        POS = [[] for d in range(DIM_Numb)]
                        pos0 = PrevPart[1]

                        if timeindex < len(TIMEpts) - 1:
                            nexttimeval = TIMEpts[timeindex + 1]
                            if nexttimeval in ParticleTimes:
                                Nextpart_index = ParticleTimes.index(nexttimeval)
                                NextPart = particledata[Nextpart_index]
                                if not isinstance(NextPart[0], str):
                                    t1 = NextPart[0]
                                    pos1 = NextPart[1]
                                    R0 = (timevalue - t0) / (t1 - t0)
                                    R1 = 1 - R0
                                    for d in range(DIM_Numb):
                                        POS[d] = R0 * pos0[d] + R1 * pos1[d]

                            if not POS[0]:
                                POS = pos0
                            Newparticledata[timeindex] = [timevalue, POS]
                            if DIM_Numb == 3:
                                for dim in range(DIM_Numb):
                                    PART[p][partORanti][dataindex][dim].append(POS[dim])
                DATApts[p][partORanti][dataindex] = Newparticledata
    if DIM_Numb == 2:
        return DATApts
    else:
        return PART
