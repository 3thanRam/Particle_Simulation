def Init(DIM_Numb, BOUNDARY_COND, L_FCT):
    from Particles.Global_Variables import init

    init(DIM_Numb, BOUNDARY_COND, L_FCT)

    from System.SystemClass import init_Syst

    init_Syst()
    from System.Position_class import init_pos

    init_pos()
