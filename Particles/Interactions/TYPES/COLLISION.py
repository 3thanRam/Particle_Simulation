import numpy as np
import vg

rng = np.random.default_rng()


from Particles.Dictionary import PARTICLE_DICT
from Particles.ParticleClass import Particle
from Particles.Global_Variables import Global_variables


BOUNDARY_COND = Global_variables.BOUNDARY_COND
if BOUNDARY_COND == 0:
    from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_PER

    BOUNDARY_FCT = BOUNDARY_FCT_PER
else:
    from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_HARD

    BOUNDARY_FCT = BOUNDARY_FCT_HARD
ROUNDDIGIT = Global_variables.ROUNDDIGIT
C_speed = Global_variables.C_speed
DIM_Numb = Global_variables.DIM_Numb
dt = Global_variables.dt
Vmax = Global_variables.Vmax


def COLLIDE(FirstAnn, F, COEFSlist, t):
    """update tracking information after collision point.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - F (list): List of dictionaries containing information about the particles in the simulation, indexed by particle type and ID.

    Returns:
    - F (list): Updated list of dictionaries containing information about the particles in the simulation.
    """

    # Extract information about the collision
    ti, xo, coltype, z1, z2, p1, id1, p2, id2 = FirstAnn

    Global_variables.ALL_TIME.append(ti)
    Global_variables.COLPTS.append([ti, xo, coltype])  # FirstAnn)

    return (F, COEFSlist)
    """
    partORAnti1=p1[0]
    partORAnti2=p2[0]

    Global_variables.TRACKING[p1[1]][partORAnti1][id1].append([ti,xo])
    Global_variables.TRACKING[p2[1]][partORAnti2][id2].append([ti,xo])


    for s in range(len(Global_variables.SYSTEM[p1[1]][partORAnti1])):
        PartS=Global_variables.SYSTEM[p1[1]][partORAnti1][s]
        if PartS.ID==id1:
            V1=PartS.V
            P1=PartS.P
            M1=PartS.M
            E1=PartS.Energy
            Eo1=M1*C_speed**2
            SYSTID1=s
            break
    for s in range(len(Global_variables.SYSTEM[p2[1]][partORAnti2])):
        PartS=Global_variables.SYSTEM[p2[1]][partORAnti2][s]
        if PartS.ID==id2:
            V2=PartS.V
            P2=PartS.P
            M2=PartS.M
            E2=PartS.Energy
            Eo2=M1*C_speed**2
            SYSTID2=s
            break
            

    
    #P1+P2=NEWP1+NEWP2
    #E1+E2=NEWE1+NEWE2

    #Eo1**2 – 2(E1*E 3 – C_speed**2*P1*P3*cosθ) + Eo3**2 = Eo4**2 –Eo2**2–2*Eo2(E1 – E3 )
    COEFLIST=[[],[]]
    for groupnumb,coefgroup in enumerate(COEFSlist):
        for partCoefindex,Coefinfo in enumerate(coefgroup):
            #coef=[V, b, Tpara, p,id,Xinter,end]
            partORanti_i,partindex_i=Coefinfo[-4]
            id_i=Coefinfo[-3]
            if (id_i==id1 and partORanti_i==partORAnti1 and partindex_i==p1[1]):
                Targs1,Bcoefs1,interX1=Coefinfo[2],Coefinfo[1],Coefinfo[-2]
                COEFLIST[0].append([groupnumb,partCoefindex])
            if (id_i==id2 and partORanti_i==partORAnti2 and partindex_i==p2[1]):
                Targs2,Bcoefs2,interX2=Coefinfo[2],Coefinfo[1],Coefinfo[-2]
                COEFLIST[1].append([groupnumb,partCoefindex])

    a1=BOUNDARY_FCT(V1,p1,id1,len(Bcoefs1)-1)
    XEND1=a1*Targs1[0]+Bcoefs1[-1]

    a2=BOUNDARY_FCT(V2,p2,id2,len(Bcoefs2)-1)
    XEND2=a2*Targs2[0]+Bcoefs2[-1]


    NEWV1,NEWE1,NEWP1,NEWXEND1=V2,E1+(P2-P1)*C_speed,P2,XEND2
    NEWV2,NEWE2,NEWP2,NEWXEND2=V1,E2+(P1-P2)*C_speed,P1,XEND1

    NewBcoefs1,NewTargs1,NewinterX1,NewVflipINFO1=[],[Targs1[0]],[],[]
    NewBcoefs2,NewTargs2,NewinterX2,NewVflipINFO2=[],[Targs1[0]],[],[]

    #NewBcoefs1,NewTargs1,NewinterX1=Targs1.copy(),Bcoefs1.copy(),interX1.copy()
    #NewBcoefs2,NewTargs2,NewinterX2=Targs2.copy(),Bcoefs2.copy(),interX2.copy()

    z1,z2=0,0
    for i1,T1 in enumerate(Targs1):
        if T1>ti:
            #if first time need to switch coll data

            #Vflipinfo[p[1]][p[0]][id].append([d_arg,Vchgsign[d_arg]])
            if z1==0:
                NewTargs2.append(ti)
                NewBcoefs2.append(Bcoefs1[i1])#NEED TO CHG V LIKE THIS ALSO
                NewinterX2.append([xo,xo])
                if len(Bcoefs1)>1:
                    NewVflipINFO2.append(Vflipinfo[p1[1]][p1[0]][id1][i1-1])
                z1=1
            NewTargs2.append(T1)
            NewBcoefs2.append(Bcoefs1[i1])
            if len(Bcoefs1)>1:
                NewinterX2.append(interX1[i1-1])
                NewVflipINFO2.append(Vflipinfo[p1[1]][p1[0]][id1][i1-1])
        else:

            NewTargs1.append(T1)
            NewBcoefs1.append(Bcoefs1[i1])
            if i1!=0 and len(Bcoefs1)>1:
                NewinterX1.append(interX1[i1-1])
                NewVflipINFO1.append(Vflipinfo[p1[1]][p1[0]][id1][i1-1])
    for i2,T2 in enumerate(Targs2[1:]):
        if T2>ti:
            if z2==0:
                NewTargs1.append(ti)
                NewBcoefs1.append(Bcoefs2[i2])#NEED TO CHG V LIKE THIS ALSO
                NewinterX1.append([xo,xo])
                if len(Bcoefs2)>1:
                    NewVflipINFO1.append(Vflipinfo[p2[1]][p2[0]][id2][i2-1])
                z2=1
            NewTargs1.append(T2)
            NewBcoefs1.append(Bcoefs2[i2])
            if len(Bcoefs2)>1:
                NewinterX1.append(interX2[i2-1])
                NewVflipINFO1.append(Vflipinfo[p2[1]][p2[0]][id2][i2-1])
        else:
            NewTargs2.append(T2)
            NewBcoefs2.append(Bcoefs2[i2])
            if i2!=0 and len(Bcoefs2)>1:
                NewinterX2.append(interX2[i2-1])
                NewVflipINFO2.append(Vflipinfo[p2[1]][p2[0]][id2][i2-1])
    
    TEMP_NewTargs1=NewTargs1[1:].copy()
    TEMP_NewTargs1.sort()
    TEMP_NewTargs1.insert(0,NewTargs1[0])
    TEMP_NewTargs2=NewTargs2[1:].copy()
    TEMP_NewTargs2.sort()
    TEMP_NewTargs2.insert(0,NewTargs2[0])
    TEMP_NewBcoefs1,TEMP_NewTargs1,TEMP_NewinterX1,TEMP_NewVflipINFO1=[],[],[],[]
    TEMP_NewBcoefs2,TEMP_NewTargs2,TEMP_NewinterX2,TEMP_NewVflipINFO2=[],[],[],[]

    for i1,T1 in enumerate(TEMP_NewTargs1):
        index1=NewTargs1.index(T1)
        TEMP_NewBcoefs1.append(NewBcoefs1[index1])
        if i1!=0:
            TEMP_NewinterX1.append(NewinterX1[index1-1])
            TEMP_NewVflipINFO1.append(NewVflipINFO1[index1-1])
    for i2,T2 in enumerate(TEMP_NewTargs2):
        index2=NewTargs2.index(T2)
        TEMP_NewBcoefs2.append(NewBcoefs2[index2])
        if i2!=0:
            TEMP_NewinterX2.append(NewinterX2[index2-1])
            TEMP_NewVflipINFO2.append(NewVflipINFO2[index2-1])



    NewTargs1=TEMP_NewTargs1
    NewTargs2=TEMP_NewTargs2
    NewBcoefs1,NewTargs1,NewinterX1,NewVflipINFO1=TEMP_NewBcoefs1,TEMP_NewTargs1,TEMP_NewinterX1,TEMP_NewVflipINFO1
    NewBcoefs2,NewTargs2,NewinterX2,NewVflipINFO2=TEMP_NewBcoefs2,TEMP_NewTargs2,TEMP_NewinterX2,TEMP_NewVflipINFO2
    #NewTargs1.sort()
    #NewTargs2.sort()

    Vflipinfo[p2[1]][p2[0]][id2]=NewVflipINFO2
    Vflipinfo[p1[1]][p1[0]][id1]=NewVflipINFO1


    #NewTargs1,NewBcoefs1,NewinterX1,NEWV1,NEWE1,NEWP1,NEWXEND1=0,0,0,0,0,0,0
    #NewTargs2,NewBcoefs2,NewinterX2,NEWV2,NEWE2,NEWP2,NEWXEND2=0,0,0,0,0,0,0
    
    for groupnumb,partCoefindex in COEFLIST[0]:
        COEFSlist[groupnumb][partCoefindex][0]=NEWV1
        COEFSlist[groupnumb][partCoefindex][1]=NewBcoefs1
        COEFSlist[groupnumb][partCoefindex][2]=NewTargs1
        COEFSlist[groupnumb][partCoefindex][-2]=NewinterX1
    for groupnumb,partCoefindex in COEFLIST[1]:
        COEFSlist[groupnumb][partCoefindex][0]=NEWV2
        COEFSlist[groupnumb][partCoefindex][1]=NewBcoefs2
        COEFSlist[groupnumb][partCoefindex][2]=NewTargs2
        COEFSlist[groupnumb][partCoefindex][-2]=NewinterX2


    #for groupnumb,coefgroup in enumerate(COEFSlist):
    #    for partCoefindex,Coefinfo in enumerate(coefgroup):
    #        #coef=[V, b, Tpara, p,id,Xinter,end]
    #        partORanti_i,partindex_i=Coefinfo[-4]
    #        id_i=Coefinfo[-3]
    #        if (id_i==id1 and partORanti_i==partORAnti1 and partindex_i==p1[1]):
    #            COEFSlist[groupnumb][partCoefindex][0]=NEWV1
    #            COEFSlist[groupnumb][partCoefindex][1]=NewBcoefs1
    #            COEFSlist[groupnumb][partCoefindex][2]=NewTargs1
    #            COEFSlist[groupnumb][partCoefindex][-2]=NewinterX1
    #        if (id_i==id2 and partORanti_i==partORAnti2 and partindex_i==p2[1]):
    #            COEFSlist[groupnumb][partCoefindex][0]=NEWV2
    #            COEFSlist[groupnumb][partCoefindex][1]=NewBcoefs2
    #            COEFSlist[groupnumb][partCoefindex][2]=NewTargs2
    #            COEFSlist[groupnumb][partCoefindex][-2]=NewinterX2


    Global_variables.SYSTEM[p1[1]][partORAnti1][SYSTID1].V=NEWV1
    Global_variables.SYSTEM[p1[1]][partORAnti1][SYSTID1].P=NEWP1
    Global_variables.SYSTEM[p1[1]][partORAnti1][SYSTID1].Energy=NEWE1
    #DOINFOLIST=[Xfin,p,id,Xinter,V,t_list,NZ]
    Global_variables.SYSTEM[p2[1]][partORAnti2][SYSTID2].V=NEWV2
    Global_variables.SYSTEM[p2[1]][partORAnti2][SYSTID2].P=NEWP2
    Global_variables.SYSTEM[p2[1]][partORAnti2][SYSTID2].Energy=NEWE2

    Z=0
    for doindex,do in enumerate(DOINFOLIST):
        do_Xend,do_p,do_id,do_Xinter,do_V,do_tparam,do_Nend=do
        do_p,do_id=do[1:3]
        if do_p[0]==partORAnti1 and do_p[1]==p1[1] and do_id==id1:
            DOINFOLIST[doindex][0]=NEWXEND1
            DOINFOLIST[doindex][3]=NewinterX1
            DOINFOLIST[doindex][4]=NEWV1
            DOINFOLIST[doindex][5]=NewTargs1
            DOINFOLIST[doindex][6]=len(NewTargs1)-1
            Z+=1
            if Z==2:
                break
        elif do_p[0]==partORAnti2 and do_p[1]==p2[1] and do_id==id2:
            DOINFOLIST[doindex][0]=NEWXEND2
            DOINFOLIST[doindex][3]=NewinterX2
            DOINFOLIST[doindex][4]=NEWV2
            DOINFOLIST[doindex][5]=NewTargs2
            DOINFOLIST[doindex][6]=len(NewTargs2)-1
            Z+=1
            if Z==2:
                break


    return(F,COEFSlist)
    """
