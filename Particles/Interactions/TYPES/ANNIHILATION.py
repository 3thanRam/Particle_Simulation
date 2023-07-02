import numpy as np
import vg
rng=np.random.default_rng()



from Particles.Dictionary import PARTICLE_DICT
from Particles.ParticleClass import Particle
from Particles.Global_Variables import Global_variables



BOUNDARY_FCT=Global_variables.BOUNDARY_FCT
BOUNDARY_COND=Global_variables.BOUNDARY_COND
ROUNDDIGIT=Global_variables.ROUNDDIGIT
C_speed=Global_variables.C_speed
DIM_Numb=Global_variables.DIM_Numb
dt=Global_variables.dt
Vmax=Global_variables.Vmax





def ANNIHILATE(FirstAnn,F,COEFSlist,t):
    """Remove the particles 1,2 involved in a collision and update tracking information and collision points.

    Args:
    - FirstAnn (list): List containing information about the collision, including time, position, and IDs of the particles involved, as well as other variables used for tracking.
    - F (list): List of dictionaries containing information about the particles in the simulation, indexed by particle type and ID.

    Returns:
    - F (list): Updated list of dictionaries containing information about the particles in the simulation.
    """
    

    # Extract information about the collision
    ti,xo,coltype,z1,z2,p1,id1,p2,id2=FirstAnn


    
    

    partORAnti1=p1[0]
    partORAnti2=p2[0]
    Grnumblist=[]
    for groupnumb,coefgroup in enumerate(COEFSlist):
        killcoef=[]
        for indtokill,coefinfo in enumerate(coefgroup):
            typetest=coefinfo[-4]
            idtest=coefinfo[-3]
            
            if typetest==p1 and idtest==id1:
                V1=coefinfo[0]
                killcoef.append(indtokill)
                Targs1=coefinfo[2]
                Xinter1=coefinfo[-2]
            elif typetest==p2 and idtest==id2:
                V2=coefinfo[0]
                killcoef.append(indtokill)
                Targs2=coefinfo[2]
                Xinter2=coefinfo[-2]
        if killcoef:
            Grnumblist.append(groupnumb)
        killcoef.sort(reverse=True)
        for coefkillindex in killcoef:
            coefgroup.pop(coefkillindex)
    p1index=p1[1]
    p2index=p2[1]
    
    # Remove the particles involved in the collision from their respective particle lists

    Etot=0
    SystKillparams=[[p1index,partORAnti1,id1],[p2index,partORAnti2,id2]]
    for pindex,partORAnti,kill_id in SystKillparams:
        for s in range(len(Global_variables.SYSTEM[pindex][partORAnti])):
            if Global_variables.SYSTEM[pindex][partORAnti][s].ID==kill_id:
                Etot+=Global_variables.SYSTEM[pindex][partORAnti][s].Energy
                Global_variables.SYSTEM[pindex][partORAnti].remove(Global_variables.SYSTEM[pindex][partORAnti][s])
                break
    
    #create photons
    if z1!=0:
        V1=BOUNDARY_FCT(V1,p1,id1,z1)
    if z2!=0:
        V2=BOUNDARY_FCT(V2,p2,id2,z2)

    if DIM_Numb==3:
        alpha,beta=vg.angle(V1, V2, look=vg.basis.z),vg.angle(V1, V2, look=vg.basis.y)
        VParam=Vmax*np.array([np.cos(alpha)*np.cos(beta),np.sin(alpha)*np.cos(beta),-np.sin(beta)])#[1,0,0] rotated by alpha,beta around y,z
    elif DIM_Numb==2:
        SD=0.1745329251994329 # 10 degrees
        v1_u,v2_u=V1/np.linalg.norm(V1),V2/np.linalg.norm(V2)
        Angle=np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        theta=Angle+np.round(rng.normal(Angle, SD, 1), ROUNDDIGIT)[0] 
        VParam=Vmax*np.array([np.cos(theta),np.sin(theta)])#[1,0] rotated by theta around x,y plane
    else:
        VParam=Vmax
    Createparams1='Post_Interaction',xo,VParam,ti,Etot/2
    Createparams2='Post_Interaction',xo,-VParam,ti,Etot/2

    def CREATE_PARTICLE_fromAnnil(ParticleType,Create_particle_param):
        if PARTICLE_DICT['photon']['Strong_Charge']!=0:
            COLOUR=[0,0,0]
        else:
            COLOUR=None
        
        
        partoranti,typeindex=ParticleType
        Crindex=typeindex
        #Crindex=TYPE_to_index_Dict[typeindex]
        Global_variables.MaxIDperPtype[Crindex][partoranti]+=1
        Cr_id=Global_variables.MaxIDperPtype[Crindex][partoranti]
        Global_variables.SYSTEM[Crindex][partoranti].append(Particle(name='photon',parity=(0,13),ID=Cr_id,ExtraParams=Create_particle_param,Colour_Charge=COLOUR))
        D=Global_variables.SYSTEM[Crindex][partoranti][-1].DO(t)
        New_Xend,New_p,New_id,New_Xinter,New_V,New_Tpara,New_end=D
        if New_end == 0:
            b =[New_Xend - New_V * float(*New_Tpara)]
        else:
            b = []
            if BOUNDARY_COND == 0:
                for r in range(New_end):
                    b.append(New_Xinter[r][0] - New_V * float(New_Tpara[1+r]))
                b.append(New_Xend - New_V * float(New_Tpara[0]))
            else:
                A = np.copy(New_V)
                for r in range(New_end):
                    b.append(New_Xinter[r][0] - A * float(New_Tpara[1+r]))
                    flipindex,flipvalue=Global_variables.Vflipinfo[Crindex][partoranti][New_id][r]
                    A[flipindex] = flipvalue
                b.append(New_Xend - A * float(New_Tpara[0]))
        Global_variables.Ntot[Crindex][partoranti]+=1
        
        if New_end==0:
            b.insert(0,'PreCreation')
            New_Tpara.append(ti)
            New_end=1
            Global_variables.Vflipinfo[Crindex][partoranti][New_id].append([0,New_V[0]])
            Xinter_LIST=[[xo,xo]]
        else:
            xinterkilllist,vflipkill_list,b_kill_list,Tpara_kill_list=[],[],[],[]
            Xinter_LIST=list(New_Xinter)
            for it,newTpar in enumerate(New_Tpara):
                if newTpar<ti:
                    Tpara_kill_list.append(newTpar)
                    b_kill_list.append(b[it-1])
                    xinterkilllist.append(it-1)
                    vflipkill_list.append(Global_variables.Vflipinfo[Crindex][partoranti][New_id][it-1])
            Global_variables.Vflipinfo[Crindex][partoranti][New_id].insert(0,[0,New_V[0]])
            for Tpara_kill in Tpara_kill_list:
                New_Tpara.remove(Tpara_kill)
            New_Tpara.insert(1,ti)
            for b_kill in b_kill_list:
                b.remove(b_kill)
            xinterkilllist.sort(reverse=True)
            Newelem=[]
            for killxinter_index in xinterkilllist:
                if len(Xinter_LIST)==1:
                    Newelem=Xinter_LIST[killxinter_index]
                Xinter_LIST.pop(killxinter_index)
            for vflipkill in vflipkill_list:
                Global_variables.Vflipinfo[Crindex][partoranti][New_id].remove(vflipkill)
            if len(Newelem)==0:
                Newelem=Xinter_LIST[-1].copy()

            Newelem[0]=xo
            Newelem[1]=xo
            Xinter_LIST.append(Newelem)
            New_end=len(New_Tpara)-1
            b.insert(0,'PreCreation')

        New_Xinter=np.array(Xinter_LIST)
        Ncoef=[New_V, b, New_Tpara, New_p,New_id,New_Xinter,New_end]
        DOd=NewDOINFOLIST[-1].copy() #to keep same format so numpy doesn't throw error when rebuilding array
        for i in range(len(DOd)):
            DOd[i]=D[i]
        DOd[-4],DOd[-1]=New_Xinter,New_end
        return(DOd,Ncoef,New_Xend)

    NewDOINFOLIST=list(Global_variables.DOINFOLIST)
    Do_info1,Coef_info1,Xend1=CREATE_PARTICLE_fromAnnil((0,13),Createparams1)
    Do_info2,Coef_info2,Xend2=CREATE_PARTICLE_fromAnnil((0,13),Createparams2)
    Xendlist=[Xend1,Xend2]
    NewCoefs=[Coef_info1,Coef_info2]
    NewDOINFOLIST.extend([Do_info1,Do_info2])
    

    
    
    Global_variables.DOINFOLIST=np.array(NewDOINFOLIST)


    NewFO=[list(F[d]) for d in range(DIM_Numb)]
    TOKILL=[[] for d in range(DIM_Numb)]
    for d in range(DIM_Numb):
        for subind,subF in enumerate(NewFO[d]):
            #[('Pos', float), ('TypeID0', int),('TypeID1', int),('index', int)]
            if (subF[1]==p1[0] and subF[2]==p1[1] and subF[3]==id1) or (subF[1]==p2[0] and subF[2]==p2[1] and subF[3]==id2):#(subF[2]==p2 and subF[0]==id2)
                TOKILL[d].append(subF)

    # If the particles have a history of collisions, update the tracking information
    if z1 > 0:
        for zi in range(z1):
            Global_variables.TRACKING[p1index][partORAnti1][id1].extend([[Targs1[zi+1],Xinter1[zi][0]],['T','X'],[Targs1[zi+1],Xinter1[zi][1]]])
            Global_variables.ALL_TIME.extend(Targs1[1:])
    if z2 > 0:
        for zi in range(z2):
            Global_variables.TRACKING[p2index][partORAnti2][id2].extend([[Targs2[zi+1],Xinter2[zi][0]],['T','X'],[Targs2[zi+1],Xinter2[zi][1]]])
            Global_variables.ALL_TIME.extend(Targs2[1:])


    Global_variables.TRACKING[p1index][partORAnti1][id1].append([ti,[*xo]])
    Global_variables.TRACKING[p2index][partORAnti2][id2].append([ti,[*xo]])
    Global_variables.ALL_TIME.append(ti)

    # Remove the information about the particles involved in the collision from the collision point list and decrement the total number of particles and add product of annihilation
    for d in range(DIM_Numb):
        NewFelem=[]
        for killparticl in TOKILL[d]:
            if len(NewFO[d])==1:
                NewFelem=killparticl.copy()
            NewFO[d].remove(killparticl)
        for Nco,Ncoef in enumerate(NewCoefs):
            N_Xd,N_type,N_id=Xendlist[Nco][d],Ncoef[-4],Ncoef[-3]
            if len(NewFelem)==0:
                NewFelem=(NewFO[d][-1]).copy()
            NewFelem[0]=N_Xd
            NewFelem[1]=N_type[0]
            NewFelem[2]=N_type[1]
            NewFelem[3]=N_id
            NewFO[d].append(NewFelem)########
    
    dtype =[('Pos', float), ('TypeID0', int),('TypeID1', int),('index', int)]# [('Pos', float), ('Type', np.ndarray),('index', int)]#[('Pos', float), ('Type', int),('index', int)]
    F=np.array(NewFO,dtype=dtype)
    #F.sort(order='Pos')

    
    
    for gnum in Grnumblist:
        for Ncoef in NewCoefs:
            COEFSlist[gnum].append(Ncoef)

    Global_variables.COLPTS.append([ti,xo,coltype])
    Global_variables.Ntot[p1index][partORAnti1]-=1
    Global_variables.Ntot[p2index][partORAnti2] -= 1

    return(F,COEFSlist)