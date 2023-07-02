import numpy as np
from Particles.Dictionary import PARTICLE_DICT
from dataclasses import dataclass, field
from typing import Callable

Numb_of_TYPES=len(PARTICLE_DICT)
PARTICLE_NAMES=[*PARTICLE_DICT.keys()]


DT=[0.1,0.1,0.05] #time step (depends on numb of dimensions)
DistList=[1e-5,1,4] #capture distance (depends on numb of dimensions)
Speed_light=1
Global_variables=[]



@dataclass(slots=True)
class Global_var:
    #Where Global variables are stored in order to commnicate them between different scripts
    
    DIM_Numb:int
    L_FCT: list
    BOUNDARY_COND:int

    L:np.ndarray=field(init=False)
    Linf:np.ndarray=field(init=False)

    V0:np.ndarray=field(init=False)
    Vmax:np.ndarray=field(init=False)

    BoundSup_V: float=field(init=False)
    BoundSup_b: float=field(init=False)
    BoundInf_V: float=field(init=False)
    BoundInf_b: float=field(init=False)

    C_speed: float=Speed_light
    Dist_min: float=field(init=False)
    dt: float=field(init=False)

    Field_DICT: dict=field(default_factory=dict)

    ALL_TIME: list=field(default_factory=list)
    FIELD: list=field(default_factory=list)
    DOINFOLIST: list=field(default_factory=list)
    COLPTS: list=field(default_factory=list)

    TRACKING: list=field(init=False)
    Vflipinfo: list=field(init=False)
    MaxIDperPtype: list=field(init=False)

    Ntot: list=field(default_factory=list)

    File_path_name:str=''
    ROUNDDIGIT:int=15

    INTERCHECK:Callable=field(init=False)
    BOUNDARY_FCT:Callable=field(init=False)
    SYSTEM:list=field(init=False)
    S_Force:np.ndarray=field(init=False)

    def __post_init__(self):
        self.L,self.Linf=self.L_FCT[0](0),self.L_FCT[1](0)
        self.V0=Speed_light*np.ones(self.DIM_Numb)
        self.Vmax=self.V0/3

        self.BoundSup_V,self.BoundSup_b,self.BoundInf_V,self.BoundInf_b=0,0,0,0
        self.Dist_min=float(DistList[self.DIM_Numb-1])
        self.dt=DT[self.DIM_Numb-1]

        Ntot=[[0,0] for Ntype in range(Numb_of_TYPES)]
        Ntot[13]=[5,0]#photons

        #Ntot[0]=[5,0]#electrons/positrons
        #Ntot[2]=[5,0]#muons/anti-muons

        Ntot[6]=[2,0]#upquark
        Ntot[7]=[1,0]#downquark

        #elec 0
        #muon 2
        #quarks: 6-11

        self.Ntot=Ntot

        self.TRACKING,self.Vflipinfo=[[[[] for ni in range(numbpart)],[[] for ni in range(numbanti)]] for numbpart,numbanti in Ntot],[[[[] for ni in range(numbpart)],[[] for ni in range(numbanti)]] for numbpart,numbanti in Ntot]
        self.MaxIDperPtype=[[ntot_part-1,ntot_anti-1] for (ntot_part,ntot_anti) in Ntot ]
        



def init(DIM_Numb,BOUNDARY_COND,L_FCT):
    global Global_variables
    Global_variables=Global_var(DIM_Numb=DIM_Numb,BOUNDARY_COND=BOUNDARY_COND,L_FCT=L_FCT)
    #INTERCHECK & BOUNDARY_FCT & Particle need certain values to be already defined and raise an error if definded during class init
    #maybe in future create init functions for them that accept needed variables as input 
    if Global_variables.DIM_Numb==1:
        from Particles.Interactions.INTERACTION_CHECK import INTERCHECK_1D
        Global_variables.INTERCHECK=INTERCHECK_1D
    else:
        from Particles.Interactions.INTERACTION_CHECK import INTERCHECK_ND
        Global_variables.INTERCHECK=INTERCHECK_ND
    if Global_variables.BOUNDARY_COND==0:
        from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_PER
        Global_variables.BOUNDARY_FCT=BOUNDARY_FCT_PER
    else:
        from ENVIRONMENT.BOUNDARY_TYPES import BOUNDARY_FCT_HARD
        Global_variables.BOUNDARY_FCT=BOUNDARY_FCT_HARD
    
    from Particles.ParticleClass import Particle
    SYSTEM=[[[],[]] for partype in range(Numb_of_TYPES)]
    Ntot=Global_variables.Ntot
    POSCENTER=np.array([1 for d in range(DIM_Numb)])
    for index, (Npart, Nantipart), in enumerate(Ntot):
        if PARTICLE_DICT[PARTICLE_NAMES[index]]['Strong_Charge']!=0:
            COLOUR=[0,0,0]
            Xparam=['INIT_Quark_CREATION',POSCENTER]
        else:
            COLOUR=None
            Xparam=None

        SYSTEM[index][0]=[Particle(name=PARTICLE_NAMES[index],parity=(0,PARTICLE_DICT[PARTICLE_NAMES[index]]["index"]),ID=i,Colour_Charge=COLOUR,ExtraParams=Xparam) for i in range(Npart)]
        if index<12:
            SYSTEM[index][1]=[Particle(name=PARTICLE_NAMES[index],parity=(1,PARTICLE_DICT[PARTICLE_NAMES[index]]["index"]),ID=i,Colour_Charge=COLOUR,ExtraParams=Xparam) for i in range(Nantipart)]
    Global_variables.SYSTEM=SYSTEM


'''def __init__(self,DIM_Numb,BOUNDARY_COND,L_FCT):
        self.DIM_Numb=DIM_Numb
        self.File_path_name=''
        self.ROUNDDIGIT=15
        self.V0=C_speed*np.ones(DIM_Numb)
        self.Vmax=self.V0/3
        self.C_speed=C_speed
        self.L_FCT=L_FCT
        self.L,self.Linf=L_FCT[0](0),L_FCT[1](0)
        self.BoundSup_V,self.BoundSup_b,self.BoundInf_V,self.BoundInf_b=0,0,0,0
        self.Dist_min=float(DistList[DIM_Numb-1])
        
        Ntot=[[0,0] for Ntype in range(Numb_of_TYPES)]
        Ntot[0]=[51,28]#electrons/positrons
        Ntot[2]=[35,47]#muons/anti-muons

        self.Ntot=Ntot
        self.dt=DT[DIM_Numb-1]
        self.BOUNDARY_COND=BOUNDARY_COND
        self.TRACKING,self.Vflipinfo=[[[[] for ni in range(numbpart)],[[] for ni in range(numbanti)]] for numbpart,numbanti in Ntot],[[[[] for ni in range(numbpart)],[[] for ni in range(numbanti)]] for numbpart,numbanti in Ntot]
        
        
        self.MaxIDperPtype=[[ntot_part-1,ntot_anti-1] for (ntot_part,ntot_anti) in Ntot ]
        self.Field_DICT={}
        self.ALL_TIME,self.FIELD,self.DOINFOLIST,self.COLPTS=[],[],[],[]'''





    
