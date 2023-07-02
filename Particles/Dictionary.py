##PARTICLE INFORMATION##
NeutrinoMasses=(0.120*10**-6)/3

PARTICLE_DICT = {
  "electron" : {
    "index":0,
    "mass" : 0.511,#Mev
    "charge" : -1,  #e
    "spin":1/2,
    "color":'blue',
    "Strong_Charge":0
  },
  "electron_neutrino" : {
    "index":1,
    "mass" : NeutrinoMasses,
    "charge" : 0,
    "spin":1/2,
    "Strong_Charge":0
  },
  "muon" : {
    "index":2,
    "mass" : 0.511,#105.66,
    "charge" : -1,
    "spin":1/2,
    "color":'red',
    "Strong_Charge":0
  },
  "muon_neutrino" : {
    "index":3,
    "mass" : NeutrinoMasses,
    "charge" : 0,
    "spin":1/2,
    "Strong_Charge":0
  },
  "tau" : {
    "index":4,
    "mass" :  0.511,#1776.86,
    "charge" : -1,
    "spin":1/2,
    "color":'green',
    "Strong_Charge":0
  },
  "tau_neutrino" : {
    "index":5,
    "mass" : NeutrinoMasses,
    "charge" : 0,
    "spin":1/2,
    "Strong_Charge":0
  },
  "up_Quark" : {
    "index":6,
    "mass" : 2.3,
    "charge" : +2/3,
    "spin":1/2,
    "color":'brown',
    "Strong_Charge":1
  }
  ,"down_Quark" : {
    "index":7,
    "mass" : 4.8,
    "charge" : -1/3,
    "spin":1/2,
    "color":'cyan',
    "Strong_Charge":1
  },
  "charm_Quark" : {
    "index":8,
    "mass" : 1275,
    "charge" : +2/3,
    "spin":1/2,
    "color":'grey',
    "Strong_Charge":1
  }
  ,"strange_Quark" : {
    "index":9,
    "mass" : 95,
    "charge" : -1/3,
    "spin":1/2,
    "color":'pink',
    "Strong_Charge":1
  },
  "top_Quark" : {
    "index":10,
    "mass" : 173210,
    "charge" : 0,
    "spin":1/2,
    "color":'lime',
    "Strong_Charge":1
  }
  ,"bottom_Quark" : {
    "index":11,
    "mass" : 4180,
    "charge" : 0,
    "spin":1/2,
    "color":'tan',
    "Strong_Charge":1
  },
  "gluon" : {
    "index":12,
    "mass" : 0,
    "charge" : 0,
    "spin":1,
    "Strong_Charge":0
  },
  "photon" : {
    "index":13,
    "mass" : 0,
    "charge" : 0,
    "spin":1,
    "color":'orange',
    "Strong_Charge":0
  },
  "zboson" : {
    "index":14,
    "mass" : 91187,
    "charge" : 0,
    "spin":1,
    "Strong_Charge":0
  },
  "w-boson" : {
    "index":15,
    "mass" : 80377,
    "charge" : -1,
    "spin":1,
    "Strong_Charge":0
  },
    "w+boson" : {
    "index":16,
    "mass" : 80377,
    "charge" : +1,
    "spin":1,
    "Strong_Charge":0
  },
  "higgs" : {
    "index":17,
    "mass" : 125250,
    "charge" : 0,
    "spin":0,
    "Strong_Charge":0
  }

} 